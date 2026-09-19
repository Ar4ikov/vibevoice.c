/**
 * @file vae_i8.h
 * @brief The int8 speech encoder of VibeVoice-ASR-BitNet, computed exactly
 *        as its reference runtime (VibeASR.cpp) computes it.
 *
 * VibeASR.cpp ships the two Conv-VAE encoders and both connectors as I8_S
 * tensors (int8, one FP32 scale per tensor) and runs them as a fully int8
 * pipeline: every op produces int8 values plus a single multiplier
 * `127 / max|y|` taken over the whole output tensor, all channels and all
 * time steps. The input audio is quantized to 8 bits the same way. The FFN
 * uses ReLU where the trained model has GELU, and the connector RMSNorm uses
 * eps = 1e-5. docs/BITNET.md has the source lines and the measured distance
 * to the float encoder.
 *
 * Why it exists here: transcripts can only match the reference if the audio
 * features do, and they are not close to the float encoder's (latent cosine
 * 0.88 acoustic, 0.95 semantic). Every integer product is exact and every
 * float epilogue repeats the expression the reference's build computes
 * (fused multiply-adds included), and add_scaled repeats its per-thread
 * partition, so at the reference's thread count the features equal
 * VibeASR.cpp's bit for bit (see vae_i8.c; tests/test_bitnet_vae.c).
 *
 * Differences to the float encoder that follow from the reference and are
 * kept on purpose:
 *   - frame count floor(n / 3200) (ggml pads no right edge) instead of ceil;
 *   - one scale per tensor means a frame depends on the whole clip, so this
 *     encoder is not streamable; long clips are cut into windows of
 *     `window_samples` encoded independently.
 *
 * Layout: activations are [time][channel], channel contiguous. Conv weights
 * are stored [out][tap][in] (reordered at load) so that an im2col row of a
 * convolution is a contiguous run of the activation buffer.
 */
#ifndef VV_VAE_I8_H
#define VV_VAE_I8_H

#include "vibevoice/types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief An int8 conv or linear layer (a linear is a conv with k = 1). */
typedef struct vv_i8_layer {
    int8_t* w;        /**< [out][k][in] (depthwise: [k][channels])        */
    float   w_scale;  /**< I8_S scale: w_real = q * w_scale               */
    float*  bias;     /**< FP32 [out]                                     */
    int     in_ch;
    int     out_ch;
    int     k;
    int     stride;
    int     depthwise;
    /** Packed copy for the SIMD GEMM (built by vv_i8vae_prepare), or NULL. */
    int8_t* packed;
} vv_i8_layer_t;

/** @brief One ConvNeXt block. */
typedef struct vv_i8_block {
    float*        norm;       /**< FP32 [C] */
    vv_i8_layer_t mixer;      /**< depthwise, k = 7                       */
    float*        gamma;      /**< FP32 [C] */
    float*        ffn_norm;   /**< FP32 [C] */
    vv_i8_layer_t fc1;        /**< C -> 4C, ReLU                          */
    vv_i8_layer_t fc2;        /**< 4C -> C                                */
    float*        ffn_gamma;  /**< FP32 [C] */
} vv_i8_block_t;

#define VV_I8VAE_MAX_STAGES 8

/**
 * Longest clip encoded as one tensor, as the reference would: 300 s. Its
 * int8 activations then take about 1.2 GB; longer clips are cut into
 * windows this long (the reference itself sizes a 10 KB/sample arena and
 * does not go that far).
 */
#define VV_I8VAE_WINDOW_SAMPLES ((int64_t)300 * 24000)

/** @brief One tower: encoder and its connector. */
typedef struct vv_i8_tower {
    int            n_stages;
    /** ds[0] is the stem (1 -> C0, stride 1), ds[i] the downsample into
     *  stage i. */
    vv_i8_layer_t  ds[VV_I8VAE_MAX_STAGES];
    vv_i8_block_t* blocks[VV_I8VAE_MAX_STAGES];
    int            depth[VV_I8VAE_MAX_STAGES];
    vv_i8_layer_t  head;      /**< C_last -> vae_dim, k = 7               */
    vv_i8_layer_t  cfc1;      /**< vae_dim -> hidden                      */
    float*         cnorm;     /**< FP32 [hidden], eps 1e-5                */
    vv_i8_layer_t  cfc2;      /**< hidden -> hidden                       */
    int            vae_dim;
    int            hidden;
} vv_i8_tower_t;

/** @brief Both towers. Owns every buffer it points to. */
typedef struct vv_i8vae {
    vv_i8_tower_t tower[2];   /**< 0 acoustic, 1 semantic                 */
    float         block_eps;  /**< encoder RMSNorm eps (1e-5)             */
    float         conn_eps;   /**< connector RMSNorm eps (1e-5)           */
    /** Threads, 0 = one per physical core (or OMP_NUM_THREADS). The count
     *  is part of the numerics: VibeASR.cpp's add_scaled result depends on
     *  its -t, and this encoder repeats that, so equal counts give equal
     *  features. */
    int           n_threads;
    void**        owned;      /**< every vv_alloc'd buffer, for freeing   */
    int           n_owned;
    int           cap_owned;
    size_t        bytes;      /**< weight bytes held                      */
} vv_i8vae_t;

/** @brief An empty encoder, ready for vv_i8vae_own(). */
vv_status_t vv_i8vae_create(vv_i8vae_t** out);

/** @brief vv_alloc `bytes` and record it as owned by `v`. */
void* vv_i8vae_own(vv_i8vae_t* v, size_t bytes);

/** @brief Free the encoder and every buffer it owns. NULL is accepted. */
void vv_i8vae_free(vv_i8vae_t* v);

/**
 * @brief Check shapes and build the packed weight copies the SIMD GEMM
 *        reads. Call once after the loader has filled the towers.
 */
vv_status_t vv_i8vae_prepare(vv_i8vae_t* v);

/** @brief Frames a clip of `n_samples` yields: floor through every stride. */
int vv_i8vae_frames(const vv_i8vae_t* v, int64_t n_samples);

/**
 * @brief Encode a clip with both towers and add their connector outputs,
 *        as VibeASR.cpp's prefill does.
 *
 * @param out      FP32 [n_frames][hidden], caller-owned, at least
 *                 vv_i8vae_frames(n_samples) rows
 * @param n_frames out: rows written
 * @param window_samples  clips longer than this are cut into windows of
 *                 this many samples (a multiple of 3200), each encoded as a
 *                 clip of its own; 0 = never cut
 */
vv_status_t vv_i8vae_encode(const vv_i8vae_t* v, const float* audio,
                            int64_t n_samples, int64_t window_samples,
                            float* out, int* n_frames);

/**
 * @brief One tower over one clip, before the towers are added: the
 *        dequantized connector output. For tests and tensor diffs.
 */
vv_status_t vv_i8vae_encode_tower(const vv_i8vae_t* v, int tower,
                                  const float* audio, int64_t n_samples,
                                  float* out, int* n_frames);

#ifdef __cplusplus
}
#endif

#endif /* VV_VAE_I8_H */
