/**
 * @file tokenizer_encoder.h
 * @brief Conv-VAE speech tokenizer encoder API.
 *
 * Two parallel tokenizer encoders:
 * - Acoustic: vae_dim=64, gaussian sampling (std=0.5)
 * - Semantic: vae_dim=128, deterministic (mean only)
 *
 * Both use same architecture: causal 1D Conv-VAE with
 * encoder_ratios=[8,5,5,4,2,2] (total compression 3200x).
 * Output frame rate: 24000 / 3200 = 7.5 Hz
 */
#ifndef VV_TOKENIZER_ENCODER_H
#define VV_TOKENIZER_ENCODER_H

#include "vibevoice/types.h"
#include "vibevoice/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Conv-VAE block types ──────────────────────────────────────────────── */

/** @brief 1D convolution layer weights. */
typedef struct vv_conv1d_weights {
    vv_tensor_t weight;   /**< [out_ch, in_ch, kernel_size] or [out_ch, 1, ks] for depthwise */
    vv_tensor_t bias;     /**< [out_ch] or empty */
    int         stride;
    int         kernel_size;
    bool        causal;
    bool        depthwise;
} vv_conv1d_weights_t;

/** @brief Single encoder block weights (norm + mixer + ffn). */
typedef struct vv_encoder_block {
    /* Mixer path: norm → depthwise_conv → residual + layer_scale */
    vv_tensor_t mixer_norm_weight;       /**< RMSNorm weight */
    vv_conv1d_weights_t mixer_conv;      /**< Depthwise conv */
    vv_tensor_t mixer_layer_scale;       /**< Learnable scale (scalar per channel) */

    /* FFN path: norm → linear1 → act → linear2 → residual + layer_scale */
    vv_tensor_t ffn_norm_weight;
    vv_tensor_t ffn_linear1_weight;      /**< [hidden, channels] */
    vv_tensor_t ffn_linear1_bias;
    vv_tensor_t ffn_linear2_weight;      /**< [channels, hidden] */
    vv_tensor_t ffn_linear2_bias;
    vv_tensor_t ffn_layer_scale;
} vv_encoder_block_t;

/** @brief Single encoder stage (downsample + N blocks). */
typedef struct vv_encoder_stage {
    vv_conv1d_weights_t downsample;      /**< Strided conv for downsampling */
    vv_encoder_block_t* blocks;
    int                 n_blocks;
    int                 channels;        /**< Output channels for this stage */
} vv_encoder_stage_t;

/** @brief FP16 GPU mirror of one encoder block's weights. */
typedef struct vv_encoder_block_gpu {
    void* norm_w;
    void* conv_w;
    void* conv_b;
    void* gamma;
    void* ffn_norm_w;
    void* ffn_gamma;
    void* l1_w;
    void* l1_b;
    void* l2_w;
    void* l2_b;
    int   ffn_hidden;
    int   channels;
    void* cache;      /**< streaming tail for the depthwise mixer conv */
    int   cache_len;
} vv_encoder_block_gpu_t;

/** @brief FP16 GPU mirror of one encoder stage. */
typedef struct vv_encoder_stage_gpu {
    void* ds_w;
    void* ds_b;
    void* ds_cache;   /**< streaming tail for the downsample conv */
    int   ds_cache_len;
    vv_encoder_block_gpu_t* blocks;
} vv_encoder_stage_gpu_t;

/** @brief Full Conv-VAE encoder. */
typedef struct vv_conv_vae_encoder {
    /* Initial convolution (1 channel → encoder_n_filters) */
    vv_conv1d_weights_t input_conv;

    /* Encoder stages */
    vv_encoder_stage_t* stages;
    int                 n_stages;

    /* Final projection to VAE dim (mean and optionally logvar) */
    vv_conv1d_weights_t proj_mean;       /**< Project to vae_dim (mean) */
    vv_conv1d_weights_t proj_logvar;     /**< Project to vae_dim (logvar), acoustic only */

    /* Config */
    int   vae_dim;
    float fix_std;
    bool  gaussian;                      /**< true for acoustic, false for semantic */
    bool  causal;

    /* GPU buffers (pre-allocated at init) */
    void* gpu_workspace;
    size_t workspace_size;

    /*
     * FP16 GPU mirrors of every weight, uploaded once on the first encode.
     * Re-uploading per block cost ~270 MB of H2D traffic and hundreds of
     * cudaMalloc/cudaFree pairs per call, which dominated audio encoding.
     */
    bool  gpu_weights_ready;
    void* input_w_gpu;
    void* input_b_gpu;
    void* proj_w_gpu;
    void* proj_b_gpu;
    void* input_cache;
    int   input_cache_len;
    void* proj_cache;
    int   proj_cache_len;
    vv_encoder_stage_gpu_t* gpu_stages;
} vv_conv_vae_encoder_t;

/* ─── API ───────────────────────────────────────────────────────────────── */

/**
 * @brief Initialize a Conv-VAE encoder from model weights.
 *
 * @param model_weights  Pointer to loaded weight tensors for this encoder
 * @param config         Acoustic or semantic tokenizer config
 * @param is_acoustic    true for acoustic (gaussian), false for semantic
 * @param encoder        Output: initialized encoder
 */
vv_status_t vv_conv_vae_init(const vv_weight_t* model_weights, int n_weights,
                              const void* config,
                              bool is_acoustic,
                              vv_conv_vae_encoder_t** encoder);

/**
 * @brief Run Conv-VAE encoder forward pass (CPU reference).
 *
 * @param encoder      Initialized encoder
 * @param audio        Input audio [n_samples] at 24kHz
 * @param n_samples    Number of input samples
 * @param output       Output: [n_frames, vae_dim] latent tokens
 * @param n_frames     Output: number of frames
 */
vv_status_t vv_conv_vae_encode_cpu(vv_conv_vae_encoder_t* encoder,
                                    const float* audio, int n_samples,
                                    float** output, int* n_frames);

/**
 * @brief Run Conv-VAE encoder forward pass on GPU.
 *
 * @param encoder      Initialized encoder (weights on GPU)
 * @param audio_gpu    Input audio on GPU [n_samples]
 * @param n_samples    Number of samples
 * @param output_gpu   Output on GPU: [n_frames, vae_dim]
 * @param n_frames     Output: number of frames
 * @param stream       CUDA stream
 */
vv_status_t vv_conv_vae_encode_dev(const vv_conv_vae_encoder_t* encoder,
                                     const void* audio_gpu, int n_samples,
                                     void** output_gpu, int* n_frames,
                                     void* stream);

/**
 * @brief Upload the encoder weights to the GPU ahead of the first encode.
 *
 * Without this the ~600 ms of host-to-device traffic lands inside the first
 * transcription and is charged to audio encoding.
 */
vv_status_t vv_conv_vae_warmup(vv_conv_vae_encoder_t* encoder);

/**
 * @brief Free Conv-VAE encoder resources.
 */
vv_status_t vv_conv_vae_free(vv_conv_vae_encoder_t* encoder);

/* ─── Acoustic latent sampling ──────────────────────────────────────────── */

/**
 * @brief Replace the acoustic mean with a draw from its distribution.
 *
 * In place over [n_frames, vae_dim], applied once to the whole clip after the
 * segments are concatenated, which is where the reference samples. A no-op for
 * VV_ACOUSTIC_MODE or `fix_std` of 0, so the caller need not special-case the
 * default.
 *
 * @param fix_std    The checkpoint's fixed spread (0.5 here).
 * @param seed       Fixes the draw; the traversal order is fixed too, so the
 *                   same seed and clip give the same latent every time.
 * @param out_scale  Optional; the scale actually drawn, for logging.
 */
vv_status_t vv_acoustic_sample(float* latents, int n_frames, int vae_dim,
                               float fix_std, vv_acoustic_sampling_t mode,
                               uint64_t seed, float* out_scale);

/** @brief "mode", "fix" or "gaussian". */
const char* vv_acoustic_sampling_name(vv_acoustic_sampling_t mode);

/** @brief Parse one of those names; VV_ACOUSTIC_SAMPLING_COUNT if unknown. */
vv_acoustic_sampling_t vv_acoustic_sampling_parse(const char* s);

#ifdef __cplusplus
}
#endif

#endif /* VV_TOKENIZER_ENCODER_H */
