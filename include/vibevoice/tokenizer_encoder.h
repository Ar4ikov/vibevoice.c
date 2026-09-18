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
 *
 * Three objects, because three different things vary at three different
 * rates:
 *
 *   vv_conv_vae_encoder_t  the architecture and the host weights, borrowed
 *                          from the loader; one per model
 *   vv_vae_weights_t       the FP16 device copy; immutable, one per device,
 *                          shared by every slot and request on it
 *   vv_vae_state_t         one audio stream's convolution tails (~700 KB);
 *                          what makes chunked encoding equal to encoding the
 *                          whole signal, for any chunk length
 *   vv_vae_arena_t         preallocated scratch, sized from (items, samples)
 *
 * vv_vae_encode takes a batch of items — segments of one file, chunks of
 * several live streams, stateless windows, in any mix — packs them along
 * time and runs every layer once over the lot. Nothing is allocated, freed
 * or synchronised inside it; it only enqueues work on the stream it is given.
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

/** @brief Single encoder stage (N blocks, then an optional downsample). */
typedef struct vv_encoder_stage {
    vv_conv1d_weights_t downsample;      /**< Strided conv into the next stage */
    vv_encoder_block_t* blocks;
    int                 n_blocks;
    int                 channels;        /**< Channels inside this stage */
} vv_encoder_stage_t;

struct vv_vae_cpu_weights;

/**
 * @brief Full Conv-VAE encoder: architecture plus borrowed FP32 host weights.
 *
 * Immutable after vv_conv_vae_init except for the CPU path's lazily built
 * FP16 copies, which are created once under the caller's control.
 */
typedef struct vv_conv_vae_encoder {
    /* Initial convolution (1 channel → encoder_n_filters) */
    vv_conv1d_weights_t input_conv;

    /* Encoder stages */
    vv_encoder_stage_t* stages;
    int                 n_stages;

    /* Final projection to VAE dim (mean) */
    vv_conv1d_weights_t proj_mean;       /**< Project to vae_dim (mean) */

    /* Config */
    int   vae_dim;
    float fix_std;
    float eps;                           /**< RMSNorm epsilon (layernorm_eps) */
    bool  gaussian;                      /**< true for acoustic, false for semantic */
    bool  causal;

    /** FP16 weights repacked for the CPU GEMMs; built on first CPU encode. */
    struct vv_vae_cpu_weights* cpu;
} vv_conv_vae_encoder_t;

/* ─── Construction ──────────────────────────────────────────────────────── */

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
 * @brief Whether the encoder has every weight the forward pass reads.
 */
bool vv_conv_vae_complete(const vv_conv_vae_encoder_t* encoder);

/** @brief Free the encoder structure and its CPU weight copies. */
vv_status_t vv_conv_vae_free(vv_conv_vae_encoder_t* encoder);

/**
 * @brief Output frames for `n_samples` of a fresh stream ending there.
 *
 * ceil(n_samples / 3200) for this model, computed layer by layer so it is
 * right for any stride schedule. The prompt can be built from this before a
 * single sample has been encoded.
 */
int vv_conv_vae_frames(const vv_conv_vae_encoder_t* encoder, int64_t n_samples);

/* ─── CPU encoder ───────────────────────────────────────────────────────── */

/**
 * @brief Run the encoder on the CPU over a whole clip.
 *
 * Time-tiled with carried convolution state, so memory is bounded by the
 * tile, not the clip; the FFNs go through the packed FP16 GEMM. Never
 * touches an accelerator.
 *
 * @param output    Out: [n_frames, vae_dim] FP32, caller frees with vv_free
 */
vv_status_t vv_conv_vae_encode_cpu(vv_conv_vae_encoder_t* encoder,
                                    const float* audio, int n_samples,
                                    float** output, int* n_frames);

/* ─── GPU encoder ───────────────────────────────────────────────────────── */

typedef struct vv_vae_weights vv_vae_weights_t;
typedef struct vv_vae_state   vv_vae_state_t;
typedef struct vv_vae_arena   vv_vae_arena_t;

/** @brief Device bytes vv_vae_weights_upload will take. */
size_t vv_vae_weights_bytes(const vv_conv_vae_encoder_t* encoder);

/**
 * @brief Upload the encoder's weights, as FP16, to the current device.
 *
 * Init-time only: this allocates and synchronises. The result is immutable
 * and may be read by any number of streams at once.
 */
vv_status_t vv_vae_weights_upload(const vv_conv_vae_encoder_t* encoder,
                                  void* stream, vv_vae_weights_t** out);
void vv_vae_weights_free(vv_vae_weights_t* w);

/** @brief The host description these weights were uploaded from. */
const vv_conv_vae_encoder_t* vv_vae_weights_encoder(const vv_vae_weights_t* w);

/** @brief Device bytes one vv_vae_state_t takes. */
size_t vv_vae_state_bytes(const vv_conv_vae_encoder_t* encoder);

/**
 * @brief Streaming state for one audio stream, on the current device.
 *
 * A state is used by one stream of work at a time: consecutive encodes that
 * share it must be ordered (same CUDA stream, or joined by events), because
 * each reads the tails the previous one wrote.
 */
vv_status_t vv_vae_state_create(const vv_vae_weights_t* w, vv_vae_state_t** out);

/** @brief Start a new stream. Host-only, costs nothing on the device. */
void vv_vae_state_reset(vv_vae_state_t* st);
void vv_vae_state_free(vv_vae_state_t* st);

/**
 * @brief Frames the next encode of `n_samples` through `st` will produce.
 *
 * `st` NULL means a stateless window (fresh context, final). Does not change
 * the state.
 */
int vv_vae_frames(const vv_vae_weights_t* w, const vv_vae_state_t* st,
                  int64_t n_samples, bool is_final);

/**
 * @brief Device bytes of an arena for batches of up to `max_items` items
 * totalling `max_samples` samples.
 *
 * Dominated by two stage-0 activation buffers, 2 x 32 channels x 2 bytes
 * per sample, plus a fixed FFN tile.
 */
size_t vv_vae_arena_bytes(const vv_conv_vae_encoder_t* encoder,
                          int max_items, int64_t max_samples);

vv_status_t vv_vae_arena_create(const vv_conv_vae_encoder_t* encoder,
                                int max_items, int64_t max_samples,
                                vv_vae_arena_t** out);
void vv_vae_arena_free(vv_vae_arena_t* a);

/** @brief Largest batch, in items and in samples, the arena was sized for. */
int     vv_vae_arena_max_items(const vv_vae_arena_t* a);
int64_t vv_vae_arena_max_samples(const vv_vae_arena_t* a);

/** @brief Bytes of the arena any encode so far has actually used. */
size_t vv_vae_arena_high_water(const vv_vae_arena_t* a);

/** @brief One piece of audio in a batched encode. */
typedef struct vv_vae_item {
    const void*     audio;       /**< device FP16 [n_samples]                 */
    int             n_samples;
    /**
     * Carried context. NULL encodes a stateless window: zero left context
     * and a final (ceil-aligned) right edge, exactly as a fresh encode of
     * just these samples would.
     */
    vv_vae_state_t* state;
    /** Last chunk of the stream: pad the right edge, then reset `state`. */
    bool            is_final;
    void*           out;         /**< device FP16 rows [frame][out_ld]        */
    int             out_ld;      /**< row stride in elements (>= vae_dim)     */
    int             skip_frames; /**< leading frames produced but not written */
    int             n_frames;    /**< out: frames produced, skipped included  */
} vv_vae_item_t;

/**
 * @brief Encode a batch of items; enqueue only.
 *
 * Every item's output equals, bit for bit, what encoding it alone would give,
 * and a stream cut into chunks of any length equals the whole stream. At most
 * vv_vae_arena_max_items() items and vv_vae_arena_max_samples() samples.
 */
vv_status_t vv_vae_encode(const vv_vae_weights_t* w, vv_vae_arena_t* a,
                          vv_vae_item_t* items, int n_items, void* stream);

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
