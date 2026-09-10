/**
 * @file inference.h
 * @brief Inference pipeline API: end-to-end audio → transcription.
 */
#ifndef VV_INFERENCE_H
#define VV_INFERENCE_H

#include "vibevoice/types.h"
#include "vibevoice/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── KV-Cache ──────────────────────────────────────────────────────────── */

/**
 * @brief KV-cache for Qwen2 transformer.
 *
 * 28 layers × 2 (K+V) × 4 KV heads × head_dim=128
 * FP16 by default. FP8 option for memory savings.
 */
typedef struct vv_kv_cache {
    void**   k_cache;         /**< [num_layers] pointers to K cache          */
    void**   v_cache;         /**< [num_layers] pointers to V cache          */
    int      num_layers;
    int      n_kv_heads;
    int      head_dim;
    int      max_seq_len;     /**< Maximum cache capacity */
    int      current_len;     /**< Current number of cached positions */
    bool     fp8;             /**< Use FP8 KV-cache for memory savings */
    bool     on_cpu;          /**< true = CPU RAM, false = GPU VRAM */
} vv_kv_cache_t;

/**
 * @brief Allocate KV-cache (GPU or CPU).
 * @param on_cpu  If true, allocate in CPU RAM instead of GPU VRAM.
 */
vv_status_t vv_kv_cache_create(vv_kv_cache_t** cache,
                                int num_layers, int n_kv_heads,
                                int head_dim, int max_seq_len,
                                bool fp8, bool on_cpu);

/**
 * @brief Append new K, V to cache at current position.
 */
vv_status_t vv_kv_cache_append(vv_kv_cache_t* cache, int layer,
                                const void* k, const void* v,
                                int seq_len, void* stream);

/**
 * @brief Get K, V pointers for a layer (for attention).
 */
vv_status_t vv_kv_cache_get(const vv_kv_cache_t* cache, int layer,
                              const void** k, const void** v, int* len);

/**
 * @brief Reset cache (for new inference).
 */
vv_status_t vv_kv_cache_reset(vv_kv_cache_t* cache);

/**
 * @brief Free KV-cache.
 */
vv_status_t vv_kv_cache_free(vv_kv_cache_t* cache);

/* ─── Layer weight streaming pool ───────────────────────────────────────── */

/**
 * @brief Double-buffered GPU staging pool for layer-by-layer weight upload.
 *
 * When VRAM is too small to hold all 28 layers at once, we keep NF4
 * weights on CPU and upload one layer's worth to a reusable GPU buffer
 * before processing it.  Double-buffering allows overlapping transfer
 * of layer i+1 while layer i is computing.
 */
#define VV_LAYER_POOL_SLOTS 2
#define VV_LAYER_TENSORS_PER_LAYER 20   /* 2 norms + 7*(packed+scales) + 4 attn bias */

typedef struct vv_layer_pool {
    void*  gpu_buf[VV_LAYER_POOL_SLOTS]; /**< Pre-allocated GPU staging  */
    size_t buf_size;                      /**< Size of each GPU buffer    */
    int    loaded[VV_LAYER_POOL_SLOTS];   /**< Layer idx in slot (-1=empty) */

    /* Saved CPU pointers for restoring after unstage */
    void*  saved_ptrs[VV_LAYER_POOL_SLOTS][VV_LAYER_TENSORS_PER_LAYER];

    bool   all_resident;  /**< true = all layers already on GPU, pool unused */
} vv_layer_pool_t;

vv_status_t vv_layer_pool_create(vv_layer_pool_t** pool,
                                  const vv_model_t* model, bool all_resident);
vv_status_t vv_layer_pool_stage(vv_layer_pool_t* pool,
                                 vv_model_t* model, int layer_idx,
                                 void* stream);
vv_status_t vv_layer_pool_unstage(vv_layer_pool_t* pool,
                                   vv_model_t* model, int layer_idx);
vv_status_t vv_layer_pool_free(vv_layer_pool_t* pool);

/* ─── Decoder ───────────────────────────────────────────────────────────── */

/**
 * @brief Per-layer forward pass on GPU.
 */
vv_status_t vv_decoder_layer_forward(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,       /**< [seq_len, hidden_size] FP16, in/out */
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    void* temp_workspace,
    size_t workspace_size,
    void* stream);

/**
 * @brief Full prefill through all 28 layers.
 * @param pool  Optional layer pool for streaming (NULL = weights on GPU).
 * @param xfer  Transfer stream for async upload (NULL = use compute).
 */
vv_status_t vv_decoder_prefill(
    vv_model_t* model,
    void* hidden_states,       /**< [seq_len, hidden_size] FP16, in/out */
    int seq_len,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream);

/**
 * @brief Single decode step through all 28 layers.
 * @param pool  Optional layer pool for streaming (NULL = weights on GPU).
 * @param xfer  Transfer stream for async upload (NULL = use compute).
 */
vv_status_t vv_decoder_step(
    vv_model_t* model,
    void* hidden_state,        /**< [1, hidden_size] FP16, in/out */
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream);

/* ─── CPU-mode decoder (Phase 3) ───────────────────────────────────────── */

vv_status_t vv_decoder_prefill_cpu(
    vv_model_t* model,
    float* hidden_states,      /**< [seq_len, hidden_size] FP32, in/out */
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size);

vv_status_t vv_decoder_step_cpu(
    vv_model_t* model,
    float* hidden_state,       /**< [1, hidden_size] FP32, in/out */
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size);

/* ─── Sampling ──────────────────────────────────────────────────────────── */

/**
 * @brief Greedy decode: argmax of logits.
 */
vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                              int32_t* token_id);

/**
 * @brief Top-k sampling.
 */
vv_status_t vv_sample_topk(const void* logits_fp16, int vocab_size,
                             int k, float temperature, int32_t* token_id);

/* ─── Full inference context ────────────────────────────────────────────── */

/* Forward declarations for opaque types */
struct vv_tokenizer;
struct vv_conv_vae_encoder;
struct vv_connector;

typedef struct vv_inference_ctx {
    vv_model_t*    model;
    vv_kv_cache_t* kv_cache;
    void*          compute_stream;
    void*          transfer_stream;
    void*          workspace;
    size_t         workspace_size;
    int            gpu_id;

    /* Placement strategy (auto-selected from VRAM budget) */
    vv_placement_t placement;
    bool           use_gpu;           /**< false for CPU-only mode */

    /* GPU-resident weight buffers (NULL if offloaded to CPU) */
    void*          embed_table_gpu;
    void*          lm_head_gpu;
    void*          final_norm_gpu;

    /* Layer weight streaming pool */
    vv_layer_pool_t* layer_pool;

    /* Text tokenizer (cached for decode loop) */
    struct vv_tokenizer* tokenizer;

    /* Audio encoding components */
    struct vv_conv_vae_encoder* acoustic_encoder;
    struct vv_conv_vae_encoder* semantic_encoder;
    struct vv_connector*        acoustic_connector;
    struct vv_connector*        semantic_connector;

    /* Model directory path (for loading tokenizer) */
    char           model_dir[512];

    /* Performance metrics from last transcribe() call */
    vv_perf_metrics_t last_perf;
} vv_inference_ctx_t;

vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               const vv_init_params_t* params,
                               vv_inference_ctx_t** ctx);
vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result);
vv_status_t vv_inference_free(vv_inference_ctx_t* ctx);
vv_status_t vv_transcription_free(vv_transcription_t* result);

/**
 * @brief Parse the model's JSON answer into a structured transcription.
 *
 * Falls back to returning the raw text with no segments when the answer is
 * not the expected JSON array.
 */
vv_status_t vv_postprocess_text(const char* text, float audio_duration,
                                 vv_transcription_t** result);

/**
 * @brief Get performance metrics from the last transcribe() call.
 */
const vv_perf_metrics_t* vv_inference_get_perf(const vv_inference_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VV_INFERENCE_H */
