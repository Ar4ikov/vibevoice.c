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
    void**   k_cache;         /**< [num_layers] pointers to K cache on GPU */
    void**   v_cache;         /**< [num_layers] pointers to V cache on GPU */
    int      num_layers;
    int      n_kv_heads;
    int      head_dim;
    int      max_seq_len;     /**< Maximum cache capacity */
    int      current_len;     /**< Current number of cached positions */
    bool     fp8;             /**< Use FP8 KV-cache for memory savings */
} vv_kv_cache_t;

/**
 * @brief Allocate KV-cache on GPU.
 */
vv_status_t vv_kv_cache_create(vv_kv_cache_t** cache,
                                int num_layers, int n_kv_heads,
                                int head_dim, int max_seq_len,
                                bool fp8);

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
 */
vv_status_t vv_decoder_prefill(
    const vv_model_t* model,
    void* hidden_states,       /**< [seq_len, hidden_size] FP16, in/out */
    int seq_len,
    vv_kv_cache_t* kv_cache,
    void* workspace,
    size_t workspace_size,
    void* stream);

/**
 * @brief Single decode step through all 28 layers.
 */
vv_status_t vv_decoder_step(
    const vv_model_t* model,
    void* hidden_state,        /**< [1, hidden_size] FP16, in/out */
    vv_kv_cache_t* kv_cache,
    void* workspace,
    size_t workspace_size,
    void* stream);

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

    /* GPU-resident weight buffers */
    void*          embed_table_gpu;
    void*          lm_head_gpu;
    void*          final_norm_gpu;

    /* Per-layer GPU buffers for NF4 dequantized weights */
    void**         layer_temp_weights;

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
                               bool kv_fp8,
                               vv_inference_ctx_t** ctx);
vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result);
vv_status_t vv_inference_free(vv_inference_ctx_t* ctx);
vv_status_t vv_transcription_free(vv_transcription_t* result);

/**
 * @brief Get performance metrics from the last transcribe() call.
 */
const vv_perf_metrics_t* vv_inference_get_perf(const vv_inference_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VV_INFERENCE_H */
