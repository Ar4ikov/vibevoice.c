/**
 * @file pipeline.c
 * @brief End-to-end inference pipeline: WAV → transcription JSON.
 *
 * Pipeline:
 * 1. Load audio → resample 24kHz → normalize -25 dBFS
 * 2. Acoustic encoder (Conv-VAE, FP16) → gaussian sample → connector
 * 3. Semantic encoder (Conv-VAE, FP16) → deterministic → connector
 * 4. Combine: acoustic_out + semantic_out → [T, 3584]
 * 5. Build input sequence (special tokens + audio features + prompt)
 * 6. LLM prefill through 28 Qwen2 layers (NF4)
 * 7. Autoregressive decode until <|endoftranscript|>
 * 8. Post-process tokens → structured JSON
 */

#include "vibevoice/inference.h"
#include "vibevoice/audio.h"
#include "vibevoice/text_tokenizer.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Portable high-resolution timer (same impl as CLI) */
#ifdef _WIN32
#include <windows.h>
static double vv_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double vv_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

/* Forward declarations */
extern vv_status_t vv_cuda_alloc(void** ptr, size_t size);
extern vv_status_t vv_cuda_free(void* ptr);
extern vv_status_t vv_cuda_memcpy_h2d(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_memcpy_d2h(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_stream_create(void** stream);
extern vv_status_t vv_cuda_stream_destroy(void* stream);
extern vv_status_t vv_cuda_stream_sync(void* stream);
extern vv_status_t vv_cuda_set_device(int device_id);
extern vv_status_t vv_cuda_get_device_info(int device_id, size_t* total_mem,
                                            size_t* free_mem, int* sm_count);

extern vv_status_t vv_embedding_cuda(
    const void* table, const int32_t* ids, void* output,
    int seq_len, int hidden_size, void* stream);

extern vv_status_t vv_rmsnorm_cuda(
    const void* input, const void* weight, void* output,
    int seq_len, int hidden_size, float eps, void* stream);

extern vv_status_t vv_gemm_fp16_cuda(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta, void* stream);

extern void vv_gemm_cleanup(void);

extern vv_status_t vv_postprocess_tokens(
    const char** token_texts, int n_tokens,
    vv_transcription_t** result);

extern vv_status_t vv_transcription_to_json(const vv_transcription_t* tr,
                                              char** json_str);

extern vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                                      int32_t* token_id);

/* Special tokens */
extern const char* VV_TOKEN_START_TRANSCRIPT;
extern const char* VV_TOKEN_END_TRANSCRIPT;
extern bool vv_is_end_token(const vv_tokenizer_t* tok, int32_t token_id);

/* ─── FP32 → FP16 conversion helper ────────────────────────────────────── */

static void float_to_half(const float* src, uint16_t* dst, int n) {
    /* Minimal FP32→FP16 conversion (round to nearest) */
    for (int i = 0; i < n; i++) {
        /* Use a union to access FP32 bits */
        union { float f; uint32_t u; } v;
        v.f = src[i];
        uint32_t sign = (v.u >> 16) & 0x8000;
        int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (v.u >> 13) & 0x3FF;

        if (exp <= 0) {
            dst[i] = (uint16_t)sign; /* flush to zero */
        } else if (exp >= 0x1F) {
            dst[i] = (uint16_t)(sign | 0x7C00); /* infinity */
        } else {
            dst[i] = (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
        }
    }
}

/* ─── Upload tensor to GPU helper ────────────────────────────────────────── */

/**
 * @brief Upload a CPU tensor to GPU in-place.
 *
 * Allocates GPU memory, copies data H→D, frees the CPU copy,
 * and replaces t->data with the GPU pointer.  Sets on_gpu = true.
 */
static vv_status_t upload_tensor_to_gpu(vv_tensor_t* t, void* stream) {
    if (!t || !t->data || t->on_gpu) return VV_OK;

    void* gpu_ptr = NULL;
    vv_status_t s = vv_cuda_alloc(&gpu_ptr, t->size_bytes);
    if (s != VV_OK) return s;

    s = vv_cuda_memcpy_h2d(gpu_ptr, t->data, t->size_bytes, stream);
    if (s != VV_OK) {
        vv_cuda_free(gpu_ptr);
        return s;
    }

    vv_free(t->data);
    t->data = gpu_ptr;
    t->on_gpu = true;
    return VV_OK;
}

/**
 * @brief Upload all NF4 layer weights (packed data, scales, layernorms)
 *        to GPU so CUDA kernels can access them.
 */
static vv_status_t upload_layer_weights(vv_model_t* model, void* stream) {
    size_t total_bytes = 0;

    for (int i = 0; i < model->num_layers; i++) {
        vv_layer_weights_t* L = &model->layers[i];
        vv_status_t s;

        /* Layernorms */
        s = upload_tensor_to_gpu(&L->input_layernorm, stream);
        if (s != VV_OK) return s;
        total_bytes += L->input_layernorm.size_bytes;

        s = upload_tensor_to_gpu(&L->post_attn_layernorm, stream);
        if (s != VV_OK) return s;
        total_bytes += L->post_attn_layernorm.size_bytes;

        /* Helper macro to upload a vv_weight_t (packed + scales) */
        #define UPLOAD_WEIGHT(w) do {                                  \
            s = upload_tensor_to_gpu(&(w).tensor, stream);             \
            if (s != VV_OK) return s;                                  \
            total_bytes += (w).tensor.size_bytes;                      \
            if ((w).quant.scales.data) {                               \
                s = upload_tensor_to_gpu(&(w).quant.scales, stream);   \
                if (s != VV_OK) return s;                              \
                total_bytes += (w).quant.scales.size_bytes;            \
            }                                                          \
        } while(0)

        UPLOAD_WEIGHT(L->attn.q_proj);
        UPLOAD_WEIGHT(L->attn.k_proj);
        UPLOAD_WEIGHT(L->attn.v_proj);
        UPLOAD_WEIGHT(L->attn.o_proj);
        UPLOAD_WEIGHT(L->mlp.gate_proj);
        UPLOAD_WEIGHT(L->mlp.up_proj);
        UPLOAD_WEIGHT(L->mlp.down_proj);

        #undef UPLOAD_WEIGHT
    }

    VV_LOG_I("inference: uploaded %d layers (%.1f MB NF4+scales+norms) to GPU",
             model->num_layers, (double)total_bytes / (1024.0 * 1024.0));
    return VV_OK;
}

/**
 * @brief Free GPU memory for all layer tensors that were uploaded.
 *
 * Must be called BEFORE vv_model_free, which would otherwise
 * try to vv_free (CPU) pointers that are actually on the GPU.
 */
static void free_layer_gpu_weights(vv_model_t* model) {
    if (!model || !model->layers) return;

    for (int i = 0; i < model->num_layers; i++) {
        vv_layer_weights_t* L = &model->layers[i];

        #define FREE_GPU_TENSOR(t) do {                 \
            if ((t).on_gpu && (t).data) {               \
                vv_cuda_free((t).data);                 \
                (t).data = NULL;                        \
                (t).on_gpu = false;                     \
            }                                           \
        } while(0)

        FREE_GPU_TENSOR(L->input_layernorm);
        FREE_GPU_TENSOR(L->post_attn_layernorm);

        #define FREE_GPU_WEIGHT(w) do {                 \
            FREE_GPU_TENSOR((w).tensor);                \
            FREE_GPU_TENSOR((w).quant.scales);          \
        } while(0)

        FREE_GPU_WEIGHT(L->attn.q_proj);
        FREE_GPU_WEIGHT(L->attn.k_proj);
        FREE_GPU_WEIGHT(L->attn.v_proj);
        FREE_GPU_WEIGHT(L->attn.o_proj);
        FREE_GPU_WEIGHT(L->mlp.gate_proj);
        FREE_GPU_WEIGHT(L->mlp.up_proj);
        FREE_GPU_WEIGHT(L->mlp.down_proj);

        #undef FREE_GPU_WEIGHT
        #undef FREE_GPU_TENSOR
    }
}

/* ─── Init ──────────────────────────────────────────────────────────────── */

vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               bool kv_fp8,
                               vv_inference_ctx_t** ctx) {
    if (!model_dir || !ctx) return VV_ERR_NULL_PTR;

    VV_LOG_I("inference: initializing from '%s' on GPU %d", model_dir, gpu_id);

    /* Set GPU */
    vv_status_t s = vv_cuda_set_device(gpu_id);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to set GPU %d", gpu_id);
        return s;
    }

    /* Allocate context */
    vv_inference_ctx_t* c = (vv_inference_ctx_t*)vv_alloc(
        sizeof(vv_inference_ctx_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));
    c->gpu_id = gpu_id;
    strncpy(c->model_dir, model_dir, sizeof(c->model_dir) - 1);

    /* Load model */
    s = vv_model_load(model_dir, &c->model);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to load model: %s", vv_status_str(s));
        vv_free(c);
        return s;
    }

    /* Create CUDA streams */
    s = vv_cuda_stream_create(&c->compute_stream);
    if (s != VV_OK) {
        vv_model_free(c->model);
        vv_free(c);
        return s;
    }
    s = vv_cuda_stream_create(&c->transfer_stream);
    if (s != VV_OK) {
        vv_cuda_stream_destroy(c->compute_stream);
        vv_model_free(c->model);
        vv_free(c);
        return s;
    }

    /* Create KV-cache */
    const vv_llm_config_t* llm = &c->model->config.llm;
    int max_seq = 8192;  /* Default max, can be overridden */
    s = vv_kv_cache_create(&c->kv_cache,
                            llm->num_hidden_layers,
                            llm->num_key_value_heads,
                            llm->head_dim,
                            max_seq, kv_fp8);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to create KV-cache");
        vv_cuda_stream_destroy(c->compute_stream);
        vv_cuda_stream_destroy(c->transfer_stream);
        vv_model_free(c->model);
        vv_free(c);
        return s;
    }

    /* ── Upload per-layer weights (NF4 packed + scales + layernorms) to GPU ── */
    s = upload_layer_weights(c->model, c->transfer_stream);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to upload layer weights to GPU");
        vv_kv_cache_free(c->kv_cache);
        vv_cuda_stream_destroy(c->compute_stream);
        vv_cuda_stream_destroy(c->transfer_stream);
        vv_model_free(c->model);
        vv_free(c);
        return s;
    }

    /* Allocate workspace (for intermediate computations).
     *
     * For prefill with seq_len=S, the decoder needs:
     *   - intermediates: S × (4·hs + 2·kv_hs + 2·inter_size) × 2  bytes
     *   - temp_weight:   inter_size × hs × 2                       bytes
     *
     * For the default model (hs=3584, inter=18944, kv_hs=512):
     *   per-token = 106496 bytes, temp_weight = 130 MB,
     *   prefill 2048 → ~340 MB total.
     *
     * We try 512 MB → 384 MB → 256 MB.
     */
    static const size_t ws_sizes[] = {
        (size_t)512 * 1024 * 1024,
        (size_t)384 * 1024 * 1024,
        (size_t)256 * 1024 * 1024,
    };
    s = VV_ERR_CUDA_OOM;
    for (int wi = 0; wi < 3; wi++) {
        c->workspace_size = ws_sizes[wi];
        s = vv_cuda_alloc(&c->workspace, c->workspace_size);
        if (s == VV_OK) break;
        VV_LOG_W("inference: failed to alloc %zu MB workspace, trying smaller",
                 c->workspace_size / (1024 * 1024));
    }
    if (s != VV_OK) {
        VV_LOG_E("inference: cannot allocate workspace");
        free_layer_gpu_weights(c->model);
        vv_kv_cache_free(c->kv_cache);
        vv_cuda_stream_destroy(c->compute_stream);
        vv_cuda_stream_destroy(c->transfer_stream);
        vv_model_free(c->model);
        vv_free(c);
        return s;
    }

    /* ── Upload embedding table to GPU ── */
    if (c->model->embed_tokens.data) {
        size_t sz = c->model->embed_tokens.size_bytes;
        s = vv_cuda_alloc(&c->embed_table_gpu, sz);
        if (s == VV_OK) {
            s = vv_cuda_memcpy_h2d(c->embed_table_gpu,
                                    c->model->embed_tokens.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload embed_tokens to GPU");
        } else {
            VV_LOG_I("inference: embed_tokens uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    /* ── Upload lm_head to GPU ── */
    if (c->model->lm_head.data) {
        size_t sz = c->model->lm_head.size_bytes;
        s = vv_cuda_alloc(&c->lm_head_gpu, sz);
        if (s == VV_OK) {
            s = vv_cuda_memcpy_h2d(c->lm_head_gpu,
                                    c->model->lm_head.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload lm_head to GPU");
        } else {
            VV_LOG_I("inference: lm_head uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    /* ── Upload final norm to GPU ── */
    if (c->model->final_norm.data) {
        size_t sz = c->model->final_norm.size_bytes;
        s = vv_cuda_alloc(&c->final_norm_gpu, sz);
        if (s == VV_OK) {
            s = vv_cuda_memcpy_h2d(c->final_norm_gpu,
                                    c->model->final_norm.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload final_norm to GPU");
        }
    }

    /* ── Load tokenizer ── */
    s = vv_tokenizer_load(model_dir, &c->tokenizer);
    if (s != VV_OK) {
        VV_LOG_W("inference: failed to load tokenizer (will retry from model dir)");
    }

    /* ── Init audio encoders ── */
    if (c->model->n_acoustic_weights > 0) {
        s = vv_conv_vae_init(c->model->acoustic_weights,
                              c->model->n_acoustic_weights,
                              &c->model->config.acoustic,
                              true, &c->acoustic_encoder);
        if (s != VV_OK) {
            VV_LOG_W("inference: failed to init acoustic encoder");
        }
    }
    if (c->model->n_semantic_weights > 0) {
        s = vv_conv_vae_init(c->model->semantic_weights,
                              c->model->n_semantic_weights,
                              &c->model->config.semantic,
                              false, &c->semantic_encoder);
        if (s != VV_OK) {
            VV_LOG_W("inference: failed to init semantic encoder");
        }
    }

    /* ── Init connectors ── */
    if (c->model->acoustic_connector_fc1.tensor.data) {
        c->acoustic_connector = (vv_connector_t*)vv_alloc(
            sizeof(vv_connector_t));
        if (c->acoustic_connector) {
            vv_connector_init(c->acoustic_connector,
                               c->model->config.acoustic_vae_dim,
                               llm->hidden_size);
            /* Wire loaded weights into connector */
            c->acoustic_connector->fc1_weight =
                c->model->acoustic_connector_fc1.tensor;
            c->acoustic_connector->fc1_bias =
                c->model->acoustic_connector_fc1.quant.packed;
            c->acoustic_connector->norm_weight =
                c->model->acoustic_connector_norm.tensor;
            c->acoustic_connector->fc2_weight =
                c->model->acoustic_connector_fc2.tensor;
            c->acoustic_connector->fc2_bias =
                c->model->acoustic_connector_fc2.quant.packed;
        }
    }
    if (c->model->semantic_connector_fc1.tensor.data) {
        c->semantic_connector = (vv_connector_t*)vv_alloc(
            sizeof(vv_connector_t));
        if (c->semantic_connector) {
            vv_connector_init(c->semantic_connector,
                               c->model->config.semantic_vae_dim,
                               llm->hidden_size);
            c->semantic_connector->fc1_weight =
                c->model->semantic_connector_fc1.tensor;
            c->semantic_connector->fc1_bias =
                c->model->semantic_connector_fc1.quant.packed;
            c->semantic_connector->norm_weight =
                c->model->semantic_connector_norm.tensor;
            c->semantic_connector->fc2_weight =
                c->model->semantic_connector_fc2.tensor;
            c->semantic_connector->fc2_bias =
                c->model->semantic_connector_fc2.quant.packed;
        }
    }

    /* Sync transfers before proceeding */
    vv_cuda_stream_sync(c->transfer_stream);

    VV_LOG_I("inference: initialized successfully (workspace=%zu MB, "
             "tokenizer=%s, encoders=%s)",
             c->workspace_size / (1024 * 1024),
             c->tokenizer ? "yes" : "no",
             c->acoustic_encoder ? "yes" : "no");
    *ctx = c;
    return VV_OK;
}

/* ─── Transcribe ────────────────────────────────────────────────────────── */

vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    if (!ctx || !audio_samples || !result) return VV_ERR_NULL_PTR;

    /* ── Zero-init metrics ── */
    vv_perf_metrics_t* perf = &ctx->last_perf;
    memset(perf, 0, sizeof(*perf));
    double t_total_start = vv_time_ms();

    vv_status_t s;
    const vv_model_config_t* cfg = &ctx->model->config;
    const vv_llm_config_t* llm = &cfg->llm;
    int hs = llm->hidden_size;
    int vocab_size = llm->vocab_size;
    int max_new_tokens = params ? params->max_new_tokens : 64000;
    if (max_new_tokens <= 0) max_new_tokens = 64000;

    perf->audio_duration_sec = (float)num_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = hs;
    perf->kv_fp8 = ctx->kv_cache ? ctx->kv_cache->fp8 : false;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    VV_LOG_I("inference: transcribing %d samples (%.2f sec), max_tokens=%d",
             num_samples, perf->audio_duration_sec, max_new_tokens);

    /* Verify we have the critical components */
    if (!ctx->embed_table_gpu || !ctx->lm_head_gpu || !ctx->final_norm_gpu) {
        VV_LOG_E("inference: GPU weight buffers not initialized");
        return VV_ERR_NULL_PTR;
    }
    if (!ctx->tokenizer) {
        VV_LOG_E("inference: tokenizer not loaded");
        return VV_ERR_NULL_PTR;
    }

    /* Reset KV-cache */
    vv_kv_cache_reset(ctx->kv_cache);

    /*
     * ═══════════════════════════════════════════════════════════════════════
     * STEP 1: Audio encoding
     * ═══════════════════════════════════════════════════════════════════════
     */
    double t_step = vv_time_ms();
    VV_LOG_I("inference: step 1 — encoding audio through tokenizers");

    float* acoustic_latents = NULL;
    float* semantic_latents = NULL;
    float* acoustic_features = NULL;
    float* semantic_features = NULL;
    float* combined_fp32 = NULL;
    int n_acoustic_frames = 0;
    int n_semantic_frames = 0;

    /* Try acoustic encoder → [T_a, acoustic_vae_dim] */
    if (ctx->acoustic_encoder &&
        ctx->acoustic_encoder->stages != NULL &&
        ctx->acoustic_encoder->input_conv.weight.data != NULL) {
        s = vv_conv_vae_encode_cpu(ctx->acoustic_encoder,
                                    audio_samples, num_samples,
                                    &acoustic_latents, &n_acoustic_frames);
        if (s == VV_OK) {
            VV_LOG_I("inference: acoustic encoder -> %d frames",
                     n_acoustic_frames);
        } else {
            VV_LOG_W("inference: acoustic encoder failed: %s (using zeros)",
                     vv_status_str(s));
        }
    }

    /* Try semantic encoder → [T_s, semantic_vae_dim] */
    if (ctx->semantic_encoder &&
        ctx->semantic_encoder->stages != NULL &&
        ctx->semantic_encoder->input_conv.weight.data != NULL) {
        s = vv_conv_vae_encode_cpu(ctx->semantic_encoder,
                                    audio_samples, num_samples,
                                    &semantic_latents, &n_semantic_frames);
        if (s == VV_OK) {
            VV_LOG_I("inference: semantic encoder -> %d frames",
                     n_semantic_frames);
        } else {
            VV_LOG_W("inference: semantic encoder failed: %s (using zeros)",
                     vv_status_str(s));
        }
    }

    /* Determine frame count */
    int n_audio_frames;
    if (n_acoustic_frames > 0 && n_semantic_frames > 0) {
        n_audio_frames = n_acoustic_frames < n_semantic_frames ?
                         n_acoustic_frames : n_semantic_frames;
    } else if (n_acoustic_frames > 0) {
        n_audio_frames = n_acoustic_frames;
    } else if (n_semantic_frames > 0) {
        n_audio_frames = n_semantic_frames;
    } else {
        n_audio_frames = num_samples / 3200;
        if (n_audio_frames < 1) n_audio_frames = 1;
        VV_LOG_W("inference: Conv-VAE encoders not wired (need GPU path) "
                 "— using %d estimated frames with zero features",
                 n_audio_frames);
    }
    perf->audio_frames = n_audio_frames;

    /* Connectors */
    if (acoustic_latents && ctx->acoustic_connector) {
        s = vv_connector_forward_cpu(ctx->acoustic_connector,
                                      acoustic_latents, n_audio_frames,
                                      &acoustic_features);
        if (s != VV_OK)
            VV_LOG_W("inference: acoustic connector failed: %s",
                     vv_status_str(s));
    }
    if (semantic_latents && ctx->semantic_connector) {
        s = vv_connector_forward_cpu(ctx->semantic_connector,
                                      semantic_latents, n_audio_frames,
                                      &semantic_features);
        if (s != VV_OK)
            VV_LOG_W("inference: semantic connector failed: %s",
                     vv_status_str(s));
    }
    if (acoustic_latents) vv_free(acoustic_latents);
    if (semantic_latents) vv_free(semantic_latents);

    /* Combine: element-wise add → [T, hs] FP32 */
    int combined_elems = n_audio_frames * hs;
    combined_fp32 = (float*)vv_alloc((size_t)combined_elems * sizeof(float));
    if (!combined_fp32) {
        if (acoustic_features) vv_free(acoustic_features);
        if (semantic_features) vv_free(semantic_features);
        return VV_ERR_OUT_OF_MEMORY;
    }

    if (acoustic_features && semantic_features) {
        for (int i = 0; i < combined_elems; i++)
            combined_fp32[i] = acoustic_features[i] + semantic_features[i];
    } else if (acoustic_features) {
        memcpy(combined_fp32, acoustic_features,
               (size_t)combined_elems * sizeof(float));
    } else if (semantic_features) {
        memcpy(combined_fp32, semantic_features,
               (size_t)combined_elems * sizeof(float));
    } else {
        memset(combined_fp32, 0, (size_t)combined_elems * sizeof(float));
    }
    if (acoustic_features) vv_free(acoustic_features);
    if (semantic_features) vv_free(semantic_features);

    /* FP32 → FP16 */
    uint16_t* combined_fp16 = (uint16_t*)vv_alloc(
        (size_t)combined_elems * sizeof(uint16_t));
    if (!combined_fp16) {
        vv_free(combined_fp32);
        return VV_ERR_OUT_OF_MEMORY;
    }
    float_to_half(combined_fp32, combined_fp16, combined_elems);
    vv_free(combined_fp32);

    perf->audio_encode_ms = vv_time_ms() - t_step;
    VV_LOG_I("inference: audio encoding done in %.1f ms (%d frames)",
             perf->audio_encode_ms, n_audio_frames);

    /*
     * ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Build input sequence
     * ═══════════════════════════════════════════════════════════════════════
     */
    t_step = vv_time_ms();
    VV_LOG_I("inference: step 2 — building input sequence");

    int start_id = vv_tokenizer_special_id(ctx->tokenizer,
                                            VV_TOKEN_START_TRANSCRIPT);
    if (start_id < 0) start_id = 0;

    int seq_len = 1 + n_audio_frames;
    perf->prefill_tokens = seq_len;

    int32_t* input_ids = (int32_t*)vv_alloc(
        (size_t)seq_len * sizeof(int32_t));
    if (!input_ids) {
        vv_free(combined_fp16);
        return VV_ERR_OUT_OF_MEMORY;
    }

    input_ids[0] = start_id;
    for (int i = 0; i < n_audio_frames; i++)
        input_ids[1 + i] = 0;

    size_t hidden_bytes = (size_t)seq_len * (size_t)hs * 2;
    void* hidden_states_gpu = NULL;
    int32_t* input_ids_gpu = NULL;

    s = vv_cuda_alloc(&hidden_states_gpu, hidden_bytes);
    if (s != VV_OK) {
        vv_free(input_ids);
        vv_free(combined_fp16);
        return s;
    }

    s = vv_cuda_alloc((void**)&input_ids_gpu,
                       (size_t)seq_len * sizeof(int32_t));
    if (s != VV_OK) {
        vv_cuda_free(hidden_states_gpu);
        vv_free(input_ids);
        vv_free(combined_fp16);
        return s;
    }

    s = vv_cuda_memcpy_h2d(input_ids_gpu, input_ids,
                            (size_t)seq_len * sizeof(int32_t),
                            ctx->compute_stream);
    vv_free(input_ids);
    if (s != VV_OK) {
        vv_cuda_free(hidden_states_gpu);
        vv_cuda_free(input_ids_gpu);
        vv_free(combined_fp16);
        return s;
    }

    s = vv_embedding_cuda(ctx->embed_table_gpu, input_ids_gpu,
                           hidden_states_gpu, seq_len, hs,
                           ctx->compute_stream);
    vv_cuda_free(input_ids_gpu);
    if (s != VV_OK) {
        VV_LOG_E("inference: embedding failed");
        vv_cuda_free(hidden_states_gpu);
        vv_free(combined_fp16);
        return s;
    }

    if (n_audio_frames > 0) {
        void* audio_dst = (uint8_t*)hidden_states_gpu + (size_t)hs * 2;
        s = vv_cuda_memcpy_h2d(audio_dst, combined_fp16,
                                (size_t)n_audio_frames * (size_t)hs * 2,
                                ctx->compute_stream);
        if (s != VV_OK) {
            VV_LOG_E("inference: audio feature upload failed");
            vv_cuda_free(hidden_states_gpu);
            vv_free(combined_fp16);
            return s;
        }
    }
    vv_free(combined_fp16);
    vv_cuda_stream_sync(ctx->compute_stream);

    perf->sequence_build_ms = vv_time_ms() - t_step;
    VV_LOG_I("inference: input sequence ready (seq_len=%d) in %.1f ms",
             seq_len, perf->sequence_build_ms);

    /*
     * ═══════════════════════════════════════════════════════════════════════
     * STEP 3: LLM Prefill
     * ═══════════════════════════════════════════════════════════════════════
     */
    t_step = vv_time_ms();
    VV_LOG_I("inference: step 3 — LLM prefill (%d tokens, %d layers)",
             seq_len, ctx->model->num_layers);

    s = vv_decoder_prefill(ctx->model, hidden_states_gpu, seq_len,
                            ctx->kv_cache, ctx->workspace,
                            ctx->workspace_size, ctx->compute_stream);
    if (s != VV_OK) {
        VV_LOG_E("inference: prefill failed: %s", vv_status_str(s));
        vv_cuda_free(hidden_states_gpu);
        return s;
    }

    /* RMSNorm → lm_head → sample first token */
    size_t one_hidden = (size_t)hs * 2;
    void* last_hidden_gpu = (uint8_t*)hidden_states_gpu +
                            (size_t)(seq_len - 1) * one_hidden;

    void* normed_gpu = NULL;
    void* logits_gpu = NULL;
    void* hidden_one_gpu = NULL;

    s = vv_cuda_alloc(&normed_gpu, one_hidden);
    if (s != VV_OK) { vv_cuda_free(hidden_states_gpu); return s; }

    s = vv_cuda_alloc(&logits_gpu, (size_t)vocab_size * 2);
    if (s != VV_OK) {
        vv_cuda_free(normed_gpu);
        vv_cuda_free(hidden_states_gpu);
        return s;
    }

    s = vv_cuda_alloc(&hidden_one_gpu, one_hidden);
    if (s != VV_OK) {
        vv_cuda_free(logits_gpu);
        vv_cuda_free(normed_gpu);
        vv_cuda_free(hidden_states_gpu);
        return s;
    }

    s = vv_rmsnorm_cuda(last_hidden_gpu, ctx->final_norm_gpu,
                         normed_gpu, 1, hs, llm->rms_norm_eps,
                         ctx->compute_stream);
    if (s != VV_OK) goto cleanup_decode;

    s = vv_gemm_fp16_cuda(normed_gpu, ctx->lm_head_gpu, logits_gpu,
                           1, vocab_size, hs, 1.0f, 0.0f,
                           ctx->compute_stream);
    if (s != VV_OK) goto cleanup_decode;

    int32_t token_id;
    vv_cuda_stream_sync(ctx->compute_stream);
    s = vv_sample_greedy(logits_gpu, vocab_size, &token_id);
    if (s != VV_OK) goto cleanup_decode;

    vv_cuda_free(hidden_states_gpu);
    hidden_states_gpu = NULL;

    perf->prefill_ms = vv_time_ms() - t_step;
    perf->ttft_ms = vv_time_ms() - t_total_start;
    perf->prefill_tok_per_sec = (perf->prefill_ms > 0.001)
        ? (double)seq_len / perf->prefill_ms * 1000.0 : 0.0;

    VV_LOG_I("inference: prefill done in %.1f ms (%.0f tok/s), "
             "first token=%d, TTFT=%.1f ms",
             perf->prefill_ms, perf->prefill_tok_per_sec,
             token_id, perf->ttft_ms);

    /*
     * ═══════════════════════════════════════════════════════════════════════
     * STEP 4: Autoregressive decode
     * ═══════════════════════════════════════════════════════════════════════
     */
    t_step = vv_time_ms();
    VV_LOG_I("inference: step 4 — autoregressive decode");

    int out_cap = max_new_tokens < 8192 ? max_new_tokens : 8192;
    int32_t* output_tokens = (int32_t*)vv_alloc(
        (size_t)out_cap * sizeof(int32_t));
    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
    int n_generated = 0;

    if (!vv_is_end_token(ctx->tokenizer, token_id)) {
        output_tokens[n_generated++] = token_id;
    }

    /* Decode loop */
    while (n_generated < max_new_tokens &&
           !vv_is_end_token(ctx->tokenizer, token_id)) {

        int32_t tok_gpu_buf[1];
        tok_gpu_buf[0] = token_id;
        int32_t* tok_id_gpu = NULL;
        s = vv_cuda_alloc((void**)&tok_id_gpu, sizeof(int32_t));
        if (s != VV_OK) break;
        vv_cuda_memcpy_h2d(tok_id_gpu, tok_gpu_buf, sizeof(int32_t),
                            ctx->compute_stream);

        s = vv_embedding_cuda(ctx->embed_table_gpu, tok_id_gpu,
                               hidden_one_gpu, 1, hs,
                               ctx->compute_stream);
        vv_cuda_free(tok_id_gpu);
        if (s != VV_OK) break;

        s = vv_decoder_step(ctx->model, hidden_one_gpu, ctx->kv_cache,
                             ctx->workspace, ctx->workspace_size,
                             ctx->compute_stream);
        if (s != VV_OK) {
            VV_LOG_E("inference: decode step %d failed: %s",
                     n_generated, vv_status_str(s));
            break;
        }

        s = vv_rmsnorm_cuda(hidden_one_gpu, ctx->final_norm_gpu,
                             normed_gpu, 1, hs, llm->rms_norm_eps,
                             ctx->compute_stream);
        if (s != VV_OK) break;

        s = vv_gemm_fp16_cuda(normed_gpu, ctx->lm_head_gpu, logits_gpu,
                               1, vocab_size, hs, 1.0f, 0.0f,
                               ctx->compute_stream);
        if (s != VV_OK) break;

        vv_cuda_stream_sync(ctx->compute_stream);
        s = vv_sample_greedy(logits_gpu, vocab_size, &token_id);
        if (s != VV_OK) break;

        if (vv_is_end_token(ctx->tokenizer, token_id)) break;

        if (n_generated >= out_cap) {
            out_cap *= 2;
            output_tokens = (int32_t*)vv_realloc(output_tokens,
                (size_t)out_cap * sizeof(int32_t));
            if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
        }
        output_tokens[n_generated++] = token_id;

        /* Progress: log every 500 tokens with running speed */
        if ((n_generated % 500) == 0) {
            double elapsed = vv_time_ms() - t_step;
            double running_tps = (elapsed > 0.001)
                ? (double)n_generated / elapsed * 1000.0 : 0.0;
            VV_LOG_I("inference: decoded %d tokens (%.1f tok/s)...",
                     n_generated, running_tps);
        }
    }

    perf->decode_ms = vv_time_ms() - t_step;
    perf->decode_tokens = n_generated;
    perf->decode_tok_per_sec = (perf->decode_ms > 0.001)
        ? (double)n_generated / perf->decode_ms * 1000.0 : 0.0;

    VV_LOG_I("inference: decode complete — %d tokens in %.1f ms (%.1f tok/s)",
             n_generated, perf->decode_ms, perf->decode_tok_per_sec);

    /*
     * ═══════════════════════════════════════════════════════════════════════
     * STEP 5: Post-processing
     * ═══════════════════════════════════════════════════════════════════════
     */
    t_step = vv_time_ms();
    VV_LOG_I("inference: step 5 — post-processing");

    const char** token_texts = (const char**)vv_alloc(
        (size_t)n_generated * sizeof(char*));
    if (!token_texts) {
        vv_free(output_tokens);
        s = VV_ERR_OUT_OF_MEMORY;
        goto cleanup_decode;
    }

    for (int i = 0; i < n_generated; i++) {
        char* decoded = NULL;
        vv_tokenizer_decode(ctx->tokenizer, &output_tokens[i], 1, &decoded);
        token_texts[i] = decoded ? decoded : "";
    }

    s = vv_postprocess_tokens(token_texts, n_generated, result);

    if (s == VV_OK && *result) {
        (*result)->duration = perf->audio_duration_sec;
    }

    for (int i = 0; i < n_generated; i++) {
        if (token_texts[i] && token_texts[i][0] != '\0')
            vv_free((void*)token_texts[i]);
    }
    vv_free((void*)token_texts);
    vv_free(output_tokens);

    perf->postprocess_ms = vv_time_ms() - t_step;

    /* ── Finalize metrics ── */
    perf->total_ms = vv_time_ms() - t_total_start;
    perf->rtf = (perf->audio_duration_sec > 0.001)
        ? (perf->total_ms / 1000.0) / perf->audio_duration_sec : 0.0;

    /* KV-cache stats */
    if (ctx->kv_cache) {
        perf->kv_cache_used = ctx->kv_cache->current_len;
        perf->kv_cache_max  = ctx->kv_cache->max_seq_len;
        perf->kv_cache_pct  = (perf->kv_cache_max > 0)
            ? 100.0f * (float)perf->kv_cache_used / (float)perf->kv_cache_max
            : 0.0f;
    }

    /* VRAM snapshot */
    {
        size_t vfree = 0, vtotal = 0;
        if (vv_cuda_get_device_info(ctx->gpu_id, &vtotal, &vfree,
                                     NULL) == VV_OK) {
            perf->vram_total_bytes = vtotal;
            perf->vram_free_bytes  = vfree;
            perf->vram_used_bytes  = vtotal - vfree;
        }
    }

    VV_LOG_I("inference: done (%.2f sec audio, %d tokens, %.1f ms total, "
             "RTF=%.3f)",
             perf->audio_duration_sec, n_generated,
             perf->total_ms, perf->rtf);

    /* Fall through to cleanup */

cleanup_decode:
    if (hidden_states_gpu) vv_cuda_free(hidden_states_gpu);
    if (normed_gpu) vv_cuda_free(normed_gpu);
    if (logits_gpu) vv_cuda_free(logits_gpu);
    if (hidden_one_gpu) vv_cuda_free(hidden_one_gpu);

    return s;
}

/* ─── Perf getter ───────────────────────────────────────────────────────── */

const vv_perf_metrics_t* vv_inference_get_perf(
    const vv_inference_ctx_t* ctx) {
    return ctx ? &ctx->last_perf : NULL;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

vv_status_t vv_inference_free(vv_inference_ctx_t* ctx) {
    if (!ctx) return VV_ERR_NULL_PTR;

    if (ctx->workspace) vv_cuda_free(ctx->workspace);
    if (ctx->embed_table_gpu) vv_cuda_free(ctx->embed_table_gpu);
    if (ctx->lm_head_gpu) vv_cuda_free(ctx->lm_head_gpu);
    if (ctx->final_norm_gpu) vv_cuda_free(ctx->final_norm_gpu);
    if (ctx->kv_cache) vv_kv_cache_free(ctx->kv_cache);
    if (ctx->compute_stream) vv_cuda_stream_destroy(ctx->compute_stream);
    if (ctx->transfer_stream) vv_cuda_stream_destroy(ctx->transfer_stream);

    if (ctx->tokenizer) vv_tokenizer_free(ctx->tokenizer);
    if (ctx->acoustic_encoder) vv_conv_vae_free(ctx->acoustic_encoder);
    if (ctx->semantic_encoder) vv_conv_vae_free(ctx->semantic_encoder);
    /* Connector weights are views into model, freed by vv_model_free */
    if (ctx->acoustic_connector) vv_free(ctx->acoustic_connector);
    if (ctx->semantic_connector) vv_free(ctx->semantic_connector);

    /* Free GPU layer weights BEFORE vv_model_free.
     * vv_tensor_free will see on_gpu=false/data=NULL and skip them. */
    if (ctx->model) {
        free_layer_gpu_weights(ctx->model);
        vv_model_free(ctx->model);
    }

    vv_gemm_cleanup();
    vv_free(ctx);

    VV_LOG_I("inference: context freed");
    return VV_OK;
}
