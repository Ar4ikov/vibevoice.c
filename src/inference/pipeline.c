/**
 * @file pipeline.c
 * @brief End-to-end inference pipeline: WAV -> transcription JSON.
 *
 * Supports multiple placement strategies based on VRAM budget:
 *   - VV_PLACE_ALL_GPU:       All layers + embed + lm_head on GPU.
 *   - VV_PLACE_STREAM_FULL:   Stream layers, embed + lm_head on GPU.
 *   - VV_PLACE_STREAM_EMBED:  Stream layers, embed on GPU, lm_head CPU.
 *   - VV_PLACE_STREAM_ALL:    Stream layers, embed + lm_head on CPU.
 *   - VV_PLACE_CPU_ONLY:      Everything on CPU, no CUDA at all.
 */

#include "vibevoice/inference.h"
#include "vibevoice/audio.h"
#include "vibevoice/text_tokenizer.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

/* Forward declarations — CUDA helpers */


extern vv_status_t vv_postprocess_tokens(
    const char** token_texts, int n_tokens,
    vv_transcription_t** result);

extern vv_status_t vv_transcription_to_json(const vv_transcription_t* tr,
                                              char** json_str);

extern vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                                      int32_t* token_id);

/* Special tokens (see special_tokens.c) */
extern const char* VV_TOKEN_IM_START;
extern const char* VV_TOKEN_IM_END;
extern const char* VV_TOKEN_ENDOFTEXT;
extern const char* VV_TOKEN_SPEECH_START;
extern const char* VV_TOKEN_SPEECH_PAD;
extern const char* VV_TOKEN_SPEECH_END;
extern const char* VV_TOKEN_START_TRANSCRIPT;
extern const char* VV_TOKEN_END_TRANSCRIPT;
extern bool vv_is_end_token(const vv_tokenizer_t* tok, int32_t token_id);

/* ─── FP32 <-> FP16 conversion helpers ──────────────────────────────────── */

static void float_to_half(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } v;
        v.f = src[i];
        uint32_t sign = (v.u >> 16) & 0x8000;
        int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (v.u >> 13) & 0x3FF;
        if (exp <= 0)      dst[i] = (uint16_t)sign;
        else if (exp >= 0x1F) dst[i] = (uint16_t)(sign | 0x7C00);
        else               dst[i] = (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
    }
}

static float half_to_float_single(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
    uint32_t frac = (uint32_t)h & 0x03FF;
    union { float f; uint32_t u; } u;
    if (expo == 0) u.u = sign;
    else if (expo == 0x1F) u.u = sign | 0x7F800000 | (frac << 13);
    else u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
    return u.f;
}

/* ─── Upload tensor to GPU helper (for all-on-GPU path) ─────────────────── */

static vv_status_t upload_tensor_to_gpu(vv_tensor_t* t, void* stream) {
    if (!t || !t->data || t->on_gpu) return VV_OK;
    void* gpu_ptr = NULL;
    vv_status_t s = vv_dev_alloc(&gpu_ptr, t->size_bytes);
    if (s != VV_OK) return s;
    s = vv_dev_memcpy_h2d(gpu_ptr, t->data, t->size_bytes, stream);
    if (s != VV_OK) { vv_dev_free(gpu_ptr); return s; }
    vv_free(t->data);
    t->data = gpu_ptr;
    t->on_gpu = true;
    return VV_OK;
}

/**
 * @brief Upload all NF4 layer weights (packed + scales + norms) to GPU.
 * Used only for VV_PLACE_ALL_GPU mode.
 */
static void attach_frontend(vv_inference_ctx_t* c);

/**
 * @brief Upload the first `n_layers` transformer layers to the GPU.
 *
 * Partial residency is the point: on a card that cannot hold all 28, the
 * layers that fit stay put and only the remainder is streamed per token, so
 * the PCIe cost scales with what is missing rather than with the whole model.
 */
static vv_status_t upload_layer_weights(vv_model_t* model, int n_layers,
                                        void* stream) {
    size_t total_bytes = 0;
    if (n_layers > model->num_layers) n_layers = model->num_layers;
    for (int i = 0; i < n_layers; i++) {
        vv_layer_weights_t* L = &model->layers[i];
        vv_status_t s;

        s = upload_tensor_to_gpu(&L->input_layernorm, stream);
        if (s != VV_OK) return s;
        total_bytes += L->input_layernorm.size_bytes;

        s = upload_tensor_to_gpu(&L->post_attn_layernorm, stream);
        if (s != VV_OK) return s;
        total_bytes += L->post_attn_layernorm.size_bytes;

        #define UPLOAD_WEIGHT(w) do {                                  \
            s = upload_tensor_to_gpu(&(w).tensor, stream);             \
            if (s != VV_OK) return s;                                  \
            total_bytes += (w).tensor.size_bytes;                      \
            if ((w).quant.scales.data) {                               \
                s = upload_tensor_to_gpu(&(w).quant.scales, stream);   \
                if (s != VV_OK) return s;                              \
                total_bytes += (w).quant.scales.size_bytes;            \
            }                                                          \
            if ((w).mins.data) {                                       \
                s = upload_tensor_to_gpu(&(w).mins, stream);           \
                if (s != VV_OK) return s;                              \
                total_bytes += (w).mins.size_bytes;                    \
            }                                                          \
            if ((w).bias.data) {                                       \
                s = upload_tensor_to_gpu(&(w).bias, stream);           \
                if (s != VV_OK) return s;                              \
                total_bytes += (w).bias.size_bytes;                    \
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
    VV_LOG_I("inference: %d/%d layers resident on GPU (%.1f MB)",
             n_layers, model->num_layers,
             (double)total_bytes / (1024.0 * 1024.0));
    return VV_OK;
}

/**
 * @brief Free GPU memory for layer tensors (call before vv_model_free).
 */
static void free_layer_gpu_weights(vv_model_t* model) {
    if (!model || !model->layers) return;
    for (int i = 0; i < model->num_layers; i++) {
        vv_layer_weights_t* L = &model->layers[i];
        #define FREE_GPU_TENSOR(t) do {                 \
            if ((t).on_gpu && (t).data) {               \
                vv_dev_free((t).data);                 \
                (t).data = NULL;                        \
                (t).on_gpu = false;                     \
            }                                           \
        } while(0)

        FREE_GPU_TENSOR(L->input_layernorm);
        FREE_GPU_TENSOR(L->post_attn_layernorm);

        #define FREE_GPU_WEIGHT(w) do {                 \
            FREE_GPU_TENSOR((w).tensor);                \
            FREE_GPU_TENSOR((w).quant.scales);          \
            FREE_GPU_TENSOR((w).mins);                  \
            FREE_GPU_TENSOR((w).bias);                  \
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

/* ═══════════════════════════════════════════════════════════════════════════
 * VRAM budget decision logic
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Calculate per-layer GPU size for one layer (NF4+scales+norms).
 */
static size_t calc_per_layer_gpu_bytes(const vv_model_t* model) {
    if (model->num_layers == 0) return 0;
    size_t total = 0;
    const vv_layer_weights_t* L = &model->layers[0];
    total += L->input_layernorm.size_bytes;
    total += L->post_attn_layernorm.size_bytes;

    #define ADD_W(w) do { total += (w).tensor.size_bytes; \
        if ((w).quant.scales.data) total += (w).quant.scales.size_bytes; \
        if ((w).mins.data) total += (w).mins.size_bytes; \
        if ((w).bias.data) total += (w).bias.size_bytes; \
    } while(0)
    ADD_W(L->attn.q_proj); ADD_W(L->attn.k_proj);
    ADD_W(L->attn.v_proj); ADD_W(L->attn.o_proj);
    ADD_W(L->mlp.gate_proj); ADD_W(L->mlp.up_proj);
    ADD_W(L->mlp.down_proj);
    #undef ADD_W
    return total;
}

static vv_placement_t decide_placement(
    float vram_budget, bool cpu_only,
    size_t free_vram,
    size_t all_layers_bytes,   /* total for 28 layers */
    size_t embed_bytes,
    size_t lm_head_bytes,
    size_t per_layer_bytes,
    size_t kv_cache_bytes,
    size_t workspace_bytes)
{
    if (cpu_only || vram_budget <= 0.0f) return VV_PLACE_CPU_ONLY;

    size_t available = (size_t)((double)free_vram * (double)vram_budget);
    size_t base = workspace_bytes + kv_cache_bytes;
    size_t staging_2 = per_layer_bytes * 2 + 16 * 256 * 2; /* 2 buffers */

    VV_LOG_I("budget: available %.1f MB, base %.1f MB (ws+kv), "
             "all_layers %.1f MB, embed %.1f MB, lm_head %.1f MB",
             (double)available / (1024.0*1024.0),
             (double)base / (1024.0*1024.0),
             (double)all_layers_bytes / (1024.0*1024.0),
             (double)embed_bytes / (1024.0*1024.0),
             (double)lm_head_bytes / (1024.0*1024.0));

    if (available < base) return VV_PLACE_CPU_ONLY;
    size_t remaining = available - base;

    if (remaining >= all_layers_bytes + embed_bytes + lm_head_bytes) {
        return VV_PLACE_ALL_GPU;
    }
    if (remaining >= staging_2 + embed_bytes + lm_head_bytes) {
        return VV_PLACE_STREAM_FULL;
    }
    if (remaining >= staging_2 + embed_bytes) {
        return VV_PLACE_STREAM_EMBED;
    }
    if (remaining >= staging_2) {
        return VV_PLACE_STREAM_ALL;
    }
    /* Not even enough for staging buffers */
    return VV_PLACE_CPU_ONLY;
}

static const char* placement_str(vv_placement_t p) {
    switch (p) {
        case VV_PLACE_ALL_GPU:      return "ALL_GPU";
        case VV_PLACE_STREAM_FULL:  return "STREAM_FULL";
        case VV_PLACE_STREAM_EMBED: return "STREAM_EMBED";
        case VV_PLACE_STREAM_ALL:   return "STREAM_ALL";
        case VV_PLACE_CPU_ONLY:     return "CPU_ONLY";
        default:                    return "?";
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Init
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               const vv_init_params_t* params,
                               vv_inference_ctx_t** ctx) {
    if (!model_dir || !ctx) return VV_ERR_NULL_PTR;

    /* Resolve params (NULL → defaults) */
    vv_init_params_t p = params ? *params : vv_init_params_default();
    bool cpu_only = p.cpu_only || p.vram_budget <= 0.0f;

    if (p.kv_format < 0 || p.kv_format >= VV_KV_FORMAT_COUNT) {
        VV_LOG_E("inference: unknown KV-cache format");
        return VV_ERR_INVALID_ARG;
    }
    /* Only the FP16 store has a CPU implementation of the append path. */
    if (cpu_only && p.kv_format != VV_KV_FP16) {
        VV_LOG_W("inference: KV format %s needs a device; "
                 "falling back to fp16 on CPU",
                 vv_kv_format_name((vv_kv_format_t)p.kv_format));
        p.kv_format = VV_KV_FP16;
    }

    VV_LOG_I("inference: initializing from '%s' %s (budget=%.0f%%)",
             model_dir,
             cpu_only ? "CPU-only" : "on GPU",
             cpu_only ? 0.0 : (double)p.vram_budget * 100.0);

    /* Set GPU (skip if CPU-only) */
    vv_status_t s = VV_OK;
    if (!cpu_only) {
        s = vv_dev_set_device(gpu_id);
        if (s != VV_OK) {
            VV_LOG_W("inference: failed to set GPU %d, falling back to CPU", gpu_id);
            cpu_only = true;
        }
    }

    /* Allocate context */
    vv_inference_ctx_t* c = (vv_inference_ctx_t*)vv_alloc(
        sizeof(vv_inference_ctx_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));
    c->gpu_id = gpu_id;
    c->use_gpu = !cpu_only;
    strncpy(c->model_dir, model_dir, sizeof(c->model_dir) - 1);

    /* Load model */
    s = vv_model_load(model_dir, &c->model);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to load model: %s", vv_status_str(s));
        vv_free(c);
        return s;
    }

    const vv_llm_config_t* llm = &c->model->config.llm;
    int max_seq = (p.max_seq_len > 0) ? p.max_seq_len : 32768;

    /* ── Decide placement strategy ── */
    if (cpu_only) {
        c->placement = VV_PLACE_CPU_ONLY;
    } else {
        size_t free_vram = 0, total_vram = 0;
        vv_dev_get_device_info(gpu_id, &total_vram, &free_vram, NULL);

        size_t per_layer = calc_per_layer_gpu_bytes(c->model);
        size_t all_layers = per_layer * (size_t)c->model->num_layers;
        size_t embed_sz = c->model->embed_tokens.size_bytes;
        size_t lm_head_sz = c->model->lm_head.size_bytes;

        size_t kv_per_token = vv_kv_cache_bytes(
            llm->num_hidden_layers, llm->num_key_value_heads,
            llm->head_dim, 1, p.kv_format);
        size_t ws_target = (size_t)512 * 1024 * 1024;

        /*
         * Shrink the KV window before giving up on resident weights: a
         * shorter context costs one long recording, while streaming the
         * layers costs every single token.
         */
        size_t available = (size_t)((double)free_vram * (double)p.vram_budget);
        size_t resident = ws_target + all_layers + embed_sz + lm_head_sz;
        if (available > resident) {
            size_t kv_room = (available - resident) / kv_per_token;
            if (kv_room < (size_t)max_seq) {
                int fitted = (int)(kv_room & ~(size_t)255);
                if (fitted >= 2048) {
                    VV_LOG_W("inference: KV window trimmed %d -> %d tokens to "
                             "keep the weights resident", max_seq, fitted);
                    max_seq = fitted;
                }
            }
        }

        size_t kv_total = (size_t)max_seq * kv_per_token;

        c->placement = decide_placement(
            p.vram_budget, false, free_vram,
            all_layers, embed_sz, lm_head_sz,
            per_layer, kv_total, ws_target);

        /*
         * When not everything fits, work out how many layers do. Two staging
         * buffers must be kept back for the ones that don't, plus embed and
         * lm_head if this placement keeps them on the GPU.
         */
        size_t fixed = ws_target + kv_total + 2 * per_layer;
        if (c->placement == VV_PLACE_STREAM_FULL)
            fixed += embed_sz + lm_head_sz;
        else if (c->placement == VV_PLACE_STREAM_EMBED)
            fixed += embed_sz;
        if (available > fixed) {
            size_t fit = (available - fixed) / per_layer;
            c->auto_resident_layers = fit > (size_t)llm->num_hidden_layers
                                      ? llm->num_hidden_layers : (int)fit;
        } else {
            c->auto_resident_layers = 0;
        }
    }

    /*
     * How many layers stay on the GPU. The placement enum only says whether
     * embed and lm_head fit; this is the finer knob, and the one that
     * actually decides the per-token PCIe bill.
     */
    if (c->placement == VV_PLACE_CPU_ONLY) {
        c->n_resident_layers = 0;
    } else if (p.gpu_layers >= 0) {
        c->n_resident_layers = p.gpu_layers < llm->num_hidden_layers
                               ? p.gpu_layers : llm->num_hidden_layers;
    } else if (c->placement == VV_PLACE_ALL_GPU) {
        c->n_resident_layers = llm->num_hidden_layers;
    } else {
        c->n_resident_layers = c->auto_resident_layers;
    }

    VV_LOG_I("inference: placement strategy = %s, %d/%d layers resident",
             placement_str(c->placement), c->n_resident_layers,
             llm->num_hidden_layers);

    /* ── CPU-only path: skip all CUDA ── */
    if (c->placement == VV_PLACE_CPU_ONLY) {
        c->use_gpu = false;
        s = vv_kv_cache_create(&c->kv_cache,
                                llm->num_hidden_layers,
                                llm->num_key_value_heads,
                                llm->head_dim,
                                max_seq, VV_KV_FP16, true);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to create CPU KV-cache");
            vv_model_free(c->model);
            vv_free(c);
            return s;
        }

        /* CPU workspace: FP32 intermediates */
        int hs = llm->hidden_size;
        int inter = llm->intermediate_size;
        int n_h = llm->num_attention_heads;
        int n_kv = llm->num_key_value_heads;
        int hd = llm->head_dim;
        /* Max per-layer: seq_len * (hs + n_h*hd + n_kv*hd + n_kv*hd + hs
         *                           + inter + inter + hs) + N*K for dequant */
        /* For decode (seq=1): relatively small. Allocate 512 MB CPU workspace. */
        c->workspace_size = (size_t)512 * 1024 * 1024;
        c->workspace = vv_alloc(c->workspace_size);
        if (!c->workspace) {
            c->workspace_size = (size_t)256 * 1024 * 1024;
            c->workspace = vv_alloc(c->workspace_size);
        }
        if (!c->workspace) {
            VV_LOG_E("inference: cannot allocate CPU workspace");
            vv_kv_cache_free(c->kv_cache);
            vv_model_free(c->model);
            vv_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }

        goto init_common;
    }

    /* ── GPU path: create streams ── */
    s = vv_dev_stream_create(&c->compute_stream);
    if (s != VV_OK) { vv_model_free(c->model); vv_free(c); return s; }
    s = vv_dev_stream_create(&c->transfer_stream);
    if (s != VV_OK) {
        vv_dev_stream_destroy(c->compute_stream);
        vv_model_free(c->model); vv_free(c); return s;
    }

    /* ── KV-cache on GPU ── */
    s = vv_kv_cache_create(&c->kv_cache,
                            llm->num_hidden_layers,
                            llm->num_key_value_heads,
                            llm->head_dim,
                            max_seq, p.kv_format, false);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to create KV-cache");
        goto fail_gpu;
    }

    /* ── Layer weights ── */
    if (c->n_resident_layers > 0) {
        s = upload_layer_weights(c->model, c->n_resident_layers,
                                 c->transfer_stream);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload layer weights");
            goto fail_gpu;
        }
    }
    s = vv_layer_pool_create(&c->layer_pool, c->model,
                             c->n_resident_layers >= c->model->num_layers);
    if (s == VV_OK && c->n_resident_layers < c->model->num_layers)
        vv_layer_pool_pin_host(c->model, c->n_resident_layers);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to create layer pool");
        goto fail_gpu;
    }

    /* ── GPU workspace ── */
    {
        static const size_t ws_sizes[] = {
            (size_t)512 * 1024 * 1024,
            (size_t)384 * 1024 * 1024,
            (size_t)256 * 1024 * 1024,
        };
        s = VV_ERR_CUDA_OOM;
        for (int wi = 0; wi < 3; wi++) {
            c->workspace_size = ws_sizes[wi];
            s = vv_dev_alloc(&c->workspace, c->workspace_size);
            if (s == VV_OK) break;
            VV_LOG_W("inference: failed to alloc %zu MB workspace, trying smaller",
                     c->workspace_size / (1024 * 1024));
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: cannot allocate workspace");
            goto fail_gpu;
        }
    }

    /* ── Embed table: on GPU unless STREAM_ALL ── */
    if (c->placement != VV_PLACE_STREAM_ALL &&
        c->model->embed_tokens.data) {
        size_t sz = c->model->embed_tokens.size_bytes;
        s = vv_dev_alloc(&c->embed_table_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->embed_table_gpu,
                                    c->model->embed_tokens.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_W("inference: embed_tokens GPU upload failed, using CPU path");
            if (c->embed_table_gpu) { vv_dev_free(c->embed_table_gpu); c->embed_table_gpu = NULL; }
        } else {
            VV_LOG_I("inference: embed_tokens uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    /* ── LM head: on GPU unless STREAM_EMBED or STREAM_ALL ── */
    if (c->placement != VV_PLACE_STREAM_EMBED &&
        c->placement != VV_PLACE_STREAM_ALL &&
        c->model->lm_head.data) {
        size_t sz = c->model->lm_head.size_bytes;
        s = vv_dev_alloc(&c->lm_head_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->lm_head_gpu,
                                    c->model->lm_head.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_W("inference: lm_head GPU upload failed, using CPU path");
            if (c->lm_head_gpu) { vv_dev_free(c->lm_head_gpu); c->lm_head_gpu = NULL; }
        } else {
            VV_LOG_I("inference: lm_head uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    /* ── Final norm: always on GPU for GPU mode ── */
    if (c->model->final_norm.data) {
        size_t sz = c->model->final_norm.size_bytes;
        s = vv_dev_alloc(&c->final_norm_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->final_norm_gpu,
                                    c->model->final_norm.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload final_norm to GPU");
        }
    }

    /* Sync transfers */
    vv_dev_stream_sync(c->transfer_stream);
    goto init_common;

fail_gpu:
    if (c->layer_pool) vv_layer_pool_free(c->layer_pool);
    if (c->kv_cache) vv_kv_cache_free(c->kv_cache);
    if (c->workspace && c->use_gpu) vv_dev_free(c->workspace);
    if (c->compute_stream) vv_dev_stream_destroy(c->compute_stream);
    if (c->transfer_stream) vv_dev_stream_destroy(c->transfer_stream);
    vv_model_free(c->model);
    vv_free(c);
    return s;

init_common:
    attach_frontend(c);

    VV_LOG_I("inference: initialized (%s, workspace=%zu MB, tokenizer=%s)",
             placement_str(c->placement),
             c->workspace_size / (1024 * 1024),
             c->tokenizer ? "yes" : "no");
    *ctx = c;
    return VV_OK;
}


/**
 * @brief Attach the speech encoders, connectors and tokenizer to a context.
 *
 * Shared by vv_inference_init and vv_inference_clone; everything here reads
 * the model's CPU-side tensors, so a clone builds its own without touching
 * the parent.
 */
static void attach_frontend(vv_inference_ctx_t* c) {
    const vv_llm_config_t* llm = &c->model->config.llm;
    vv_status_t s;

    s = vv_tokenizer_load(c->model_dir, &c->tokenizer);
    if (s != VV_OK)
        VV_LOG_W("inference: failed to load tokenizer from '%s'", c->model_dir);

    if (c->model->n_acoustic_weights > 0) {
        s = vv_conv_vae_init(c->model->acoustic_weights,
                              c->model->n_acoustic_weights,
                              &c->model->config.acoustic,
                              true, &c->acoustic_encoder);
        if (s != VV_OK) VV_LOG_W("inference: failed to init acoustic encoder");
    }
    if (c->model->n_semantic_weights > 0) {
        s = vv_conv_vae_init(c->model->semantic_weights,
                              c->model->n_semantic_weights,
                              &c->model->config.semantic,
                              false, &c->semantic_encoder);
        if (s != VV_OK) VV_LOG_W("inference: failed to init semantic encoder");
    }

    if (c->model->acoustic_connector_fc1.tensor.data) {
        c->acoustic_connector = (vv_connector_t*)vv_alloc(sizeof(vv_connector_t));
        if (c->acoustic_connector) {
            vv_connector_init(c->acoustic_connector,
                               c->model->config.acoustic_vae_dim,
                               llm->hidden_size);
            c->acoustic_connector->fc1_weight = c->model->acoustic_connector_fc1.tensor;
            c->acoustic_connector->fc1_bias   = c->model->acoustic_connector_fc1.quant.packed;
            c->acoustic_connector->norm_weight = c->model->acoustic_connector_norm.tensor;
            c->acoustic_connector->fc2_weight = c->model->acoustic_connector_fc2.tensor;
            c->acoustic_connector->fc2_bias   = c->model->acoustic_connector_fc2.quant.packed;
        }
    }
    if (c->model->semantic_connector_fc1.tensor.data) {
        c->semantic_connector = (vv_connector_t*)vv_alloc(sizeof(vv_connector_t));
        if (c->semantic_connector) {
            vv_connector_init(c->semantic_connector,
                               c->model->config.semantic_vae_dim,
                               llm->hidden_size);
            c->semantic_connector->fc1_weight = c->model->semantic_connector_fc1.tensor;
            c->semantic_connector->fc1_bias   = c->model->semantic_connector_fc1.quant.packed;
            c->semantic_connector->norm_weight = c->model->semantic_connector_norm.tensor;
            c->semantic_connector->fc2_weight = c->model->semantic_connector_fc2.tensor;
            c->semantic_connector->fc2_bias   = c->model->semantic_connector_fc2.quant.packed;
        }
    }

    if (c->use_gpu) {
        double t_w = vv_time_ms();
        if (c->acoustic_encoder) vv_conv_vae_warmup(c->acoustic_encoder);
        if (c->semantic_encoder) vv_conv_vae_warmup(c->semantic_encoder);
        VV_LOG_I("inference: speech encoder weights staged to GPU (%.0f ms)",
                 vv_time_ms() - t_w);
    }
}

vv_status_t vv_inference_clone(const vv_inference_ctx_t* parent,
                                const vv_init_params_t* params,
                                vv_inference_ctx_t** out) {
    if (!parent || !out) return VV_ERR_NULL_PTR;
    if (!parent->use_gpu || parent->placement != VV_PLACE_ALL_GPU) {
        VV_LOG_E("inference: clone needs a parent with all layers resident");
        return VV_ERR_UNSUPPORTED;
    }

    vv_init_params_t p = params ? *params : vv_init_params_default();
    const vv_llm_config_t* llm = &parent->model->config.llm;
    const int max_seq = (p.max_seq_len > 0) ? p.max_seq_len : 32768;

    vv_inference_ctx_t* c =
        (vv_inference_ctx_t*)vv_alloc(sizeof(vv_inference_ctx_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));

    c->is_clone  = true;
    c->gpu_id    = parent->gpu_id;
    c->use_gpu   = true;
    c->placement = parent->placement;
    c->model     = parent->model;
    c->layer_pool = parent->layer_pool;      /* all_resident: stateless */
    c->embed_table_gpu = parent->embed_table_gpu;
    c->lm_head_gpu     = parent->lm_head_gpu;
    c->final_norm_gpu  = parent->final_norm_gpu;
    memcpy(c->model_dir, parent->model_dir, sizeof(c->model_dir));

    vv_status_t s = vv_dev_stream_create(&c->compute_stream);
    if (s != VV_OK) { vv_free(c); return s; }
    s = vv_dev_stream_create(&c->transfer_stream);
    if (s != VV_OK) {
        vv_dev_stream_destroy(c->compute_stream);
        vv_free(c);
        return s;
    }

    s = vv_kv_cache_create(&c->kv_cache, llm->num_hidden_layers,
                           llm->num_key_value_heads, llm->head_dim,
                           max_seq, p.kv_format, false);
    if (s != VV_OK) goto fail;

    c->workspace_size = parent->workspace_size;
    s = vv_dev_alloc(&c->workspace, c->workspace_size);
    if (s != VV_OK) {
        c->workspace_size = (size_t)256 * 1024 * 1024;
        s = vv_dev_alloc(&c->workspace, c->workspace_size);
    }
    if (s != VV_OK) goto fail;

    attach_frontend(c);

    VV_LOG_I("inference: cloned context (workspace=%zu MB, kv=%.0f MB)",
             c->workspace_size / (1024 * 1024),
             (double)c->kv_cache->bytes_total / (1024.0 * 1024.0));
    *out = c;
    return VV_OK;

fail:
    if (c->kv_cache) vv_kv_cache_free(c->kv_cache);
    if (c->workspace) vv_dev_free(c->workspace);
    vv_dev_stream_destroy(c->compute_stream);
    vv_dev_stream_destroy(c->transfer_stream);
    vv_free(c);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Debug tensor dumping
 *
 * Set VV_DUMP_DIR to a directory and every intermediate below is written as a
 * raw little-endian binary blob, so tools/compare_ref.py can diff them against
 * the PyTorch reference dump element by element.
 * ═══════════════════════════════════════════════════════════════════════════ */


#define VV_ARGMAX_PARTIALS 256

static const char* dump_dir(void) {
    static const char* d = NULL;
    static bool probed = false;
    if (!probed) { d = getenv("VV_DUMP_DIR"); probed = true; }
    return (d && d[0]) ? d : NULL;
}

static void dump_raw(const char* name, const void* data, size_t bytes) {
    const char* d = dump_dir();
    if (!d || !data || bytes == 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", d, name);
    FILE* f = fopen(path, "wb");
    if (!f) { VV_LOG_W("dump: cannot write %s", path); return; }
    fwrite(data, 1, bytes, f);
    fclose(f);
    VV_LOG_D("dump: wrote %s (%zu bytes)", path, bytes);
}

static void dump_f32(const char* name, const float* v, size_t n) {
    dump_raw(name, v, n * sizeof(float));
}

static void dump_i32(const char* name, const int32_t* v, size_t n) {
    dump_raw(name, v, n * sizeof(int32_t));
}

static void dump_f16_as_f32(const char* name, const uint16_t* v, size_t n) {
    if (!dump_dir() || !v || n == 0) return;
    float* tmp = (float*)vv_alloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < n; i++) tmp[i] = half_to_float_single(v[i]);
    dump_f32(name, tmp, n);
    vv_free(tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ChatML prompt builder for ASR
 *
 * Format (from vibevoice_asr_processor.py):
 *   <|im_start|>system\n
 *   You are a helpful assistant that transcribes audio input
 *   into text output in JSON format.<|im_end|>\n
 *   <|im_start|>user\n
 *   <|object_ref_start|>[box_start × N]<|object_ref_end|>\n
 *   This is a {dur:.2f} seconds audio, please transcribe it
 *   with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * Audio embeddings replace the N box_start positions.
 * ═══════════════════════════════════════════════════════════════════════════ */

static vv_status_t build_asr_prompt(
    const vv_tokenizer_t* tok,
    int n_audio_frames,
    float audio_duration_sec,
    const char* context_info,       /* hotwords / extra info, may be NULL */
    int32_t** out_ids,
    int* out_seq_len,
    int* out_audio_offset)          /* index of first box_start token */
{
    /* Look up special token IDs */
    int im_start = vv_tokenizer_special_id(tok, VV_TOKEN_IM_START);
    int im_end   = vv_tokenizer_special_id(tok, VV_TOKEN_IM_END);
    int sp_start = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_START);
    int sp_pad   = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_PAD);
    int sp_end   = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_END);

    VV_LOG_D("prompt tokens: im_start=%d im_end=%d speech_start=%d "
             "speech_pad=%d speech_end=%d",
             im_start, im_end, sp_start, sp_pad, sp_end);

    if (im_start < 0 || im_end < 0 ||
        sp_start < 0 || sp_pad < 0 || sp_end < 0) {
        VV_LOG_E("prompt: missing critical special tokens in tokenizer");
        return VV_ERR_INVALID_ARG;
    }

    /*
     * Exact training/inference format, reproduced from
     * vibevoice_asr_processor.py::_process_single_audio + the ASR chat
     * template ("<|im_start|>{role}\n{content}<|im_end|>\n" per message).
     *
     * NOTE: the processor does NOT append a generation prompt — the model
     * emits "<|im_start|>assistant\n" itself as its first tokens.
     */
    static const char* SYSTEM_MSG =
        "<|im_start|>system\n"
        "You are a helpful assistant that transcribes audio input into text "
        "output in JSON format.<|im_end|>\n"
        "<|im_start|>user\n";

    int32_t *pre_ids = NULL, *post_ids = NULL;
    int n_pre = 0, n_post = 0;

    vv_status_t st = vv_tokenizer_encode(tok, SYSTEM_MSG, &pre_ids, &n_pre);
    if (st != VV_OK) return st;

    {
        char buf[1024];
        static const char* KEYS = "Start time, End time, Speaker ID, Content";
        if (context_info && context_info[0]) {
            snprintf(buf, sizeof(buf),
                     "\nThis is a %.2f seconds audio, with extra info: %s\n\n"
                     "Please transcribe it with these keys: %s<|im_end|>\n",
                     (double)audio_duration_sec, context_info, KEYS);
        } else {
            snprintf(buf, sizeof(buf),
                     "\nThis is a %.2f seconds audio, please transcribe it "
                     "with these keys: %s<|im_end|>\n",
                     (double)audio_duration_sec, KEYS);
        }
        st = vv_tokenizer_encode(tok, buf, &post_ids, &n_post);
        if (st != VV_OK) { vv_free(pre_ids); return st; }
    }

    /* [pre] <|object_ref_start|> [pad × N] <|object_ref_end|> [post] */
    int total = n_pre + 1 + n_audio_frames + 1 + n_post;
    int32_t* ids = (int32_t*)vv_alloc((size_t)total * sizeof(int32_t));
    if (!ids) { vv_free(pre_ids); vv_free(post_ids); return VV_ERR_OUT_OF_MEMORY; }

    int p = 0;
    for (int i = 0; i < n_pre; i++) ids[p++] = pre_ids[i];
    ids[p++] = sp_start;
    int audio_off = p;
    for (int i = 0; i < n_audio_frames; i++) ids[p++] = sp_pad;
    ids[p++] = sp_end;
    for (int i = 0; i < n_post; i++) ids[p++] = post_ids[i];

    vv_free(pre_ids);
    vv_free(post_ids);

    VV_LOG_I("prompt: %d tokens (audio frames %d at offset %d, audio=%.2f sec)",
             p, n_audio_frames, audio_off, (double)audio_duration_sec);
    {
        char line[4096];
        int w = 0, shown = 0;
        for (int i = 0; i < p && w < (int)sizeof(line) - 16; i++) {
            if (i == audio_off && n_audio_frames > 0) {
                w += snprintf(line + w, sizeof(line) - (size_t)w,
                              "[%d x%d] ", sp_pad, n_audio_frames);
                i += n_audio_frames - 1;
                continue;
            }
            w += snprintf(line + w, sizeof(line) - (size_t)w, "%d ", ids[i]);
            shown++;
        }
        VV_LOG_D("prompt ids (%d shown): %s", shown, line);
    }

    *out_ids = ids;
    *out_seq_len = p;
    *out_audio_offset = audio_off;
    return VV_OK;
}


/**
 * @brief Join hotwords into the processor's `context_info` string.
 *
 * The reference processor embeds it as "with extra info: <...>"; passing the
 * comma-joined hotword list there is how the model is steered towards rare
 * names in the upstream demo.
 */
static void build_context_info(const vv_inference_params_t* params,
                               char* buf, size_t buf_size) {
    buf[0] = '\0';
    if (!params || !params->hotwords || params->num_hotwords <= 0) return;
    size_t w = 0;
    for (int i = 0; i < params->num_hotwords; i++) {
        const char* hw = params->hotwords[i];
        if (!hw || !hw[0]) continue;
        int n = snprintf(buf + w, buf_size - w, "%s%s", w ? ", " : "", hw);
        if (n < 0 || (size_t)n >= buf_size - w) break;
        w += (size_t)n;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — GPU path
 * ═══════════════════════════════════════════════════════════════════════════ */

static vv_status_t transcribe_gpu(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    vv_perf_metrics_t* perf = &ctx->last_perf;
    double t_total_start = vv_time_ms();
    vv_status_t s;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    int hs = llm->hidden_size;
    int vocab_size = llm->vocab_size;
    int max_new_tokens = params ? params->max_new_tokens : 64000;
    if (max_new_tokens <= 0) max_new_tokens = 64000;

    perf->audio_duration_sec = (float)num_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = hs;
    perf->kv_format = ctx->kv_cache ? ctx->kv_cache->format : VV_KV_FP16;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    VV_LOG_I("inference: GPU transcribe %d samples (%.2f sec)",
             num_samples, perf->audio_duration_sec);

    if (!ctx->final_norm_gpu || !ctx->tokenizer) {
        VV_LOG_E("inference: missing critical components");
        return VV_ERR_NULL_PTR;
    }

    /* Check: need either GPU embed or CPU fallback */
    bool embed_on_cpu = (ctx->embed_table_gpu == NULL);
    bool lm_head_on_cpu = (ctx->lm_head_gpu == NULL);

    vv_kv_cache_reset(ctx->kv_cache);

    /* ═══ STEP 1: Audio encoding ═══ */
    double t_step = vv_time_ms();
    VV_LOG_I("inference: step 1 — encoding audio");

    float* acoustic_latents = NULL, *semantic_latents = NULL;
    float* acoustic_features = NULL, *semantic_features = NULL;
    float* combined_fp32 = NULL;
    int n_acoustic_frames = 0, n_semantic_frames = 0;

    /* Temporarily free GPU workspace to give Conv-VAE encoder more VRAM */
    void* saved_workspace = NULL;
    size_t saved_workspace_size = 0;
    if (ctx->use_gpu && ctx->workspace) {
        saved_workspace = ctx->workspace;
        saved_workspace_size = ctx->workspace_size;
        vv_dev_free(ctx->workspace);
        ctx->workspace = NULL;
        VV_LOG_D("inference: freed GPU workspace (%zu MB) for audio encoding",
                 saved_workspace_size / (1024*1024));
    }

    dump_f32("c_audio24k", audio_samples, (size_t)num_samples);

    if (ctx->acoustic_encoder &&
        ctx->acoustic_encoder->stages != NULL &&
        ctx->acoustic_encoder->input_conv.weight.data != NULL) {
        s = vv_conv_vae_encode_cpu(ctx->acoustic_encoder,
                                    audio_samples, num_samples,
                                    &acoustic_latents, &n_acoustic_frames);
        if (s != VV_OK) VV_LOG_W("inference: acoustic encoder failed");
    }
    if (ctx->semantic_encoder &&
        ctx->semantic_encoder->stages != NULL &&
        ctx->semantic_encoder->input_conv.weight.data != NULL) {
        s = vv_conv_vae_encode_cpu(ctx->semantic_encoder,
                                    audio_samples, num_samples,
                                    &semantic_latents, &n_semantic_frames);
        if (s != VV_OK) VV_LOG_W("inference: semantic encoder failed");
    }

    /* Re-allocate GPU workspace */
    if (saved_workspace_size > 0 && ctx->use_gpu) {
        s = vv_dev_alloc(&ctx->workspace, saved_workspace_size);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to re-allocate GPU workspace");
            return s;
        }
        ctx->workspace_size = saved_workspace_size;
        VV_LOG_D("inference: re-allocated GPU workspace (%zu MB)",
                 saved_workspace_size / (1024*1024));
    }

    int n_audio_frames;
    if (n_acoustic_frames > 0 && n_semantic_frames > 0)
        n_audio_frames = n_acoustic_frames < n_semantic_frames ?
                         n_acoustic_frames : n_semantic_frames;
    else if (n_acoustic_frames > 0) n_audio_frames = n_acoustic_frames;
    else if (n_semantic_frames > 0) n_audio_frames = n_semantic_frames;
    else { n_audio_frames = num_samples / 3200; if (n_audio_frames < 1) n_audio_frames = 1; }
    perf->audio_frames = n_audio_frames;

    if (acoustic_latents)
        dump_f32("c_ac_mean", acoustic_latents,
                 (size_t)n_audio_frames * (size_t)ctx->acoustic_encoder->vae_dim);
    if (semantic_latents)
        dump_f32("c_sem_mean", semantic_latents,
                 (size_t)n_audio_frames * (size_t)ctx->semantic_encoder->vae_dim);

    /* Connectors */
    if (acoustic_latents && ctx->acoustic_connector)
        vv_connector_forward_auto(ctx->acoustic_connector,
                                   acoustic_latents, n_audio_frames,
                                   &acoustic_features);
    if (semantic_latents && ctx->semantic_connector)
        vv_connector_forward_auto(ctx->semantic_connector,
                                   semantic_latents, n_audio_frames,
                                   &semantic_features);
    if (acoustic_latents) vv_free(acoustic_latents);
    if (semantic_latents) vv_free(semantic_latents);

    int combined_elems = n_audio_frames * hs;
    combined_fp32 = (float*)vv_alloc((size_t)combined_elems * sizeof(float));
    if (!combined_fp32) {
        if (acoustic_features) vv_free(acoustic_features);
        if (semantic_features) vv_free(semantic_features);
        return VV_ERR_OUT_OF_MEMORY;
    }
    if (acoustic_features && semantic_features)
        for (int i = 0; i < combined_elems; i++)
            combined_fp32[i] = acoustic_features[i] + semantic_features[i];
    else if (acoustic_features) memcpy(combined_fp32, acoustic_features, (size_t)combined_elems * sizeof(float));
    else if (semantic_features) memcpy(combined_fp32, semantic_features, (size_t)combined_elems * sizeof(float));
    else memset(combined_fp32, 0, (size_t)combined_elems * sizeof(float));
    if (acoustic_features)
        dump_f32("c_ac_feat", acoustic_features, (size_t)combined_elems);
    if (semantic_features)
        dump_f32("c_sem_feat", semantic_features, (size_t)combined_elems);
    dump_f32("c_combined", combined_fp32, (size_t)combined_elems);

    if (acoustic_features) vv_free(acoustic_features);
    if (semantic_features) vv_free(semantic_features);

    /* ═══ Diagnostic: check feature statistics ═══ */
    {
        float fmin = combined_fp32[0], fmax = combined_fp32[0];
        float fsum = 0.0f;
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < combined_elems; i++) {
            float v = combined_fp32[i];
            if (v != v) nan_count++;
            else if (v > 1e30f || v < -1e30f) inf_count++;
            else { if (v < fmin) fmin = v; if (v > fmax) fmax = v; }
            fsum += v;
        }
        float fmean = fsum / (float)combined_elems;
        VV_LOG_D("inference: combined features [%d x %d]: min=%.4f max=%.4f mean=%.6f nan=%d inf=%d",
                 n_audio_frames, hs, fmin, fmax, fmean, nan_count, inf_count);
    }

    uint16_t* combined_fp16 = (uint16_t*)vv_alloc((size_t)combined_elems * sizeof(uint16_t));
    if (!combined_fp16) { vv_free(combined_fp32); return VV_ERR_OUT_OF_MEMORY; }
    float_to_half(combined_fp32, combined_fp16, combined_elems);
    vv_free(combined_fp32);
    combined_fp32 = NULL;

    perf->audio_encode_ms = vv_time_ms() - t_step;

    VV_LOG_D("inference: step 1 complete — audio encoded + connectors done (%.0f ms)",
             vv_time_ms() - t_step);

    /* ═══ STEP 2: Build ChatML input sequence ═══ */
    t_step = vv_time_ms();

    int32_t* input_ids = NULL;
    int seq_len = 0;
    int audio_offset = 0;  /* index where box_start tokens (= audio frames) begin */

    char ctx_info[512];
    build_context_info(params, ctx_info, sizeof(ctx_info));

    s = build_asr_prompt(ctx->tokenizer, n_audio_frames,
                          perf->audio_duration_sec, ctx_info,
                          &input_ids, &seq_len, &audio_offset);
    if (s != VV_OK) { vv_free(combined_fp16); return s; }

    perf->prefill_tokens = seq_len;
    dump_i32("c_input_ids", input_ids, (size_t)seq_len);

    size_t hidden_bytes = (size_t)seq_len * (size_t)hs * 2;
    void* hidden_states_gpu = NULL;
    int32_t* input_ids_gpu = NULL;

    s = vv_dev_alloc(&hidden_states_gpu, hidden_bytes);
    if (s != VV_OK) { vv_free(input_ids); vv_free(combined_fp16); return s; }

    /* Embedding: all tokens (text + speech_pad placeholders) */
    if (embed_on_cpu) {
        float* embed_fp32 = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
        if (!embed_fp32) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); vv_free(combined_fp16); return VV_ERR_OUT_OF_MEMORY; }
        vv_embedding_cpu(ctx->model->embed_tokens.data, input_ids, embed_fp32, seq_len, hs);
        uint16_t* embed_fp16 = (uint16_t*)vv_alloc(hidden_bytes);
        if (!embed_fp16) { vv_free(embed_fp32); vv_dev_free(hidden_states_gpu); vv_free(input_ids); vv_free(combined_fp16); return VV_ERR_OUT_OF_MEMORY; }
        float_to_half(embed_fp32, embed_fp16, seq_len * hs);
        vv_free(embed_fp32);
        vv_dev_memcpy_h2d(hidden_states_gpu, embed_fp16, hidden_bytes, ctx->compute_stream);
        vv_free(embed_fp16);
    } else {
        s = vv_dev_alloc((void**)&input_ids_gpu, (size_t)seq_len * sizeof(int32_t));
        if (s != VV_OK) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); vv_free(combined_fp16); return s; }
        vv_dev_memcpy_h2d(input_ids_gpu, input_ids, (size_t)seq_len * sizeof(int32_t), ctx->compute_stream);
        s = vv_embedding_dev(ctx->embed_table_gpu, input_ids_gpu, hidden_states_gpu, seq_len, hs, ctx->compute_stream);
        vv_dev_free(input_ids_gpu);
        if (s != VV_OK) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); vv_free(combined_fp16); return s; }
    }
    vv_free(input_ids);

    /* Overlay audio features at the speech_pad positions */
    if (n_audio_frames > 0) {
        void* audio_dst = (uint8_t*)hidden_states_gpu
                        + (size_t)audio_offset * (size_t)hs * 2;
        vv_dev_memcpy_h2d(audio_dst, combined_fp16,
                            (size_t)n_audio_frames * (size_t)hs * 2,
                            ctx->compute_stream);
    }
    vv_free(combined_fp16);
    vv_dev_stream_sync(ctx->compute_stream);

    /* ── Diagnostic: compare text-embedding vs audio-feature magnitudes ── */
    {
        const int sample_dim = (hs < 32) ? hs : 32;  /* sample first 32 dims */
        uint16_t diag_buf[32];
        float sum_sq;

        /* Text embedding: read token at position 0 (im_start) */
        vv_dev_memcpy_d2h(diag_buf, hidden_states_gpu,
                            (size_t)sample_dim * 2, NULL);
        sum_sq = 0.0f;
        for (int d = 0; d < sample_dim; d++) {
            float v = half_to_float_single(diag_buf[d]);
            sum_sq += v * v;
        }
        VV_LOG_I("diag: text embed[0] L2(first %d dims) = %.6f", sample_dim, sqrtf(sum_sq));

        /* Audio features: read first audio frame at audio_offset */
        if (n_audio_frames > 0) {
            void* audio_pos = (uint8_t*)hidden_states_gpu
                            + (size_t)audio_offset * (size_t)hs * 2;
            vv_dev_memcpy_d2h(diag_buf, audio_pos,
                                (size_t)sample_dim * 2, NULL);
            sum_sq = 0.0f;
            for (int d = 0; d < sample_dim; d++) {
                float v = half_to_float_single(diag_buf[d]);
                sum_sq += v * v;
            }
            VV_LOG_I("diag: audio feat[0] L2(first %d dims) = %.6f", sample_dim, sqrtf(sum_sq));
        }
    }

    perf->sequence_build_ms = vv_time_ms() - t_step;

    if (dump_dir()) {
        size_t n = (size_t)seq_len * (size_t)hs;
        uint16_t* h = (uint16_t*)vv_alloc(n * 2);
        if (h) {
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(h, hidden_states_gpu, n * 2, NULL);
            dump_f16_as_f32("c_embeds", h, n);
            vv_free(h);
        }
    }

    /* ═══ STEP 3: LLM Prefill ═══ */
    t_step = vv_time_ms();
    VV_LOG_I("inference: prefill %d tokens, %d layers", seq_len, ctx->model->num_layers);

    s = vv_decoder_prefill(ctx->model, hidden_states_gpu, seq_len,
                            ctx->kv_cache, ctx->layer_pool,
                            ctx->workspace, ctx->workspace_size,
                            ctx->compute_stream, ctx->transfer_stream);
    if (s != VV_OK) {
        VV_LOG_E("inference: prefill failed: %s", vv_status_str(s));
        vv_dev_free(hidden_states_gpu);
        return s;
    }

    if (dump_dir()) {
        size_t n = (size_t)seq_len * (size_t)hs;
        uint16_t* h = (uint16_t*)vv_alloc(n * 2);
        if (h) {
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(h, hidden_states_gpu, n * 2, NULL);
            dump_f16_as_f32("c_prefill_hidden", h, n);
            vv_free(h);
        }
    }

    /* RMSNorm → lm_head → first token */
    size_t one_hidden = (size_t)hs * 2;
    void* last_hidden_gpu = (uint8_t*)hidden_states_gpu + (size_t)(seq_len-1) * one_hidden;

    void* normed_gpu = NULL;
    void* logits_gpu = NULL;
    void* hidden_one_gpu = NULL;
    /*
     * Everything the decode loop touches is allocated once here: cudaMalloc
     * serialises against the stream, so even a 4-byte allocation per token
     * would cost more than the layer it feeds.
     */
    void* logits_f32_gpu = NULL;
    void* argmax_v_gpu   = NULL;
    void* argmax_i_gpu   = NULL;
    void* token_out_gpu  = NULL;
    int32_t* tok_id_gpu  = NULL;
    s = vv_dev_alloc(&normed_gpu, one_hidden);
    if (s != VV_OK) { vv_dev_free(hidden_states_gpu); return s; }
    s = vv_dev_alloc(&hidden_one_gpu, one_hidden);
    if (s != VV_OK) { vv_dev_free(normed_gpu); vv_dev_free(hidden_states_gpu); return s; }
    if (!lm_head_on_cpu) {
        if (vv_dev_alloc(&logits_f32_gpu, (size_t)vocab_size * sizeof(float)) != VV_OK ||
            vv_dev_alloc(&argmax_v_gpu, VV_ARGMAX_PARTIALS * sizeof(float)) != VV_OK ||
            vv_dev_alloc(&argmax_i_gpu, VV_ARGMAX_PARTIALS * sizeof(int32_t)) != VV_OK ||
            vv_dev_alloc(&token_out_gpu, sizeof(int32_t)) != VV_OK ||
            vv_dev_alloc((void**)&tok_id_gpu, sizeof(int32_t)) != VV_OK) {
            s = VV_ERR_CUDA_OOM;
            goto cleanup_decode;
        }
    }

    s = vv_rmsnorm_dev(last_hidden_gpu, ctx->final_norm_gpu,
                         normed_gpu, 1, hs, llm->rms_norm_eps,
                         ctx->compute_stream);
    if (s != VV_OK) goto cleanup_decode;

    /* LM head — GPU or CPU */
    int32_t token_id;
    if (lm_head_on_cpu) {
        /* Download normed, do GEMM on CPU, argmax on CPU */
        uint16_t* normed_cpu = (uint16_t*)vv_alloc(one_hidden);
        if (!normed_cpu) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
        vv_dev_stream_sync(ctx->compute_stream);
        vv_dev_memcpy_d2h(normed_cpu, normed_gpu, one_hidden, NULL);
        /* FP16 → FP32 */
        float* normed_f32 = (float*)vv_alloc((size_t)hs * sizeof(float));
        float* logits_f32 = (float*)vv_alloc((size_t)vocab_size * sizeof(float));
        float* lm_w_f32 = NULL;
        if (!normed_f32 || !logits_f32) {
            if (normed_f32) vv_free(normed_f32);
            if (logits_f32) vv_free(logits_f32);
            vv_free(normed_cpu);
            s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode;
        }
        for (int d = 0; d < hs; d++) normed_f32[d] = half_to_float_single(normed_cpu[d]);
        vv_free(normed_cpu);
        /* lm_head GEMM: [1, hs] @ [vocab, hs]^T = [1, vocab] */
        lm_w_f32 = (float*)vv_alloc((size_t)vocab_size * hs * sizeof(float));
        if (!lm_w_f32) { vv_free(normed_f32); vv_free(logits_f32); s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
        {
            const uint16_t* lm = (const uint16_t*)ctx->model->lm_head.data;
            for (int i = 0; i < vocab_size * hs; i++) lm_w_f32[i] = half_to_float_single(lm[i]);
        }
        /* output[j] = sum_k normed[k] * lm_head[j][k] */
        for (int j = 0; j < vocab_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < hs; k++) sum += normed_f32[k] * lm_w_f32[j * hs + k];
            logits_f32[j] = sum;
        }
        vv_free(lm_w_f32); vv_free(normed_f32);
        vv_sample_greedy_cpu(logits_f32, vocab_size, &token_id);
        vv_free(logits_f32);
    } else {
        s = vv_lm_head_gemv_dev(normed_gpu, ctx->lm_head_gpu, logits_f32_gpu,
                                  vocab_size, hs, ctx->compute_stream);
        if (s != VV_OK) goto cleanup_decode;
        s = vv_argmax_dev(logits_f32_gpu, vocab_size, argmax_v_gpu,
                            argmax_i_gpu, token_out_gpu, NULL,
                            ctx->compute_stream);
        if (s != VV_OK) goto cleanup_decode;
        vv_dev_stream_sync(ctx->compute_stream);
        vv_dev_memcpy_d2h(&token_id, token_out_gpu, sizeof(int32_t), NULL);
        if (dump_dir()) {
            float* lg = (float*)vv_alloc((size_t)vocab_size * sizeof(float));
            if (lg) {
                vv_dev_memcpy_d2h(lg, logits_f32_gpu,
                                    (size_t)vocab_size * sizeof(float), NULL);
                dump_f32("c_prefill_logits", lg, (size_t)vocab_size);
                vv_free(lg);
            }
        }
    }

    vv_dev_free(hidden_states_gpu); hidden_states_gpu = NULL;

    perf->prefill_ms = vv_time_ms() - t_step;
    perf->ttft_ms = vv_time_ms() - t_total_start;
    (void)0;
    perf->prefill_tok_per_sec = (perf->prefill_ms > 0.001)
        ? (double)seq_len / perf->prefill_ms * 1000.0 : 0.0;
    VV_LOG_I("inference: prefill %.1f ms (%.0f tok/s), TTFT=%.1f ms",
             perf->prefill_ms, perf->prefill_tok_per_sec, perf->ttft_ms);

    /* ═══ STEP 4: Autoregressive decode ═══ */
    t_step = vv_time_ms();
    int out_cap = max_new_tokens < 8192 ? max_new_tokens : 8192;
    int32_t* output_tokens = (int32_t*)vv_alloc((size_t)out_cap * sizeof(int32_t));
    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
    int n_generated = 0;

    /* Split the decode cost so the next optimisation targets the real hot
     * spot: 28 transformer layers vs the 152k-row LM head + sampling. */
    double t_layers_ms = 0.0, t_head_ms = 0.0, t_embed_ms = 0.0;

    /*
     * Live token echo. Worth watching during a long transcription, pure
     * noise inside the chat and mic loops, so it follows the log level.
     */
    const bool echo_tokens = vv_log_get_level() >= VV_LOG_INFO;
    if (echo_tokens) fprintf(stderr, "\n--- token stream ---\n");
    if (!vv_is_end_token(ctx->tokenizer, token_id)) {
        output_tokens[n_generated++] = token_id;
        /* Stream first token */
        char* first_text = NULL;
        vv_tokenizer_decode(ctx->tokenizer, &token_id, 1, &first_text);
        if (first_text) {
            if (echo_tokens) fprintf(stderr, "%s", first_text);
            fflush(stderr);
            vv_free(first_text);
        }
    }

    while (n_generated < max_new_tokens &&
           !vv_is_end_token(ctx->tokenizer, token_id)) {

        double t_tok = vv_time_ms();

        /* Embed token */
        if (embed_on_cpu) {
            float embed_f32[4096]; /* head_dim <= 4096 */
            vv_embedding_cpu(ctx->model->embed_tokens.data, &token_id,
                              embed_f32, 1, hs);
            uint16_t embed_h[4096];
            float_to_half(embed_f32, embed_h, hs);
            vv_dev_memcpy_h2d(hidden_one_gpu, embed_h, one_hidden,
                                ctx->compute_stream);
        } else {
            int32_t tok_buf[1]; tok_buf[0] = token_id;
            vv_dev_memcpy_h2d(tok_id_gpu, tok_buf, sizeof(int32_t), ctx->compute_stream);
            s = vv_embedding_dev(ctx->embed_table_gpu, tok_id_gpu,
                                   hidden_one_gpu, 1, hs, ctx->compute_stream);
            if (s != VV_OK) break;
        }

        vv_dev_stream_sync(ctx->compute_stream);
        t_embed_ms += vv_time_ms() - t_tok;
        t_tok = vv_time_ms();

        /* Decoder step */
        s = vv_decoder_step(ctx->model, hidden_one_gpu, ctx->kv_cache,
                             ctx->layer_pool, ctx->workspace,
                             ctx->workspace_size, ctx->compute_stream,
                             ctx->transfer_stream);
        if (s != VV_OK) { VV_LOG_E("inference: decode step %d failed", n_generated); break; }

        vv_dev_stream_sync(ctx->compute_stream);
        t_layers_ms += vv_time_ms() - t_tok;
        t_tok = vv_time_ms();

        /* RMSNorm + LM head + sample */
        s = vv_rmsnorm_dev(hidden_one_gpu, ctx->final_norm_gpu,
                             normed_gpu, 1, hs, llm->rms_norm_eps,
                             ctx->compute_stream);
        if (s != VV_OK) break;

        if (lm_head_on_cpu) {
            uint16_t normed_h[4096];
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(normed_h, normed_gpu, one_hidden, NULL);
            float normed_f[4096], logits_f[152064]; /* max vocab */
            for (int d = 0; d < hs; d++) normed_f[d] = half_to_float_single(normed_h[d]);
            /* Simplified: use model->lm_head directly (FP16 on CPU) */
            const uint16_t* lm = (const uint16_t*)ctx->model->lm_head.data;
            for (int j = 0; j < vocab_size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hs; k++)
                    sum += normed_f[k] * half_to_float_single(lm[j*hs+k]);
                logits_f[j] = sum;
            }
            vv_sample_greedy_cpu(logits_f, vocab_size, &token_id);
        } else {
            s = vv_lm_head_gemv_dev(normed_gpu, ctx->lm_head_gpu,
                                      logits_f32_gpu, vocab_size, hs,
                                      ctx->compute_stream);
            if (s != VV_OK) break;
            s = vv_argmax_dev(logits_f32_gpu, vocab_size, argmax_v_gpu,
                                argmax_i_gpu, token_out_gpu, NULL,
                                ctx->compute_stream);
            if (s != VV_OK) break;
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(&token_id, token_out_gpu, sizeof(int32_t), NULL);
        }

        t_head_ms += vv_time_ms() - t_tok;

        if (vv_is_end_token(ctx->tokenizer, token_id)) break;

        if (n_generated >= out_cap) {
            out_cap *= 2;
            output_tokens = (int32_t*)vv_realloc(output_tokens,
                (size_t)out_cap * sizeof(int32_t));
            if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
        }
        output_tokens[n_generated++] = token_id;

        /* Stream token text to stderr for live monitoring */
        {
            char* tok_text = NULL;
            vv_tokenizer_decode(ctx->tokenizer, &token_id, 1, &tok_text);
            if (tok_text) {
                if (echo_tokens) fprintf(stderr, "%s", tok_text);
                fflush(stderr);
                vv_free(tok_text);
            }
        }

        if ((n_generated % 500) == 0) {
            double elapsed = vv_time_ms() - t_step;
            VV_LOG_I("inference: decoded %d tokens (%.1f tok/s)...",
                     n_generated, (elapsed > 0.001) ? (double)n_generated / elapsed * 1000.0 : 0.0);
        }
    }
    if (echo_tokens)
        fprintf(stderr, "\n--- end stream (%d tokens) ---\n",
                n_generated);
    fflush(stderr);

    perf->decode_layers_ms = t_layers_ms;
    perf->decode_head_ms   = t_head_ms;
    perf->decode_embed_ms  = t_embed_ms;
    perf->decode_ms = vv_time_ms() - t_step;
    perf->decode_tokens = n_generated;
    perf->decode_tok_per_sec = (perf->decode_ms > 0.001)
        ? (double)n_generated / perf->decode_ms * 1000.0 : 0.0;

    /* ═══ STEP 5: Post-processing ═══ */
    t_step = vv_time_ms();
    const char** token_texts = (const char**)vv_alloc((size_t)n_generated * sizeof(char*));
    if (!token_texts) { vv_free(output_tokens); s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
    for (int i = 0; i < n_generated; i++) {
        char* decoded = NULL;
        vv_tokenizer_decode(ctx->tokenizer, &output_tokens[i], 1, &decoded);
        token_texts[i] = decoded ? decoded : "";
    }
    s = vv_postprocess_tokens(token_texts, n_generated, result);
    if (s == VV_OK && *result) (*result)->duration = perf->audio_duration_sec;
    for (int i = 0; i < n_generated; i++)
        if (token_texts[i] && token_texts[i][0] != '\0') vv_free((void*)token_texts[i]);
    vv_free((void*)token_texts);
    vv_free(output_tokens);
    perf->postprocess_ms = vv_time_ms() - t_step;

    /* Finalize metrics */
    perf->total_ms = vv_time_ms() - t_total_start;
    perf->rtf = (perf->audio_duration_sec > 0.001)
        ? (perf->total_ms / 1000.0) / perf->audio_duration_sec : 0.0;
    if (ctx->kv_cache) {
        perf->kv_cache_used = ctx->kv_cache->current_len;
        perf->kv_cache_max  = ctx->kv_cache->max_seq_len;
        perf->kv_cache_pct  = (perf->kv_cache_max > 0)
            ? 100.0f * (float)perf->kv_cache_used / (float)perf->kv_cache_max : 0.0f;
    }
    {
        size_t vfree = 0, vtotal = 0;
        if (vv_dev_get_device_info(ctx->gpu_id, &vtotal, &vfree, NULL) == VV_OK) {
            perf->vram_total_bytes = vtotal;
            perf->vram_free_bytes  = vfree;
            perf->vram_used_bytes  = vtotal - vfree;
        }
    }

cleanup_decode:
    if (hidden_states_gpu) vv_dev_free(hidden_states_gpu);
    if (normed_gpu) vv_dev_free(normed_gpu);
    if (logits_gpu) vv_dev_free(logits_gpu);
    if (hidden_one_gpu) vv_dev_free(hidden_one_gpu);
    if (logits_f32_gpu) vv_dev_free(logits_f32_gpu);
    if (argmax_v_gpu) vv_dev_free(argmax_v_gpu);
    if (argmax_i_gpu) vv_dev_free(argmax_i_gpu);
    if (token_out_gpu) vv_dev_free(token_out_gpu);
    if (tok_id_gpu) vv_dev_free(tok_id_gpu);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — CPU path
 * ═══════════════════════════════════════════════════════════════════════════ */

static vv_status_t transcribe_cpu(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    vv_perf_metrics_t* perf = &ctx->last_perf;
    double t_total_start = vv_time_ms();
    vv_status_t s;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    int hs = llm->hidden_size;
    int vocab_size = llm->vocab_size;
    int max_new_tokens = params ? params->max_new_tokens : 64000;
    if (max_new_tokens <= 0) max_new_tokens = 64000;

    perf->audio_duration_sec = (float)num_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = hs;
    perf->kv_format = VV_KV_FP16;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    VV_LOG_I("inference: CPU transcribe %d samples (%.2f sec)",
             num_samples, perf->audio_duration_sec);

    if (!ctx->tokenizer) { VV_LOG_E("inference: no tokenizer"); return VV_ERR_NULL_PTR; }
    vv_kv_cache_reset(ctx->kv_cache);

    /* ═══ STEP 1: Audio encoding (same as GPU path) ═══ */
    double t_step = vv_time_ms();
    float* acoustic_latents = NULL, *semantic_latents = NULL;
    float* acoustic_features = NULL, *semantic_features = NULL;
    int n_acoustic_frames = 0, n_semantic_frames = 0;

    /* Temporarily free GPU workspace to give Conv-VAE encoder more VRAM */
    void* saved_ws2 = NULL;
    size_t saved_ws2_size = 0;
    if (ctx->use_gpu && ctx->workspace) {
        saved_ws2 = ctx->workspace;
        saved_ws2_size = ctx->workspace_size;
        vv_dev_free(ctx->workspace);
        ctx->workspace = NULL;
    }

    if (ctx->acoustic_encoder && ctx->acoustic_encoder->stages &&
        ctx->acoustic_encoder->input_conv.weight.data) {
        vv_conv_vae_encode_cpu(ctx->acoustic_encoder, audio_samples, num_samples,
                                &acoustic_latents, &n_acoustic_frames);
    }
    if (ctx->semantic_encoder && ctx->semantic_encoder->stages &&
        ctx->semantic_encoder->input_conv.weight.data) {
        vv_conv_vae_encode_cpu(ctx->semantic_encoder, audio_samples, num_samples,
                                &semantic_latents, &n_semantic_frames);
    }

    if (saved_ws2_size > 0 && ctx->use_gpu) {
        vv_dev_alloc(&ctx->workspace, saved_ws2_size);
        ctx->workspace_size = saved_ws2_size;
    }

    int n_audio_frames;
    if (n_acoustic_frames > 0 && n_semantic_frames > 0)
        n_audio_frames = n_acoustic_frames < n_semantic_frames ? n_acoustic_frames : n_semantic_frames;
    else if (n_acoustic_frames > 0) n_audio_frames = n_acoustic_frames;
    else if (n_semantic_frames > 0) n_audio_frames = n_semantic_frames;
    else { n_audio_frames = num_samples / 3200; if (n_audio_frames < 1) n_audio_frames = 1; }
    perf->audio_frames = n_audio_frames;

    if (acoustic_latents && ctx->acoustic_connector)
        vv_connector_forward_auto(ctx->acoustic_connector, acoustic_latents, n_audio_frames, &acoustic_features);
    if (semantic_latents && ctx->semantic_connector)
        vv_connector_forward_auto(ctx->semantic_connector, semantic_latents, n_audio_frames, &semantic_features);
    if (acoustic_latents) vv_free(acoustic_latents);
    if (semantic_latents) vv_free(semantic_latents);

    int combined_elems = n_audio_frames * hs;
    float* combined = (float*)vv_alloc((size_t)combined_elems * sizeof(float));
    if (!combined) {
        if (acoustic_features) vv_free(acoustic_features);
        if (semantic_features) vv_free(semantic_features);
        return VV_ERR_OUT_OF_MEMORY;
    }
    if (acoustic_features && semantic_features)
        for (int i = 0; i < combined_elems; i++) combined[i] = acoustic_features[i] + semantic_features[i];
    else if (acoustic_features) memcpy(combined, acoustic_features, (size_t)combined_elems * sizeof(float));
    else if (semantic_features) memcpy(combined, semantic_features, (size_t)combined_elems * sizeof(float));
    else memset(combined, 0, (size_t)combined_elems * sizeof(float));
    if (acoustic_features) vv_free(acoustic_features);
    if (semantic_features) vv_free(semantic_features);

    perf->audio_encode_ms = vv_time_ms() - t_step;

    /* ═══ STEP 2: Build ChatML input sequence (FP32) ═══ */
    t_step = vv_time_ms();
    int32_t* prompt_ids = NULL;
    int seq_len = 0;
    int audio_offset = 0;

    char ctx_info[512];
    build_context_info(params, ctx_info, sizeof(ctx_info));

    s = build_asr_prompt(ctx->tokenizer, n_audio_frames,
                          perf->audio_duration_sec, ctx_info,
                          &prompt_ids, &seq_len, &audio_offset);
    if (s != VV_OK) { vv_free(combined); return s; }
    perf->prefill_tokens = seq_len;

    /* Allocate FP32 hidden states on CPU */
    float* hidden = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
    if (!hidden) { vv_free(prompt_ids); vv_free(combined); return VV_ERR_OUT_OF_MEMORY; }

    /* Embed all prompt tokens */
    vv_embedding_cpu(ctx->model->embed_tokens.data, prompt_ids, hidden, seq_len, hs);
    vv_free(prompt_ids);

    /* Overlay audio features at speech_pad positions */
    if (n_audio_frames > 0) {
        memcpy(hidden + (size_t)audio_offset * hs,
               combined, (size_t)combined_elems * sizeof(float));
    }
    vv_free(combined);

    perf->sequence_build_ms = vv_time_ms() - t_step;

    /* ═══ STEP 3: Prefill ═══ */
    t_step = vv_time_ms();
    s = vv_decoder_prefill_cpu(ctx->model, hidden, seq_len, ctx->kv_cache,
                                (float*)ctx->workspace, ctx->workspace_size);
    if (s != VV_OK) { vv_free(hidden); return s; }

    /* Final norm + lm_head + sample (CPU FP32) */
    float* norm_w = (float*)vv_alloc((size_t)hs * sizeof(float));
    if (!norm_w) { vv_free(hidden); return VV_ERR_OUT_OF_MEMORY; }
    {
        const uint16_t* nw = (const uint16_t*)ctx->model->final_norm.data;
        for (int d = 0; d < hs; d++) norm_w[d] = half_to_float_single(nw[d]);
    }

    float* normed = (float*)vv_alloc((size_t)hs * sizeof(float));
    float* logits = (float*)vv_alloc((size_t)vocab_size * sizeof(float));
    float* hidden_one = (float*)vv_alloc((size_t)hs * sizeof(float));
    if (!normed || !logits || !hidden_one) {
        if (normed) vv_free(normed);
        if (logits) vv_free(logits);
        if (hidden_one) vv_free(hidden_one);
        vv_free(norm_w); vv_free(hidden);
        return VV_ERR_OUT_OF_MEMORY;
    }

    /* Norm last hidden */
    float* last_hidden = hidden + (size_t)(seq_len-1) * hs;
    vv_rmsnorm_cpu(last_hidden, norm_w, normed, 1, hs, llm->rms_norm_eps);

    /* LM head GEMM: [1, hs] @ [vocab, hs]^T → [1, vocab] */
    {
        const uint16_t* lm = (const uint16_t*)ctx->model->lm_head.data;
        for (int j = 0; j < vocab_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < hs; k++) sum += normed[k] * half_to_float_single(lm[j*hs+k]);
            logits[j] = sum;
        }
    }
    vv_free(hidden); hidden = NULL;

    int32_t token_id;
    vv_sample_greedy_cpu(logits, vocab_size, &token_id);

    perf->prefill_ms = vv_time_ms() - t_step;
    perf->ttft_ms = vv_time_ms() - t_total_start;
    perf->prefill_tok_per_sec = (perf->prefill_ms > 0.001)
        ? (double)seq_len / perf->prefill_ms * 1000.0 : 0.0;

    /* ═══ STEP 4: Decode loop (CPU) ═══ */
    t_step = vv_time_ms();
    int out_cap = max_new_tokens < 8192 ? max_new_tokens : 8192;
    int32_t* output_tokens = (int32_t*)vv_alloc((size_t)out_cap * sizeof(int32_t));
    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; goto cpu_cleanup; }
    int n_generated = 0;
    if (!vv_is_end_token(ctx->tokenizer, token_id))
        output_tokens[n_generated++] = token_id;

    while (n_generated < max_new_tokens && !vv_is_end_token(ctx->tokenizer, token_id)) {
        /* Embed */
        vv_embedding_cpu(ctx->model->embed_tokens.data, &token_id,
                          hidden_one, 1, hs);
        /* Decoder step */
        s = vv_decoder_step_cpu(ctx->model, hidden_one, ctx->kv_cache,
                                 (float*)ctx->workspace, ctx->workspace_size);
        if (s != VV_OK) break;

        /* Norm + LM head */
        vv_rmsnorm_cpu(hidden_one, norm_w, normed, 1, hs, llm->rms_norm_eps);
        {
            const uint16_t* lm = (const uint16_t*)ctx->model->lm_head.data;
            for (int j = 0; j < vocab_size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hs; k++) sum += normed[k] * half_to_float_single(lm[j*hs+k]);
                logits[j] = sum;
            }
        }
        vv_sample_greedy_cpu(logits, vocab_size, &token_id);
        if (vv_is_end_token(ctx->tokenizer, token_id)) break;

        if (n_generated >= out_cap) {
            out_cap *= 2;
            output_tokens = (int32_t*)vv_realloc(output_tokens, (size_t)out_cap * sizeof(int32_t));
            if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
        }
        output_tokens[n_generated++] = token_id;

        if ((n_generated % 100) == 0) {
            double elapsed = vv_time_ms() - t_step;
            VV_LOG_I("inference: CPU decoded %d tokens (%.1f tok/s)...",
                     n_generated, (elapsed > 0.001) ? (double)n_generated / elapsed * 1000.0 : 0.0);
        }
    }

    perf->decode_ms = vv_time_ms() - t_step;
    perf->decode_tokens = n_generated;
    perf->decode_tok_per_sec = (perf->decode_ms > 0.001)
        ? (double)n_generated / perf->decode_ms * 1000.0 : 0.0;

    /* ═══ STEP 5: Post-processing ═══ */
    t_step = vv_time_ms();
    {
        const char** token_texts = (const char**)vv_alloc((size_t)n_generated * sizeof(char*));
        if (token_texts) {
            for (int i = 0; i < n_generated; i++) {
                char* decoded = NULL;
                vv_tokenizer_decode(ctx->tokenizer, &output_tokens[i], 1, &decoded);
                token_texts[i] = decoded ? decoded : "";
            }
            s = vv_postprocess_tokens(token_texts, n_generated, result);
            if (s == VV_OK && *result) (*result)->duration = perf->audio_duration_sec;
            for (int i = 0; i < n_generated; i++)
                if (token_texts[i] && token_texts[i][0] != '\0') vv_free((void*)token_texts[i]);
            vv_free((void*)token_texts);
        }
    }
    vv_free(output_tokens);
    perf->postprocess_ms = vv_time_ms() - t_step;

    perf->total_ms = vv_time_ms() - t_total_start;
    perf->rtf = (perf->audio_duration_sec > 0.001)
        ? (perf->total_ms / 1000.0) / perf->audio_duration_sec : 0.0;
    if (ctx->kv_cache) {
        perf->kv_cache_used = ctx->kv_cache->current_len;
        perf->kv_cache_max  = ctx->kv_cache->max_seq_len;
        perf->kv_cache_pct  = (perf->kv_cache_max > 0)
            ? 100.0f * (float)perf->kv_cache_used / (float)perf->kv_cache_max : 0.0f;
    }

cpu_cleanup:
    vv_free(normed);
    vv_free(logits);
    vv_free(hidden_one);
    vv_free(norm_w);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — dispatcher
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    if (!ctx || !audio_samples || !result) return VV_ERR_NULL_PTR;

    memset(&ctx->last_perf, 0, sizeof(ctx->last_perf));

    if (ctx->placement == VV_PLACE_CPU_ONLY) {
        return transcribe_cpu(ctx, audio_samples, num_samples, params, result);
    } else {
        return transcribe_gpu(ctx, audio_samples, num_samples, params, result);
    }
}

/* ─── Perf getter ───────────────────────────────────────────────────────── */

const vv_perf_metrics_t* vv_inference_get_perf(
    const vv_inference_ctx_t* ctx) {
    return ctx ? &ctx->last_perf : NULL;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

vv_status_t vv_inference_free(vv_inference_ctx_t* ctx) {
    if (!ctx) return VV_ERR_NULL_PTR;

    if (ctx->use_gpu) {
        if (ctx->workspace) vv_dev_free(ctx->workspace);
        if (!ctx->is_clone) {
            if (ctx->embed_table_gpu) vv_dev_free(ctx->embed_table_gpu);
            if (ctx->lm_head_gpu) vv_dev_free(ctx->lm_head_gpu);
            if (ctx->final_norm_gpu) vv_dev_free(ctx->final_norm_gpu);
            if (ctx->layer_pool) vv_layer_pool_free(ctx->layer_pool);
        }
    } else {
        if (ctx->workspace) vv_free(ctx->workspace);
    }

    if (ctx->kv_cache) vv_kv_cache_free(ctx->kv_cache);
    if (ctx->use_gpu) {
        if (ctx->compute_stream) vv_dev_stream_destroy(ctx->compute_stream);
        if (ctx->transfer_stream) vv_dev_stream_destroy(ctx->transfer_stream);
    }

    if (ctx->tokenizer) vv_tokenizer_free(ctx->tokenizer);
    if (ctx->acoustic_encoder) vv_conv_vae_free(ctx->acoustic_encoder);
    if (ctx->semantic_encoder) vv_conv_vae_free(ctx->semantic_encoder);
    if (ctx->acoustic_connector) vv_free(ctx->acoustic_connector);
    if (ctx->semantic_connector) vv_free(ctx->semantic_connector);

    if (ctx->model && !ctx->is_clone) {
        if (ctx->use_gpu && ctx->placement == VV_PLACE_ALL_GPU)
            free_layer_gpu_weights(ctx->model);
        vv_model_free(ctx->model);
    }

    if (ctx->use_gpu && !ctx->is_clone) vv_gemm_cleanup();
    vv_free(ctx);

    VV_LOG_I("inference: context freed");
    return VV_OK;
}
