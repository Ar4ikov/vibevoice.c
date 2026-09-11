/**
 * @file decoder.c
 * @brief Qwen2 transformer decoder: per-layer forward with optional
 *        layer-weight streaming and CPU-only fallback.
 *
 * Per layer (28 total):
 * 1. RMSNorm -> Q, K, V (NF4 dequant + GEMM)
 * 2. RoPE -> GQA Attention -> O proj -> Residual
 * 3. RMSNorm -> SwiGLU MLP (NF4 dequant + GEMM) -> Residual
 *
 * Layer pool integration:
 *   If a vv_layer_pool_t is provided and !all_resident, the prefill/step
 *   functions stage each layer's weights to GPU just before processing,
 *   then unstage (restore CPU pointers) afterward.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/cpu_kernels.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"


/* ─── CUDA kernel forward declarations ──────────────────────────────────── */


/* ═══════════════════════════════════════════════════════════════════════════
 * Layer Pool — stage / unstage helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Calculate total GPU buffer size needed for one layer's tensors.
 */
static size_t layer_gpu_size(const vv_layer_weights_t* L) {
    size_t total = 0;
    total += L->input_layernorm.size_bytes;
    total += L->post_attn_layernorm.size_bytes;

    #define ADD_WEIGHT_SIZE(w) do {                   \
        total += (w).tensor.size_bytes;               \
        if ((w).quant.scales.data)                    \
            total += (w).quant.scales.size_bytes;     \
        if ((w).mins.data)                            \
            total += (w).mins.size_bytes;             \
        if ((w).bias.data)                            \
            total += (w).bias.size_bytes;             \
    } while(0)

    ADD_WEIGHT_SIZE(L->attn.q_proj);
    ADD_WEIGHT_SIZE(L->attn.k_proj);
    ADD_WEIGHT_SIZE(L->attn.v_proj);
    ADD_WEIGHT_SIZE(L->attn.o_proj);
    ADD_WEIGHT_SIZE(L->mlp.gate_proj);
    ADD_WEIGHT_SIZE(L->mlp.up_proj);
    ADD_WEIGHT_SIZE(L->mlp.down_proj);

    #undef ADD_WEIGHT_SIZE
    return total;
}

vv_status_t vv_layer_pool_create(vv_layer_pool_t** pool,
                                  const vv_model_t* model,
                                  bool all_resident) {
    if (!pool || !model) return VV_ERR_NULL_PTR;

    vv_layer_pool_t* p = (vv_layer_pool_t*)vv_alloc(sizeof(vv_layer_pool_t));
    if (!p) return VV_ERR_OUT_OF_MEMORY;
    memset(p, 0, sizeof(*p));
    p->all_resident = all_resident;
    p->loaded[0] = p->loaded[1] = -1;

    /*
     * Which layers are permanently resident is fixed here, before anything is
     * staged. It cannot be re-derived later from tensor.on_gpu: staging sets
     * that flag too, and confusing "already staged" with "resident" skips the
     * wait for the copy that just started.
     */
    for (p->n_resident = 0; p->n_resident < model->num_layers; p->n_resident++)
        if (!model->layers[p->n_resident].attn.q_proj.tensor.on_gpu) break;

    if (!all_resident && model->num_layers > 0) {
        /* Find max layer size */
        size_t max_sz = 0;
        for (int i = 0; i < model->num_layers; i++) {
            size_t sz = layer_gpu_size(&model->layers[i]);
            if (sz > max_sz) max_sz = sz;
        }
        /*
         * Every staged tensor lands on a 256-byte boundary, so the slack has
         * to cover one alignment gap per tensor rather than a guessed
         * handful -- at 20 live tensors the old 16 was already short.
         */
        p->buf_size = max_sz + (size_t)VV_LAYER_TENSORS_PER_LAYER * 256;

        for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
            vv_status_t st = vv_dev_alloc(&p->gpu_buf[s], p->buf_size);
            if (st != VV_OK) {
                /* Fall back to single buffer */
                if (s == 1) {
                    VV_LOG_W("layer_pool: only 1 staging buffer (no double-buffering)");
                    break;
                }
                vv_free(p);
                return st;
            }
        }
        p->n_bufs = p->gpu_buf[1] ? 2 : 1;
        for (int s = 0; s < p->n_bufs; s++) {
            vv_dev_event_create(&p->ready_ev[s]);
            vv_dev_event_create(&p->done_ev[s]);
        }
        VV_LOG_I("layer_pool: %d staging buffer(s) of %.1f MB each",
                 p->n_bufs, (double)p->buf_size / (1024.0 * 1024.0));
    }

    *pool = p;
    return VV_OK;
}

/**
 * @brief Stage layer weights to GPU: copy CPU→GPU, patch pointers.
 */
vv_status_t vv_layer_pool_stage(vv_layer_pool_t* pool,
                                 vv_model_t* model, int layer_idx,
                                 void* stream) {
    if (!pool || pool->all_resident) return VV_OK;
    if (layer_idx < 0 || layer_idx >= model->num_layers)
        return VV_ERR_INVALID_ARG;

    /*
     * With partial offload the first N layers already live on the GPU. They
     * must not pass through a staging slot: that would evict a layer that is
     * actually streamed, and unstaging one would hand a device pointer back
     * as if it were host memory.
     */
    if (layer_idx < pool->n_resident) return VV_OK;

    int slot = layer_idx % VV_LAYER_POOL_SLOTS;
    /* If only 1 buffer, always use slot 0 */
    if (!pool->gpu_buf[1]) slot = 0;

    /* Unstage whatever was previously in this slot */
    if (pool->loaded[slot] >= 0 && pool->loaded[slot] != layer_idx) {
        vv_layer_pool_unstage(pool, model, pool->loaded[slot]);
    }
    if (pool->loaded[slot] == layer_idx) return VV_OK; /* already staged */

    vv_layer_weights_t* L = &model->layers[layer_idx];
    uint8_t* buf = (uint8_t*)pool->gpu_buf[slot];
    size_t off = 0;
    int idx = 0;

    /* A NULL saved pointer tells unstage there is nothing to restore. */
    #define STAGE_TENSOR(t) do {                                        \
        pool->saved_ptrs[slot][idx] = (t).on_gpu ? NULL : (t).data;    \
        if ((t).data && !(t).on_gpu && (t).size_bytes > 0) {           \
            /* Align to 256 bytes */                                    \
            off = (off + 255) & ~(size_t)255;                          \
            if (off + (t).size_bytes > pool->buf_size) {               \
                VV_LOG_E("layer_pool: staging overflow on layer %d",    \
                         layer_idx);                                    \
                return VV_ERR_OVERFLOW;                                 \
            }                                                          \
            vv_dev_memcpy_h2d(buf + off, (t).data,                    \
                               (t).size_bytes, stream);                \
            (t).data = buf + off;                                      \
            (t).on_gpu = true;                                         \
            off += (t).size_bytes;                                     \
        }                                                              \
        idx++;                                                         \
    } while(0)

    STAGE_TENSOR(L->input_layernorm);
    STAGE_TENSOR(L->post_attn_layernorm);
    STAGE_TENSOR(L->attn.q_proj.tensor);
    STAGE_TENSOR(L->attn.q_proj.quant.scales);
    STAGE_TENSOR(L->attn.q_proj.mins);
    STAGE_TENSOR(L->attn.q_proj.bias);
    STAGE_TENSOR(L->attn.k_proj.tensor);
    STAGE_TENSOR(L->attn.k_proj.quant.scales);
    STAGE_TENSOR(L->attn.k_proj.mins);
    STAGE_TENSOR(L->attn.k_proj.bias);
    STAGE_TENSOR(L->attn.v_proj.tensor);
    STAGE_TENSOR(L->attn.v_proj.quant.scales);
    STAGE_TENSOR(L->attn.v_proj.mins);
    STAGE_TENSOR(L->attn.v_proj.bias);
    STAGE_TENSOR(L->attn.o_proj.tensor);
    STAGE_TENSOR(L->attn.o_proj.quant.scales);
    STAGE_TENSOR(L->attn.o_proj.mins);
    STAGE_TENSOR(L->attn.o_proj.bias);
    STAGE_TENSOR(L->mlp.gate_proj.tensor);
    STAGE_TENSOR(L->mlp.gate_proj.quant.scales);
    STAGE_TENSOR(L->mlp.gate_proj.mins);
    STAGE_TENSOR(L->mlp.up_proj.tensor);
    STAGE_TENSOR(L->mlp.up_proj.quant.scales);
    STAGE_TENSOR(L->mlp.up_proj.mins);
    STAGE_TENSOR(L->mlp.down_proj.tensor);
    STAGE_TENSOR(L->mlp.down_proj.quant.scales);
    STAGE_TENSOR(L->mlp.down_proj.mins);

    #undef STAGE_TENSOR

    pool->loaded[slot] = layer_idx;
    return VV_OK;
}

/**
 * @brief Unstage layer: restore original CPU pointers.
 */
vv_status_t vv_layer_pool_unstage(vv_layer_pool_t* pool,
                                   vv_model_t* model, int layer_idx) {
    if (!pool || pool->all_resident) return VV_OK;
    if (layer_idx < 0 || layer_idx >= model->num_layers)
        return VV_ERR_INVALID_ARG;

    int slot = -1;
    for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
        if (pool->loaded[s] == layer_idx) { slot = s; break; }
    }
    if (slot < 0) return VV_OK; /* not staged */

    vv_layer_weights_t* L = &model->layers[layer_idx];
    int idx = 0;

    #define UNSTAGE_TENSOR(t) do {                                      \
        if (pool->saved_ptrs[slot][idx]) {                             \
            (t).data = pool->saved_ptrs[slot][idx];                    \
            (t).on_gpu = false;                                        \
        }                                                              \
        idx++;                                                         \
    } while(0)

    UNSTAGE_TENSOR(L->input_layernorm);
    UNSTAGE_TENSOR(L->post_attn_layernorm);
    UNSTAGE_TENSOR(L->attn.q_proj.tensor);
    UNSTAGE_TENSOR(L->attn.q_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.q_proj.mins);
    UNSTAGE_TENSOR(L->attn.q_proj.bias);
    UNSTAGE_TENSOR(L->attn.k_proj.tensor);
    UNSTAGE_TENSOR(L->attn.k_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.k_proj.mins);
    UNSTAGE_TENSOR(L->attn.k_proj.bias);
    UNSTAGE_TENSOR(L->attn.v_proj.tensor);
    UNSTAGE_TENSOR(L->attn.v_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.v_proj.mins);
    UNSTAGE_TENSOR(L->attn.v_proj.bias);
    UNSTAGE_TENSOR(L->attn.o_proj.tensor);
    UNSTAGE_TENSOR(L->attn.o_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.o_proj.mins);
    UNSTAGE_TENSOR(L->attn.o_proj.bias);
    UNSTAGE_TENSOR(L->mlp.gate_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.gate_proj.quant.scales);
    UNSTAGE_TENSOR(L->mlp.gate_proj.mins);
    UNSTAGE_TENSOR(L->mlp.up_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.up_proj.quant.scales);
    UNSTAGE_TENSOR(L->mlp.up_proj.mins);
    UNSTAGE_TENSOR(L->mlp.down_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.down_proj.quant.scales);
    UNSTAGE_TENSOR(L->mlp.down_proj.mins);

    #undef UNSTAGE_TENSOR

    pool->loaded[slot] = -1;
    return VV_OK;
}

vv_status_t vv_layer_pool_free(vv_layer_pool_t* pool) {
    if (!pool) return VV_OK;
    for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
        if (pool->gpu_buf[s]) vv_dev_free(pool->gpu_buf[s]);
        if (pool->ready_ev[s]) vv_dev_event_destroy(pool->ready_ev[s]);
        if (pool->done_ev[s]) vv_dev_event_destroy(pool->done_ev[s]);
    }
    vv_free(pool);
    return VV_OK;
}

/* ─── Overlapped weight streaming ───────────────────────────────────────── */

static int pool_slot(const vv_layer_pool_t* pool, int layer_idx) {
    return (pool->n_bufs > 1) ? (layer_idx % VV_LAYER_POOL_SLOTS) : 0;
}

/** @brief VV_NO_PREFETCH=1 stages each layer synchronously (for bisecting). */
static bool prefetch_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("VV_NO_PREFETCH");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

vv_status_t vv_layer_prefetch_begin(vv_layer_pool_t* pool, vv_model_t* model,
                                    int layer_idx, void* xfer_stream) {
    if (!pool || pool->all_resident || !model) return VV_OK;
    if (layer_idx < 0 || layer_idx >= model->num_layers) return VV_OK;
    if (prefetch_disabled()) return VV_OK;
    if (layer_idx < pool->n_resident) return VV_OK;

    const int slot = pool_slot(pool, layer_idx);
    if (pool->loaded[slot] == layer_idx) return VV_OK;

    /*
     * Overwriting a slot is only safe once the compute that read it is done.
     * With a single staging buffer there is nothing to overlap with, so the
     * caller's prefetch-ahead is simply skipped and vv_layer_prefetch_wait
     * stages the layer synchronously instead.
     */
    if (pool->loaded[slot] >= 0) {
        if (!pool->done_valid[slot]) return VV_OK;
        vv_dev_stream_wait_event(xfer_stream, pool->done_ev[slot]);
    }

    vv_status_t s = vv_layer_pool_stage(pool, model, layer_idx, xfer_stream);
    if (s != VV_OK) return s;

    pool->done_valid[slot] = false;
    return vv_dev_event_record(pool->ready_ev[slot], xfer_stream);
}

vv_status_t vv_layer_prefetch_wait(vv_layer_pool_t* pool, vv_model_t* model,
                                   int layer_idx, void* compute_stream) {
    if (!pool || pool->all_resident || !model) return VV_OK;
    if (layer_idx < pool->n_resident) return VV_OK;

    const int slot = pool_slot(pool, layer_idx);
    if (pool->loaded[slot] != layer_idx) {
        /* Not prefetched (single buffer, or the slot was still busy). */
        vv_status_t s = vv_layer_pool_stage(pool, model, layer_idx,
                                            compute_stream);
        if (s != VV_OK) return s;
        pool->done_valid[slot] = false;
        return VV_OK;
    }
    return vv_dev_stream_wait_event(compute_stream, pool->ready_ev[slot]);
}

vv_status_t vv_layer_prefetch_done(vv_layer_pool_t* pool, int layer_idx,
                                   void* compute_stream) {
    if (!pool || pool->all_resident) return VV_OK;
    if (layer_idx < pool->n_resident) return VV_OK;
    const int slot = pool_slot(pool, layer_idx);
    if (pool->loaded[slot] != layer_idx) return VV_OK;
    pool->done_valid[slot] = true;
    return vv_dev_event_record(pool->done_ev[slot], compute_stream);
}

vv_status_t vv_layer_pool_pin_host(vv_model_t* model, int first_streamed) {
    if (!model) return VV_ERR_NULL_PTR;
    if (first_streamed >= model->num_layers) return VV_OK;

    size_t pinned = 0;
    const double t0 = vv_time_ms();

    #define PIN(t) do {                                                             if ((t).data && !(t).on_gpu && (t).size_bytes >= 65536) {                       if (vv_dev_host_register((t).data, (t).size_bytes) == VV_OK)                    pinned += (t).size_bytes;                                           }                                                                       } while (0)
    #define PIN_W(w) do { PIN((w).tensor); PIN((w).quant.scales);                                     PIN((w).mins); PIN((w).bias); } while (0)

    for (int i = first_streamed; i < model->num_layers; i++) {
        vv_layer_weights_t* L = &model->layers[i];
        PIN_W(L->attn.q_proj); PIN_W(L->attn.k_proj);
        PIN_W(L->attn.v_proj); PIN_W(L->attn.o_proj);
        PIN_W(L->mlp.gate_proj); PIN_W(L->mlp.up_proj);
        PIN_W(L->mlp.down_proj);
    }
    #undef PIN_W
    #undef PIN

    if (pinned)
        VV_LOG_I("layer_pool: page-locked %.1f MB of streamed weights "
                 "(%.0f ms)", (double)pinned / (1024.0 * 1024.0),
                 vv_time_ms() - t0);
    return VV_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU decoder — per-layer forward (unchanged core logic)
 * ═══════════════════════════════════════════════════════════════════════════ */


/**
 * @brief One quantised linear layer: y[M,N] = x[M,K] @ dequant(W)^T + bias.
 *
 * Single-token decode takes the fused path, which reads the 4-bit weights
 * straight into the MAC instead of materialising an FP16 copy first — the
 * scratch round-trip costs ~5x the memory traffic and dominates decode.
 */
static vv_status_t quant_linear(
    const vv_weight_t* w, const void* x, void* y,
    void* scratch, int M, int N, int K, void* stream)
{
    vv_status_t s;

    if (w->quant_kind == VV_QUANT_INT4G) {
        if (M == 1) {
            s = vv_awq_gemv_dev(x, (const uint32_t*)w->tensor.data,
                                (const uint32_t*)w->mins.data,
                                w->quant.scales.data, w->bias.data,
                                y, N, K, w->group_size, stream);
            if (s == VV_OK) return VV_OK;      /* bias already folded in */
            if (s != VV_ERR_UNSUPPORTED) return s;
        }
        s = vv_awq_gemm_dev(x, (const uint32_t*)w->tensor.data,
                            (const uint32_t*)w->mins.data,
                            w->quant.scales.data, y, scratch,
                            M, N, K, w->group_size, stream);
    } else if (w->quant_kind == VV_QUANT_NF4) {
        if (M == 1) {
            s = vv_nf4_gemv_dev(x, (const uint8_t*)w->tensor.data,
                                 w->quant.scales.data, w->bias.data,
                                 y, N, K, stream);
            if (s == VV_OK) return VV_OK;      /* bias already folded in */
            if (s != VV_ERR_UNSUPPORTED) return s;
        }
        s = vv_nf4_gemm_dev(x, (const uint8_t*)w->tensor.data,
                             w->quant.scales.data, y, scratch,
                             M, N, K, 64, stream);
    } else {
        s = vv_gemm_fp16_dev(x, w->tensor.data, y, M, N, K,
                              1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    if (w->bias.data)
        s = vv_bias_add_dev(y, w->bias.data, M, N, stream);
    return s;
}

/**
 * @brief Dump a GPU FP16 buffer as FP32 to $VV_DUMP_DIR (debug builds only).
 */
static void dump_gpu_fp16(const char* name, const void* gpu, size_t n,
                          void* stream) {
    if (!vv_debug_dump_dir() || !gpu || n == 0) return;
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    if (!h) return;
    vv_dev_stream_sync(stream);
    vv_dev_memcpy_d2h(h, gpu, n * 2, NULL);
    float* f = (float*)vv_alloc(n * sizeof(float));
    if (f) {
        for (size_t i = 0; i < n; i++) f[i] = vv_half_to_float(h[i]);
        vv_debug_dump(name, f, n * sizeof(float));
        vv_free(f);
    }
    vv_free(h);
}

static vv_status_t decoder_layer_impl(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    void* temp_workspace,
    size_t workspace_size,
    void* stream)
{
    if (!layer || !config || !hidden_states || !kv_cache ||
        !temp_workspace) {
        return VV_ERR_NULL_PTR;
    }

    int hs = config->hidden_size;
    int n_heads = config->num_attention_heads;
    int n_kv_heads = config->num_key_value_heads;
    int head_dim = config->head_dim;
    int inter_size = config->intermediate_size;
    vv_status_t s;

    /*
     * Workspace layout:
     * [0]: norm_out    [seq_len * hs] FP16
     * [1]: q           [seq_len * n_heads * head_dim] FP16
     * [2]: k           [seq_len * n_kv_heads * head_dim] FP16
     * [3]: v           [seq_len * n_kv_heads * head_dim] FP16
     * [4]: attn_out    [seq_len * hs] FP16
     * [5]: gate        [seq_len * inter_size] FP16
     * [6]: up          [seq_len * inter_size] FP16
     * [7]: mlp_out     [seq_len * hs] FP16
     * [8]: temp_weight [max(N*K)] FP16 for NF4 dequant
     */
    uint8_t* wp = (uint8_t*)temp_workspace;
    size_t offset = 0;

    void* norm_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* q_buf = wp + offset;
    offset += (size_t)seq_len * n_heads * head_dim * 2;

    void* k_buf = wp + offset;
    offset += (size_t)seq_len * n_kv_heads * head_dim * 2;

    void* v_buf = wp + offset;
    offset += (size_t)seq_len * n_kv_heads * head_dim * 2;

    void* attn_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* gate_buf = wp + offset;
    offset += (size_t)seq_len * inter_size * 2;

    void* up_buf = wp + offset;
    offset += (size_t)seq_len * inter_size * 2;

    void* mlp_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* temp_weight = wp + offset;

    /* 1. Input LayerNorm */
    s = vv_rmsnorm_dev(hidden_states, layer->input_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 2. Q, K, V projections (+ bias if present) */
    s = quant_linear(&layer->attn.q_proj, norm_out, q_buf, temp_weight,
                     seq_len, n_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.k_proj, norm_out, k_buf, temp_weight,
                     seq_len, n_kv_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.v_proj, norm_out, v_buf, temp_weight,
                     seq_len, n_kv_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1) {
        dump_gpu_fp16("c_l0_norm", norm_out, (size_t)seq_len * hs, stream);
        dump_gpu_fp16("c_l0_q_prerope", q_buf,
                      (size_t)seq_len * n_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_k_prerope", k_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_v", v_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
    }

    /* 3. RoPE */
    s = vv_rope_dev(q_buf, seq_len, n_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;
    s = vv_rope_dev(k_buf, seq_len, n_kv_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1) {
        dump_gpu_fp16("c_l0_q", q_buf,
                      (size_t)seq_len * n_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_k", k_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
    }

    /* 4. KV-cache append */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, stream);
    if (s != VV_OK) return s;

    /* 5. GQA attention over the cache (queries of this chunk see all of it) */
    {
        const void* k_cached;
        const void* v_cached;
        int cache_len;
        vv_kv_cache_get(kv_cache, layer_idx, &k_cached, &v_cached, &cache_len);
        /*
         * current_len only advances on the last layer, so derive the true
         * length from this call's position instead.
         */
        const int actual_cache_len = position_offset + seq_len;
        const vv_kv_format_t fmt = (vv_kv_format_t)kv_cache->format;

        if (fmt == VV_KV_FP16) {
            if (seq_len > 1) {
                s = vv_gqa_attention_prefill_cached_dev(
                    q_buf, k_cached, v_cached, attn_out,
                    n_heads, n_kv_heads, head_dim,
                    seq_len, position_offset, actual_cache_len, true, stream);
            } else {
                s = vv_gqa_attention_decode_dev(
                    q_buf, k_cached, v_cached, attn_out,
                    n_heads, n_kv_heads, head_dim, actual_cache_len, stream);
            }
        } else {
            const void *k_meta, *v_meta;
            vv_kv_cache_get_meta(kv_cache, layer_idx, &k_meta, &v_meta);
            /*
             * TurboQuant stores rotated vectors. The transform is orthogonal,
             * so instead of inverting it per key we rotate Q once here and
             * undo the rotation on the output; the kernels then read stored
             * values directly.
             */
            if (vv_kv_rotates(fmt)) {
                s = vv_kv_rotate_dev(q_buf, n_heads, head_dim, seq_len, stream);
                if (s != VV_OK) return s;
            }
            if (seq_len > 1) {
                s = vv_gqa_attention_prefill_q_dev(
                    q_buf, k_cached, v_cached, k_meta, v_meta, attn_out,
                    n_heads, n_kv_heads, head_dim,
                    seq_len, position_offset, actual_cache_len, true,
                    (int)fmt, stream);
            } else {
                s = vv_gqa_attention_decode_q_dev(
                    q_buf, k_cached, v_cached, k_meta, v_meta, attn_out,
                    n_heads, n_kv_heads, head_dim, actual_cache_len,
                    (int)fmt, stream);
            }
            if (s == VV_OK && vv_kv_rotates(fmt))
                s = vv_kv_unrotate_dev(attn_out, n_heads, head_dim,
                                       seq_len, stream);
        }
    }
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1)
        dump_gpu_fp16("c_l0_attn", attn_out, (size_t)seq_len * hs, stream);

    /* 6. O projection + bias + residual */
    s = quant_linear(&layer->attn.o_proj, attn_out, norm_out, temp_weight,
                     seq_len, hs, hs, stream);
    if (s != VV_OK) return s;
    s = vv_residual_add_dev(hidden_states, norm_out,
                              seq_len * hs, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1)
        dump_gpu_fp16("c_l0_postattn", hidden_states, (size_t)seq_len * hs, stream);

    /* 7. Post-attention LayerNorm */
    s = vv_rmsnorm_dev(hidden_states, layer->post_attn_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 8. MLP: gate + up (+ bias if present) */
    s = quant_linear(&layer->mlp.gate_proj, norm_out, gate_buf, temp_weight,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->mlp.up_proj, norm_out, up_buf, temp_weight,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    /* 9. SwiGLU */
    s = vv_swiglu_dev(gate_buf, up_buf, gate_buf,
                        seq_len * inter_size, stream);
    if (s != VV_OK) return s;

    /* 10. Down projection + bias + residual */
    s = quant_linear(&layer->mlp.down_proj, gate_buf, mlp_out, temp_weight,
                     seq_len, hs, inter_size, stream);
    if (s != VV_OK) return s;
    s = vv_residual_add_dev(hidden_states, mlp_out,
                              seq_len * hs, stream);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU decoder — public API (with layer pool support)
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_decoder_layer_forward(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    void* temp_workspace,
    size_t workspace_size,
    void* stream)
{
    return decoder_layer_impl(layer, config, hidden_states, kv_cache,
                               layer_idx, position_offset, 1,
                               temp_workspace, workspace_size, stream);
}

vv_status_t vv_decoder_step(
    vv_model_t* model,
    void* hidden_state,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;

    int position = kv_cache->current_len;
    bool streaming = pool && !pool->all_resident;

    if (streaming) vv_layer_prefetch_begin(pool, model, 0, xfer_stream);

    for (int i = 0; i < model->num_layers; i++) {
        if (streaming) {
            vv_layer_prefetch_wait(pool, model, i, compute_stream);
            if (i + 1 < model->num_layers)
                vv_layer_prefetch_begin(pool, model, i + 1, xfer_stream);
        }

        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            hidden_state, kv_cache, i, position, 1,
            workspace, workspace_size, compute_stream);

        if (streaming) vv_layer_prefetch_done(pool, i, compute_stream);

        if (s != VV_OK) {
            VV_LOG_E("decoder: layer %d failed: %s", i, vv_status_str(s));
            return s;
        }
    }

    return VV_OK;
}

vv_status_t vv_decoder_prefill(
    vv_model_t* model,
    void* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;

    VV_LOG_I("decoder: prefill %d tokens through %d layers%s",
             seq_len, model->num_layers,
             (pool && !pool->all_resident) ? " (streaming)" : "");

    bool streaming = pool && !pool->all_resident;

    const vv_llm_config_t* cfg = &model->config.llm;
    const int hs = cfg->hidden_size;

    /*
     * Chunked prefill. Activation scratch grows linearly with the number of
     * tokens in flight (the two 18944-wide MLP buffers dominate), so a
     * 30-minute prompt would need gigabytes if run in one shot. Chunking caps
     * that at a fixed budget; correctness is preserved because each chunk
     * attends to the whole KV cache written by the chunks before it.
     */
    size_t per_token = (size_t)(hs * 3
                       + cfg->num_attention_heads * cfg->head_dim
                       + 2 * cfg->num_key_value_heads * cfg->head_dim
                       + 2 * cfg->intermediate_size) * 2;
    size_t weight_scratch = (size_t)cfg->intermediate_size * hs * 2;
    int chunk = seq_len;
    if (workspace_size > weight_scratch + per_token) {
        size_t budget = (workspace_size - weight_scratch) / per_token;
        if (budget < 1) budget = 1;
        if (budget > 2048) budget = 2048;
        if ((int)budget < chunk) chunk = (int)budget;
    }
    if (chunk < 1) chunk = 1;
    if (chunk < seq_len)
        VV_LOG_I("decoder: prefill in %d chunks of %d tokens",
                 (seq_len + chunk - 1) / chunk, chunk);

    for (int start = 0; start < seq_len; start += chunk) {
      const int len = (start + chunk <= seq_len) ? chunk : (seq_len - start);
      void* chunk_hidden = (uint8_t*)hidden_states + (size_t)start * hs * 2;

      if (streaming) vv_layer_prefetch_begin(pool, model, 0, xfer_stream);

      for (int i = 0; i < model->num_layers; i++) {

        if (streaming) {
            vv_layer_prefetch_wait(pool, model, i, compute_stream);
            if (i + 1 < model->num_layers)
                vv_layer_prefetch_begin(pool, model, i + 1, xfer_stream);
        }

        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            chunk_hidden, kv_cache, i, start, len,
            workspace, workspace_size, compute_stream);

        if (streaming) vv_layer_prefetch_done(pool, i, compute_stream);

        if (s != VV_OK) {
            VV_LOG_E("decoder: prefill layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }

        if (start + len == seq_len && vv_debug_dump_dir() && start == 0) {
            size_t n = (size_t)seq_len * (size_t)model->config.llm.hidden_size;
            uint16_t* h = (uint16_t*)vv_alloc(n * 2);
            if (h) {
                vv_dev_stream_sync(compute_stream);
                vv_dev_memcpy_d2h(h, hidden_states, n * 2, NULL);
                float* f = (float*)vv_alloc(n * sizeof(float));
                if (f) {
                    for (size_t j = 0; j < n; j++) f[j] = vv_half_to_float(h[j]);
                    char nm[64];
                    snprintf(nm, sizeof(nm), "c_layer%02d", i + 1);
                    vv_debug_dump(nm, f, n * sizeof(float));
                    vv_free(f);
                }
                vv_free(h);
            }
        }
      }
    }

    return VV_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU decoder — per-layer forward (FP32, all on CPU)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** NF4 lookup table (for scale conversion) */
extern const float VV_NF4_TABLE[16];

/**
 * @brief Convert FP16 scale tensor to FP32 float array.
 * Caller must free returned pointer.
 */
static float* fp16_scales_to_fp32(const vv_tensor_t* scales) {
    if (!scales || !scales->data || scales->size_bytes == 0) return NULL;
    int n = (int)(scales->size_bytes / 2); /* FP16 = 2 bytes */
    float* out = (float*)vv_alloc((size_t)n * sizeof(float));
    if (!out) return NULL;
    const uint16_t* src = (const uint16_t*)scales->data;
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = ((uint32_t)h & 0x8000) << 16;
        uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
        uint32_t frac = (uint32_t)h & 0x03FF;
        union { float f; uint32_t u; } u;
        if (expo == 0)
            u.u = sign;
        else if (expo == 0x1F)
            u.u = sign | 0x7F800000 | (frac << 13);
        else
            u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
        out[i] = u.f;
    }
    return out;
}

/**
 * @brief Convert FP16 weight tensor to FP32.
 * Caller must free returned pointer.
 */
static float* fp16_to_fp32(const void* data, int n_elements) {
    if (!data || n_elements <= 0) return NULL;
    float* out = (float*)vv_alloc((size_t)n_elements * sizeof(float));
    if (!out) return NULL;
    const uint16_t* src = (const uint16_t*)data;
    for (int i = 0; i < n_elements; i++) {
        uint16_t h = src[i];
        uint32_t sign = ((uint32_t)h & 0x8000) << 16;
        uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
        uint32_t frac = (uint32_t)h & 0x03FF;
        union { float f; uint32_t u; } u;
        if (expo == 0)
            u.u = sign;
        else if (expo == 0x1F)
            u.u = sign | 0x7F800000 | (frac << 13);
        else
            u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
        out[i] = u.f;
    }
    return out;
}

static vv_status_t decoder_layer_cpu(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    float* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    float* workspace,
    size_t workspace_size)
{
    int hs = config->hidden_size;
    int n_heads = config->num_attention_heads;
    int n_kv_heads = config->num_key_value_heads;
    int head_dim = config->head_dim;
    int inter_size = config->intermediate_size;
    vv_status_t s;

    /* Workspace layout (FP32): each element 4 bytes */
    float* wp = workspace;
    size_t off = 0;

    float* norm_out = wp + off; off += seq_len * hs;
    float* q_buf    = wp + off; off += seq_len * n_heads * head_dim;
    float* k_buf    = wp + off; off += seq_len * n_kv_heads * head_dim;
    float* v_buf    = wp + off; off += seq_len * n_kv_heads * head_dim;
    float* attn_out = wp + off; off += seq_len * hs;
    float* gate_buf = wp + off; off += seq_len * inter_size;
    float* up_buf   = wp + off; off += seq_len * inter_size;
    float* mlp_out  = wp + off; off += seq_len * hs;
    float* temp_w   = wp + off;
    /* remaining workspace used for temp dequant weight */

    /* Convert FP16 layernorm weights to FP32 */
    float* ln1_w = fp16_to_fp32(layer->input_layernorm.data,
                                 hs);
    float* ln2_w = fp16_to_fp32(layer->post_attn_layernorm.data,
                                 hs);
    if (!ln1_w || !ln2_w) {
        if (ln1_w) vv_free(ln1_w);
        if (ln2_w) vv_free(ln2_w);
        return VV_ERR_OUT_OF_MEMORY;
    }

    /* 1. RMSNorm */
    s = vv_rmsnorm_cpu(hidden_states, ln1_w, norm_out,
                        seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) goto cleanup_norms;

    /* 2. Q, K, V projections (NF4 or FP16→FP32) + bias */
    #define CPU_PROJ(w, out_buf, M, N, K_dim) do {                        \
        if ((w).is_quantized) {                                           \
            float* sc = fp16_scales_to_fp32(&(w).quant.scales);           \
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }    \
            s = vv_nf4_gemm_cpu(norm_out,                                 \
                (const uint8_t*)(w).tensor.data, sc,                      \
                (out_buf), temp_w, (M), (N), (K_dim), 64);               \
            vv_free(sc);                                                  \
        } else {                                                          \
            float* wf = fp16_to_fp32((w).tensor.data, (N) * (K_dim));     \
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }    \
            /* output = input @ weight^T */                               \
            for (int _i = 0; _i < (M); _i++) {                           \
                for (int _j = 0; _j < (N); _j++) {                       \
                    float sum = 0.0f;                                     \
                    for (int _k = 0; _k < (K_dim); _k++)                  \
                        sum += norm_out[_i*(K_dim)+_k] * wf[_j*(K_dim)+_k]; \
                    (out_buf)[_i*(N)+_j] = sum;                           \
                }                                                         \
            }                                                             \
            vv_free(wf);                                                  \
        }                                                                 \
        if (s != VV_OK) goto cleanup_norms;                               \
        /* Add bias if present */                                         \
        if ((w).bias.data) {                                              \
            float* bf = fp16_to_fp32((w).bias.data, (N));                 \
            if (bf) {                                                     \
                for (int _i = 0; _i < (M); _i++)                         \
                    for (int _j = 0; _j < (N); _j++)                     \
                        (out_buf)[_i*(N)+_j] += bf[_j];                   \
                vv_free(bf);                                              \
            }                                                             \
        }                                                                 \
    } while(0)

    CPU_PROJ(layer->attn.q_proj, q_buf, seq_len, n_heads * head_dim, hs);
    CPU_PROJ(layer->attn.k_proj, k_buf, seq_len, n_kv_heads * head_dim, hs);
    CPU_PROJ(layer->attn.v_proj, v_buf, seq_len, n_kv_heads * head_dim, hs);

    /* 3. RoPE */
    s = vv_rope_cpu(q_buf, seq_len, n_heads, head_dim,
                     position_offset, config->rope_theta);
    if (s != VV_OK) goto cleanup_norms;
    s = vv_rope_cpu(k_buf, seq_len, n_kv_heads, head_dim,
                     position_offset, config->rope_theta);
    if (s != VV_OK) goto cleanup_norms;

    /* 4. KV cache append (CPU cache uses memcpy) */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, NULL);
    if (s != VV_OK) goto cleanup_norms;

    /* 5. GQA Attention */
    if (seq_len > 1) {
        s = vv_attention_prefill_cpu(q_buf, k_buf, v_buf, attn_out,
                                      n_heads, n_kv_heads, head_dim,
                                      seq_len, true);
    } else {
        const void* kc; const void* vc; int cl;
        vv_kv_cache_get(kv_cache, layer_idx, &kc, &vc, &cl);
        /* Include the just-appended token */
        int actual_cl = position_offset + seq_len;
        s = vv_attention_decode_cpu(q_buf, (const float*)kc,
                                     (const float*)vc, attn_out,
                                     n_heads, n_kv_heads, head_dim, actual_cl);
    }
    if (s != VV_OK) goto cleanup_norms;

    /* 6. O projection + bias + residual */
    {
        /* Reuse norm_out as temp */
        float* o_out = norm_out; /* overwrite ok, done with input norm */

        if (layer->attn.o_proj.is_quantized) {
            float* sc = fp16_scales_to_fp32(&layer->attn.o_proj.quant.scales);
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            s = vv_nf4_gemm_cpu(attn_out,
                (const uint8_t*)layer->attn.o_proj.tensor.data, sc,
                o_out, temp_w, seq_len, hs, hs, 64);
            vv_free(sc);
        } else {
            float* wf = fp16_to_fp32(layer->attn.o_proj.tensor.data, hs * hs);
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            for (int _i = 0; _i < seq_len; _i++)
                for (int _j = 0; _j < hs; _j++) {
                    float sum = 0.0f;
                    for (int _k = 0; _k < hs; _k++)
                        sum += attn_out[_i*hs+_k] * wf[_j*hs+_k];
                    o_out[_i*hs+_j] = sum;
                }
            vv_free(wf);
        }
        if (s != VV_OK) goto cleanup_norms;
        /* Add bias if present */
        if (layer->attn.o_proj.bias.data) {
            float* bf = fp16_to_fp32(layer->attn.o_proj.bias.data, hs);
            if (bf) {
                for (int _i = 0; _i < seq_len; _i++)
                    for (int _j = 0; _j < hs; _j++)
                        o_out[_i*hs+_j] += bf[_j];
                vv_free(bf);
            }
        }
        vv_residual_add_cpu(hidden_states, o_out, seq_len * hs);
    }

    /* 7. Post-attention LayerNorm */
    s = vv_rmsnorm_cpu(hidden_states, ln2_w, norm_out,
                        seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) goto cleanup_norms;

    /* 8. MLP: gate + up + SwiGLU + down + residual */
    CPU_PROJ(layer->mlp.gate_proj, gate_buf, seq_len, inter_size, hs);
    CPU_PROJ(layer->mlp.up_proj,   up_buf,   seq_len, inter_size, hs);

    #undef CPU_PROJ

    s = vv_swiglu_cpu(gate_buf, up_buf, gate_buf, seq_len * inter_size);
    if (s != VV_OK) goto cleanup_norms;

    /* Down proj + bias */
    {
        if (layer->mlp.down_proj.is_quantized) {
            float* sc = fp16_scales_to_fp32(&layer->mlp.down_proj.quant.scales);
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            s = vv_nf4_gemm_cpu(gate_buf,
                (const uint8_t*)layer->mlp.down_proj.tensor.data, sc,
                mlp_out, temp_w, seq_len, hs, inter_size, 64);
            vv_free(sc);
        } else {
            float* wf = fp16_to_fp32(layer->mlp.down_proj.tensor.data,
                                      hs * inter_size);
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            for (int _i = 0; _i < seq_len; _i++)
                for (int _j = 0; _j < hs; _j++) {
                    float sum = 0.0f;
                    for (int _k = 0; _k < inter_size; _k++)
                        sum += gate_buf[_i*inter_size+_k] * wf[_j*inter_size+_k];
                    mlp_out[_i*hs+_j] = sum;
                }
            vv_free(wf);
        }
        if (s != VV_OK) goto cleanup_norms;
        /* Add bias if present */
        if (layer->mlp.down_proj.bias.data) {
            float* bf = fp16_to_fp32(layer->mlp.down_proj.bias.data, hs);
            if (bf) {
                for (int _i = 0; _i < seq_len; _i++)
                    for (int _j = 0; _j < hs; _j++)
                        mlp_out[_i*hs+_j] += bf[_j];
                vv_free(bf);
            }
        }
        vv_residual_add_cpu(hidden_states, mlp_out, seq_len * hs);
    }

cleanup_norms:
    vv_free(ln1_w);
    vv_free(ln2_w);
    return s;
}

/* ─── CPU prefill / step ────────────────────────────────────────────────── */

vv_status_t vv_decoder_prefill_cpu(
    vv_model_t* model,
    float* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;

    VV_LOG_I("decoder: CPU prefill %d tokens through %d layers",
             seq_len, model->num_layers);

    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_cpu(
            &model->layers[i], &model->config.llm,
            hidden_states, kv_cache, i, 0, seq_len,
            workspace, workspace_size);
        if (s != VV_OK) {
            VV_LOG_E("decoder: CPU prefill layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}

vv_status_t vv_decoder_step_cpu(
    vv_model_t* model,
    float* hidden_state,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;

    int position = kv_cache->current_len;

    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_cpu(
            &model->layers[i], &model->config.llm,
            hidden_state, kv_cache, i, position, 1,
            workspace, workspace_size);
        if (s != VV_OK) {
            VV_LOG_E("decoder: CPU step layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}
