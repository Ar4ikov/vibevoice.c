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
#include "vibevoice/quant.h"

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

/** @brief GPU buffer size needed for one layer's tensors. */
static size_t layer_gpu_size(const vv_layer_weights_t* L) {
    return vv_layer_bytes(L);
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

    vv_tensor_t* ts[VV_LAYER_TENSOR_SLOTS];
    const int nt = vv_layer_tensors(&model->layers[layer_idx], ts);
    uint8_t* buf = (uint8_t*)pool->gpu_buf[slot];
    size_t off = 0;

    /* A NULL saved pointer tells unstage there is nothing to restore. */
    for (int idx = 0; idx < nt; idx++) {
        vv_tensor_t* t = ts[idx];
        pool->saved_ptrs[slot][idx] = t->on_gpu ? NULL : t->data;
        if (!t->data || t->on_gpu || t->size_bytes == 0) continue;
        off = (off + 255) & ~(size_t)255;       /* 256-byte aligned */
        if (off + t->size_bytes > pool->buf_size) {
            VV_LOG_E("layer_pool: staging overflow on layer %d", layer_idx);
            return VV_ERR_OVERFLOW;
        }
        vv_status_t s = vv_dev_memcpy_h2d(buf + off, t->data, t->size_bytes,
                                          stream);
        if (s != VV_OK) return s;
        t->data = buf + off;
        t->on_gpu = true;
        off += t->size_bytes;
    }

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

    vv_tensor_t* ts[VV_LAYER_TENSOR_SLOTS];
    const int nt = vv_layer_tensors(&model->layers[layer_idx], ts);
    for (int idx = 0; idx < nt; idx++) {
        if (!pool->saved_ptrs[slot][idx]) continue;
        ts[idx]->data = pool->saved_ptrs[slot][idx];
        ts[idx]->on_gpu = false;
    }

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
    return vv_layer_pool_pin_range(model, first_streamed,
                                   model->num_layers - first_streamed);
}

vv_status_t vv_layer_pool_pin_range(vv_model_t* model, int first, int count) {
    if (!model) return VV_ERR_NULL_PTR;
    if (first < 0) first = 0;
    if (first + count > model->num_layers) count = model->num_layers - first;
    if (count <= 0) return VV_OK;

    size_t pinned = 0;
    const double t0 = vv_time_ms();

    /* Small tensors (norms, biases) are not worth a registration each. */
    for (int i = first; i < first + count; i++) {
        vv_tensor_t* ts[VV_LAYER_TENSOR_SLOTS];
        const int nt = vv_layer_tensors(&model->layers[i], ts);
        for (int k = 0; k < nt; k++) {
            vv_tensor_t* t = ts[k];
            if (!t->data || t->on_gpu || t->size_bytes < 65536) continue;
            if (vv_dev_host_register(t->data, t->size_bytes) == VV_OK)
                pinned += t->size_bytes;
        }
    }

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
    void* scratch, size_t scratch_bytes, int M, int N, int K, void* stream)
{
    vv_status_t s;

    if (w->quant_kind == VV_QUANT_INT4G &&
        w->int4g_layout == VV_INT4G_GPU) {
        /* W4A16 kernels: bias fused, no dequantized copy of the weight. */
        if (M == 1)
            return vv_w4a16_gemv_dev(x, w->tensor.data,
                                     w->quant.scales.data, w->bias.data,
                                     y, N, K, w->group_size, stream);
        return vv_w4a16_gemm_dev(x, w->tensor.data, w->quant.scales.data,
                                 w->bias.data, y, scratch, scratch_bytes,
                                 M, N, K, w->group_size, stream);
    }

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
    } else if (w->quant_kind == VV_QUANT_INT8) {
        if (M == 1)
            return vv_int8_gemv_dev(x, (const int8_t*)w->tensor.data,
                                    (const float*)w->quant.scales.data,
                                    w->bias.data, y, N, K, stream);
        s = vv_int8_gemm_dev(x, (const int8_t*)w->tensor.data,
                             (const float*)w->quant.scales.data, y, scratch,
                             M, N, K, stream);
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

/**
 * @brief Bytes the split-K decode attention reserves at the workspace front.
 *
 * Its partials have to survive between the split kernel and the combine
 * kernel, and they belong to one context: a shared buffer let two concurrent
 * requests read each other's slices, which showed up as words drifting in the
 * second transcript. The workspace is per-context, so carving it from there
 * gets the lifetime and the ownership right at once.
 */
static size_t decode_scratch_bytes(const vv_llm_config_t* cfg) {
    return vv_gqa_decode_scratch_bytes(cfg->num_attention_heads,
                                       cfg->head_dim);
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
     * A decode step is captured once and replayed for every token, so nothing
     * in it may carry the position as a kernel argument. One row means decode;
     * prefill keeps the host-side scalars, which is what its chunking needs.
     */
    const bool dev_pos = (seq_len == 1) && !kv_cache->on_cpu && kv_cache->d_len;
    const int* d_pos  = dev_pos ? (const int*)kv_cache->d_len : NULL;
    const int* d_next = dev_pos ? (const int*)kv_cache->d_len_next : NULL;

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
     * with the decode-attention scratch ahead of all of it.
     */
    uint8_t* wp = (uint8_t*)temp_workspace;
    size_t offset = 0;

    void* decode_scratch = wp;
    offset += decode_scratch_bytes(config);

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
    const size_t temp_bytes = workspace_size > offset
                            ? workspace_size - offset : 0;

    /* 1. Input LayerNorm */
    s = vv_rmsnorm_dev(hidden_states, layer->input_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 2. Q, K, V projections (+ bias if present) */
    s = quant_linear(&layer->attn.q_proj, norm_out, q_buf, temp_weight, temp_bytes,
                     seq_len, n_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.k_proj, norm_out, k_buf, temp_weight, temp_bytes,
                     seq_len, n_kv_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.v_proj, norm_out, v_buf, temp_weight, temp_bytes,
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
                      position_offset, d_pos, config->rope_theta, stream);
    if (s != VV_OK) return s;
    s = vv_rope_dev(k_buf, seq_len, n_kv_heads, head_dim,
                      position_offset, d_pos, config->rope_theta, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1) {
        dump_gpu_fp16("c_l0_q", q_buf,
                      (size_t)seq_len * n_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_k", k_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
    }

    /* 4. KV-cache append */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, dev_pos, stream);
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

        if (vv_kv_is_raw(fmt)) {
            if (seq_len > 1) {
                s = vv_gqa_attention_prefill_cached_dev(
                    q_buf, k_cached, v_cached, attn_out,
                    n_heads, n_kv_heads, head_dim,
                    seq_len, position_offset, actual_cache_len, true, stream);
            } else {
                s = vv_gqa_attention_decode_dev(
                    q_buf, k_cached, v_cached, attn_out,
                    n_heads, n_kv_heads, head_dim, actual_cache_len,
                    d_next, decode_scratch, stream);
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
                    d_next, (int)fmt, decode_scratch, stream);
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
    s = quant_linear(&layer->attn.o_proj, attn_out, norm_out, temp_weight, temp_bytes,
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
    s = quant_linear(&layer->mlp.gate_proj, norm_out, gate_buf, temp_weight, temp_bytes,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->mlp.up_proj, norm_out, up_buf, temp_weight, temp_bytes,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    /* 9. SwiGLU */
    s = vv_swiglu_dev(gate_buf, up_buf, gate_buf,
                        seq_len * inter_size, stream);
    if (s != VV_OK) return s;

    /* 10. Down projection + bias + residual */
    s = quant_linear(&layer->mlp.down_proj, gate_buf, mlp_out, temp_weight, temp_bytes,
                     seq_len, hs, inter_size, stream);
    if (s != VV_OK) return s;
    s = vv_residual_add_dev(hidden_states, mlp_out,
                              seq_len * hs, stream);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU decoder — public API (with layer pool support)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One layer, for inspection. Not the decode entry point.
 *
 * vv_decoder_step is: it advances the device-side position the decode kernels
 * read, and a single layer called on its own would rotate and attend at
 * whatever position happens to be there.
 */
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
    void* xfer_stream,
    int first_layer,
    int n_layers)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;
    const int last_layer = first_layer + n_layers;
    if (first_layer < 0 || n_layers <= 0 || last_layer > model->num_layers)
        return VV_ERR_INVALID_ARG;

    int position = kv_cache->current_len;
    bool streaming = pool && !pool->all_resident;

    /*
     * The layers read `d_len` (the position) and `d_len_next` (what attention
     * covers once this token's K and V are written). Both are advanced here,
     * on the stream, so the whole step is a function of device state and can be
     * replayed: `d_len_next` before the layers use it, `d_len` after they do.
     */
    const bool dev_pos = !kv_cache->on_cpu && kv_cache->d_len;
    if (dev_pos) {
        vv_status_t ps = vv_pos_add_dev((int*)kv_cache->d_len_next,
                                        (const int*)kv_cache->d_len, 1,
                                        compute_stream);
        if (ps != VV_OK) return ps;
    }

    if (streaming)
        vv_layer_prefetch_begin(pool, model, first_layer, xfer_stream);

    for (int i = first_layer; i < last_layer; i++) {
        if (streaming) {
            vv_layer_prefetch_wait(pool, model, i, compute_stream);
            if (i + 1 < last_layer)
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

    if (dev_pos) {
        vv_status_t ps = vv_pos_add_dev((int*)kv_cache->d_len,
                                        (const int*)kv_cache->d_len, 1,
                                        compute_stream);
        if (ps != VV_OK) return ps;
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
    void* xfer_stream,
    int first_layer,
    int n_layers)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;
    const int last_layer = first_layer + n_layers;
    if (first_layer < 0 || n_layers <= 0 || last_layer > model->num_layers)
        return VV_ERR_INVALID_ARG;

    if (n_layers == model->num_layers)
        VV_LOG_I("decoder: prefill %d tokens through %d layers%s",
                 seq_len, n_layers,
                 (pool && !pool->all_resident) ? " (streaming)" : "");
    else
        VV_LOG_I("decoder: prefill %d tokens through layers %d..%d%s",
                 seq_len, first_layer, last_layer - 1,
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
    size_t weight_scratch = (size_t)cfg->intermediate_size * hs * 2
                          + decode_scratch_bytes(cfg);
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

    /*
     * Positions continue from whatever the cache already holds: zero for a
     * fresh prompt, the running length when a streaming session prefills its
     * next chunk on top of what it has decoded. Refused up front rather than
     * half-written when it would not fit.
     */
    const int base = kv_cache->current_len;
    if ((long long)base + seq_len > (long long)kv_cache->max_seq_len) {
        VV_LOG_E("decoder: prefill of %d tokens at %d overflows the %d-token "
                 "KV window", seq_len, base, kv_cache->max_seq_len);
        return VV_ERR_OVERFLOW;
    }

    for (int start = 0; start < seq_len; start += chunk) {
      const int len = (start + chunk <= seq_len) ? chunk : (seq_len - start);
      void* chunk_hidden = (uint8_t*)hidden_states + (size_t)start * hs * 2;

      if (streaming)
          vv_layer_prefetch_begin(pool, model, first_layer, xfer_stream);

      for (int i = first_layer; i < last_layer; i++) {

        if (streaming) {
            vv_layer_prefetch_wait(pool, model, i, compute_stream);
            if (i + 1 < last_layer)
                vv_layer_prefetch_begin(pool, model, i + 1, xfer_stream);
        }

        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            chunk_hidden, kv_cache, i, base + start, len,
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
 * CPU decoder — per-layer forward (FP32 activations, quantized weights)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Run one projection in whatever format its weights are stored in. */
static vv_status_t cpu_proj(const vv_weight_t* w, const float* in, float* out,
                            int M, int N, int K) {
    switch (w->quant_kind) {
    case VV_QUANT_INT4G:
        return vv_int4g_gemm_cpu(in, (const uint8_t*)w->tensor.data,
                                 w->quant.scales.data, w->mins.data,
                                 w->bias.data, out, M, N, K, w->group_size);
    case VV_QUANT_NF4:
        return vv_nf4_gemm_cpu(in, (const uint8_t*)w->tensor.data,
                               w->quant.scales.data, w->bias.data,
                               out, M, N, K);
    case VV_QUANT_INT8:
        return vv_int8_gemm_cpu(in, (const int8_t*)w->tensor.data,
                                (const float*)w->quant.scales.data,
                                w->bias.data, out, M, N, K);
    default:
        return vv_gemm_f16w_cpu(in, w->tensor.data, w->bias.data, out,
                                M, N, K);
    }
}

/**
 * @brief One transformer layer on the CPU, FP32 activations.
 *
 * Weights are read in the form they were loaded in. Nothing is converted or
 * materialised here: the earlier version allocated an FP32 copy of every
 * weight and every scale vector on every call, which is 13 GB of traffic and
 * 200-odd mallocs per token.
 */
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
    const int hs = config->hidden_size;
    const int n_heads = config->num_attention_heads;
    const int n_kv_heads = config->num_key_value_heads;
    const int head_dim = config->head_dim;
    const int inter = config->intermediate_size;
    vv_status_t s;

    const size_t need = (size_t)seq_len *
        (3 * (size_t)hs + (size_t)n_heads * head_dim +
         2 * (size_t)n_kv_heads * head_dim + 2 * (size_t)inter);
    if (workspace_size < need * sizeof(float)) return VV_ERR_OUT_OF_MEMORY;

    float* wp = workspace;
    size_t off = 0;
    float* norm_out = wp + off; off += (size_t)seq_len * hs;
    float* q_buf    = wp + off; off += (size_t)seq_len * n_heads * head_dim;
    float* k_buf    = wp + off; off += (size_t)seq_len * n_kv_heads * head_dim;
    float* v_buf    = wp + off; off += (size_t)seq_len * n_kv_heads * head_dim;
    float* attn_out = wp + off; off += (size_t)seq_len * hs;
    float* gate_buf = wp + off; off += (size_t)seq_len * inter;
    float* up_buf   = wp + off; off += (size_t)seq_len * inter;
    float* mlp_out  = wp + off;

    /** Run one projection in whatever format its weights are stored in. */
    #define CPU_PROJ(w, in, out_buf, M, N, KK) do { s = cpu_proj(&(w), (in), (out_buf), (M), (N), (KK)); if (s != VV_OK) return s; } while (0)

    s = vv_rmsnorm_cpu(hidden_states, layer->input_layernorm.data, norm_out,
                       seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) return s;

    CPU_PROJ(layer->attn.q_proj, norm_out, q_buf, seq_len, n_heads * head_dim, hs);
    CPU_PROJ(layer->attn.k_proj, norm_out, k_buf, seq_len, n_kv_heads * head_dim, hs);
    CPU_PROJ(layer->attn.v_proj, norm_out, v_buf, seq_len, n_kv_heads * head_dim, hs);

    s = vv_rope_cpu(q_buf, seq_len, n_heads, head_dim,
                    position_offset, config->rope_theta);
    if (s != VV_OK) return s;
    s = vv_rope_cpu(k_buf, seq_len, n_kv_heads, head_dim,
                    position_offset, config->rope_theta);
    if (s != VV_OK) return s;

    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf, seq_len,
                           false, NULL);
    if (s != VV_OK) return s;

    {
        const void *kc, *vc;
        int cl;
        vv_kv_cache_get(kv_cache, layer_idx, &kc, &vc, &cl);
        /* current_len only advances on the last layer; derive the truth. */
        const int kv_len = position_offset + seq_len;
        s = vv_attention_prefill_cpu(q_buf, (const float*)kc, (const float*)vc,
                                     attn_out, n_heads, n_kv_heads, head_dim,
                                     seq_len, position_offset, kv_len, true);
        if (s != VV_OK) return s;
    }

    CPU_PROJ(layer->attn.o_proj, attn_out, norm_out, seq_len, hs, hs);
    vv_residual_add_cpu(hidden_states, norm_out, seq_len * hs);

    s = vv_rmsnorm_cpu(hidden_states, layer->post_attn_layernorm.data,
                       norm_out, seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) return s;

    CPU_PROJ(layer->mlp.gate_proj, norm_out, gate_buf, seq_len, inter, hs);
    CPU_PROJ(layer->mlp.up_proj,   norm_out, up_buf,   seq_len, inter, hs);
    s = vv_swiglu_cpu(gate_buf, up_buf, gate_buf, seq_len * inter);
    if (s != VV_OK) return s;

    CPU_PROJ(layer->mlp.down_proj, gate_buf, mlp_out, seq_len, hs, inter);
    vv_residual_add_cpu(hidden_states, mlp_out, seq_len * hs);

    #undef CPU_PROJ
    return VV_OK;
}

vv_status_t vv_decoder_prefill_cpu(
    vv_model_t* model,
    float* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size)
{
    if (!model || !hidden_states || !kv_cache || !workspace)
        return VV_ERR_NULL_PTR;
    if (seq_len <= 0) return VV_ERR_INVALID_ARG;

    const vv_llm_config_t* cfg = &model->config.llm;
    const int hs = cfg->hidden_size;

    /*
     * The activations of one layer cost this many floats per token (the two
     * intermediate-wide MLP buffers dominate), and the workspace is fixed.
     * Unchunked, 512 MB ran out at about 2.5K tokens — five minutes of
     * audio on the 7B. Each chunk attends to everything cached before it,
     * so chunking changes how the work is cut, not what it computes.
     */
    const size_t per_token = sizeof(float) *
        (3 * (size_t)hs + (size_t)cfg->num_attention_heads * cfg->head_dim +
         2 * (size_t)cfg->num_key_value_heads * cfg->head_dim +
         2 * (size_t)cfg->intermediate_size);
    size_t fit = workspace_size / per_token;
    if (fit < 1) return VV_ERR_OUT_OF_MEMORY;
    if (fit > 2048) fit = 2048;
    const int chunk = (int)fit < seq_len ? (int)fit : seq_len;

    const int base = kv_cache->current_len;
    if ((long long)base + seq_len > (long long)kv_cache->max_seq_len) {
        VV_LOG_E("decoder: CPU prefill of %d tokens at %d overflows the "
                 "%d-token KV window", seq_len, base, kv_cache->max_seq_len);
        return VV_ERR_OVERFLOW;
    }

    if (chunk < seq_len)
        VV_LOG_I("decoder: CPU prefill %d tokens through %d layers in %d "
                 "chunks of %d", seq_len, model->num_layers,
                 (seq_len + chunk - 1) / chunk, chunk);
    else
        VV_LOG_I("decoder: CPU prefill %d tokens through %d layers",
                 seq_len, model->num_layers);

    for (int start = 0; start < seq_len; start += chunk) {
        const int len = (start + chunk <= seq_len) ? chunk : seq_len - start;
        float* h = hidden_states + (size_t)start * hs;
        for (int i = 0; i < model->num_layers; i++) {
            vv_status_t s = decoder_layer_cpu(
                &model->layers[i], cfg, h, kv_cache, i, base + start, len,
                workspace, workspace_size);
            if (s != VV_OK) {
                VV_LOG_E("decoder: CPU prefill layer %d failed: %s",
                         i, vv_status_str(s));
                return s;
            }
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
