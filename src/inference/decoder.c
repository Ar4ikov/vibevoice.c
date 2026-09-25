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
#include "vibevoice/bitnet.h"

#include <math.h>
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
/**
 * @brief Carve the per-token int8 activation of a ternary projection out of
 *        the scratch: q [M][K], then the FP32 multipliers and int32 row sums.
 */
static vv_status_t ternary_act(const void* x, int M, int K, void* scratch,
                               size_t scratch_bytes, int8_t** q, float** sc,
                               int32_t** sum, void* stream) {
    const size_t qb = ((size_t)M * K + 255) & ~(size_t)255;
    const size_t mb = ((size_t)M * 4 + 255) & ~(size_t)255;
    if (!scratch || scratch_bytes < qb + 2 * mb) return VV_ERR_OVERFLOW;
    *q = (int8_t*)scratch;
    *sc = (float*)((uint8_t*)scratch + qb);
    *sum = (int32_t*)((uint8_t*)scratch + qb + mb);
    return vv_act_quant_i8_dev(x, 1, M, K, *q, *sc, *sum, stream);
}

static vv_status_t ternary_linear(const vv_weight_t* w, const int8_t* q,
                                  const float* sc, const int32_t* sum, void* y,
                                  int M, int N, int K, void* stream) {
    return vv_ternary_gemm_dev(q, sum, sc, (const uint8_t*)w->tensor.data,
                               w->tscale, (const float*)w->bias.data, NULL, y,
                               1, M, N, K, stream);
}

/**
 * @brief The small-M kernel format of a weight, or -1 when it has none.
 *
 * Dense, INT8 and NF4 weights: the formats whose rows 9..64 used to go
 * through an FP16 copy of the weight and the tile GEMM. vv_skinny_linear_dev
 * gives the same bits without either.
 */
static int skinny_format(const vv_weight_t* w) {
    switch (w->quant_kind) {
        case VV_QUANT_NONE: return VV_SKINNY_F16;
        case VV_QUANT_INT8: return VV_SKINNY_INT8;
        case VV_QUANT_NF4:  return VV_SKINNY_NF4;
        default:            return -1;
    }
}

static bool skinny_rows(int M) {
    return M >= VV_SKINNY_M_MIN && M <= VV_SKINNY_M_MAX;
}

static vv_skinny_proj_t skinny_proj(const vv_weight_t* w, void* y, int N) {
    vv_skinny_proj_t p;
    p.w = w->tensor.data;
    p.scales = w->quant_kind == VV_QUANT_NONE ? NULL : w->quant.scales.data;
    p.bias = w->bias.data;
    p.y = y;
    p.N = N;
    return p;
}

/** @brief A W4A16 (GPU layout) weight as the kernels take it. */
static vv_w4a16_proj_t w4a16_proj(const vv_weight_t* w, void* y, int N) {
    vv_w4a16_proj_t p;
    p.packed = w->tensor.data;
    p.sz     = w->quant.scales.data;
    p.bias   = w->bias.data;
    p.y      = y;
    p.N      = N;
    return p;
}

static vv_status_t quant_linear(
    const vv_weight_t* w, const void* x, void* y,
    void* scratch, size_t scratch_bytes, int M, int N, int K, void* stream)
{
    vv_status_t s;

    /* 9..64 rows (a streaming chunk): bias fused, no weight scratch. */
    if (skinny_rows(M) && skinny_format(w) >= 0) {
        const vv_skinny_proj_t p = skinny_proj(w, y, N);
        s = vv_skinny_linear_dev(x, skinny_format(w), &p, 1, M, K, 1.0f,
                                 stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
    }

    if (w->quant_kind == VV_QUANT_TERNARY) {
        /* W1.58A8: quantize the rows, then the ternary product (bias fused). */
        int8_t* q; float* sc; int32_t* sum;
        s = ternary_act(x, M, K, scratch, scratch_bytes, &q, &sc, &sum, stream);
        return s == VV_OK ? ternary_linear(w, q, sc, sum, y, M, N, K, stream) : s;
    }

    /*
     * Act-order GPTQ: the weight's input channels were sorted by group at
     * load, so the activations go through the same order first. The
     * gathered copy sits at the front of the scratch; the kernels get the
     * rest. Graph-capturable: the arguments do not change between tokens.
     */
    if (w->perm.data) {
        const size_t xb = ((size_t)M * K * 2 + 255) & ~(size_t)255;
        const bool legacy_gemm = M > 1 && w->quant_kind == VV_QUANT_INT4G &&
                                 w->int4g_layout != VV_INT4G_GPU;
        const size_t need = xb + (legacy_gemm ? (size_t)N * K * 2 : 0);
        if (!scratch || scratch_bytes < need) {
            VV_LOG_E("decoder: act-order '%s' needs %zu bytes of scratch "
                     "for %d rows, %zu available", w->name, need, M,
                     scratch_bytes);
            return VV_ERR_OVERFLOW;
        }
        s = vv_w4a16_gather_dev(x, (const int32_t*)w->perm.data, scratch,
                                M, K, stream);
        if (s != VV_OK) return s;
        x = scratch;
        scratch = (uint8_t*)scratch + xb;
        scratch_bytes -= xb;
    }

    if (w->quant_kind == VV_QUANT_INT4G &&
        w->int4g_layout == VV_INT4G_GPU) {
        /* W4A16 kernels: bias fused, no dequantized copy of the weight. Up
         * to 16 rows the tensor-core GEMV, a row's bits its own (see
         * linear_rows); where it declines, the FP16-chain GEMV. */
        if (M <= VV_W4A16_MV_MAX_ROWS) {
            const vv_w4a16_proj_t p = w4a16_proj(w, y, N);
            s = vv_w4a16_mv_dev(x, M, &p, 1, K, w->group_size, stream);
            if (s != VV_ERR_UNSUPPORTED) return s;
        }
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
 * @brief Several projections of the same x: y_i = x @ W_i^T + b_i.
 *
 * Up to 16 rows, W4A16 weights of one group size go through a single GEMV
 * launch (q/k/v, gate/up): the 512-row k and v projections are too short
 * to get past the kernel's ramp-up on their own and ride along behind q
 * instead. So do 9..64 rows of dense, INT8 or NF4 weights, through the
 * small-M kernel: alone, k and v would be 8-16 blocks on an 82-SM card.
 * Everything else is one quant_linear each.
 */
static vv_status_t quant_linear_group(
    const vv_weight_t* const* ws, void* const* ys, const int* Ns, int n,
    const void* x, void* scratch, size_t scratch_bytes, int M, int K,
    void* stream)
{
    bool tern = true;
    for (int i = 0; i < n; i++) tern = tern && ws[i]->quant_kind == VV_QUANT_TERNARY;
    if (tern) {
        /* One quantization of x feeds q/k/v (gate/up), as in the reference. */
        int8_t* q; float* sc; int32_t* sum;
        vv_status_t s = ternary_act(x, M, K, scratch, scratch_bytes, &q, &sc,
                                    &sum, stream);
        for (int i = 0; i < n && s == VV_OK; i++)
            s = ternary_linear(ws[i], q, sc, sum, ys[i], M, Ns[i], K, stream);
        return s;
    }
    if (skinny_rows(M) && n <= 3 && skinny_format(ws[0]) >= 0) {
        bool same = true;
        for (int i = 1; i < n; i++)
            same = same && skinny_format(ws[i]) == skinny_format(ws[0]);
        if (same) {
            vv_skinny_proj_t p[3];
            for (int i = 0; i < n; i++) p[i] = skinny_proj(ws[i], ys[i], Ns[i]);
            vv_status_t s = vv_skinny_linear_dev(x, skinny_format(ws[0]), p, n,
                                                 M, K, 1.0f, stream);
            if (s != VV_ERR_UNSUPPORTED) return s;
        }
    }
    bool fuse = M <= VV_W4A16_MV_MAX_ROWS && n <= 3;
    for (int i = 0; i < n && fuse; i++)
        fuse = ws[i]->quant_kind == VV_QUANT_INT4G &&
               ws[i]->int4g_layout == VV_INT4G_GPU && !ws[i]->perm.data &&
               ws[i]->group_size == ws[0]->group_size;
    if (fuse) {
        vv_w4a16_proj_t p[3];
        for (int i = 0; i < n; i++) p[i] = w4a16_proj(ws[i], ys[i], Ns[i]);
        vv_status_t s = vv_w4a16_mv_dev(x, M, p, n, K, ws[0]->group_size,
                                        stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
        if (M == 1)
            return vv_w4a16_gemv_multi_dev(x, p, n, K, ws[0]->group_size,
                                           stream);
    }
    for (int i = 0; i < n; i++) {
        vv_status_t s = quant_linear(ws[i], x, ys[i], scratch, scratch_bytes,
                                     M, Ns[i], K, stream);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

/* ─── Verified blocks: M rows, each exactly as its own decode step ────────── */

/*
 * A drafted block is checked by running its M rows through the model at
 * once, and the tokens that come out have to be the ones M decode steps
 * would have produced. So every op here computes each row exactly as the
 * one-token op does:
 *
 *   W4A16 (GPU layout)   the tensor-core GEMV a step runs too, whose rows
 *                        are independent (vv_w4a16_mv_dev); where it
 *                        declines, the multi-row twin of the FP16-chain GEMV
 *   FP16 dense           the M <= 8 CUDA-core kernel one decode step runs
 *   ternary, W8A8/W4A8   integer sums, exact in any order, for M <= 8
 *   NF4, INT8, legacy    the one-token GEMV once per row (exact, slower)
 *
 * and attention is vv_attn_decode_rows: every row its own decode.
 */
#define VERIFY_M_MAX 16

static vv_status_t linear_rows(const vv_weight_t* w, const void* x, void* y,
                               void* scratch, size_t scratch_bytes, int M,
                               int N, int K, void* stream)
{
    vv_status_t s;
    if (w->perm.data) {
        /* The gather is a copy per row: exact. The kernels read its copy. */
        const size_t xb = ((size_t)M * K * 2 + 255) & ~(size_t)255;
        if (!scratch || scratch_bytes < xb) return VV_ERR_OVERFLOW;
        s = vv_w4a16_gather_dev(x, (const int32_t*)w->perm.data, scratch, M,
                                K, stream);
        if (s != VV_OK) return s;
        x = scratch;
        scratch = (uint8_t*)scratch + xb;
        scratch_bytes -= xb;
    }
    if (w->quant_kind == VV_QUANT_INT4G && w->int4g_layout == VV_INT4G_GPU) {
        /* Declines for the step exactly when it declines here. */
        const vv_w4a16_proj_t p = w4a16_proj(w, y, N);
        s = vv_w4a16_mv_dev(x, M, &p, 1, K, w->group_size, stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
        return vv_w4a16_gemv_rows_dev(x, M, &p, 1, K, w->group_size, stream);
    }
    if (w->quant_kind == VV_QUANT_TERNARY || w->quant_kind == VV_QUANT_NONE) {
        /* Up to 8 rows these are the kernels a decode step runs. */
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = M - m0 < 8 ? M - m0 : 8;
            s = quant_linear(w, (const uint8_t*)x + (size_t)m0 * K * 2,
                             (uint8_t*)y + (size_t)m0 * N * 2, scratch,
                             scratch_bytes, mm, N, K, stream);
            if (s != VV_OK) return s;
        }
        return VV_OK;
    }
    for (int m = 0; m < M; m++) {
        s = quant_linear(w, (const uint8_t*)x + (size_t)m * K * 2,
                         (uint8_t*)y + (size_t)m * N * 2, scratch,
                         scratch_bytes, 1, N, K, stream);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

static vv_status_t linear_rows_group(
    const vv_weight_t* const* ws, void* const* ys, const int* Ns, int n,
    const void* x, void* scratch, size_t scratch_bytes, int M, int K,
    void* stream)
{
    bool fuse = n <= 3;
    for (int i = 0; i < n && fuse; i++)
        fuse = ws[i]->quant_kind == VV_QUANT_INT4G &&
               ws[i]->int4g_layout == VV_INT4G_GPU && !ws[i]->perm.data &&
               ws[i]->group_size == ws[0]->group_size;
    if (fuse) {
        vv_w4a16_proj_t p[3];
        for (int i = 0; i < n; i++) p[i] = w4a16_proj(ws[i], ys[i], Ns[i]);
        vv_status_t s = vv_w4a16_mv_dev(x, M, p, n, K, ws[0]->group_size,
                                        stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
        return vv_w4a16_gemv_rows_dev(x, M, p, n, K, ws[0]->group_size,
                                      stream);
    }
    bool tern = true;
    for (int i = 0; i < n; i++) tern = tern && ws[i]->quant_kind == VV_QUANT_TERNARY;
    if (tern && M <= 8)
        return quant_linear_group(ws, ys, Ns, n, x, scratch, scratch_bytes, M,
                                  K, stream);
    for (int i = 0; i < n; i++) {
        vv_status_t s = linear_rows(ws[i], x, ys[i], scratch, scratch_bytes,
                                    M, Ns[i], K, stream);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

/* ─── Int8 activations (W8A8, W4A8) ─────────────────────────────────────── */

/**
 * @brief Whether a layer runs on int8 activations, and on which device.
 *
 * All seven projections must ask for it (`act_int8`) and share one weight
 * kind, so that one quantized input serves q/k/v and one serves gate/up.
 * The GPU reads INT4 in the W4A16 layout, the CPU in the row-major one with
 * its integer zeros; act-order weights gather FP16 inputs and stay on the
 * FP16 path. On the GPU the fused quantizers also bound the row width
 * (VV_ACT_QUANT_MIN_K..VV_ACT_QUANT_MAX_K). A layer that does not qualify
 * runs its (identical) weights on FP16 activations instead, so this never
 * refuses a model.
 */
static int a8_kmax(const vv_llm_config_t* c);

static bool layer_a8(const vv_layer_weights_t* L, const vv_llm_config_t* c,
                     bool gpu) {
    if (gpu && (a8_kmax(c) > VV_ACT_QUANT_MAX_K ||
                c->hidden_size < VV_ACT_QUANT_MIN_K))
        return false;
    vv_weight_t* p[7];
    vv_layer_projections((vv_layer_weights_t*)L, p);
    for (int i = 0; i < 7; i++) {
        const vv_weight_t* w = p[i];
        if (!w->act_int8 || w->perm.data || w->quant_kind != p[0]->quant_kind)
            return false;
        if (w->quant_kind == VV_QUANT_INT8) continue;
        if (w->quant_kind != VV_QUANT_INT4G || (w->group_size % 32) != 0)
            return false;
        if (gpu ? w->int4g_layout != VV_INT4G_GPU
                : (w->int4g_layout != VV_INT4G_ROWMAJOR || !w->zeros.data))
            return false;
    }
    return true;
}

/** @brief Widest input any projection of this model reads. */
static int a8_kmax(const vv_llm_config_t* c) {
    int k = c->hidden_size;
    if (c->intermediate_size > k) k = c->intermediate_size;
    if (c->num_attention_heads * c->head_dim > k)
        k = c->num_attention_heads * c->head_dim;
    return k;
}

/** @brief Round a workspace offset up to 256 bytes. */
static size_t ws_align(size_t off) { return (off + 255) & ~(size_t)255; }

/** @brief Bytes of int8 activations, scales and per-32 sums for M rows. */
static size_t a8_bytes(const vv_llm_config_t* c, int M) {
    const size_t k = (size_t)a8_kmax(c);
    return ws_align((size_t)M * k) + ws_align((size_t)M * 4) +
           ws_align((size_t)M * (k / 32) * 4) + 256;
}

/**
 * @brief Int8 activations shared by the projections that read one input.
 *
 * Quantized once by the op that produces the input (RMSNorm, SwiGLU, or the
 * attention output) and read by every projection behind it.
 */
typedef struct a8_act {
    int8_t*  xq;     /**< [M][K] int8, in `layout` order                    */
    float*   sx;     /**< [M] per-token scale                               */
    int32_t* xsum;   /**< [M][K/32] sums of xq, for the W4A8 zero points     */
    int      layout; /**< vv_q8_layout_t the quantizers write               */
} a8_act_t;

/**
 * @brief One W8A8 / W4A8 projection: y = W . act + bias (+ residual).
 *
 * `residual` may be `y` itself, which is how o_proj and down_proj add into
 * the hidden state without a separate kernel.
 */
static vv_status_t a8_linear(const vv_weight_t* w, const a8_act_t* a,
                             void* y, const void* residual,
                             int M, int N, int K, void* stream)
{
    if (w->quant_kind == VV_QUANT_INT8)
        return vv_w8a8_linear_dev(a->xq, a->sx, (const int8_t*)w->tensor.data,
                                  (const float*)w->quant.scales.data,
                                  w->bias.data, residual, y, 0, M, N, K,
                                  VV_I8_PATH_AUTO, stream);
    return vv_w4a8_linear_dev(a->xq, a->layout, a->sx, a->xsum,
                              w->tensor.data, w->quant.scales.data,
                              w->group_size, w->bias.data, residual, y, 0,
                              M, N, K, VV_I8_PATH_AUTO, stream);
}

/**
 * @brief Projections that read the same int8 input (q/k/v, gate/up): one
 *        GEMV launch for decode-sized M, one GEMM each otherwise.
 */
static vv_status_t a8_group(const vv_weight_t* const* ws, void* const* ys,
                            const int* Ns, int n, const a8_act_t* a,
                            int M, int K, void* stream)
{
    vv_i8_proj_t p[3];
    for (int i = 0; i < n; i++) {
        p[i].w    = ws[i]->tensor.data;
        p[i].sw   = ws[i]->quant.scales.data;
        p[i].sz   = ws[i]->quant.scales.data;
        p[i].bias = ws[i]->bias.data;
        p[i].y    = ys[i];
        p[i].N    = Ns[i];
    }
    return vv_i8_linear_multi_dev(a->xq, a->layout, a->sx, a->xsum,
                                  ws[0]->quant_kind == VV_QUANT_INT4G, p, n,
                                  ws[0]->group_size, M, K, stream);
}

/* ─── SmoothQuant calibration ───────────────────────────────────────────── */

/*
 * A layer with `calib_absmax` set (vv_smooth_calibrate, smooth.h) records
 * the per-channel absmax of each projection input into its row: the two
 * RMSNorm outputs, the attention output and the SwiGLU output. Only dense
 * FP16 layers are calibrated, which is what the calibration pass loads.
 */
#define CALIB_ATTN_IN(c)  0
#define CALIB_MLP_IN(c)   ((c)->hidden_size)
#define CALIB_ATTN_OUT(c) (2 * (c)->hidden_size)
#define CALIB_MLP_MID(c)  (2 * (c)->hidden_size + \
                           (c)->num_attention_heads * (c)->head_dim)

static vv_status_t calib_dev(const vv_layer_weights_t* L, int off,
                             const void* x, int M, int K, void* stream) {
    if (!L->calib_absmax) return VV_OK;
    return vv_col_absmax_dev(x, M, K, L->calib_absmax + off, stream);
}

static void calib_cpu(const vv_layer_weights_t* L, int off, const float* x,
                      int M, int K) {
    if (!L->calib_absmax) return;
    float* acc = L->calib_absmax + off;
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            const float v = fabsf(x[(size_t)m * K + k]);
            if (v > acc[k]) acc[k] = v;
        }
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
    return vv_attn_scratch_bytes(cfg->num_attention_heads,
                                 cfg->num_key_value_heads, cfg->head_dim);
}

/** @brief Buffer slot of `layer` in `t`, or -1 when it is not tapped. */
static int tap_slot(const vv_taps_t* t, int layer) {
    if (!t || !t->buf) return -1;
    for (int i = 0; i < t->n; i++)
        if (t->layers[i] == layer) return i;
    return -1;
}

static vv_status_t decoder_layer_body(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    void* temp_workspace,
    size_t workspace_size,
    void* stream,
    const vv_verify_opts_t* vo);

/**
 * @brief One layer, then its output rows into their slot of `taps` (rows
 *        `tap_row0`..) when the layer is tapped. `vo` set: the rows of a
 *        verified block (vv_decoder_verify).
 */
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
    void* stream,
    const vv_taps_t* taps,
    int tap_row0,
    const vv_verify_opts_t* vo)
{
    vv_status_t s = decoder_layer_body(layer, config, hidden_states, kv_cache,
                                       layer_idx, position_offset, seq_len,
                                       temp_workspace, workspace_size, stream,
                                       vo);
    const int slot = tap_slot(taps, layer_idx);
    if (s != VV_OK || slot < 0) return s;
    if (tap_row0 < 0 || tap_row0 + seq_len > taps->rows) return VV_ERR_OVERFLOW;
    const size_t row = (size_t)config->hidden_size * 2;
    return vv_dev_memcpy2d_d2d(
        (uint8_t*)taps->buf + ((size_t)tap_row0 * taps->n + slot) * row,
        row * (size_t)taps->n, hidden_states, row, row, (size_t)seq_len,
        stream);
}

static vv_status_t decoder_layer_body(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    void* temp_workspace,
    size_t workspace_size,
    void* stream,
    const vv_verify_opts_t* vo)
{
    if (!layer || !config || !hidden_states || !kv_cache ||
        !temp_workspace) {
        return VV_ERR_NULL_PTR;
    }
    const bool verify = vo != NULL;
    /* Each row with its decode step's arithmetic, or the prefill's. */
    const bool rows_exact = verify && !vo->fast;

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
     * A verified block brings its own device positions when it is replayed
     * (vo->d_pos: row 0's, rows follow it).
     */
    const bool dev_pos = !verify && (seq_len == 1) && !kv_cache->on_cpu &&
                         kv_cache->d_len;
    const int* d_pos  = dev_pos ? (const int*)kv_cache->d_len
                                : (verify ? vo->d_pos : NULL);
    const int* d_next = dev_pos ? (const int*)kv_cache->d_len_next
                                : (verify ? vo->d_next : NULL);

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
    offset += verify ? vv_attn_rows_scratch_bytes(n_heads, head_dim, seq_len)
                     : decode_scratch_bytes(config);

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

    /*
     * Int8-activation layers (W8A8, W4A8) dequantize nothing; the scratch
     * holds the quantized activations instead. INT4 weights want them in
     * the order the GEMV or the GEMM reads (q8.h), INT8 in column order.
     */
    const bool a8 = layer_a8(layer, config, true);
    a8_act_t act = { NULL, NULL, NULL, VV_Q8_NATURAL };
    if (a8) {
        if (layer->attn.q_proj.quant_kind == VV_QUANT_INT4G)
            act.layout = vv_w4a8_layout_for(seq_len);
        const size_t kmax = (size_t)a8_kmax(config);
        size_t o = ws_align(offset);
        act.xq = (int8_t*)(wp + o);    o = ws_align(o + (size_t)seq_len * kmax);
        act.sx = (float*)(wp + o);     o = ws_align(o + (size_t)seq_len * 4);
        act.xsum = (int32_t*)(wp + o); o += (size_t)seq_len * (kmax / 32) * 4;
        if (o > workspace_size) return VV_ERR_OUT_OF_MEMORY;
    }

    /* 1. Input LayerNorm (fused with the int8 quantizer on A8 layers) */
    if (a8)
        s = vv_rmsnorm_q8_dev(hidden_states, layer->input_layernorm.data,
                              seq_len, hs, config->rms_norm_eps, act.layout,
                              act.xq, act.sx, act.xsum, stream);
    else
        s = vv_rmsnorm_dev(hidden_states, layer->input_layernorm.data,
                           norm_out, seq_len, hs,
                           config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 2. Q, K, V projections (+ bias if present) */
    if (a8) {
        const vv_weight_t* ws[3] = { &layer->attn.q_proj, &layer->attn.k_proj,
                                     &layer->attn.v_proj };
        void* ys[3] = { q_buf, k_buf, v_buf };
        const int ns[3] = { n_heads * head_dim, n_kv_heads * head_dim,
                            n_kv_heads * head_dim };
        s = a8_group(ws, ys, ns, 3, &act, seq_len, hs, stream);
        if (s != VV_OK) return s;
    } else {
        const vv_weight_t* ws[3] = { &layer->attn.q_proj, &layer->attn.k_proj,
                                     &layer->attn.v_proj };
        void* ys[3] = { q_buf, k_buf, v_buf };
        const int ns[3] = { n_heads * head_dim, n_kv_heads * head_dim,
                            n_kv_heads * head_dim };
        s = rows_exact ? linear_rows_group(ws, ys, ns, 3, norm_out,
                                           temp_weight, temp_bytes, seq_len,
                                           hs, stream)
                       : quant_linear_group(ws, ys, ns, 3, norm_out, temp_weight,
                                            temp_bytes, seq_len, hs, stream);
        if (s != VV_OK) return s;
    }

    if (!a8 && (s = calib_dev(layer, CALIB_ATTN_IN(config), norm_out,
                              seq_len, hs, stream)) != VV_OK)
        return s;

    if (layer_idx == 0 && seq_len > 1 && !a8) {
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
    s = vv_kv_cache_append_at(kv_cache, layer_idx, k_buf, v_buf, seq_len,
                              d_pos, stream);
    if (s != VV_OK) return s;

    /* 5. GQA attention over the cache (queries of this chunk see all of it) */
    {
        /*
         * current_len only advances on the last layer, so derive the true
         * length from this call's position instead.
         */
        const int actual_cache_len = position_offset + seq_len;
        const vv_kv_view_t view = vv_kv_cache_view(kv_cache, layer_idx);
        const int backend = kv_cache->attn_backend;
        const bool rotates = vv_kv_rotates((vv_kv_format_t)kv_cache->format);

        /*
         * TurboQuant stores rotated vectors. The transform is orthogonal, so
         * instead of inverting it per key we rotate Q once here and undo the
         * rotation on the output; the kernels then read stored values
         * directly, whichever backend they belong to.
         */
        if (rotates) {
            s = vv_kv_rotate_dev(q_buf, n_heads, head_dim, seq_len, stream);
            if (s != VV_OK) return s;
        }
        if (verify)
            s = vv_attn_decode_rows(vo->attn_backend >= 0 ? vo->attn_backend
                                                          : backend,
                                    q_buf, &view, attn_out, n_heads, seq_len,
                                    position_offset + 1, d_next,
                                    decode_scratch, stream);
        else if (seq_len > 1)
            s = vv_attn_prefill(backend, q_buf, &view, attn_out, n_heads,
                                seq_len, position_offset, actual_cache_len,
                                true, decode_scratch, stream);
        else
            s = vv_attn_decode(backend, q_buf, &view, attn_out, n_heads,
                               actual_cache_len, d_next, decode_scratch,
                               stream);
        if (s == VV_OK && rotates)
            s = vv_kv_unrotate_dev(attn_out, n_heads, head_dim, seq_len,
                                   stream);
    }
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1)
        dump_gpu_fp16("c_l0_attn", attn_out, (size_t)seq_len * hs, stream);

    if (a8) {
        /*
         * 6-10 on int8 activations. o_proj and down_proj write the residual
         * sum straight into the hidden state; the quantizers are fused into
         * the RMSNorm and the SwiGLU that feed the next projections.
         */
        s = vv_act_quant_dev(attn_out, seq_len, n_heads * head_dim,
                             act.layout, act.xq, act.sx, act.xsum, stream);
        if (s == VV_OK)
            s = a8_linear(&layer->attn.o_proj, &act, hidden_states,
                          hidden_states, seq_len, hs, n_heads * head_dim,
                          stream);
        if (s == VV_OK)
            s = vv_rmsnorm_q8_dev(hidden_states,
                                  layer->post_attn_layernorm.data, seq_len,
                                  hs, config->rms_norm_eps, act.layout,
                                  act.xq, act.sx, act.xsum, stream);
        if (s == VV_OK) {
            const vv_weight_t* ws[2] = { &layer->mlp.gate_proj,
                                         &layer->mlp.up_proj };
            void* ys[2] = { gate_buf, up_buf };
            const int ns[2] = { inter_size, inter_size };
            s = a8_group(ws, ys, ns, 2, &act, seq_len, hs, stream);
        }
        if (s == VV_OK)
            s = vv_swiglu_q8_dev(gate_buf, up_buf, seq_len, inter_size,
                                 act.layout, act.xq, act.sx, act.xsum,
                                 stream);
        if (s == VV_OK)
            s = a8_linear(&layer->mlp.down_proj, &act, hidden_states,
                          hidden_states, seq_len, hs, inter_size, stream);
        (void)mlp_out;
        return s;
    }

    s = calib_dev(layer, CALIB_ATTN_OUT(config), attn_out, seq_len,
                  n_heads * head_dim, stream);
    if (s != VV_OK) return s;

    /* 6. O projection + bias + residual */
    s = rows_exact ? linear_rows(&layer->attn.o_proj, attn_out, norm_out,
                                 temp_weight, temp_bytes, seq_len, hs, hs,
                                 stream)
                   : quant_linear(&layer->attn.o_proj, attn_out, norm_out,
                                  temp_weight, temp_bytes, seq_len, hs, hs,
                                  stream);
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
    s = calib_dev(layer, CALIB_MLP_IN(config), norm_out, seq_len, hs, stream);
    if (s != VV_OK) return s;

    /* 8. MLP: gate + up (+ bias if present) */
    {
        const vv_weight_t* ws[2] = { &layer->mlp.gate_proj,
                                     &layer->mlp.up_proj };
        void* ys[2] = { gate_buf, up_buf };
        const int ns[2] = { inter_size, inter_size };
        s = rows_exact ? linear_rows_group(ws, ys, ns, 2, norm_out,
                                           temp_weight, temp_bytes, seq_len,
                                           hs, stream)
                       : quant_linear_group(ws, ys, ns, 2, norm_out, temp_weight,
                                            temp_bytes, seq_len, hs, stream);
        if (s != VV_OK) return s;
    }

    /* 9. SwiGLU */
    s = vv_swiglu_dev(gate_buf, up_buf, gate_buf,
                        seq_len * inter_size, stream);
    if (s != VV_OK) return s;
    s = calib_dev(layer, CALIB_MLP_MID(config), gate_buf, seq_len,
                  inter_size, stream);
    if (s != VV_OK) return s;

    /* 10. Down projection + bias + residual */
    s = rows_exact ? linear_rows(&layer->mlp.down_proj, gate_buf, mlp_out,
                                 temp_weight, temp_bytes, seq_len, hs,
                                 inter_size, stream)
                   : quant_linear(&layer->mlp.down_proj, gate_buf, mlp_out,
                                  temp_weight, temp_bytes, seq_len, hs,
                                  inter_size, stream);
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
                               temp_workspace, workspace_size, stream,
                               NULL, 0, NULL);
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
    return vv_decoder_step_taps(model, hidden_state, kv_cache, pool, workspace,
                                workspace_size, compute_stream, xfer_stream,
                                first_layer, n_layers, NULL);
}

vv_status_t vv_decoder_step_taps(
    vv_model_t* model,
    void* hidden_state,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    int first_layer,
    int n_layers,
    const vv_taps_t* taps)
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
    /* A paged cache needs a page for this position. The caller maps ahead
     * of any capture, so inside one this finds the page already there. */
    {
        vv_status_t rs = vv_kv_cache_reserve(kv_cache, position + 1,
                                             compute_stream);
        if (rs != VV_OK) return rs;
    }

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
            workspace, workspace_size, compute_stream, taps, 0, NULL);

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
    return vv_decoder_prefill_taps(model, hidden_states, seq_len, kv_cache,
                                   pool, workspace, workspace_size,
                                   compute_stream, xfer_stream, first_layer,
                                   n_layers, NULL);
}

vv_status_t vv_decoder_prefill_taps(
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
    int n_layers,
    const vv_taps_t* taps)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;
    const int last_layer = first_layer + n_layers;
    if (first_layer < 0 || n_layers <= 0 || last_layer > model->num_layers)
        return VV_ERR_INVALID_ARG;

    /* A prefill on top of a filled cache is a streaming chunk, one every
     * three seconds per session: not worth a line each. */
    const vv_log_level_t lvl = kv_cache->current_len ? VV_LOG_DEBUG
                                                     : VV_LOG_INFO;
    if (n_layers == model->num_layers)
        vv_log(lvl, "decoder: prefill %d tokens through %d layers%s",
               seq_len, n_layers,
               (pool && !pool->all_resident) ? " (streaming)" : "");
    else
        vv_log(lvl, "decoder: prefill %d tokens through layers %d..%d%s",
               seq_len, first_layer, last_layer - 1,
               (pool && !pool->all_resident) ? " (streaming)" : "");

    bool streaming = pool && !pool->all_resident;

    const vv_llm_config_t* cfg = &model->config.llm;
    const int hs = cfg->hidden_size;

    {
        vv_status_t rs = vv_kv_cache_reserve(
            kv_cache, kv_cache->current_len + seq_len, compute_stream);
        if (rs != VV_OK) {
            VV_LOG_E("decoder: no KV room for %d more positions", seq_len);
            return rs;
        }
    }

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
    /*
     * Int8-activation layers dequantize no weight; they carve their int8
     * activations, scales and sums instead, which grow with the chunk.
     */
    {
        bool any_a8 = false, all_a8 = true;
        for (int i = first_layer; i < last_layer; i++) {
            const bool a = layer_a8(&model->layers[i], cfg, true);
            any_a8 |= a;
            all_a8 &= a;
        }
        if (all_a8) weight_scratch = decode_scratch_bytes(cfg) + 4 * 256;
        if (any_a8) per_token += a8_bytes(cfg, 1);
    }
    int chunk = seq_len;
    if (workspace_size > weight_scratch + per_token) {
        size_t budget = (workspace_size - weight_scratch) / per_token;
        if (budget < 1) budget = 1;
        if (budget > 2048) budget = 2048;
        if ((int)budget < chunk) chunk = (int)budget;
    }
    if (chunk < 1) chunk = 1;
    /* Handed over chunk by chunk, the taps only have to hold one chunk. */
    if (taps && taps->on_chunk && chunk > taps->rows) chunk = taps->rows;
    if (chunk < 1) return VV_ERR_INVALID_ARG;
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
            workspace, workspace_size, compute_stream, taps,
            (taps && taps->on_chunk) ? 0
                                     : (taps ? base + start - taps->row_base : 0),
            NULL);

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
      if (taps && taps->on_chunk) {
          vv_status_t s = taps->on_chunk(taps->user, base + start, len,
                                         compute_stream);
          if (s != VV_OK) return s;
      }
    }

    return VV_OK;
}

vv_status_t vv_decoder_verify(
    vv_model_t* model,
    void* hidden_states,
    int rows,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    const vv_taps_t* taps,
    const vv_verify_opts_t* opts)
{
    /* No options: the host's positions and the cache's attention. */
    static const vv_verify_opts_t host_pos = { NULL, NULL, -1, false };
    const vv_verify_opts_t* vo = opts ? opts : &host_pos;
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;
    if (rows < 1 || rows > vv_decoder_verify_rows_max(model))
        return VV_ERR_INVALID_ARG;
    if ((vo->d_pos != NULL) != (vo->d_next != NULL)) return VV_ERR_INVALID_ARG;
    const int base = kv_cache->current_len;
    if (base + rows > kv_cache->max_seq_len) return VV_ERR_OVERFLOW;
    vv_status_t s = vv_kv_cache_reserve(kv_cache, base + rows, compute_stream);
    if (s != VV_OK) return s;
    const bool streaming = pool && !pool->all_resident;
    if (streaming)
        vv_layer_prefetch_begin(pool, model, 0, xfer_stream);
    for (int i = 0; i < model->num_layers; i++) {
        if (streaming) {
            vv_layer_prefetch_wait(pool, model, i, compute_stream);
            if (i + 1 < model->num_layers)
                vv_layer_prefetch_begin(pool, model, i + 1, xfer_stream);
        }
        s = decoder_layer_impl(&model->layers[i], &model->config.llm,
                               hidden_states, kv_cache, i, base, rows,
                               workspace, workspace_size, compute_stream,
                               taps, taps ? base - taps->row_base : 0, vo);
        if (streaming) vv_layer_prefetch_done(pool, i, compute_stream);
        if (s != VV_OK) {
            VV_LOG_E("decoder: verify layer %d failed: %s", i,
                     vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}

int vv_decoder_verify_rows_max(const vv_model_t* model) {
    if (!model) return VERIFY_M_MAX;
    for (int i = 0; i < model->num_layers; i++) {
        const vv_layer_weights_t* L = &model->layers[i];
        if (!layer_a8(L, &model->config.llm, true) ||
            L->attn.q_proj.quant_kind != VV_QUANT_INT4G)
            continue;
        /* W4A8: the rows that still take the decode step's GEMV layout */
        int m = VERIFY_M_MAX;
        while (m > 1 && vv_w4a8_layout_for(m) != VV_Q8_W4_GEMV) m--;
        return m;
    }
    return VERIFY_M_MAX;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU decoder — per-layer forward (FP32 activations, quantized weights)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One CPU projection, in whatever format its weights are stored in.
 *
 * An act-order INT4G weight (w->perm set) had its input channels sorted by
 * group at load; its input is gathered into `gather` first.
 */
static vv_status_t cpu_proj(const vv_weight_t* w, const float* in,
                            float* out, int M, int N, int K,
                            float* gather, size_t gather_n)
{
    if (w->perm.data) {
        const int32_t* perm = (const int32_t*)w->perm.data;
        if (!gather || gather_n < (size_t)M * K) {
            VV_LOG_E("decoder: act-order '%s' needs %zu floats of CPU "
                     "workspace for %d rows, %zu left", w->name,
                     (size_t)M * K, M, gather_n);
            return VV_ERR_OVERFLOW;
        }
        for (int m = 0; m < M; m++) {
            const float* src = in + (size_t)m * K;
            float* dst = gather + (size_t)m * K;
            for (int j = 0; j < K; j++) dst[j] = src[perm[j]];
        }
        in = gather;
    }
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
    case VV_QUANT_TERNARY: {
        /*
         * W1.58A8: per-token int8 rows, then the exact ternary product with
         * ggml's epilogue and the FP32 bias. asr-bitnet layers normally run
         * whole through vv_bitnet_layer_cpu; this is for any other caller.
         * The int8 rows and their multipliers live in the gather space.
         */
        const size_t need = ((size_t)M * K + 3) / 4 + (size_t)M;
        if (w->perm.data || !gather || gather_n < need) {
            VV_LOG_E("decoder: ternary '%s' needs %zu floats of CPU "
                     "workspace for %d rows, %zu left", w->name, need, M,
                     gather_n);
            return VV_ERR_OVERFLOW;
        }
        float* sc = gather;
        int8_t* q = (int8_t*)(gather + M);
        vv_status_t s = vv_act_quant_i8_cpu(in, M, K, q, sc, NULL);
        if (s != VV_OK) return s;
        return vv_ternary_linear_cpu(q, sc, (const uint8_t*)w->tensor.data,
                                     w->tscale, (const float*)w->bias.data,
                                     out, M, N, K);
    }
    default:
        return vv_gemm_f16w_cpu(in, w->tensor.data, w->bias.data, out,
                                M, N, K);
    }
}

/** @brief Whether any projection carries an act-order permutation. */
static bool model_has_act_order(vv_model_t* model) {
    for (int i = 0; i < model->num_layers; i++) {
        vv_weight_t* p[7];
        const int n = vv_layer_projections(&model->layers[i], p);
        for (int k = 0; k < n; k++)
            if (p[k]->perm.data) return true;
    }
    return false;
}

/**
 * @brief One transformer layer on the CPU, FP32 activations.
 *
 * Weights are read in the form they were loaded in. Nothing is converted or
 * materialised here: the earlier version allocated an FP32 copy of every
 * weight and every scale vector on every call, which is 13 GB of traffic and
 * 200-odd mallocs per token.
 */
static vv_status_t decoder_layer_cpu_body(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    float* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    float* workspace,
    size_t workspace_size);

/** @brief One CPU layer, then its output rows into `taps` when tapped. */
static vv_status_t decoder_layer_cpu(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    float* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    float* workspace,
    size_t workspace_size,
    const vv_taps_t* taps,
    int tap_row0)
{
    vv_status_t s = decoder_layer_cpu_body(layer, config, hidden_states,
                                           kv_cache, layer_idx,
                                           position_offset, seq_len,
                                           workspace, workspace_size);
    const int slot = tap_slot(taps, layer_idx);
    if (s != VV_OK || slot < 0) return s;
    if (tap_row0 < 0 || tap_row0 + seq_len > taps->rows) return VV_ERR_OVERFLOW;
    const size_t hs = (size_t)config->hidden_size;
    float* dst = (float*)taps->buf;
    for (int r = 0; r < seq_len; r++)
        memcpy(dst + ((size_t)(tap_row0 + r) * taps->n + slot) * hs,
               hidden_states + (size_t)r * hs, hs * sizeof(float));
    return VV_OK;
}

static vv_status_t decoder_layer_cpu_body(
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

    /* Int8 activations, scales and per-32 sums, counted in floats. */
    const bool a8 = layer_a8(layer, config, false);
    const size_t kmax = (size_t)a8_kmax(config);
    const size_t a8_floats = a8
        ? (size_t)seq_len * (kmax / 4 + 1 + kmax / 32) + 64 : 0;
    /* BitNet: its own layer, with the reference's numerics (bitnet_lm.c). */
    if (layer->attn.q_proj.quant_kind == VV_QUANT_TERNARY)
        return vv_bitnet_layer_cpu(layer, config, hidden_states, kv_cache,
                                   layer_idx, position_offset, seq_len,
                                   workspace, workspace_size);

    const size_t need = (size_t)seq_len *
        (3 * (size_t)hs + (size_t)n_heads * head_dim +
         2 * (size_t)n_kv_heads * head_dim + 2 * (size_t)inter) + a8_floats;
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
    /* Whatever is left past the layer's buffers gathers act-order inputs. */
    float* gather = wp + need;
    const size_t gather_n = workspace_size / sizeof(float) - need;

    /*
     * Int8-activation layers: the quantized input lives after mlp_out and is
     * rebuilt by CPU_Q8 whenever the projections' input changes. The CPU
     * W4A8 kernel reads row-major INT4 bytes, so its activations come in
     * the nibble order (q8.h).
     */
    int8_t*  a8_xq = NULL;
    float*   a8_sx = NULL;
    int32_t* a8_xs = NULL;
    const int a8_layout = layer->attn.q_proj.quant_kind == VV_QUANT_INT4G
                        ? VV_Q8_NIBBLE : VV_Q8_NATURAL;
    if (a8) {
        float* base = mlp_out + (size_t)seq_len * hs;
        a8_xq = (int8_t*)base;
        a8_sx = base + (size_t)seq_len * (kmax / 4);
        a8_xs = (int32_t*)(a8_sx + seq_len);
    }

    /** Quantize `in` [seq_len][KK] for the projections behind it. */
    #define CPU_Q8(in, KK) do {                                              \
        if (a8) {                                                           \
            s = vv_quant_act_q8_cpu((in), seq_len, (KK), a8_layout,         \
                                    a8_xq, a8_sx, a8_xs);                   \
            if (s != VV_OK) return s;                                       \
        }                                                                   \
    } while (0)

    /** Run one projection in whatever format its weights are stored in. */
    #define CPU_PROJ(w, in, out_buf, M, N, KK) do {                          \
        if (a8 && (w).quant_kind == VV_QUANT_INT8)                          \
            s = vv_w8a8_gemm_cpu(a8_xq, a8_sx,                              \
                    (const int8_t*)(w).tensor.data,                         \
                    (const float*)(w).quant.scales.data, (w).bias.data,     \
                    (out_buf), (M), (N), (KK));                             \
        else if (a8)                                                        \
            s = vv_w4a8_gemm_cpu(a8_xq, a8_sx, a8_xs,                       \
                    (const uint8_t*)(w).tensor.data, (w).quant.scales.data, \
                    (const uint8_t*)(w).zeros.data, (w).group_size,         \
                    (w).bias.data, (out_buf), (M), (N), (KK));              \
        else                                                                \
            s = cpu_proj(&(w), (in), (out_buf), (M), (N), (KK),             \
                         gather, gather_n);                                 \
        if (s != VV_OK) return s;                                           \
    } while (0)

    s = vv_rmsnorm_cpu(hidden_states, layer->input_layernorm.data, norm_out,
                       seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) return s;

    calib_cpu(layer, CALIB_ATTN_IN(config), norm_out, seq_len, hs);
    CPU_Q8(norm_out, hs);
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

    calib_cpu(layer, CALIB_ATTN_OUT(config), attn_out, seq_len,
              n_heads * head_dim);
    CPU_Q8(attn_out, n_heads * head_dim);
    CPU_PROJ(layer->attn.o_proj, attn_out, norm_out, seq_len, hs,
             n_heads * head_dim);
    vv_residual_add_cpu(hidden_states, norm_out, seq_len * hs);

    s = vv_rmsnorm_cpu(hidden_states, layer->post_attn_layernorm.data,
                       norm_out, seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) return s;

    calib_cpu(layer, CALIB_MLP_IN(config), norm_out, seq_len, hs);
    CPU_Q8(norm_out, hs);
    CPU_PROJ(layer->mlp.gate_proj, norm_out, gate_buf, seq_len, inter, hs);
    CPU_PROJ(layer->mlp.up_proj,   norm_out, up_buf,   seq_len, inter, hs);
    s = vv_swiglu_cpu(gate_buf, up_buf, gate_buf, seq_len * inter);
    if (s != VV_OK) return s;

    calib_cpu(layer, CALIB_MLP_MID(config), gate_buf, seq_len, inter);
    CPU_Q8(gate_buf, inter);
    CPU_PROJ(layer->mlp.down_proj, gate_buf, mlp_out, seq_len, hs, inter);
    vv_residual_add_cpu(hidden_states, mlp_out, seq_len * hs);

    #undef CPU_PROJ
    #undef CPU_Q8
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
    return vv_decoder_prefill_cpu_taps(model, hidden_states, seq_len,
                                       kv_cache, workspace, workspace_size,
                                       NULL);
}

vv_status_t vv_decoder_prefill_cpu_taps(
    vv_model_t* model,
    float* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size,
    const vv_taps_t* taps)
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
    size_t per_token = sizeof(float) *
        (3 * (size_t)hs + (size_t)cfg->num_attention_heads * cfg->head_dim +
         2 * (size_t)cfg->num_key_value_heads * cfg->head_dim +
         2 * (size_t)cfg->intermediate_size);
    /* Act-order INT4 weights gather their input past the layer's buffers:
       one row of the widest K per token. */
    if (model_has_act_order(model))
        per_token += sizeof(float) *
            (size_t)(cfg->intermediate_size > hs ? cfg->intermediate_size : hs);
    /* Int8 activations: a byte per column plus a scale and the sums. */
    bool any_a8 = false;
    for (int i = 0; i < model->num_layers; i++)
        any_a8 |= layer_a8(&model->layers[i], cfg, false);
    if (any_a8)
        per_token += sizeof(float) * ((size_t)a8_kmax(cfg) / 4 + 1 +
                                      (size_t)a8_kmax(cfg) / 32) + 64;
    size_t fit = workspace_size / per_token;
    if (model->num_layers > 0 &&
        model->layers[0].attn.q_proj.quant_kind == VV_QUANT_TERNARY) {
        /* The BitNet layer's workspace grows with the cache it attends to. */
        const int kv_end = kv_cache->current_len + seq_len;
        const size_t fixed = vv_bitnet_layer_cpu_bytes(cfg, 0, kv_end);
        const size_t per = vv_bitnet_layer_cpu_bytes(cfg, 1, kv_end) - fixed;
        fit = workspace_size > fixed ? (workspace_size - fixed) / per : 0;
    }
    if (fit < 1) return VV_ERR_OUT_OF_MEMORY;
    if (fit > 2048) fit = 2048;
    if (taps && taps->on_chunk && fit > (size_t)taps->rows)
        fit = (size_t)taps->rows;
    if (fit < 1) return VV_ERR_INVALID_ARG;
    const int chunk = (int)fit < seq_len ? (int)fit : seq_len;

    const int base = kv_cache->current_len;
    if ((long long)base + seq_len > (long long)kv_cache->max_seq_len) {
        VV_LOG_E("decoder: CPU prefill of %d tokens at %d overflows the "
                 "%d-token KV window", seq_len, base, kv_cache->max_seq_len);
        return VV_ERR_OVERFLOW;
    }

    if (kv_cache->current_len)
        VV_LOG_D("decoder: CPU prefill %d tokens at %d", seq_len,
                 kv_cache->current_len);
    else if (chunk < seq_len)
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
                workspace, workspace_size, taps,
                (taps && taps->on_chunk) ? 0
                                         : (taps ? base + start - taps->row_base : 0));
            if (s != VV_OK) {
                VV_LOG_E("decoder: CPU prefill layer %d failed: %s",
                         i, vv_status_str(s));
                return s;
            }
        }
        if (taps && taps->on_chunk) {
            vv_status_t s = taps->on_chunk(taps->user, base + start, len, NULL);
            if (s != VV_OK) return s;
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
            workspace, workspace_size, NULL, 0);
        if (s != VV_OK) {
            VV_LOG_E("decoder: CPU step layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}
