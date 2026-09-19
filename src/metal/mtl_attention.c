/**
 * @file mtl_attention.c
 * @brief Attention and KV-cache ops on Metal (kernels/attention.metal).
 *
 * The three backend names select two kernel families here. Decode is one
 * split-KV kernel for every backend -- the port of fa2's, whose arithmetic
 * CUDA's fa1 decode kernels also share -- over any KV format, contiguous or
 * paged. Prefill is the scalar fa1 kernel for `fa1`, and the GQA-packed
 * simdgroup-matrix kernel for `fa2` and `flashinfer`, which reads every
 * format and follows page tables.
 */

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/vibevoice.h"
#include "metal_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ATT_D 128

/* Split-KV decode geometry and scratch layout, as src/cuda/decode_split.h. */
#define DEC_WARPS        8
#define DEC_POS_PER_WARP 128
#define DEC_MAX_PARTS    256

static size_t parts_bytes(int n_q_heads, int head_dim) {
    const size_t rows = (size_t)n_q_heads * DEC_MAX_PARTS;
    return rows * ((size_t)head_dim + 2) * sizeof(float);
}

static int n_parts_for(int cache_len) {
    int n = (cache_len + DEC_POS_PER_WARP - 1) / DEC_POS_PER_WARP;
    if (n < 1) n = 1;
    if (n > DEC_MAX_PARTS) n = DEC_MAX_PARTS;
    return ((n + DEC_WARPS - 1) / DEC_WARPS) * DEC_WARPS;
}

static const char* fmt_suffix(int fmt) {
    switch (fmt) {
        case VV_KV_FP16:     return "fp16";
        case VV_KV_FP8_E4M3: return "fp8_e4m3";
        case VV_KV_FP8_E5M2: return "fp8_e5m2";
        case VV_KV_TQ4:      return "tq4";
        case VV_KV_TQ3:      return "tq3";
        case VV_KV_TQ2:      return "tq2";
        case VV_KV_TQ1_5:    return "tq1_5";
        default:             return NULL;
    }
}

static bool gqa_ok(int n_q_heads, int n_kv_heads, int head_dim) {
    if (head_dim != ATT_D || n_kv_heads <= 0 || n_q_heads % n_kv_heads)
        return false;
    const int g = n_q_heads / n_kv_heads;
    return g == 1 || g == 2 || g == 4 || g == 6 || g == 7 || g == 8;
}

static vv_mtl_launch_t launch(const char* kernel, const void* params,
                              size_t params_size) {
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    l.kernel = kernel;
    l.params = params;
    l.params_size = params_size;
    l.grid[1] = l.grid[2] = 1;
    l.block[1] = l.block[2] = 1;
    return l;
}

/* ─── Backends ──────────────────────────────────────────────────────────── */

int vv_attn_resolve(int requested, int kv_format, bool paged,
                    int n_q_heads, int n_kv_heads, int head_dim) {
    /* CUDA's rules with tensor cores present: the pipeline decides paging
     * from the answer, so the answers have to agree. */
    const bool shape_ok = head_dim == ATT_D && n_kv_heads > 0 &&
                          n_q_heads % n_kv_heads == 0 &&
                          n_q_heads / n_kv_heads <= 16;
    const bool fi_ok = shape_ok;
    const bool fa2_ok = shape_ok && gqa_ok(n_q_heads, n_kv_heads, head_dim);
    const bool fa2_pages = fa2_ok && kv_format == VV_KV_FP16;
    if (paged) {
        if (requested == VV_ATTN_FLASHINFER && fi_ok) return VV_ATTN_FLASHINFER;
        if (fa2_pages) return VV_ATTN_FA2;
        return fi_ok ? VV_ATTN_FLASHINFER : VV_ATTN_FA1;
    }
    switch (requested) {
        case VV_ATTN_FLASHINFER:
            if (fi_ok) return VV_ATTN_FLASHINFER;
            return fa2_ok ? VV_ATTN_FA2 : VV_ATTN_FA1;
        case VV_ATTN_FA1:
            return VV_ATTN_FA1;
        default:
            return fa2_ok ? VV_ATTN_FA2 : VV_ATTN_FA1;
    }
}

size_t vv_attn_scratch_bytes(int n_q_heads, int n_kv_heads, int head_dim) {
    (void)n_kv_heads;
    return parts_bytes(n_q_heads, head_dim);
}

size_t vv_gqa_decode_scratch_bytes(int n_q_heads, int head_dim) {
    return parts_bytes(n_q_heads, head_dim);
}

int vv_attn_decode_shape(int backend, int n_q_heads, int n_kv_heads,
                         int cache_len) {
    (void)backend; (void)n_q_heads; (void)n_kv_heads;
    return n_parts_for(cache_len);
}

int vv_gqa_decode_shape(int cache_len) { return n_parts_for(cache_len); }

/* ─── Decode ────────────────────────────────────────────────────────────── */

typedef struct {
    int n_kv_heads, G, cache_len, use_dev_len, n_parts, bpv, use_pt;
    float scale;
} dec_p;

typedef struct { int n_parts; } comb_p;

static vv_status_t decode_run(const void* q, const vv_kv_view_t* kv,
                              void* out, int n_q_heads, int cache_len,
                              const int* d_cache_len, void* scratch,
                              void* stream) {
    if (!q || !kv || !kv->k || !kv->v || !out || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len <= 0) return VV_OK;
    if (!gqa_ok(n_q_heads, kv->n_kv_heads, kv->head_dim))
        return VV_ERR_UNSUPPORTED;
    const char* fs = fmt_suffix(kv->format);
    if (!fs) return VV_ERR_UNSUPPORTED;
    if (kv->format >= VV_KV_TQ4 && (!kv->k_meta || !kv->v_meta))
        return VV_ERR_NULL_PTR;

    const int n_kv = kv->n_kv_heads;
    const int G = n_q_heads / n_kv;
    const int n_parts = n_parts_for(cache_len);
    const size_t rows = (size_t)n_q_heads * DEC_MAX_PARTS;
    float* part_o = (float*)scratch;
    float* part_m = part_o + rows * ATT_D;
    float* part_l = part_m + rows;

    /*
     * FP16 spreads the group's heads over simdgroups that read the same
     * addresses; a quantized cache packs them into one so each vector is
     * decoded once, as soon as there are enough slices to fill the GPU.
     */
    const bool packed = G == 1 ||
        (kv->format != VV_KV_FP16 && n_kv * n_parts >= 4 * vv_mtl_core_count());
    char name[64];
    snprintf(name, sizeof(name), "vv_attn_decode_%s_h%d", fs, packed ? G : 1);

    dec_p p = { n_kv, G, cache_len, d_cache_len != NULL, n_parts,
                vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D),
                kv->page_table != NULL, 1.0f / sqrtf((float)ATT_D) };
    vv_mtl_launch_t l = launch(name, &p, sizeof(p));
    l.bufs[0] = q; l.bufs[1] = kv->k; l.bufs[2] = kv->v;
    l.bufs[3] = kv->k_meta; l.bufs[4] = kv->v_meta; l.bufs[5] = kv->page_table;
    l.bufs[6] = part_o; l.bufs[7] = part_m; l.bufs[8] = part_l;
    l.bufs[9] = d_cache_len; l.nbufs = 10;
    l.grid[0] = (uint32_t)n_kv;
    l.grid[1] = (uint32_t)(packed ? n_parts / DEC_WARPS : n_parts);
    l.block[0] = 32u * (uint32_t)(packed ? DEC_WARPS : G);
    vv_status_t s = vv_mtl_run(stream, &l);
    if (s != VV_OK) return s;

    comb_p c = { n_parts };
    vv_mtl_launch_t m = launch("vv_attn_combine", &c, sizeof(c));
    m.bufs[0] = part_o; m.bufs[1] = part_m; m.bufs[2] = part_l; m.bufs[3] = out;
    m.nbufs = 4;
    m.grid[0] = (uint32_t)n_q_heads;
    m.block[0] = ATT_D;
    return vv_mtl_run(stream, &m);
}

vv_status_t vv_attn_decode(int backend, const void* q, const vv_kv_view_t* kv,
                           void* out, int n_q_heads, int cache_len,
                           const int* d_cache_len, void* scratch,
                           void* stream) {
    (void)backend;
    return decode_run(q, kv, out, n_q_heads, cache_len, d_cache_len, scratch,
                      stream);
}

static vv_kv_view_t plain_view(const void* k, const void* v, const void* km,
                               const void* vm, int format, int n_kv_heads,
                               int head_dim) {
    vv_kv_view_t kv;
    memset(&kv, 0, sizeof(kv));
    kv.k = k; kv.v = v; kv.k_meta = km; kv.v_meta = vm;
    kv.format = format; kv.n_kv_heads = n_kv_heads; kv.head_dim = head_dim;
    return kv;
}

vv_status_t vv_gqa_attention_decode_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    const int* d_cache_len, void* scratch, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;
    const vv_kv_view_t kv = plain_view(k_cache, v_cache, NULL, NULL,
                                       VV_KV_FP16, n_kv_heads, head_dim);
    return decode_run(q, &kv, output, n_q_heads, cache_len, d_cache_len,
                      scratch, stream);
}

vv_status_t vv_gqa_attention_decode_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    const int* d_cache_len, int kv_format, void* scratch, void* stream) {
    if (!q || !k_store || !v_store || !output || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;
    const vv_kv_view_t kv = plain_view(k_store, v_store, k_meta, v_meta,
                                       kv_format, n_kv_heads, head_dim);
    return decode_run(q, &kv, output, n_q_heads, cache_len, d_cache_len,
                      scratch, stream);
}

/* ─── Prefill ───────────────────────────────────────────────────────────── */

typedef struct {
    int n_q_heads, n_kv_heads, G, q_len, q_offset, kv_len, causal, bpv, use_pt;
    float scale;
} pre_p;

static vv_status_t prefill_run(bool scalar, const void* q,
                               const vv_kv_view_t* kv, void* out,
                               int n_q_heads, int q_len, int q_offset,
                               int kv_len, bool causal, void* stream) {
    if (!q || !kv || !kv->k || !kv->v || !out) return VV_ERR_NULL_PTR;
    if (q_len <= 0 || kv_len <= 0) return VV_OK;
    if (kv->head_dim != ATT_D || kv->n_kv_heads <= 0 ||
        n_q_heads % kv->n_kv_heads)
        return VV_ERR_UNSUPPORTED;
    const char* fs = fmt_suffix(kv->format);
    if (!fs) return VV_ERR_UNSUPPORTED;
    if (kv->format >= VV_KV_TQ4 && (!kv->k_meta || !kv->v_meta))
        return VV_ERR_NULL_PTR;

    const int G = n_q_heads / kv->n_kv_heads;
    pre_p p = { n_q_heads, kv->n_kv_heads, G, q_len, q_offset, kv_len,
                causal ? 1 : 0,
                vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D),
                kv->page_table != NULL, 1.0f / sqrtf((float)ATT_D) };
    char name[64];
    snprintf(name, sizeof(name), "vv_attn_prefill_%s_%s",
             scalar ? "fa1" : "mma", fs);
    vv_mtl_launch_t l = launch(name, &p, sizeof(p));
    l.bufs[0] = q; l.bufs[1] = kv->k; l.bufs[2] = kv->v;
    l.bufs[3] = kv->k_meta; l.bufs[4] = kv->v_meta; l.bufs[5] = kv->page_table;
    l.bufs[6] = out; l.nbufs = 7;
    if (scalar) {
        l.grid[0] = (uint32_t)n_q_heads;
        l.grid[1] = (uint32_t)((q_len + 7) / 8);
        l.block[0] = 256;
    } else {
        l.grid[0] = (uint32_t)(((int64_t)q_len * G + 31) / 32);
        l.grid[1] = (uint32_t)kv->n_kv_heads;
        l.block[0] = 128;
    }
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_attn_prefill(int backend, const void* q, const vv_kv_view_t* kv,
                            void* out, int n_q_heads, int q_len, int q_offset,
                            int kv_len, bool causal, void* scratch,
                            void* stream) {
    (void)scratch;
    return prefill_run(backend == VV_ATTN_FA1, q, kv, out, n_q_heads, q_len,
                       q_offset, kv_len, causal, stream);
}

/** @brief The backend VV_ATTN / VV_ATTN_MMA ask for (the legacy entry). */
static int env_backend(void) {
    const char* e = getenv("VV_ATTN");
    const char* m = getenv("VV_ATTN_MMA");
    if (e && strcmp(e, "fa1") == 0) return VV_ATTN_FA1;
    if (e && strcmp(e, "fa2") == 0) return VV_ATTN_FA2;
    if (e && (strcmp(e, "flashinfer") == 0 || strcmp(e, "fi") == 0))
        return VV_ATTN_FLASHINFER;
    if (m && m[0] == '0') return VV_ATTN_FA1;
    return VV_ATTN_AUTO;
}

vv_status_t vv_gqa_attention_prefill_cached_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int q_len, int q_offset,
    int kv_len, bool causal, void* stream) {
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;
    const int backend = vv_attn_resolve(env_backend(), VV_KV_FP16, false,
                                        n_q_heads, n_kv_heads, head_dim);
    const vv_kv_view_t kv = plain_view(k_cache, v_cache, NULL, NULL,
                                       VV_KV_FP16, n_kv_heads, head_dim);
    return vv_attn_prefill(backend, q, &kv, output, n_q_heads, q_len,
                           q_offset, kv_len, causal, NULL, stream);
}

vv_status_t vv_gqa_attention_prefill_dev(
    const void* q, const void* k, const void* v, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int seq_len, bool causal,
    void* stream) {
    return vv_gqa_attention_prefill_cached_dev(q, k, v, output, n_q_heads,
                                               n_kv_heads, head_dim, seq_len,
                                               0, seq_len, causal, stream);
}

vv_status_t vv_gqa_attention_prefill_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int q_len, int q_offset,
    int kv_len, bool causal, int kv_format, void* stream) {
    if (!q || !k_store || !v_store || !output) return VV_ERR_NULL_PTR;
    const vv_kv_view_t kv = plain_view(k_store, v_store, k_meta, v_meta,
                                       kv_format, n_kv_heads, head_dim);
    /* The scalar kernel, as the CUDA entry point runs it. */
    return prefill_run(true, q, &kv, output, n_q_heads, q_len, q_offset,
                       kv_len, causal, stream);
}

/* ─── KV store ──────────────────────────────────────────────────────────── */

typedef struct {
    int n_kv_heads, head_dim, pos0, use_dev_pos, n_pos, bpv, use_pt, has_meta,
        has_ref, chunks;
} store_p;

typedef struct { int n_kv_heads, head_dim, n_pos; } ref_p;

vv_status_t vv_kv_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, const int* page_table, void* stream) {
    if (!k_fp16 || !v_fp16 || !k_store || !v_store) return VV_ERR_NULL_PTR;
    if (n_positions <= 0) return VV_OK;

    if (kv_format == VV_KV_FP16) {
        if (head_dim % 8) return VV_ERR_UNSUPPORTED;
        store_p p = { n_kv_heads, head_dim, pos, d_pos != NULL, n_positions,
                      0, page_table != NULL, 0, 0, head_dim / 8 };
        vv_mtl_launch_t l = launch("vv_kv_store_raw", &p, sizeof(p));
        l.bufs[0] = k_fp16; l.bufs[1] = v_fp16; l.bufs[2] = k_store;
        l.bufs[3] = v_store; l.bufs[4] = d_pos; l.bufs[5] = page_table;
        l.nbufs = 6;
        vv_mtl_grid1(&l, (uint64_t)n_positions * n_kv_heads * (head_dim / 8),
                     256);
        return vv_mtl_run(stream, &l);
    }
    if (head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    const char* fs = fmt_suffix(kv_format);
    if (!fs) return VV_ERR_UNSUPPORTED;
    if (kv_format >= VV_KV_TQ4 && (!k_meta || !v_meta)) return VV_ERR_NULL_PTR;

    vv_status_t s;
    if (k_ref && build_ref) {
        ref_p r = { n_kv_heads, head_dim, n_positions };
        vv_mtl_launch_t l = launch("vv_kv_build_ref", &r, sizeof(r));
        l.bufs[0] = k_fp16; l.bufs[1] = k_ref; l.nbufs = 2;
        l.grid[0] = (uint32_t)n_kv_heads;
        l.block[0] = (uint32_t)head_dim;
        if ((s = vv_mtl_run(stream, &l)) != VV_OK) return s;
    }

    char name[48];
    snprintf(name, sizeof(name), "vv_kv_store_%s", fs);
    const int total = n_positions * n_kv_heads;
    for (int which = 0; which < 2; which++) {
        store_p p = { n_kv_heads, head_dim, pos, d_pos != NULL, n_positions,
                      vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim),
                      page_table != NULL,
                      (which ? v_meta : k_meta) != NULL,
                      which == 0 && k_ref != NULL, 0 };
        vv_mtl_launch_t l = launch(name, &p, sizeof(p));
        l.bufs[0] = which ? v_fp16 : k_fp16;
        l.bufs[1] = which ? v_store : k_store;
        l.bufs[2] = which ? v_meta : k_meta;
        l.bufs[3] = which ? NULL : k_ref;
        l.bufs[4] = d_pos; l.bufs[5] = page_table; l.nbufs = 6;
        l.grid[0] = (uint32_t)((total + 3) / 4);
        l.block[0] = 128;
        if ((s = vv_mtl_run(stream, &l)) != VV_OK) return s;
    }
    return VV_OK;
}

vv_status_t vv_kv_quant_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, void* stream) {
    if (kv_format == VV_KV_FP16) return VV_ERR_UNSUPPORTED;
    return vv_kv_store_dev(k_fp16, v_fp16, k_store, v_store, k_meta, v_meta,
                           k_ref, build_ref, n_kv_heads, head_dim, pos, d_pos,
                           n_positions, kv_format, NULL, stream);
}

typedef struct { int n; int id[256]; } map_p;

vv_status_t vv_kv_page_map_dev(int* page_table, int first, int n,
                               const int* pages, void* stream) {
    if (!page_table || (!pages && n > 0)) return VV_ERR_NULL_PTR;
    for (int i = 0; i < n; i += 256) {
        map_p p;
        p.n = (n - i < 256) ? n - i : 256;
        memcpy(p.id, pages + i, (size_t)p.n * sizeof(int));
        vv_mtl_launch_t l = launch("vv_kv_page_map", &p,
                                   sizeof(int) * (size_t)(1 + p.n));
        l.bufs[0] = page_table + first + i; l.nbufs = 1;
        l.grid[0] = 1;
        l.block[0] = 256;
        const vv_status_t s = vv_mtl_run(stream, &l);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

typedef struct { int n_heads, head_dim, rows, inverse; } rot_p;

static vv_status_t rotate(void* x, int n_heads, int head_dim, int rows,
                          int inverse, void* stream) {
    if (!x) return VV_ERR_NULL_PTR;
    if (head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    if (rows <= 0 || n_heads <= 0) return VV_OK;
    rot_p p = { n_heads, head_dim, rows, inverse };
    vv_mtl_launch_t l = launch("vv_kv_rotate", &p, sizeof(p));
    l.bufs[0] = x; l.nbufs = 1;
    l.grid[0] = (uint32_t)((rows * n_heads + 3) / 4);
    l.block[0] = 128;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_kv_rotate_dev(void* x, int n_heads, int head_dim, int rows,
                             void* stream) {
    return rotate(x, n_heads, head_dim, rows, 0, stream);
}

vv_status_t vv_kv_unrotate_dev(void* x, int n_heads, int head_dim, int rows,
                               void* stream) {
    return rotate(x, n_heads, head_dim, rows, 1, stream);
}

typedef struct { int n_kv_heads, head_dim, n_pos, bpv, has_meta, has_ref; } deq_p;

vv_status_t vv_kv_dequant_dev(const void* store, const void* meta,
                              const void* ref, void* out_fp16, int n_kv_heads,
                              int head_dim, int n_pos, int kv_format,
                              void* stream) {
    if (!store || !out_fp16) return VV_ERR_NULL_PTR;
    if (head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    if (kv_format == VV_KV_FP16 || !fmt_suffix(kv_format))
        return VV_ERR_UNSUPPORTED;
    if (n_pos <= 0) return VV_OK;
    if (kv_format >= VV_KV_TQ4 && !meta) return VV_ERR_NULL_PTR;
    char name[48];
    snprintf(name, sizeof(name), "vv_kv_dequant_%s", fmt_suffix(kv_format));
    deq_p p = { n_kv_heads, head_dim, n_pos,
                vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim),
                meta != NULL, ref != NULL };
    vv_mtl_launch_t l = launch(name, &p, sizeof(p));
    l.bufs[0] = store; l.bufs[1] = meta; l.bufs[2] = ref; l.bufs[3] = out_fp16;
    l.nbufs = 4;
    l.grid[0] = (uint32_t)((n_pos * n_kv_heads + 3) / 4);
    l.block[0] = 128;
    return vv_mtl_run(stream, &l);
}
