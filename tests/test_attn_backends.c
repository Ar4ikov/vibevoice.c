/**
 * @file test_attn_backends.c
 * @brief Every attention backend against an FP64 reference, plus the
 *        properties the decode loop relies on: paging changes nothing, a
 *        replayed graph equals a direct launch.
 *
 * ctest runs this once per backend (VV_ATTN=fa1|fa2|flashinfer); run by hand
 * with VV_ATTN unset it walks all three. For each backend:
 *
 *  - prefill, FP16 cache: lengths on and off every 16/64 boundary, causal and
 *    not, resumed at an offset into the cache, small-q against a long cache
 *    (flashinfer's split-KV path), for GQA 28/4 (Qwen2-7B) and 12/2 (BitNet);
 *  - prefill and decode over fp8, fp8-e5m2, tq4 and tq2 caches, against the
 *    reference on the values the store actually kept (vv_kv_dequant_dev), so
 *    the kernel is judged, not the quantizer;
 *  - decode at 1, 1023, 1024, 1025, 4097, 16385 and 32767 positions — both
 *    sides of every split bucket — with the length read from device memory;
 *  - fa2: decode bit-identical to fa1's per-head kernels on every format,
 *    which is what lets `auto` pick it without moving a transcript;
 *  - backends that read pages (flashinfer; fa2 on FP16): a paged cache with
 *    a shuffled page table and a partial last page, written through the
 *    paged store, must give bit-identical prefill and decode to the
 *    contiguous one;
 *  - a decode captured into a graph at one length and replayed at the next
 *    twenty is bit-identical to launching it directly.
 *
 * `--bench` adds prefill TFLOP/s (1K/4K/8K causal) and decode effective KV
 * bandwidth (bytes of K and V the step must read, over its time).
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HEAD_DIM 128

static uint32_t rng_state = 0x9E3779B9u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}
static uint32_t irand(uint32_t n) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (rng_state >> 8) % n;
}

static int failures = 0;
static int backend = 0;
static void* g_stream = NULL;
static void* g_qtmp = NULL;       /* rotated copy of Q for TurboQuant runs */

typedef struct { int nq, nkv; const char* name; } gqa_t;
static const gqa_t GQA[2] = { {28, 4, "28/4"}, {12, 2, "12/2"} };

/* ─── Reference ──────────────────────────────────────────────────────────── */

/**
 * @brief Softmax(QKᵀ/√d)·V in FP64 for one head over rows [0, q_len).
 * K and V are FP16 [kv_len][nkv][d] as the reference sees them.
 */
static void reference_head(const uint16_t* Q, const uint16_t* K,
                           const uint16_t* V, double* out, int nq, int nkv,
                           int head, int q_len, int q_offset, int kv_len,
                           bool causal)
{
    const int kv_head = head / (nq / nkv);
    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    double* p = (double*)malloc((size_t)kv_len * sizeof(double));
    for (int r = 0; r < q_len; r++) {
        const uint16_t* q = Q + ((size_t)r * nq + head) * HEAD_DIM;
        const int q_abs = q_offset + r;
        double qf[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) qf[d] = vv_half_to_float(q[d]);
        double m = -1e300;
        for (int c = 0; c < kv_len; c++) {
            if (causal && c > q_abs) { p[c] = -1e300; continue; }
            const uint16_t* k = K + ((size_t)c * nkv + kv_head) * HEAD_DIM;
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += qf[d] * (double)vv_half_to_float(k[d]);
            p[c] = dot * scale;
            if (p[c] > m) m = p[c];
        }
        double sum = 0.0;
        for (int c = 0; c < kv_len; c++) {
            p[c] = (p[c] > -1e300) ? exp(p[c] - m) : 0.0;
            sum += p[c];
        }
        double* o = out + (size_t)r * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) o[d] = 0.0;
        for (int c = 0; c < kv_len; c++) {
            if (p[c] == 0.0) continue;
            const double w = p[c] / sum;
            const uint16_t* v = V + ((size_t)c * nkv + kv_head) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                o[d] += w * (double)vv_half_to_float(v[d]);
        }
    }
    free(p);
}

/** @brief Compare heads {0, G-1, G, nq-1} of `got` to the reference. */
static void compare(const char* what, const uint16_t* Q, const uint16_t* K,
                    const uint16_t* V, const uint16_t* got, int nq, int nkv,
                    int q_len, int q_offset, int kv_len, bool causal,
                    double tol)
{
    const int G = nq / nkv;
    const int heads[4] = { 0, G - 1, G, nq - 1 };
    double* ref = (double*)malloc((size_t)q_len * HEAD_DIM * sizeof(double));
    double worst = 0.0, min_cos = 1.0;
    for (int hi = 0; hi < 4; hi++) {
        const int h = heads[hi];
        reference_head(Q, K, V, ref, nq, nkv, h, q_len, q_offset, kv_len, causal);
        for (int r = 0; r < q_len; r++) {
            double dot = 0, na = 0, nb = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                const double a = ref[(size_t)r * HEAD_DIM + d];
                const double b = vv_half_to_float(got[((size_t)r * nq + h) * HEAD_DIM + d]);
                const double e = fabs(a - b);
                if (e > worst || e != e) worst = (e == e) ? e : 1e9;
                dot += a * b; na += a * a; nb += b * b;
            }
            const double c = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 1.0;
            if (c < min_cos) min_cos = c;
        }
    }
    free(ref);
    const bool ok = worst < tol && min_cos > 0.9999;
    printf("  %-4s %-44s max|err| %.2e  min cos %.7f\n",
           ok ? "ok" : "FAIL", what, worst, min_cos);
    if (!ok) failures++;
}

/* ─── A device cache built from host data ────────────────────────────────── */

typedef struct {
    int fmt, nkv, len, bpv;
    void *k, *v, *km, *vm, *kr;   /* store (contiguous) */
    uint16_t *hk, *hv;            /* what the reference compares against */
} cache_t;

static void fill_half(uint16_t* h, size_t n, float amp, float bias) {
    for (size_t i = 0; i < n; i++) h[i] = vv_float_to_half(frand() * amp + bias);
}

/**
 * @brief Build a cache of `len` positions. For quantized formats the host
 *        copy is replaced by what the store actually kept.
 */
static int cache_make(cache_t* c, int fmt, int nkv, int len) {
    memset(c, 0, sizeof(*c));
    c->fmt = fmt; c->nkv = nkv; c->len = len;
    c->bpv = vv_kv_bytes_per_vec((vv_kv_format_t)fmt, HEAD_DIM);
    const size_t n = (size_t)len * nkv * HEAD_DIM;
    c->hk = (uint16_t*)malloc(n * 2);
    c->hv = (uint16_t*)malloc(n * 2);
    /* Keys with a large shared component, as Qwen2's are. */
    fill_half(c->hk, n, 1.5f, 0.0f);
    for (size_t i = 0; i < n; i++)
        c->hk[i] = vv_float_to_half(vv_half_to_float(c->hk[i])
                                    + (float)((i % HEAD_DIM) % 7) * 0.5f);
    fill_half(c->hv, n, 1.0f, 0.0f);

    void *dk = NULL, *dv = NULL;
    if (vv_dev_alloc(&dk, n * 2) != VV_OK || vv_dev_alloc(&dv, n * 2) != VV_OK)
        return -1;
    vv_dev_memcpy_h2d(dk, c->hk, n * 2, NULL);
    vv_dev_memcpy_h2d(dv, c->hv, n * 2, NULL);
    if (fmt == VV_KV_FP16) { c->k = dk; c->v = dv; return 0; }

    const size_t sb = (size_t)len * nkv * c->bpv;
    if (vv_dev_alloc(&c->k, sb) != VV_OK || vv_dev_alloc(&c->v, sb) != VV_OK ||
        vv_dev_alloc(&c->kr, (size_t)nkv * HEAD_DIM * 2) != VV_OK) return -1;
    if (vv_kv_has_meta((vv_kv_format_t)fmt)) {
        if (vv_dev_alloc(&c->km, (size_t)len * nkv * 2) != VV_OK ||
            vv_dev_alloc(&c->vm, (size_t)len * nkv * 2) != VV_OK) return -1;
    }
    if (vv_kv_store_dev(dk, dv, c->k, c->v, c->km, c->vm, c->kr, true, nkv,
                        HEAD_DIM, 0, NULL, len, fmt, NULL, g_stream) != VV_OK)
        return -1;
    vv_kv_dequant_dev(c->k, c->km, c->kr, dk, nkv, HEAD_DIM, len, fmt, g_stream);
    vv_kv_dequant_dev(c->v, c->vm, NULL, dv, nkv, HEAD_DIM, len, fmt, g_stream);
    vv_dev_stream_sync(g_stream);
    vv_dev_memcpy_d2h(c->hk, dk, n * 2, NULL);
    vv_dev_memcpy_d2h(c->hv, dv, n * 2, NULL);
    vv_dev_free(dk); vv_dev_free(dv);
    return 0;
}

static void cache_free(cache_t* c) {
    if (c->k) vv_dev_free(c->k);
    if (c->v) vv_dev_free(c->v);
    if (c->km) vv_dev_free(c->km);
    if (c->vm) vv_dev_free(c->vm);
    if (c->kr) vv_dev_free(c->kr);
    free(c->hk); free(c->hv);
}

static vv_kv_view_t cache_view(const cache_t* c) {
    vv_kv_view_t v;
    memset(&v, 0, sizeof(v));
    v.k = c->k; v.v = c->v; v.k_meta = c->km; v.v_meta = c->vm;
    v.format = c->fmt; v.n_kv_heads = c->nkv; v.head_dim = HEAD_DIM;
    return v;
}

/** @brief The backend this test runs for this format, as a context would. */
static int backend_for(int fmt, bool paged, const gqa_t* g) {
    return vv_attn_resolve(backend, fmt, paged, g->nq, g->nkv, HEAD_DIM);
}

/**
 * @brief Run attention through the dispatch; rotates for TurboQuant exactly
 *        the way the decoder does. `decode` means q_len 1 at the end.
 */
static vv_status_t run_attn(int be, const vv_kv_view_t* kv, void* dQ,
                            void* dO, int nq, int q_len, int q_off,
                            int kv_len, bool causal, bool decode,
                            const int* d_len, void* scratch)
{
    const bool rot = vv_kv_rotates((vv_kv_format_t)kv->format);
    vv_status_t s = VV_OK;
    if (rot) {
        /* Rotate a copy: a rotation there and back is not the identity in
         * FP16, and the caller may run the same Q twice. */
        s = vv_dev_memcpy_d2d(g_qtmp, dQ, (size_t)q_len * nq * HEAD_DIM * 2,
                              g_stream);
        dQ = g_qtmp;
        if (s == VV_OK) s = vv_kv_rotate_dev(dQ, nq, HEAD_DIM, q_len, g_stream);
    }
    if (s != VV_OK) return s;
    if (decode)
        s = vv_attn_decode(be, dQ, kv, dO, nq, kv_len, d_len, scratch, g_stream);
    else
        s = vv_attn_prefill(be, dQ, kv, dO, nq, q_len, q_off, kv_len, causal,
                            scratch, g_stream);
    if (s == VV_OK && rot) s = vv_kv_unrotate_dev(dO, nq, HEAD_DIM, q_len, g_stream);
    return s;
}

/* ─── Prefill ────────────────────────────────────────────────────────────── */

static void check_prefill(const gqa_t* g, int fmt, int q_len, int q_off,
                          int kv_len, bool causal)
{
    cache_t c;
    if (cache_make(&c, fmt, g->nkv, kv_len) != 0) { failures++; return; }
    const int be = backend_for(fmt, false, g);
    const size_t qn = (size_t)q_len * g->nq * HEAD_DIM;
    uint16_t* hq = (uint16_t*)malloc(qn * 2);
    uint16_t* ho = (uint16_t*)malloc(qn * 2);
    fill_half(hq, qn, 1.0f, 0.0f);
    void *dQ = NULL, *dO = NULL, *scratch = NULL;
    vv_dev_alloc(&dQ, qn * 2);
    vv_dev_alloc(&dO, qn * 2);
    vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv, HEAD_DIM));
    vv_dev_memcpy_h2d(dQ, hq, qn * 2, NULL);
    vv_dev_memset(dO, 0xFF, qn * 2);                  /* NaN if unwritten */

    const vv_kv_view_t kv = cache_view(&c);
    vv_status_t s = run_attn(be, &kv, dQ, dO, g->nq, q_len, q_off, kv_len,
                             causal, false, NULL, scratch);
    vv_dev_stream_sync(g_stream);
    char what[96];
    snprintf(what, sizeof(what), "prefill %s %s q=%d @%d kv=%d %s [%s]",
             g->name, vv_kv_format_name((vv_kv_format_t)fmt), q_len, q_off,
             kv_len, causal ? "c" : "nc",
             vv_attn_backend_name((vv_attn_backend_t)be));
    if (s != VV_OK) {
        printf("  FAIL %s -> %s\n", what, vv_status_str(s));
        failures++;
    } else {
        vv_dev_memcpy_d2h(ho, dO, qn * 2, NULL);
        const double tol = (fmt == VV_KV_FP16) ? 4e-3 : 8e-3;
        compare(what, hq, c.hk, c.hv, ho, g->nq, g->nkv, q_len, q_off,
                kv_len, causal, tol);
    }
    vv_dev_free(dQ); vv_dev_free(dO); vv_dev_free(scratch);
    free(hq); free(ho);
    cache_free(&c);
}

/* ─── Decode ─────────────────────────────────────────────────────────────── */

static void check_decode(const gqa_t* g, int fmt, int len)
{
    cache_t c;
    if (cache_make(&c, fmt, g->nkv, len) != 0) { failures++; return; }
    const int be = backend_for(fmt, false, g);
    const size_t qn = (size_t)g->nq * HEAD_DIM;
    uint16_t* hq = (uint16_t*)malloc(qn * 2);
    uint16_t* ho = (uint16_t*)malloc(qn * 2);
    fill_half(hq, qn, 1.0f, 0.0f);
    void *dQ = NULL, *dO = NULL, *scratch = NULL, *dlen = NULL;
    vv_dev_alloc(&dQ, qn * 2);
    vv_dev_alloc(&dO, qn * 2);
    vv_dev_alloc(&dlen, sizeof(int));
    vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv, HEAD_DIM));
    vv_dev_memcpy_h2d(dQ, hq, qn * 2, NULL);
    vv_dev_memcpy_h2d(dlen, &len, sizeof(int), NULL);
    vv_dev_memset(dO, 0xFF, qn * 2);

    const vv_kv_view_t kv = cache_view(&c);
    vv_status_t s = run_attn(be, &kv, dQ, dO, g->nq, 1, len - 1, len, false,
                             true, (const int*)dlen, scratch);
    vv_dev_stream_sync(g_stream);
    char what[96];
    snprintf(what, sizeof(what), "decode  %s %s len=%d [%s]", g->name,
             vv_kv_format_name((vv_kv_format_t)fmt), len,
             vv_attn_backend_name((vv_attn_backend_t)be));
    if (s != VV_OK) {
        printf("  FAIL %s -> %s\n", what, vv_status_str(s));
        failures++;
    } else {
        vv_dev_memcpy_d2h(ho, dO, qn * 2, NULL);
        const double tol = (fmt == VV_KV_FP16) ? 4e-3 : 8e-3;
        compare(what, hq, c.hk, c.hv, ho, g->nq, g->nkv, 1, len - 1, len,
                false, tol);
    }
    vv_dev_free(dQ); vv_dev_free(dO); vv_dev_free(scratch); vv_dev_free(dlen);
    free(hq); free(ho);
    cache_free(&c);
}

/* ─── fa2 decode == fa1 decode, bit for bit ──────────────────────────────── */

static void check_decode_exact(const gqa_t* g, int fmt, int len)
{
    cache_t c;
    if (cache_make(&c, fmt, g->nkv, len) != 0) { failures++; return; }
    const size_t qn = (size_t)g->nq * HEAD_DIM;
    uint16_t* hq = (uint16_t*)malloc(qn * 2);
    uint16_t* h1 = (uint16_t*)malloc(qn * 2);
    uint16_t* h2 = (uint16_t*)malloc(qn * 2);
    fill_half(hq, qn, 1.0f, 0.0f);
    void *dQ = NULL, *dO = NULL, *scratch = NULL, *dlen = NULL;
    vv_dev_alloc(&dQ, qn * 2);
    vv_dev_alloc(&dO, qn * 2);
    vv_dev_alloc(&dlen, sizeof(int));
    vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv, HEAD_DIM));
    vv_dev_memcpy_h2d(dQ, hq, qn * 2, NULL);
    vv_dev_memcpy_h2d(dlen, &len, sizeof(int), NULL);

    const vv_kv_view_t kv = cache_view(&c);
    vv_status_t a = run_attn(VV_ATTN_FA1, &kv, dQ, dO, g->nq, 1, len - 1, len,
                             false, true, (const int*)dlen, scratch);
    vv_dev_stream_sync(g_stream);
    vv_dev_memcpy_d2h(h1, dO, qn * 2, NULL);
    vv_status_t b = run_attn(VV_ATTN_FA2, &kv, dQ, dO, g->nq, 1, len - 1, len,
                             false, true, (const int*)dlen, scratch);
    vv_dev_stream_sync(g_stream);
    vv_dev_memcpy_d2h(h2, dO, qn * 2, NULL);
    const bool bad = a != VV_OK || b != VV_OK || memcmp(h1, h2, qn * 2) != 0;
    printf("  %-4s decode fa2 == fa1 bit for bit %s %s len=%d\n",
           bad ? "FAIL" : "ok", g->name,
           vv_kv_format_name((vv_kv_format_t)fmt), len);
    if (bad) failures++;
    vv_dev_free(dQ); vv_dev_free(dO); vv_dev_free(scratch); vv_dev_free(dlen);
    free(hq); free(h1); free(h2);
    cache_free(&c);
}

/* ─── Paged == contiguous ────────────────────────────────────────────────── */

/**
 * @brief Write the same K/V through the store into a slab and into shuffled
 *        pages of a pool (chunked prefill, then single-position decode
 *        appends at a device position), and demand identical attention.
 */
static void check_paged(const gqa_t* g, int fmt, int len)
{
    const int be = backend;
    const int nkv = g->nkv, nq = g->nq;
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)fmt, HEAD_DIM);
    const bool meta = vv_kv_has_meta((vv_kv_format_t)fmt);
    const int pages = (len + VV_KV_PAGE_SIZE - 1) / VV_KV_PAGE_SIZE;
    const int pool_pages = pages + 5;
    const size_t n = (size_t)len * nkv * HEAD_DIM;
    uint16_t* hk = (uint16_t*)malloc(n * 2);
    uint16_t* hv = (uint16_t*)malloc(n * 2);
    fill_half(hk, n, 2.0f, 0.5f);
    fill_half(hv, n, 1.0f, 0.0f);

    void *dk = NULL, *dv = NULL;
    vv_dev_alloc(&dk, n * 2); vv_dev_alloc(&dv, n * 2);
    vv_dev_memcpy_h2d(dk, hk, n * 2, NULL);
    vv_dev_memcpy_h2d(dv, hv, n * 2, NULL);

    /* Slab and pool, both zeroed so nothing but the store differs. */
    const size_t slab_b = (size_t)len * nkv * bpv;
    const size_t pool_b = (size_t)pool_pages * VV_KV_PAGE_SIZE * nkv * bpv;
    const size_t slab_m = (size_t)len * nkv * 2;
    const size_t pool_m = (size_t)pool_pages * VV_KV_PAGE_SIZE * nkv * 2;
    void *sk = NULL, *sv = NULL, *pk = NULL, *pv = NULL;
    void *skm = NULL, *svm = NULL, *pkm = NULL, *pvm = NULL;
    void *skr = NULL, *pkr = NULL, *table = NULL, *dpos = NULL;
    vv_dev_alloc(&sk, slab_b); vv_dev_alloc(&sv, slab_b);
    vv_dev_alloc(&pk, pool_b); vv_dev_alloc(&pv, pool_b);
    vv_dev_memset(sk, 0, slab_b); vv_dev_memset(sv, 0, slab_b);
    vv_dev_memset(pk, 0, pool_b); vv_dev_memset(pv, 0, pool_b);
    if (meta) {
        vv_dev_alloc(&skm, slab_m); vv_dev_alloc(&svm, slab_m);
        vv_dev_alloc(&pkm, pool_m); vv_dev_alloc(&pvm, pool_m);
    }
    if (fmt != VV_KV_FP16) {
        vv_dev_alloc(&skr, (size_t)nkv * HEAD_DIM * 2);
        vv_dev_alloc(&pkr, (size_t)nkv * HEAD_DIM * 2);
    }
    vv_dev_alloc(&table, (size_t)pages * sizeof(int));
    vv_dev_alloc(&dpos, sizeof(int));

    /* A shuffled page table: logical page i lives at pool page perm[i]. */
    int* perm = (int*)malloc((size_t)pool_pages * sizeof(int));
    for (int i = 0; i < pool_pages; i++) perm[i] = i;
    for (int i = pool_pages - 1; i > 0; i--) {
        const int j = (int)irand((uint32_t)i + 1);
        const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    vv_kv_page_map_dev((int*)table, 0, pages, perm, g_stream);

    /* Prefill in uneven chunks, then the last positions one at a time at a
     * device-side position, as decode writes them. */
    const int tail = len > 8 ? 5 : 0;
    const int body = len - tail;
    const size_t row = (size_t)nkv * HEAD_DIM;
    vv_status_t s = VV_OK;
    for (int p = 0; p < body && s == VV_OK; ) {
        int m = 97 + (int)irand(200);
        if (p + m > body) m = body - p;
        const bool first = (p == 0);
        s = vv_kv_store_dev((uint16_t*)dk + p * row, (uint16_t*)dv + p * row,
                            sk, sv, skm, svm, skr, first, nkv, HEAD_DIM, p,
                            NULL, m, fmt, NULL, g_stream);
        if (s == VV_OK)
            s = vv_kv_store_dev((uint16_t*)dk + p * row,
                                (uint16_t*)dv + p * row, pk, pv, pkm, pvm,
                                pkr, first, nkv, HEAD_DIM, p, NULL, m, fmt,
                                (const int*)table, g_stream);
        p += m;
    }
    for (int p = body; p < len && s == VV_OK; p++) {
        vv_dev_memcpy_h2d(dpos, &p, sizeof(int), g_stream);
        s = vv_kv_store_dev((uint16_t*)dk + p * row, (uint16_t*)dv + p * row,
                            sk, sv, skm, svm, skr, false, nkv, HEAD_DIM, 0,
                            (const int*)dpos, 1, fmt, NULL, g_stream);
        if (s == VV_OK)
            s = vv_kv_store_dev((uint16_t*)dk + p * row,
                                (uint16_t*)dv + p * row, pk, pv, pkm, pvm,
                                pkr, false, nkv, HEAD_DIM, 0,
                                (const int*)dpos, 1, fmt, (const int*)table,
                                g_stream);
        vv_dev_stream_sync(g_stream);
    }

    vv_kv_view_t slab, paged;
    memset(&slab, 0, sizeof(slab));
    slab.k = sk; slab.v = sv; slab.k_meta = skm; slab.v_meta = svm;
    slab.format = fmt; slab.n_kv_heads = nkv; slab.head_dim = HEAD_DIM;
    paged = slab;
    paged.k = pk; paged.v = pv; paged.k_meta = pkm; paged.v_meta = pvm;
    paged.page_table = (const int*)table;

    /* Prefill rows at the end (split-KV path) and a full prefill, then a
     * decode; each through both, compared bit for bit. */
    const int q_cases[3][2] = { { 3, len - 3 }, { 40, len - 40 }, { len, 0 } };
    void *dQ = NULL, *o1 = NULL, *o2 = NULL, *scratch = NULL, *dl = NULL;
    const size_t qn = (size_t)len * nq * HEAD_DIM;
    uint16_t* hq = (uint16_t*)malloc(qn * 2);
    uint16_t* h1 = (uint16_t*)malloc(qn * 2);
    uint16_t* h2 = (uint16_t*)malloc(qn * 2);
    fill_half(hq, qn, 1.0f, 0.0f);
    vv_dev_alloc(&dQ, qn * 2); vv_dev_alloc(&o1, qn * 2); vv_dev_alloc(&o2, qn * 2);
    vv_dev_alloc(&scratch, vv_attn_scratch_bytes(nq, nkv, HEAD_DIM));
    vv_dev_alloc(&dl, sizeof(int));
    vv_dev_memcpy_h2d(dQ, hq, qn * 2, NULL);
    vv_dev_memcpy_h2d(dl, &len, sizeof(int), NULL);

    int bad = (s != VV_OK);
    for (int ci = 0; ci < 4 && !bad; ci++) {
        const bool dec = (ci == 3);
        const int ql = dec ? 1 : q_cases[ci][0];
        const int qo = dec ? len - 1 : q_cases[ci][1];
        if (ql <= 0 || qo < 0) continue;
        vv_dev_memset(o1, 0, qn * 2);
        vv_dev_memset(o2, 0, qn * 2);
        vv_status_t a = run_attn(be, &slab, dQ, o1, nq, ql, qo,
                                 len, true, dec, (const int*)dl, scratch);
        vv_status_t b = run_attn(be, &paged, dQ, o2, nq, ql,
                                 qo, len, true, dec, (const int*)dl, scratch);
        vv_dev_stream_sync(g_stream);
        const size_t on = (size_t)ql * nq * HEAD_DIM;
        vv_dev_memcpy_d2h(h1, o1, on * 2, NULL);
        vv_dev_memcpy_d2h(h2, o2, on * 2, NULL);
        if (a != VV_OK || b != VV_OK || memcmp(h1, h2, on * 2) != 0) bad = 1;
    }
    printf("  %-4s paged == contiguous %s %s len=%d (%d pages, shuffled) [%s]\n",
           bad ? "FAIL" : "ok", g->name,
           vv_kv_format_name((vv_kv_format_t)fmt), len, pages,
           vv_attn_backend_name((vv_attn_backend_t)be));
    if (bad) failures++;

    vv_dev_free(dQ); vv_dev_free(o1); vv_dev_free(o2); vv_dev_free(scratch);
    vv_dev_free(dl); vv_dev_free(dk); vv_dev_free(dv);
    vv_dev_free(sk); vv_dev_free(sv); vv_dev_free(pk); vv_dev_free(pv);
    if (meta) { vv_dev_free(skm); vv_dev_free(svm); vv_dev_free(pkm); vv_dev_free(pvm); }
    if (skr) vv_dev_free(skr);
    if (pkr) vv_dev_free(pkr);
    vv_dev_free(table); vv_dev_free(dpos);
    free(perm); free(hk); free(hv); free(hq); free(h1); free(h2);
}

/* ─── Graph replay == direct launch ──────────────────────────────────────── */

static void check_graph(const gqa_t* g, int fmt)
{
    const int L = 1025, steps = 20, cap = L + steps;
    cache_t c;
    if (cache_make(&c, fmt, g->nkv, cap) != 0) { failures++; return; }
    const int be = backend_for(fmt, false, g);
    const size_t qn = (size_t)g->nq * HEAD_DIM;
    uint16_t* hq = (uint16_t*)malloc(qn * 2);
    uint16_t* h1 = (uint16_t*)malloc(qn * 2 * steps);
    uint16_t* h2 = (uint16_t*)malloc(qn * 2 * steps);
    fill_half(hq, qn, 1.0f, 0.0f);
    void *dQ = NULL, *dO = NULL, *scratch = NULL, *dl = NULL;
    vv_dev_alloc(&dQ, qn * 2); vv_dev_alloc(&dO, qn * 2);
    vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv, HEAD_DIM));
    vv_dev_alloc(&dl, sizeof(int));
    vv_dev_memcpy_h2d(dQ, hq, qn * 2, NULL);
    const vv_kv_view_t kv = cache_view(&c);

    int bad = 0;
    for (int i = 0; i < steps && !bad; i++) {
        const int len = L + i;
        vv_dev_memcpy_h2d(dl, &len, sizeof(int), g_stream);
        if (run_attn(be, &kv, dQ, dO, g->nq, 1, len - 1, L, false, true,
                     (const int*)dl, scratch) != VV_OK) bad = 1;
        vv_dev_stream_sync(g_stream);
        vv_dev_memcpy_d2h(h1 + qn * i, dO, qn * 2, NULL);
    }

    void* exec = NULL;
    if (!bad && vv_dev_graph_begin(g_stream) == VV_OK) {
        const vv_status_t s = run_attn(be, &kv, dQ, dO, g->nq, 1, L - 1, L,
                                       false, true, (const int*)dl, scratch);
        if (vv_dev_graph_end(g_stream, &exec) != VV_OK || s != VV_OK) bad = 1;
    } else {
        bad = 1;
    }
    for (int i = 0; i < steps && !bad; i++) {
        const int len = L + i;
        vv_dev_memcpy_h2d(dl, &len, sizeof(int), g_stream);
        if (vv_dev_graph_launch(exec, g_stream) != VV_OK) bad = 1;
        vv_dev_stream_sync(g_stream);
        vv_dev_memcpy_d2h(h2 + qn * i, dO, qn * 2, NULL);
    }
    if (exec) vv_dev_graph_destroy(exec);
    if (!bad && memcmp(h1, h2, qn * 2 * steps) != 0) bad = 1;
    printf("  %-4s graph replay == direct %s %s len %d..%d [%s]\n",
           bad ? "FAIL" : "ok", g->name,
           vv_kv_format_name((vv_kv_format_t)fmt), L, L + steps - 1,
           vv_attn_backend_name((vv_attn_backend_t)be));
    if (bad) failures++;
    vv_dev_free(dQ); vv_dev_free(dO); vv_dev_free(scratch); vv_dev_free(dl);
    free(hq); free(h1); free(h2);
    cache_free(&c);
}

/* ─── Throughput ─────────────────────────────────────────────────────────── */

static void bench_prefill(int seq) {
    const gqa_t* g = &GQA[0];
    const int be = backend_for(VV_KV_FP16, false, g);
    const size_t qn = (size_t)seq * g->nq * HEAD_DIM;
    const size_t kn = (size_t)seq * g->nkv * HEAD_DIM;
    void *dQ = NULL, *dK = NULL, *dV = NULL, *dO = NULL, *scratch = NULL;
    if (vv_dev_alloc(&dQ, qn * 2) != VV_OK || vv_dev_alloc(&dK, kn * 2) != VV_OK ||
        vv_dev_alloc(&dV, kn * 2) != VV_OK || vv_dev_alloc(&dO, qn * 2) != VV_OK ||
        vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv, HEAD_DIM)) != VV_OK)
        return;
    vv_dev_memset(dQ, 0x11, qn * 2);
    vv_dev_memset(dK, 0x11, kn * 2);
    vv_dev_memset(dV, 0x11, kn * 2);
    vv_kv_view_t kv;
    memset(&kv, 0, sizeof(kv));
    kv.k = dK; kv.v = dV; kv.format = VV_KV_FP16; kv.n_kv_heads = g->nkv;
    kv.head_dim = HEAD_DIM;
    for (int i = 0; i < 2; i++)
        vv_attn_prefill(be, dQ, &kv, dO, g->nq, seq, 0, seq, true, scratch, g_stream);
    vv_dev_stream_sync(g_stream);
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        const int iters = 5;
        const double t0 = vv_time_ms();
        for (int i = 0; i < iters; i++)
            vv_attn_prefill(be, dQ, &kv, dO, g->nq, seq, 0, seq, true, scratch,
                            g_stream);
        vv_dev_stream_sync(g_stream);
        const double ms = (vv_time_ms() - t0) / iters;
        if (ms < best) best = ms;
    }
    const double flop = 2.0 * 2.0 * 0.5 * (double)seq * seq * g->nq * HEAD_DIM;
    printf("  prefill %5d: %8.3f ms  %6.1f TFLOP/s [%s]\n", seq, best,
           flop / (best * 1e-3) / 1e12,
           vv_attn_backend_name((vv_attn_backend_t)be));
    vv_dev_free(dQ); vv_dev_free(dK); vv_dev_free(dV); vv_dev_free(dO);
    vv_dev_free(scratch);
}

/*
 * Decode reads the whole cache once per call, so its figure is K+V bytes over
 * time -- but only if those bytes come from DRAM. One 1K fp16 cache is 2 MB
 * and a 4K one 8 MB against the 3090's 6 MB L2; timed back to back on the
 * same buffers they are L2 hits. So the calls rotate over enough copies to
 * pass 64 MB, each filled with random data (a constant pattern is also what
 * a compressing memory system likes best).
 */
#define BENCH_ROTATE_BYTES ((size_t)64 << 20)
#define BENCH_MAX_COPIES   32

static void fill_random(void* dev, size_t bytes, int fmt, bool is_meta) {
    uint8_t* h = (uint8_t*)malloc(bytes);
    if (!h) return;
    if (fmt == VV_KV_FP16 || is_meta) {
        /* Halves: values for K/V, small positive scales for metadata. */
        uint16_t* hv = (uint16_t*)h;
        for (size_t i = 0; i < bytes / 2; i++)
            hv[i] = vv_float_to_half(is_meta ? 0.01f + 0.5f * (frand() + 1.0f)
                                             : frand());
    } else {
        /* Codes: keep fp8 clear of its NaN/Inf encodings. */
        for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t)(irand(256) & 0xBB);
    }
    vv_dev_memcpy_h2d(dev, h, bytes, g_stream);
    vv_dev_stream_sync(g_stream);
    free(h);
}

static void bench_decode(int fmt, int len) {
    const gqa_t* g = &GQA[0];
    const int be = backend_for(fmt, false, g);
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)fmt, HEAD_DIM);
    const bool meta = vv_kv_has_meta((vv_kv_format_t)fmt);
    const size_t sb = (size_t)len * g->nkv * bpv;
    const size_t mb = meta ? (size_t)len * g->nkv * 2 : 0;
    const size_t per_copy = 2 * (sb + mb);
    int n_copies = (int)((BENCH_ROTATE_BYTES + per_copy - 1) / per_copy);
    if (n_copies < 2) n_copies = 2;
    if (n_copies > BENCH_MAX_COPIES) n_copies = BENCH_MAX_COPIES;

    void *dQ = NULL, *dO = NULL, *scratch = NULL, *dl = NULL;
    void *K[BENCH_MAX_COPIES] = {0}, *V[BENCH_MAX_COPIES] = {0};
    void *KM[BENCH_MAX_COPIES] = {0}, *VM[BENCH_MAX_COPIES] = {0};
    vv_kv_view_t kv[BENCH_MAX_COPIES];
    bool ok = vv_dev_alloc(&dQ, (size_t)g->nq * HEAD_DIM * 2) == VV_OK &&
              vv_dev_alloc(&dO, (size_t)g->nq * HEAD_DIM * 2) == VV_OK &&
              vv_dev_alloc(&scratch, vv_attn_scratch_bytes(g->nq, g->nkv,
                                                           HEAD_DIM)) == VV_OK &&
              vv_dev_alloc(&dl, sizeof(int)) == VV_OK;
    for (int c = 0; ok && c < n_copies; c++) {
        ok = vv_dev_alloc(&K[c], sb) == VV_OK && vv_dev_alloc(&V[c], sb) == VV_OK;
        if (ok && meta)
            ok = vv_dev_alloc(&KM[c], mb) == VV_OK &&
                 vv_dev_alloc(&VM[c], mb) == VV_OK;
        if (!ok) break;
        fill_random(K[c], sb, fmt, false);
        fill_random(V[c], sb, fmt, false);
        if (meta) { fill_random(KM[c], mb, fmt, true); fill_random(VM[c], mb, fmt, true); }
        memset(&kv[c], 0, sizeof(kv[c]));
        kv[c].k = K[c]; kv[c].v = V[c]; kv[c].k_meta = KM[c]; kv[c].v_meta = VM[c];
        kv[c].format = fmt; kv[c].n_kv_heads = g->nkv; kv[c].head_dim = HEAD_DIM;
    }
    if (ok) {
        fill_random(dQ, (size_t)g->nq * HEAD_DIM * 2, VV_KV_FP16, false);
        vv_dev_memcpy_h2d(dl, &len, sizeof(int), g_stream);
        for (int i = 0; i < n_copies; i++)
            vv_attn_decode(be, dQ, &kv[i], dO, g->nq, len, (const int*)dl,
                           scratch, g_stream);
        vv_dev_stream_sync(g_stream);
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            const int iters = n_copies * ((50 + n_copies - 1) / n_copies);
            const double t0 = vv_time_ms();
            for (int i = 0; i < iters; i++)
                vv_attn_decode(be, dQ, &kv[i % n_copies], dO, g->nq, len,
                               (const int*)dl, scratch, g_stream);
            vv_dev_stream_sync(g_stream);
            const double ms = (vv_time_ms() - t0) / iters;
            if (ms < best) best = ms;
        }
        const double bytes = (double)per_copy;
        printf("  decode %-8s %6d: %7.1f us  %6.1f GB/s of KV from DRAM "
               "(%d caches, %.0f MB) [%s]\n",
               vv_kv_format_name((vv_kv_format_t)fmt), len, best * 1e3,
               bytes / (best * 1e-3) / 1e9, n_copies,
               (double)per_copy * n_copies / (1024.0 * 1024.0),
               vv_attn_backend_name((vv_attn_backend_t)be));
    } else {
        printf("  decode %-8s %6d: skipped, allocation failed\n",
               vv_kv_format_name((vv_kv_format_t)fmt), len);
    }
    for (int c = 0; c < n_copies; c++) {
        if (K[c]) vv_dev_free(K[c]);
        if (V[c]) vv_dev_free(V[c]);
        if (KM[c]) vv_dev_free(KM[c]);
        if (VM[c]) vv_dev_free(VM[c]);
    }
    if (dQ) vv_dev_free(dQ);
    if (dO) vv_dev_free(dO);
    if (scratch) vv_dev_free(scratch);
    if (dl) vv_dev_free(dl);
}

/* ─── Driver ─────────────────────────────────────────────────────────────── */

static void run_backend(bool bench) {
    printf("\n=== backend %s ===\n", vv_attn_backend_name((vv_attn_backend_t)backend));
    static const int fmts[5] = { VV_KV_FP16, VV_KV_FP8_E4M3, VV_KV_FP8_E5M2,
                                 VV_KV_TQ4, VV_KV_TQ2 };
    for (int gi = 0; gi < 2; gi++) {
        const gqa_t* g = &GQA[gi];
        /* FP16 prefill: tile edges, causal and not, resumed offsets. */
        const int shapes[][4] = {
            {1, 0, 1, 1}, {16, 0, 16, 1}, {17, 0, 17, 1}, {64, 0, 64, 1},
            {65, 0, 65, 1}, {100, 0, 100, 1}, {257, 0, 257, 1},
            {64, 0, 64, 0}, {100, 0, 257, 0},
            {64, 64, 128, 1}, {37, 100, 137, 1}, {128, 63, 191, 1},
            {1, 2999, 3000, 1}, {7, 1500, 1507, 1}, {28, 3000, 3028, 1},
            {300, 1000, 1300, 1},
        };
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
            check_prefill(g, VV_KV_FP16, shapes[i][0], shapes[i][1],
                          shapes[i][2], shapes[i][3] != 0);
        for (int f = 1; f < 5; f++) {
            check_prefill(g, fmts[f], 100, 0, 100, true);
            check_prefill(g, fmts[f], 28, 1000, 1028, true);
        }
        /* Decode on both sides of every split bucket. */
        static const int lens[7] = { 1, 1023, 1024, 1025, 4097, 16385, 32767 };
        for (int i = 0; i < 7; i++) check_decode(g, VV_KV_FP16, lens[i]);
        for (int f = 1; f < 5; f++) {
            check_decode(g, fmts[f], 1025);
            check_decode(g, fmts[f], 4097);
        }
        check_graph(g, VV_KV_FP16);
        check_graph(g, VV_KV_TQ4);
        if (backend == VV_ATTN_FA2 &&
            backend_for(VV_KV_FP16, false, g) == VV_ATTN_FA2) {
            for (int i = 0; i < 7; i++)
                check_decode_exact(g, VV_KV_FP16, lens[i]);
            /* Short caches take the spread layout, long ones the packed
             * one on quantized formats: both must be exact. */
            for (int f = 1; f < 5; f++) {
                check_decode_exact(g, fmts[f], 1025);
                check_decode_exact(g, fmts[f], 4097);
                check_decode_exact(g, fmts[f], 32767);
            }
        }
        for (int f = 0; f < 5; f++)
            if (backend_for(fmts[f], true, g) == backend)
                check_paged(g, fmts[f], 1000);
        if (backend_for(VV_KV_FP16, true, g) == backend) {
            check_paged(g, VV_KV_FP16, 64);
            check_paged(g, VV_KV_FP16, 4097);
        }
    }
    if (bench) {
        printf("\n--- throughput ---\n");
        bench_prefill(1024);
        bench_prefill(4096);
        bench_prefill(8192);
        const int dl[6] = { 256, 1024, 4096, 8192, 16384, 32768 };
        for (int i = 0; i < 6; i++) bench_decode(VV_KV_FP16, dl[i]);
        bench_decode(VV_KV_FP8_E4M3, 16384);
        bench_decode(VV_KV_TQ4, 16384);
    }
}

int main(int argc, char** argv) {
#ifndef VV_HAS_ACCEL
    (void)argc; (void)argv;
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(0, &total, &freem, NULL) != VV_OK || total == 0) {
        printf("SKIP: no device available\n");
        return 0;
    }
    bool bench = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--bench") == 0) bench = true;
    if (vv_dev_alloc(&g_qtmp, (size_t)64 << 20) != VV_OK ||
        vv_dev_stream_create(&g_stream) != VV_OK) {
        printf("SKIP: cannot create a stream\n");
        return 0;
    }

    const vv_attn_backend_t want = vv_attn_backend_from_env(VV_ATTN_AUTO);
    if (want == VV_ATTN_AUTO) {
        for (int b = VV_ATTN_FA1; b <= VV_ATTN_FLASHINFER; b++) {
            backend = b;
            run_backend(bench);
        }
    } else {
        backend = (int)want;
        run_backend(bench);
    }
    vv_dev_stream_destroy(g_stream);

    if (failures) {
        printf("\n=== %d check(s) failed ===\n", failures);
        return 1;
    }
    printf("\n=== every backend matches the reference ===\n");
    return 0;
#endif
}
