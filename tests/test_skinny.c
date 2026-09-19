/**
 * @file test_skinny.c
 * @brief Linear layers at 9..64 rows on the GPU: the small-M tensor-core
 *        kernels (vv_skinny_linear_dev) against the dequant + tile GEMM path
 *        they replace, byte for byte, and both against the CPU kernels and
 *        an FP64 reference.
 *
 * For dense FP16, per-channel INT8 and NF4 weights:
 *   - every M where the tiling changes (9, 15, 16, 17, 29, 31, 32, 33, 47,
 *     48, 49, 63, 64) on a k/v-sized and a ragged projection (N not a
 *     multiple of the block, K of one stage), a few on the 7B and 1.5B
 *     shapes;
 *   - the output bytes equal dequant -> vv_gemm_fp16_tile_dev ->
 *     vv_bias_add_dev, with and without bias, and so do those of the public
 *     entry points (vv_gemm_fp16_dev, vv_int8_gemm_dev, vv_nf4_gemm_dev),
 *     whichever kernel they end up in;
 *   - q/k/v and gate/up in one launch give the bytes of separate launches;
 *   - every output against the CPU kernel, sampled ones against FP64 over
 *     the FP16 weight the GPU multiplies;
 *   - what it does not take (M outside 9..64, K off its tile, a misaligned
 *     input) is declined without a launch, and the entry points then run
 *     the tile path.
 * Every kernel instance is reached: each format at 1..4 row tiles.
 *
 * VV_SKINNY_BENCH=1 adds timings (2: timings only): each 7B projection at
 * M = 9..64 through the small-M kernel and through the path it replaces,
 * and q/k/v, gate/up in one launch against separate ones. 3: only the four
 * launches a decoder layer makes, small-M kernel only (for tile sweeps).
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/cpu_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern const float VV_NF4_TABLE[16];

static uint32_t rng_state = 0x2545F491u;
static uint32_t urand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float frand(void) {                       /* [-1, 1) */
    return ((float)(urand() >> 8) / 8388608.0f) - 1.0f;
}

static int failures = 0;
static void* g_stream = NULL;
static bool g_declined = false;      /* the device has no small-M kernels */

static const char* FMT[3] = { "fp16", "int8", "nf4" };

/** @brief One weight in one format: host copy, device copy, and the FP16
 *         values the GPU multiplies, as float. */
typedef struct {
    int fmt, N, K;
    uint16_t* w16;        /* F16  [N][K]                                  */
    int8_t*   q8;         /* INT8 [N][K]                                  */
    float*    s8;         /* INT8 [N]                                     */
    uint8_t*  p4;         /* NF4  [N][K/2], high nibble first             */
    uint16_t* a4;         /* NF4  absmax [N*K/64]                         */
    uint16_t* bias;       /* [N]                                          */
    float*    wref;       /* [N][K]                                       */
    void *dw, *ds, *dbias;
} weight_t;

static void weight_free(weight_t* w) {
    free(w->w16); free(w->q8); free(w->s8); free(w->p4); free(w->a4);
    free(w->bias); free(w->wref);
    vv_dev_free(w->dw); vv_dev_free(w->ds); vv_dev_free(w->dbias);
    memset(w, 0, sizeof(*w));
}

/**
 * @brief A random weight. Some rows and blocks are scaled down far enough
 *        that the dequantized FP16 values are subnormal, where a rounding
 *        difference would show first.
 */
static int weight_make(weight_t* w, int fmt, int N, int K) {
    memset(w, 0, sizeof(*w));
    w->fmt = fmt; w->N = N; w->K = K;
    const size_t nk = (size_t)N * K;
    w->wref = (float*)malloc(nk * sizeof(float));
    w->bias = (uint16_t*)malloc((size_t)N * 2);
    for (int n = 0; n < N; n++)
        w->bias[n] = vv_float_to_half_rne(frand() * 0.5f);
    size_t wbytes = 0, sbytes = 0;
    const void* hs = NULL;
    const void* hw = NULL;
    if (fmt == VV_SKINNY_F16) {
        w->w16 = (uint16_t*)malloc(nk * 2);
        for (size_t i = 0; i < nk; i++) {
            const float v = frand() * ((i % 97) == 5 ? 1e-5f : 0.05f);
            w->w16[i] = vv_float_to_half_rne(v);
            w->wref[i] = vv_half_to_float(w->w16[i]);
        }
        hw = w->w16; wbytes = nk * 2;
    } else if (fmt == VV_SKINNY_INT8) {
        w->q8 = (int8_t*)malloc(nk);
        w->s8 = (float*)malloc((size_t)N * sizeof(float));
        for (int n = 0; n < N; n++)
            w->s8[n] = (n % 13 == 7) ? 3e-8f
                     : 0.0002f + 0.001f * (float)(urand() & 1023) / 1024.0f;
        for (size_t i = 0; i < nk; i++) {
            w->q8[i] = (int8_t)((int)(urand() % 255u) - 127);
            const float v = (float)w->q8[i] * w->s8[i / K];
            w->wref[i] = vv_half_to_float(vv_float_to_half_rne(v));
        }
        hw = w->q8; wbytes = nk;
        hs = w->s8; sbytes = (size_t)N * sizeof(float);
    } else {
        const size_t nb = nk / 64;
        w->p4 = (uint8_t*)malloc(nk / 2);
        w->a4 = (uint16_t*)malloc(nb * 2);
        for (size_t i = 0; i < nk / 2; i++) w->p4[i] = (uint8_t)urand();
        for (size_t b = 0; b < nb; b++)
            w->a4[b] = vv_float_to_half_rne(
                (b % 29 == 3) ? 2e-5f
                              : 0.005f + 0.05f * (float)(urand() & 1023) / 1024.0f);
        for (size_t i = 0; i < nk; i++) {
            const uint8_t byte = w->p4[i / 2];
            const int code = (i & 1) ? (byte & 15) : (byte >> 4);
            const float v = VV_NF4_TABLE[code] * vv_half_to_float(w->a4[i / 64]);
            w->wref[i] = vv_half_to_float(vv_float_to_half_rne(v));
        }
        hw = w->p4; wbytes = nk / 2;
        hs = w->a4; sbytes = nb * 2;
    }
    if (vv_dev_alloc(&w->dw, wbytes) != VV_OK ||
        vv_dev_alloc(&w->dbias, (size_t)N * 2) != VV_OK ||
        (sbytes && vv_dev_alloc(&w->ds, sbytes) != VV_OK)) {
        printf("  FAIL allocation for %s %dx%d\n", FMT[fmt], N, K);
        failures++;
        return 1;
    }
    vv_dev_memcpy_h2d(w->dw, hw, wbytes, NULL);
    vv_dev_memcpy_h2d(w->dbias, w->bias, (size_t)N * 2, NULL);
    if (sbytes) vv_dev_memcpy_h2d(w->ds, hs, sbytes, NULL);
    return 0;
}

/** @brief Random FP16 activations [M][K] with a few outliers, and as float. */
static void make_x(int M, int K, uint16_t** hx, float** fx) {
    *hx = (uint16_t*)malloc((size_t)M * K * 2);
    *fx = (float*)malloc((size_t)M * K * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) {
        float v = frand();
        if ((urand() & 255) == 0) v *= 24.0f;
        (*hx)[i] = vv_float_to_half_rne(v);
        (*fx)[i] = vv_half_to_float((*hx)[i]);
    }
}

/**
 * @brief The path the small-M kernels replace: FP16 copy of the weight,
 *        the 128x128 tile GEMM, then the bias as its own kernel.
 */
static vv_status_t run_old(const weight_t* w, const void* dx, void* dy,
                           void* tmp, int M, bool bias) {
    const void* wf = w->dw;
    vv_status_t s = VV_OK;
    if (w->fmt == VV_SKINNY_INT8)
        s = vv_dequant_int8_dev((const int8_t*)w->dw, (const float*)w->ds, tmp,
                                w->N, w->K, g_stream);
    else if (w->fmt == VV_SKINNY_NF4)
        s = vv_dequant_nf4_dev((const uint8_t*)w->dw, w->ds, tmp,
                               w->N * w->K, 64, g_stream);
    if (w->fmt != VV_SKINNY_F16) wf = tmp;
    if (s == VV_OK)
        s = vv_gemm_fp16_tile_dev(dx, wf, dy, M, w->N, w->K, 1.0f, 0.0f,
                                  g_stream);
    if (s == VV_OK && bias)
        s = vv_bias_add_dev(dy, w->dbias, M, w->N, g_stream);
    return s;
}

/** @brief The public entry point of the format, then the bias. */
static vv_status_t run_public(const weight_t* w, const void* dx, void* dy,
                              void* tmp, int M, bool bias) {
    vv_status_t s;
    if (w->fmt == VV_SKINNY_F16)
        s = vv_gemm_fp16_dev(dx, w->dw, dy, M, w->N, w->K, 1.0f, 0.0f,
                             g_stream);
    else if (w->fmt == VV_SKINNY_INT8)
        s = vv_int8_gemm_dev(dx, (const int8_t*)w->dw, (const float*)w->ds,
                             dy, tmp, M, w->N, w->K, g_stream);
    else
        s = vv_nf4_gemm_dev(dx, (const uint8_t*)w->dw, w->ds, dy, tmp,
                            M, w->N, w->K, 64, g_stream);
    if (s == VV_OK && bias)
        s = vv_bias_add_dev(dy, w->dbias, M, w->N, g_stream);
    return s;
}

static vv_status_t run_new(const weight_t* w, const void* dx, void* dy,
                           int M, bool bias) {
    const vv_skinny_proj_t p = { w->dw, w->ds, bias ? w->dbias : NULL, dy,
                                 w->N };
    return vv_skinny_linear_dev(dx, w->fmt, &p, 1, M, w->K, 1.0f, g_stream);
}

static size_t count_diff(const uint16_t* a, const uint16_t* b, size_t n,
                         size_t* first) {
    size_t d = 0;
    *first = (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) { if (!d) *first = i; d++; }
    return d;
}

/**
 * @brief GPU outputs hy [M][N] against the CPU kernel (all) and FP64 over
 *        the FP16 weight (sampled).
 *
 * The FP64 bound is two FP16 roundings of the output (the sum, then the sum
 * plus bias) and a twentieth of sqrt(sum (w x)^2), far more than FP32
 * accumulation of K such products can drift. The rows whose weights are
 * all tiny are the ones where only the first term is left.
 */
static bool verify_ref(const weight_t* w, const float* fx, const uint16_t* hy,
                       int M, bool bias, double* worst_cpu_out,
                       double* worst_ref_out) {
    const int N = w->N, K = w->K;
    float* cref = (float*)malloc((size_t)M * N * sizeof(float));
    const void* b = bias ? w->bias : NULL;
    vv_status_t s;
    if (w->fmt == VV_SKINNY_F16)
        s = vv_gemm_f16w_cpu(fx, w->w16, b, cref, M, N, K);
    else if (w->fmt == VV_SKINNY_INT8)
        s = vv_int8_gemm_cpu(fx, w->q8, w->s8, b, cref, M, N, K);
    else
        s = vv_nf4_gemm_cpu(fx, w->p4, w->a4, b, cref, M, N, K);
    double worst_cpu = s == VV_OK ? 0.0 : 1e9, num = 0, da = 0, db = 0;
    for (size_t i = 0; i < (size_t)M * N && s == VV_OK; i++) {
        const double g = vv_half_to_float(hy[i]), r = cref[i];
        num += g * r; da += g * g; db += r * r;
        const double e = fabs(g - r) / (fabs(r) + 1.0);
        if (e > worst_cpu || e != e) worst_cpu = (e != e) ? 1e9 : e;
    }
    const double cosv = num / (sqrt(da) * sqrt(db) + 1e-300);
    double worst = 0;
    for (int i = 0; i < 256; i++) {
        const int m = (int)(urand() % (uint32_t)M);
        const int n = (int)(urand() % (uint32_t)N);
        double acc = 0.0, sq = 0.0;
        const float* wr = w->wref + (size_t)n * K;
        const float* xr = fx + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            const double p = (double)wr[k] * xr[k];
            acc += p; sq += p * p;
        }
        const double b = bias ? vv_half_to_float(w->bias[n]) : 0.0;
        const double sum = acc;
        acc += b;
        const double g = vv_half_to_float(hy[(size_t)m * N + n]);
        const double bound = (fabs(sum) + fabs(acc)) / 2048.0
                           + 0.05 * sqrt(sq) + 1e-6;
        const double e = fabs(g - acc) / bound;
        if (e > worst || e != e) worst = (e != e) ? 1e9 : e;
    }
    free(cref);
    *worst_cpu_out = worst_cpu;
    *worst_ref_out = worst;
    return worst <= 1.0 && worst_cpu < 0.05 && (da == 0 || cosv > 0.99999);
}

/**
 * @brief One weight at one M: small-M kernel, public entry point and the
 *        old path, with and without bias. `may_decline`: a shape the small-M
 *        kernel is allowed to turn down (the bytes must match either way).
 */
static void check_m(const weight_t* w, int M, void* tmp, bool may_decline) {
    const int N = w->N, K = w->K;
    uint16_t* hx; float* fx;
    make_x(M, K, &hx, &fx);
    const size_t ny = (size_t)M * N;
    uint16_t* h_old = (uint16_t*)malloc(ny * 2);
    uint16_t* h_new = (uint16_t*)malloc(ny * 2);
    uint16_t* h_pub = (uint16_t*)malloc(ny * 2);
    void *dx = NULL, *d_old = NULL, *d_new = NULL, *d_pub = NULL;
    vv_dev_alloc(&dx, (size_t)M * K * 2);
    vv_dev_alloc(&d_old, ny * 2);
    vv_dev_alloc(&d_new, ny * 2);
    vv_dev_alloc(&d_pub, ny * 2);
    vv_dev_memcpy_h2d(dx, hx, (size_t)M * K * 2, NULL);

    for (int bias = 0; bias < 2; bias++) {
        vv_dev_memset_async(d_new, 0xFF, ny * 2, g_stream);  /* NaN until */
        vv_dev_memset_async(d_pub, 0xFF, ny * 2, g_stream);  /* written   */
        vv_status_t so = run_old(w, dx, d_old, tmp, M, bias);
        vv_status_t sn = run_new(w, dx, d_new, M, bias);
        vv_status_t sp = run_public(w, dx, d_pub, tmp, M, bias);
        vv_dev_stream_sync(g_stream);
        char tag[96];
        snprintf(tag, sizeof(tag), "%-4s %5dx%-5d M=%-2d %s", FMT[w->fmt], N, K,
                 M, bias ? "bias" : "    ");
        if (so != VV_OK || sp != VV_OK ||
            (sn != VV_OK && sn != VV_ERR_UNSUPPORTED)) {
            printf("  FAIL %s: old %d, small-M %d, entry point %d\n", tag, so,
                   sn, sp);
            failures++;
            continue;
        }
        if (sn == VV_ERR_UNSUPPORTED && !may_decline) g_declined = true;
        vv_dev_memcpy_d2h(h_old, d_old, ny * 2, NULL);
        vv_dev_memcpy_d2h(h_new, d_new, ny * 2, NULL);
        vv_dev_memcpy_d2h(h_pub, d_pub, ny * 2, NULL);
        size_t f1 = 0, f2 = 0;
        const size_t dn = sn == VV_OK ? count_diff(h_new, h_old, ny, &f1) : 0;
        const size_t dp = count_diff(h_pub, h_old, ny, &f2);
        double wc = 0, wr = 0;
        const bool ref_ok =
            verify_ref(w, fx, sn == VV_OK ? h_new : h_old, M, bias, &wc, &wr);
        const bool ok = dn == 0 && dp == 0 && ref_ok;
        printf("  %s %s  %s%s  cpu %.1e  fp64 %.3f of bound\n",
               ok ? "ok  " : "FAIL", tag,
               sn == VV_OK ? "bytes = tile path" : "declined",
               dp ? ", entry point differs" : "", wc, wr);
        if (dn)
            printf("       %zu of %zu outputs differ from the tile path, first "
                   "[%zu][%zu]: %04x vs %04x\n", dn, ny, f1 / N, f1 % N,
                   h_new[f1], h_old[f1]);
        if (dp)
            printf("       entry point: %zu differ, first [%zu][%zu]\n", dp,
                   f2 / N, f2 % N);
        if (!ok) failures++;
    }
    vv_dev_free(dx); vv_dev_free(d_old); vv_dev_free(d_new); vv_dev_free(d_pub);
    free(hx); free(fx); free(h_old); free(h_new); free(h_pub);
}

static void run_shape(int fmt, int N, int K, const int* ms, int nm,
                      void* tmp, bool may_decline) {
    weight_t w;
    if (weight_make(&w, fmt, N, K)) { weight_free(&w); return; }
    for (int i = 0; i < nm; i++) check_m(&w, ms[i], tmp, may_decline);
    weight_free(&w);
}

/** @brief Projections of one x in one launch against one launch each. */
static void check_multi(int fmt, const int* Ns, int n, int K, int M) {
    weight_t w[3];
    memset(w, 0, sizeof(w));
    bool made = true;
    for (int i = 0; i < n && made; i++)
        made = weight_make(&w[i], fmt, Ns[i], K) == 0;
    if (made) {
        uint16_t* hx; float* fx;
        make_x(M, K, &hx, &fx);
        void* dx = NULL;
        vv_dev_alloc(&dx, (size_t)M * K * 2);
        vv_dev_memcpy_h2d(dx, hx, (size_t)M * K * 2, NULL);
        void* ys[3] = { NULL, NULL, NULL };
        void* yf[3] = { NULL, NULL, NULL };
        vv_skinny_proj_t p[3];
        vv_status_t s = VV_OK;
        for (int i = 0; i < n; i++) {
            vv_dev_alloc(&ys[i], (size_t)M * Ns[i] * 2);
            vv_dev_alloc(&yf[i], (size_t)M * Ns[i] * 2);
            vv_dev_memset_async(yf[i], 0xFF, (size_t)M * Ns[i] * 2,
                                g_stream);
            if (s == VV_OK) s = run_new(&w[i], dx, ys[i], M, true);
            const vv_skinny_proj_t q = { w[i].dw, w[i].ds, w[i].dbias, yf[i],
                                         Ns[i] };
            p[i] = q;
        }
        const vv_status_t sf = s == VV_OK
            ? vv_skinny_linear_dev(dx, fmt, p, n, M, K, 1.0f, g_stream) : s;
        vv_dev_stream_sync(g_stream);
        if (s == VV_ERR_UNSUPPORTED && sf == VV_ERR_UNSUPPORTED) {
            printf("  skip fused %s x%d: declined\n", FMT[fmt], n);
        } else if (s != VV_OK || sf != VV_OK) {
            printf("  FAIL fused %s x%d M=%d: separate %d, fused %d\n",
                   FMT[fmt], n, M, s, sf);
            failures++;
        } else {
            size_t bad = 0, first = 0;
            for (int i = 0; i < n; i++) {
                const size_t ny = (size_t)M * Ns[i];
                uint16_t* a = (uint16_t*)malloc(ny * 2);
                uint16_t* b = (uint16_t*)malloc(ny * 2);
                vv_dev_memcpy_d2h(a, ys[i], ny * 2, NULL);
                vv_dev_memcpy_d2h(b, yf[i], ny * 2, NULL);
                bad += count_diff(a, b, ny, &first);
                free(a); free(b);
            }
            printf("  %s fused %-4s %s M=%-2d  %s\n", bad ? "FAIL" : "ok  ",
                   FMT[fmt], n == 3 ? "q/k/v  " : "gate/up", M,
                   bad ? "outputs differ from separate launches"
                       : "bytes = separate launches");
            if (bad) failures++;
        }
        for (int i = 0; i < n; i++) { vv_dev_free(ys[i]); vv_dev_free(yf[i]); }
        vv_dev_free(dx);
        free(hx); free(fx);
    }
    for (int i = 0; i < n; i++) weight_free(&w[i]);
}

/**
 * @brief Shapes it must decline without launching, and what the public
 *        entry points do with them instead.
 */
static void check_declines(void* tmp) {
    weight_t w;
    if (weight_make(&w, VV_SKINNY_F16, 512, 3584)) { weight_free(&w); return; }
    void *dx = NULL, *dy = NULL;
    vv_dev_alloc(&dx, (size_t)65 * 3584 * 2 + 16);
    vv_dev_alloc(&dy, (size_t)65 * 512 * 2);
    struct { int M; int off; const char* what; } cases[] = {
        { 8, 0, "M = 8" }, { 65, 0, "M = 65" }, { 29, 2, "input not 16-byte aligned" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const vv_skinny_proj_t p = { w.dw, NULL, NULL, dy, 512 };
        const vv_status_t s = vv_skinny_linear_dev(
            (const uint8_t*)dx + cases[i].off, VV_SKINNY_F16, &p, 1,
            cases[i].M, 3584, 1.0f, g_stream);
        const bool ok = s == VV_ERR_UNSUPPORTED;
        printf("  %s declines %s\n", ok ? "ok  " : "FAIL", cases[i].what);
        if (!ok) failures++;
    }
    vv_dev_free(dx); vv_dev_free(dy);
    weight_free(&w);

    /* K a multiple of 64 but not of 128: INT8 and NF4 tiles decline it,
     * their entry points go through the FP16 copy -- same bytes. */
    const int ms[] = { 29 };
    run_shape(VV_SKINNY_INT8, 512, 3648, ms, 1, tmp, true);
    run_shape(VV_SKINNY_NF4, 512, 3648, ms, 1, tmp, true);
    run_shape(VV_SKINNY_F16, 512, 3656, ms, 1, tmp, true);
}

/* ─── Benchmark ──────────────────────────────────────────────────────────── */

typedef vv_status_t (*launch_fn)(void* ctx, int rep);

/** @brief Time `iters` launches captured into one graph; ms per launch. */
static double time_graph(launch_fn fn, void* ctx, int reps, int iters) {
    for (int i = 0; i < reps; i++)
        if (fn(ctx, i) != VV_OK) return -1;
    vv_dev_stream_sync(g_stream);
    void* exec = NULL;
    if (vv_dev_graph_begin(g_stream) != VV_OK) return -1;
    for (int i = 0; i < iters; i++) fn(ctx, i % reps);
    if (vv_dev_graph_end(g_stream, &exec) != VV_OK || !exec) return -1;
    const double warm0 = vv_time_ms();
    while (vv_time_ms() - warm0 < 150.0) {
        vv_dev_graph_launch(exec, g_stream);
        vv_dev_stream_sync(g_stream);
    }
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        const double t0 = vv_time_ms();
        vv_dev_graph_launch(exec, g_stream);
        vv_dev_stream_sync(g_stream);
        const double t = (vv_time_ms() - t0) / iters;
        if (t < best) best = t;
    }
    vv_dev_graph_destroy(exec);
    return best;
}

typedef struct {
    int fmt, M, K, n, reps, mode;      /* mode 0 small-M, 1 old path */
    int N[3];
    void** w[3];
    void** s[3];
    void *x, *y[3], *tmp;
    int fused;
} bench_t;

static vv_status_t bench_launch(void* p, int rep) {
    bench_t* c = (bench_t*)p;
    if (c->mode == 0 && c->fused) {
        vv_skinny_proj_t pr[3];
        for (int i = 0; i < c->n; i++) {
            const vv_skinny_proj_t q = { c->w[i][rep], c->s[i] ? c->s[i][rep] : NULL,
                                         NULL, c->y[i], c->N[i] };
            pr[i] = q;
        }
        return vv_skinny_linear_dev(c->x, c->fmt, pr, c->n, c->M, c->K, 1.0f,
                                    g_stream);
    }
    for (int i = 0; i < c->n; i++) {
        weight_t w;
        memset(&w, 0, sizeof(w));
        w.fmt = c->fmt; w.N = c->N[i]; w.K = c->K;
        w.dw = c->w[i][rep];
        w.ds = c->s[i] ? c->s[i][rep] : NULL;
        const vv_status_t s = c->mode == 0
            ? run_new(&w, c->x, c->y[i], c->M, false)
            : run_old(&w, c->x, c->y[i], c->tmp, c->M, false);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

static void bench_set(int fmt, const char* name, const int* Ns, int n, int K,
                      const int* ms, int nm, void* tmp, bool new_only) {
    bench_t c;
    memset(&c, 0, sizeof(c));
    c.fmt = fmt; c.K = K; c.n = n; c.tmp = tmp;
    double bytes = 0;
    size_t wb[3], sb[3];
    for (int i = 0; i < n; i++) {
        c.N[i] = Ns[i];
        const size_t nk = (size_t)Ns[i] * K;
        wb[i] = fmt == VV_SKINNY_F16 ? nk * 2 : fmt == VV_SKINNY_INT8 ? nk : nk / 2;
        sb[i] = fmt == VV_SKINNY_F16 ? 0
              : fmt == VV_SKINNY_INT8 ? (size_t)Ns[i] * 4 : nk / 64 * 2;
        bytes += (double)(wb[i] + sb[i]);
    }
    /* enough copies that the set does not stay in L2 */
    int reps = (int)((256u << 20) / bytes) + 1;
    if (reps > 32) reps = 32;
    c.reps = reps;
    for (int i = 0; i < n; i++) {
        c.w[i] = (void**)calloc(reps, sizeof(void*));
        c.s[i] = sb[i] ? (void**)calloc(reps, sizeof(void*)) : NULL;
        for (int r = 0; r < reps; r++) {
            vv_dev_alloc(&c.w[i][r], wb[i]);
            vv_dev_memset(c.w[i][r], 0x11, wb[i]);
            if (sb[i]) {
                vv_dev_alloc(&c.s[i][r], sb[i]);
                vv_dev_memset(c.s[i][r], 0x11, sb[i]);
            }
        }
        vv_dev_alloc(&c.y[i], (size_t)64 * Ns[i] * 2);
    }
    vv_dev_alloc(&c.x, (size_t)64 * K * 2);
    vv_dev_memset(c.x, 0x11, (size_t)64 * K * 2);

    for (int j = 0; j < nm; j++) {
        c.M = ms[j];
        const int iters = 40;
        c.mode = 0; c.fused = n > 1;
        const double tf = n > 1 ? time_graph(bench_launch, &c, reps, iters) : -1;
        if (new_only) {
            const double t = n > 1 ? tf : time_graph(bench_launch, &c, reps,
                                                     iters);
            printf("  %-4s %-8s M=%-2d  %7.1f us %6.1f GB/s\n", FMT[fmt], name,
                   c.M, t * 1e3, bytes / (t * 1e-3) / 1e9);
            continue;
        }
        c.fused = 0;
        const double tn = time_graph(bench_launch, &c, reps, iters);
        c.mode = 1;
        const double to = time_graph(bench_launch, &c, reps, iters);
        if (n > 1)
            printf("  %-4s %-8s M=%-2d  fused %7.1f us %6.1f GB/s   separate "
                   "%7.1f us   old %8.1f us   x%.2f\n", FMT[fmt], name, c.M,
                   tf * 1e3, bytes / (tf * 1e-3) / 1e9, tn * 1e3, to * 1e3,
                   to / tf);
        else
            printf("  %-4s %-8s M=%-2d  new %7.1f us %6.1f GB/s   old %8.1f us"
                   "   x%.2f\n", FMT[fmt], name, c.M, tn * 1e3,
                   bytes / (tn * 1e-3) / 1e9, to * 1e3, to / tn);
    }
    for (int i = 0; i < n; i++) {
        for (int r = 0; r < reps; r++) {
            vv_dev_free(c.w[i][r]);
            if (c.s[i]) vv_dev_free(c.s[i][r]);
        }
        free(c.w[i]); free(c.s[i]);
        vv_dev_free(c.y[i]);
    }
    vv_dev_free(c.x);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
#ifndef VV_HAS_ACCEL
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(0, &total, &freem, NULL) != VV_OK || total == 0) {
        printf("SKIP: no device available\n");
        return 0;
    }
    vv_dev_set_device(0);
    if (vv_dev_stream_create(&g_stream) != VV_OK) {
        printf("FAIL: stream\n");
        return 1;
    }
    /* The decoder's dequant scratch for a 7B layer, and the old path's here. */
    void* tmp = NULL;
    if (vv_dev_alloc(&tmp, (size_t)18944 * 3584 * 2) != VV_OK) {
        printf("FAIL: scratch\n");
        return 1;
    }
    printf("=== small-M linear, 9..64 rows ===\n");

    const char* b = getenv("VV_SKINNY_BENCH");
    const bool bench_only = b && (b[0] == '2' || b[0] == '3');
    const bool bench = b && (b[0] == '1' || b[0] == '2');
    const bool bench_layer = b && b[0] == '3';

    if (!bench_only) {
        const int ms_all[] = { 9, 15, 16, 17, 29, 31, 32, 33, 47, 48, 49, 63, 64 };
        const int n_all = (int)(sizeof(ms_all) / sizeof(ms_all[0]));
        const int ms_few[] = { 9, 29, 64 };
        const int n_few = 3;
        for (int f = 0; f < 3; f++) {
            /* every tiling on a k/v-sized and a ragged projection */
            run_shape(f, 512, 3584, ms_all, n_all, tmp, false);
            run_shape(f, 520, 256, ms_all, n_all, tmp, false);
            /* the 7B and 1.5B projections */
            run_shape(f, 3584, 3584, ms_few, n_few, tmp, false);
            run_shape(f, 18944, 3584, ms_few, n_few, tmp, false);
            run_shape(f, 3584, 18944, ms_few, n_few, tmp, false);
            run_shape(f, 8960, 1536, ms_few, n_few, tmp, false);
            run_shape(f, 1536, 8960, ms_few, n_few, tmp, false);
            /* one launch for q/k/v and for gate/up */
            const int qkv[3] = { 3584, 512, 512 }, gu[2] = { 18944, 18944 };
            check_multi(f, qkv, 3, 3584, 29);
            check_multi(f, gu, 2, 3584, 29);
        }
        check_declines(tmp);
        if (g_declined)
            printf("  (small-M kernels not available here: sm < 80 or "
                   "VV_SKINNY=0; the entry points were checked alone)\n");
    }

    if (bench) {
        printf("\n--- by projection and M, graph-replayed, weights cycled "
               "past L2 ---\n");
        const int ms[] = { 9, 16, 29, 32, 48, 64 };
        const int nm = 6;
        for (int f = 0; f < 3; f++) {
            const int qo[1] = { 3584 }, kv[1] = { 512 }, gate[1] = { 18944 };
            const int qkv[3] = { 3584, 512, 512 }, gu[2] = { 18944, 18944 };
            bench_set(f, "q/o", qo, 1, 3584, ms, nm, tmp, false);
            bench_set(f, "k/v", kv, 1, 3584, ms, nm, tmp, false);
            bench_set(f, "gate/up", gate, 1, 3584, ms, nm, tmp, false);
            bench_set(f, "down", qo, 1, 18944, ms, nm, tmp, false);
            bench_set(f, "q+k+v", qkv, 3, 3584, ms, nm, tmp, false);
            bench_set(f, "gate+up", gu, 2, 3584, ms, nm, tmp, false);
        }
    }
    if (bench_layer) {
        printf("\n--- one 7B layer's launches, small-M kernel only ---\n");
        const int ms[] = { 9, 16, 29, 32, 48, 64 };
        const int qo[1] = { 3584 };
        const int qkv[3] = { 3584, 512, 512 }, gu[2] = { 18944, 18944 };
        for (int f = 0; f < 3; f++) {
            bench_set(f, "q+k+v", qkv, 3, 3584, ms, 6, tmp, true);
            bench_set(f, "o", qo, 1, 3584, ms, 6, tmp, true);
            bench_set(f, "gate+up", gu, 2, 3584, ms, 6, tmp, true);
            bench_set(f, "down", qo, 1, 18944, ms, 6, tmp, true);
        }
    }

    vv_dev_free(tmp);
    vv_dev_stream_destroy(g_stream);
    if (failures) {
        printf("\n=== %d check(s) failed ===\n", failures);
        return 1;
    }
    printf("\n=== small-M linear layers match the tile path byte for byte ===\n");
    return 0;
#endif
}
