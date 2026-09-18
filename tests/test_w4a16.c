/**
 * @file test_w4a16.c
 * @brief INT4 group-affine (AWQ / GPTQ) linear layers on the GPU against the
 *        CPU kernel and an FP64 reference, plus a GEMV / GEMM benchmark.
 *
 * What is checked:
 *   - the AWQ and GPTQ repacks produce the same codes, scales and exact
 *     integer zeros from the same logical weight, and act-order GPTQ is
 *     refused rather than loaded wrong;
 *   - vv_w4a16_gemv_dev / vv_w4a16_gemm_dev at the 7B and 1.5B projection
 *     shapes, M in {1,2,7,8,9,16,28,64,285,2048}, group 32/64/128, symmetric
 *     and asymmetric zeros, against vv_int4g_gemm_cpu (all outputs) and an
 *     FP64 reference of the exact dequant (q - z) * s (sampled outputs);
 *   - the dequant fallback (VV_W4A16_MMA=0 runs the whole suite through it).
 *
 * VV_W4A16_BENCH=1 adds timings (2: timings only): GEMV GB/s per projection shape and GEMM
 * TFLOP/s by M, for the new kernels and the row-major ones they replace.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/quant.h"
#include "vibevoice/cpu_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rng_state = 0x9E3779B9u;
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

static const int AWQ_ORDER[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };

/** @brief One logical INT4 weight and everything derived from it. */
typedef struct {
    int N, K, G, ng;
    uint8_t*  q;          /* [N][K] codes 0..15                          */
    uint8_t*  z;          /* [N][ng] zero points                          */
    uint16_t* s;          /* [N][ng] FP16 scales                          */
    /* row-major runtime form (vv_awq_repack) */
    uint8_t*  packed;
    uint16_t* scales;
    uint16_t* mins;
    uint8_t*  zeros;
    /* GPU form */
    uint8_t*  gpacked;
    uint16_t* sz;
    float*    wref;       /* [N][K] exact FP16 dequant as float           */
} qweight_t;

static void qweight_free(qweight_t* w) {
    free(w->q); free(w->z); free(w->s);
    free(w->packed); free(w->scales); free(w->mins); free(w->zeros);
    free(w->gpacked); free(w->sz); free(w->wref);
    memset(w, 0, sizeof(*w));
}

/**
 * @brief Random weight, packed the AutoAWQ way, repacked by the runtime.
 *
 * Also packs it the GPTQ way and checks vv_gptq_repack lands on the same
 * bytes, which is the GPTQ layout test.
 */
static int make_weight(qweight_t* w, int N, int K, int G, bool sym) {
    memset(w, 0, sizeof(*w));
    w->N = N; w->K = K; w->G = G; w->ng = K / G;
    const int ng = w->ng;
    w->q = (uint8_t*)malloc((size_t)N * K);
    w->z = (uint8_t*)malloc((size_t)N * ng);
    w->s = (uint16_t*)malloc((size_t)N * ng * 2);
    for (size_t i = 0; i < (size_t)N * K; i++) w->q[i] = (uint8_t)(urand() & 15);
    for (size_t i = 0; i < (size_t)N * ng; i++) {
        /* GPTQ v1 stores z - 1, so keep z >= 1 to stay representable */
        w->z[i] = sym ? 8 : (uint8_t)(1 + urand() % 15);
        w->s[i] = vv_float_to_half(0.002f + 0.02f * (float)(urand() & 1023) / 1024.0f);
    }

    /* AutoAWQ tensors */
    uint32_t* qw = (uint32_t*)calloc((size_t)K * (N / 8), 4);
    uint32_t* qz = (uint32_t*)calloc((size_t)ng * (N / 8), 4);
    uint16_t* sc = (uint16_t*)malloc((size_t)ng * N * 2);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            qw[(size_t)k * (N / 8) + n / 8] |=
                (uint32_t)w->q[(size_t)n * K + k] << (4 * AWQ_ORDER[n & 7]);
    for (int g = 0; g < ng; g++)
        for (int n = 0; n < N; n++) {
            qz[(size_t)g * (N / 8) + n / 8] |=
                (uint32_t)w->z[(size_t)n * ng + g] << (4 * AWQ_ORDER[n & 7]);
            sc[(size_t)g * N + n] = w->s[(size_t)n * ng + g];
        }

    w->packed = (uint8_t*)malloc((size_t)N * K / 2);
    w->scales = (uint16_t*)malloc((size_t)N * ng * 2);
    w->mins   = (uint16_t*)malloc((size_t)N * ng * 2);
    w->zeros  = (uint8_t*)malloc((size_t)N * ng);
    vv_status_t s = vv_awq_repack(qw, qz, sc, K, N, G, 0, w->packed,
                                  w->scales, w->mins, w->zeros);
    int bad = (s != VV_OK);
    if (!bad && (memcmp(w->zeros, w->z, (size_t)N * ng) != 0 ||
                 memcmp(w->scales, w->s, (size_t)N * ng * 2) != 0)) {
        printf("  FAIL awq repack: zeros/scales differ\n");
        bad = 1;
    }

    /* GPTQ tensors of the same weight: qweight [K/8][N], qzeros z - 1 */
    uint32_t* gw = (uint32_t*)calloc((size_t)(K / 8) * N, 4);
    uint32_t* gz = (uint32_t*)calloc((size_t)ng * (N / 8), 4);
    int32_t* gidx = (int32_t*)malloc((size_t)K * 4);
    for (int k = 0; k < K; k++) {
        gidx[k] = k / G;
        for (int n = 0; n < N; n++)
            gw[(size_t)(k / 8) * N + n] |=
                (uint32_t)w->q[(size_t)n * K + k] << (4 * (k & 7));
    }
    for (int g = 0; g < ng; g++)
        for (int n = 0; n < N; n++)
            gz[(size_t)g * (N / 8) + n / 8] |=
                (uint32_t)(w->z[(size_t)n * ng + g] - 1) << (4 * (n & 7));
    uint8_t*  p2 = (uint8_t*)malloc((size_t)N * K / 2);
    uint16_t* s2 = (uint16_t*)malloc((size_t)N * ng * 2);
    uint16_t* m2 = (uint16_t*)malloc((size_t)N * ng * 2);
    uint8_t*  z2 = (uint8_t*)malloc((size_t)N * ng);
    s = vv_gptq_repack(gw, gz, sc, gidx, K, N, G, 1, p2, s2, m2, z2);
    if (s != VV_OK || memcmp(p2, w->packed, (size_t)N * K / 2) ||
        memcmp(s2, w->scales, (size_t)N * ng * 2) ||
        memcmp(m2, w->mins, (size_t)N * ng * 2) ||
        memcmp(z2, w->zeros, (size_t)N * ng)) {
        printf("  FAIL gptq repack differs from awq repack (N=%d K=%d G=%d)\n",
               N, K, G);
        bad = 1;
    }
    /* act-order: swap two channels of different groups -> refused */
    if (K / G >= 2) {
        gidx[0] = 1; gidx[G] = 0;
        s = vv_gptq_repack(gw, gz, sc, gidx, K, N, G, 1, p2, s2, m2, z2);
        if (s != VV_ERR_UNSUPPORTED) {
            printf("  FAIL gptq act-order g_idx was accepted (%d)\n", s);
            bad = 1;
        }
    }
    free(gw); free(gz); free(gidx); free(p2); free(s2); free(m2); free(z2);
    free(qw); free(qz); free(sc);

    /* GPU layout */
    w->gpacked = (uint8_t*)malloc((size_t)N * K / 2);
    w->sz = (uint16_t*)malloc((size_t)N * ng * 4);
    memcpy(w->gpacked, w->packed, (size_t)N * K / 2);
    s = vv_int4g_to_gpu_layout(w->gpacked, w->scales, w->zeros, N, K, G,
                               w->sz);
    if (s != VV_OK) { printf("  FAIL gpu layout: %d\n", s); bad = 1; }

    /* Exact dequant, rounded to FP16 as the reference does */
    w->wref = (float*)malloc((size_t)N * K * sizeof(float));
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            const int g = k / G;
            const float v = (float)((int)w->q[(size_t)n * K + k] -
                                    (int)w->z[(size_t)n * ng + g]) *
                            vv_half_to_float(w->s[(size_t)n * ng + g]);
            w->wref[(size_t)n * K + k] = vv_half_to_float(vv_float_to_half(v));
        }
    if (bad) failures++;
    return bad;
}

/** @brief Device copies of one weight in both layouts. */
typedef struct {
    void *gp, *gsz, *rp, *rs, *rm, *bias;
} dweight_t;

static int upload_weight(const qweight_t* w, const uint16_t* bias,
                         dweight_t* d) {
    memset(d, 0, sizeof(*d));
    const size_t pb = (size_t)w->N * w->K / 2, gb = (size_t)w->N * w->ng * 2;
    if (vv_dev_alloc(&d->gp, pb) || vv_dev_alloc(&d->gsz, gb * 2) ||
        vv_dev_alloc(&d->rp, pb) || vv_dev_alloc(&d->rs, gb) ||
        vv_dev_alloc(&d->rm, gb) || vv_dev_alloc(&d->bias, (size_t)w->N * 2))
        return 1;
    vv_dev_memcpy_h2d(d->gp, w->gpacked, pb, NULL);
    vv_dev_memcpy_h2d(d->gsz, w->sz, gb * 2, NULL);
    vv_dev_memcpy_h2d(d->rp, w->packed, pb, NULL);
    vv_dev_memcpy_h2d(d->rs, w->scales, gb, NULL);
    vv_dev_memcpy_h2d(d->rm, w->mins, gb, NULL);
    vv_dev_memcpy_h2d(d->bias, bias, (size_t)w->N * 2, NULL);
    return 0;
}

static void free_dweight(dweight_t* d) {
    vv_dev_free(d->gp); vv_dev_free(d->gsz); vv_dev_free(d->rp);
    vv_dev_free(d->rs); vv_dev_free(d->rm); vv_dev_free(d->bias);
}

static void* g_scratch = NULL;
static size_t g_scratch_bytes = 0;
static void* g_stream = NULL;

/**
 * @brief Run one M through the GPU path and check it.
 */
static void check_m(const qweight_t* w, const dweight_t* d,
                    const uint16_t* bias, int M, const char* tag)
{
    const int N = w->N, K = w->K;
    uint16_t* hx = (uint16_t*)malloc((size_t)M * K * 2);
    float*    fx = (float*)malloc((size_t)M * K * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) {
        float v = frand();
        if ((urand() & 255) == 0) v *= 24.0f;      /* a few outliers */
        hx[i] = vv_float_to_half(v);
        fx[i] = vv_half_to_float(hx[i]);
    }
    uint16_t* hy = (uint16_t*)malloc((size_t)M * N * 2);
    float* cref = (float*)malloc((size_t)M * N * sizeof(float));

    void *dx = NULL, *dy = NULL;
    vv_dev_alloc(&dx, (size_t)M * K * 2);
    vv_dev_alloc(&dy, (size_t)M * N * 2);
    vv_dev_memcpy_h2d(dx, hx, (size_t)M * K * 2, NULL);

    vv_status_t s = (M == 1)
        ? vv_w4a16_gemv_dev(dx, d->gp, d->gsz, d->bias, dy, N, K, w->G,
                            g_stream)
        : vv_w4a16_gemm_dev(dx, d->gp, d->gsz, d->bias, dy, g_scratch,
                            g_scratch_bytes, M, N, K, w->G, g_stream);
    vv_dev_stream_sync(g_stream);
    if (s != VV_OK) {
        printf("  FAIL %s M=%d: kernel returned %d\n", tag, M, s);
        failures++;
        goto done;
    }
    vv_dev_memcpy_d2h(hy, dy, (size_t)M * N * 2, NULL);

    {
        /* All outputs against the CPU kernel (FP16 mins, FP32 math). */
        vv_int4g_gemm_cpu(fx, w->packed, w->scales, w->mins, bias, cref,
                          M, N, K, w->G);
        double num = 0, da = 0, db = 0, worst_cpu = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            const double g = vv_half_to_float(hy[i]), r = cref[i];
            num += g * r; da += g * g; db += r * r;
            const double e = fabs(g - r) / (fabs(r) + 1.0);
            if (e > worst_cpu || e != e) worst_cpu = (e != e) ? 1e9 : e;
        }
        const double cosv = num / (sqrt(da) * sqrt(db) + 1e-300);

        /*
         * Sampled outputs against FP64 with the exact FP16 dequant. The
         * error scale is sqrt(sum (w x)^2), what FP32/FP16 accumulation of
         * K products of that size can drift by.
         */
        double worst = 0;
        const int samples = 512;
        for (int i = 0; i < samples; i++) {
            const int m = (int)(urand() % (uint32_t)M);
            const int n = (int)(urand() % (uint32_t)N);
            double acc = vv_half_to_float(bias[n]), sq = 0;
            const float* wr = w->wref + (size_t)n * K;
            const float* xr = fx + (size_t)m * K;
            for (int k = 0; k < K; k++) {
                const double p = (double)wr[k] * xr[k];
                acc += p; sq += p * p;
            }
            const double g = vv_half_to_float(hy[(size_t)m * N + n]);
            const double e = fabs(g - acc) / (fabs(acc) * 2e-3 + sqrt(sq) + 1e-6);
            if (e > worst || e != e) worst = (e != e) ? 1e9 : e;
        }
        const bool ok = worst < 0.05 && cosv > 0.99999 && worst_cpu < 0.05;
        printf("  %s %-26s M=%-5d exact %.4f  cpu %.2e  cos %.8f\n",
               ok ? "ok  " : "FAIL", tag, M, worst, worst_cpu, cosv);
        if (!ok) failures++;
    }
done:
    vv_dev_free(dx); vv_dev_free(dy);
    free(hx); free(fx); free(hy); free(cref);
}

static void run_shape(int N, int K, int G, bool sym, const int* ms, int nm) {
    qweight_t w;
    char tag[64];
    snprintf(tag, sizeof(tag), "%dx%d g%d %s", N, K, G, sym ? "sym" : "asym");
    if (make_weight(&w, N, K, G, sym)) { qweight_free(&w); return; }
    uint16_t* bias = (uint16_t*)malloc((size_t)N * 2);
    for (int n = 0; n < N; n++) bias[n] = vv_float_to_half(frand() * 0.5f);
    dweight_t d;
    if (upload_weight(&w, bias, &d)) {
        printf("  FAIL %s: allocation\n", tag);
        failures++;
    } else {
        for (int i = 0; i < nm; i++) check_m(&w, &d, bias, ms[i], tag);
    }
    free_dweight(&d);
    free(bias);
    qweight_free(&w);
}

/* ─── Benchmark ──────────────────────────────────────────────────────────── */

/** @brief Time `iters` launches captured into one graph; ms per launch. */
typedef vv_status_t (*launch_fn)(void* ctx, int rep);

static double time_graph(launch_fn fn, void* ctx, int reps, int iters) {
    for (int i = 0; i < reps; i++) fn(ctx, i);          /* warm up */
    vv_dev_stream_sync(g_stream);
    void* exec = NULL;
    if (vv_dev_graph_begin(g_stream) != VV_OK) return -1;
    for (int i = 0; i < iters; i++) fn(ctx, i % reps);
    if (vv_dev_graph_end(g_stream, &exec) != VV_OK || !exec) return -1;
    /* Let the clocks leave the idle state before anything is timed. */
    const double warm0 = vv_time_ms();
    while (vv_time_ms() - warm0 < 150.0) {
        vv_dev_graph_launch(exec, g_stream);
        vv_dev_stream_sync(g_stream);
    }
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        vv_dev_graph_launch(exec, g_stream);
        vv_dev_stream_sync(g_stream);
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
    int M, N, K, G, reps;
    void **gp, **gsz, **rp, **rs, **rm;
    void *x, *y, *bias;
    int legacy;
} bench_ctx_t;

static vv_status_t launch_bench(void* p, int rep) {
    bench_ctx_t* c = (bench_ctx_t*)p;
    if (c->legacy) {
        if (c->M == 1)
            return vv_awq_gemv_dev(c->x, (const uint32_t*)c->rp[rep],
                                   (const uint32_t*)c->rm[rep], c->rs[rep],
                                   c->bias, c->y, c->N, c->K, c->G, g_stream);
        return vv_awq_gemm_dev(c->x, (const uint32_t*)c->rp[rep],
                               (const uint32_t*)c->rm[rep], c->rs[rep], c->y,
                               g_scratch, c->M, c->N, c->K, c->G, g_stream);
    }
    if (c->M == 1)
        return vv_w4a16_gemv_dev(c->x, c->gp[rep], c->gsz[rep], c->bias, c->y,
                                 c->N, c->K, c->G, g_stream);
    return vv_w4a16_gemm_dev(c->x, c->gp[rep], c->gsz[rep], c->bias, c->y,
                             g_scratch, g_scratch_bytes, c->M, c->N, c->K,
                             c->G, g_stream);
}

static void bench_shape(const char* name, int N, int K, const int* ms, int nm) {
    const int G = 128;
    const size_t pb = (size_t)N * K / 2, gb = (size_t)N * (K / G) * 2;
    /* enough copies that the set does not fit in L2 */
    int reps = (int)((256u << 20) / (pb + 2 * gb)) + 1;
    if (reps > 64) reps = 64;
    bench_ctx_t c;
    memset(&c, 0, sizeof(c));
    c.N = N; c.K = K; c.G = G; c.reps = reps;
    c.gp = (void**)calloc(reps, sizeof(void*));
    c.gsz = (void**)calloc(reps, sizeof(void*));
    c.rp = (void**)calloc(reps, sizeof(void*));
    c.rs = (void**)calloc(reps, sizeof(void*));
    c.rm = (void**)calloc(reps, sizeof(void*));
    for (int r = 0; r < reps; r++) {
        vv_dev_alloc(&c.gp[r], pb);  vv_dev_memset(c.gp[r], 0x5A, pb);
        vv_dev_alloc(&c.gsz[r], gb * 2); vv_dev_memset(c.gsz[r], 0x11, gb * 2);
        vv_dev_alloc(&c.rp[r], pb);  vv_dev_memset(c.rp[r], 0x5A, pb);
        vv_dev_alloc(&c.rs[r], gb);  vv_dev_memset(c.rs[r], 0x11, gb);
        vv_dev_alloc(&c.rm[r], gb);  vv_dev_memset(c.rm[r], 0x11, gb);
    }
    int maxm = 1;
    for (int i = 0; i < nm; i++) if (ms[i] > maxm) maxm = ms[i];
    vv_dev_alloc(&c.x, (size_t)maxm * K * 2);
    vv_dev_memset(c.x, 0x11, (size_t)maxm * K * 2);
    vv_dev_alloc(&c.y, (size_t)maxm * N * 2);
    vv_dev_alloc(&c.bias, (size_t)N * 2);
    vv_dev_memset(c.bias, 0, (size_t)N * 2);

    for (int i = 0; i < nm; i++) {
        c.M = ms[i];
        const int iters = c.M <= 16 ? 200 : (c.M <= 256 ? 40 : 8);
        c.legacy = 0;
        const double tn = time_graph(launch_bench, &c, reps, iters);
        c.legacy = 1;
        const double to = time_graph(launch_bench, &c, reps, iters);
        const double bytes = (double)pb + 2.0 * gb;
        const double flop = 2.0 * c.M * (double)N * K;
        if (c.M == 1)
            printf("  %-10s %5dx%-5d M=1     new %7.1f us %6.1f GB/s   "
                   "old %7.1f us %6.1f GB/s   x%.2f\n", name, N, K,
                   tn * 1e3, bytes / (tn * 1e-3) / 1e9,
                   to * 1e3, bytes / (to * 1e-3) / 1e9, to / tn);
        else
            printf("  %-10s %5dx%-5d M=%-5d new %8.1f us %6.2f TFLOP/s "
                   "%6.1f GB/s   old %8.1f us %6.2f TFLOP/s   x%.2f\n",
                   name, N, K, c.M, tn * 1e3, flop / (tn * 1e-3) / 1e12,
                   bytes / (tn * 1e-3) / 1e9,
                   to * 1e3, flop / (to * 1e-3) / 1e12, to / tn);
    }
    for (int r = 0; r < reps; r++) {
        vv_dev_free(c.gp[r]); vv_dev_free(c.gsz[r]); vv_dev_free(c.rp[r]);
        vv_dev_free(c.rs[r]); vv_dev_free(c.rm[r]);
    }
    free(c.gp); free(c.gsz); free(c.rp); free(c.rs); free(c.rm);
    vv_dev_free(c.x); vv_dev_free(c.y); vv_dev_free(c.bias);
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
    /* Same size the decoder gives the dequant scratch for a 7B layer. */
    g_scratch_bytes = (size_t)18944 * 3584 * 2;
    if (vv_dev_alloc(&g_scratch, g_scratch_bytes) != VV_OK) {
        printf("FAIL: scratch\n");
        return 1;
    }
    const char* mma = getenv("VV_W4A16_MMA");
    printf("=== W4A16 GEMV/GEMM (%s) ===\n",
           (mma && mma[0] == '0') ? "dequant fallback forced" : "as built");

    const int ms_all[] = { 1, 2, 7, 8, 9, 16, 28, 64, 285, 2048 };
    const int n_all = (int)(sizeof(ms_all) / sizeof(ms_all[0]));
    const int ms_few[] = { 1, 3, 9, 64 };
    const int n_few = (int)(sizeof(ms_few) / sizeof(ms_few[0]));

    const char* b = getenv("VV_W4A16_BENCH");
    const bool bench_only = b && (b[0] == '2' || b[0] == '3');
    const bool bench = b && (b[0] >= '1' && b[0] <= '3');

    /* 7B projections, AWQ checkpoint's group size, asymmetric */
    if (!bench_only) {
    run_shape(3584, 3584, 128, false, ms_all, n_all);
    run_shape(512, 3584, 128, false, ms_all, n_all);
    run_shape(18944, 3584, 128, false, ms_all, n_all);
    run_shape(3584, 18944, 128, false, ms_all, n_all);
    /* 1.5B projections */
    run_shape(1536, 1536, 128, false, ms_all, n_all);
    run_shape(256, 1536, 128, false, ms_all, n_all);
    run_shape(8960, 1536, 128, false, ms_all, n_all);
    run_shape(1536, 8960, 128, false, ms_all, n_all);
    /* other group sizes, symmetric zeros */
    run_shape(512, 3584, 32, false, ms_few, n_few);
    run_shape(1536, 8960, 32, true, ms_few, n_few);
    run_shape(3584, 3584, 64, false, ms_few, n_few);
    run_shape(256, 1536, 64, true, ms_few, n_few);
    run_shape(3584, 3584, 128, true, ms_few, n_few);
    }

    /* VV_W4A16_BENCH_M=16,28 restricts the GEMM sweep (0 = GEMV only). */
    int mg[16] = { 2, 4, 8, 9, 16, 28, 64, 128, 285, 1024, 2048 };
    int nmg = 11;
    const char* bm = getenv("VV_W4A16_BENCH_M");
    if (bm && bm[0]) {
        nmg = 0;
        for (const char* p = bm; *p && nmg < 16; ) {
            const int v = atoi(p);
            if (v > 0) mg[nmg++] = v;
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const bool bench_gemv = !(bm && bm[0] && nmg > 0);

    if (bench && bench_gemv) {
        printf("\n--- GEMV (M=1), graph-replayed, weights cycled past L2 ---\n");
        const int m1[] = { 1 };
        bench_shape("q/o", 3584, 3584, m1, 1);
        bench_shape("k/v", 512, 3584, m1, 1);
        bench_shape("gate/up", 18944, 3584, m1, 1);
        bench_shape("down", 3584, 18944, m1, 1);
        bench_shape("1.5B q/o", 1536, 1536, m1, 1);
        bench_shape("1.5B k/v", 256, 1536, m1, 1);
        bench_shape("1.5B g/u", 8960, 1536, m1, 1);
        bench_shape("1.5B down", 1536, 8960, m1, 1);
    }
    if (bench && nmg > 0) {
        printf("\n--- GEMM by M ---\n");
        bench_shape("q/o", 3584, 3584, mg, nmg);
        bench_shape("k/v", 512, 3584, mg, nmg);
        bench_shape("gate/up", 18944, 3584, mg, nmg);
        bench_shape("down", 3584, 18944, mg, nmg);
    }

    vv_dev_free(g_scratch);
    vv_dev_stream_destroy(g_stream);
    if (failures) {
        printf("\n=== %d check(s) failed ===\n", failures);
        return 1;
    }
    printf("\n=== all W4A16 checks match the reference ===\n");
    return 0;
#endif
}
