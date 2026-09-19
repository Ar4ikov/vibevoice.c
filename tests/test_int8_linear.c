/**
 * @file test_int8_linear.c
 * @brief W8A8 / W4A8: activation quantizers and linear kernels, CPU and GPU.
 *
 * What has to hold, and how it is checked:
 *
 *  1. Every activation quantizer (CPU, GPU plain, GPU fused into RMSNorm and
 *     SwiGLU) produces the same bytes, the same scale and the same per-32
 *     sums as a scalar reference with round-half-even, on identical inputs.
 *     The fused GPU ones are compared against "unfused op, then quantize".
 *  2. The integer part is exact: with every scale set to 1 and FP32 output,
 *     each kernel must return the int64 reference dot product bit for bit.
 *  3. With real scales, bias, a residual and FP16 output, results sit within
 *     FP16 rounding of an FP64 reference.
 *
 * Shapes are the 7B's and the 1.5B's projections plus a seam shape, and M
 * covers 1, 2, 7, 8, 9, 16, 17, 64, 285 and 921 so that every kernel path and
 * every partial tile runs. Large products are verified on a sample of
 * entries (including the last row and column), small ones completely.
 *
 * `test_int8_linear --bench` also prints GEMM TOPS and GEMV GB/s.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...) do {                                      \
    checks++;                                                      \
    if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__);     \
                   printf("\n"); }                                 \
} while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}
static float frand(void) { return (float)(rnd() & 0xFFFFFF) / 8388608.0f - 1.0f; }

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* ─── Reference quantizer ───────────────────────────────────────────────── */

static void ref_quant(const float* x, int K, int layout, int8_t* q, float* sx,
                      int32_t* xs) {
    float amax = 0.0f;
    for (int k = 0; k < K; k++) if (fabsf(x[k]) > amax) amax = fabsf(x[k]);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    *sx = amax / 127.0f;
    for (int c = 0; c < K / 32; c++) {
        int s = 0;
        for (int j = 0; j < 32; j++) {
            /* nearbyint: round half to even in the default mode */
            double v = nearbyint((double)(x[c * 32 + j] * inv));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            s += (int)v;
            q[c * 32 + vv_q8_pos(layout, j)] = (int8_t)v;
        }
        xs[c] = s;
    }
}

static void test_cpu_quantizer(void) {
    printf("cpu quantizer (bit-exact vs reference)\n");
    const int Ks[] = { 32, 1536, 3584, 8960, 18944 };
    for (int ki = 0; ki < 5; ki++) {
        const int K = Ks[ki], M = 5;
        float* x = (float*)malloc((size_t)M * K * 4);
        for (int i = 0; i < M * K; i++) x[i] = frand() * 7.0f;
        /* row 1 all zero, row 2 exact .5 ties: amax 127 so inv = 1 */
        memset(x + K, 0, (size_t)K * 4);
        for (int k = 0; k < K; k++) x[2 * K + k] = (float)((k % 254) - 127) + 0.5f * ((k & 1) ? 1 : -1);
        x[2 * K] = 127.0f;
        int8_t* q = (int8_t*)malloc((size_t)M * K);
        int8_t* r = (int8_t*)malloc((size_t)M * K);
        int32_t* xs = (int32_t*)malloc((size_t)M * K / 32 * 4);
        int32_t* rs = (int32_t*)malloc((size_t)M * K / 32 * 4);
        float sx[5], rx[5];
        for (int nib = 0; nib < VV_Q8_LAYOUT_COUNT; nib++) {
            vv_quant_act_q8_cpu(x, M, K, nib, q, sx, xs);
            int bad = 0;
            for (int m = 0; m < M; m++) {
                ref_quant(x + (size_t)m * K, K, nib, r + (size_t)m * K, &rx[m],
                          rs + (size_t)m * (K / 32));
                if (memcmp(&sx[m], &rx[m], 4) != 0) bad++;
            }
            if (memcmp(q, r, (size_t)M * K) != 0) bad++;
            if (memcmp(xs, rs, (size_t)M * K / 32 * 4) != 0) bad++;
            CHECK(bad == 0, "cpu quantizer K=%d layout=%d", K, nib);
        }
        free(x); free(q); free(r); free(xs); free(rs);
    }
}

/* ─── Test problem ──────────────────────────────────────────────────────── */

typedef struct {
    int N, K, G;
    int8_t* w8;          /* [N][K]        */
    float* sw;           /* [N]           */
    uint8_t* w4;         /* [N][K/2]      */
    uint16_t* s4;        /* [N][K/G] FP16 */
    uint16_t* s4one;     /* all 1.0       */
    float* sone;         /* all 1.0       */
    uint8_t* z4;         /* [N][K/G]      */
    uint16_t* bias;      /* [N] FP16      */
    uint8_t* w4g;        /* w4 in the W4A16 GPU layout          */
    uint16_t* sz;        /* half2 {s4, z4} [N][K/G]             */
    uint16_t* szone;     /* half2 {1, z4}                       */
} problem_t;

static problem_t make_problem(int N, int K, int G) {
    problem_t p;
    memset(&p, 0, sizeof(p));
    p.N = N; p.K = K; p.G = G;
    const int ng = K / G;
    p.w8 = (int8_t*)malloc((size_t)N * K);
    p.w4 = (uint8_t*)malloc((size_t)N * K / 2);
    p.sw = (float*)malloc((size_t)N * 4);
    p.sone = (float*)malloc((size_t)N * 4);
    p.s4 = (uint16_t*)malloc((size_t)N * ng * 2);
    p.s4one = (uint16_t*)malloc((size_t)N * ng * 2);
    p.z4 = (uint8_t*)malloc((size_t)N * ng);
    p.bias = (uint16_t*)malloc((size_t)N * 2);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        int v = (int)(rnd() % 255) - 127;
        p.w8[i] = (int8_t)v;
    }
    for (size_t i = 0; i < (size_t)N * K / 2; i++) p.w4[i] = (uint8_t)rnd();
    for (int n = 0; n < N; n++) {
        p.sw[n] = 0.0005f + 0.002f * (frand() + 1.0f);
        p.sone[n] = 1.0f;
        p.bias[n] = vv_float_to_half(frand() * 0.5f);
    }
    for (size_t i = 0; i < (size_t)N * ng; i++) {
        p.s4[i] = vv_float_to_half(0.002f + 0.01f * (frand() + 1.0f));
        p.s4one[i] = vv_float_to_half(1.0f);
        p.z4[i] = (uint8_t)(rnd() % 16);
    }
    /* What the GPU reads: the same codes, rearranged by the conversion the
       loader uses, and the group data as half2 {scale, zero}. */
    p.w4g = (uint8_t*)malloc((size_t)N * K / 2);
    p.sz = (uint16_t*)malloc((size_t)N * ng * 4);
    p.szone = (uint16_t*)malloc((size_t)N * ng * 4);
    memcpy(p.w4g, p.w4, (size_t)N * K / 2);
    vv_int4g_to_gpu_layout(p.w4g, p.s4, p.z4, N, K, G, p.sz);
    {
        uint8_t* tmp = (uint8_t*)malloc((size_t)N * K / 2);
        memcpy(tmp, p.w4, (size_t)N * K / 2);
        vv_int4g_to_gpu_layout(tmp, p.s4one, p.z4, N, K, G, p.szone);
        free(tmp);
    }
    return p;
}

static void free_problem(problem_t* p) {
    free(p->w8); free(p->w4); free(p->sw); free(p->sone); free(p->s4);
    free(p->s4one); free(p->z4); free(p->bias); free(p->w4g); free(p->sz);
    free(p->szone);
}

/* Exact integer parts for one (m, n). W4: per group (sum (q - z) x). */
static int64_t ref_w8_dot(const problem_t* p, const int8_t* x, int n) {
    int64_t a = 0;
    const int8_t* w = p->w8 + (size_t)n * p->K;
    for (int k = 0; k < p->K; k++) a += (int64_t)w[k] * x[k];
    return a;
}

static double ref_w4(const problem_t* p, const int8_t* x, int n, int unit) {
    const int ng = p->K / p->G;
    double acc = 0.0;
    for (int g = 0; g < ng; g++) {
        int64_t d = 0;
        const int z = p->z4[(size_t)n * ng + g];
        for (int k = g * p->G; k < (g + 1) * p->G; k++) {
            const uint8_t b = p->w4[(size_t)n * (p->K / 2) + k / 2];
            const int q = (k & 1) ? (b & 15) : (b >> 4);
            d += (int64_t)(q - z) * x[k];
        }
        const double s = unit ? 1.0
            : (double)vv_half_to_float(p->s4[(size_t)n * ng + g]);
        acc += s * (double)d;
    }
    return acc;
}

/* Entries to verify: all of them when cheap, else a sample plus the edges. */
static int pick_entries(int M, int N, int K, int* ms, int* ns, int cap) {
    if ((double)M * N * K <= 6e7 && M * N <= cap) {
        int c = 0;
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            ms[c] = m; ns[c] = n; c++;
        }
        return c;
    }
    int c = 0;
    const int edge[4][2] = { {0, 0}, {M - 1, N - 1}, {0, N - 1}, {M - 1, 0} };
    for (int i = 0; i < 4; i++) { ms[c] = edge[i][0]; ns[c] = edge[i][1]; c++; }
    while (c < cap) { ms[c] = (int)(rnd() % M); ns[c] = (int)(rnd() % N); c++; }
    return c;
}

/* ─── CPU kernels ───────────────────────────────────────────────────────── */

static void test_cpu_linear(const problem_t* p, int M) {
    const int K = p->K, N = p->N;
    float* x = (float*)malloc((size_t)M * K * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = frand() * 3.0f;
    int8_t* xn = (int8_t*)malloc((size_t)M * K);
    int8_t* xb = (int8_t*)malloc((size_t)M * K);
    float* sx = (float*)malloc((size_t)M * 4);
    int32_t* xs = (int32_t*)malloc((size_t)M * (K / 32) * 4);
    float* out = (float*)malloc((size_t)M * N * 4);
    vv_quant_act_q8_cpu(x, M, K, VV_Q8_NATURAL, xn, sx, xs);
    vv_quant_act_q8_cpu(x, M, K, VV_Q8_NIBBLE, xb, sx, xs);

    enum { CAP = 3000 };
    static int ms[CAP], ns[CAP];
    const int cnt = pick_entries(M, N, K, ms, ns, CAP);

    /* W8A8, unit scales: exact. */
    float one_m[1024];
    for (int i = 0; i < M && i < 1024; i++) one_m[i] = 1.0f;
    vv_w8a8_gemm_cpu(xn, one_m, p->w8, p->sone, NULL, out, M, N, K);
    int bad = 0;
    for (int i = 0; i < cnt; i++) {
        const int64_t r = ref_w8_dot(p, xn + (size_t)ms[i] * K, ns[i]);
        if ((double)out[(size_t)ms[i] * N + ns[i]] != (double)r) bad++;
    }
    CHECK(bad == 0, "cpu w8a8 exact M=%d N=%d K=%d: %d/%d", M, N, K, bad, cnt);

    vv_w8a8_gemm_cpu(xn, sx, p->w8, p->sw, p->bias, out, M, N, K);
    bad = 0;
    for (int i = 0; i < cnt; i++) {
        const int m = ms[i], n = ns[i];
        const double r = (double)ref_w8_dot(p, xn + (size_t)m * K, n) *
                         sx[m] * p->sw[n] + vv_half_to_float(p->bias[n]);
        const double o = out[(size_t)m * N + n];
        if (fabs(o - r) > 1e-5 * fabs(r) + 1e-5) bad++;
    }
    CHECK(bad == 0, "cpu w8a8 scaled M=%d N=%d K=%d: %d/%d", M, N, K, bad, cnt);

    /* W4A8. The CPU kernel reads the nibble layout; reference uses natural. */
    vv_w4a8_gemm_cpu(xb, one_m, xs, p->w4, p->s4one, p->z4, p->G, NULL, out,
                     M, N, K);
    bad = 0;
    for (int i = 0; i < cnt; i++) {
        const double r = ref_w4(p, xn + (size_t)ms[i] * K, ns[i], 1);
        if ((double)out[(size_t)ms[i] * N + ns[i]] != r) bad++;
    }
    CHECK(bad == 0, "cpu w4a8 exact M=%d N=%d K=%d G=%d: %d/%d", M, N, K,
          p->G, bad, cnt);

    vv_w4a8_gemm_cpu(xb, sx, xs, p->w4, p->s4, p->z4, p->G, p->bias, out,
                     M, N, K);
    bad = 0;
    for (int i = 0; i < cnt; i++) {
        const int m = ms[i], n = ns[i];
        const double r = ref_w4(p, xn + (size_t)m * K, n, 0) * sx[m] +
                         vv_half_to_float(p->bias[n]);
        const double o = out[(size_t)m * N + n];
        if (fabs(o - r) > 1e-5 * fabs(r) + 1e-4) bad++;
    }
    CHECK(bad == 0, "cpu w4a8 scaled M=%d N=%d K=%d: %d/%d", M, N, K, bad, cnt);

    free(x); free(xn); free(xb); free(sx); free(xs); free(out);
}

/* ─── GPU ───────────────────────────────────────────────────────────────── */

static void* dup_dev(const void* h, size_t n) {
    void* d = NULL;
    if (vv_dev_alloc(&d, n ? n : 16) != VV_OK) return NULL;
    if (n && h) vv_dev_memcpy_h2d(d, h, n, NULL);
    return d;
}

static void test_gpu_quantizers(void* st) {
    printf("gpu quantizers (bit-exact vs CPU on the same FP16 values)\n");
    const int Ks[] = { 1536, 3584, 8960, 18944 };
    const int Ms[] = { 1, 7, 285 };
    for (int ki = 0; ki < 4; ki++) for (int mi = 0; mi < 3; mi++) {
        const int K = Ks[ki], M = Ms[mi];
        const size_t n = (size_t)M * K;
        uint16_t* h = (uint16_t*)malloc(n * 2);
        uint16_t* h2 = (uint16_t*)malloc(n * 2);
        uint16_t* hw = (uint16_t*)malloc((size_t)K * 2);
        float* f = (float*)malloc(n * 4);
        for (size_t i = 0; i < n; i++) h[i] = vv_float_to_half(frand() * 4.0f);
        for (size_t i = 0; i < n; i++) h2[i] = vv_float_to_half(frand() * 3.0f);
        for (int i = 0; i < K; i++) hw[i] = vv_float_to_half(0.5f + frand() * 0.4f);
        void* dx = dup_dev(h, n * 2);
        void* dx2 = dup_dev(h2, n * 2);
        void* dw = dup_dev(hw, (size_t)K * 2);
        void* dy = dup_dev(NULL, n * 2);
        int8_t* dq = (int8_t*)dup_dev(NULL, n);
        float* dsx = (float*)dup_dev(NULL, (size_t)M * 4);
        int32_t* dxs = (int32_t*)dup_dev(NULL, (size_t)M * (K / 32) * 4);
        int8_t* q = (int8_t*)malloc(n), * rq = (int8_t*)malloc(n);
        float* sx = (float*)malloc((size_t)M * 4), * rsx = (float*)malloc((size_t)M * 4);
        int32_t* xs = (int32_t*)malloc((size_t)M * (K / 32) * 4);
        int32_t* rxs = (int32_t*)malloc((size_t)M * (K / 32) * 4);
        uint16_t* hy = (uint16_t*)malloc(n * 2);

        for (int wl = 0; wl < 3 * VV_Q8_LAYOUT_COUNT; wl++) {
            const int which = wl % 3, lay = wl / 3;
            vv_status_t s;
            if (which == 0) {
                s = vv_act_quant_dev(dx, M, K, lay, dq, dsx, dxs, st);
                memcpy(hy, h, n * 2);
            } else if (which == 1) {
                s = vv_rmsnorm_dev(dx, dw, dy, M, K, 1e-6f, st);
                if (s == VV_OK) s = vv_rmsnorm_q8_dev(dx, dw, M, K, 1e-6f, lay, dq, dsx, dxs, st);
            } else {
                s = vv_swiglu_dev(dx, dx2, dy, (int)n, st);
                if (s == VV_OK) s = vv_swiglu_q8_dev(dx, dx2, M, K, lay, dq, dsx, dxs, st);
            }
            vv_dev_stream_sync(st);
            if (which > 0) vv_dev_memcpy_d2h(hy, dy, n * 2, NULL);
            vv_dev_memcpy_d2h(q, dq, n, NULL);
            vv_dev_memcpy_d2h(sx, dsx, (size_t)M * 4, NULL);
            vv_dev_memcpy_d2h(xs, dxs, (size_t)M * (K / 32) * 4, NULL);
            for (size_t i = 0; i < n; i++) f[i] = vv_half_to_float(hy[i]);
            vv_quant_act_q8_cpu(f, M, K, lay, rq, rsx, rxs);
            const int same = s == VV_OK && memcmp(q, rq, n) == 0 &&
                             memcmp(sx, rsx, (size_t)M * 4) == 0 &&
                             memcmp(xs, rxs, (size_t)M * (K / 32) * 4) == 0;
            static const char* nm[3] = { "act_quant", "rmsnorm_q8", "swiglu_q8" };
            CHECK(same, "gpu %s M=%d K=%d layout=%d (status %d)", nm[which],
                  M, K, lay, (int)s);
        }
        vv_dev_free(dx); vv_dev_free(dx2); vv_dev_free(dw); vv_dev_free(dy);
        vv_dev_free(dq); vv_dev_free(dsx); vv_dev_free(dxs);
        free(h); free(h2); free(hw); free(f); free(q); free(rq); free(sx);
        free(rsx); free(xs); free(rxs); free(hy);
    }
}

/** SmoothQuant calibration's per-column absmax, accumulated over calls. */
static void test_gpu_col_absmax(void* st) {
    printf("gpu calibration absmax (exact vs CPU)\n");
    const int Ks[] = { 3584, 18944, 100 };
    const int Ms[] = { 1, 70, 285 };
    for (int ki = 0; ki < 3; ki++) for (int mi = 0; mi < 3; mi++) {
        const int K = Ks[ki], M = Ms[mi];
        const size_t n = (size_t)M * K;
        uint16_t* h = (uint16_t*)malloc(n * 2);
        float* ref = (float*)malloc((size_t)K * 4);
        float* got = (float*)malloc((size_t)K * 4);
        for (int k = 0; k < K; k++) ref[k] = (k % 5 == 0) ? 3.5f : 0.0f;
        void* dacc = dup_dev(ref, (size_t)K * 4);      /* prior maxima */
        void* dx = dup_dev(NULL, n * 2);
        vv_status_t s = VV_OK;
        for (int pass = 0; pass < 2 && s == VV_OK; pass++) {
            for (size_t i = 0; i < n; i++)
                h[i] = vv_float_to_half(frand() * (1.0f + (float)(i % K) / K) *
                                        (pass ? 4.0f : 2.0f));
            for (size_t i = 0; i < n; i++) {
                const float v = fabsf(vv_half_to_float(h[i]));
                if (v > ref[i % K]) ref[i % K] = v;
            }
            vv_dev_memcpy_h2d(dx, h, n * 2, st);
            s = vv_col_absmax_dev(dx, M, K, (float*)dacc, st);
        }
        vv_dev_stream_sync(st);
        vv_dev_memcpy_d2h(got, dacc, (size_t)K * 4, NULL);
        CHECK(s == VV_OK && memcmp(got, ref, (size_t)K * 4) == 0,
              "gpu col_absmax M=%d K=%d, two passes (status %d)", M, K, (int)s);
        vv_dev_free(dacc); vv_dev_free(dx);
        free(h); free(ref); free(got);
    }
}

typedef struct {
    void *w8, *sw, *sone, *w4, *sz, *szone, *bias;
} dev_problem_t;

static dev_problem_t upload(const problem_t* p) {
    dev_problem_t d;
    const size_t ng = (size_t)p->N * (p->K / p->G);
    d.w8 = dup_dev(p->w8, (size_t)p->N * p->K);
    d.sw = dup_dev(p->sw, (size_t)p->N * 4);
    d.sone = dup_dev(p->sone, (size_t)p->N * 4);
    d.w4 = dup_dev(p->w4g, (size_t)p->N * p->K / 2);
    d.sz = dup_dev(p->sz, ng * 4);
    d.szone = dup_dev(p->szone, ng * 4);
    d.bias = dup_dev(p->bias, (size_t)p->N * 2);
    return d;
}

static void free_dev(dev_problem_t* d) {
    vv_dev_free(d->w8); vv_dev_free(d->sw); vv_dev_free(d->sone);
    vv_dev_free(d->w4); vv_dev_free(d->sz); vv_dev_free(d->szone);
    vv_dev_free(d->bias);
}

static void test_gpu_linear(const problem_t* p, const dev_problem_t* d, int M,
                            void* st) {
    const int K = p->K, N = p->N;
    const size_t mk = (size_t)M * K, mn = (size_t)M * N;
    float* x = (float*)malloc(mk * 4);
    for (size_t i = 0; i < mk; i++) x[i] = frand() * 3.0f;
    int8_t* xq = (int8_t*)malloc(mk);
    float* sx = (float*)malloc((size_t)M * 4);
    float* one = (float*)malloc((size_t)M * 4);
    int32_t* xs = (int32_t*)malloc((size_t)M * (K / 32) * 4);
    vv_quant_act_q8_cpu(x, M, K, VV_Q8_NATURAL, xq, sx, xs);
    /* The W4 kernels read the GPU-layout orders: GEMV and MMA/SIMT. */
    int8_t* xqn = (int8_t*)malloc(mk);
    int8_t* xqm = (int8_t*)malloc(mk);
    vv_quant_act_q8_cpu(x, M, K, VV_Q8_W4_GEMV, xqn, sx, xs);
    vv_quant_act_q8_cpu(x, M, K, VV_Q8_W4_MMA, xqm, sx, xs);
    for (int m = 0; m < M; m++) one[m] = 1.0f;
    uint16_t* res = (uint16_t*)malloc(mn * 2);
    for (size_t i = 0; i < mn; i++) res[i] = vv_float_to_half(frand());

    int8_t* dxq = (int8_t*)dup_dev(xq, mk);
    int8_t* dxqn = (int8_t*)dup_dev(xqn, mk);
    int8_t* dxqm = (int8_t*)dup_dev(xqm, mk);
    float* dsx = (float*)dup_dev(sx, (size_t)M * 4);
    float* done = (float*)dup_dev(one, (size_t)M * 4);
    int32_t* dxs = (int32_t*)dup_dev(xs, (size_t)M * (K / 32) * 4);
    void* dres = dup_dev(res, mn * 2);
    void* dy = dup_dev(NULL, mn * 4);
    float* yf = (float*)malloc(mn * 4);
    uint16_t* yh = (uint16_t*)malloc(mn * 2);

    enum { CAP = 3000 };
    static int ms[CAP], ns[CAP];
    const int cnt = pick_entries(M, N, K, ms, ns, CAP);

    const int paths[3] = { VV_I8_PATH_GEMV, VV_I8_PATH_MMA, VV_I8_PATH_SIMT };
    static const char* pn[4] = { "auto", "gemv", "mma", "simt" };
    for (int pi = 0; pi < 3; pi++) {
        const int path = paths[pi];
        if (path == VV_I8_PATH_GEMV && M > 8) continue;
        for (int w4 = 0; w4 < 2; w4++) {
            /* W8A8 reads natural order; W4A8 the GEMV or the MMA order. */
            const int lay = !w4 ? VV_Q8_NATURAL
                          : path == VV_I8_PATH_GEMV ? VV_Q8_W4_GEMV : VV_Q8_W4_MMA;
            const int8_t* ax = lay == VV_Q8_W4_GEMV ? dxqn
                             : lay == VV_Q8_W4_MMA ? dxqm : dxq;
            /* exact integer part */
            vv_status_t s = w4
                ? vv_w4a8_linear_dev(ax, lay, done, dxs, d->w4, d->szone, p->G,
                                     NULL, NULL, dy, 1, M, N, K, path, st)
                : vv_w8a8_linear_dev(dxq, done, (const int8_t*)d->w8,
                                     (const float*)d->sone, NULL, NULL, dy, 1,
                                     M, N, K, path, st);
            if (s == VV_ERR_UNSUPPORTED && path == VV_I8_PATH_MMA) continue;
            vv_dev_stream_sync(st);
            vv_dev_memcpy_d2h(yf, dy, mn * 4, NULL);
            int bad = 0;
            for (int i = 0; i < cnt; i++) {
                const int m = ms[i], n = ns[i];
                const double r = w4 ? ref_w4(p, xq + (size_t)m * K, n, 1)
                                    : (double)ref_w8_dot(p, xq + (size_t)m * K, n);
                if ((double)yf[(size_t)m * N + n] != r) {
                    if (bad < 3) printf("    m=%d n=%d got %.1f want %.1f\n",
                                        m, n, yf[(size_t)m * N + n], r);
                    bad++;
                }
            }
            CHECK(s == VV_OK && bad == 0, "gpu %s %s exact M=%d N=%d K=%d: "
                  "%d/%d (status %d)", w4 ? "w4a8" : "w8a8",
                  pn[path], M, N, K, bad, cnt, (int)s);

            /* real scales, bias and a residual, FP16 out (y = res + ...) */
            vv_dev_memcpy_h2d(dy, res, mn * 2, NULL);
            s = w4
                ? vv_w4a8_linear_dev(ax, lay, dsx, dxs, d->w4, d->sz, p->G,
                                     d->bias, dy, dy, 0, M, N, K, path, st)
                : vv_w8a8_linear_dev(dxq, dsx, (const int8_t*)d->w8,
                                     (const float*)d->sw, d->bias, dy, dy, 0,
                                     M, N, K, path, st);
            vv_dev_stream_sync(st);
            vv_dev_memcpy_d2h(yh, dy, mn * 2, NULL);
            bad = 0;
            double worst = 0.0;
            for (int i = 0; i < cnt; i++) {
                const int m = ms[i], n = ns[i];
                const double core = w4 ? ref_w4(p, xq + (size_t)m * K, n, 0) * sx[m]
                    : (double)ref_w8_dot(p, xq + (size_t)m * K, n) * sx[m] * p->sw[n];
                const double r = core + vv_half_to_float(p->bias[n]) +
                                 vv_half_to_float(res[(size_t)m * N + n]);
                const double o = vv_half_to_float(yh[(size_t)m * N + n]);
                const double err = fabs(o - r);
                const double tol = fabs(r) * (1.0 / 1024.0) + 1e-3;
                if (err / tol > worst) worst = err / tol;
                if (err > tol) bad++;
            }
            CHECK(s == VV_OK && bad == 0, "gpu %s %s fp16 M=%d N=%d K=%d: %d/%d "
                  "worst %.2f tol", w4 ? "w4a8" : "w8a8",
                  pn[path], M, N, K, bad, cnt, worst);
        }
    }
    /* Shared-input launches (q/k/v, gate/up): three projections of the
       same input in one GEMV grid, each checked like a single one. */
    if (M <= 8) {
        void* dy2 = dup_dev(NULL, mn * 2 * 3);
        for (int w4 = 0; w4 < 2; w4++) {
            vv_i8_proj_t pr[3];
            for (int i = 0; i < 3; i++) {
                pr[i].w = w4 ? d->w4 : d->w8;
                pr[i].sw = d->sw;
                pr[i].sz = d->sz;
                pr[i].bias = i == 1 ? NULL : d->bias;
                pr[i].y = (uint8_t*)dy2 + (size_t)i * mn * 2;
                pr[i].N = i == 2 ? (N + 1) / 2 : N;   /* a shorter third */
            }
            vv_status_t s = vv_i8_linear_multi_dev(
                w4 ? dxqn : dxq, w4 ? VV_Q8_W4_GEMV : VV_Q8_NATURAL, dsx, dxs,
                w4, pr, 3, p->G, M, K, st);
            vv_dev_stream_sync(st);
            int bad = 0;
            for (int i = 0; i < 3 && s == VV_OK; i++) {
                const int Ni = pr[i].N;
                vv_dev_memcpy_d2h(yh, pr[i].y, (size_t)M * Ni * 2, NULL);
                for (int e = 0; e < cnt; e++) {
                    const int m = ms[e], n = ns[e];
                    if (n >= Ni) continue;
                    const double core = w4
                        ? ref_w4(p, xq + (size_t)m * K, n, 0) * sx[m]
                        : (double)ref_w8_dot(p, xq + (size_t)m * K, n) * sx[m] * p->sw[n];
                    const double r = core +
                        (pr[i].bias ? vv_half_to_float(p->bias[n]) : 0.0);
                    const double o = vv_half_to_float(yh[(size_t)m * Ni + n]);
                    if (fabs(o - r) > fabs(r) * (1.0 / 1024.0) + 1e-3) bad++;
                }
            }
            CHECK(s == VV_OK && bad == 0, "gpu %s multi M=%d N=%d K=%d: %d "
                  "(status %d)", w4 ? "w4a8" : "w8a8", M, N, K, bad, (int)s);
        }
        vv_dev_free(dy2);
    }

    vv_dev_free(dxq); vv_dev_free(dxqn); vv_dev_free(dxqm); vv_dev_free(dsx);
    vv_dev_free(done);
    vv_dev_free(dxs); vv_dev_free(dres); vv_dev_free(dy);
    free(xqn); free(xqm); free(x); free(xq); free(sx); free(one); free(xs); free(res); free(yf);
    free(yh);
}

/* ─── Benchmark ─────────────────────────────────────────────────────────── */

static void bench(void* st) {
    printf("\nbench (best of 20; W8A8 and W4A8, FP16 out)\n");
    struct { int N, K; const char* nm; } sh[] = {
        { 4608, 3584, "qkv-sized 7B" }, { 3584, 3584, "o 7B" },
        { 18944, 3584, "gate/up 7B" }, { 3584, 18944, "down 7B" },
        { 8960, 1536, "gate/up 1.5B" }, { 512, 3584, "k/v 7B" },
    };
    const int Ms[] = { 1, 8, 28, 64, 256, 2048 };
    for (int si = 0; si < 6; si++) {
        problem_t p = make_problem(sh[si].N, sh[si].K, 128);
        dev_problem_t d = upload(&p);
        const int K = p.K, N = p.N;
        for (int mi = 0; mi < 6; mi++) {
            const int M = Ms[mi];
            int8_t* dxq = (int8_t*)dup_dev(NULL, (size_t)M * K);
            vv_dev_memset(dxq, 1, (size_t)M * K);
            float* dsx = (float*)dup_dev(NULL, (size_t)M * 4);
            vv_dev_memset(dsx, 0, (size_t)M * 4);
            int32_t* dxs = (int32_t*)dup_dev(NULL, (size_t)M * (K / 32) * 4);
            vv_dev_memset(dxs, 0, (size_t)M * (K / 32) * 4);
            void* dy = dup_dev(NULL, (size_t)M * N * 2);
            for (int w4 = 0; w4 < 2; w4++) {
                /*
                 * 20 launches back to back per sample, best of 5 samples, so
                 * the ~5 us launch-plus-sync overhead does not swamp the
                 * decode-sized kernels (in the model they run inside a graph).
                 */
                double best = 1e30;
                for (int it = 0; it < 6; it++) {
                    vv_dev_stream_sync(st);
                    const double t0 = now_ms();
                    for (int r = 0; r < 20; r++) {
                        if (w4) vv_w4a8_linear_dev(dxq, vv_w4a8_layout_for(M),
                                                   dsx, dxs, d.w4, d.sz, 128,
                                                   NULL, NULL, dy, 0, M, N, K,
                                                   VV_I8_PATH_AUTO, st);
                        else    vv_w8a8_linear_dev(dxq, dsx, (const int8_t*)d.w8,
                                                   (const float*)d.sw, NULL, NULL,
                                                   dy, 0, M, N, K,
                                                   VV_I8_PATH_AUTO, st);
                    }
                    vv_dev_stream_sync(st);
                    const double t = (now_ms() - t0) / 20.0;
                    if (it >= 1 && t < best) best = t;
                }
                const double wbytes = w4 ? (double)N * K / 2 : (double)N * K;
                const double tops = 2.0 * M * N * K / (best * 1e-3) / 1e12;
                const double gbs = (wbytes + (double)M * K) / (best * 1e-3) / 1e9;
                printf("  %-13s %s  M=%-5d %8.3f ms  %7.1f TOPS  %7.1f GB/s\n",
                       sh[si].nm, w4 ? "W4A8 " : "W8A8 ", M, best, tops, gbs);
            }
            /* The same weights on FP16 activations: the W4A16 kernels. */
            {
                void* dx = dup_dev(NULL, (size_t)M * K * 2);
                vv_dev_memset(dx, 0, (size_t)M * K * 2);
                void* scratch = dup_dev(NULL, (size_t)N * K * 2);
                double best = 1e30;
                for (int it = 0; it < 6; it++) {
                    vv_dev_stream_sync(st);
                    const double t0 = now_ms();
                    for (int r = 0; r < 20; r++) {
                        if (M == 1)
                            vv_w4a16_gemv_dev(dx, d.w4, d.sz, NULL, dy, N, K,
                                              128, st);
                        else
                            vv_w4a16_gemm_dev(dx, d.w4, d.sz, NULL, dy,
                                              scratch, (size_t)N * K * 2,
                                              M, N, K, 128, st);
                    }
                    vv_dev_stream_sync(st);
                    const double t = (now_ms() - t0) / 20.0;
                    if (it >= 1 && t < best) best = t;
                }
                printf("  %-13s W4A16 M=%-5d %8.3f ms  %7.1f TFLOPS\n",
                       sh[si].nm, M, best,
                       2.0 * M * N * K / (best * 1e-3) / 1e12);
                vv_dev_free(dx);
                vv_dev_free(scratch);
            }
            vv_dev_free(dxq); vv_dev_free(dsx); vv_dev_free(dxs); vv_dev_free(dy);
        }
        free_dev(&d);
        free_problem(&p);
    }
}

/*
 * --bench-cpu: the CPU kernels on the 7B projections, against the FP32
 * INT4G (W4A16) kernel they would replace. M = 143 is jfk's prefill. The
 * int8 timings include quantizing the activations, which the decoder pays
 * once per input (shared by q/k/v and by gate/up).
 */
static void bench_cpu(void) {
    printf("\ncpu bench (best of 5, %s)\n", vv_cpu_int8_isa());
    const struct { const char* nm; int N, K; } sh[] = {
        { "q/o 7B", 3584, 3584 }, { "gate 7B", 18944, 3584 },
        { "down 7B", 3584, 18944 },
    };
    const int Ms[] = { 1, 143 };
    for (int si = 0; si < 3; si++) {
        problem_t p = make_problem(sh[si].N, sh[si].K, 128);
        const int N = p.N, K = p.K, ng = K / p.G;
        uint16_t* mins = (uint16_t*)malloc((size_t)N * ng * 2);
        for (size_t i = 0; i < (size_t)N * ng; i++)
            mins[i] = vv_float_to_half(-(float)p.z4[i] *
                                       vv_half_to_float(p.s4[i]));
        for (int mi = 0; mi < 2; mi++) {
            const int M = Ms[mi];
            float* x = (float*)malloc((size_t)M * K * 4);
            float* y = (float*)malloc((size_t)M * N * 4);
            int8_t* xq = (int8_t*)malloc((size_t)M * K);
            float* sx = (float*)malloc((size_t)M * 4);
            int32_t* xs = (int32_t*)malloc((size_t)M * (K / 32) * 4);
            for (size_t i = 0; i < (size_t)M * K; i++) x[i] = frand();
            for (int kind = 0; kind < 3; kind++) {
                double best = 1e30;
                for (int it = 0; it < 5; it++) {
                    const double t0 = now_ms();
                    if (kind == 0) {
                        vv_int4g_gemm_cpu(x, p.w4, p.s4, mins, p.bias, y,
                                          M, N, K, p.G);
                    } else if (kind == 1) {
                        vv_quant_act_q8_cpu(x, M, K, VV_Q8_NIBBLE, xq, sx, xs);
                        vv_w4a8_gemm_cpu(xq, sx, xs, p.w4, p.s4, p.z4, p.G,
                                         p.bias, y, M, N, K);
                    } else {
                        vv_quant_act_q8_cpu(x, M, K, VV_Q8_NATURAL, xq, sx,
                                            NULL);
                        vv_w8a8_gemm_cpu(xq, sx, p.w8, p.sw, p.bias, y,
                                         M, N, K);
                    }
                    const double t = now_ms() - t0;
                    if (t < best) best = t;
                }
                static const char* names[3] = { "W4A16", "W4A8 ", "W8A8 " };
                printf("  %-8s %s M=%-4d %8.2f ms  %6.1f GOPS\n", sh[si].nm,
                       names[kind], M, best,
                       2.0 * M * N * K / (best * 1e-3) / 1e9);
            }
            free(x); free(y); free(xq); free(sx); free(xs);
        }
        free(mins);
        free_problem(&p);
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench-cpu") == 0) {
        bench_cpu();
        return 0;
    }
    const int do_bench = argc > 1 && strcmp(argv[1], "--bench") == 0;
    printf("=== int8 linear (W8A8 / W4A8) ===  CPU ISA: %s\n",
           vv_cpu_int8_isa());

    const int Ms[] = { 1, 2, 7, 8, 9, 16, 17, 64, 285, 921 };
    const int nM = (int)(sizeof(Ms) / sizeof(Ms[0]));
    struct { int N, K, G; } shapes[] = {
        { 512, 3584, 128 }, { 3584, 3584, 128 }, { 18944, 3584, 128 },
        { 3584, 18944, 128 },                                   /* 7B   */
        { 256, 1536, 128 }, { 1536, 1536, 128 }, { 8960, 1536, 128 },
        { 1536, 8960, 128 },                                    /* 1.5B */
        { 300, 640, 64 }, { 136, 640, 32 }, { 77, 256, 256 },   /* seams */
        { 64, 1024, 512 },          /* a group longer than one int16 run */
    };
    const int nS = (int)(sizeof(shapes) / sizeof(shapes[0]));
    /*
     * --quick: the seam shapes and M <= 17 only, for emulated targets
     * (qemu-aarch64 runs the NEON dotprod kernels at a few GOPS).
     */
    const int quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
    const int s_first = quick ? nS - 4 : 0;
    const int m_last = quick ? 7 : nM;

    if (!do_bench) {
        test_cpu_quantizer();
        printf("cpu linear\n");
        for (int si = s_first; si < nS; si++) {
            problem_t p = make_problem(shapes[si].N, shapes[si].K, shapes[si].G);
            for (int mi = 0; mi < m_last; mi++) test_cpu_linear(&p, Ms[mi]);
            free_problem(&p);
        }
    }
    if (quick) {
        printf("\n=== %d/%d checks passed (quick, CPU only) ===\n",
               checks - failures, checks);
        return failures ? 1 : 0;
    }

    if (vv_dev_device_count() <= 0) {
        printf("no GPU: device tests SKIPPED\n");
    } else {
        vv_dev_set_device(0);
        void* st = NULL;
        vv_dev_stream_create(&st);
        if (do_bench) {
            bench(st);
        } else {
            test_gpu_quantizers(st);
            test_gpu_col_absmax(st);
            printf("gpu linear\n");
            for (int si = 0; si < nS; si++) {
                problem_t p = make_problem(shapes[si].N, shapes[si].K,
                                           shapes[si].G);
                dev_problem_t d = upload(&p);
                for (int mi = 0; mi < nM; mi++)
                    test_gpu_linear(&p, &d, Ms[mi], st);
                free_dev(&d);
                free_problem(&p);
            }
        }
        vv_dev_stream_destroy(st);
    }

    printf("\n=== %d/%d checks passed ===\n", checks - failures, checks);
    return failures ? 1 : 0;
}
