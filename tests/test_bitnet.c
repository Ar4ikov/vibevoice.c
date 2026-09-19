/**
 * @file test_bitnet.c
 * @brief BitNet integer kernels against scalar references, exactly.
 *
 * Everything under test is an integer product followed by a fixed sequence of
 * FP32 operations, so the comparisons here are `==`, not tolerances: every
 * ISA the CPU offers (scalar, AVX2, AVX-VNNI, AVX512-VNNI, NEON) and the GPU
 * kernels must produce the same int32 accumulators and the same FP32 bits.
 *
 * The references are deliberately written the slow, obvious way (unpack to
 * {-1, 0, 1}, multiply, add), and the activation quantizer's reference is a
 * transcription of ggml's quantize_row_i8_s from the VibeASR.cpp fork.
 *
 * Run it under qemu-aarch64 to cover NEON (with and without +dotprod).
 */

#include "vibevoice/bitnet.h"
#include "vibevoice/device.h"
#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static uint32_t rng = 12345u;

static uint32_t urand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static float frand(void) { return (float)(urand() >> 8) / 16777216.0f * 2.0f - 1.0f; }

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("  FAIL: " __VA_ARGS__);               \
            printf("\n");                                 \
            failures++;                                   \
        }                                                 \
    } while (0)

/* ─── References ────────────────────────────────────────────────────────── */

/* ggml fork, ggml-quants.c quantize_row_i8_s, transcribed. The reference
 * build contracts nearest_int(x[i] * s) into fma(x, s, 1.5 * 2^23), so the
 * product is rounded once, together with the magic constant. */
static inline int ref_nearest_int_mul(float x, float s) {
    float val = fmaf(x, s, 12582912.f);
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
static void ref_quantize_row_i8_s(const float* x, int8_t* y, int n, float* s_out,
                                  int32_t* sum_out) {
    double min = 0.00001;
    double max = min;
    for (int i = 0; i < n; ++i) max = fmax(max, (double)fabs((double)x[i]));
    float s = 127 / max;
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) {
        int v = ref_nearest_int_mul(x[i], s);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        sum += v;
        y[i] = (int8_t)v;
    }
    *s_out = s;
    *sum_out = sum;
}

/* ggml fork, ggml-lm-mad.cpp quantize_i2_s (x86 ACT_PARALLEL branch) */
static void ref_quantize_i2_s(const float* src, uint8_t* dst, int64_t n, float* scale) {
    double max = 0;
    for (int64_t i = 0; i < n; ++i) max = fmax(max, (double)fabs((double)src[i]));
    memset(dst, 0, (size_t)(n / 4));
    for (int64_t i = 0; i < n / 128; i++)
        for (int j = 0; j < 128; j++) {
            const double v = src[i * 128 + j];
            uint8_t q = fabs(v) < 1e-6 ? 1 : (v * max > 0 ? 2 : 0);
            dst[i * 32 + j % 32] |= (uint8_t)(q << (6 - 2 * (j / 32)));
        }
    *scale = (float)max;
}

static int32_t ref_tern_dot(const uint8_t* codes, const int8_t* x, int K) {
    int8_t* t = (int8_t*)malloc((size_t)K);
    vv_ternary_unpack_row(codes, K, t);
    int32_t s = 0;
    for (int k = 0; k < K; k++) s += (int32_t)t[k] * x[k];
    free(t);
    return s;
}

/* ─── ISA iteration ─────────────────────────────────────────────────────── */

static const struct { int id; const char* name; } ISAS[] = {
    { 1, "scalar" }, { 2, "AVX2" }, { 3, "AVX-VNNI" }, { 4, "AVX512-VNNI" }, { 5, "NEON" },
};

/* ─── CPU tests ─────────────────────────────────────────────────────────── */

static void test_pack_roundtrip(void) {
    printf("pack/unpack round trip\n");
    enum { N = 7, K = 384 };
    int8_t* t = malloc(N * K), *u = malloc(K);
    uint8_t* c = malloc(N * K / 4);
    for (int i = 0; i < N * K; i++) t[i] = (int8_t)((int)(urand() % 3) - 1);
    CHECK(vv_ternary_pack(t, N, K, c) == VV_OK, "pack");
    for (int n = 0; n < N; n++) {
        vv_ternary_unpack_row(c + n * K / 4, K, u);
        CHECK(memcmp(u, t + n * K, K) == 0, "row %d differs", n);
    }
    free(t); free(u); free(c);
}

static void test_ternarize(void) {
    printf("ternarize == converter + quantize_i2_s\n");
    enum { N = 64, K = 1536 };
    float* w = malloc(sizeof(float) * N * K);
    float* wq = malloc(sizeof(float) * N * K);
    uint8_t* got = malloc(N * K / 4);
    uint8_t* ref = malloc(N * K / 4);
    for (int i = 0; i < N * K; i++) w[i] = frand() * 0.05f;
    w[5] = 0.0f; /* exact zeros must become code 1 */

    float scale = 0.0f;
    CHECK(vv_ternarize_f32(w, N, K, got, &scale) == VV_OK, "ternarize");

    /* the converter, spelled out: s = 1/mean|w|; w' = clamp(round(w s)) / s */
    double acc = 0;
    for (int i = 0; i < N * K; i++) acc += fabs((double)w[i]);
    float mean = (float)(acc / (N * K));
    const float s = 1.0f / (mean < 1e-5f ? 1e-5f : mean);
    for (int i = 0; i < N * K; i++) {
        float r = nearbyintf(w[i] * s);
        r = r > 1 ? 1 : (r < -1 ? -1 : r);
        wq[i] = r / s;
    }
    float ref_scale = 0.0f;
    ref_quantize_i2_s(wq, ref, (int64_t)N * K, &ref_scale);
    CHECK(memcmp(got, ref, N * K / 4) == 0, "codes differ from the reference");
    CHECK(scale == ref_scale, "scale %.9g vs %.9g", scale, ref_scale);

    int zeros = 0, pos = 0, neg = 0;
    int8_t* t = malloc(K);
    for (int n = 0; n < N; n++) {
        vv_ternary_unpack_row(got + n * K / 4, K, t);
        for (int k = 0; k < K; k++) {
            zeros += t[k] == 0; pos += t[k] > 0; neg += t[k] < 0;
            /* dequantized value equals the fake-quant value exactly */
            if ((float)t[k] * scale != wq[n * K + k] &&
                !(t[k] == 0 && wq[n * K + k] == 0.0f)) {
                CHECK(0, "dequant mismatch at %d,%d", n, k);
                n = N;
                break;
            }
        }
    }
    printf("  codes: %d neg / %d zero / %d pos, scale %.6g\n", neg, zeros, pos, scale);
    free(t); free(w); free(wq); free(got); free(ref);
}

static void test_act_quant(void) {
    printf("activation quantizer == ggml quantize_row_i8_s\n");
    enum { M = 9, K = 1536 };
    float* x = malloc(sizeof(float) * M * K);
    int8_t* q = malloc(M * K), *rq = malloc(K);
    float sc[M];
    int32_t sum[M];
    for (int i = 0; i < M * K; i++) x[i] = frand() * (i % 7 == 0 ? 40.0f : 3.0f);
    /* rows with ties (exact .5 after scaling), all zeros, and a spike */
    for (int k = 0; k < K; k++) x[1 * K + k] = (float)((k % 255) - 127) * 0.5f;
    for (int k = 0; k < K; k++) x[2 * K + k] = 0.0f;
    x[3 * K + 17] = 1e6f;
    /* row 4: values whose product with s rounds to k + 0.5 in FP32 while the
     * exact product does not -- the fused and the unfused quantizer differ */
    {
        float* r = x + 4 * K;
        r[0] = 3.1f;
        const float s = (float)(127.0 / 3.1);
        int hits = 0;
        for (int k = 1; k < K; k++) {
            const float t = ((float)(k % 200) - 100.0f + 0.5f) / s;
            uint32_t u;
            memcpy(&u, &t, 4);
            u += (uint32_t)(k % 5) - 2u;          /* a few ulps either side */
            memcpy(&r[k], &u, 4);
            const float p = r[k] * s;
            if (p - floorf(p) == 0.5f && fmaf(r[k], s, -p) != 0.0f) hits++;
        }
        CHECK(hits > 0, "row 4 has no fused-rounding case");
    }
    for (int pass = 0; pass < 2; pass++) {
        /* the scalar fmaf path, then the best ISA's (AVX2 + FMA on x86) */
        vv_bitnet_cpu_force_isa(pass == 0 ? 1 : 0);
        memset(q, 0, (size_t)M * K);
        CHECK(vv_act_quant_i8_cpu(x, M, K, q, sc, sum) == VV_OK, "quant");
        for (int m = 0; m < M; m++) {
            float rs;
            int32_t rsum;
            ref_quantize_row_i8_s(x + m * K, rq, K, &rs, &rsum);
            CHECK(memcmp(rq, q + m * K, K) == 0, "pass %d row %d codes", pass, m);
            CHECK(rs == sc[m], "row %d scale %.9g vs %.9g", m, sc[m], rs);
            CHECK(rsum == sum[m], "pass %d row %d sum", pass, m);
        }
    }
    vv_bitnet_cpu_force_isa(0);
    free(x); free(q); free(rq);
}

static void test_ternary_gemm_isas(void) {
    printf("ternary x int8 exact on every ISA\n");
    static const int KS[] = { 128, 1536, 2048, 2176, 4224 };
    static const int MS[] = { 1, 3, 4, 5, 64, 70 };
    enum { N = 37 };
    const int Kmax = 4224, Mmax = 70;
    uint8_t* codes = malloc((size_t)N * Kmax / 4);
    int8_t* x = malloc((size_t)Mmax * Kmax);
    int32_t* ref = malloc(sizeof(int32_t) * Mmax * N);
    int32_t* got = malloc(sizeof(int32_t) * Mmax * N);
    float* y = malloc(sizeof(float) * Mmax * N);
    float* bias = malloc(sizeof(float) * N);
    float* sc = malloc(sizeof(float) * Mmax);
    for (int n = 0; n < N; n++) bias[n] = frand();
    for (int m = 0; m < Mmax; m++) sc[m] = 1.0f + (float)m * 0.37f;

    for (size_t ki = 0; ki < sizeof(KS) / sizeof(KS[0]); ki++) {
        const int K = KS[ki];
        for (int pattern = 0; pattern < 3; pattern++) {
            /* 0: random; 1: all +1 weights x +127 (int16 lane at its max
             * positive after 16 blocks); 2: all +1 x -128 (its minimum) */
            for (int i = 0; i < N * K / 4; i++)
                codes[i] = pattern ? 0xAA : (uint8_t)urand();
            /* code 3 does not occur in real files; keep codes in 0..2 */
            if (!pattern)
                for (int i = 0; i < N * K / 4; i++) {
                    uint8_t b = codes[i], o = 0;
                    for (int f = 0; f < 4; f++) {
                        uint8_t c = (b >> (2 * f)) & 3;
                        if (c == 3) c = urand() % 3;
                        o |= (uint8_t)(c << (2 * f));
                    }
                    codes[i] = o;
                }
            for (int i = 0; i < Mmax * K; i++)
                x[i] = pattern == 0 ? (int8_t)(urand() & 0xFF) : (pattern == 1 ? 127 : -128);
            for (size_t mi = 0; mi < sizeof(MS) / sizeof(MS[0]); mi++) {
                const int M = MS[mi];
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                        ref[m * N + n] = ref_tern_dot(codes + (size_t)n * K / 4, x + (size_t)m * K, K);
                for (size_t is = 0; is < sizeof(ISAS) / sizeof(ISAS[0]); is++) {
                    if (vv_bitnet_cpu_force_isa(ISAS[is].id) != VV_OK) continue;
                    memset(got, 0, sizeof(int32_t) * M * N);
                    CHECK(vv_ternary_gemm_i32_cpu(x, codes, got, M, N, K) == VV_OK, "gemm");
                    CHECK(memcmp(got, ref, sizeof(int32_t) * M * N) == 0,
                          "%s K=%d M=%d pattern %d", ISAS[is].name, K, M, pattern);
                    /* FP32 epilogue, same two operations in the same order */
                    const float ws = 0.0421f;
                    CHECK(vv_ternary_linear_cpu(x, sc, codes, ws, bias, y, M, N, K) == VV_OK, "linear");
                    int bad = 0;
                    for (int m = 0; m < M && !bad; m++)
                        for (int n = 0; n < N; n++) {
                            const float want = (float)ref[m * N + n] / sc[m] * ws + bias[n];
                            if (memcmp(&want, &y[m * N + n], 4) != 0) { bad = 1; break; }
                        }
                    CHECK(!bad, "%s linear epilogue K=%d M=%d", ISAS[is].name, K, M);
                }
            }
        }
    }
    vv_bitnet_cpu_force_isa(0);
    free(codes); free(x); free(ref); free(got); free(y); free(bias); free(sc);
}

static void test_i8_gemm_isas(void) {
    printf("int8 x int8 exact on every ISA\n");
    static const int KS[] = { 8, 32, 40, 100, 1536, 2048 };
    static const int MS[] = { 1, 2, 3, 9 };
    enum { N = 13 };
    const int Kmax = 2048, Mmax = 9;
    int8_t* w = malloc((size_t)N * Kmax);
    int8_t* a = malloc((size_t)Mmax * Kmax);
    int32_t* ref = malloc(sizeof(int32_t) * Mmax * N);
    int32_t* got = malloc(sizeof(int32_t) * Mmax * N);
    float* y = malloc(sizeof(float) * Mmax * N);
    float bias[N];
    for (int n = 0; n < N; n++) bias[n] = frand() * 3;

    for (size_t ki = 0; ki < sizeof(KS) / sizeof(KS[0]); ki++) {
        const int K = KS[ki];
        for (int pattern = 0; pattern < 2; pattern++) {
            for (int i = 0; i < N * K; i++) {
                int v = (int)(urand() % 255) - 127; /* weights stay in [-127, 127] */
                w[i] = (int8_t)(pattern ? (i & 1 ? -127 : 127) : v);
            }
            for (int i = 0; i < Mmax * K; i++)
                a[i] = (int8_t)(pattern ? (i & 1 ? 127 : -128) : (int)(urand() & 0xFF) - 128);
            for (size_t mi = 0; mi < sizeof(MS) / sizeof(MS[0]); mi++) {
                const int M = MS[mi];
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++) {
                        int32_t s = 0;
                        for (int k = 0; k < K; k++) s += (int32_t)a[m * K + k] * w[n * K + k];
                        ref[m * N + n] = s;
                    }
                for (size_t is = 0; is < sizeof(ISAS) / sizeof(ISAS[0]); is++) {
                    if (vv_bitnet_cpu_force_isa(ISAS[is].id) != VV_OK) continue;
                    CHECK(vv_i8_gemm_i32_cpu(a, w, got, M, N, K) == VV_OK, "gemm");
                    CHECK(memcmp(got, ref, sizeof(int32_t) * M * N) == 0,
                          "%s K=%d M=%d pattern %d", ISAS[is].name, K, M, pattern);
                    float amax = -1.0f;
                    const float as = 3.17f, ws = 0.0123f;
                    CHECK(vv_i8_linear_cpu(a, as, w, ws, bias, y, &amax, M, N, K) == VV_OK, "linear");
                    float want_max = 0.0f;
                    int bad = 0;
                    const float d = ws / as;
                    for (int i = 0; i < M * N; i++) {
                        const float want = (float)ref[i] * d + bias[i % N];
                        if (memcmp(&want, &y[i], 4) != 0) bad = 1;
                        if (fabsf(want) > want_max) want_max = fabsf(want);
                    }
                    CHECK(!bad, "%s linear epilogue K=%d M=%d", ISAS[is].name, K, M);
                    CHECK(amax == want_max, "%s absmax", ISAS[is].name);
                }
            }
        }
    }
    vv_bitnet_cpu_force_isa(0);
    free(w); free(a); free(ref); free(got); free(y);
}

static void test_head_argmax(void) {
    printf("int8 head argmax\n");
    enum { V = 1003, K = 256 };
    int8_t* w = malloc(V * K);
    float* ws = malloc(sizeof(float) * V);
    int8_t q[K];
    for (int i = 0; i < V * K; i++) w[i] = (int8_t)((int)(urand() % 255) - 127);
    for (int i = 0; i < K; i++) q[i] = (int8_t)((int)(urand() % 255) - 127);
    for (int n = 0; n < V; n++) ws[n] = 0.01f + frand() * 0.001f;
    /* plant a tie between two rows at the top (w = q maximizes the dot
     * product); the lower index must win */
    for (int k = 0; k < K; k++) w[900 * K + k] = w[700 * K + k] = q[k] < -127 ? -127 : q[k];
    ws[700] = ws[900] = 1.0f;
    const float s = 2.5f;
    for (size_t is = 0; is < sizeof(ISAS) / sizeof(ISAS[0]); is++) {
        if (vv_bitnet_cpu_force_isa(ISAS[is].id) != VV_OK) continue;
        float best = -FLT_MAX;
        int bi = -1;
        for (int n = 0; n < V; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)w[n * K + k] * q[k];
            const float v = (float)acc * ws[n] / s;
            if (v > best) { best = v; bi = n; }
        }
        int32_t tok = -1;
        float val = 0;
        CHECK(vv_i8_head_argmax_cpu(q, s, w, ws, V, K, &tok, &val) == VV_OK, "head");
        CHECK(tok == bi && val == best && tok == 700, "%s: token %d (%g) vs %d (%g)", ISAS[is].name, tok, val, bi, best);
    }
    vv_bitnet_cpu_force_isa(0);
    free(w); free(ws);
}

/* ggml_vec_dot_f16 (x86 F16C: 4 x 8 FMA lanes over 32, then the reduction)
 * of an F16 row with an activation rounded to F16, transcribed */
static float ref_vec_dot_f16(const uint16_t* w, const float* x, int K) {
    float acc[4][8];
    memset(acc, 0, sizeof(acc));
    for (int i = 0; i < K; i += 32)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 8; l++) {
                const float xh = vv_half_to_float(vv_float_to_half_rne(x[i + 8 * j + l]));
                acc[j][l] = fmaf(vv_half_to_float(w[i + 8 * j + l]), xh, acc[j][l]);
            }
    float s0[8], t0[4];
    for (int l = 0; l < 8; l++) s0[l] = (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]);
    for (int l = 0; l < 4; l++) t0[l] = s0[l] + s0[l + 4];
    return (t0[0] + t0[1]) + (t0[2] + t0[3]);
}

static void test_f16_head(void) {
    printf("F16 head: full scan == ggml, int8 filter == full scan\n");
    enum { V = 5003, K = 1536 };
    uint16_t* w = malloc(sizeof(uint16_t) * V * K);
    int8_t* q = malloc((size_t)V * K);
    float* sc = malloc(sizeof(float) * V);
    float* bd = malloc(sizeof(float) * 3 * V);
    float* x = malloc(sizeof(float) * K);
    const size_t sb = vv_bitnet_head_filter_scratch(V, K);
    void* scratch = malloc(sb);
    for (size_t i = 0; i < (size_t)V * K; i++) w[i] = vv_float_to_half(frand() * 0.05f);
    /* a few rows close to each other at the top, and an exact tie */
    for (int k = 0; k < K; k++) {
        w[(size_t)4000 * K + k] = vv_float_to_half(0.04f * (float)((k % 13) - 6) / 6.0f);
        w[(size_t)77 * K + k] = w[(size_t)4000 * K + k];            /* tie, lower id */
        w[(size_t)3000 * K + k] = vv_float_to_half(0.0399f * (float)((k % 13) - 6) / 6.0f);
    }
    CHECK(vv_bitnet_head_filter_build(w, V, K, q, sc, bd) == VV_OK, "filter build");
    for (int trial = 0; trial < 4; trial++) {
        for (int k = 0; k < K; k++)
            x[k] = trial < 2 ? (float)((k % 13) - 6) * (1.0f + 0.01f * frand())
                             : frand() * 3.0f;
        int32_t rt = -1;
        float rv = -INFINITY;
        for (int r = 0; r < V; r++) {
            const float v = ref_vec_dot_f16(w + (size_t)r * K, x, K);
            if (rt < 0 || v > rv) { rv = v; rt = r; }
        }
        int32_t t1 = -2, t2 = -3;
        float v1 = 0, v2 = 0;
        int n = 0;
        CHECK(vv_bitnet_head_f16_argmax_cpu(x, w, V, K, &t1, &v1) == VV_OK, "scan");
        CHECK(vv_bitnet_head_filtered_argmax_cpu(x, w, q, sc, bd, V, K, scratch, sb,
                                                 &t2, &v2, &n) == VV_OK, "filter");
        CHECK(t1 == rt && v1 == rv, "trial %d scan %d %.9g vs ggml %d %.9g", trial,
              t1, v1, rt, rv);
        CHECK(t2 == rt && v2 == rv, "trial %d filter %d %.9g vs ggml %d %.9g", trial,
              t2, v2, rt, rv);
        CHECK(n > 0 && n < V / 4, "trial %d: %d rows scored exactly", trial, n);
        if (trial < 2) CHECK(rt == 77, "trial %d tie resolved to %d, not 77", trial, rt);
    }
    free(w); free(q); free(sc); free(bd); free(x); free(scratch);
}

static void test_requant(void) {
    printf("requantize\n");
    enum { N = 1000 };
    float y[N];
    int8_t q[N];
    for (int i = 0; i < N; i++) y[i] = frand() * 7.0f;
    y[3] = 7.0f;
    y[4] = -3.5f; /* exact tie after scaling by 127/7 = 18.142..: not a tie */
    float sc = 0;
    for (int relu = 0; relu < 2; relu++) {
        CHECK(vv_i8s_requant_cpu(y, N, 7.0f, relu, q, &sc) == VV_OK, "requant");
        const float inv = 127.0f / 7.0f;
        int bad = 0;
        for (int i = 0; i < N; i++) {
            float v = y[i] * inv;
            const float lo = relu ? 0.0f : -127.0f;
            v = v < lo ? lo : (v > 127 ? 127 : v);
            if (q[i] != (int8_t)nearbyintf(v)) bad = 1;
        }
        CHECK(!bad && sc == inv, "relu=%d", relu);
    }
}

/* ─── GPU tests ─────────────────────────────────────────────────────────── */

#ifdef VV_HAS_ACCEL
static void* ST = NULL;

static void* dup_to_dev(const void* h, size_t n) {
    void* d = NULL;
    if (vv_dev_alloc(&d, n) != VV_OK) return NULL;
    vv_dev_memcpy_h2d(d, h, n, ST);
    return d;
}

static void test_gpu(void) {
    printf("GPU kernels vs CPU, bit for bit\n");
    /* ── activation quantizer, f32 and f16 input ── */
    {
        enum { M = 5, K = 1536 };
        float* x = malloc(sizeof(float) * M * K);
        uint16_t* xh = malloc(2 * M * K);
        for (int i = 0; i < M * K; i++) {
            xh[i] = vv_float_to_half(frand() * 9.0f);
            x[i] = vv_half_to_float(xh[i]);
        }
        int8_t q[M * K], gq[M * K];
        float sc[M], gsc[M];
        int32_t sum[M], gsum[M];
        vv_act_quant_i8_cpu(x, M, K, q, sc, sum);
        for (int f16 = 0; f16 < 2; f16++) {
            void* dx = dup_to_dev(f16 ? (void*)xh : (void*)x, (f16 ? 2 : 4) * (size_t)M * K);
            void *dq = NULL, *ds = NULL, *dsum = NULL;
            vv_dev_alloc(&dq, M * K);
            vv_dev_alloc(&ds, 4 * M);
            vv_dev_alloc(&dsum, 4 * M);
            CHECK(vv_act_quant_i8_dev(dx, f16, M, K, dq, ds, dsum, ST) == VV_OK, "act quant dev");
            vv_dev_memcpy_d2h(gq, dq, M * K, ST);
            vv_dev_memcpy_d2h(gsc, ds, 4 * M, ST);
            vv_dev_memcpy_d2h(gsum, dsum, 4 * M, ST);
            vv_dev_stream_sync(ST);
            CHECK(memcmp(q, gq, M * K) == 0 && memcmp(sc, gsc, 4 * M) == 0 &&
                  memcmp(sum, gsum, 4 * M) == 0, "act quant f16=%d", f16);
            vv_dev_free(dx); vv_dev_free(dq); vv_dev_free(ds); vv_dev_free(dsum);
        }
        free(x); free(xh);
    }

    /* ── ternary GEMV / GEMM ── */
    static const int MS[] = { 1, 5, 8, 9, 64, 131 };
    static const int NS[] = { 256, 1536, 72 };
    static const int KS[] = { 1536, 8960, 128 };
    for (size_t si = 0; si < 3; si++) {
        const int N = NS[si], K = KS[si];
        const int Mmax = 131;
        uint8_t* codes = malloc((size_t)N * K / 4);
        int8_t* x = malloc((size_t)Mmax * K);
        float* w = malloc(sizeof(float) * N * K);
        for (int i = 0; i < N * K; i++) w[i] = frand();
        float ws;
        vv_ternarize_f32(w, N, K, codes, &ws);
        float* xf = malloc(sizeof(float) * Mmax * K);
        for (int i = 0; i < Mmax * K; i++) xf[i] = frand() * 4;
        float* sc = malloc(sizeof(float) * Mmax);
        int32_t* sum = malloc(sizeof(int32_t) * Mmax);
        vv_act_quant_i8_cpu(xf, Mmax, K, x, sc, sum);
        float* bias = malloc(sizeof(float) * N);
        for (int n = 0; n < N; n++) bias[n] = frand();
        int32_t* acc = malloc(sizeof(int32_t) * Mmax * N);
        int32_t* gacc = malloc(sizeof(int32_t) * Mmax * N);
        float* y = malloc(sizeof(float) * Mmax * N);
        float* gy = malloc(sizeof(float) * Mmax * N);
        void* dc = dup_to_dev(codes, (size_t)N * K / 4);
        void* dx = dup_to_dev(x, (size_t)Mmax * K);
        void* ds = dup_to_dev(sc, 4 * (size_t)Mmax);
        void* dsum = dup_to_dev(sum, 4 * (size_t)Mmax);
        void* db = dup_to_dev(bias, 4 * (size_t)N);
        void *dacc = NULL, *dy = NULL;
        vv_dev_alloc(&dacc, 4 * (size_t)Mmax * N);
        vv_dev_alloc(&dy, 4 * (size_t)Mmax * N);
        for (size_t mi = 0; mi < sizeof(MS) / sizeof(MS[0]); mi++) {
            const int M = MS[mi];
            vv_ternary_gemm_i32_cpu(x, codes, acc, M, N, K);
            vv_ternary_linear_cpu(x, sc, codes, ws, bias, y, M, N, K);
            CHECK(vv_ternary_gemm_dev(dx, dsum, ds, dc, ws, db, dacc, dy, 0, M, N, K, ST) == VV_OK,
                  "ternary dev");
            vv_dev_memcpy_d2h(gacc, dacc, 4 * (size_t)M * N, ST);
            vv_dev_memcpy_d2h(gy, dy, 4 * (size_t)M * N, ST);
            vv_dev_stream_sync(ST);
            CHECK(memcmp(acc, gacc, 4 * (size_t)M * N) == 0, "ternary acc N=%d K=%d M=%d", N, K, M);
            CHECK(memcmp(y, gy, 4 * (size_t)M * N) == 0, "ternary y N=%d K=%d M=%d", N, K, M);
        }
        vv_dev_free(dc); vv_dev_free(dx); vv_dev_free(ds); vv_dev_free(dsum);
        vv_dev_free(db); vv_dev_free(dacc); vv_dev_free(dy);
        free(codes); free(x); free(w); free(xf); free(sc); free(sum); free(bias);
        free(acc); free(gacc); free(y); free(gy);
    }

    /* ── int8 GEMM + requant (aligned and ragged K) ── */
    static const int IK[] = { 1536, 8, 200 };
    static const int IN[] = { 128, 32, 70 };
    for (size_t si = 0; si < 3; si++) {
        const int N = IN[si], K = IK[si];
        static const int IM[] = { 1, 7, 9, 300 };
        const int Mmax = 300;
        int8_t* a = malloc((size_t)Mmax * K);
        int8_t* w = malloc((size_t)N * K);
        for (int i = 0; i < Mmax * K; i++) a[i] = (int8_t)((int)(urand() % 255) - 127);
        for (int i = 0; i < N * K; i++) w[i] = (int8_t)((int)(urand() % 255) - 127);
        float bias[128];
        for (int n = 0; n < N; n++) bias[n] = frand();
        const float as = 12.5f, ws = 0.003f;
        int32_t* acc = malloc(4 * (size_t)Mmax * N), *gacc = malloc(4 * (size_t)Mmax * N);
        float* y = malloc(4 * (size_t)Mmax * N), *gy = malloc(4 * (size_t)Mmax * N);
        int8_t* q = malloc((size_t)Mmax * N), *gq = malloc((size_t)Mmax * N);
        void* da = dup_to_dev(a, (size_t)Mmax * K);
        void* dw = dup_to_dev(w, (size_t)N * K);
        void* das = dup_to_dev(&as, 4);
        void* db = dup_to_dev(bias, 4 * (size_t)N);
        void *dacc = NULL, *dy = NULL, *dmax = NULL, *dq = NULL, *dos = NULL;
        vv_dev_alloc(&dacc, 4 * (size_t)Mmax * N);
        vv_dev_alloc(&dy, 4 * (size_t)Mmax * N);
        vv_dev_alloc(&dmax, 4);
        vv_dev_alloc(&dq, (size_t)Mmax * N);
        vv_dev_alloc(&dos, 4);
        for (size_t mi = 0; mi < 4; mi++) {
            const int M = IM[mi];
            float amax = 0, gmax = 0, osc = 0, gosc = 0;
            vv_i8_gemm_i32_cpu(a, w, acc, M, N, K);
            vv_i8_linear_cpu(a, as, w, ws, bias, y, &amax, M, N, K);
            vv_i8s_requant_cpu(y, (int64_t)M * N, amax, 1, q, &osc);
            vv_dev_memset_async(dmax, 0, 4, ST);
            CHECK(vv_i8_gemm_dev(da, das, dw, ws, db, dacc, dy, dmax, M, N, K, ST) == VV_OK, "i8 dev");
            CHECK(vv_i8s_requant_dev(dy, (int64_t)M * N, dmax, 1, dq, dos, ST) == VV_OK, "requant dev");
            vv_dev_memcpy_d2h(gacc, dacc, 4 * (size_t)M * N, ST);
            vv_dev_memcpy_d2h(gy, dy, 4 * (size_t)M * N, ST);
            vv_dev_memcpy_d2h(&gmax, dmax, 4, ST);
            vv_dev_memcpy_d2h(gq, dq, (size_t)M * N, ST);
            vv_dev_memcpy_d2h(&gosc, dos, 4, ST);
            vv_dev_stream_sync(ST);
            CHECK(memcmp(acc, gacc, 4 * (size_t)M * N) == 0, "i8 acc N=%d K=%d M=%d", N, K, M);
            CHECK(memcmp(y, gy, 4 * (size_t)M * N) == 0, "i8 y N=%d K=%d M=%d", N, K, M);
            CHECK(amax == gmax, "i8 absmax %g vs %g", amax, gmax);
            CHECK(memcmp(q, gq, (size_t)M * N) == 0 && osc == gosc, "requant N=%d K=%d M=%d", N, K, M);
        }
        vv_dev_free(da); vv_dev_free(dw); vv_dev_free(das); vv_dev_free(db);
        vv_dev_free(dacc); vv_dev_free(dy); vv_dev_free(dmax); vv_dev_free(dq); vv_dev_free(dos);
        free(a); free(w); free(acc); free(gacc); free(y); free(gy); free(q); free(gq);
    }

    /* ── int8 head argmax ── */
    {
        enum { V = 151936, K = 1536 };
        int8_t* w = malloc((size_t)V * K);
        float* ws = malloc(4 * (size_t)V);
        int8_t q[K];
        for (size_t i = 0; i < (size_t)V * K; i++) w[i] = (int8_t)((int)(urand() % 255) - 127);
        for (int i = 0; i < K; i++) q[i] = (int8_t)((int)(urand() % 255) - 127);
        for (int n = 0; n < V; n++) ws[n] = 0.01f + frand() * 0.001f;
        for (int k = 0; k < K; k++) w[(size_t)1234 * K + k] = w[(size_t)150000 * K + k] = q[k];
        ws[1234] = ws[150000] = 1.0f;
        const float s = 3.0f;
        int32_t tok;
        float val;
        vv_i8_head_argmax_cpu(q, s, w, ws, V, K, &tok, &val);
        void* dw = dup_to_dev(w, (size_t)V * K);
        void* dws = dup_to_dev(ws, 4 * (size_t)V);
        void* dq = dup_to_dev(q, K);
        void* ds = dup_to_dev(&s, 4);
        void *dt = NULL, *dv = NULL, *scr = NULL;
        vv_dev_alloc(&dt, 4);
        vv_dev_alloc(&dv, 4);
        vv_dev_alloc(&scr, vv_i8_head_argmax_scratch_bytes(V));
        CHECK(vv_i8_head_argmax_dev(dq, ds, dw, dws, V, K, dt, dv, scr, ST) == VV_OK, "head dev");
        int32_t gt;
        float gv;
        vv_dev_memcpy_d2h(&gt, dt, 4, ST);
        vv_dev_memcpy_d2h(&gv, dv, 4, ST);
        vv_dev_stream_sync(ST);
        CHECK(gt == tok && gv == val && tok == 1234, "head: gpu %d (%g) cpu %d (%g)", gt, gv, tok, val);
        vv_dev_free(dw); vv_dev_free(dws); vv_dev_free(dq); vv_dev_free(ds);
        vv_dev_free(dt); vv_dev_free(dv); vv_dev_free(scr);
        free(w); free(ws);
    }
}
#endif

int main(void) {
    printf("bitnet kernels: %s\n", vv_bitnet_cpu_isa());
    test_pack_roundtrip();
    test_ternarize();
    test_act_quant();
    test_ternary_gemm_isas();
    test_i8_gemm_isas();
    test_head_argmax();
    test_f16_head();
    test_requant();
#ifdef VV_HAS_ACCEL
    if (vv_dev_device_count() > 0) test_gpu();
    else printf("SKIP GPU: no device\n");
#else
    printf("SKIP GPU: built without an accelerator backend\n");
#endif
    if (failures) {
        printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("all bitnet kernel checks passed\n");
    return 0;
}
