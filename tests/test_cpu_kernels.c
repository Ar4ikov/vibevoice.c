/**
 * @file test_cpu_kernels.c
 * @brief Check every CPU kernel against a plain scalar reference.
 *
 * The CPU path has a SIMD specialisation chosen at runtime, so "it works"
 * has to be established per architecture, not once. Running this binary
 * under qemu-aarch64 is what makes the Apple Silicon claim testable from a
 * machine that has no Apple Silicon: same sources, same test, ARM code path.
 */

#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 987654321u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;
}

static const float NF4_REF[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f
};

static int failures = 0;

/** @brief Compare against a tolerance scaled by the magnitude involved. */
static void check(const char* what, const float* got, const float* ref,
                  int n, double tol) {
    double max_abs = 0.0, mag = 0.0;
    int worst = 0;
    for (int i = 0; i < n; i++) {
        const double d = fabs((double)got[i] - (double)ref[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
        if (fabs((double)ref[i]) > mag) mag = fabs((double)ref[i]);
    }
    const double rel = max_abs / (mag + 1e-9);
    const int ok = rel <= tol;
    printf("  %-22s max|err|=%.3e rel=%.2e %s\n", what, max_abs, rel,
           ok ? "ok" : "FAIL");
    if (!ok) {
        printf("      worst at %d: got %.6f want %.6f\n",
               worst, got[worst], ref[worst]);
        failures++;
    }
}

int main(void) {
    printf("cpu kernels: %s, %d thread(s)\n",
           vv_cpu_simd_name(), vv_cpu_threads());

    /*
     * The shapes are chosen to land on every seam in the packed GEMM, since
     * a GEMM that is right on round numbers is right on nothing in
     * particular. N = 77 leaves a ragged tail past the last 16-column panel;
     * K = 640 gives an undersized last k-block; and the M list crosses the
     * threshold below which packing is skipped, the 6-row panel boundary,
     * and the row count above which the activation block is split.
     */
    enum { M = 400, N = 77, K = 640, GROUP = 128 };
    static const int MS[] = { 1, 3, 4, 7, 13, M };

    float* x = malloc(sizeof(float) * M * K);
    float* got = malloc(sizeof(float) * M * N);
    float* ref = malloc(sizeof(float) * M * N);
    uint8_t* packed = malloc((size_t)N * K / 2);
    uint16_t* scales = malloc(sizeof(uint16_t) * N * K / 64);
    uint16_t* gscales = malloc(sizeof(uint16_t) * N * (K / GROUP));
    uint16_t* gmins = malloc(sizeof(uint16_t) * N * (K / GROUP));
    uint16_t* wf16 = malloc(sizeof(uint16_t) * N * K);
    uint16_t* bias = malloc(sizeof(uint16_t) * N);
    if (!x || !got || !ref || !packed || !scales || !gscales || !gmins ||
        !wf16 || !bias) return 1;

    for (int i = 0; i < M * K; i++) x[i] = frand();
    for (size_t i = 0; i < (size_t)N * K / 2; i++)
        packed[i] = (uint8_t)((rng = rng * 1103515245u + 12345u) >> 16);
    for (int i = 0; i < N * K / 64; i++)
        scales[i] = vv_float_to_half(0.02f + 0.3f * fabsf(frand()));
    for (int i = 0; i < N * (K / GROUP); i++) {
        gscales[i] = vv_float_to_half(0.01f + 0.1f * fabsf(frand()));
        gmins[i] = vv_float_to_half(-0.5f * fabsf(frand()));
    }
    for (int i = 0; i < N * K; i++) wf16[i] = vv_float_to_half(frand());
    for (int i = 0; i < N; i++) bias[i] = vv_float_to_half(frand());

    for (size_t s = 0; s < sizeof(MS) / sizeof(MS[0]); s++) {
        const int m_rows = MS[s];
        char what[48];

        /* ── NF4 GEMM ── */
        for (int m = 0; m < m_rows; m++)
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    const uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
                    const int code = (k & 1) ? (byte & 0xF) : (byte >> 4);
                    acc += (double)x[(size_t)m * K + k] * NF4_REF[code] *
                           vv_half_to_float(
                               scales[(size_t)n * (K / 64) + k / 64]);
                }
                ref[(size_t)m * N + n] = (float)acc + vv_half_to_float(bias[n]);
            }
        vv_nf4_gemm_cpu(x, packed, scales, bias, got, m_rows, N, K);
        snprintf(what, sizeof(what), "nf4_gemm M=%d", m_rows);
        check(what, got, ref, m_rows * N, 1e-5);

        /* ── INT4 group-affine GEMM ── */
        for (int m = 0; m < m_rows; m++)
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    const uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
                    const int q = (k & 1) ? (byte & 0xF) : (byte >> 4);
                    const size_t g = (size_t)n * (K / GROUP) + k / GROUP;
                    acc += (double)x[(size_t)m * K + k] *
                           ((float)q * vv_half_to_float(gscales[g]) +
                            vv_half_to_float(gmins[g]));
                }
                ref[(size_t)m * N + n] = (float)acc + vv_half_to_float(bias[n]);
            }
        vv_int4g_gemm_cpu(x, packed, gscales, gmins, bias, got,
                          m_rows, N, K, GROUP);
        snprintf(what, sizeof(what), "int4g_gemm M=%d", m_rows);
        check(what, got, ref, m_rows * N, 1e-5);

        /* ── Dense FP16 weight ── */
        for (int m = 0; m < m_rows; m++)
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++)
                    acc += (double)x[(size_t)m * K + k] *
                           vv_half_to_float(wf16[(size_t)n * K + k]);
                ref[(size_t)m * N + n] = (float)acc + vv_half_to_float(bias[n]);
            }
        vv_gemm_f16w_cpu(x, wf16, bias, got, m_rows, N, K);
        snprintf(what, sizeof(what), "gemm_f16w M=%d", m_rows);
        check(what, got, ref, m_rows * N, 1e-5);
    }

    /* ── RMSNorm ── */
    {
        const int rows = 4, n = 128;
        float* in = malloc(sizeof(float) * rows * n);
        float* out = malloc(sizeof(float) * rows * n);
        float* rr = malloc(sizeof(float) * rows * n);
        uint16_t* w = malloc(sizeof(uint16_t) * n);
        for (int i = 0; i < rows * n; i++) in[i] = frand() * 3.0f;
        for (int i = 0; i < n; i++) w[i] = vv_float_to_half(1.0f + frand() * 0.2f);
        for (int r = 0; r < rows; r++) {
            double ss = 0.0;
            for (int i = 0; i < n; i++) ss += (double)in[r * n + i] * in[r * n + i];
            const float inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
            for (int i = 0; i < n; i++)
                rr[r * n + i] = in[r * n + i] * inv * vv_half_to_float(w[i]);
        }
        vv_rmsnorm_cpu(in, w, out, rows, n, 1e-6f);
        check("rmsnorm", out, rr, rows * n, 1e-5);
        free(in); free(out); free(rr); free(w);
    }

    /* ── SwiGLU ── */
    {
        const int n = 1024;
        float* g = malloc(sizeof(float) * n);
        float* u = malloc(sizeof(float) * n);
        float* o = malloc(sizeof(float) * n);
        float* rr = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) { g[i] = frand() * 4.0f; u[i] = frand(); }
        for (int i = 0; i < n; i++)
            rr[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
        vv_swiglu_cpu(g, u, o, n);
        check("swiglu", o, rr, n, 1e-6);
        free(g); free(u); free(o); free(rr);
    }

    /* ── Attention: decode must equal the last row of prefill ── */
    {
        const int nq = 8, nkv = 2, hd = 64, T = 37;
        float* q = malloc(sizeof(float) * T * nq * hd);
        float* k = malloc(sizeof(float) * T * nkv * hd);
        float* v = malloc(sizeof(float) * T * nkv * hd);
        float* full = malloc(sizeof(float) * T * nq * hd);
        float* one = malloc(sizeof(float) * nq * hd);
        for (int i = 0; i < T * nq * hd; i++) q[i] = frand();
        for (int i = 0; i < T * nkv * hd; i++) { k[i] = frand(); v[i] = frand(); }

        vv_attention_prefill_cpu(q, k, v, full, nq, nkv, hd, T, 0, T, true);
        vv_attention_decode_cpu(q + (size_t)(T - 1) * nq * hd, k, v, one,
                                nq, nkv, hd, T);
        check("attention decode", one, full + (size_t)(T - 1) * nq * hd,
              nq * hd, 1e-5);
        free(q); free(k); free(v); free(full); free(one);
    }

    free(x); free(got); free(ref); free(packed); free(scales);
    free(gscales); free(gmins); free(wf16); free(bias);

    printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
