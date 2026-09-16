/**
 * @file test_attention.c
 * @brief Prefill attention against a CPU reference, and how fast it runs.
 *
 * The prefill kernel is the one place where a wrong answer is easy to miss:
 * attention over noise still looks like attention over noise, and the model
 * degrades into plausible-sounding text rather than crashing. So this checks
 * the kernel against an FP64 reference computed the obvious way, at sequence
 * lengths that land on and off every tile boundary, with and without the
 * chunked-prefill offset.
 *
 * Run twice by ctest: once as built (tensor cores on Ampere and later) and
 * once with VV_ATTN_MMA=0, which forces the scalar kernel. Both have to agree
 * with the reference, and the timings printed at the end are what the two are
 * compared on.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_Q_HEADS  28
#define N_KV_HEADS 4
#define HEAD_DIM   128

static uint32_t rng_state = 0x243F6A88u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

static int failures = 0;

/**
 * @brief Softmax(QKᵀ/√d)·V in FP64 for one head, for the rows of one query.
 *
 * Only the heads the caller names, because the reference is O(S²·d) and the
 * point is to catch a wrong kernel, not to re-run it.
 */
static void reference_head(const uint16_t* Q, const uint16_t* K,
                           const uint16_t* V, double* out,
                           int head, int q_len, int q_offset, int kv_len,
                           bool causal)
{
    const int q_stride  = N_Q_HEADS * HEAD_DIM;
    const int kv_stride = N_KV_HEADS * HEAD_DIM;
    const int kv_head   = head / (N_Q_HEADS / N_KV_HEADS);
    const double scale  = 1.0 / sqrt((double)HEAD_DIM);

    double* p = (double*)malloc((size_t)kv_len * sizeof(double));
    for (int r = 0; r < q_len; r++) {
        const uint16_t* q = Q + (size_t)r * q_stride + head * HEAD_DIM;
        const int q_abs = q_offset + r;
        double m = -1e300;
        for (int c = 0; c < kv_len; c++) {
            if (causal && c > q_abs) { p[c] = -1e300; continue; }
            const uint16_t* k = K + (size_t)c * kv_stride + kv_head * HEAD_DIM;
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += (double)vv_half_to_float(q[d]) *
                       (double)vv_half_to_float(k[d]);
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
        if (sum <= 0.0) continue;
        for (int c = 0; c < kv_len; c++) {
            if (p[c] == 0.0) continue;
            const double w = p[c] / sum;
            const uint16_t* v = V + (size_t)c * kv_stride + kv_head * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                o[d] += w * (double)vv_half_to_float(v[d]);
        }
    }
    free(p);
}

/**
 * @brief One shape: build inputs, run the kernel, diff two heads.
 *
 * `q_offset` > 0 is the chunked-prefill case, where this chunk's queries have
 * to attend to every key the earlier chunks wrote, not just their own.
 */
static void check(int q_len, int q_offset, int kv_len, bool causal,
                  const char* what)
{
    const size_t q_elems  = (size_t)q_len * N_Q_HEADS * HEAD_DIM;
    const size_t kv_elems = (size_t)kv_len * N_KV_HEADS * HEAD_DIM;

    uint16_t* hQ = (uint16_t*)malloc(q_elems * 2);
    uint16_t* hK = (uint16_t*)malloc(kv_elems * 2);
    uint16_t* hV = (uint16_t*)malloc(kv_elems * 2);
    uint16_t* hO = (uint16_t*)malloc(q_elems * 2);
    double*   ref = (double*)malloc((size_t)q_len * HEAD_DIM * sizeof(double));
    if (!hQ || !hK || !hV || !hO || !ref) { failures++; return; }

    for (size_t i = 0; i < q_elems; i++)  hQ[i] = vv_float_to_half(frand());
    for (size_t i = 0; i < kv_elems; i++) hK[i] = vv_float_to_half(frand() * 1.5f);
    for (size_t i = 0; i < kv_elems; i++) hV[i] = vv_float_to_half(frand() * 2.0f);

    void *dQ = NULL, *dK = NULL, *dV = NULL, *dO = NULL;
    if (vv_dev_alloc(&dQ, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&dK, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dV, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dO, q_elems * 2)  != VV_OK) {
        printf("  FAIL %s: allocation\n", what);
        failures++;
        return;
    }
    vv_dev_memcpy_h2d(dQ, hQ, q_elems * 2, NULL);
    vv_dev_memcpy_h2d(dK, hK, kv_elems * 2, NULL);
    vv_dev_memcpy_h2d(dV, hV, kv_elems * 2, NULL);
    vv_dev_memset(dO, 0, q_elems * 2);

    vv_status_t s = vv_gqa_attention_prefill_cached_dev(
        dQ, dK, dV, dO, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
        q_len, q_offset, kv_len, causal, NULL);
    vv_dev_stream_sync(NULL);
    if (s != VV_OK) {
        printf("  FAIL %s: kernel returned %d\n", what, s);
        failures++;
        goto done;
    }
    vv_dev_memcpy_d2h(hO, dO, q_elems * 2, NULL);

    {
        /* Head 0 shares a KV head with six others; head 27 is the last one. */
        const int heads[2] = { 0, N_Q_HEADS - 1 };
        double worst = 0.0, worst_cos = 1.0;
        for (int hi = 0; hi < 2; hi++) {
            const int head = heads[hi];
            reference_head(hQ, hK, hV, ref, head, q_len, q_offset, kv_len,
                           causal);
            for (int r = 0; r < q_len; r++) {
                const uint16_t* got = hO + (size_t)r * N_Q_HEADS * HEAD_DIM
                                    + head * HEAD_DIM;
                const double* want = ref + (size_t)r * HEAD_DIM;
                double num = 0.0, da = 0.0, db = 0.0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    const double g = vv_half_to_float(got[d]);
                    const double e = fabs(g - want[d]);
                    if (e > worst) worst = e;
                    num += g * want[d]; da += g * g; db += want[d] * want[d];
                }
                if (da > 0.0 && db > 0.0) {
                    const double c = num / (sqrt(da) * sqrt(db));
                    if (c < worst_cos) worst_cos = c;
                }
            }
        }
        /*
         * Both kernels accumulate in FP32 from FP16 inputs against an FP64
         * reference, so a few thousandths of absolute error on values of order
         * one is the arithmetic, not a bug. A wrong kernel misses by far more.
         */
        const bool ok = worst < 4e-3 && worst_cos > 0.9999;
        printf("  %s %-34s max|err| %.2e  min cos %.7f\n",
               ok ? "ok  " : "FAIL", what, worst, worst_cos);
        if (!ok) failures++;
    }

done:
    vv_dev_free(dQ); vv_dev_free(dK); vv_dev_free(dV); vv_dev_free(dO);
    free(hQ); free(hK); free(hV); free(hO); free(ref);
}

/** @brief Causal self-attention at one length, timed over several runs. */
static void bench(int seq)
{
    const size_t q_elems  = (size_t)seq * N_Q_HEADS * HEAD_DIM;
    const size_t kv_elems = (size_t)seq * N_KV_HEADS * HEAD_DIM;

    void *dQ = NULL, *dK = NULL, *dV = NULL, *dO = NULL;
    if (vv_dev_alloc(&dQ, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&dK, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dV, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dO, q_elems * 2)  != VV_OK) return;
    vv_dev_memset(dQ, 0x11, q_elems * 2);
    vv_dev_memset(dK, 0x11, kv_elems * 2);
    vv_dev_memset(dV, 0x11, kv_elems * 2);

    for (int i = 0; i < 2; i++)
        vv_gqa_attention_prefill_cached_dev(dQ, dK, dV, dO, N_Q_HEADS,
                                            N_KV_HEADS, HEAD_DIM, seq, 0, seq,
                                            true, NULL);
    vv_dev_stream_sync(NULL);

    const int iters = 5;
    const double t0 = vv_time_ms();
    for (int i = 0; i < iters; i++)
        vv_gqa_attention_prefill_cached_dev(dQ, dK, dV, dO, N_Q_HEADS,
                                            N_KV_HEADS, HEAD_DIM, seq, 0, seq,
                                            true, NULL);
    vv_dev_stream_sync(NULL);
    const double ms = (vv_time_ms() - t0) / iters;

    /* Causal: half the pairs, two matmuls of head_dim MACs each. */
    const double macs = 0.5 * (double)seq * (double)seq
                      * N_Q_HEADS * HEAD_DIM * 2.0;
    printf("  seq %6d: %8.3f ms   %6.1f TFLOP/s\n",
           seq, ms, 2.0 * macs / (ms * 1e-3) / 1e12);

    vv_dev_free(dQ); vv_dev_free(dK); vv_dev_free(dV); vv_dev_free(dO);
}

int main(void) {
#ifndef VV_HAS_ACCEL
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(0, &total, &freem, NULL) != VV_OK || total == 0) {
        printf("SKIP: no device available\n");
        return 0;
    }
    const char* forced = getenv("VV_ATTN_MMA");
    printf("=== prefill attention (%s) ===\n",
           (forced && forced[0] == '0') ? "scalar kernel forced"
                                        : "as built");

    /* On and off every tile boundary: 16 (fragment), 64 (block and KV tile). */
    check(1,    0, 1,    true,  "q=1 kv=1 causal");
    check(16,   0, 16,   true,  "q=16 kv=16 causal");
    check(17,   0, 17,   true,  "q=17 kv=17 causal");
    check(64,   0, 64,   true,  "q=64 kv=64 causal");
    check(65,   0, 65,   true,  "q=65 kv=65 causal");
    check(100,  0, 100,  true,  "q=100 kv=100 causal");
    check(128,  0, 128,  true,  "q=128 kv=128 causal");
    check(257,  0, 257,  true,  "q=257 kv=257 causal");
    check(64,   0, 64,   false, "q=64 kv=64 non-causal");
    check(100,  0, 257,  false, "q=100 kv=257 non-causal");

    /* Chunked prefill: this chunk starts partway into the cache. */
    check(64,  64,  128, true,  "q=64 @64 kv=128 causal");
    check(37, 100,  137, true,  "q=37 @100 kv=137 causal");
    check(128, 63,  191, true,  "q=128 @63 kv=191 causal");

    printf("\n--- throughput ---\n");
    bench(1024);
    bench(4096);
    bench(8192);

    if (failures) {
        printf("\n=== %d shape(s) failed ===\n", failures);
        return 1;
    }
    printf("\n=== all shapes match the reference ===\n");
    return 0;
#endif
}
