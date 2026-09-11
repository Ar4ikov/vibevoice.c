/**
 * @file awq_repack.c
 * @brief Turn an AutoAWQ / GPTQ checkpoint's tensors into the runtime layout.
 *
 * AutoAWQ writes, for a Linear(in=K, out=N):
 *   qweight  int32 [K][N/8]      8 output columns packed per word
 *   qzeros   int32 [K/G][N/8]    same packing, one entry per group
 *   scales   FP16  [K/G][N]
 *
 * and within a word the eight logical columns appear in the order
 * {0, 4, 1, 5, 2, 6, 3, 7} — a side effect of how AWQ's CUDA kernels want to
 * unpack them with a single subtract-and-shift trick.
 *
 * None of that suits a GEMV, which wants to walk K contiguously for one
 * output row. This repacks once at load into:
 *   packed   uint8 [N][K/2]      high nibble = even k
 *   scales   FP16  [N][K/G]
 *   mins     FP16  [N][K/G]      m = -zero * scale
 *
 * so that dequantisation in the kernel is w = q * scale + min.
 *
 * GPTQ uses the same three tensors with the same packing; its qzeros are
 * stored off by one (the value is zero_point - 1), which `zero_bias` handles.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/quant.h"

#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Logical column j of a packed word lives at nibble slot AWQ_ORDER[j]. */
static const int AWQ_ORDER[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };

vv_status_t vv_awq_repack(const uint32_t* qweight, const uint32_t* qzeros,
                          const uint16_t* scales, int K, int N, int group_size,
                          int zero_bias,
                          uint8_t* out_packed, uint16_t* out_scales,
                          uint16_t* out_mins)
{
    if (!qweight || !qzeros || !scales || !out_packed || !out_scales ||
        !out_mins) {
        return VV_ERR_NULL_PTR;
    }
    if (K <= 0 || N <= 0 || (N & 7) != 0 || (K & 1) != 0)
        return VV_ERR_INVALID_ARG;
    if (group_size <= 0 || (K % group_size) != 0)
        return VV_ERR_INVALID_ARG;

    const int n_words = N / 8;
    const int n_groups = K / group_size;

    /* Shift for logical column j inside its word, precomputed. */
    int shift[8];
    for (int j = 0; j < 8; j++) shift[j] = 4 * AWQ_ORDER[j];

    /*
     * Parallelise over output rows: each thread owns one destination row of
     * packed/scales/mins, so nothing is shared and no locking is needed.
     */
    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (n = 0; n < N; n++) {
        const int word = n >> 3;
        const int sh   = shift[n & 7];

        uint8_t*  prow = out_packed + (size_t)n * (K / 2);
        uint16_t* srow = out_scales + (size_t)n * n_groups;
        uint16_t* mrow = out_mins   + (size_t)n * n_groups;

        for (int g = 0; g < n_groups; g++) {
            const float sc = vv_half_to_float(scales[(size_t)g * N + n]);
            const uint32_t zw = qzeros[(size_t)g * n_words + word];
            const float z = (float)((zw >> sh) & 0xFu) + (float)zero_bias;
            srow[g] = vv_float_to_half(sc);
            mrow[g] = vv_float_to_half(-z * sc);

            const int k0 = g * group_size;
            for (int k = k0; k < k0 + group_size; k += 2) {
                const uint32_t w0 = qweight[(size_t)k * n_words + word];
                const uint32_t w1 = qweight[(size_t)(k + 1) * n_words + word];
                const uint8_t q0 = (uint8_t)((w0 >> sh) & 0xFu);
                const uint8_t q1 = (uint8_t)((w1 >> sh) & 0xFu);
                prow[k >> 1] = (uint8_t)((q0 << 4) | q1);
            }
        }
    }

    return VV_OK;
}

vv_status_t vv_int4g_quantize(const float* w, int N, int K, int group_size,
                              uint8_t* out_packed, uint16_t* out_scales,
                              uint16_t* out_mins)
{
    if (!w || !out_packed || !out_scales || !out_mins) return VV_ERR_NULL_PTR;
    if (group_size <= 0 || (K % group_size) != 0 || (K & 1) != 0)
        return VV_ERR_INVALID_ARG;

    const int n_groups = K / group_size;

    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (n = 0; n < N; n++) {
        const float* wrow = w + (size_t)n * K;
        uint8_t*  prow = out_packed + (size_t)n * (K / 2);
        uint16_t* srow = out_scales + (size_t)n * n_groups;
        uint16_t* mrow = out_mins   + (size_t)n * n_groups;

        for (int g = 0; g < n_groups; g++) {
            const int k0 = g * group_size;
            float lo = wrow[k0], hi = wrow[k0];
            for (int k = k0 + 1; k < k0 + group_size; k++) {
                if (wrow[k] < lo) lo = wrow[k];
                if (wrow[k] > hi) hi = wrow[k];
            }
            /* Asymmetric range over 16 levels; a flat group gets scale 0. */
            const float sc = (hi > lo) ? (hi - lo) / 15.0f : 0.0f;
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            srow[g] = vv_float_to_half(sc);
            mrow[g] = vv_float_to_half(lo);

            for (int k = k0; k < k0 + group_size; k += 2) {
                int q0 = (int)((wrow[k]     - lo) * inv + 0.5f);
                int q1 = (int)((wrow[k + 1] - lo) * inv + 0.5f);
                if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
                prow[(k - 0) >> 1] = (uint8_t)((q0 << 4) | q1);
            }
        }
    }
    return VV_OK;
}
