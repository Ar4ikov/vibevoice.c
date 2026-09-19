/**
 * @file load_quant.c
 * @brief Quantizing dense weights at load time: the option, NF4 and INT8.
 *
 * INT4G lives with the AWQ repack (awq_repack.c), which already had a
 * min/max quantizer; NF4 and INT8 are here. All of them work row by row, so
 * they give the same bytes whatever the thread count.
 *
 * The loader calls these on a few rows at a time from inside its own
 * parallel loop, so their `omp parallel for` only forks when it is handed a
 * large block; on a small one it runs in the calling thread.
 */

#include "vibevoice/quant.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

/** Below this many rows a quantizer does not open a parallel region. */
#define VV_QUANT_PAR_ROWS 64

static const char* const LOAD_QUANT_NAMES[VV_LOAD_QUANT_COUNT] = {
    "auto", "none", "nf4", "int4", "int8",
};

vv_load_quant_t vv_load_quant_parse(const char* name) {
    if (!name) return VV_LOAD_QUANT_COUNT;
    for (int i = 0; i < VV_LOAD_QUANT_COUNT; i++)
        if (strcmp(name, LOAD_QUANT_NAMES[i]) == 0) return (vv_load_quant_t)i;
    return VV_LOAD_QUANT_COUNT;
}

const char* vv_load_quant_name(vv_load_quant_t q) {
    return (q >= 0 && q < VV_LOAD_QUANT_COUNT) ? LOAD_QUANT_NAMES[q] : "?";
}

/* The codebook every NF4 kernel in the tree decodes with. */
extern const float VV_NF4_TABLE[16];

/**
 * @brief Index of the level nearest `x`; ties go to the lower level.
 *
 * Counting the midpoints below x gives the index directly. A binary search
 * does a third of the compares, but its four branches depend on the data
 * and mispredict about half the time on Gaussian weights: it measured
 * 12.6 ns per weight against 1.6 ns for this, which compiles to compares
 * and adds with no branch at all.
 */
static inline int nf4_nearest(float x, const float mids[15]) {
    int c = 0;
    for (int j = 0; j < 15; j++) c += (x > mids[j]);
    return c;
}

vv_status_t vv_nf4_quantize(const float* w, int N, int K,
                            uint8_t* out_packed, uint16_t* out_scales) {
    if (!w || !out_packed || !out_scales) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K % VV_NF4_BLOCK) != 0) return VV_ERR_INVALID_ARG;

    float mids[15];
    for (int j = 0; j < 15; j++)
        mids[j] = 0.5f * (VV_NF4_TABLE[j] + VV_NF4_TABLE[j + 1]);

    const int blocks = K / VV_NF4_BLOCK;
    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N >= VV_QUANT_PAR_ROWS)
#endif
    for (n = 0; n < N; n++) {
        const float* row = w + (size_t)n * K;
        uint8_t* prow = out_packed + (size_t)n * (K / 2);
        uint16_t* srow = out_scales + (size_t)n * blocks;
        for (int b = 0; b < blocks; b++) {
            const float* x = row + (size_t)b * VV_NF4_BLOCK;
            float amax = 0.0f;
            for (int i = 0; i < VV_NF4_BLOCK; i++) {
                const float a = fabsf(x[i]);
                if (a > amax) amax = a;     /* NaN never wins a compare */
            }
            const uint16_t sh = vv_float_to_half_rne(amax);
            srow[b] = sh;
            const float sc = vv_half_to_float(sh);
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            uint8_t* pb = prow + (size_t)b * (VV_NF4_BLOCK / 2);
            for (int i = 0; i < VV_NF4_BLOCK; i += 2) {
                float v0 = x[i] * inv, v1 = x[i + 1] * inv;
                if (!(v0 == v0)) v0 = 0.0f;
                if (!(v1 == v1)) v1 = 0.0f;
                pb[i >> 1] = (uint8_t)((nf4_nearest(v0, mids) << 4) |
                                       nf4_nearest(v1, mids));
            }
        }
    }
    return VV_OK;
}

vv_status_t vv_int8_quantize_rows(const float* w, int N, int K,
                                  int8_t* out_q, float* out_scale) {
    if (!w || !out_q || !out_scale) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;

    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N >= VV_QUANT_PAR_ROWS)
#endif
    for (n = 0; n < N; n++) {
        const float* row = w + (size_t)n * K;
        int8_t* q = out_q + (size_t)n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = fabsf(row[k]);
            if (a > amax) amax = a;         /* NaN never wins a compare */
        }
        const float scale = amax / 127.0f;
        out_scale[n] = scale;
        if (!(scale > 0.0f) || !(amax < INFINITY)) {
            /* All zero (or an infinity, which no scale can represent). */
            memset(q, 0, (size_t)K);
            if (!(amax < INFINITY)) out_scale[n] = 0.0f;
            continue;
        }
        for (int k = 0; k < K; k++) {
            float v = rintf(row[k] / scale);  /* default mode: half to even */
            if (!(v == v)) v = 0.0f;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[k] = (int8_t)v;
        }
    }
    return VV_OK;
}
