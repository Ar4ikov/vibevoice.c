/**
 * @file load_quant.c
 * @brief Quantizing dense weights at load time: the option and NF4.
 *
 * INT4G lives with the AWQ repack (awq_repack.c), which already had a
 * min/max quantizer; NF4 is here. Both work row by row, so they give the
 * same bytes whatever the thread count.
 */

#include "vibevoice/quant.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

static const char* const LOAD_QUANT_NAMES[VV_LOAD_QUANT_COUNT] = {
    "auto", "none", "nf4", "int4",
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
#define NF4_LEVELS VV_NF4_TABLE

/** @brief Index of the level nearest `x`; ties go to the lower level. */
static int nf4_nearest(float x) {
    /* Binary search over the 15 midpoints: four compares, not fifteen. */
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (x > NF4_LEVELS[mid]) lo = mid; else hi = mid;
    }
    /* x lies in [LEVELS[lo], LEVELS[hi]] (or beyond an end). */
    const float mid = 0.5f * (NF4_LEVELS[lo] + NF4_LEVELS[hi]);
    return (x > mid) ? hi : lo;
}

vv_status_t vv_nf4_quantize(const float* w, int N, int K,
                            uint8_t* out_packed, uint16_t* out_scales) {
    if (!w || !out_packed || !out_scales) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K % VV_NF4_BLOCK) != 0) return VV_ERR_INVALID_ARG;

    const int blocks = K / VV_NF4_BLOCK;
    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
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
            const uint16_t sh = vv_float_to_half(amax);
            srow[b] = sh;
            const float sc = vv_half_to_float(sh);
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            uint8_t* pb = prow + (size_t)b * (VV_NF4_BLOCK / 2);
            for (int i = 0; i < VV_NF4_BLOCK; i += 2) {
                float v0 = x[i] * inv, v1 = x[i + 1] * inv;
                if (!(v0 == v0)) v0 = 0.0f;
                if (!(v1 == v1)) v1 = 0.0f;
                pb[i >> 1] = (uint8_t)((nf4_nearest(v0) << 4) | nf4_nearest(v1));
            }
        }
    }
    return VV_OK;
}
