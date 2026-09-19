/*
 * common.metal -- shared by every kernel file (they are compiled as one
 * library, this one first).
 *
 * Conventions, the same for every kernel:
 *   [[buffer(0)]]   the launch's parameter struct, `constant`
 *   [[buffer(1..)]] device buffers, in the order the C wrapper lists them
 *   grid            threadgroups, as a CUDA grid; block = threads per group
 *
 * The library is compiled with safe math: IEEE results, precise
 * transcendentals, and a*b+c fused only inside one expression (what Apple
 * clang does to the CPU kernels). Where a CUDA kernel calls fmaf the port
 * calls fma, and where a result must match the CPU bit for bit the kernel
 * writes the CPU's expression.
 */
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define VV_SIMD 32

/* Launch geometry, CUDA spelling. */
#define VV_GRID_ARGS                                                         \
    uint3 blockIdx  [[threadgroup_position_in_grid]],                        \
    uint3 threadIdx [[thread_position_in_threadgroup]],                      \
    uint3 blockDim  [[threads_per_threadgroup]],                             \
    uint  lane      [[thread_index_in_simdgroup]],                           \
    uint  warp      [[simdgroup_index_in_threadgroup]]

/* A warp sum in CUDA's butterfly order (xor 16, 8, 4, 2, 1), so a port of a
 * warp_sum kernel adds its partials exactly as the original does. */
static inline float vv_warp_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

static inline float vv_warp_max(float v) {
    v = max(v, simd_shuffle_xor(v, 16));
    v = max(v, simd_shuffle_xor(v, 8));
    v = max(v, simd_shuffle_xor(v, 4));
    v = max(v, simd_shuffle_xor(v, 2));
    v = max(v, simd_shuffle_xor(v, 1));
    return v;
}

/* CUDA's __shfl_down_sync reduction: lane 0 ends up with the sum. */
static inline float vv_warp_sum_down(float v) {
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

/* NF4 code book, the exact bitsandbytes values the CUDA kernels use. */
constant float vv_nf4_lut[16] = {
    -1.0f,                 -0.6961928009986877f,  -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};

/*
 * erff for the exact GELU. The Metal library has no erf; this is FreeBSD's
 * s_erff.c as musl carries it, which the CPU kernels' libm erff agrees with
 * to an ulp or two -- well inside the FP16 rounding every GELU output gets.
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 * Conversion to float by Ian Lance Taylor, Cygnus Support.
 */
static inline float vv_erfc2(uint ix, float x) {
    const float erx = 8.4506291151e-01f;
    if (ix < 0x3fa00000u) {                     /* |x| < 1.25 */
        const float s = fabs(x) - 1.0f;
        const float P = -2.3621185683e-03f + s * (4.1485610604e-01f + s * (-3.7220788002e-01f
                      + s * (3.1834661961e-01f + s * (-1.1089469492e-01f + s * (3.5478305072e-02f
                      + s * -2.1663755178e-03f)))));
        const float Q = 1.0f + s * (1.0642088205e-01f + s * (5.4039794207e-01f + s * (7.1828655899e-02f
                      + s * (1.2617121637e-01f + s * (1.3637083583e-02f + s * 1.1984500103e-02f)))));
        return 1.0f - erx - P / Q;
    }
    x = fabs(x);
    const float s = 1.0f / (x * x);
    float R, S;
    if (ix < 0x4036db6du) {                     /* |x| < 1/0.35 */
        R = -9.8649440333e-03f + s * (-6.9385856390e-01f + s * (-1.0558626175e+01f + s * (-6.2375331879e+01f
          + s * (-1.6239666748e+02f + s * (-1.8460508728e+02f + s * (-8.1287437439e+01f + s * -9.8143291473e+00f))))));
        S = 1.0f + s * (1.9651271820e+01f + s * (1.3765776062e+02f + s * (4.3456588745e+02f + s * (6.4538726807e+02f
          + s * (4.2900814819e+02f + s * (1.0863500214e+02f + s * (6.5702495575e+00f + s * -6.0424413532e-02f)))))));
    } else {
        R = -9.8649431020e-03f + s * (-7.9928326607e-01f + s * (-1.7757955551e+01f + s * (-1.6063638306e+02f
          + s * (-6.3756646729e+02f + s * (-1.0250950928e+03f + s * -4.8351919556e+02f)))));
        S = 1.0f + s * (3.0338060379e+01f + s * (3.2579251099e+02f + s * (1.5367296143e+03f + s * (3.1998581543e+03f
          + s * (2.5530502930e+03f + s * (4.7452853394e+02f + s * -2.2440952301e+01f))))));
    }
    const float z = as_type<float>(as_type<uint>(x) & 0xffffe000u);
    return exp(-z * z - 0.5625f) * exp((z - x) * (z + x) + R / S) / x;
}

static inline float vv_erf(float x) {
    const uint bits = as_type<uint>(x);
    const bool sign = (bits >> 31) != 0u;
    const uint ix = bits & 0x7fffffffu;
    if (ix >= 0x7f800000u) return 1.0f - 2.0f * (float)sign + 1.0f / x;
    if (ix < 0x3f580000u) {                     /* |x| < 0.84375 */
        if (ix < 0x31800000u) return 0.125f * (8.0f * x + 1.0270333290e+00f * x);
        const float z = x * x;
        const float r = 1.2837916613e-01f + z * (-3.2504209876e-01f + z * (-2.8481749818e-02f
                      + z * (-5.7702702470e-03f + z * -2.3763017452e-05f)));
        const float s = 1.0f + z * (3.9791721106e-01f + z * (6.5022252500e-02f + z * (5.0813062117e-03f
                      + z * (1.3249473704e-04f + z * -3.9602282413e-06f))));
        return x + x * (r / s);
    }
    const float y = ix < 0x40c00000u ? 1.0f - vv_erfc2(ix, x) : 1.0f - 0x1p-120f;
    return sign ? -y : y;
}

static inline float vv_gelu(float x) {
    return 0.5f * x * (1.0f + vv_erf(x * 0.7071067811865476f));
}
