/**
 * @file i8_common.cuh
 * @brief Shared pieces of the int8-activation linear kernels (W8A8, W4A8).
 *
 * gemv_i8.cu holds the decode kernels (M <= 8, dp4a) and gemm_i8.cu the
 * prefill ones (mma.sync s8 on sm_80+, dp4a tiles elsewhere) plus the public
 * entry points that choose between them.
 */
#ifndef VV_I8_COMMON_CUH
#define VV_I8_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/** @brief Everything a W8A8 or W4A8 launch needs; passed by value. */
struct vv_i8_args {
    const int8_t*  xq;       /* [M][K] int8 activations                     */
    const float*   sx;       /* [M] per-token scale                         */
    const int32_t* xsum;     /* [M][K/32] sums of xq (W4A8 GEMV)            */
    const void*    w;        /* W8: int8 [N][K]; W4: GPU-layout [N][K/2]    */
    const float*   sw;       /* W8: FP32 [N]                                */
    const half2*   sz;       /* W4: {scale, zero} [N][K/G]                  */
    const half*    bias;     /* [N] or NULL                                 */
    const half*    res;      /* [M][N] or NULL; may alias y                 */
    void*          y;        /* [M][N] FP16, or FP32 when y_f32             */
    int y_f32;
    int x_layout;            /* vv_q8_layout_t of xq                        */
    int M, N, K, G;
};

/** @brief Write one output: v + bias[n] + res[m,n], FP16 or FP32. */
__device__ __forceinline__ void i8_store(const vv_i8_args& p, int m, int n,
                                         float v) {
    const size_t idx = (size_t)m * p.N + n;
    if (p.bias) v += __half2float(p.bias[n]);
    if (p.res)  v += __half2float(p.res[idx]);
    if (p.y_f32) ((float*)p.y)[idx] = v;
    else         ((half*)p.y)[idx] = __float2half(v);
}

__device__ __forceinline__ int i8_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    for (int i = 0; i < 4; i++)
        c += (int)(signed char)(a >> (8 * i)) * (int)(signed char)(b >> (8 * i));
    return c;
#endif
}

/**
 * @brief Four nibbles of a W4A16 GPU-layout word -> four signed bytes q - z.
 *
 * `h` holds nibble slots 0..3 (or 4..7, shifted down) in its low 16 bits;
 * byte i of the result is slot i. `zrep` is the zero point replicated into
 * all four bytes. The difference is in [-15, 15], exact as s8, so the MMA
 * needs no correction term.
 */
__device__ __forceinline__ uint32_t i8_unpack4_sub(uint32_t h, uint32_t zrep) {
    const uint32_t a = __byte_perm(h, 0u, 0x4140);   /* b0, 0, b1, 0 */
    const uint32_t v = (a & 0x000F000Fu) | ((a << 4) & 0x0F000F00u);
    return __vsub4(v, zrep);
}

#endif /* VV_I8_COMMON_CUH */
