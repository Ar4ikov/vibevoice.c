/**
 * @file nf4_gemv.cu
 * @brief Fused NF4 dequantise + GEMV / narrow GEMM for the decode path.
 *
 * The generic path (dequantise the whole weight into FP16 scratch, then call
 * cuBLAS) moves ~5x more bytes than necessary when M == 1: 3.2 GB of packed
 * weights are read, 13 GB of FP16 are written, and then those 13 GB are read
 * back by the GEMM. Single-token decode is purely bandwidth bound, so the
 * fused kernel below keeps the weights in their 4-bit form all the way into
 * the multiply-accumulate and touches each byte exactly once.
 *
 * Layout (bitsandbytes, blocksize 64, row-major [N, K]):
 *   element (n, k) -> flat index e = n * K + k
 *   packed[e / 2]  -> high nibble when e is even, low nibble when odd
 *   scales[e / 64] -> FP16 absmax for that block (already de-nested by the
 *                     loader, so no second-level dequantisation here)
 *
 * K is always a multiple of 256 for Qwen2-7B (3584 and 18944), which lets
 * every lane consume one aligned uint32 (8 nibbles) that never straddles a
 * scale block.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define NF4_GEMV_WARPS 8      /* warps per block == rows per block */

/** NF4 code book — exact bitsandbytes values. */
__device__ __constant__ float c_gemv_nf4[16] = {
    -1.0f,                 -0.6961928009986877f,  -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

/**
 * @brief y[n] = sum_k dequant(W)[n,k] * x[k] (+ bias[n]), one warp per row.
 */
__global__ void nf4_gemv_kernel(
    const half*    __restrict__ x,       /* [K]      */
    const uint8_t* __restrict__ w,       /* [N*K/2]  */
    const half*    __restrict__ scales,  /* [N*K/64] */
    const half*    __restrict__ bias,    /* [N] or NULL */
    half*          __restrict__ y,       /* [N]      */
    int N, int K)
{
    __shared__ float lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16)
        lut[threadIdx.x] = c_gemv_nf4[threadIdx.x];
    __syncthreads();

    const int row = blockIdx.x * NF4_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const uint32_t* wrow = (const uint32_t*)(w + (size_t)row * (K >> 1));
    const half*     srow = scales + (size_t)row * (K >> 6);

    float acc = 0.0f;

    /* 32 lanes x 8 elements = 256 elements per iteration */
    for (int base = 0; base < K; base += 256) {
        const int e0 = base + lane * 8;
        const uint32_t bits = wrow[(e0 >> 1) >> 2];   /* 4 bytes = 8 nibbles */
        const float sc = __half2float(srow[e0 >> 6]);

        const float4 xv = *(const float4*)(x + e0);   /* 8 halves, 16 B */
        const half2* xh = (const half2*)&xv;

#pragma unroll
        for (int b = 0; b < 4; b++) {
            const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
            const float w0 = lut[byte >> 4]  * sc;
            const float w1 = lut[byte & 0xF] * sc;
            const float2 xx = __half22float2(xh[b]);
            acc = fmaf(w0, xx.x, acc);
            acc = fmaf(w1, xx.y, acc);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        if (bias) acc += __half2float(bias[row]);
        y[row] = __float2half(acc);
    }
}

/**
 * @brief Same as above for a handful of rows of x (M <= 8), one warp per
 *        (row, token) pair. Keeps short prefill chunks off the scratch path.
 */
__global__ void nf4_gemm_small_kernel(
    const half*    __restrict__ x,       /* [M, K]   */
    const uint8_t* __restrict__ w,       /* [N*K/2]  */
    const half*    __restrict__ scales,
    const half*    __restrict__ bias,
    half*          __restrict__ y,       /* [M, N]   */
    int M, int N, int K)
{
    __shared__ float lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16)
        lut[threadIdx.x] = c_gemv_nf4[threadIdx.x];
    __syncthreads();

    const int row = blockIdx.x * NF4_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const uint32_t* wrow = (const uint32_t*)(w + (size_t)row * (K >> 1));
    const half*     srow = scales + (size_t)row * (K >> 6);

    float acc[8];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    for (int base = 0; base < K; base += 256) {
        const int e0 = base + lane * 8;
        const uint32_t bits = wrow[(e0 >> 1) >> 2];
        const float sc = __half2float(srow[e0 >> 6]);

        float wv[8];
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
            wv[b * 2]     = lut[byte >> 4]  * sc;
            wv[b * 2 + 1] = lut[byte & 0xF] * sc;
        }
        for (int m = 0; m < M; m++) {
            const float4 xv = *(const float4*)(x + (size_t)m * K + e0);
            const half2* xh = (const half2*)&xv;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const float2 xx = __half22float2(xh[b]);
                acc[m] = fmaf(wv[b * 2],     xx.x, acc[m]);
                acc[m] = fmaf(wv[b * 2 + 1], xx.y, acc[m]);
            }
        }
    }

    for (int m = 0; m < M; m++) {
        float v = warp_sum(acc[m]);
        if (lane == 0) {
            if (bias) v += __half2float(bias[row]);
            y[(size_t)m * N + row] = __float2half(v);
        }
    }
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief Fused NF4 GEMV: y[N] = dequant(W)[N,K] @ x[K] + bias.
 *
 * @param x       Input activations [K] FP16 (device)
 * @param packed  NF4 weight [N*K/2] uint8 (device)
 * @param scales  Per-64-element FP16 scales [N*K/64] (device)
 * @param bias    Optional [N] FP16 (device) or NULL
 * @param y       Output [N] FP16 (device)
 */
vv_status_t vv_nf4_gemv_cuda(
    const void* x, const uint8_t* packed, const void* scales,
    const void* bias, void* y, int N, int K, void* stream)
{
    if (!x || !packed || !scales || !y) return VV_ERR_NULL_PTR;
    if ((K & 255) != 0) return VV_ERR_UNSUPPORTED;   /* caller falls back */

    dim3 block(32, NF4_GEMV_WARPS);
    dim3 grid((N + NF4_GEMV_WARPS - 1) / NF4_GEMV_WARPS);

    nf4_gemv_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)x, packed, (const half*)scales,
        (const half*)bias, (half*)y, N, K);

    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Fused NF4 GEMM for a small number of tokens (M <= 8).
 */
vv_status_t vv_nf4_gemm_small_cuda(
    const void* x, const uint8_t* packed, const void* scales,
    const void* bias, void* y, int M, int N, int K, void* stream)
{
    if (!x || !packed || !scales || !y) return VV_ERR_NULL_PTR;
    if (M < 1 || M > 8) return VV_ERR_UNSUPPORTED;
    if ((K & 255) != 0) return VV_ERR_UNSUPPORTED;

    dim3 block(32, NF4_GEMV_WARPS);
    dim3 grid((N + NF4_GEMV_WARPS - 1) / NF4_GEMV_WARPS);

    nf4_gemm_small_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)x, packed, (const half*)scales,
        (const half*)bias, (half*)y, M, N, K);

    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
