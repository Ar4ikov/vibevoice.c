/**
 * @file awq_gemv.cu
 * @brief Fused INT4 group-affine dequantise + GEMV / GEMM.
 *
 * This is the compute side of AWQ support. AutoAWQ stores weights K-major
 * ([in_features, out_features/8] int32), which is the wrong way round for a
 * GEMV: walking k for one output row would stride by N/8 words and read one
 * useful uint32 per 128-byte transaction. The loader therefore repacks once,
 * at startup, into the same row-major shape the NF4 path uses:
 *
 *   packed  [N][K/2] uint8   high nibble = even k, low nibble = odd k
 *   scales  [N][K/G] FP16
 *   mins    [N][K/G] FP16    m = -zero * scale, folded so dequant is one FMA
 *
 * Dequantisation is then w = q * scale + min, which is cheaper than NF4's
 * codebook lookup, and each lane reads one aligned uint32 covering 8 weights
 * that never straddle a group boundary (G is 32, 64 or 128; all divide 8's
 * multiples evenly).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "vibevoice/device.h"

#define AWQ_GEMV_WARPS 8      /* warps per block == rows per block */

__device__ __forceinline__ float awq_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

/**
 * @brief y[n] = sum_k (q[n,k] * s + m) * x[k] (+ bias[n]), one warp per row.
 */
__global__ void awq_gemv_kernel(
    const half*    __restrict__ x,       /* [K]       */
    const uint8_t* __restrict__ w,       /* [N][K/2]  */
    const half*    __restrict__ scales,  /* [N][K/G]  */
    const half*    __restrict__ mins,    /* [N][K/G]  */
    const half*    __restrict__ bias,    /* [N] or NULL */
    half*          __restrict__ y,       /* [N]       */
    int N, int K, int gshift)
{
    const int row = blockIdx.x * AWQ_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const int groups = K >> gshift;
    const uint32_t* wrow = (const uint32_t*)(w + (size_t)row * (K >> 1));
    const half* srow = scales + (size_t)row * groups;
    const half* mrow = mins   + (size_t)row * groups;

    float acc = 0.0f;

    /* 32 lanes x 8 weights = 256 weights per iteration */
    for (int base = 0; base < K; base += 256) {
        const int e0 = base + lane * 8;
        const uint32_t bits = wrow[(e0 >> 1) >> 2];
        const int g = e0 >> gshift;
        const float sc = __half2float(srow[g]);
        const float mn = __half2float(mrow[g]);

        const float4 xv = *(const float4*)(x + e0);
        const half2* xh = (const half2*)&xv;

#pragma unroll
        for (int b = 0; b < 4; b++) {
            const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
            const float2 xx = __half22float2(xh[b]);
            acc = fmaf(fmaf((float)(byte >> 4),  sc, mn), xx.x, acc);
            acc = fmaf(fmaf((float)(byte & 0xF), sc, mn), xx.y, acc);
        }
    }

    acc = awq_warp_sum(acc);
    if (lane == 0) {
        if (bias) acc += __half2float(bias[row]);
        y[row] = __float2half(acc);
    }
}

/** @brief Same, for a handful of query rows (M <= 8). */
__global__ void awq_gemm_small_kernel(
    const half*    __restrict__ x,       /* [M, K]   */
    const uint8_t* __restrict__ w,
    const half*    __restrict__ scales,
    const half*    __restrict__ mins,
    const half*    __restrict__ bias,
    half*          __restrict__ y,       /* [M, N]   */
    int M, int N, int K, int gshift)
{
    const int row = blockIdx.x * AWQ_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const int groups = K >> gshift;
    const uint32_t* wrow = (const uint32_t*)(w + (size_t)row * (K >> 1));
    const half* srow = scales + (size_t)row * groups;
    const half* mrow = mins   + (size_t)row * groups;

    float acc[8];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    for (int base = 0; base < K; base += 256) {
        const int e0 = base + lane * 8;
        const uint32_t bits = wrow[(e0 >> 1) >> 2];
        const int g = e0 >> gshift;
        const float sc = __half2float(srow[g]);
        const float mn = __half2float(mrow[g]);

        float wv[8];
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
            wv[2 * b + 0] = fmaf((float)(byte >> 4),  sc, mn);
            wv[2 * b + 1] = fmaf((float)(byte & 0xF), sc, mn);
        }
        for (int m = 0; m < M; m++) {
            const float4 xv = *(const float4*)(x + (size_t)m * K + e0);
            const half2* xh = (const half2*)&xv;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const float2 xx = __half22float2(xh[b]);
                acc[m] = fmaf(wv[2 * b + 0], xx.x, acc[m]);
                acc[m] = fmaf(wv[2 * b + 1], xx.y, acc[m]);
            }
        }
    }

    for (int m = 0; m < M; m++) {
        const float v = awq_warp_sum(acc[m]);
        if (lane == 0) {
            float o = v;
            if (bias) o += __half2float(bias[row]);
            y[(size_t)m * N + row] = __float2half(o);
        }
    }
}

/** @brief Expand [N][K/2] INT4 into [N][K] FP16 for the wide-GEMM path. */
__global__ void awq_dequant_kernel(
    const uint8_t* __restrict__ w,
    const half*    __restrict__ scales,
    const half*    __restrict__ mins,
    half*          __restrict__ out,
    int N, int K, int gshift)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)N * (K >> 3);      /* one uint32 per thread */
    if (tid >= total) return;

    const int per_row = K >> 3;
    const int row = (int)(tid / per_row);
    const int wi  = (int)(tid % per_row);
    const int e0  = wi * 8;

    const int groups = K >> gshift;
    const uint32_t bits = ((const uint32_t*)(w + (size_t)row * (K >> 1)))[wi];
    const float sc = __half2float(scales[(size_t)row * groups + (e0 >> gshift)]);
    const float mn = __half2float(mins[(size_t)row * groups + (e0 >> gshift)]);

    half vals[8];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
        vals[2 * b + 0] = __float2half(fmaf((float)(byte >> 4),  sc, mn));
        vals[2 * b + 1] = __float2half(fmaf((float)(byte & 0xF), sc, mn));
    }
    *(float4*)(out + (size_t)row * K + e0) = *(const float4*)vals;
}

/* ─── C entry points ─────────────────────────────────────────────────────── */

extern "C" {

static int gshift_of(int group_size) {
    switch (group_size) {
        case 32:  return 5;
        case 64:  return 6;
        case 128: return 7;
        case 256: return 8;
        default:  return -1;
    }
}

vv_status_t vv_awq_gemv_dev(
    const void* x, const uint32_t* packed, const uint32_t* mins,
    const void* scales, const void* bias, void* y,
    int N, int K, int group_size, void* stream)
{
    if (!x || !packed || !scales || !mins || !y) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 255) != 0) return VV_ERR_UNSUPPORTED;

    dim3 block(32, AWQ_GEMV_WARPS);
    dim3 grid((N + AWQ_GEMV_WARPS - 1) / AWQ_GEMV_WARPS);
    awq_gemv_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)x, (const uint8_t*)packed, (const half*)scales,
        (const half*)mins, (const half*)bias, (half*)y, N, K, gs);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dequant_awq_dev(
    const uint32_t* packed, const uint32_t* mins, const void* scales,
    void* output_fp16, int N, int K, int group_size, void* stream)
{
    if (!packed || !scales || !mins || !output_fp16) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 7) != 0) return VV_ERR_UNSUPPORTED;

    const size_t total = (size_t)N * (K >> 3);
    const int threads = 256;
    const size_t blocks = (total + threads - 1) / threads;
    awq_dequant_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)packed, (const half*)scales, (const half*)mins,
        (half*)output_fp16, N, K, gs);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_awq_gemm_dev(
    const void* input_fp16, const uint32_t* packed, const uint32_t* mins,
    const void* scales, void* output_fp16, void* temp_weight_fp16,
    int M, int N, int K, int group_size, void* stream)
{
    if (!input_fp16 || !packed || !scales || !mins || !output_fp16)
        return VV_ERR_NULL_PTR;

    const int gs = gshift_of(group_size);
    if (gs < 0) return VV_ERR_UNSUPPORTED;

    if (M <= 8 && (K & 255) == 0) {
        dim3 block(32, AWQ_GEMV_WARPS);
        dim3 grid((N + AWQ_GEMV_WARPS - 1) / AWQ_GEMV_WARPS);
        awq_gemm_small_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16, (const uint8_t*)packed,
            (const half*)scales, (const half*)mins, NULL,
            (half*)output_fp16, M, N, K, gs);
        return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
    }

    if (!temp_weight_fp16) return VV_ERR_NULL_PTR;
    vv_status_t s = vv_dequant_awq_dev(packed, mins, scales, temp_weight_fp16,
                                       N, K, group_size, stream);
    if (s != VV_OK) return s;
    return vv_gemm_fp16_dev(input_fp16, temp_weight_fp16, output_fp16,
                            M, N, K, 1.0f, 0.0f, stream);
}

} /* extern "C" */
