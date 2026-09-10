/**
 * @file lm_head.cu
 * @brief Fused LM-head GEMV + argmax for the decode loop.
 *
 * The generic path went cuBLAS -> FP16 logits -> 304 KB device-to-host copy ->
 * CPU argmax over 152 064 entries. Two problems: the D2H copy plus CPU scan
 * costs about as much as the GEMV itself, and FP16 logits only resolve to
 * ~0.03 near the top of the range, which is enough to flip argmax between
 * near-tied tokens. This keeps logits in FP32 and reduces on the GPU, so only
 * 8 bytes cross the bus per token.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#define LMH_WARPS   8
#define ARGMAX_BLK  256

__device__ __forceinline__ float lmh_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

/** @brief logits[n] = sum_k W[n,k] * x[k], FP32 accumulation, warp per row. */
__global__ void lm_head_gemv_kernel(
    const half*  __restrict__ x,      /* [K]    */
    const half*  __restrict__ W,      /* [V, K] */
    float*       __restrict__ logits, /* [V]    */
    int V, int K)
{
    const int row = blockIdx.x * LMH_WARPS + threadIdx.y;
    if (row >= V) return;
    const int lane = threadIdx.x;

    const half* wrow = W + (size_t)row * K;
    float acc = 0.0f;

    /* 32 lanes x 8 halves = 256 elements per step */
    int k = lane * 8;
    for (; k + 8 <= K; k += 256) {
        const float4 wv = *(const float4*)(wrow + k);
        const float4 xv = *(const float4*)(x + k);
        const half2* wh = (const half2*)&wv;
        const half2* xh = (const half2*)&xv;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const float2 a = __half22float2(wh[b]);
            const float2 c = __half22float2(xh[b]);
            acc = fmaf(a.x, c.x, acc);
            acc = fmaf(a.y, c.y, acc);
        }
    }
    /* tail (K is 3584 here, so this never runs, but keep it correct) */
    for (int t = (K & ~255) + lane; t < K; t += 32)
        acc = fmaf(__half2float(wrow[t]), __half2float(x[t]), acc);

    acc = lmh_warp_sum(acc);
    if (lane == 0) logits[row] = acc;
}

/** @brief Stage 1: per-block (max, idx) over a chunk of the logit vector. */
__global__ void argmax_partial_kernel(
    const float* __restrict__ logits, int V,
    float* __restrict__ pmax, int* __restrict__ pidx)
{
    __shared__ float sv[ARGMAX_BLK];
    __shared__ int   si[ARGMAX_BLK];

    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int   bidx = 0;

    for (int i = blockIdx.x * ARGMAX_BLK + tid; i < V;
         i += gridDim.x * ARGMAX_BLK) {
        const float v = logits[i];
        if (v > best) { best = v; bidx = i; }
    }
    sv[tid] = best;
    si[tid] = bidx;
    __syncthreads();

    for (int s = ARGMAX_BLK / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}

/** @brief Stage 2: reduce the per-block candidates to a single token id. */
__global__ void argmax_final_kernel(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    int n, int* __restrict__ out_token, float* __restrict__ out_value)
{
    __shared__ float sv[ARGMAX_BLK];
    __shared__ int   si[ARGMAX_BLK];

    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int   bidx = 0;
    for (int i = tid; i < n; i += ARGMAX_BLK) {
        if (pmax[i] > best) { best = pmax[i]; bidx = pidx[i]; }
    }
    sv[tid] = best;
    si[tid] = bidx;
    __syncthreads();

    for (int s = ARGMAX_BLK / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_token = si[0];
        if (out_value) *out_value = sv[0];
    }
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief LM head GEMV into FP32 logits.
 *
 * @param x       [K] FP16 device (already RMSNormed hidden state)
 * @param W       [V, K] FP16 device
 * @param logits  [V] FP32 device
 */
vv_status_t vv_lm_head_gemv_cuda(
    const void* x, const void* W, void* logits,
    int V, int K, void* stream)
{
    if (!x || !W || !logits) return VV_ERR_NULL_PTR;

    dim3 block(32, LMH_WARPS);
    dim3 grid((V + LMH_WARPS - 1) / LMH_WARPS);

    lm_head_gemv_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)x, (const half*)W, (float*)logits, V, K);

    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Greedy argmax over FP32 logits, entirely on device.
 *
 * @param logits     [V] FP32 device
 * @param scratch_v  [VV_ARGMAX_PARTIALS] FP32 device scratch
 * @param scratch_i  [VV_ARGMAX_PARTIALS] int32 device scratch
 * @param out_token  [1] int32 device
 * @param out_value  [1] FP32 device or NULL
 */
vv_status_t vv_argmax_cuda(
    const void* logits, int V,
    void* scratch_v, void* scratch_i,
    void* out_token, void* out_value, void* stream)
{
    if (!logits || !scratch_v || !scratch_i || !out_token) return VV_ERR_NULL_PTR;

    const int nblocks = 256;   /* must match VV_ARGMAX_PARTIALS */
    argmax_partial_kernel<<<nblocks, ARGMAX_BLK, 0, (cudaStream_t)stream>>>(
        (const float*)logits, V, (float*)scratch_v, (int*)scratch_i);
    argmax_final_kernel<<<1, ARGMAX_BLK, 0, (cudaStream_t)stream>>>(
        (const float*)scratch_v, (const int*)scratch_i, nblocks,
        (int*)out_token, (float*)out_value);

    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
