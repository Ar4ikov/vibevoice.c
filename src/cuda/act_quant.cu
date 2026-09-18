/**
 * @file act_quant.cu
 * @brief Per-token int8 activation quantization, fused into the op before it.
 *
 * W8A8 and W4A8 layers read int8 activations. Each of the three places a
 * projection's input is produced gets a fused form here:
 *
 *   RMSNorm  -> q/k/v and gate/up       vv_rmsnorm_q8_dev
 *   SwiGLU   -> down                    vv_swiglu_q8_dev
 *   attention output -> o               vv_act_quant_dev
 *
 * One block per row. The row is produced into shared memory as the FP16
 * value the unfused op writes (same expression, same reduction order), its
 * absmax is reduced, and it is quantized:
 *
 *   sx = amax / 127,  q = round-half-even(v * (127 / amax)),  |q| <= 127
 *
 * Every step uses an explicitly rounded intrinsic, so `--use_fast_math`
 * cannot turn the division into an approximation and the result is
 * bit-identical to the CPU reference (vv_quant_act_q8_cpu) on the same FP16
 * values. The tests check exactly that.
 *
 * Launch shape depends only on M and K, and nothing is read from the host,
 * so the decode step that contains these stays capturable.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "vibevoice/device.h"

#define AQ_THREADS 256
#define AQ_MAX_K   22528          /* the FP16 row in shared memory, < 48 KB */

__device__ __forceinline__ float aq_block_max(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();                       /* red[] may still be in use */
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (AQ_THREADS / 32)) ? red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int off = 4; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

/**
 * @brief Quantize the FP16 row held in shared memory.
 *
 * Thread t owns 8-element chunks t, t + 256, ...; four consecutive threads
 * cover one 32-column run, which is what the xsum needs.
 */
__device__ __forceinline__ void aq_quantize_row(
    const half* row, int K, int nibble, int8_t* xq, float* sx,
    int32_t* xsum, float* red)
{
    const int tid = threadIdx.x;
    float amax = 0.0f;
    for (int i = tid; i < K; i += AQ_THREADS)
        amax = fmaxf(amax, fabsf(__half2float(row[i])));
    amax = aq_block_max(amax, red);

    const float inv = (amax > 0.0f) ? __fdiv_rn(127.0f, amax) : 0.0f;
    if (tid == 0) *sx = __fdiv_rn(amax, 127.0f);

    const int chunks = K >> 3;
    for (int base = 0; base < chunks; base += AQ_THREADS) {
        const int c = base + tid;
        int s = 0;
        if (c < chunks) {
            const uint4 raw = *(const uint4*)(row + (c << 3));
            const half* h = (const half*)&raw;
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int q = __float2int_rn(__fmul_rn(__half2float(h[j]), inv));
                q = max(-127, min(127, q));
                s += q;
                const uint32_t b = (uint32_t)(q & 0xFF);
                if (j < 4) lo |= b << (8 * j);
                else       hi |= b << (8 * (j - 4));
            }
            if (!nibble) {
                *(uint2*)(xq + (c << 3)) = make_uint2(lo, hi);
            } else {
                /* chunk c is columns 8i..8i+7 of its 32-run, i = c & 3:
                 * its evens land at 4i.., its odds at 16 + 4i.. */
                const uint32_t ev = __byte_perm(lo, hi, 0x6420);
                const uint32_t od = __byte_perm(lo, hi, 0x7531);
                int8_t* base = xq + ((c >> 2) << 5) + ((c & 3) << 2);
                *(uint32_t*)base = ev;
                *(uint32_t*)(base + 16) = od;
            }
        }
        if (xsum) {
            s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
            s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
            if (c < chunks && (tid & 3) == 0) xsum[c >> 2] = s;
        }
    }
}

__global__ void act_quant_kernel(const half* __restrict__ x, int K,
                                 int nibble, int8_t* __restrict__ xq,
                                 float* __restrict__ sx,
                                 int32_t* __restrict__ xsum)
{
    extern __shared__ __align__(16) unsigned char aq_smem[];
    __shared__ float red[AQ_THREADS / 32];
    half* row = (half*)aq_smem;
    const size_t r = blockIdx.x;
    const half* xr = x + r * K;
    for (int c = threadIdx.x; c < (K >> 3); c += AQ_THREADS)
        ((uint4*)row)[c] = ((const uint4*)xr)[c];
    __syncthreads();
    aq_quantize_row(row, K, nibble, xq + r * K, sx + r,
                    xsum ? xsum + r * (K >> 5) : NULL, red);
}

/*
 * The FP16 value written here is the one rmsnorm_kernel (rmsnorm.cu) writes:
 * the same per-thread strided sum of squares over 256 threads, the same tree
 * reduction and the same expressions. Keep the two in step.
 */
__global__ void rmsnorm_q8_kernel(const half* __restrict__ x,
                                  const half* __restrict__ weight, int K,
                                  float eps, int nibble,
                                  int8_t* __restrict__ xq,
                                  float* __restrict__ sx,
                                  int32_t* __restrict__ xsum)
{
    extern __shared__ __align__(16) unsigned char aq_smem[];
    __shared__ float sdata[AQ_THREADS];
    __shared__ float red[AQ_THREADS / 32];
    half* row = (half*)aq_smem;
    const int tid = threadIdx.x;
    const size_t r = blockIdx.x;
    const half* xr = x + r * K;

    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)K + eps);

    for (int i = tid; i < K; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float w = __half2float(weight[i]);
        row[i] = __float2half(v * rms * w);
    }
    __syncthreads();
    aq_quantize_row(row, K, nibble, xq + r * K, sx + r,
                    xsum ? xsum + r * (K >> 5) : NULL, red);
}

/* The FP16 value is swiglu_vec2_kernel's (swiglu.cu): same expression. */
__global__ void swiglu_q8_kernel(const half* __restrict__ gate,
                                 const half* __restrict__ up, int K,
                                 int nibble, int8_t* __restrict__ xq,
                                 float* __restrict__ sx,
                                 int32_t* __restrict__ xsum)
{
    extern __shared__ __align__(16) unsigned char aq_smem[];
    __shared__ float red[AQ_THREADS / 32];
    half2* row2 = (half2*)aq_smem;
    const size_t r = blockIdx.x;
    const half2* g2 = (const half2*)(gate + r * K);
    const half2* u2 = (const half2*)(up + r * K);
    for (int i = threadIdx.x; i < (K >> 1); i += AQ_THREADS) {
        half2 g = g2[i];
        half2 u = u2[i];
        float g0 = __half2float(__low2half(g));
        float g1 = __half2float(__high2half(g));
        float u0 = __half2float(__low2half(u));
        float u1 = __half2float(__high2half(u));
        float s0 = g0 / (1.0f + expf(-g0)) * u0;
        float s1 = g1 / (1.0f + expf(-g1)) * u1;
        row2[i] = __floats2half2_rn(s0, s1);
    }
    __syncthreads();
    aq_quantize_row((const half*)row2, K, nibble, xq + r * K, sx + r,
                    xsum ? xsum + r * (K >> 5) : NULL, red);
}

extern "C" {

static vv_status_t aq_check(int M, int K, const void* a, const int8_t* xq,
                            const float* sx) {
    if (!a || !xq || !sx) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    if ((K & 31) != 0 || K > AQ_MAX_K) return VV_ERR_UNSUPPORTED;
    return VV_OK;
}

vv_status_t vv_act_quant_dev(const void* x, int M, int K, int layout,
                             int8_t* xq, float* sx, int32_t* xsum,
                             void* stream)
{
    vv_status_t s = aq_check(M, K, x, xq, sx);
    if (s != VV_OK) return s;
    act_quant_kernel<<<M, AQ_THREADS, (size_t)K * 2, (cudaStream_t)stream>>>(
        (const half*)x, K, layout == VV_Q8_NIBBLE, xq, sx, xsum);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_rmsnorm_q8_dev(const void* x, const void* weight,
                              int M, int K, float eps, int layout,
                              int8_t* xq, float* sx, int32_t* xsum,
                              void* stream)
{
    vv_status_t s = aq_check(M, K, x, xq, sx);
    if (s != VV_OK) return s;
    if (!weight) return VV_ERR_NULL_PTR;
    /* vv_rmsnorm_dev narrows its block below 256 columns; match it or bail. */
    if (K < AQ_THREADS) return VV_ERR_UNSUPPORTED;
    rmsnorm_q8_kernel<<<M, AQ_THREADS, (size_t)K * 2, (cudaStream_t)stream>>>(
        (const half*)x, (const half*)weight, K, eps, layout == VV_Q8_NIBBLE,
        xq, sx, xsum);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_swiglu_q8_dev(const void* gate, const void* up, int M, int K,
                             int layout, int8_t* xq, float* sx,
                             int32_t* xsum, void* stream)
{
    vv_status_t s = aq_check(M, K, gate, xq, sx);
    if (s != VV_OK) return s;
    if (!up) return VV_ERR_NULL_PTR;
    swiglu_q8_kernel<<<M, AQ_THREADS, (size_t)K * 2, (cudaStream_t)stream>>>(
        (const half*)gate, (const half*)up, K, layout == VV_Q8_NIBBLE,
        xq, sx, xsum);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
