/**
 * @file int8_gemv.cu
 * @brief Per-channel INT8 weights against FP16 activations (W8A16).
 *
 * The layout is what `--quant int8` produces at load:
 *
 *   q       [N][K] int8, row-major
 *   scales  [N]    FP32, w[n,k] = q[n,k] * scales[n]
 *
 * One scale per row means the scale leaves the inner loop entirely: each
 * warp accumulates sum_k q * x in FP32 and multiplies once at the end. Each
 * lane reads 16 weights (one uint4) and 16 activations (two float4) per
 * step, so a row needs K % 16 == 0 — true of every shape the runtime loads
 * (3584, 18944, 1536, 8960).
 *
 * Decode (M = 1) reads each weight byte once, half the bytes of FP16 and
 * twice those of the 4-bit formats. Small batches (M <= 8) reuse the
 * converted weights across rows; larger M dequantizes into the FP16 scratch
 * and runs the tensor-core GEMM, like the other formats do.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "vibevoice/device.h"

#define I8_GEMV_WARPS 8      /* warps per block == rows per block */
#define I8_SMALL_M    8

__device__ __forceinline__ float i8_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

/** @brief The 16 int8 codes of one uint4, as floats. */
__device__ __forceinline__ void i8_unpack16(const uint4 bits, float w[16]) {
    const uint32_t words[4] = { bits.x, bits.y, bits.z, bits.w };
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const char4 c = *reinterpret_cast<const char4*>(&words[j]);
        w[4 * j + 0] = (float)c.x;
        w[4 * j + 1] = (float)c.y;
        w[4 * j + 2] = (float)c.z;
        w[4 * j + 3] = (float)c.w;
    }
}

/** @brief sum over 16 lanes of w * x, x being 16 halves at `x`. */
__device__ __forceinline__ float i8_dot16(const float w[16], const half* x) {
    const float4 xa = *(const float4*)(x);
    const float4 xb = *(const float4*)(x + 8);
    const half2* ha = (const half2*)&xa;
    const half2* hb = (const half2*)&xb;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 a = __half22float2(ha[j]);
        const float2 b = __half22float2(hb[j]);
        acc = fmaf(w[2 * j + 0], a.x, acc);
        acc = fmaf(w[2 * j + 1], a.y, acc);
        acc = fmaf(w[8 + 2 * j + 0], b.x, acc);
        acc = fmaf(w[8 + 2 * j + 1], b.y, acc);
    }
    return acc;
}

/**
 * @brief y[n] = scale[n] * sum_k q[n,k] * x[k] (+ bias[n]), one warp per row.
 */
__global__ void int8_gemv_kernel(
    const half*   __restrict__ x,       /* [K]     */
    const int8_t* __restrict__ q,       /* [N][K]  */
    const float*  __restrict__ scales,  /* [N]     */
    const half*   __restrict__ bias,    /* [N] or NULL */
    half*         __restrict__ y,       /* [N]     */
    int N, int K)
{
    const int row = blockIdx.x * I8_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;
    const int lane = threadIdx.x;
    const int8_t* wrow = q + (size_t)row * K;

    float acc = 0.0f;
    /* 32 lanes x 16 weights = 512 weights per step. */
    for (int e0 = lane * 16; e0 < K; e0 += 512) {
        float w[16];
        i8_unpack16(*(const uint4*)(wrow + e0), w);
        acc += i8_dot16(w, x + e0);
    }

    acc = i8_warp_sum(acc);
    if (lane == 0) {
        acc *= scales[row];
        if (bias) acc += __half2float(bias[row]);
        y[row] = __float2half(acc);
    }
}

/** @brief Same for M <= 8 query rows; each weight is converted once. */
__global__ void int8_gemm_small_kernel(
    const half*   __restrict__ x,       /* [M, K]  */
    const int8_t* __restrict__ q,
    const float*  __restrict__ scales,
    half*         __restrict__ y,       /* [M, N]  */
    int M, int N, int K)
{
    const int row = blockIdx.x * I8_GEMV_WARPS + threadIdx.y;
    if (row >= N) return;
    const int lane = threadIdx.x;
    const int8_t* wrow = q + (size_t)row * K;

    /* Fully unrolled over the 8 possible rows so acc[] stays in registers;
       rows past M are predicated off. */
    float acc[I8_SMALL_M];
#pragma unroll
    for (int m = 0; m < I8_SMALL_M; m++) acc[m] = 0.0f;

    for (int e0 = lane * 16; e0 < K; e0 += 512) {
        float w[16];
        i8_unpack16(*(const uint4*)(wrow + e0), w);
#pragma unroll
        for (int m = 0; m < I8_SMALL_M; m++)
            if (m < M) acc[m] += i8_dot16(w, x + (size_t)m * K + e0);
    }

    const float s = scales[row];
#pragma unroll
    for (int m = 0; m < I8_SMALL_M; m++) {
        if (m >= M) break;
        const float v = i8_warp_sum(acc[m]);
        if (lane == 0) y[(size_t)m * N + row] = __float2half(v * s);
    }
}

/** @brief out[n,k] = q[n,k] * scales[n] as FP16, 16 values per thread. */
__global__ void int8_dequant_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ scales,
    half* __restrict__ out, int N, int K)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int per_row = K >> 4;
    if (tid >= (size_t)N * per_row) return;
    const int row = (int)(tid / per_row);
    const int e0 = (int)(tid % per_row) * 16;

    float w[16];
    i8_unpack16(*(const uint4*)(q + (size_t)row * K + e0), w);
    const float s = scales[row];
    half v[16];
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = __float2half(w[j] * s);
    float4* dst = (float4*)(out + (size_t)row * K + e0);
    dst[0] = *(const float4*)&v[0];
    dst[1] = *(const float4*)&v[8];
}

/* ─── C entry points ─────────────────────────────────────────────────────── */

extern "C" {

vv_status_t vv_int8_gemv_dev(
    const void* x, const int8_t* q, const float* scales, const void* bias,
    void* y, int N, int K, void* stream)
{
    if (!x || !q || !scales || !y) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;

    dim3 block(32, I8_GEMV_WARPS);
    dim3 grid((N + I8_GEMV_WARPS - 1) / I8_GEMV_WARPS);
    int8_gemv_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)x, q, scales, (const half*)bias, (half*)y, N, K);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dequant_int8_dev(
    const int8_t* q, const float* scales, void* output_fp16,
    int N, int K, void* stream)
{
    if (!q || !scales || !output_fp16) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;

    const size_t total = (size_t)N * (size_t)(K >> 4);
    const int threads = 256;
    const size_t blocks = (total + threads - 1) / threads;
    int8_dequant_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        q, scales, (half*)output_fp16, N, K);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_int8_gemm_dev(
    const void* input_fp16, const int8_t* q, const float* scales,
    void* output_fp16, void* temp_weight_fp16,
    int M, int N, int K, void* stream)
{
    if (!input_fp16 || !q || !scales || !output_fp16) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;

    if (M <= I8_SMALL_M) {
        dim3 block(32, I8_GEMV_WARPS);
        dim3 grid((N + I8_GEMV_WARPS - 1) / I8_GEMV_WARPS);
        int8_gemm_small_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16, q, scales, (half*)output_fp16, M, N, K);
        return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
    }

    if (!temp_weight_fp16) return VV_ERR_NULL_PTR;
    vv_status_t s = vv_dequant_int8_dev(q, scales, temp_weight_fp16, N, K,
                                        stream);
    if (s != VV_OK) return s;
    return vv_gemm_fp16_dev(input_fp16, temp_weight_fp16, output_fp16,
                            M, N, K, 1.0f, 0.0f, stream);
}

} /* extern "C" */
