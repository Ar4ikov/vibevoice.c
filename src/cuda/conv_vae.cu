/**
 * @file conv_vae.cu
 * @brief Small elementwise ops shared by the decoder: residual add, bias
 * add, FP32 <-> FP16.
 *
 * The Conv-VAE encoder's own kernels live in vae_kernels.cu.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Simple residual add: x += y
 */
static __global__ void residual_add_kernel(
    half* __restrict__ x,
    const half* __restrict__ y,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float xv = __half2float(x[idx]);
    float yv = __half2float(y[idx]);
    x[idx] = __float2half(xv + yv);
}

/**
 * @brief Launch simple residual add.
 */
vv_status_t vv_residual_add_dev(void* x, const void* y, int total,
                                   void* stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    residual_add_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)x, (const half*)y, total);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Broadcast bias add kernel: output[i*N + j] += bias[j]
 *
 * Adds a 1-D bias vector [N] to each of M rows in a [M, N] matrix.
 */
static __global__ void bias_add_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int j = idx % N;
    float val = __half2float(output[idx]);
    float b   = __half2float(bias[j]);
    output[idx] = __float2half(val + b);
}

/**
 * @brief Add bias [N] to each row of [M, N] FP16 matrix in-place.
 */
vv_status_t vv_bias_add_dev(void* output, const void* bias,
                              int M, int N, void* stream) {
    if (!output || !bias) return VV_ERR_NULL_PTR;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bias_add_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)output, (const half*)bias, M, N);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* ─── FP32 ↔ FP16 conversion kernels ──────────────────────────────────── */

static __global__ void fp32_to_fp16_kernel(const float* __restrict__ in,
                                            half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

static __global__ void fp16_to_fp32_kernel(const half* __restrict__ in,
                                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

vv_status_t vv_fp32_to_fp16_dev(const void* in_fp32, void* out_fp16,
                                   int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp32_to_fp16_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)in_fp32, (half*)out_fp16, n);
    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_fp16_to_fp32_dev(const void* in_fp16, void* out_fp32,
                                   int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp16_to_fp32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)in_fp16, (float*)out_fp32, n);
    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
