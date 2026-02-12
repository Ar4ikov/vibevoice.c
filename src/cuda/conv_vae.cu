/**
 * @file conv_vae.cu
 * @brief Conv-VAE speech tokenizer encoder — GPU implementation.
 *
 * This file orchestrates the Conv-VAE forward pass on GPU using
 * the conv1d, rmsnorm, and other CUDA kernels.
 *
 * Architecture per encoder stage:
 * - D blocks, each: norm → depthwise_conv → residual + layer_scale
 *                    norm → ffn(linear1→silu→linear2) → residual + scale
 * - Downsample: strided 1D conv
 *
 * Final: project to vae_dim (mean + optional logvar for acoustic)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief Residual add kernel: x += y * scale
 *
 * For layer_scale: each channel has its own scale factor.
 */
static __global__ void residual_add_scaled_kernel(
    half* __restrict__ x,
    const half* __restrict__ y,
    const half* __restrict__ scale,
    int channels, int length)
{
    int c = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= channels || t >= length) return;

    int idx = c * length + t;
    float xv = __half2float(x[idx]);
    float yv = __half2float(y[idx]);
    float sv = __half2float(scale[c]);
    x[idx] = __float2half(xv + yv * sv);
}

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
 * @brief SiLU activation kernel (in-place).
 */
static __global__ void silu_kernel(half* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float v = __half2float(data[idx]);
    data[idx] = __float2half(v / (1.0f + expf(-v)));
}

/**
 * @brief Launch residual add with layer scale.
 */
vv_status_t vv_residual_add_scaled_cuda(
    void* x, const void* y, const void* scale,
    int channels, int length, void* stream)
{
    dim3 block(256);
    dim3 grid((length + 255) / 256, channels);

    residual_add_scaled_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (half*)x, (const half*)y, (const half*)scale, channels, length);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Launch simple residual add.
 */
vv_status_t vv_residual_add_cuda(void* x, const void* y, int total,
                                   void* stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    residual_add_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)x, (const half*)y, total);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Launch SiLU activation in-place.
 */
vv_status_t vv_silu_cuda(void* data, int total, void* stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    silu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)data, total);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief GELU activation kernel.
 *
 * GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))
 */
static __global__ void gelu_kernel(half* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float x = __half2float(data[idx]);
    float c = 0.7978845608028654f;
    float inner = c * (x + 0.044715f * x * x * x);
    data[idx] = __float2half(0.5f * x * (1.0f + tanhf(inner)));
}

vv_status_t vv_gelu_cuda(void* data, int total, void* stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    gelu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)data, total);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
