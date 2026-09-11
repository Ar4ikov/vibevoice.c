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

#include "vibevoice/device.h"

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
vv_status_t vv_residual_add_scaled_dev(
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
 * @brief Launch SiLU activation in-place.
 */
vv_status_t vv_silu_dev(void* data, int total, void* stream) {
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
 * Exact form GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))), matching
 * transformers' ACT2FN["gelu"] (== F.gelu) used by the tokenizer FFN.
 * The tanh approximation is NOT interchangeable here: it drifts by up to
 * ~1e-3 absolute, which compounds across 26 residual blocks.
 */
static __global__ void gelu_kernel(half* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float x = __half2float(data[idx]);
    data[idx] = __float2half(0.5f * x * (1.0f + erff(x * 0.7071067811865476f)));
}

vv_status_t vv_gelu_dev(void* data, int total, void* stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    gelu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (half*)data, total);

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

/* ─── Channel-first RMSNorm kernel ────────────────────────────────────── */

/**
 * @brief RMSNorm for channel-first layout [channels, length].
 *
 * Each thread handles one timestep: normalizes across the channel dimension.
 */
static __global__ void rmsnorm_channel_first_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int channels, int length, float eps)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= length) return;

    float sum_sq = 0.0f;
    for (int c = 0; c < channels; c++) {
        float v = __half2float(input[(size_t)c * length + t]);
        sum_sq += v * v;
    }
    float rms_inv = rsqrtf(sum_sq / (float)channels + eps);

    for (int c = 0; c < channels; c++) {
        float v = __half2float(input[(size_t)c * length + t]);
        float w = __half2float(weight[c]);
        output[(size_t)c * length + t] = __float2half(v * rms_inv * w);
    }
}

vv_status_t vv_rmsnorm_channel_first_dev(
    const void* input, const void* weight, void* output,
    int channels, int length, float eps, void* stream) {
    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    rmsnorm_channel_first_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)input, (const half*)weight, (half*)output,
        channels, length, eps);

    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Channel bias add for channel-first layout: output[c * len + t] += bias[c]
 *
 * Data is [channels, length], bias is [channels].
 * Each channel c gets bias[c] added to all its timesteps.
 */
static __global__ void channel_bias_add_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    int channels, int length)
{
    int c = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels || t >= length) return;

    int idx = c * length + t;
    float val = __half2float(output[idx]);
    float b   = __half2float(bias[c]);
    output[idx] = __float2half(val + b);
}

vv_status_t vv_channel_bias_add_dev(void* output, const void* bias,
                                      int channels, int length, void* stream) {
    if (!output || !bias) return VV_ERR_NULL_PTR;

    dim3 block(256);
    dim3 grid((length + 255) / 256, channels);

    channel_bias_add_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (half*)output, (const half*)bias, channels, length);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* ─── Gather / scatter for tiled FFN on strided channel-first data ─────── */

/**
 * @brief Gather a contiguous [channels, tile_len] tile from a strided [channels, full_len] buffer.
 *
 * src layout: channel c at offset c * full_len
 * dst layout: channel c at offset c * tile_len  (contiguous)
 */
static __global__ void gather_tile_kernel(
    const half* __restrict__ src, half* __restrict__ dst,
    int channels, int full_len, int tile_offset, int tile_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * tile_len;
    if (i >= total) return;
    int c = i / tile_len;
    int t = i % tile_len;
    dst[i] = src[(size_t)c * full_len + tile_offset + t];
}

/**
 * @brief Scatter a contiguous [channels, tile_len] tile back to a strided [channels, full_len] buffer.
 */
static __global__ void scatter_tile_kernel(
    const half* __restrict__ src, half* __restrict__ dst,
    int channels, int full_len, int tile_offset, int tile_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * tile_len;
    if (i >= total) return;
    int c = i / tile_len;
    int t = i % tile_len;
    dst[(size_t)c * full_len + tile_offset + t] = src[i];
}

vv_status_t vv_gather_tile_dev(const void* src, void* dst,
                                 int channels, int full_len,
                                 int tile_offset, int tile_len, void* stream) {
    int total = channels * tile_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gather_tile_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)src, (half*)dst, channels, full_len, tile_offset, tile_len);
    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_scatter_tile_dev(const void* src, void* dst,
                                  int channels, int full_len,
                                  int tile_offset, int tile_len, void* stream) {
    int total = channels * tile_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_tile_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)src, (half*)dst, channels, full_len, tile_offset, tile_len);
    return (cudaGetLastError() == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
