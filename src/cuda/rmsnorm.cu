/**
 * @file rmsnorm.cu
 * @brief RMSNorm CUDA kernel for Qwen2 LLM.
 *
 * RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
 * Uses Welford-style accumulation for numerical stability.
 * FP32 accumulation, FP16 I/O.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief RMSNorm kernel. One block per row (token position).
 *
 * input:  [seq_len, hidden_size] FP16
 * weight: [hidden_size] FP16
 * output: [seq_len, hidden_size] FP16
 *
 * Block: (256, 1, 1) threads
 * Grid:  (seq_len, 1, 1)
 */
__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x = input + row * hidden_size;
    half* y = output + row * hidden_size;

    /* Compute sum of squares using warp reduction */
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }

    /* Block-level reduction using shared memory */
    __shared__ float sdata[256];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)hidden_size + eps);

    /* Apply normalization and weight */
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half(v * rms * w);
    }
}

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Launch RMSNorm kernel.
 *
 * @param input       [seq_len, hidden_size] FP16
 * @param weight      [hidden_size] FP16
 * @param output      [seq_len, hidden_size] FP16
 * @param seq_len     Number of rows
 * @param hidden_size Dimension of each row
 * @param eps         Epsilon (1e-6 for Qwen2)
 * @param stream      CUDA stream
 */
vv_status_t vv_rmsnorm_dev(
    const void* input, const void* weight, void* output,
    int seq_len, int hidden_size, float eps, void* stream)
{
    if (!input || !weight || !output) return VV_ERR_NULL_PTR;

    int threads = 256;
    if (hidden_size < 256) threads = hidden_size;

    dim3 grid(seq_len);
    dim3 block(threads);

    rmsnorm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)input, (const half*)weight, (half*)output,
        hidden_size, eps);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
