/**
 * @file swiglu.cu
 * @brief SwiGLU activation CUDA kernel.
 *
 * SwiGLU(gate, up) = SiLU(gate) * up
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Used in Qwen2 MLP:
 *   gate = W_gate @ x     [seq_len, intermediate_size]
 *   up   = W_up   @ x     [seq_len, intermediate_size]
 *   hidden = SwiGLU(gate, up)
 *   output = W_down @ hidden
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief Fused SwiGLU kernel.
 *
 * gate:   [n, dim] FP16
 * up:     [n, dim] FP16
 * output: [n, dim] FP16 = SiLU(gate) * up
 */
__global__ void swiglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    /* SiLU(g) = g * sigmoid(g) */
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu_g * u);
}

/**
 * @brief Vectorized SwiGLU using half2.
 */
__global__ void swiglu_vec2_kernel(
    const half2* __restrict__ gate,
    const half2* __restrict__ up,
    half2* __restrict__ output,
    int total_half2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_half2) return;

    half2 g = gate[idx];
    half2 u = up[idx];

    float g0 = __half2float(__low2half(g));
    float g1 = __half2float(__high2half(g));
    float u0 = __half2float(__low2half(u));
    float u1 = __half2float(__high2half(u));

    float s0 = g0 / (1.0f + expf(-g0)) * u0;
    float s1 = g1 / (1.0f + expf(-g1)) * u1;

    output[idx] = __floats2half2_rn(s0, s1);
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief Launch fused SwiGLU kernel.
 *
 * output = SiLU(gate) * up
 */
vv_status_t vv_swiglu_cuda(
    const void* gate, const void* up, void* output,
    int n_elements, void* stream)
{
    if (!gate || !up || !output) return VV_ERR_NULL_PTR;

    /* Use vectorized version if aligned */
    if (n_elements % 2 == 0) {
        int n_half2 = n_elements / 2;
        int threads = 256;
        int blocks = (n_half2 + threads - 1) / threads;

        swiglu_vec2_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const half2*)gate, (const half2*)up, (half2*)output, n_half2);
    } else {
        int threads = 256;
        int blocks = (n_elements + threads - 1) / threads;

        swiglu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const half*)gate, (const half*)up, (half*)output, n_elements);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
