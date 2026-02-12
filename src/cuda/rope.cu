/**
 * @file rope.cu
 * @brief Rotary Position Embedding (RoPE) CUDA kernel.
 *
 * Qwen2 RoPE parameters:
 * - theta = 1,000,000
 * - head_dim = 128
 * - Applied to Q and K tensors in-place
 *
 * For position p and dimension pair (2i, 2i+1):
 *   cos_theta = cos(p * theta^(-2i/d))
 *   sin_theta = sin(p * theta^(-2i/d))
 *   q_new[2i]   = q[2i]   * cos_theta - q[2i+1] * sin_theta
 *   q_new[2i+1] = q[2i]   * sin_theta + q[2i+1] * cos_theta
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

/**
 * @brief Apply RoPE to Q or K tensor in-place.
 *
 * x: [n_heads, seq_len, head_dim] FP16
 *
 * Grid: (seq_len, n_heads)
 * Block: (head_dim/2)
 */
__global__ void rope_kernel(
    half* __restrict__ x,
    int seq_len, int n_heads, int head_dim,
    int position_offset,
    float theta)
{
    int pos = blockIdx.x + position_offset;
    int head = blockIdx.y;
    int pair = threadIdx.x;  /* 0 .. head_dim/2 - 1 */

    if (pair >= head_dim / 2) return;

    int seq_idx = blockIdx.x;  /* actual index in the tensor */

    /* Compute rotation angle */
    float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    /* Index into [n_heads, seq_len, head_dim] */
    int base = head * seq_len * head_dim + seq_idx * head_dim;
    int idx0 = base + pair * 2;
    int idx1 = base + pair * 2 + 1;

    float x0 = __half2float(x[idx0]);
    float x1 = __half2float(x[idx1]);

    x[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
    x[idx1] = __float2half(x0 * sin_a + x1 * cos_a);
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief Apply RoPE to Q or K tensor.
 *
 * @param x              [n_heads, seq_len, head_dim] FP16, modified in-place
 * @param seq_len        Sequence length
 * @param n_heads        Number of heads
 * @param head_dim       Dimension per head (128 for Qwen2)
 * @param position_offset Starting position (for KV-cache decode)
 * @param theta          RoPE base frequency (1e6 for Qwen2)
 * @param stream         CUDA stream
 */
vv_status_t vv_rope_cuda(
    void* x, int seq_len, int n_heads, int head_dim,
    int position_offset, float theta, void* stream)
{
    if (!x) return VV_ERR_NULL_PTR;

    dim3 grid(seq_len, n_heads);
    dim3 block(head_dim / 2);

    rope_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (half*)x, seq_len, n_heads, head_dim,
        position_offset, theta);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
