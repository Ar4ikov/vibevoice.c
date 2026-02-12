/**
 * @file rope.cu
 * @brief Rotary Position Embedding (RoPE) CUDA kernel.
 *
 * Qwen2 RoPE uses the "half-split" dimension pairing convention:
 *   pair (d, d + half_dim)  for d = 0 .. half_dim-1
 *
 * For position p and dimension pair d:
 *   freq   = theta^(-2d / head_dim)
 *   angle  = pos * freq
 *   x_new[d]             = x[d]             * cos(angle) - x[d+half_dim] * sin(angle)
 *   x_new[d + half_dim]  = x[d]             * sin(angle) + x[d+half_dim] * cos(angle)
 *
 * LAYOUT: x is [seq_len, n_heads, head_dim] FP16 (seq-major, heads interleaved).
 *         This matches GEMM output [seq_len, n_heads * head_dim].
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

/**
 * @brief Apply RoPE to Q or K tensor in-place.
 *
 * x: [seq_len, n_heads, head_dim] FP16  (seq-major)
 *
 * Grid:  (seq_len, n_heads)
 * Block: (head_dim / 2)
 *
 * Each thread handles one dimension pair (d, d + half_dim).
 */
__global__ void rope_kernel(
    half* __restrict__ x,
    int seq_len, int n_heads, int head_dim,
    int position_offset,
    float theta)
{
    int seq_idx = blockIdx.x;
    int head = blockIdx.y;
    int d = threadIdx.x;          /* 0 .. head_dim/2 - 1 */
    int half_dim = head_dim / 2;

    if (d >= half_dim) return;
    if (seq_idx >= seq_len) return;

    int pos = seq_idx + position_offset;

    /* Compute rotation angle — Qwen2 convention:
     * freq = theta^(-2d / head_dim)  = 1 / theta^(2d / head_dim)
     */
    float freq = powf(theta, -2.0f * (float)d / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    /* Index into [seq_len, n_heads, head_dim]  (seq-major, half-split) */
    int base = seq_idx * n_heads * head_dim + head * head_dim;
    int idx0 = base + d;                /* dimension d */
    int idx1 = base + d + half_dim;     /* dimension d + half_dim */

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
 * @param x              [seq_len, n_heads, head_dim] FP16, modified in-place
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
