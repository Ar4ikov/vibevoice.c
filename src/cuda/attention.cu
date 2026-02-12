/**
 * @file attention.cu
 * @brief Grouped-Query Attention (GQA) CUDA kernel.
 *
 * Qwen2 attention: 28 query heads, 4 key-value heads (GQA ratio 7:1).
 * head_dim = 128.
 *
 * For decode phase (single token query), this is essentially a
 * batched dot-product attention over the KV-cache.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

/**
 * @brief GQA attention for decode (single query token).
 *
 * q:     [n_q_heads, 1, head_dim] FP16
 * k_cache: [n_kv_heads, cache_len, head_dim] FP16
 * v_cache: [n_kv_heads, cache_len, head_dim] FP16
 * output:  [n_q_heads, 1, head_dim] FP16
 *
 * Grid: (n_q_heads, 1, 1)
 * Block: (256, 1, 1) — one block per query head
 */
__global__ void gqa_decode_attention_kernel(
    const half* __restrict__ q,         /* [n_q_heads, head_dim] */
    const half* __restrict__ k_cache,   /* [n_kv_heads, cache_len, head_dim] */
    const half* __restrict__ v_cache,   /* [n_kv_heads, cache_len, head_dim] */
    half* __restrict__ output,          /* [n_q_heads, head_dim] */
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len,
    float scale)
{
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    int kv_head = q_head / (n_q_heads / n_kv_heads);  /* GQA mapping */

    extern __shared__ float shared[];
    float* attn_scores = shared;  /* [cache_len] */

    /* Step 1: Compute attention scores (Q · K^T) */
    const half* q_vec = q + q_head * head_dim;
    const half* k_base = k_cache + kv_head * cache_len * head_dim;

    /* Each thread computes scores for a subset of cache positions */
    float max_score = -FLT_MAX;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        const half* k_vec = k_base + t * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        dot *= scale;
        attn_scores[t] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    /* Find global max for numerical stability */
    __shared__ float s_max[256];
    s_max[tid] = max_score;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid])
            s_max[tid] = s_max[tid + s];
        __syncthreads();
    }
    float global_max = s_max[0];

    /* Step 2: Softmax */
    float sum_exp = 0.0f;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        float e = expf(attn_scores[t] - global_max);
        attn_scores[t] = e;
        sum_exp += e;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_sum[0] + 1e-8f);

    for (int t = tid; t < cache_len; t += blockDim.x) {
        attn_scores[t] *= inv_sum;
    }
    __syncthreads();

    /* Step 3: Weighted sum of V (attention @ V) */
    const half* v_base = v_cache + kv_head * cache_len * head_dim;
    half* out_vec = output + q_head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            sum += attn_scores[t] * __half2float(v_base[t * head_dim + d]);
        }
        out_vec[d] = __float2half(sum);
    }
}

/**
 * @brief GQA attention for prefill (multiple query tokens).
 *
 * Simple tiled implementation for prefill phase.
 * For large sequence lengths, Flash Attention should be used instead.
 */
__global__ void gqa_prefill_attention_kernel(
    const half* __restrict__ q,         /* [n_q_heads, seq_len, head_dim] */
    const half* __restrict__ k,         /* [n_kv_heads, seq_len, head_dim] */
    const half* __restrict__ v,         /* [n_kv_heads, seq_len, head_dim] */
    half* __restrict__ output,          /* [n_q_heads, seq_len, head_dim] */
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len,
    float scale,
    bool causal)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    int tid = threadIdx.x;
    int kv_head = q_head / (n_q_heads / n_kv_heads);

    if (q_pos >= seq_len) return;

    int max_k = causal ? (q_pos + 1) : seq_len;

    extern __shared__ float shared[];
    float* scores = shared;

    const half* q_vec = q + q_head * seq_len * head_dim + q_pos * head_dim;
    const half* k_base = k + kv_head * seq_len * head_dim;

    /* Compute scores */
    float max_s = -FLT_MAX;
    for (int t = tid; t < max_k; t += blockDim.x) {
        const half* k_vec = k_base + t * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        dot *= scale;
        scores[t] = dot;
        if (dot > max_s) max_s = dot;
    }
    /* Mask future positions */
    for (int t = tid; t < seq_len; t += blockDim.x) {
        if (t >= max_k) scores[t] = -FLT_MAX;
    }
    __syncthreads();

    /* Max reduction */
    __shared__ float s_max[256];
    s_max[tid] = max_s;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid])
            s_max[tid] = s_max[tid + s];
        __syncthreads();
    }
    float gmax = s_max[0];

    /* Softmax */
    float sum_e = 0.0f;
    for (int t = tid; t < max_k; t += blockDim.x) {
        float e = expf(scores[t] - gmax);
        scores[t] = e;
        sum_e += e;
    }
    for (int t = tid; t < seq_len; t += blockDim.x) {
        if (t >= max_k) scores[t] = 0.0f;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum_e;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float inv_s = 1.0f / (s_sum[0] + 1e-8f);

    for (int t = tid; t < max_k; t += blockDim.x) {
        scores[t] *= inv_s;
    }
    __syncthreads();

    /* Weighted V */
    const half* v_base = v + kv_head * seq_len * head_dim;
    half* out = output + q_head * seq_len * head_dim + q_pos * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < max_k; t++) {
            sum += scores[t] * __half2float(v_base[t * head_dim + d]);
        }
        out[d] = __float2half(sum);
    }
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief GQA attention for decode (single new token).
 */
vv_status_t vv_gqa_attention_decode_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, void* stream)
{
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;

    float scale = 1.0f / sqrtf((float)head_dim);
    int threads = 256;
    size_t shared_bytes = (size_t)cache_len * sizeof(float);

    /* Cap shared memory; fall back for very long sequences */
    if (shared_bytes > 48 * 1024) {
        /* For very long cache, would need tiled approach */
        shared_bytes = 48 * 1024;
    }

    dim3 grid(n_q_heads);
    dim3 block(threads);

    gqa_decode_attention_kernel<<<grid, block, shared_bytes,
                                   (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache,
        (half*)output,
        n_q_heads, n_kv_heads, head_dim, cache_len, scale);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief GQA attention for prefill (full sequence).
 */
vv_status_t vv_gqa_attention_prefill_cuda(
    const void* q, const void* k, const void* v,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream)
{
    if (!q || !k || !v || !output) return VV_ERR_NULL_PTR;

    float scale = 1.0f / sqrtf((float)head_dim);
    int threads = 256;
    if (head_dim < 256) threads = head_dim;

    size_t shared_bytes = (size_t)seq_len * sizeof(float);

    dim3 grid(n_q_heads, seq_len);
    dim3 block(threads);

    gqa_prefill_attention_kernel<<<grid, block, shared_bytes,
                                    (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k, (const half*)v, (half*)output,
        n_q_heads, n_kv_heads, head_dim, seq_len, scale, causal);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
