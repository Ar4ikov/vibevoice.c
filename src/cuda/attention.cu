/**
 * @file attention.cu
 * @brief Flash Attention 2 GQA kernels — prefill (tiled online softmax)
 *        and decode (tiled with online softmax, no O(n) shared mem).
 *
 * Qwen2 attention: 28 query heads, 4 key-value heads (GQA ratio 7:1).
 * head_dim = 128.
 *
 * MEMORY LAYOUT (seq-major, matching GEMM output):
 *   Q:  [seq_len, n_q_heads,  head_dim]   — stride between positions = n_q_heads * head_dim
 *   K:  [seq_len, n_kv_heads, head_dim]   — stride between positions = n_kv_heads * head_dim
 *   V:  [seq_len, n_kv_heads, head_dim]   — same
 *   O:  [seq_len, n_q_heads,  head_dim]   — same as Q
 *   KV-cache: [cache_len, n_kv_heads, head_dim]  — written by kv_cache_append
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

/* ─── Tile sizes ─────────────────────────────────────────────────────────── */

#define FA2_BR          4      /* Q rows per block (= warps per block)     */
#define FA2_BC          32     /* K/V positions per tile (= warp size)     */
#define FA2_THREADS     128    /* blockDim = (32, 4) = FA2_BC * FA2_BR     */

#define DECODE_THREADS  128    /* 4 warps for decode                       */

/* ─── Warp-level reduce helpers ──────────────────────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Flash Attention 2 — PREFILL (multi-token query)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Layout: Q/K/V/O are seq-major [seq_len, n_heads, head_dim].
 *
 * Grid:  (n_q_heads, ceil(seq_len / FA2_BR))
 * Block: (32, FA2_BR) = (32, 4) = 128 threads
 *
 * Each warp (32 threads) processes one Q row.
 * Each thread handles (head_dim / 32) elements of the output vector.
 *
 * Shared memory:
 *   K_tile [FA2_BC][head_dim] half   (32 * 128 * 2 =  8 KB)
 *   V_tile [FA2_BC][head_dim] half   (32 * 128 * 2 =  8 KB)
 *   Q_smem [FA2_BR][head_dim] half   ( 4 * 128 * 2 =  1 KB)
 *   Total: ~17 KB  (fits in default 48 KB shared)
 */
__global__ void flash_attn2_prefill_kernel(
    const half* __restrict__ Q,     /* [seq_len, n_q_heads, head_dim]  */
    const half* __restrict__ K,     /* [seq_len, n_kv_heads, head_dim] */
    const half* __restrict__ V,     /* [seq_len, n_kv_heads, head_dim] */
    half* __restrict__ O,           /* [seq_len, n_q_heads, head_dim]  */
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, float scale, bool causal)
{
    const int head     = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int warp_id  = threadIdx.y;   /* 0 .. FA2_BR-1 */
    const int lane     = threadIdx.x;   /* 0 .. 31       */
    const int kv_head  = head / (n_q_heads / n_kv_heads);

    const int q_row = tile_row * FA2_BR + warp_id;
    if (q_row >= seq_len) return;

    const int max_kv = causal ? (q_row + 1) : seq_len;
    const int dims_per_thread = head_dim / 32;
    const int dim_base = lane * dims_per_thread;

    /* Strides for seq-major layout */
    const int q_stride = n_q_heads * head_dim;   /* stride between positions in Q/O */
    const int kv_stride = n_kv_heads * head_dim;  /* stride between positions in K/V */

    /* ── Shared memory ── */
    extern __shared__ char smem_raw[];
    half* K_tile = (half*)smem_raw;
    half* V_tile = K_tile + FA2_BC * head_dim;
    half* Q_smem = V_tile + FA2_BC * head_dim;

    /* ── Load Q row into shared memory ── */
    /* Q[q_row, head, :] is at offset: q_row * q_stride + head * head_dim */
    {
        const int q_offset = q_row * q_stride + head * head_dim;
        for (int d = lane; d < head_dim; d += 32) {
            Q_smem[warp_id * head_dim + d] = Q[q_offset + d];
        }
    }
    __syncthreads();

    /* ── Per-thread accumulators (registers) ── */
    float o_reg[8];
    for (int d = 0; d < dims_per_thread; d++) o_reg[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    /* ── Iterate over KV tiles ── */
    const int num_kv_tiles = (max_kv + FA2_BC - 1) / FA2_BC;

    for (int tj = 0; tj < num_kv_tiles; tj++) {
        const int kv_start = tj * FA2_BC;
        const int tile_len = ((kv_start + FA2_BC) <= max_kv)
                             ? FA2_BC : (max_kv - kv_start);

        /* ── Cooperative load K_tile and V_tile ── */
        /* K[pos, kv_head, :] at offset: pos * kv_stride + kv_head * head_dim */
        {
            const int total_elems = FA2_BC * head_dim;
            const int tid_flat = warp_id * 32 + lane;
            for (int idx = tid_flat; idx < total_elems; idx += FA2_THREADS) {
                int r = idx / head_dim;    /* row within tile (0..31) */
                int c = idx % head_dim;    /* dim within head */
                int global_pos = kv_start + r;
                if (global_pos < seq_len) {
                    int kv_off = global_pos * kv_stride + kv_head * head_dim + c;
                    K_tile[r * head_dim + c] = K[kv_off];
                    V_tile[r * head_dim + c] = V[kv_off];
                } else {
                    K_tile[r * head_dim + c] = __float2half(0.0f);
                    V_tile[r * head_dim + c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        /* ── Compute scores for this tile ── */
        float score;
        {
            float dot = 0.0f;
            const half* q_ptr = Q_smem + warp_id * head_dim;
            const half* k_ptr = K_tile + lane * head_dim;
            if (lane < tile_len) {
                for (int d = 0; d < head_dim; d++) {
                    dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
                }
                dot *= scale;
                if (causal && (kv_start + lane) > q_row) {
                    dot = -FLT_MAX;
                }
            } else {
                dot = -FLT_MAX;
            }
            score = dot;
        }

        /* ── Online softmax update ── */
        float tile_max = warp_reduce_max(score);
        float m_new = fmaxf(m_i, tile_max);

        float alpha = expf(m_i - m_new);
        for (int d = 0; d < dims_per_thread; d++) {
            o_reg[d] *= alpha;
        }
        l_i *= alpha;

        float p = (score > -FLT_MAX + 1.0f) ? expf(score - m_new) : 0.0f;

        float tile_sum = warp_reduce_sum(p);
        l_i += tile_sum;

        /* ── Accumulate P @ V_tile ── */
        for (int k = 0; k < tile_len; k++) {
            float pk = __shfl_sync(0xFFFFFFFF, p, k);
            if (pk > 0.0f) {
                const half* v_row = V_tile + k * head_dim + dim_base;
                for (int d = 0; d < dims_per_thread; d++) {
                    o_reg[d] += pk * __half2float(v_row[d]);
                }
            }
        }

        m_i = m_new;
        __syncthreads();
    }

    /* ── Final normalization and write output ── */
    /* O[q_row, head, :] at offset: q_row * q_stride + head * head_dim */
    float inv_l = (l_i > 1e-8f) ? (1.0f / l_i) : 0.0f;
    const int out_offset = q_row * q_stride + head * head_dim;
    for (int d = 0; d < dims_per_thread; d++) {
        O[out_offset + dim_base + d] = __float2half(o_reg[d] * inv_l);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Flash Decode — single-token attention with online softmax
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Q layout:        [1, n_q_heads, head_dim] = [n_q_heads * head_dim] flat
 * KV-cache layout: [cache_len, n_kv_heads, head_dim]  (seq-major)
 * Output layout:   [1, n_q_heads, head_dim] = [n_q_heads * head_dim] flat
 *
 * Grid:  (n_q_heads)
 * Block: (DECODE_THREADS) = 128 threads
 *
 * Each thread handles one output dimension (head_dim <= DECODE_THREADS).
 * Uses online softmax, iterating over the full KV cache.
 *
 * Shared memory:
 *   q_shared [head_dim] half        (128 * 2 = 256 bytes)
 *   reduce_buf [DECODE_THREADS]     (128 * 4 = 512 bytes)
 *   Total: < 1 KB
 */
__global__ void flash_decode_attention_kernel(
    const half* __restrict__ q,         /* [n_q_heads, head_dim] flat          */
    const half* __restrict__ k_cache,   /* [cache_len, n_kv_heads, head_dim]   */
    const half* __restrict__ v_cache,   /* [cache_len, n_kv_heads, head_dim]   */
    half* __restrict__ output,          /* [n_q_heads, head_dim] flat          */
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, float scale)
{
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = q_head / (n_q_heads / n_kv_heads);

    /* Stride between positions in KV cache (seq-major) */
    const int kv_stride = n_kv_heads * head_dim;

    __shared__ half q_shared[256];      /* head_dim <= 256 */
    __shared__ float reduce_buf[DECODE_THREADS];

    /* Load Q into shared */
    if (tid < head_dim) {
        q_shared[tid] = q[q_head * head_dim + tid];
    }
    __syncthreads();

    const bool active = (tid < head_dim);

    /* Online softmax accumulators */
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o_acc = 0.0f;

    for (int t = 0; t < cache_len; t++) {
        /* K[t, kv_head, d] at offset: t * kv_stride + kv_head * head_dim + d */
        const int kv_offset = t * kv_stride + kv_head * head_dim;

        /* Compute dot product Q · K[t] — all threads participate */
        float partial = 0.0f;
        if (active) {
            partial = __half2float(q_shared[tid])
                    * __half2float(k_cache[kv_offset + tid]);
        }

        /* Block-level reduction for dot product */
        reduce_buf[tid] = partial;
        __syncthreads();
        for (int s = DECODE_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        float dot = reduce_buf[0] * scale;

        /* Online softmax update */
        float m_new = fmaxf(m_i, dot);
        float alpha_val = expf(m_i - m_new);
        float p = expf(dot - m_new);

        if (active) {
            o_acc = alpha_val * o_acc
                  + p * __half2float(v_cache[kv_offset + tid]);
        }
        l_i = alpha_val * l_i + p;
        m_i = m_new;
    }

    /* Final normalization and write */
    if (active) {
        float inv_l = (l_i > 1e-8f) ? (1.0f / l_i) : 0.0f;
        output[q_head * head_dim + tid] = __float2half(o_acc * inv_l);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C wrappers
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

#include "vibevoice/types.h"

vv_status_t vv_gqa_attention_decode_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, void* stream)
{
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;

    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(n_q_heads);
    dim3 block(DECODE_THREADS);
    size_t shared_bytes = (size_t)head_dim * 2 + DECODE_THREADS * sizeof(float);

    flash_decode_attention_kernel<<<grid, block, shared_bytes,
                                     (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache,
        (half*)output,
        n_q_heads, n_kv_heads, head_dim, cache_len, scale);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_prefill_cuda(
    const void* q, const void* k, const void* v,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream)
{
    if (!q || !k || !v || !output) return VV_ERR_NULL_PTR;
    if (seq_len == 0) return VV_OK;

    float scale = 1.0f / sqrtf((float)head_dim);

    int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
    dim3 grid(n_q_heads, num_q_tiles);
    dim3 block(32, FA2_BR);

    size_t shared_bytes = (size_t)(2 * FA2_BC + FA2_BR) * head_dim * sizeof(half);

    flash_attn2_prefill_kernel<<<grid, block, shared_bytes,
                                  (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k, (const half*)v, (half*)output,
        n_q_heads, n_kv_heads, head_dim, seq_len, scale, causal);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
