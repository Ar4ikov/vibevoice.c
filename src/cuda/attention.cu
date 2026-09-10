/**
 * @file attention.cu
 * @brief GQA attention — tiled prefill with online softmax and a split-KV
 *        flash decode.
 *
 * Qwen2 attention: 28 query heads, 4 key-value heads (GQA ratio 7:1),
 * head_dim = 128.
 *
 * MEMORY LAYOUT (seq-major, matching GEMM output):
 *   Q:  [seq_len, n_q_heads,  head_dim]
 *   K:  [seq_len, n_kv_heads, head_dim]
 *   V:  [seq_len, n_kv_heads, head_dim]
 *   O:  [seq_len, n_q_heads,  head_dim]
 *   KV-cache: [cache_len, n_kv_heads, head_dim]  — written by kv_cache_append
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

/* ─── Tile sizes ─────────────────────────────────────────────────────────── */

#define FA2_BR          4      /* Q rows per block (= warps per block)     */
#define FA2_BC          32     /* K/V positions per tile (= warp size)     */
#define FA2_THREADS     128    /* blockDim = (32, 4)                       */

#define DECODE_WARPS    4
#define DECODE_DPT      4      /* head_dim / 32 — dims held per lane       */

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
 * PREFILL — multi-token query, tiled online softmax
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Grid:  (n_q_heads, ceil(seq_len / FA2_BR))
 * Block: (32, FA2_BR) = 128 threads; one warp per query row.
 *
 * Shared memory: K_tile + V_tile + Q_smem = (2*32 + 4) * 128 halves ≈ 17 KB.
 */
__global__ void flash_attn2_prefill_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, float scale, bool causal)
{
    const int head     = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;
    const int kv_head  = head / (n_q_heads / n_kv_heads);

    const int q_row  = tile_row * FA2_BR + warp_id;
    const bool alive = (q_row < q_len);
    const int q_abs  = q_offset + q_row;

    /*
     * The KV loop bound must be UNIFORM across the block: every warp has to
     * execute the same number of __syncthreads() calls. Deriving it from the
     * per-row causal limit (q_row + 1) desynchronises the barriers and
     * silently corrupts the shared K/V tiles. Per-row causality is applied as
     * a score mask instead.
     */
    const int tile_last = tile_row * FA2_BR + FA2_BR - 1;
    int block_max_kv = causal ? (q_offset + tile_last + 1) : kv_len;
    if (block_max_kv > kv_len) block_max_kv = kv_len;

    const int dims_per_thread = head_dim / 32;
    const int dim_base = lane * dims_per_thread;

    const int q_stride  = n_q_heads  * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    extern __shared__ char smem_raw[];
    half* K_tile = (half*)smem_raw;
    half* V_tile = K_tile + FA2_BC * head_dim;
    half* Q_smem = V_tile + FA2_BC * head_dim;

    {
        const int q_base = q_row * q_stride + head * head_dim;
        for (int d = lane; d < head_dim; d += 32)
            Q_smem[warp_id * head_dim + d] =
                alive ? Q[q_base + d] : __float2half(0.0f);
    }
    __syncthreads();

    float o_reg[8];
    for (int d = 0; d < dims_per_thread; d++) o_reg[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const int num_kv_tiles = (block_max_kv + FA2_BC - 1) / FA2_BC;

    for (int tj = 0; tj < num_kv_tiles; tj++) {
        const int kv_start = tj * FA2_BC;

        {
            const int total_elems = FA2_BC * head_dim;
            const int tid_flat = warp_id * 32 + lane;
            for (int idx = tid_flat; idx < total_elems; idx += FA2_THREADS) {
                int r = idx / head_dim;
                int c = idx % head_dim;
                int global_pos = kv_start + r;
                if (global_pos < kv_len) {
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

        const int kv_pos = kv_start + lane;
        float score = -FLT_MAX;
        if (alive && kv_pos < kv_len && (!causal || kv_pos <= q_abs)) {
            const half* q_ptr = Q_smem + warp_id * head_dim;
            const half* k_ptr = K_tile + lane * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
            score = dot * scale;
        }

        const float tile_max = warp_reduce_max(score);
        const float m_new = fmaxf(m_i, tile_max);
        const bool  have_any = (m_new > -FLT_MAX);

        const float alpha = (have_any && m_i > -FLT_MAX)
                            ? __expf(m_i - m_new) : 1.0f;
        for (int d = 0; d < dims_per_thread; d++) o_reg[d] *= alpha;
        l_i *= alpha;

        const float p = (score > -FLT_MAX && have_any)
                        ? __expf(score - m_new) : 0.0f;
        l_i += warp_reduce_sum(p);

        const int tile_valid = min(FA2_BC, kv_len - kv_start);
        for (int k = 0; k < tile_valid; k++) {
            const float pk = __shfl_sync(0xFFFFFFFF, p, k);
            if (pk != 0.0f) {
                const half* v_row = V_tile + k * head_dim + dim_base;
                for (int d = 0; d < dims_per_thread; d++)
                    o_reg[d] += pk * __half2float(v_row[d]);
            }
        }

        if (have_any) m_i = m_new;
        __syncthreads();
    }

    if (alive) {
        const float inv_l = (l_i > 1e-20f) ? (1.0f / l_i) : 0.0f;
        const int out_offset = q_row * q_stride + head * head_dim;
        for (int d = 0; d < dims_per_thread; d++)
            O[out_offset + dim_base + d] = __float2half(o_reg[d] * inv_l);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DECODE — single-token query, KV range split across warps
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The straightforward version walked the cache one position at a time with a
 * full block reduction (plus two __syncthreads) per position and only
 * n_heads blocks of parallelism — hopeless once the cache holds a few
 * thousand speech frames. Here each warp owns a contiguous slice and keeps
 * its own online-softmax state (m, l, o); a second kernel merges the slices.
 * Lane l holds dims 4l..4l+3, so every K/V read is one coalesced 256-byte
 * transaction per position.
 */
__global__ void flash_decode_split_kernel(
    const half* __restrict__ q,         /* [n_q_heads, head_dim]             */
    const half* __restrict__ k_cache,   /* [cache_len, n_kv_heads, head_dim] */
    const half* __restrict__ v_cache,
    float* __restrict__ part_o,         /* [n_q_heads, n_parts, head_dim]    */
    float* __restrict__ part_m,         /* [n_q_heads, n_parts]              */
    float* __restrict__ part_l,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, int n_parts, float scale)
{
    const int q_head  = blockIdx.x;
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int part    = blockIdx.y * DECODE_WARPS + warp_id;
    if (part >= n_parts) return;

    const int kv_head   = q_head / (n_q_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int d0        = lane * DECODE_DPT;

    const int chunk = (cache_len + n_parts - 1) / n_parts;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > cache_len) end = cache_len;

    float o_acc[DECODE_DPT];
#pragma unroll
    for (int d = 0; d < DECODE_DPT; d++) o_acc[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    float qreg[DECODE_DPT];
    {
        const half* qh = q + (size_t)q_head * head_dim + d0;
#pragma unroll
        for (int d = 0; d < DECODE_DPT; d++) qreg[d] = __half2float(qh[d]);
    }

    for (int t = begin; t < end; t++) {
        const size_t off = (size_t)t * kv_stride
                         + (size_t)kv_head * head_dim + d0;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < DECODE_DPT; d++)
            dot = fmaf(qreg[d], __half2float(k_cache[off + d]), dot);
        dot = warp_reduce_sum(dot) * scale;

        const float m_new = fmaxf(m_i, dot);
        const float alpha = (m_i > -FLT_MAX) ? __expf(m_i - m_new) : 0.0f;
        const float p     = __expf(dot - m_new);
#pragma unroll
        for (int d = 0; d < DECODE_DPT; d++)
            o_acc[d] = o_acc[d] * alpha + p * __half2float(v_cache[off + d]);
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    float* po = part_o + ((size_t)q_head * n_parts + part) * head_dim + d0;
#pragma unroll
    for (int d = 0; d < DECODE_DPT; d++) po[d] = o_acc[d];
    if (lane == 0) {
        part_m[q_head * n_parts + part] = (end > begin) ? m_i : -FLT_MAX;
        part_l[q_head * n_parts + part] = l_i;
    }
}

/** @brief Merge per-slice online-softmax states into the final output. */
__global__ void flash_decode_combine_kernel(
    const float* __restrict__ part_o,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    half* __restrict__ output,
    int n_parts, int head_dim)
{
    const int q_head = blockIdx.x;
    const int d      = threadIdx.x;

    extern __shared__ float sh[];
    float* s_m = sh;
    float* s_l = sh + n_parts;

    for (int i = d; i < n_parts; i += blockDim.x) {
        s_m[i] = part_m[q_head * n_parts + i];
        s_l[i] = part_l[q_head * n_parts + i];
    }
    __syncthreads();

    if (d >= head_dim) return;

    float gmax = -FLT_MAX;
    for (int i = 0; i < n_parts; i++) gmax = fmaxf(gmax, s_m[i]);

    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n_parts; i++) {
        if (s_m[i] <= -FLT_MAX) continue;
        const float w = __expf(s_m[i] - gmax);
        num += w * part_o[((size_t)q_head * n_parts + i) * head_dim + d];
        den += w * s_l[i];
    }
    output[q_head * head_dim + d] =
        __float2half(den > 1e-20f ? num / den : 0.0f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C wrappers
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

#include "vibevoice/types.h"

/*
 * Split-decode scratch: allocated on first use and reused for the session —
 * the decode hot path must never call cudaMalloc.
 */
#define VV_DECODE_MAX_PARTS 64
static float* s_part_o = NULL;
static float* s_part_m = NULL;
static float* s_part_l = NULL;
static int    s_part_heads = 0;
static int    s_part_dim   = 0;

static vv_status_t ensure_decode_scratch(int n_heads, int head_dim) {
    if (s_part_o && n_heads <= s_part_heads && head_dim <= s_part_dim)
        return VV_OK;
    if (s_part_o) { cudaFree(s_part_o); cudaFree(s_part_m); cudaFree(s_part_l); }
    s_part_heads = n_heads;
    s_part_dim   = head_dim;
    size_t no = (size_t)n_heads * VV_DECODE_MAX_PARTS * head_dim * sizeof(float);
    size_t nm = (size_t)n_heads * VV_DECODE_MAX_PARTS * sizeof(float);
    if (cudaMalloc((void**)&s_part_o, no) != cudaSuccess) return VV_ERR_CUDA_OOM;
    if (cudaMalloc((void**)&s_part_m, nm) != cudaSuccess) return VV_ERR_CUDA_OOM;
    if (cudaMalloc((void**)&s_part_l, nm) != cudaSuccess) return VV_ERR_CUDA_OOM;
    return VV_OK;
}

void vv_attention_cleanup(void) {
    if (s_part_o) { cudaFree(s_part_o); s_part_o = NULL; }
    if (s_part_m) { cudaFree(s_part_m); s_part_m = NULL; }
    if (s_part_l) { cudaFree(s_part_l); s_part_l = NULL; }
    s_part_heads = s_part_dim = 0;
}

vv_status_t vv_gqa_attention_decode_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, void* stream)
{
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;
    if (head_dim != DECODE_DPT * 32) return VV_ERR_UNSUPPORTED;

    vv_status_t st = ensure_decode_scratch(n_q_heads, head_dim);
    if (st != VV_OK) return st;

    /* ~512 cache positions per warp, rounded up to whole blocks. */
    int n_parts = (cache_len + 511) / 512;
    if (n_parts < 1) n_parts = 1;
    if (n_parts > VV_DECODE_MAX_PARTS) n_parts = VV_DECODE_MAX_PARTS;
    n_parts = ((n_parts + DECODE_WARPS - 1) / DECODE_WARPS) * DECODE_WARPS;

    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(n_q_heads, n_parts / DECODE_WARPS);
    dim3 block(32, DECODE_WARPS);
    flash_decode_split_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache,
        s_part_o, s_part_m, s_part_l,
        n_q_heads, n_kv_heads, head_dim, cache_len, n_parts, scale);

    size_t shbytes = (size_t)n_parts * 2 * sizeof(float);
    int cthreads = head_dim > n_parts ? head_dim : n_parts;
    flash_decode_combine_kernel<<<n_q_heads, cthreads, shbytes,
                                  (cudaStream_t)stream>>>(
        s_part_o, s_part_m, s_part_l, (half*)output, n_parts, head_dim);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Multi-token attention against a KV cache.
 *
 * @param q         [q_len, n_q_heads, head_dim] queries for this chunk
 * @param k_cache   [kv_len, n_kv_heads, head_dim] keys, absolute positions
 * @param v_cache   matching values
 * @param q_offset  absolute position of the first query row
 * @param kv_len    number of valid cache positions (>= q_offset + q_len)
 *
 * Chunked prefill needs this: the queries of chunk N must see every key from
 * position 0, not just the ones inside the chunk.
 */
vv_status_t vv_gqa_attention_prefill_cached_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal, void* stream)
{
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;
    if (q_len == 0 || kv_len == 0) return VV_OK;

    float scale = 1.0f / sqrtf((float)head_dim);

    int num_q_tiles = (q_len + FA2_BR - 1) / FA2_BR;
    dim3 grid(n_q_heads, num_q_tiles);
    dim3 block(32, FA2_BR);

    size_t shared_bytes = (size_t)(2 * FA2_BC + FA2_BR) * head_dim * sizeof(half);

    flash_attn2_prefill_kernel<<<grid, block, shared_bytes,
                                  (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache,
        (half*)output, n_q_heads, n_kv_heads, head_dim,
        q_len, q_offset, kv_len, scale, causal);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_prefill_cuda(
    const void* q, const void* k, const void* v,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream)
{
    return vv_gqa_attention_prefill_cached_cuda(
        q, k, v, output, n_q_heads, n_kv_heads, head_dim,
        seq_len, 0, seq_len, causal, stream);
}

} /* extern "C" */
