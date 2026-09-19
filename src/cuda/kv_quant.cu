/**
 * @file kv_quant.cu
 * @brief Quantized KV-cache storage and the attention kernels that read it.
 *
 * See include/vibevoice/kv_quant.h for the format definitions and for why the
 * Hadamard rotation lets attention work directly on stored values.
 *
 * Layout, per layer:
 *   store: [max_seq_len][n_kv_heads][bytes_per_vec]  bytes
 *   meta:  [max_seq_len][n_kv_heads]                 FP16 RMS (TurboQuant only)
 *
 * Sub-byte codes are laid out so that lane L of a warp owns dims 4L..4L+3 —
 * the same split the FP16 decode kernel uses — which means its codes are a
 * contiguous run of 4*bits bits at offset 4*bits*L. The warp loads the whole
 * vector as uint32 words (one per lane, fully coalesced) and each lane then
 * shuffles the one or two words its bits land in.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include "decode_split.h"

#include "kv_codec.cuh"


/* ─── Store kernel ───────────────────────────────────────────────────────── */

/**
 * One warp handles one (position, kv_head) vector of both K and V.
 * Grid is flattened over positions x heads x {K, V}.
 */
template <int FMT>
__global__ void kv_store_kernel(
    const half* __restrict__ src,      /* [n_pos, n_kv_heads, 128]  */
    uint8_t* __restrict__ store,       /* [max_pos, n_kv_heads, BPV] */
    half* __restrict__ meta,           /* [max_pos, n_kv_heads] or NULL */
    const half* __restrict__ ref,      /* [n_kv_heads, 128] or NULL  */
    int n_kv_heads, int head_dim, int pos0,
    const int* __restrict__ d_pos0, int n_pos, int bpv,
    const int* __restrict__ page_table)
{
    const int warp_global = blockIdx.x * (blockDim.y) + threadIdx.y;
    const int total = n_pos * n_kv_heads;
    if (warp_global >= total) return;

    /* Device-side when a graph recorded this launch; see rope_kernel. */
    if (d_pos0) pos0 = *d_pos0;

    const int p    = warp_global / n_kv_heads;
    const int h    = warp_global % n_kv_heads;
    const int lane = threadIdx.x;
    const int d0   = lane * 4;

    const half* s = src + ((size_t)p * n_kv_heads + h) * head_dim + d0;
    const int row = kv_phys_row(page_table, pos0 + p);
    uint8_t* dst = store + ((size_t)row * n_kv_heads + h) * (size_t)bpv;

    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __half2float(s[i]);

    if (ref) {
        const half* r = ref + (size_t)h * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] -= __half2float(r[i]);
    }

    if (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2) {
        uint32_t packed = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (FMT == VV_KV_FP8_E4M3) ? f2e4m3(v[i]) : f2e5m2(v[i]);
            packed |= b << (8 * i);
        }
        ((uint32_t*)dst)[lane] = packed;
        return;
    }

    /* TurboQuant: rotate, then quantize against the vector's own RMS. */
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
    warp_wht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;

    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) ss = fmaf(v[i], v[i], ss);
    const float sigma = sqrtf(fmaxf(warp_sum(ss) / (float)head_dim, 1e-12f));

    const uint32_t bits = pack4<FMT>(v, 1.0f / sigma);

    /* Lanes write overlapping words, so assemble through shared memory. */
    extern __shared__ uint32_t s_pack[];
    uint32_t* buf = s_pack + threadIdx.y * 16;
    if (lane < 16) buf[lane] = 0u;
    __syncwarp();

    const int nb = KVBits<FMT>::N;
    const int bit = nb * lane;
    atomicOr(&buf[bit >> 5], bits << (bit & 31));
    if ((bit & 31) + nb > 32)
        atomicOr(&buf[(bit >> 5) + 1], bits >> (32 - (bit & 31)));
    __syncwarp();

    const int nwords = bpv / 4;
    if (lane < nwords) ((uint32_t*)dst)[lane] = buf[lane];
    if (lane == 0 && meta)
        meta[(size_t)row * n_kv_heads + h] = __float2half(sigma);
}

/**
 * @brief FP16 store: a straight copy, except that each position lands on the
 *        row its page says. One thread moves 16 bytes.
 */
__global__ void kv_store_raw_kernel(
    const uint4* __restrict__ k_src, const uint4* __restrict__ v_src,
    uint4* __restrict__ k_dst, uint4* __restrict__ v_dst,
    int n_kv_heads, int chunks, int pos0, const int* __restrict__ d_pos0,
    int n_pos, const int* __restrict__ page_table)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per_pos = (long long)n_kv_heads * chunks;
    if (i >= per_pos * n_pos) return;
    if (d_pos0) pos0 = *d_pos0;
    const int p = (int)(i / per_pos);
    const long long r = i - (long long)p * per_pos;
    const long long dst = (long long)kv_phys_row(page_table, pos0 + p) * per_pos + r;
    k_dst[dst] = k_src[i];
    v_dst[dst] = v_src[i];
}

#define VV_KV_MAP_BATCH 32

/** @brief Page ids for one map launch, passed by value. */
typedef struct { int id[VV_KV_MAP_BATCH]; } kv_page_ids_t;

__global__ void kv_page_map_kernel(int* __restrict__ table, kv_page_ids_t ids,
                                   int n)
{
    if ((int)threadIdx.x < n) table[threadIdx.x] = ids.id[threadIdx.x];
}

/**
 * @brief Mean key over a run of positions, one block per head.
 *
 * Called once per layer on the first prefill chunk. Any fixed vector would be
 * correct — softmax cancels it — so a mean over the first chunk is plenty.
 */
__global__ void kv_build_ref_kernel(
    const half* __restrict__ src, half* __restrict__ ref,
    int n_kv_heads, int head_dim, int n_pos)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    float acc = 0.0f;
    for (int p = 0; p < n_pos; ++p)
        acc += __half2float(src[((size_t)p * n_kv_heads + h) * head_dim + d]);
    ref[(size_t)h * head_dim + d] = __float2half(acc / (float)n_pos);
}

/* ─── Rotation kernels (applied to Q before, O after, attention) ─────────── */

__global__ void kv_rotate_kernel(half* __restrict__ x, int n_heads,
                                 int head_dim, int rows, bool inverse)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= rows * n_heads) return;
    const int lane = threadIdx.x;
    const int d0 = lane * 4;

    half* p = x + (size_t)idx * head_dim + d0;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __half2float(p[i]);

    if (!inverse) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;
    } else {
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __float2half(v[i]);
}

/* ─── Flash decode over a quantized cache ────────────────────────────────── */

#define QD_WARPS   VV_DECODE_WARPS

template <int FMT>
__global__ void q_decode_split_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ k_store,
    const uint8_t* __restrict__ v_store,
    const half* __restrict__ k_meta,
    const half* __restrict__ v_meta,
    float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, const int* __restrict__ d_cache_len,
    int n_parts, int bpv, float scale)
{
    const int q_head  = blockIdx.x;
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int part    = blockIdx.y * QD_WARPS + warp_id;
    if (part >= n_parts) return;

    if (d_cache_len) cache_len = *d_cache_len;

    const int kv_head = q_head / (n_q_heads / n_kv_heads);
    const int d0 = lane * 4;

    const int chunk = (cache_len + n_parts - 1) / n_parts;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > cache_len) end = cache_len;

    float o_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.0f;

    float qreg[4];
    {
        const half* qh = q + (size_t)q_head * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) qreg[i] = __half2float(qh[i]);
    }

    for (int t = begin; t < end; ++t) {
        const size_t vo = ((size_t)t * n_kv_heads + kv_head) * (size_t)bpv;
        const size_t mo = (size_t)t * n_kv_heads + kv_head;

        float kv[4];
        kv_read4<FMT>(k_store + vo, k_meta ? k_meta + mo : NULL, lane, kv);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dot = fmaf(qreg[i], kv[i], dot);
        dot = warp_sum(dot) * scale;

        const float m_new = fmaxf(m_i, dot);
        const float alpha = (m_i > -FLT_MAX) ? __expf(m_i - m_new) : 0.0f;
        const float p = __expf(dot - m_new);

        kv_read4<FMT>(v_store + vo, v_meta ? v_meta + mo : NULL, lane, kv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[i] = o_acc[i] * alpha + p * kv[i];
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    float* po = part_o + ((size_t)q_head * n_parts + part) * head_dim + d0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) po[i] = o_acc[i];
    if (lane == 0) {
        part_m[q_head * n_parts + part] = (end > begin) ? m_i : -FLT_MAX;
        part_l[q_head * n_parts + part] = l_i;
    }
}

__global__ void q_decode_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, half* __restrict__ output,
    int n_parts, int head_dim)
{
    const int q_head = blockIdx.x;
    const int d = threadIdx.x;

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
    for (int i = 0; i < n_parts; ++i) gmax = fmaxf(gmax, s_m[i]);

    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n_parts; ++i) {
        if (s_m[i] <= -FLT_MAX) continue;
        const float w = __expf(s_m[i] - gmax);
        num += w * part_o[((size_t)q_head * n_parts + i) * head_dim + d];
        den += w * s_l[i];
    }
    output[q_head * head_dim + d] = __float2half(den > 1e-20f ? num / den : 0.0f);
}

/**
 * @brief Unpack a stored cache back to FP16 — used by tests and by the CPU
 *        offload path, never on the GPU hot path.
 */
template <int FMT>
__global__ void kv_dequant_kernel(
    const uint8_t* __restrict__ store, const half* __restrict__ meta,
    const half* __restrict__ ref,
    half* __restrict__ out, int n_kv_heads, int head_dim, int n_pos, int bpv)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= n_pos * n_kv_heads) return;
    const int lane = threadIdx.x;
    const int d0 = lane * 4;

    float v[4];
    kv_read4<FMT>(store + (size_t)idx * bpv, meta ? meta + idx : NULL, lane, v);

    if (FMT >= VV_KV_TQ4) {          /* undo the rotation for inspection */
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }
    if (ref) {
        const half* r = ref + (size_t)(idx % n_kv_heads) * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += __half2float(r[i]);
    }
    half* p = out + (size_t)idx * head_dim + d0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __float2half(v[i]);
}

/* ─── Flash prefill over a quantized cache ───────────────────────────────── */

#define QP_BR      8
#define QP_BC      32
#define QP_THREADS 256
#define QP_KV_PAD  2

template <int FMT>
__global__ void q_prefill_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ k_store,
    const uint8_t* __restrict__ v_store,
    const half* __restrict__ k_meta,
    const half* __restrict__ v_meta,
    half* __restrict__ O,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, int bpv, float scale, bool causal)
{
    const int head     = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;
    const int kv_head  = head / (n_q_heads / n_kv_heads);

    const int q_row  = tile_row * QP_BR + warp_id;
    const bool alive = (q_row < q_len);
    const int q_abs  = q_offset + q_row;

    const int tile_last = tile_row * QP_BR + QP_BR - 1;
    int block_max_kv = causal ? (q_offset + tile_last + 1) : kv_len;
    if (block_max_kv > kv_len) block_max_kv = kv_len;

    const int q_stride = n_q_heads * head_dim;
    const int kv_row   = head_dim + QP_KV_PAD;

    extern __shared__ char smem_raw[];
    half* K_tile = (half*)smem_raw;
    half* V_tile = K_tile + QP_BC * kv_row;
    half* Q_smem = V_tile + QP_BC * kv_row;

    {
        const int q_base = q_row * q_stride + head * head_dim;
        for (int d = lane; d < head_dim; d += 32)
            Q_smem[warp_id * head_dim + d] =
                alive ? Q[q_base + d] : __float2half(0.0f);
    }
    __syncthreads();

    float o_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.0f;
    const int d0 = lane * 4;

    const int num_kv_tiles = (block_max_kv + QP_BC - 1) / QP_BC;

    for (int tj = 0; tj < num_kv_tiles; ++tj) {
        const int kv_start = tj * QP_BC;

        /* One warp per cached position: the packed layout is warp-shaped. */
        for (int r = warp_id; r < QP_BC; r += QP_BR) {
            const int gp = kv_start + r;
            float kk[4], vv[4];
            if (gp < kv_len) {
                const size_t vo = ((size_t)gp * n_kv_heads + kv_head) * (size_t)bpv;
                const size_t mo = (size_t)gp * n_kv_heads + kv_head;
                kv_read4<FMT>(k_store + vo, k_meta ? k_meta + mo : NULL, lane, kk);
                kv_read4<FMT>(v_store + vo, v_meta ? v_meta + mo : NULL, lane, vv);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) { kk[i] = 0.f; vv[i] = 0.f; }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                K_tile[r * kv_row + d0 + i] = __float2half(kk[i]);
                V_tile[r * kv_row + d0 + i] = __float2half(vv[i]);
            }
        }
        __syncthreads();

        const int kv_pos = kv_start + lane;
        float score = -FLT_MAX;
        if (alive && kv_pos < kv_len && (!causal || kv_pos <= q_abs)) {
            const half2* q_ptr = (const half2*)(Q_smem + warp_id * head_dim);
            const half2* k_ptr = (const half2*)(K_tile + lane * kv_row);
            float dot = 0.0f;
            for (int d = 0; d < head_dim / 2; ++d) {
                const float2 a = __half22float2(q_ptr[d]);
                const float2 b = __half22float2(k_ptr[d]);
                dot = fmaf(a.x, b.x, dot);
                dot = fmaf(a.y, b.y, dot);
            }
            score = dot * scale;
        }

        const float tile_max = warp_max(score);
        const float m_new = fmaxf(m_i, tile_max);
        const bool have_any = (m_new > -FLT_MAX);
        const float alpha = (have_any && m_i > -FLT_MAX) ? __expf(m_i - m_new) : 1.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_reg[i] *= alpha;
        l_i *= alpha;

        const float p = (score > -FLT_MAX && have_any) ? __expf(score - m_new) : 0.0f;
        l_i += warp_sum(p);

        const int tile_valid = min(QP_BC, kv_len - kv_start);
        for (int k = 0; k < tile_valid; ++k) {
            const float pk = __shfl_sync(0xFFFFFFFFu, p, k);
            if (pk != 0.0f) {
                const half* v_row = V_tile + k * kv_row + d0;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    o_reg[i] += pk * __half2float(v_row[i]);
            }
        }
        if (have_any) m_i = m_new;
        __syncthreads();
    }

    if (alive) {
        const float inv_l = (l_i > 1e-20f) ? (1.0f / l_i) : 0.0f;
        const int out_offset = q_row * q_stride + head * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            O[out_offset + d0 + i] = __float2half(o_reg[i] * inv_l);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C entry points
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

vv_status_t vv_kv_rotate_dev(void* x, int n_heads, int head_dim,
                             int rows, void* stream)
{
    if (!x) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int total = rows * n_heads;
    dim3 block(32, 4);
    kv_rotate_kernel<<<(total + 3) / 4, block, 0, (cudaStream_t)stream>>>(
        (half*)x, n_heads, head_dim, rows, false);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_kv_unrotate_dev(void* x, int n_heads, int head_dim,
                               int rows, void* stream)
{
    if (!x) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int total = rows * n_heads;
    dim3 block(32, 4);
    kv_rotate_kernel<<<(total + 3) / 4, block, 0, (cudaStream_t)stream>>>(
        (half*)x, n_heads, head_dim, rows, true);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

#define DISPATCH_Q(fmt, CALL)                                   \
    switch (fmt) {                                              \
        case VV_KV_FP8_E4M3: CALL(VV_KV_FP8_E4M3); break;       \
        case VV_KV_FP8_E5M2: CALL(VV_KV_FP8_E5M2); break;       \
        case VV_KV_TQ4:      CALL(VV_KV_TQ4);      break;       \
        case VV_KV_TQ3:      CALL(VV_KV_TQ3);      break;       \
        case VV_KV_TQ2:      CALL(VV_KV_TQ2);      break;       \
        case VV_KV_TQ1_5:    CALL(VV_KV_TQ1_5);    break;       \
        default: return VV_ERR_UNSUPPORTED;                     \
    }

vv_status_t vv_kv_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, const int* page_table, void* stream)
{
    if (!k_fp16 || !v_fp16 || !k_store || !v_store) return VV_ERR_NULL_PTR;
    if (n_positions <= 0) return VV_OK;
    cudaStream_t st = (cudaStream_t)stream;

    if (kv_format == VV_KV_FP16) {
        /* 16-byte chunks: head_dim/8 per vector, one thread each. */
        if (head_dim % 8) return VV_ERR_UNSUPPORTED;
        const int chunks = head_dim / 8;
        const long long total = (long long)n_positions * n_kv_heads * chunks;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        kv_store_raw_kernel<<<blocks, threads, 0, st>>>(
            (const uint4*)k_fp16, (const uint4*)v_fp16,
            (uint4*)k_store, (uint4*)v_store, n_kv_heads, chunks,
            pos, d_pos, n_positions, page_table);
        return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
    }
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const int total = n_positions * n_kv_heads;
    const int wpb = 4;                       /* warps per block */
    dim3 block(32, wpb);
    const int blocks = (total + wpb - 1) / wpb;
    const size_t sh = (size_t)wpb * 16 * sizeof(uint32_t);

    if (k_ref && build_ref) {
        kv_build_ref_kernel<<<n_kv_heads, head_dim, 0, st>>>(
            (const half*)k_fp16, (half*)k_ref, n_kv_heads, head_dim,
            n_positions);
    }

#define STORE_CALL(F)                                                            kv_store_kernel<F><<<blocks, block, sh, st>>>(                                   (const half*)k_fp16, (uint8_t*)k_store, (half*)k_meta,                       (const half*)k_ref, n_kv_heads, head_dim, pos, d_pos,                       n_positions, bpv, page_table);                                          kv_store_kernel<F><<<blocks, block, sh, st>>>(                                   (const half*)v_fp16, (uint8_t*)v_store, (half*)v_meta,                       NULL, n_kv_heads, head_dim, pos, d_pos, n_positions, bpv,                   page_table);

    DISPATCH_Q(kv_format, STORE_CALL)
#undef STORE_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_kv_quant_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, void* stream)
{
    if (kv_format == VV_KV_FP16) return VV_ERR_UNSUPPORTED;
    return vv_kv_store_dev(k_fp16, v_fp16, k_store, v_store, k_meta, v_meta,
                           k_ref, build_ref, n_kv_heads, head_dim, pos, d_pos,
                           n_positions, kv_format, NULL, stream);
}

vv_status_t vv_kv_page_map_dev(int* page_table, int first, int n,
                               const int* pages, void* stream)
{
    if (!page_table || (!pages && n > 0)) return VV_ERR_NULL_PTR;
    /* Page ids ride in the launch arguments, a batch at a time: nothing to
     * stage in pinned memory, and the writes are ordered on the stream. */
    for (int i = 0; i < n; i += VV_KV_MAP_BATCH) {
        kv_page_ids_t ids;
        const int m = (n - i < VV_KV_MAP_BATCH) ? n - i : VV_KV_MAP_BATCH;
        for (int j = 0; j < m; ++j) ids.id[j] = pages[i + j];
        kv_page_map_kernel<<<1, VV_KV_MAP_BATCH, 0, (cudaStream_t)stream>>>(
            page_table + first + i, ids, m);
    }
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_kv_dequant_dev(
    const void* store, const void* meta, const void* ref, void* out_fp16,
    int n_kv_heads, int head_dim, int n_pos, int kv_format, void* stream)
{
    if (!store || !out_fp16) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const int total = n_pos * n_kv_heads;
    dim3 block(32, 4);
    const int blocks = (total + 3) / 4;
    cudaStream_t s = (cudaStream_t)stream;

#define DEQ_CALL(F)                                                              kv_dequant_kernel<F><<<blocks, block, 0, s>>>(                                   (const uint8_t*)store, (const half*)meta, (const half*)ref,                  (half*)out_fp16, n_kv_heads, head_dim, n_pos, bpv);

    DISPATCH_Q(kv_format, DEQ_CALL)
#undef DEQ_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_decode_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    const int* d_cache_len, int kv_format, void* scratch, void* stream)
{
    if (!q || !k_store || !v_store || !output || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;

    const vv_decode_parts_t part =
        vv_decode_parts(scratch, n_q_heads, head_dim);
    const int n_parts = vv_decode_n_parts(cache_len);

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(n_q_heads, n_parts / QD_WARPS);
    dim3 block(32, QD_WARPS);
    cudaStream_t s = (cudaStream_t)stream;

#define DEC_CALL(F)                                                          \
    q_decode_split_kernel<F><<<grid, block, 0, s>>>(                         \
        (const half*)q, (const uint8_t*)k_store, (const uint8_t*)v_store,    \
        (const half*)k_meta, (const half*)v_meta,                            \
        part.o, part.m, part.l,                                              \
        n_q_heads, n_kv_heads, head_dim, cache_len, d_cache_len,             \
        n_parts, bpv, scale);

    DISPATCH_Q(kv_format, DEC_CALL)
#undef DEC_CALL

    const size_t shb = (size_t)n_parts * 2 * sizeof(float);
    const int cthreads = head_dim > n_parts ? head_dim : n_parts;
    q_decode_combine_kernel<<<n_q_heads, cthreads, shb, s>>>(
        part.o, part.m, part.l, (half*)output, n_parts, head_dim);

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_prefill_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal,
    int kv_format, void* stream)
{
    if (!q || !k_store || !v_store || !output) return VV_ERR_NULL_PTR;
    if (q_len == 0 || kv_len == 0) return VV_OK;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(n_q_heads, (q_len + QP_BR - 1) / QP_BR);
    dim3 block(32, QP_BR);
    const size_t shb = (size_t)(2 * QP_BC * (head_dim + QP_KV_PAD)
                                + QP_BR * head_dim) * sizeof(half);
    cudaStream_t s = (cudaStream_t)stream;

#define PRE_CALL(F)                                                          \
    q_prefill_kernel<F><<<grid, block, shb, s>>>(                            \
        (const half*)q, (const uint8_t*)k_store, (const uint8_t*)v_store,    \
        (const half*)k_meta, (const half*)v_meta, (half*)output,             \
        n_q_heads, n_kv_heads, head_dim, q_len, q_offset, kv_len,            \
        bpv, scale, causal);

    DISPATCH_Q(kv_format, PRE_CALL)
#undef PRE_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
