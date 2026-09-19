/**
 * @file attention_decode.cu
 * @brief Split-KV decode attention with GQA packing, in exactly the
 *        arithmetic of the per-head kernels it replaces.
 *
 * The per-head decode kernels (flash_decode_split_kernel for FP16,
 * q_decode_split_kernel for the quantized formats) give each of the 28 query
 * heads its own warps, so the 7 heads that share a KV head read — and for
 * TurboQuant, unpack — the same K and V seven times.
 *
 * Here the G heads of a group run together, in one of two layouts
 * (gqa_decode_kernel): spread — one warp per head, the group's warps in one
 * block, so a position comes from DRAM once and from L1 the other G - 1
 * times — or packed — one warp carrying all G heads' online-softmax state,
 * so a quantized vector is also decoded once. Everything a head computes is
 * the same sequence of FP32 operations the per-head kernel performed, on the
 * same slice boundaries (vv_decode_n_parts), merged by a kernel with the
 * same order of operations — so the result is bit-identical, which the
 * tests check, and the transcripts cannot move.
 *
 * Tensor cores would read K/V once too (the flashinfer backend does), but
 * they round P to FP16 and sum in a different order, and that is enough to
 * move a timestamp by 10 ms on the 32-minute file.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "attn_common.cuh"
#include "decode_split.h"

#define GD_WARPS VV_DECODE_WARPS

/* Packed layout from this many warps per SM (see gqa_decode_kernel). */
#ifndef GD_PACKED_WARPS_PER_SM
#define GD_PACKED_WARPS_PER_SM 4
#endif

/*
 * A position's K or V is read in two steps: `gd_fetch` issues the lane's
 * global loads, `gd_decode` turns them into the 4 floats kv_read4 (or the
 * FP16 kernel's __half2float) would have produced. The loop fetches position
 * t + 1 before it computes position t, so the load latency — which is what
 * bounds a warp walking its slice one position at a time — hides behind the
 * G heads' arithmetic. The arithmetic itself is untouched.
 */
template <int FMT> struct GdRaw { uint32_t w; half sigma; };
template <> struct GdRaw<VV_KV_FP16> { uint2 w; };

template <int FMT>
__device__ __forceinline__ void gd_fetch(
    GdRaw<FMT>& r, const uint8_t* __restrict__ store,
    const half* __restrict__ meta, size_t vi, int bpv, int lane)
{
    if (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2) {
        r.w = ((const uint32_t*)(store + vi * (size_t)bpv))[lane];
    } else {
        /* TurboQuant: warp_fetch_bits' one coalesced word per lane. */
        const int nwords = (KVBits<FMT>::N * 32 + 31) / 32;
        const uint32_t* w = (const uint32_t*)(store + vi * (size_t)bpv);
        r.w = (lane < nwords) ? w[lane] : 0u;
        r.sigma = meta[vi];
    }
}

template <>
__device__ __forceinline__ void gd_fetch<VV_KV_FP16>(
    GdRaw<VV_KV_FP16>& r, const uint8_t* __restrict__ store,
    const half* __restrict__ meta, size_t vi, int bpv, int lane)
{
    (void)meta; (void)bpv;
    r.w = *(const uint2*)((const half*)store + vi * ATT_D + lane * 4);
}

template <int FMT>
__device__ __forceinline__ void gd_decode(const GdRaw<FMT>& r, int lane,
                                          float out[4])
{
    if (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (r.w >> (8 * i)) & 0xFFu;
            out[i] = (FMT == VV_KV_FP8_E4M3) ? e4m32f(b) : e5m22f(b);
        }
    } else {
        /* warp_fetch_bits after its load, then unpack4. */
        const int NB = KVBits<FMT>::N;
        const int bit = NB * lane;
        const int wi  = bit >> 5;
        const int sh  = bit & 31;
        const uint32_t lo = __shfl_sync(0xFFFFFFFFu, r.w, wi);
        const uint32_t hi = __shfl_sync(0xFFFFFFFFu, r.w, (wi + 1) & 31);
        const uint32_t v  = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
        unpack4<FMT>(v & ((1u << NB) - 1u), __half2float(r.sigma), out);
    }
}

template <>
__device__ __forceinline__ void gd_decode<VV_KV_FP16>(
    const GdRaw<VV_KV_FP16>& r, int lane, float out[4])
{
    (void)lane;
    const __half2 a = *(const __half2*)&r.w.x;
    const __half2 b = *(const __half2*)&r.w.y;
    out[0] = __low2float(a);  out[1] = __high2float(a);
    out[2] = __low2float(b);  out[3] = __high2float(b);
}

/**
 * Two layouts, one arithmetic. Each warp walks one slice for H query heads.
 *
 *  H == G (packed): grid (n_kv_heads, n_parts / GD_WARPS), block
 *    (32, GD_WARPS); warp w of block y takes slice y * GD_WARPS + w for all
 *    G heads of the group. One load per position; n_kv_heads * n_parts warps.
 *  H == 1 (spread): grid (n_kv_heads, n_parts), block (32, G); warp w takes
 *    head w of the group on slice y. G warps of one block load the same
 *    addresses, so the other G - 1 hit L1; n_q_heads * n_parts warps.
 *
 * A short cache has few slices (8 up to 1K positions): packed would run 32
 * warps on an 82-SM card. Which one the host picks is in
 * vv_attn_gqa_decode_dev.
 */
template <int FMT, int H>
__global__ __launch_bounds__(32 * GD_WARPS)
void gqa_decode_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ k_store, const uint8_t* __restrict__ v_store,
    const half* __restrict__ k_meta, const half* __restrict__ v_meta,
    const int* __restrict__ page_table,
    float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l,
    int n_kv_heads, int G, int cache_len, const int* __restrict__ d_cache_len,
    int n_parts, int bpv, float scale)
{
    const int kv_head = blockIdx.x;
    const int lane    = threadIdx.x;
    const int part    = (H == 1) ? (int)blockIdx.y
                                 : (int)(blockIdx.y * GD_WARPS + threadIdx.y);
    const int head0   = kv_head * G + ((H == 1) ? (int)threadIdx.y : 0);
    if (part >= n_parts) return;

    /* A replayed graph keeps its arguments: the length comes from device
     * memory, and n_parts only sizes the grid within a shape bucket. */
    if (d_cache_len) cache_len = *d_cache_len;

    const int chunk = (cache_len + n_parts - 1) / n_parts;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > cache_len) end = cache_len;

    float qr[H][4], o[H][4], m[H], l[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) {
        const half* qh = q + (size_t)(head0 + h) * ATT_D + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qr[h][i] = __half2float(qh[i]);
            o[h][i] = 0.0f;
        }
        m[h] = -FLT_MAX;
        l[h] = 0.0f;
    }

    GdRaw<FMT> rk, rv;
    if (begin < end) {
        const size_t vi = (size_t)kv_phys_row(page_table, begin) * n_kv_heads
                        + kv_head;
        gd_fetch<FMT>(rk, k_store, k_meta, vi, bpv, lane);
        gd_fetch<FMT>(rv, v_store, v_meta, vi, bpv, lane);
    }
    for (int t = begin; t < end; ++t) {
        /* Next position's loads first; the last iteration re-reads its own
         * position rather than branch, which keeps the loop uniform. */
        GdRaw<FMT> nk, nv;
        {
            const int tn = (t + 1 < end) ? t + 1 : t;
            const size_t vi = (size_t)kv_phys_row(page_table, tn) * n_kv_heads
                            + kv_head;
            gd_fetch<FMT>(nk, k_store, k_meta, vi, bpv, lane);
            gd_fetch<FMT>(nv, v_store, v_meta, vi, bpv, lane);
        }

        float kv[4];
        gd_decode<FMT>(rk, lane, kv);

        float p[H], alpha[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qr[h][i], kv[i], dot);
            dot = warp_sum(dot) * scale;
            const float m_new = fmaxf(m[h], dot);
            alpha[h] = (m[h] > -FLT_MAX) ? __expf(m[h] - m_new) : 0.0f;
            p[h] = __expf(dot - m_new);
            m[h] = m_new;
        }

        gd_decode<FMT>(rv, lane, kv);
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                o[h][i] = o[h][i] * alpha[h] + p[h] * kv[i];
            l[h] = l[h] * alpha[h] + p[h];
        }
        rk = nk;
        rv = nv;
    }

    #pragma unroll
    for (int h = 0; h < H; ++h) {
        const size_t row = (size_t)(head0 + h) * n_parts + part;
        float* po = part_o + row * ATT_D + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) po[i] = o[h][i];
        if (lane == 0) {
            part_m[row] = (end > begin) ? m[h] : -FLT_MAX;
            part_l[row] = l[h];
        }
    }
}

extern "C" {

#include "vibevoice/types.h"
#include "vibevoice/device.h"
#include "attn_internal.h"

bool vv_attn_gqa_decode_ok(int n_q_heads, int n_kv_heads, int head_dim) {
    if (head_dim != ATT_D || n_kv_heads <= 0 || n_q_heads % n_kv_heads)
        return false;
    const int g = n_q_heads / n_kv_heads;
    return g == 1 || g == 2 || g == 4 || g == 6 || g == 7 || g == 8;
}

vv_status_t vv_attn_gqa_decode_dev(
    const void* q, const vv_kv_view_t* kv, void* out, int n_q_heads,
    int cache_len, const int* d_cache_len, void* scratch, void* stream)
{
    if (!q || !kv || !kv->k || !kv->v || !out || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len <= 0) return VV_OK;
    if (!vv_attn_gqa_decode_ok(n_q_heads, kv->n_kv_heads, kv->head_dim))
        return VV_ERR_UNSUPPORTED;

    const int n_kv = kv->n_kv_heads;
    const int G = n_q_heads / n_kv;
    const int n_parts = vv_decode_n_parts(cache_len);
    const vv_decode_parts_t pt = vv_decode_parts(scratch, n_q_heads, ATT_D);
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D);
    const float scale = 1.0f / sqrtf((float)ATT_D);
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t* K = (const uint8_t*)kv->k;
    const uint8_t* V = (const uint8_t*)kv->v;
    const half* Km = (const half*)kv->k_meta;
    const half* Vm = (const half*)kv->v_meta;

    /*
     * Spread for FP16: its loads are plain and L1 serves the group's other
     * warps, so more warps in flight win at every length (3090: 27 us at 1K,
     * 93 at 16K, 157 at 32K, against 62 / 202 / 324 for fa1; packed takes
     * 135 at 16K). A quantized vector costs real work to decode, which
     * packing does once instead of G times — so once there are enough slices
     * to keep every SM busy it packs (fp8 at 16K: 148 us packed, 190 spread).
     * Both choices depend only on n_parts, so the graph shape is unchanged.
     */
    const int sm_count = vv_attn_device_sm_count();
    const bool packed = G == 1 ||
        (kv->format != VV_KV_FP16 &&
         n_kv * n_parts >= GD_PACKED_WARPS_PER_SM * sm_count);
    const dim3 grid = packed ? dim3(n_kv, n_parts / GD_WARPS)
                             : dim3(n_kv, n_parts);
    const dim3 block = packed ? dim3(32, GD_WARPS) : dim3(32, G);
#define GD_LAUNCH(F, HH)                                                       \
    gqa_decode_kernel<F, HH><<<grid, block, 0, st>>>(                          \
        (const half*)q, K, V, Km, Vm, kv->page_table, pt.o, pt.m, pt.l,        \
        n_kv, G, cache_len, d_cache_len, n_parts, bpv, scale)
#define GD_FMT(F)                                                              \
    if (!packed) { GD_LAUNCH(F, 1); } else switch (G) {                        \
        case 1: GD_LAUNCH(F, 1); break;                                        \
        case 2: GD_LAUNCH(F, 2); break;                                        \
        case 4: GD_LAUNCH(F, 4); break;                                        \
        case 6: GD_LAUNCH(F, 6); break;                                        \
        case 7: GD_LAUNCH(F, 7); break;                                        \
        default: GD_LAUNCH(F, 8); break;                                       \
    }
    switch (kv->format) {
        case VV_KV_FP16:     GD_FMT(VV_KV_FP16);     break;
        case VV_KV_FP8_E4M3: GD_FMT(VV_KV_FP8_E4M3); break;
        case VV_KV_FP8_E5M2: GD_FMT(VV_KV_FP8_E5M2); break;
        case VV_KV_TQ4:      GD_FMT(VV_KV_TQ4);      break;
        case VV_KV_TQ3:      GD_FMT(VV_KV_TQ3);      break;
        case VV_KV_TQ2:      GD_FMT(VV_KV_TQ2);      break;
        case VV_KV_TQ1_5:    GD_FMT(VV_KV_TQ1_5);    break;
        default: return VV_ERR_UNSUPPORTED;
    }
#undef GD_FMT
#undef GD_LAUNCH

    att_combine_kernel<<<n_q_heads, ATT_D, 0, st>>>(pt.o, pt.m, pt.l,
                                                   (half*)out, n_parts);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
