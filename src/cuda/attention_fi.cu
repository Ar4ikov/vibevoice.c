/**
 * @file attention_fi.cu
 * @brief FlashAttention-2 and FlashInfer-style attention on tensor cores,
 *        and the one dispatch point every backend goes through.
 *
 * ## GQA packing
 *
 * Qwen2-7B has 7 query heads per KV head (BitNet 6). Kernels that give each
 * query head its own block read every K/V tile once per head. Here a block
 * belongs to a KV head and its rows are (position, head-in-group) pairs:
 * row r is position r / G, head kv_head * G + r % G. Every tile a block loads
 * serves every head that reads it, and because a 64-row tile spans only
 * ~9 positions, the causal bound of a block stays tight.
 *
 * ## Two kernels
 *
 * `att_rows_kernel` — prefill. Four warps own 16 rows each; K and V stream
 * through shared memory tile by tile. The two tiles are loaded on separate
 * cp.async groups so V(t) arrives while S = Q·K(t)ᵀ is computed and K(t+1)
 * arrives during P·V(t): the overlap of a double buffer in the footprint of
 * a single one (34 KB, so two blocks fit an SM beside their registers).
 * Tiles the causal mask cannot touch skip the mask arithmetic entirely.
 *
 * `att_split_kernel` — decode and small-q prefill. One 16-row fragment per
 * block (for decode: the G heads of one KV head), the four warps each taking
 * 16 positions of every tile, and the cache split into `n_parts` ranges over
 * a third grid dimension. Partial (o, m, l) states go to the caller's scratch
 * and `att_combine_kernel` merges them. K/V is read once per KV head, not
 * once per query head, and on tensor cores rather than one FMA per lane.
 *
 * Both read any cache format (att_load_tile) and a paged cache.
 *
 * ## Uniform loop bounds
 *
 * Every warp of a block reaches the same __syncthreads() count: the KV loop
 * bound is computed per block, never per row, and per-row causality is a
 * score mask (CLAUDE.md, bug #4).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "attn_common.cuh"
#include "decode_split.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Prefill: 64 packed rows per block
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Warps (16 rows each) per prefill block. Eight share each K/V tile between
 * twice the rows but, at 232 registers a thread, fit one block per SM
 * instead of two: 37/53/57 TFLOP/s at 1K/4K/8K against 45/61/64 for four. */
#ifndef ATT_ROWS_WARPS
#define ATT_ROWS_WARPS 4
#endif
#define ATT_ROWS_BR (ATT_ROWS_WARPS * 16)

template <int FMT>
__global__ __launch_bounds__(ATT_ROWS_WARPS * 32)
void att_rows_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const half* __restrict__ K_meta, const half* __restrict__ V_meta,
    const int* __restrict__ page_table,
    half* __restrict__ O,
    int n_q_heads, int n_kv_heads, int q_len, int q_offset, int kv_len,
    int bpv, float scale, bool causal)
{
#if ATT_HAS_MMA
    const int kv_head = blockIdx.x;
    const int G       = n_q_heads / n_kv_heads;
    const int n_rows  = q_len * G;
    const int tid     = threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = lane >> 2;
    const int quad    = lane & 3;

    const int row0 = blockIdx.y * ATT_ROWS_BR;
    const int r_lo = row0 + warp * 16 + grp;
    const int r_hi = r_lo + 8;
    const bool ok_lo = r_lo < n_rows, ok_hi = r_hi < n_rows;
    const int p_lo = r_lo / G, p_hi = r_hi / G;
    const int h_lo = kv_head * G + (r_lo - p_lo * G);
    const int h_hi = kv_head * G + (r_hi - p_hi * G);
    const size_t q_stride = (size_t)n_q_heads * ATT_D;

    extern __shared__ __align__(16) char att_smem[];
    half* Ks = (half*)att_smem;
    half* Vs = Ks + ATT_TILE;

    uint32_t a_q[ATT_D / 16][4];
    att_load_q(a_q, Q + p_lo * q_stride + (size_t)h_lo * ATT_D, ok_lo,
               Q + p_hi * q_stride + (size_t)h_hi * ATT_D, ok_hi, quad);

    float o[ATT_D / 8][4];
    #pragma unroll
    for (int n = 0; n < ATT_D / 8; ++n)
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_lo = -FLT_MAX, m_hi = -FLT_MAX, l_lo = 0.0f, l_hi = 0.0f;

    /* Block-uniform bounds: the loop runs to the last row's causal limit,
     * and tiles wholly below the first row's limit need no mask. */
    const int last_row  = min(row0 + ATT_ROWS_BR, n_rows) - 1;
    const int first_pos = row0 / G, last_pos = last_row / G;
    int kv_bound = causal ? q_offset + last_pos + 1 : kv_len;
    if (kv_bound > kv_len) kv_bound = kv_len;
    int full_to = causal ? q_offset + first_pos + 1 : kv_len;
    if (full_to > kv_len) full_to = kv_len;
    const int n_tiles = (kv_bound + ATT_BC - 1) / ATT_BC;
    const int qa_lo = q_offset + p_lo, qa_hi = q_offset + p_hi;
    __shared__ float lut[16];
    att_q_lut_init<FMT>(lut, tid);
    if (AttQ<FMT>::LUT) __syncthreads();

    float s[ATT_BC / 8][4];
    auto qk = [&](int kv0) {
        /* S = Q·Kᵀ: 8 n-tiles of 8 positions, 8 k-steps. One ldmatrix.x4
         * brings the B operand of two k-steps. */
        #pragma unroll
        for (int n = 0; n < ATT_BC / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            const half* kr = Ks + (n * 8 + (lane & 7)) * ATT_LD + (lane >> 3) * 8;
            #pragma unroll
            for (int kk = 0; kk < ATT_D / 32; ++kk) {
                uint32_t b[4];
                att_ldsm_x4(b, kr + kk * 32);
                att_mma(s[n], a_q[2 * kk],     b[0], b[1]);
                att_mma(s[n], a_q[2 * kk + 1], b[2], b[3]);
            }
        }
        if (kv0 + ATT_BC <= full_to)
            att_softmax<ATT_BC / 8, false>(s, o, m_lo, m_hi, l_lo, l_hi,
                                           scale, kv0, quad, kv_len,
                                           causal, qa_lo, qa_hi);
        else
            att_softmax<ATT_BC / 8, true>(s, o, m_lo, m_hi, l_lo, l_hi,
                                          scale, kv0, quad, kv_len,
                                          causal, qa_lo, qa_hi);
    };
    auto pv = [&]() {
        /* O += P·V. Two n-tiles of S make one A fragment; ldmatrix.trans
         * hands out V's B operand for two output n-tiles at once.
         *
         * A quantized cache used to be read by FP32 scalar kernels only, so
         * there P also goes in as FP16 hi + lo (att_split_pair) to stay
         * that close to them; on FP16 it stays one FP16, as the tensor-core
         * prefill always had it. */
        constexpr bool SPLIT_P = FMT != VV_KV_FP16;
        #pragma unroll
        for (int ks = 0; ks < ATT_BC / 16; ++ks) {
            uint32_t a_p[4], a_l[4];
            att_split_pair(s[2 * ks][0],     s[2 * ks][1],     a_p[0], a_l[0]);
            att_split_pair(s[2 * ks][2],     s[2 * ks][3],     a_p[1], a_l[1]);
            att_split_pair(s[2 * ks + 1][0], s[2 * ks + 1][1], a_p[2], a_l[2]);
            att_split_pair(s[2 * ks + 1][2], s[2 * ks + 1][3], a_p[3], a_l[3]);
            const half* vr = Vs + (ks * 16 + (lane & 15)) * ATT_LD + (lane >> 4) * 8;
            #pragma unroll
            for (int n2 = 0; n2 < ATT_D / 16; ++n2) {
                uint32_t b[4];
                att_ldsm_x4_t(b, vr + n2 * 16);
                att_mma(o[2 * n2],     a_p, b[0], b[1]);
                att_mma(o[2 * n2 + 1], a_p, b[2], b[3]);
                if (SPLIT_P) {
                    att_mma(o[2 * n2],     a_l, b[0], b[1]);
                    att_mma(o[2 * n2 + 1], a_l, b[2], b[3]);
                }
            }
        }
    };
    att_run_tiles<FMT>(Ks, Vs, K, V, K_meta, V_meta, page_table, lut, 0,
                       n_tiles, kv_len, n_kv_heads, kv_head, bpv, tid, qk, pv);

    const float inv_lo = (l_lo > 1e-20f) ? 1.0f / l_lo : 0.0f;
    const float inv_hi = (l_hi > 1e-20f) ? 1.0f / l_hi : 0.0f;
    half* o_lo = O + p_lo * q_stride + (size_t)h_lo * ATT_D;
    half* o_hi = O + p_hi * q_stride + (size_t)h_hi * ATT_D;
    #pragma unroll
    for (int n = 0; n < ATT_D / 8; ++n) {
        const int c = n * 8 + quad * 2;
        if (ok_lo) *(uint32_t*)(o_lo + c) = att_pack(o[n][0] * inv_lo, o[n][1] * inv_lo);
        if (ok_hi) *(uint32_t*)(o_hi + c) = att_pack(o[n][2] * inv_hi, o[n][3] * inv_hi);
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Decode and small-q prefill: 16 packed rows per block, split KV
 * ═══════════════════════════════════════════════════════════════════════════ */

template <int FMT>
__global__ __launch_bounds__(ATT_THREADS)
void att_split_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const half* __restrict__ K_meta, const half* __restrict__ V_meta,
    const int* __restrict__ page_table,
    float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l,
    int n_q_heads, int n_kv_heads, int q_len, int q_offset,
    int kv_len, const int* __restrict__ d_kv_len, int n_parts,
    int bpv, float scale, bool causal, int verify_rows)
{
#if ATT_HAS_MMA
    /* A replayed graph keeps its arguments, so the length comes from device
     * memory; n_parts only sizes the grid and is fixed within a bucket. */
    if (d_kv_len) kv_len = *d_kv_len;

    /*
     * The rows of a verified block (verify_rows > 1, one query row each):
     * blockIdx.y is the row r, a decode at kv_len + r with the split count
     * and partials that decode would have had.
     */
    int vr = 0;
    if (verify_rows > 1) {
        vr = (int)blockIdx.y;
        kv_len += vr;
        n_parts = vv_fi_decode_parts(n_kv_heads, kv_len);
        Q += (size_t)vr * n_q_heads * ATT_D;
        const size_t per = (size_t)n_q_heads * VV_DECODE_MAX_PARTS;
        part_o += (size_t)vr * per * ATT_D;
        part_m += (size_t)vr * per;
        part_l += (size_t)vr * per;
        if ((int)blockIdx.z >= n_parts) return;
    }

    const int kv_head = blockIdx.x;
    const int part    = blockIdx.z;
    const int G       = n_q_heads / n_kv_heads;
    const int n_rows  = q_len * G;
    const int tid     = threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = lane >> 2;
    const int quad    = lane & 3;

    const int row0 = verify_rows > 1 ? 0 : blockIdx.y * 16;
    const int r_lo = row0 + grp, r_hi = r_lo + 8;
    const bool ok_lo = r_lo < n_rows, ok_hi = r_hi < n_rows;
    const int p_lo = r_lo / G, p_hi = r_hi / G;
    const int h_lo = kv_head * G + (r_lo - p_lo * G);
    const int h_hi = kv_head * G + (r_hi - p_hi * G);
    const size_t q_stride = (size_t)n_q_heads * ATT_D;

    extern __shared__ __align__(16) char att_smem[];
    half* Ks = (half*)att_smem;
    half* Vs = Ks + ATT_TILE;

    uint32_t a_q[ATT_D / 16][4];
    att_load_q(a_q, Q + p_lo * q_stride + (size_t)h_lo * ATT_D, ok_lo,
               Q + p_hi * q_stride + (size_t)h_hi * ATT_D, ok_hi, quad);

    float o[ATT_D / 8][4];
    #pragma unroll
    for (int n = 0; n < ATT_D / 8; ++n)
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_lo = -FLT_MAX, m_hi = -FLT_MAX, l_lo = 0.0f, l_hi = 0.0f;

    /* This part's slice: the same partition for every row tile, aligned to
     * whole tiles, then clipped to what these rows may see. */
    const int last_row = min(row0 + 16, n_rows) - 1;
    const int first_pos = row0 / G, last_pos = last_row / G;
    int bound = causal ? q_offset + last_pos + 1 : kv_len;
    if (bound > kv_len) bound = kv_len;
    int full_to = causal ? q_offset + first_pos + 1 : kv_len;
    if (full_to > kv_len) full_to = kv_len;

    int chunk = (kv_len + n_parts - 1) / n_parts;
    chunk = (chunk + ATT_BC - 1) / ATT_BC * ATT_BC;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > bound) end = bound;
    const int n_tiles = end > begin ? (end - begin + ATT_BC - 1) / ATT_BC : 0;
    const int qa_lo = q_offset + p_lo, qa_hi = q_offset + p_hi;
    __shared__ float lut[16];
    att_q_lut_init<FMT>(lut, tid);
    if (AttQ<FMT>::LUT) __syncthreads();

    float s[2][4];
    auto qk = [&](int kv0) {
        /* This warp's 16 positions of the tile: two n-tiles. */
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            const half* kr = Ks + (warp * 16 + n * 8 + (lane & 7)) * ATT_LD
                           + (lane >> 3) * 8;
            #pragma unroll
            for (int kk = 0; kk < ATT_D / 32; ++kk) {
                uint32_t b[4];
                att_ldsm_x4(b, kr + kk * 32);
                att_mma(s[n], a_q[2 * kk],     b[0], b[1]);
                att_mma(s[n], a_q[2 * kk + 1], b[2], b[3]);
            }
        }
        const int pos0 = kv0 + warp * 16;
        if (kv0 + ATT_BC <= full_to)
            att_softmax<2, false>(s, o, m_lo, m_hi, l_lo, l_hi, scale,
                                  pos0, quad, kv_len, causal, qa_lo, qa_hi);
        else
            att_softmax<2, true>(s, o, m_lo, m_hi, l_lo, l_hi, scale,
                                 pos0, quad, kv_len, causal, qa_lo, qa_hi);
    };
    auto pv = [&]() {
        /*
         * P goes in as two FP16 halves, P = hi + lo, so it keeps ~22 bits
         * instead of 11: the scalar decode this replaces kept P in FP32, and
         * a single rounding of it was enough to move a timestamp by 10 ms
         * on the 32-minute file. The second MMA is free here — decode reads
         * a whole K/V tile per 16 rows and is bound by memory, not math.
         */
        uint32_t a_p[4], a_l[4];
        att_split_pair(s[0][0], s[0][1], a_p[0], a_l[0]);
        att_split_pair(s[0][2], s[0][3], a_p[1], a_l[1]);
        att_split_pair(s[1][0], s[1][1], a_p[2], a_l[2]);
        att_split_pair(s[1][2], s[1][3], a_p[3], a_l[3]);
        const half* vr = Vs + (warp * 16 + (lane & 15)) * ATT_LD + (lane >> 4) * 8;
        #pragma unroll
        for (int n2 = 0; n2 < ATT_D / 16; ++n2) {
            uint32_t b[4];
            att_ldsm_x4_t(b, vr + n2 * 16);
            att_mma(o[2 * n2],     a_p, b[0], b[1]);
            att_mma(o[2 * n2 + 1], a_p, b[2], b[3]);
            att_mma(o[2 * n2],     a_l, b[0], b[1]);
            att_mma(o[2 * n2 + 1], a_l, b[2], b[3]);
        }
    };
    att_run_tiles<FMT>(Ks, Vs, K, V, K_meta, V_meta, page_table, lut, begin,
                       n_tiles, kv_len, n_kv_heads, kv_head, bpv, tid, qk, pv);

    /* Merge the four warps' states through the (now idle) tile memory. */
    float* sh_o = (float*)att_smem;                       /* [4][16][128] */
    float* sh_m = sh_o + ATT_WARPS * 16 * ATT_D;          /* [4][16]      */
    float* sh_l = sh_m + ATT_WARPS * 16;
    #pragma unroll
    for (int n = 0; n < ATT_D / 8; ++n) {
        const int c = n * 8 + quad * 2;
        float* lo = sh_o + ((size_t)warp * 16 + grp) * ATT_D + c;
        float* hi = lo + 8 * ATT_D;
        lo[0] = o[n][0]; lo[1] = o[n][1];
        hi[0] = o[n][2]; hi[1] = o[n][3];
    }
    if (quad == 0) {
        sh_m[warp * 16 + grp]     = m_lo;
        sh_m[warp * 16 + grp + 8] = m_hi;
        sh_l[warp * 16 + grp]     = l_lo;
        sh_l[warp * 16 + grp + 8] = l_hi;
    }
    __syncthreads();

    const int d = tid;                                    /* 128 threads */
    for (int r = 0; r < 16; ++r) {
        const int row = row0 + r;
        if (row >= n_rows) break;
        float mx = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < ATT_WARPS; ++w) mx = fmaxf(mx, sh_m[w * 16 + r]);
        float acc = 0.0f, l = 0.0f;
        if (mx > -FLT_MAX) {
            #pragma unroll
            for (int w = 0; w < ATT_WARPS; ++w) {
                const float mw = sh_m[w * 16 + r];
                if (mw <= -FLT_MAX) continue;
                const float f = __expf(mw - mx);
                acc += f * sh_o[((size_t)w * 16 + r) * ATT_D + d];
                l   += f * sh_l[w * 16 + r];
            }
        }
        const int pos = row / G;
        const int head = kv_head * G + (row - pos * G);
        const size_t out_row = (size_t)pos * n_q_heads + head;
        const size_t pi = out_row * n_parts + part;
        part_o[pi * ATT_D + d] = acc;
        if (d == 0) { part_m[pi] = mx; part_l[pi] = l; }
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host side
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

#include "vibevoice/types.h"
#include "vibevoice/device.h"
#include "attn_internal.h"

#define ATT_DISPATCH(fmt, CALL)                                   \
    switch (fmt) {                                                \
        case VV_KV_FP16:     CALL(VV_KV_FP16);     break;         \
        case VV_KV_FP8_E4M3: CALL(VV_KV_FP8_E4M3); break;         \
        case VV_KV_FP8_E5M2: CALL(VV_KV_FP8_E5M2); break;         \
        case VV_KV_TQ4:      CALL(VV_KV_TQ4);      break;         \
        case VV_KV_TQ3:      CALL(VV_KV_TQ3);      break;         \
        case VV_KV_TQ2:      CALL(VV_KV_TQ2);      break;         \
        case VV_KV_TQ1_5:    CALL(VV_KV_TQ1_5);    break;         \
        default: return VV_ERR_UNSUPPORTED;                       \
    }

/** @brief Compute capability and SM count of the current device, cached per
 *         thread and device (a process may drive two different cards). */
static void att_device(int* sm, int* n_sm) {
    static thread_local int cached = -1, c_sm = 0, c_n = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { *sm = 0; *n_sm = 0; return; }
    if (dev != cached) {
        int ma = 0, mi = 0, n = 0;
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        c_sm = ma * 10 + mi; c_n = n; cached = dev;
    }
    *sm = c_sm; *n_sm = c_n;
}

int vv_attn_device_sm(void) {
    int sm, n;
    att_device(&sm, &n);
    return sm;
}

int vv_attn_device_sm_count(void) {
    int sm, n;
    att_device(&sm, &n);
    return n;
}

/** @brief Split count for flashinfer decode; steps every 1024 positions. */
static int att_fi_decode_parts(int n_kv_heads, int cache_len) {
    return vv_fi_decode_parts(n_kv_heads, cache_len);
}

vv_status_t vv_attn_fi_prefill_dev(
    bool packed_only, const void* q, const vv_kv_view_t* kv, void* out,
    int n_q_heads, int q_len, int q_offset, int kv_len, bool causal,
    void* scratch, void* stream)
{
    if (!q || !kv || !kv->k || !kv->v || !out) return VV_ERR_NULL_PTR;
    if (q_len <= 0 || kv_len <= 0) return VV_OK;
    if (kv->head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    const int n_kv = kv->n_kv_heads;
    if (n_kv <= 0 || n_q_heads % n_kv) return VV_ERR_UNSUPPORTED;
    const int G = n_q_heads / n_kv;
    if (G > 16) return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D);
    const float scale = 1.0f / sqrtf((float)ATT_D);
    cudaStream_t st = (cudaStream_t)stream;
    const int fmt = kv->format;
    const uint8_t* K = (const uint8_t*)kv->k;
    const uint8_t* V = (const uint8_t*)kv->v;
    const half* Km = (const half*)kv->k_meta;
    const half* Vm = (const half*)kv->v_meta;

    /*
     * Few rows against a long cache — a chat turn, a streaming chunk — leave
     * most SMs idle in the row kernel. Split the cache instead, as long as
     * the partials fit the scratch every context already owns.
     *
     * Only when the row kernel would leave more than half the SMs empty:
     * above that the split gains little, and it sums in a different order,
     * so keeping the row kernel there keeps prefill bit-identical to the
     * FA2 kernel (the tail chunk of a long file, 113 tokens on the 32-minute
     * one, used to move a timestamp by 10 ms).
     */
    const int n_rows = q_len * G;
    int sm = 0, n_sm = 0;
    att_device(&sm, &n_sm);
    const int row_blocks = n_kv * ((n_rows + ATT_ROWS_BR - 1) / ATT_ROWS_BR);
    if (!packed_only && scratch && 2 * row_blocks <= n_sm &&
        q_len <= VV_DECODE_MAX_PARTS && kv_len >= 2 * ATT_BC) {
        const int tiles16 = (n_rows + 15) / 16;
        const int base = n_kv * tiles16;
        int parts = (2 * n_sm + base - 1) / base;
        int cap = VV_DECODE_MAX_PARTS / q_len;
        const int by_len = (kv_len + 4 * ATT_BC - 1) / (4 * ATT_BC);
        if (cap > by_len) cap = by_len;
        if (parts > cap) parts = cap;
        if (parts >= 2) {
            const vv_decode_parts_t pt = vv_decode_parts(scratch, n_q_heads, ATT_D);
            dim3 grid(n_kv, tiles16, parts);
#define SPLIT_CALL(F)                                                          \
            att_split_kernel<F><<<grid, ATT_THREADS, ATT_SMEM_BYTES, st>>>(    \
                (const half*)q, K, V, Km, Vm, kv->page_table,                  \
                pt.o, pt.m, pt.l, n_q_heads, n_kv, q_len, q_offset, kv_len,    \
                NULL, parts, bpv, scale, causal, 0);
            ATT_DISPATCH(fmt, SPLIT_CALL)
#undef SPLIT_CALL
            att_combine_kernel<<<q_len * n_q_heads, ATT_D, 0, st>>>(
                pt.o, pt.m, pt.l, (half*)out, parts);
            return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
        }
    }

    dim3 grid(n_kv, (n_rows + ATT_ROWS_BR - 1) / ATT_ROWS_BR);
#define ROWS_CALL(F)                                                           \
    att_rows_kernel<F><<<grid, ATT_ROWS_WARPS * 32, ATT_SMEM_BYTES, st>>>(             \
        (const half*)q, K, V, Km, Vm, kv->page_table, (half*)out,              \
        n_q_heads, n_kv, q_len, q_offset, kv_len, bpv, scale, causal);
    ATT_DISPATCH(fmt, ROWS_CALL)
#undef ROWS_CALL
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_attn_fi_decode_dev(
    const void* q, const vv_kv_view_t* kv, void* out, int n_q_heads,
    int cache_len, const int* d_cache_len, void* scratch, void* stream)
{
    if (!q || !kv || !kv->k || !kv->v || !out || !scratch)
        return VV_ERR_NULL_PTR;
    if (cache_len <= 0) return VV_OK;
    if (kv->head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    const int n_kv = kv->n_kv_heads;
    if (n_kv <= 0 || n_q_heads % n_kv || n_q_heads / n_kv > 16)
        return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D);
    const float scale = 1.0f / sqrtf((float)ATT_D);
    cudaStream_t st = (cudaStream_t)stream;
    const int parts = att_fi_decode_parts(n_kv, cache_len);
    const vv_decode_parts_t pt = vv_decode_parts(scratch, n_q_heads, ATT_D);
    const uint8_t* K = (const uint8_t*)kv->k;
    const uint8_t* V = (const uint8_t*)kv->v;
    const half* Km = (const half*)kv->k_meta;
    const half* Vm = (const half*)kv->v_meta;

    dim3 grid(n_kv, 1, parts);
#define DEC_CALL(F)                                                            \
    att_split_kernel<F><<<grid, ATT_THREADS, ATT_SMEM_BYTES, st>>>(            \
        (const half*)q, K, V, Km, Vm, kv->page_table, pt.o, pt.m, pt.l,        \
        n_q_heads, n_kv, 1, 0, cache_len, d_cache_len, parts, bpv,             \
        scale, false, 0);
    ATT_DISPATCH(kv->format, DEC_CALL)
#undef DEC_CALL
    att_combine_kernel<<<n_q_heads, ATT_D, 0, st>>>(pt.o, pt.m, pt.l,
                                                   (half*)out, parts);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* `rows` consecutive flashinfer decodes in one launch (a verified block). */
static vv_status_t vv_attn_fi_decode_rows_dev(
    const void* q, const vv_kv_view_t* kv, void* out, int n_q_heads,
    int rows, int cache_len, const int* d_cache_len, void* scratch,
    void* stream)
{
    if (!q || !kv || !kv->k || !kv->v || !out || !scratch)
        return VV_ERR_NULL_PTR;
    if (rows == 1)
        return vv_attn_fi_decode_dev(q, kv, out, n_q_heads, cache_len,
                                     d_cache_len, scratch, stream);
    if (kv->head_dim != ATT_D) return VV_ERR_UNSUPPORTED;
    const int n_kv = kv->n_kv_heads;
    if (n_kv <= 0 || n_q_heads % n_kv || n_q_heads / n_kv > 16)
        return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv->format, ATT_D);
    const float scale = 1.0f / sqrtf((float)ATT_D);
    cudaStream_t st = (cudaStream_t)stream;
    const int parts = att_fi_decode_parts(n_kv, cache_len + rows - 1);
    const vv_decode_parts_t pt = vv_decode_parts_rows(scratch, n_q_heads,
                                                      ATT_D, rows);
    const uint8_t* K = (const uint8_t*)kv->k;
    const uint8_t* V = (const uint8_t*)kv->v;
    const half* Km = (const half*)kv->k_meta;
    const half* Vm = (const half*)kv->v_meta;

    dim3 grid(n_kv, rows, parts);
#define DEC_CALL(F)                                                            \
    att_split_kernel<F><<<grid, ATT_THREADS, ATT_SMEM_BYTES, st>>>(            \
        (const half*)q, K, V, Km, Vm, kv->page_table, pt.o, pt.m, pt.l,        \
        n_q_heads, n_kv, 1, 0, cache_len, d_cache_len, parts, bpv,             \
        scale, false, rows);
    ATT_DISPATCH(kv->format, DEC_CALL)
#undef DEC_CALL
    att_combine_rows_kernel<<<dim3(n_q_heads, rows), ATT_D, 0, st>>>(
        pt.o, pt.m, pt.l, (half*)out, n_q_heads, cache_len, d_cache_len, n_kv);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* ─── The dispatch point ─────────────────────────────────────────────────── */

/*
 * What each backend runs:
 *
 *             prefill, FP16 KV       prefill, quantized KV   decode
 *   fa1       scalar, per head       scalar, per head        scalar, per head
 *   fa2       tensor cores, packed   scalar, per head        GQA-packed scalar
 *   flashinfer tensor cores, packed, split KV for few rows;  tensor cores,
 *              quantized tiles decoded into shared memory     packed, split
 *
 * fa2 computes exactly what the kernels before the backends did: its FP16
 * prefill does the per-row arithmetic of the tensor-core kernel it replaced
 * (packing only changes which rows share a block), and its decode that of
 * the per-head decode, which fa1 still is (the tests compare them bit for
 * bit). So `auto` takes fa2 without moving a single timestamp. fa1's FP16
 * prefill is the scalar kernel that VV_ATTN_MMA=0 used to force. flashinfer
 * rounds P and sums in its own order: within FP16 rounding of the others,
 * fastest by far, and opt-in.
 *
 * Paging: fa2 reads a page table on FP16 (its prefill and decode both take
 * one); flashinfer on every format; fa1 never.
 */
int vv_attn_resolve(int requested, int kv_format, bool paged,
                    int n_q_heads, int n_kv_heads, int head_dim)
{
    int sm = 0, n_sm = 0;
    att_device(&sm, &n_sm);
    const bool tc = sm >= 75;
    const bool shape_ok = head_dim == ATT_D && n_kv_heads > 0 &&
                          n_q_heads % n_kv_heads == 0 &&
                          n_q_heads / n_kv_heads <= 16;
    const bool fi_ok = tc && shape_ok;
    const bool fa2_ok = tc && shape_ok &&
                        vv_attn_gqa_decode_ok(n_q_heads, n_kv_heads, head_dim);
    const bool fa2_pages = fa2_ok && kv_format == VV_KV_FP16;

    if (paged) {
        if (requested == VV_ATTN_FLASHINFER && fi_ok) return VV_ATTN_FLASHINFER;
        if (fa2_pages) return VV_ATTN_FA2;
        return fi_ok ? VV_ATTN_FLASHINFER : VV_ATTN_FA1;
    }

    switch (requested) {
        case VV_ATTN_FLASHINFER:
            if (fi_ok) return VV_ATTN_FLASHINFER;
            return fa2_ok ? VV_ATTN_FA2 : VV_ATTN_FA1;
        case VV_ATTN_FA1:
            return VV_ATTN_FA1;
        default:  /* fa2, auto */
            return fa2_ok ? VV_ATTN_FA2 : VV_ATTN_FA1;
    }
}

size_t vv_attn_scratch_bytes(int n_q_heads, int n_kv_heads, int head_dim) {
    (void)n_kv_heads;
    /* Every backend's partials use the split-decode layout at its cap. */
    return vv_decode_parts_bytes(n_q_heads, head_dim);
}

int vv_attn_decode_shape(int backend, int n_q_heads, int n_kv_heads,
                         int cache_len)
{
    (void)n_q_heads;
    if (backend == VV_ATTN_FLASHINFER)
        return 1000 + att_fi_decode_parts(n_kv_heads, cache_len);
    return vv_decode_n_parts(cache_len);
}

vv_status_t vv_attn_prefill(int backend, const void* q, const vv_kv_view_t* kv,
                            void* out, int n_q_heads, int q_len, int q_offset,
                            int kv_len, bool causal, void* scratch,
                            void* stream)
{
    if (!q || !kv || !out) return VV_ERR_NULL_PTR;
    if (q_len <= 0 || kv_len <= 0) return VV_OK;
    const bool raw = kv->format == VV_KV_FP16;

    switch (backend) {
        case VV_ATTN_FLASHINFER:
            return vv_attn_fi_prefill_dev(false, q, kv, out, n_q_heads, q_len,
                                          q_offset, kv_len, causal, scratch,
                                          stream);
        case VV_ATTN_FA2:
            if (raw)
                return vv_attn_fi_prefill_dev(true, q, kv, out, n_q_heads,
                                              q_len, q_offset, kv_len, causal,
                                              NULL, stream);
            break;
        default:
            break;
    }
    if (kv->page_table) return VV_ERR_UNSUPPORTED;
    if (raw)
        return vv_attn_fa1_prefill_dev(q, kv->k, kv->v, out, n_q_heads,
                                       kv->n_kv_heads, kv->head_dim, q_len,
                                       q_offset, kv_len, causal, stream);
    return vv_gqa_attention_prefill_q_dev(
        q, kv->k, kv->v, kv->k_meta, kv->v_meta, out, n_q_heads,
        kv->n_kv_heads, kv->head_dim, q_len, q_offset, kv_len, causal,
        kv->format, stream);
}

vv_status_t vv_attn_decode(int backend, const void* q, const vv_kv_view_t* kv,
                           void* out, int n_q_heads, int cache_len,
                           const int* d_cache_len, void* scratch,
                           void* stream)
{
    if (!q || !kv || !out) return VV_ERR_NULL_PTR;
    if (backend == VV_ATTN_FLASHINFER)
        return vv_attn_fi_decode_dev(q, kv, out, n_q_heads, cache_len,
                                     d_cache_len, scratch, stream);
    if (backend == VV_ATTN_FA2 &&
        vv_attn_gqa_decode_ok(n_q_heads, kv->n_kv_heads, kv->head_dim))
        return vv_attn_gqa_decode_dev(q, kv, out, n_q_heads, cache_len,
                                      d_cache_len, scratch, stream);
    if (kv->page_table) return VV_ERR_UNSUPPORTED;
    if (kv->format == VV_KV_FP16)
        return vv_gqa_attention_decode_dev(q, kv->k, kv->v, out, n_q_heads,
                                           kv->n_kv_heads, kv->head_dim,
                                           cache_len, d_cache_len, scratch,
                                           stream);
    return vv_gqa_attention_decode_q_dev(
        q, kv->k, kv->v, kv->k_meta, kv->v_meta, out, n_q_heads,
        kv->n_kv_heads, kv->head_dim, cache_len, d_cache_len, kv->format,
        scratch, stream);
}

size_t vv_attn_rows_scratch_bytes(int n_q_heads, int head_dim, int rows) {
    return vv_decode_parts_rows_bytes(n_q_heads, head_dim, rows > 0 ? rows : 1);
}

vv_status_t vv_attn_decode_rows(int backend, const void* q,
                                const vv_kv_view_t* kv, void* out,
                                int n_q_heads, int rows, int cache_len,
                                const int* d_cache_len, void* scratch,
                                void* stream)
{
    if (!q || !kv || !out) return VV_ERR_NULL_PTR;
    if (rows < 1) return VV_ERR_INVALID_ARG;
    if (backend == VV_ATTN_FLASHINFER)
        return vv_attn_fi_decode_rows_dev(q, kv, out, n_q_heads, rows,
                                          cache_len, d_cache_len, scratch,
                                          stream);
    if (backend == VV_ATTN_FA2 &&
        vv_attn_gqa_decode_ok(n_q_heads, kv->n_kv_heads, kv->head_dim))
        return vv_attn_gqa_decode_rows_dev(q, kv, out, n_q_heads, rows,
                                           cache_len, d_cache_len, scratch,
                                           stream);
    /* fa1: its one-row kernels, one row after another. Exact, not fast. */
    if (d_cache_len) return VV_ERR_UNSUPPORTED;
    const size_t row = (size_t)n_q_heads * (size_t)kv->head_dim * 2;
    for (int r = 0; r < rows; r++) {
        const vv_status_t s = vv_attn_decode(
            backend, (const uint8_t*)q + (size_t)r * row, kv,
            (uint8_t*)out + (size_t)r * row, n_q_heads, cache_len + r, NULL,
            scratch, stream);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

} /* extern "C" */
