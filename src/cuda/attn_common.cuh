/**
 * @file attn_common.cuh
 * @brief Tensor-core building blocks shared by the FlashAttention-2 and the
 *        FlashInfer-style kernels: mma/ldmatrix/cp.async wrappers with an
 *        sm_75 fallback, and the K/V tile loaders for every cache format.
 *
 * ## Fragment layout (PTX ISA, mma.m16n8k16, f16 in / f32 accumulate)
 *
 *   A reg 0 -> row g,   k 2q..2q+1        C reg 0,1 -> row g,   col 2q..2q+1
 *   A reg 1 -> row g+8, k 2q..2q+1        C reg 2,3 -> row g+8, col 2q..2q+1
 *   A reg 2 -> row g,   k 2q+8..
 *   A reg 3 -> row g+8, k 2q+8..          B reg 0 -> k 2q..2q+1,   col g
 *                                         B reg 1 -> k 2q+8..,     col g
 *
 * with g = lane / 4 and q = lane % 4. Two adjacent C tiles are exactly one A
 * fragment, so softmax probabilities feed P·V straight from registers.
 *
 * ## Turing
 *
 * sm_75 has ldmatrix and `mma.m16n8k8` but not the k16 shape or cp.async.
 * A k16 product is two k8 products over the halves of the same registers
 * (A regs {0,1} then {2,3}, B reg 0 then 1), and a cp.async becomes a
 * 16-byte load and store. The kernels are written once; only these wrappers
 * know the difference.
 */
#ifndef VV_CUDA_ATTN_COMMON_CUH
#define VV_CUDA_ATTN_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#include "kv_codec.cuh"

/* ─── Geometry shared by every tensor-core attention kernel ─────────────── */

#define ATT_D        128                 /* head_dim these kernels take    */
#define ATT_BC       64                  /* KV positions per tile = a page */
#define ATT_LD       (ATT_D + 8)         /* 136: conflict-free for ldmatrix */
#define ATT_WARPS    4
#define ATT_THREADS  (ATT_WARPS * 32)
#define ATT_TILE     (ATT_BC * ATT_LD)   /* halves in one K or V tile      */
#define ATT_SMEM_BYTES ((size_t)2 * ATT_TILE * sizeof(half))   /* 34816 */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define ATT_HAS_MMA 1
#else
#define ATT_HAS_MMA 0
#endif

/** @brief d += a · b over k = 16. */
__device__ __forceinline__ void att_mma(float d[4], const uint32_t a[4],
                                        uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

/** @brief Four 8x8 b16 tiles; lanes 8i..8i+7 address the rows of tile i. */
__device__ __forceinline__ void att_ldsm_x4(uint32_t r[4], const half* p)
{
#if ATT_HAS_MMA
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a) : "memory");
#else
    (void)p; r[0] = r[1] = r[2] = r[3] = 0u;
#endif
}

/** @brief The same, each tile transposed on the way (the V operand). */
__device__ __forceinline__ void att_ldsm_x4_t(uint32_t r[4], const half* p)
{
#if ATT_HAS_MMA
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a) : "memory");
#else
    (void)p; r[0] = r[1] = r[2] = r[3] = 0u;
#endif
}

/** @brief 16 bytes global -> shared, zero-filled when `pred` is false. */
__device__ __forceinline__ void att_cp16(half* s, const void* g, bool pred)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(a), "l"(g), "r"(n) : "memory");
#else
    *(uint4*)s = pred ? *(const uint4*)g : make_uint4(0u, 0u, 0u, 0u);
#endif
}

__device__ __forceinline__ void att_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

/** @brief Wait until at most N committed groups are still in flight. */
template <int N>
__device__ __forceinline__ void att_cp_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

__device__ __forceinline__ float att_quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    return v;
}

__device__ __forceinline__ float att_quad_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    return v;
}

__device__ __forceinline__ uint32_t att_pack(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t*)&h;
}

/**
 * @brief Two floats as a pair of FP16 pairs whose sum is them to ~22 bits:
 *        `hi` the rounded values, `lo` what rounding dropped.
 *
 * For probabilities in [0, 1] the remainder turns subnormal below p ~ 2^-3,
 * where its absolute error (under 2^-25) no longer matters.
 */
__device__ __forceinline__ void att_split_pair(float a, float b,
                                               uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - f.x, b - f.y);
    hi = *(const uint32_t*)&h;
    lo = *(const uint32_t*)&l;
}

/**
 * @brief The A fragments of 16 query rows, straight from global memory.
 *
 * Rows are (position, head) pairs; a row that does not exist reads as zero
 * and is dropped at the store. Loaded once per block, so no shared staging.
 */
__device__ __forceinline__ void att_load_q(uint32_t a_q[ATT_D / 16][4],
                                           const half* q_lo, bool ok_lo,
                                           const half* q_hi, bool ok_hi,
                                           int quad)
{
    #pragma unroll
    for (int s = 0; s < ATT_D / 16; ++s) {
        const int c0 = s * 16 + quad * 2;
        a_q[s][0] = ok_lo ? *(const uint32_t*)(q_lo + c0)     : 0u;
        a_q[s][1] = ok_hi ? *(const uint32_t*)(q_hi + c0)     : 0u;
        a_q[s][2] = ok_lo ? *(const uint32_t*)(q_lo + c0 + 8) : 0u;
        a_q[s][3] = ok_hi ? *(const uint32_t*)(q_hi + c0 + 8) : 0u;
    }
}

/**
 * @brief Bring one 64-position tile of K or V into shared memory as FP16.
 *
 * FP16 goes through cp.async (or a plain copy on Turing) and the caller
 * commits and waits. Every other format is decoded by one warp per position,
 * the layout the store wrote it in, and lands before this returns. FP8
 * values are exact in FP16; TurboQuant values round, as they always have in
 * the scalar prefill.
 *
 * Positions at or past `kv_len` become zeros, not whatever the store holds:
 * a masked probability is 0, and 0 times a stale NaN is still NaN.
 *
 * `kv0` is a multiple of ATT_BC, so the tile sits inside one page.
 */
template <int FMT>
__device__ __forceinline__ void att_load_tile(
    half* __restrict__ tile, const uint8_t* __restrict__ store,
    const half* __restrict__ meta, const int* __restrict__ page_table,
    int kv0, int kv_len, int n_kv_heads, int kv_head, int bpv, int tid)
{
    const int prow0 = page_table
        ? page_table[kv0 / VV_KV_PAGE_SIZE] * VV_KV_PAGE_SIZE : kv0;

    if (FMT == VV_KV_FP16) {
        const half* base = (const half*)store;
        #pragma unroll
        for (int i = 0; i < (ATT_BC * ATT_D / 8) / ATT_THREADS; ++i) {
            const int idx = i * ATT_THREADS + tid;
            const int r = idx >> 4;              /* 16 chunks per row */
            const int c = (idx & 15) * 8;
            const bool ok = kv0 + r < kv_len;
            const half* g = base
                + ((size_t)(prow0 + r) * n_kv_heads + kv_head) * ATT_D + c;
            att_cp16(tile + r * ATT_LD + c, ok ? g : base, ok);
        }
    } else {
        const int warp = tid >> 5, lane = tid & 31;
        #pragma unroll 4
        for (int r = warp; r < ATT_BC; r += ATT_WARPS) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (kv0 + r < kv_len) {                       /* warp-uniform */
                const size_t vi = (size_t)(prow0 + r) * n_kv_heads + kv_head;
                kv_read4<FMT>(store + vi * (size_t)bpv,
                              meta ? meta + vi : NULL, lane, v);
            }
            uint2 w;
            w.x = att_pack(v[0], v[1]);
            w.y = att_pack(v[2], v[3]);
            *(uint2*)(tile + r * ATT_LD + lane * 4) = w;
        }
    }
}

/* ─── Quantized tiles: fetch to registers, decode into shared memory ────── */

/*
 * A quantized tile cannot go through cp.async — it has to be decoded — so it
 * is split in two: `att_q_fetch` issues the global loads into registers and
 * `att_q_commit` decodes them into the FP16 tile. The kernels issue a fetch,
 * do the matmul the previous tile is waiting on, and only then commit, so
 * the load latency hides behind tensor-core work the same way cp.async hides
 * it for FP16.
 *
 * Each thread owns whole runs of one vector rather than a lane's 4 values,
 * so nothing is shuffled: FP8 is 16 values per 16-byte load, TurboQuant a
 * run of 32 values (64 for tq1.5), which is a whole number of 32-bit words
 * for every bit width because value d sits at bit bits*d of the vector.
 */

template <int FMT> struct AttQ {
    enum {
        FP8    = (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2),
        TQ     = (FMT >= VV_KV_TQ4 && FMT <= VV_KV_TQ1_5),
        RUN    = FP8 ? 16 : (FMT == VV_KV_TQ1_5 ? 64 : 32), /* values/run */
        RUNS   = (ATT_BC * ATT_D / RUN) / ATT_THREADS,      /* per thread */
        /* 32-bit words per run: FP8 one byte per value, TQ bits/4 per 4. */
        WORDS  = FP8 ? 4 : (TQ ? (KVBits<FMT>::N * RUN / 4) / 32 : 1),
        LUT    = (FMT == VV_KV_TQ4) ? 16 : (FMT == VV_KV_TQ3) ? 8
               : (FMT == VV_KV_TQ2) ? 4 : (FMT == VV_KV_TQ1_5) ? 16 : 0,
    };
};

/** @brief What one thread holds between fetch and commit. */
template <int FMT> struct AttQRaw {
    uint32_t w[AttQ<FMT>::RUNS][AttQ<FMT>::WORDS];
    float    sigma[AttQ<FMT>::RUNS];
};

/** @brief The codebook, copied to shared memory once per block. */
template <int FMT>
__device__ __forceinline__ void att_q_lut_init(float* lut, int tid)
{
    if (tid < AttQ<FMT>::LUT) {
        if (FMT == VV_KV_TQ4)        lut[tid] = c_lev4[tid];
        else if (FMT == VV_KV_TQ3)   lut[tid] = c_lev3[tid];
        else if (FMT == VV_KV_TQ2)   lut[tid] = c_lev2[tid];
        else if (FMT == VV_KV_TQ1_5) lut[tid] = c_vq15[tid];
    }
}

template <int FMT>
__device__ __forceinline__ void att_q_fetch(
    AttQRaw<FMT>& raw, const uint8_t* __restrict__ store,
    const half* __restrict__ meta, const int* __restrict__ page_table,
    int kv0, int kv_len, int n_kv_heads, int kv_head, int bpv, int tid)
{
    typedef AttQ<FMT> Q;
    if (!Q::FP8 && !Q::TQ) return;
    const int prow0 = page_table
        ? page_table[kv0 / VV_KV_PAGE_SIZE] * VV_KV_PAGE_SIZE : kv0;
    const int runs_per_row = ATT_D / Q::RUN;
    #pragma unroll
    for (int i = 0; i < Q::RUNS; ++i) {
        const int idx = i * ATT_THREADS + tid;
        const int r = idx / runs_per_row;
        const int j = idx % runs_per_row;
        const bool ok = kv0 + r < kv_len;
        const size_t vi = (size_t)(prow0 + r) * n_kv_heads + kv_head;
        const uint32_t* src = (const uint32_t*)(store + vi * (size_t)bpv)
                            + j * Q::WORDS;
        if (Q::WORDS == 4) {
            const uint4 u = ok ? *(const uint4*)src : make_uint4(0u, 0u, 0u, 0u);
            raw.w[i][0] = u.x; raw.w[i][1] = u.y;
            raw.w[i][2 % Q::WORDS] = u.z; raw.w[i][3 % Q::WORDS] = u.w;
        } else if (Q::WORDS == 2) {
            const uint2 u = ok ? *(const uint2*)src : make_uint2(0u, 0u);
            raw.w[i][0] = u.x; raw.w[i][1 % Q::WORDS] = u.y;
        } else {
            #pragma unroll
            for (int k = 0; k < Q::WORDS; ++k) raw.w[i][k] = ok ? src[k] : 0u;
        }
        raw.sigma[i] = (Q::TQ && ok) ? __half2float(meta[vi]) : 0.0f;
    }
}

/** @brief Bits [b, b + n) of a word run, n <= 32, b a compile-time value. */
template <int WORDS>
__device__ __forceinline__ uint32_t att_bits(const uint32_t (&w)[WORDS],
                                             int b, int n)
{
    const int k = b >> 5, sh = b & 31;
    uint32_t v = w[k] >> sh;
    if (sh + n > 32 && k + 1 < WORDS) v |= w[k + 1] << (32 - sh);
    return v & ((n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u));
}

template <int FMT>
__device__ __forceinline__ void att_q_commit(
    const AttQRaw<FMT>& raw, half* __restrict__ tile,
    const float* __restrict__ lut, int tid)
{
    typedef AttQ<FMT> Q;
    if (!Q::FP8 && !Q::TQ) return;
    const int runs_per_row = ATT_D / Q::RUN;
    #pragma unroll
    for (int i = 0; i < Q::RUNS; ++i) {
        const int idx = i * ATT_THREADS + tid;
        const int r = idx / runs_per_row;
        const int j = idx % runs_per_row;
        uint32_t h[Q::RUN / 2];                    /* packed half2 */
        if (Q::FP8) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t x = raw.w[i][k % Q::WORDS];
                if (FMT == VV_KV_FP8_E5M2) {
                    /* E5M2 is the top byte of an FP16: shift into place. */
                    h[2 * k]     = ((x << 8) & 0xFF00u) | ((x << 16) & 0xFF000000u);
                    h[2 * k + 1] = ((x >> 8) & 0xFF00u) | (x & 0xFF000000u);
                } else {
                    /* E4M3: sign and the 7 other bits land as an FP16 with
                     * bias 15 instead of 7; one exact multiply by 2^8 fixes
                     * the exponent, subnormals included. */
                    const uint32_t lo = ((x & 0x80u) << 8)  | ((x & 0x7Fu) << 7)
                                      | ((x & 0x8000u) << 16) | ((x & 0x7F00u) << 15);
                    const uint32_t y = x >> 16;
                    const uint32_t hi = ((y & 0x80u) << 8)  | ((y & 0x7Fu) << 7)
                                      | ((y & 0x8000u) << 16) | ((y & 0x7F00u) << 15);
                    const __half2 s = __float2half2_rn(256.0f);
                    __half2 a = __hmul2(*(const __half2*)&lo, s);
                    __half2 b = __hmul2(*(const __half2*)&hi, s);
                    h[2 * k]     = *(const uint32_t*)&a;
                    h[2 * k + 1] = *(const uint32_t*)&b;
                }
            }
        } else {
            const float sg = raw.sigma[i];
            if (FMT == VV_KV_TQ1_5) {
                #pragma unroll
                for (int p = 0; p < Q::RUN / 2; ++p) {
                    const uint32_t c = att_bits(raw.w[i], 3 * p, 3);
                    h[p] = att_pack(sg * lut[2 * c], sg * lut[2 * c + 1]);
                }
            } else {
                const int B = KVBits<FMT>::N / 4;
                #pragma unroll
                for (int p = 0; p < Q::RUN / 2; ++p) {
                    const uint32_t c0 = att_bits(raw.w[i], B * (2 * p), B);
                    const uint32_t c1 = att_bits(raw.w[i], B * (2 * p + 1), B);
                    h[p] = att_pack(sg * lut[c0], sg * lut[c1]);
                }
            }
        }
        uint4* dst = (uint4*)(tile + r * ATT_LD + j * Q::RUN);
        #pragma unroll
        for (int k = 0; k < Q::RUN / 8; ++k)
            dst[k] = make_uint4(h[4 * k], h[4 * k + 1], h[4 * k + 2], h[4 * k + 3]);
    }
}

/**
 * @brief Walk `n_tiles` KV tiles from `begin`, calling `fs(kv0)` once K is
 *        in shared memory and `fpv()` once V is.
 *
 * FP16: K and V on separate cp.async groups (V(t) lands during S(t), K(t+1)
 * during P·V(t)), four barriers per tile. Quantized: fetch-to-registers ahead
 * of the matmul that hides it, decode after, two barriers per tile. Either
 * way every thread reaches every barrier: the bounds are per block.
 */
template <int FMT, typename FS, typename FPV>
__device__ __forceinline__ void att_run_tiles(
    half* Ks, half* Vs,
    const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const half* __restrict__ K_meta, const half* __restrict__ V_meta,
    const int* __restrict__ page_table, const float* lut,
    int begin, int n_tiles, int kv_len, int n_kv_heads, int kv_head, int bpv,
    int tid, FS&& fs, FPV&& fpv)
{
    /* The first ATT_THREADS threads move tiles; a wider block's other warps
     * only compute (they still meet every barrier). */
    const bool ld = tid < ATT_THREADS;
    if (FMT == VV_KV_FP16) {
        if (ld && n_tiles > 0)
            att_load_tile<FMT>(Ks, K, K_meta, page_table, begin, kv_len,
                               n_kv_heads, kv_head, bpv, tid);
        att_cp_commit();
        for (int t = 0; t < n_tiles; ++t) {
            const int kv0 = begin + t * ATT_BC;
            if (ld)
                att_load_tile<FMT>(Vs, V, V_meta, page_table, kv0, kv_len,
                                   n_kv_heads, kv_head, bpv, tid);
            att_cp_commit();
            att_cp_wait<1>();                 /* K(t) has landed */
            __syncthreads();
            fs(kv0);
            __syncthreads();                  /* everyone is done with K(t) */
            if (ld && t + 1 < n_tiles)
                att_load_tile<FMT>(Ks, K, K_meta, page_table, kv0 + ATT_BC,
                                   kv_len, n_kv_heads, kv_head, bpv, tid);
            att_cp_commit();
            att_cp_wait<1>();                 /* V(t) has landed */
            __syncthreads();
            fpv();
            __syncthreads();                  /* everyone is done with V(t) */
        }
        att_cp_wait<0>();
    } else {
        AttQRaw<FMT> raw;
        if (ld && n_tiles > 0) {
            att_q_fetch<FMT>(raw, K, K_meta, page_table, begin, kv_len,
                             n_kv_heads, kv_head, bpv, tid);
            att_q_commit<FMT>(raw, Ks, lut, tid);
        }
        for (int t = 0; t < n_tiles; ++t) {
            const int kv0 = begin + t * ATT_BC;
            if (ld)
                att_q_fetch<FMT>(raw, V, V_meta, page_table, kv0, kv_len,
                                 n_kv_heads, kv_head, bpv, tid);
            __syncthreads();                  /* K(t) visible, P·V(t-1) done */
            fs(kv0);
            if (ld) att_q_commit<FMT>(raw, Vs, lut, tid);
            if (ld && t + 1 < n_tiles)
                att_q_fetch<FMT>(raw, K, K_meta, page_table, kv0 + ATT_BC,
                                 kv_len, n_kv_heads, kv_head, bpv, tid);
            __syncthreads();                  /* V(t) visible, S(t) done */
            fpv();
            if (ld && t + 1 < n_tiles) att_q_commit<FMT>(raw, Ks, lut, tid);
        }
        __syncthreads();
    }
}

/**
 * @brief Online-softmax step for one 16-row fragment over `NT` n-tiles.
 *
 * Scales S, masks unless the tile is known fully visible, and rescales the
 * running output. Probabilities are left in `s`. Natural-base `__expf` on
 * the scaled score, exactly as the kernels these replace computed it: with
 * the same MMA order, FP16 prefill stays bit-identical to them.
 *
 * @param pos0   KV position of n-tile 0's first column
 * @param qa_lo  Absolute position of row g (row g+8: qa_hi)
 */
template <int NT, bool MASK>
__device__ __forceinline__ void att_softmax(
    float s[NT][4], float o[ATT_D / 8][4],
    float& m_lo, float& m_hi, float& l_lo, float& l_hi,
    float scale, int pos0, int quad, int kv_len, bool causal,
    int qa_lo, int qa_hi)
{
    float mr_lo = -FLT_MAX, mr_hi = -FLT_MAX;
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v = s[n][i] * scale;
            if (MASK) {
                const int kp = pos0 + n * 8 + quad * 2 + (i & 1);
                const int qa = (i < 2) ? qa_lo : qa_hi;
                const bool ok = kp < kv_len && (!causal || kp <= qa);
                v = ok ? v : -FLT_MAX;
            }
            s[n][i] = v;
            if (i < 2) mr_lo = fmaxf(mr_lo, v);
            else       mr_hi = fmaxf(mr_hi, v);
        }
    }
    mr_lo = att_quad_max(mr_lo);
    mr_hi = att_quad_max(mr_hi);

    const float mn_lo = fmaxf(m_lo, mr_lo);
    const float mn_hi = fmaxf(m_hi, mr_hi);
    const float a_lo = (mn_lo > -FLT_MAX && m_lo > -FLT_MAX)
                       ? __expf(m_lo - mn_lo) : 1.0f;
    const float a_hi = (mn_hi > -FLT_MAX && m_hi > -FLT_MAX)
                       ? __expf(m_hi - mn_hi) : 1.0f;

    float sum_lo = 0.0f, sum_hi = 0.0f;
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float m = (i < 2) ? mn_lo : mn_hi;
            float p;
            if (MASK) p = (s[n][i] > -FLT_MAX && m > -FLT_MAX)
                          ? __expf(s[n][i] - m) : 0.0f;
            else      p = __expf(s[n][i] - m);
            s[n][i] = p;
            if (i < 2) sum_lo += p; else sum_hi += p;
        }
    }
    l_lo = l_lo * a_lo + att_quad_sum(sum_lo);
    l_hi = l_hi * a_hi + att_quad_sum(sum_hi);

    #pragma unroll
    for (int n = 0; n < ATT_D / 8; ++n) {
        o[n][0] *= a_lo; o[n][1] *= a_lo;
        o[n][2] *= a_hi; o[n][3] *= a_hi;
    }
    m_lo = mn_lo;
    m_hi = mn_hi;
}

/**
 * @brief Merge split-KV partials: one block per output row, one thread per
 *        dimension.
 *
 * Partials are [row][n_parts][head_dim], rows being [position][head], so the
 * row index is also the output row.
 */
static __global__ void att_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, half* __restrict__ out, int n_parts)
{
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    const float* pm = part_m + (size_t)row * n_parts;
    const float* pl = part_l + (size_t)row * n_parts;

    float gmax = -FLT_MAX;
    for (int i = 0; i < n_parts; ++i) gmax = fmaxf(gmax, pm[i]);

    float num = 0.0f, den = 0.0f;
    if (gmax > -FLT_MAX) {
        for (int i = 0; i < n_parts; ++i) {
            const float mi = pm[i];
            if (mi <= -FLT_MAX) continue;
            const float w = __expf(mi - gmax);
            num += w * part_o[((size_t)row * n_parts + i) * ATT_D + d];
            den += w * pl[i];
        }
    }
    out[(size_t)row * ATT_D + d] = __float2half(den > 1e-20f ? num / den : 0.0f);
}

#endif /* VV_CUDA_ATTN_COMMON_CUH */
