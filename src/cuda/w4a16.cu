/**
 * @file w4a16.cu
 * @brief Fast INT4 group-affine (AWQ / GPTQ) linear layers: GEMV and a
 *        Marlin-style tensor-core GEMM that share one weight layout.
 *
 * Weight layout ("GPU layout", produced once at upload by
 * vv_int4g_to_gpu_layout from the row-major INT4G form):
 *
 *   packed [N][K/2] bytes. Every row is still contiguous in k, cut into
 *          16-byte chunks of 32 weights. Inside a chunk, 32-bit word t
 *          (t = 0..3) holds the k-pairs t, t+4, t+8 and t+12 (pair p is
 *          k = 2p, 2p+1), nibble slot s of the word being
 *              pair t + 4*(s & 3), element (s >> 2) of the pair.
 *   sz     [N][K/G] half2 = { scale, zero } with the zero point kept as the
 *          exact integer the checkpoint stored.
 *
 * Why this order: one LOP3 with the 0x6400 magic turns nibbles 0/4, 1/5,
 * 2/6 and 3/7 of a word into half2 values 1024 + q (or 1024 + 16q), which
 * are exactly the pairs {t, t+4, t+8, t+12}. For the GEMV these are the
 * natural half2 pairs of x; for mma.m16n8k16 they are the B fragment of
 * lane (g, t) for two consecutive k16 steps (b0 = pair t, b1 = pair t+4),
 * so the tensor-core kernel dequantizes straight into registers with no
 * shuffles and no shared-memory round trip of FP16 weights.
 *
 * Dequantization is w = (q - z) * s in FP16, which is the reference's own
 * formula: q - z is exact (small integers), and the one multiply rounds
 * exactly like PyTorch's (qweight - qzeros) * scales does. The old path
 * computed q * s + fp16(-z * s), which differs in the last bit.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "vibevoice/device.h"

/* ─── Small helpers ──────────────────────────────────────────────────────── */

__device__ __forceinline__ uint32_t w4_lop3(uint32_t a, uint32_t mask,
                                            uint32_t magic) {
    uint32_t r;
    /* (a & mask) | magic */
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
        : "=r"(r) : "r"(a), "r"(mask), "r"(magic));
    return r;
}

__device__ __forceinline__ half2 w4_u2h(uint32_t v) {
    return *reinterpret_cast<half2*>(&v);
}

__device__ __forceinline__ uint32_t w4_h2u(half2 v) {
    return *reinterpret_cast<uint32_t*>(&v);
}

/** @brief Per-group constants: s, 1024 + z and -(64 + z), each as half2. */
struct w4_group {
    half2 s2;
    half2 c1;
    half2 c16;
};

__device__ __forceinline__ w4_group w4_make_group(half2 sz) {
    w4_group g;
    g.s2  = __half2half2(__low2half(sz));
    const half z = __high2half(sz);
    g.c1  = __half2half2(__hadd(z, __float2half(1024.0f)));
    g.c16 = __half2half2(__hneg(__hadd(z, __float2half(64.0f))));
    return g;
}

/**
 * @brief One packed word -> four half2 weights (q - z) * s.
 *
 * out[j] is pair t + 4j of the chunk the word belongs to.
 */
__device__ __forceinline__ void w4_dequant_word(uint32_t w, const w4_group& g,
                                                half2 out[4]) {
    const uint32_t top = w >> 8;
    const uint32_t MAGIC = 0x64006400u;
    const half2 h0 = w4_u2h(w4_lop3(w,   0x000F000Fu, MAGIC));
    const half2 h1 = w4_u2h(w4_lop3(w,   0x00F000F0u, MAGIC));
    const half2 h2 = w4_u2h(w4_lop3(top, 0x000F000Fu, MAGIC));
    const half2 h3 = w4_u2h(w4_lop3(top, 0x00F000F0u, MAGIC));
    const half2 inv16 = __float2half2_rn(0.0625f);
    out[0] = __hmul2(__hsub2(h0, g.c1), g.s2);
    out[1] = __hmul2(__hfma2(h1, inv16, g.c16), g.s2);
    out[2] = __hmul2(__hsub2(h2, g.c1), g.s2);
    out[3] = __hmul2(__hfma2(h3, inv16, g.c16), g.s2);
}

/** @brief 16-byte weight load that does not pollute L1 (read once). */
__device__ __forceinline__ uint4 w4_ld_stream(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GEMV: y[N] = W . x + bias, for up to W4V_MAX_SEGS projections that share x
 *
 * LPR lanes share R output rows (LPR 16..128; above 32 the rows span several
 * warps and the partial sums meet in shared memory). Each lane walks the
 * rows' 16-byte chunks with a stride of LPR; the x values of a chunk are
 * loaded once and used for all R rows. Per chunk and row the 32 products
 * are formed as four HFMA2 chains of four and flushed to FP32, so no FP16
 * partial ever sums more than four terms.
 *
 * Several projections of the same input (q/k/v, gate/up) go in one launch:
 * the grid is the concatenation of their row blocks and each block looks up
 * which projection it belongs to. The 0.9 MB k and v projections on their
 * own are too short to get past the kernel's ramp-up; behind q they ride
 * along at q's bandwidth.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define W4V_THREADS 128
#define W4V_MAX_SEGS 3

/** @brief The projections of one launch, passed by value (kernel params). */
struct w4v_segs {
    const uint4* w[W4V_MAX_SEGS];
    const half2* sz[W4V_MAX_SEGS];
    const half*  bias[W4V_MAX_SEGS];
    half*        y[W4V_MAX_SEGS];
    int          n[W4V_MAX_SEGS];
    int          first_block[W4V_MAX_SEGS];   /* unused: INT_MAX */
};

template <int LPR, int R>
__global__ void __launch_bounds__(W4V_THREADS)
w4a16_gemv_kernel(const half* __restrict__ x, const w4v_segs segs,
                  int K, int gshift)
{
    constexpr int GPB = W4V_THREADS / LPR;            /* lane groups/block */
    constexpr int WIDTH = LPR < 32 ? LPR : 32;

    /* Which projection this block serves; selects, not indexed loads. */
    const int bx = (int)blockIdx.x;
    const int seg = bx >= segs.first_block[2] ? 2
                  : (bx >= segs.first_block[1] ? 1 : 0);
    const uint4* w = seg == 0 ? segs.w[0] : (seg == 1 ? segs.w[1] : segs.w[2]);
    const half2* sz = seg == 0 ? segs.sz[0]
                    : (seg == 1 ? segs.sz[1] : segs.sz[2]);
    const half* bias = seg == 0 ? segs.bias[0]
                     : (seg == 1 ? segs.bias[1] : segs.bias[2]);
    half* y = seg == 0 ? segs.y[0] : (seg == 1 ? segs.y[1] : segs.y[2]);
    const int N = seg == 0 ? segs.n[0] : (seg == 1 ? segs.n[1] : segs.n[2]);
    const int blk = bx - (seg == 0 ? 0 : (seg == 1 ? segs.first_block[1]
                                                   : segs.first_block[2]));

    const int row0 = (blk * GPB + (int)threadIdx.x / LPR) * R;
    const int lr  = (int)threadIdx.x % LPR;

    const int nch = row0 < N ? (K >> 5) : 0;
    const int ngroups = K >> gshift;
    const uint4* wrow[R];
    const half2* szrow[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int rr = row0 + r < N ? row0 + r : N - 1;
        wrow[r]  = w  + (size_t)rr * (K >> 5);
        szrow[r] = sz + (size_t)rr * ngroups;
    }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    for (int c = lr; c < nch; c += LPR) {
        uint4 wv[R];
        half2 sv[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            wv[r] = w4_ld_stream(wrow[r] + c);
            sv[r] = szrow[r][(c << 5) >> gshift];
        }
        const uint4* xp = (const uint4*)(x + (c << 5));
        uint4 xv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) xv[i] = __ldg(xp + i);
        const half2* xh = (const half2*)xv;           /* pairs 0..15 */
#pragma unroll
        for (int r = 0; r < R; r++) {
            const w4_group g = w4_make_group(sv[r]);
            const uint32_t wd[4] = { wv[r].x, wv[r].y, wv[r].z, wv[r].w };
#pragma unroll
            for (int t = 0; t < 4; t++) {
                half2 wh[4];
                w4_dequant_word(wd[t], g, wh);
                /* word t, slot j -> pair t + 4j */
                half2 p = __hmul2(wh[0], xh[t]);
                p = __hfma2(wh[1], xh[t + 4],  p);
                p = __hfma2(wh[2], xh[t + 8],  p);
                p = __hfma2(wh[3], xh[t + 12], p);
                const float2 f = __half22float2(p);
                acc[r] += f.x + f.y;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int off = WIDTH / 2; off > 0; off >>= 1)
            acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off, WIDTH);
    }
    if (LPR > 32) {
        constexpr int WPG = LPR / 32;                  /* warps per group */
        __shared__ float red[R][W4V_THREADS / 32];
        const int warp = (int)threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int r = 0; r < R; r++) red[r][warp] = acc[r];
        }
        __syncthreads();
        if (lr == 0) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                float v = 0.0f;
#pragma unroll
                for (int i = 0; i < WPG; i++) v += red[r][warp + i];
                acc[r] = v;
            }
        }
    }
    if (lr == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= N) break;
            const float b = bias ? __half2float(bias[row]) : 0.0f;
            y[row] = __float2half(acc[r] + b);
        }
    }
}

/*
 * The same GEMV for up to 8 rows of x (a drafted block being checked). Every
 * output is the one-row kernel's arithmetic exactly: lane lr of a lane group
 * takes chunks lr, lr + LPR, ... of its weight row in that order, forms the
 * same four HFMA2 chains per chunk, flushes them to FP32 the same way and
 * reduces over the same butterfly -- so LPR must be the one-row launch's.
 *
 * What differs is where x comes from and how much weight a lane carries. x
 * (8 rows x K) is staged in shared memory in as few tiles as fit (one for
 * K = 3584, four for 18944), laid out [row][word][chunk] so a warp reads
 * consecutive 16-byte words; within a tile the lane groups stream their
 * weights as the one-row kernel does, with no barrier. A lane group takes R
 * weight rows at once (R never changes how a row is summed), dequantizes
 * each chunk once and reads each x chunk from shared memory once for all R,
 * and loads the next chunk's weights before it computes the current one.
 */
#define W4R_THREADS 128
#define W4R_MX 8
/* Chunks of x per tile: 8 rows x 64 chunks x 64 bytes = 32 KB, three
 * blocks to an SM. */
#define W4R_TILE_MAX 64

template <int LPR, int R>
__global__ void __launch_bounds__(W4R_THREADS)
w4a16_gemv_rows_kernel(const half* __restrict__ x, int M, const w4v_segs segs,
                       int K, int gshift, int tile_chunks)
{
    constexpr int GPB = W4R_THREADS / LPR;
    constexpr int WIDTH = LPR < 32 ? LPR : 32;
    extern __shared__ uint4 xs[];            /* [W4R_MX][4][tile_chunks] */

    const int bx = (int)blockIdx.x;
    const int seg = bx >= segs.first_block[2] ? 2
                  : (bx >= segs.first_block[1] ? 1 : 0);
    const uint4* w = seg == 0 ? segs.w[0] : (seg == 1 ? segs.w[1] : segs.w[2]);
    const half2* sz = seg == 0 ? segs.sz[0]
                    : (seg == 1 ? segs.sz[1] : segs.sz[2]);
    const half* bias = seg == 0 ? segs.bias[0]
                     : (seg == 1 ? segs.bias[1] : segs.bias[2]);
    half* y = seg == 0 ? segs.y[0] : (seg == 1 ? segs.y[1] : segs.y[2]);
    const int N = seg == 0 ? segs.n[0] : (seg == 1 ? segs.n[1] : segs.n[2]);
    const int blk = bx - (seg == 0 ? 0 : (seg == 1 ? segs.first_block[1]
                                                   : segs.first_block[2]));

    const int lg  = (int)threadIdx.x / LPR;
    const int lr  = (int)threadIdx.x % LPR;
    const int row0 = (blk * GPB + lg) * R;
    const int nch = K >> 5;
    const int ngroups = K >> gshift;
    const bool live = row0 < N;

    const uint4* wrow[R];
    const half2* szrow[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int rr = row0 + r < N ? row0 + r : N - 1;
        wrow[r]  = w  + (size_t)rr * nch;
        szrow[r] = sz + (size_t)rr * ngroups;
    }

    float acc[R][W4R_MX];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < W4R_MX; m++) acc[r][m] = 0.0f;

    for (int t0 = 0; t0 < nch; t0 += tile_chunks) {
        const int tc = nch - t0 < tile_chunks ? nch - t0 : tile_chunks;
        if (t0 > 0) __syncthreads();              /* last tile is read */
        for (int e = (int)threadIdx.x; e < W4R_MX * 4 * tc; e += W4R_THREADS) {
            const int m = e / (4 * tc);
            const int j = (e / tc) & 3;
            const int cc = e % tc;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (m < M)
                v = __ldg((const uint4*)(x + (size_t)m * K +
                                         ((size_t)(t0 + cc) << 5)) + j);
            xs[(m * 4 + j) * tile_chunks + cc] = v;
        }
        __syncthreads();
        if (!live) continue;

        /* This lane's chunks of the tile: t0 + lr + i * LPR. */
        int c = t0 + lr;
        const int c_end = t0 + tc;
        uint4 wv[R];
        half2 sv[R];
        if (c < c_end) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                wv[r] = w4_ld_stream(wrow[r] + c);
                sv[r] = szrow[r][(c << 5) >> gshift];
            }
        }
        for (; c < c_end; c += LPR) {
            half2 wh[R][4][4];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const w4_group g = w4_make_group(sv[r]);
                const uint32_t wd[4] = { wv[r].x, wv[r].y, wv[r].z, wv[r].w };
#pragma unroll
                for (int t = 0; t < 4; t++) w4_dequant_word(wd[t], g, wh[r][t]);
            }
            /* The next chunk's weights are in flight while this one runs. */
            const int cn = c + LPR;
            if (cn < c_end) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    wv[r] = w4_ld_stream(wrow[r] + cn);
                    sv[r] = szrow[r][(cn << 5) >> gshift];
                }
            }
            const int cc = c - t0;
#pragma unroll
            for (int m = 0; m < W4R_MX; m++) {
                if (m < M) {
                    uint4 xv[4];
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        xv[j] = xs[(m * 4 + j) * tile_chunks + cc];
                    const half2* xh = (const half2*)xv;
#pragma unroll
                    for (int r = 0; r < R; r++) {
#pragma unroll
                        for (int t = 0; t < 4; t++) {
                            half2 p = __hmul2(wh[r][t][0], xh[t]);
                            p = __hfma2(wh[r][t][1], xh[t + 4],  p);
                            p = __hfma2(wh[r][t][2], xh[t + 8],  p);
                            p = __hfma2(wh[r][t][3], xh[t + 12], p);
                            const float2 f = __half22float2(p);
                            acc[r][m] += f.x + f.y;
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < W4R_MX; m++)
#pragma unroll
            for (int off = WIDTH / 2; off > 0; off >>= 1)
                acc[r][m] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][m], off, WIDTH);
    if (LPR > 32) {
        constexpr int WPG = LPR / 32;
        __shared__ float red[R][W4R_MX][W4R_THREADS / 32];
        const int warp = (int)threadIdx.x >> 5;
        __syncthreads();
        if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int m = 0; m < W4R_MX; m++) red[r][m][warp] = acc[r][m];
        }
        __syncthreads();
        if (lr == 0) {
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int m = 0; m < W4R_MX; m++) {
                    float v = 0.0f;
#pragma unroll
                    for (int i = 0; i < WPG; i++) v += red[r][m][warp + i];
                    acc[r][m] = v;
                }
        }
    }
    if (lr == 0 && live) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= N) break;
            const float b = bias ? __half2float(bias[row]) : 0.0f;
#pragma unroll
            for (int m = 0; m < W4R_MX; m++)
                if (m < M) y[(size_t)m * N + row] = __float2half(acc[r][m] + b);
        }
    }
}

/** @brief out[m][j] = x[m][perm[j]]: act-order activations, 8 columns/thread. */
__global__ void w4a16_gather_kernel(const half* __restrict__ x,
                                    const int32_t* __restrict__ perm,
                                    half* __restrict__ out, int M, int K)
{
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const size_t total = (size_t)M * K;
    if (i >= total) return;
    const int m = (int)(i / K);
    const int j = (int)(i % K);
    const half* xr = x + (size_t)m * K;
    const int4 p0 = *(const int4*)(perm + j);
    const int4 p1 = *(const int4*)(perm + j + 4);
    half v[8];
    v[0] = xr[p0.x]; v[1] = xr[p0.y]; v[2] = xr[p0.z]; v[3] = xr[p0.w];
    v[4] = xr[p1.x]; v[5] = xr[p1.y]; v[6] = xr[p1.z]; v[7] = xr[p1.w];
    *(uint4*)(out + i) = *(const uint4*)v;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tensor-core GEMM (sm_80+): C[M,N] = A[M,K] . W^T (+ bias)
 *
 * Block tile BM x BN x BK, WM x WN warps, warp tile (MT*16) x (BN/WN).
 * A goes through shared memory with cp.async and ldmatrix; W stays packed
 * INT4 all the way into shared memory (cp.async, rows padded so the per-lane
 * 32-bit reads are conflict-free) and is dequantized in registers straight
 * into the B fragments. BK is what sets the length of each contiguous run
 * read from a weight row: at 32 bytes (BK = 64) the DRAM pages thrash and a
 * 16-row GEMM moves weights at a third of the GEMV's rate, so the small-M
 * tiles use BK = 128/256. With a split-K grid (gridDim.z > 1) each slice
 * writes FP32 partials and a second kernel sums them and adds the bias.
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define W4G_SM80 1
#endif

template <int BM, int BN, int BK>
struct w4g_shape {
    static constexpr int ASTR = BK + 8;               /* halves per A row  */
    static constexpr int WSTR = BK / 2 + 16;          /* bytes per W row   */
    static constexpr int SSTR = BK / 32;              /* half2 per sz row  */
    static constexpr int A_BYTES = BM * ASTR * 2;
    static constexpr int W_BYTES = BN * WSTR;
    static constexpr int S_BYTES = BN * SSTR * 4;
    static constexpr int STAGE = A_BYTES + W_BYTES + S_BYTES;
};

__device__ __forceinline__ void w4g_cp_async(void* smem, const void* gmem,
                                             int bytes, bool valid) {
#ifdef W4G_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    const int src = valid ? bytes : 0;
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(s), "l"(gmem), "r"(src));
    else if (bytes == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                     :: "r"(s), "l"(gmem), "r"(src));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                     :: "r"(s), "l"(gmem), "r"(src));
#else
    (void)smem; (void)gmem; (void)bytes; (void)valid;
#endif
}

__device__ __forceinline__ void w4g_commit(void) {
#ifdef W4G_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N_PENDING>
__device__ __forceinline__ void w4g_wait(void) {
#ifdef W4G_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_PENDING));
#endif
}

__device__ __forceinline__ void w4g_ldmatrix_x4(uint32_t r[4], const void* p) {
#ifdef W4G_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#else
    (void)p; r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void w4g_mma(float d[4], const uint32_t a[4],
                                        uint32_t b0, uint32_t b1) {
#ifdef W4G_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

template <int BM, int BN, int BK, int WM, int WN, int MT, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
w4a16_gemm_kernel(const half*    __restrict__ A,
                  const uint8_t* __restrict__ W,
                  const half2*   __restrict__ sz,
                  const half*    __restrict__ bias,
                  half*          __restrict__ C,
                  float*         __restrict__ partial,
                  int M, int N, int K, int gshift, int kt_per_split)
{
    typedef w4g_shape<BM, BN, BK> S;
    constexpr int THREADS = WM * WN * 32;
    constexpr int NT = BN / WN / 8;                  /* n8 tiles per warp  */
    constexpr int KC = BK / 32;                      /* 32-k chunks/stage  */
    static_assert(BM == WM * MT * 16, "tile shape");
    static_assert(NT >= 1 && BN == WN * NT * 8, "tile shape");

    extern __shared__ __align__(16) uint8_t w4g_smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, t = lane & 3;

    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int kt_total = K / BK;
    const int kt_begin = blockIdx.z * kt_per_split;
    int kt_end = kt_begin + kt_per_split;
    if (kt_end > kt_total) kt_end = kt_total;
    const int ngroups = K >> gshift;
    const int half_k = K >> 1;
    /* sz entries one stage spans, and how many go per cp.async */
    const int nsz = (BK >> gshift) > 0 ? (BK >> gshift) : 1;
    const int piece = nsz < 4 ? nsz : 4;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    auto stage_a = [&](int s) -> half* {
        return (half*)(w4g_smem + s * S::STAGE);
    };
    auto stage_w = [&](int s) -> uint8_t* {
        return w4g_smem + s * S::STAGE + S::A_BYTES;
    };
    auto stage_s = [&](int s) -> half2* {
        return (half2*)(w4g_smem + s * S::STAGE + S::A_BYTES + S::W_BYTES);
    };

    auto load_stage = [&](int s, int kt) {
        half* As = stage_a(s);
        uint8_t* Ws = stage_w(s);
        half2* Ss = stage_s(s);
        const int k0 = kt * BK;
        constexpr int A_CH = BK / 8;                 /* 16-byte chunks/row */
#pragma unroll
        for (int i = tid; i < BM * A_CH; i += THREADS) {
            const int r = i / A_CH, c8 = i % A_CH;
            const bool ok = (m0 + r) < M;
            const half* src = ok ? A + (size_t)(m0 + r) * K + k0 + c8 * 8 : A;
            w4g_cp_async(As + r * S::ASTR + c8 * 8, src, 16, ok);
        }
        constexpr int W_CH = BK / 32;
#pragma unroll
        for (int i = tid; i < BN * W_CH; i += THREADS) {
            const int r = i / W_CH, part = i % W_CH;
            const bool ok = (n0 + r) < N;
            const uint8_t* src = ok
                ? W + (size_t)(n0 + r) * half_k + (k0 >> 1) + part * 16 : W;
            w4g_cp_async(Ws + r * S::WSTR + part * 16, src, 16, ok);
        }
        const int per_row = nsz / piece;
        for (int i = tid; i < BN * per_row; i += THREADS) {
            const int r = i / per_row, p = i % per_row;
            const bool ok = (n0 + r) < N;
            const half2* src = ok
                ? sz + (size_t)(n0 + r) * ngroups + (k0 >> gshift) + p * piece
                : sz;
            w4g_cp_async(Ss + r * S::SSTR + p * piece, src, piece * 4, ok);
        }
    };

    /* Prologue */
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (kt_begin + s < kt_end) load_stage(s, kt_begin + s);
        w4g_commit();
    }

    for (int kt = kt_begin; kt < kt_end; kt++) {
        w4g_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < kt_end) load_stage((nk - kt_begin) % STAGES, nk);
            w4g_commit();
        }
        const int s = (kt - kt_begin) % STAGES;
        const half* As = stage_a(s);
        const uint8_t* Ws = stage_w(s);
        const half2* Ss = stage_s(s);

#pragma unroll
        for (int kc = 0; kc < KC; kc++) {
            uint32_t a[MT][2][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    const int r = wm * MT * 16 + mt * 16 + (lane & 15);
                    const int c = kc * 32 + ks * 16 + (lane >> 4) * 8;
                    w4g_ldmatrix_x4(a[mt][ks], As + r * S::ASTR + c);
                }
            const int gi = (kc * 32) >> gshift;
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const int nr = wn * NT * 8 + nt * 8 + g;
                const uint32_t wq =
                    *(const uint32_t*)(Ws + nr * S::WSTR + kc * 16 + t * 4);
                const w4_group grp = w4_make_group(Ss[nr * S::SSTR + gi]);
                half2 b[4];
                w4_dequant_word(wq, grp, b);
                const uint32_t b0 = w4_h2u(b[0]), b1 = w4_h2u(b[1]);
                const uint32_t b2 = w4_h2u(b[2]), b3 = w4_h2u(b[3]);
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    w4g_mma(acc[mt][nt], a[mt][0], b0, b1);
                    w4g_mma(acc[mt][nt], a[mt][1], b2, b3);
                }
            }
        }
    }
    w4g_wait<0>();

    /* Epilogue */
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int col = n0 + wn * NT * 8 + nt * 8 + 2 * t;
            if (col >= N) continue;
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                const int row = m0 + wm * MT * 16 + mt * 16 + g + hr * 8;
                if (row >= M) continue;
                float v0 = acc[mt][nt][2 * hr + 0];
                float v1 = acc[mt][nt][2 * hr + 1];
                if (partial) {
                    float2* p = (float2*)(partial
                        + ((size_t)blockIdx.z * M + row) * N + col);
                    *p = make_float2(v0, v1);
                } else {
                    if (bias) {
                        v0 += __half2float(bias[col]);
                        v1 += __half2float(bias[col + 1]);
                    }
                    *(half2*)(C + (size_t)row * N + col) =
                        __floats2half2_rn(v0, v1);
                }
            }
        }
    }
}

/** @brief Sum split-K partials [S][M][N] into C (+ bias). Two columns/thread. */
__global__ void w4a16_splitk_reduce_kernel(const float* __restrict__ partial,
                                           const half*  __restrict__ bias,
                                           half*        __restrict__ C,
                                           int M, int N, int S)
{
    const size_t i2 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total2 = (size_t)M * N / 2;
    if (i2 >= total2) return;
    const size_t i = i2 * 2;
    const int col = (int)(i % N);
    float v0 = 0.0f, v1 = 0.0f;
    for (int s = 0; s < S; s++) {
        const float2 p = *(const float2*)(partial + (size_t)s * M * N + i);
        v0 += p.x; v1 += p.y;
    }
    if (bias) {
        v0 += __half2float(bias[col]);
        v1 += __half2float(bias[col + 1]);
    }
    *(half2*)(C + i) = __floats2half2_rn(v0, v1);
}

/** @brief GPU layout -> dense FP16 [N][K]; one packed word per thread. */
__global__ void w4a16_dequant_kernel(const uint32_t* __restrict__ w,
                                     const half2*    __restrict__ sz,
                                     half*           __restrict__ out,
                                     int N, int K, int gshift)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)N * (K >> 3);
    if (tid >= total) return;
    const int per_row = K >> 3;
    const int row = (int)(tid / per_row);
    const int wi  = (int)(tid % per_row);
    const int c   = wi >> 2, t = wi & 3;
    const int k0  = c << 5;
    const w4_group g = w4_make_group(sz[(size_t)row * (K >> gshift)
                                        + (k0 >> gshift)]);
    half2 v[4];
    w4_dequant_word(w[tid], g, v);
    half2* orow = (half2*)(out + (size_t)row * K + k0);
#pragma unroll
    for (int j = 0; j < 4; j++) orow[t + 4 * j] = v[j];
}

/* ─── Host side ──────────────────────────────────────────────────────────── */

namespace {

int w4_gshift(int group_size) {
    switch (group_size) {
        case 32:  return 5;
        case 64:  return 6;
        case 128: return 7;
        case 256: return 8;
        default:  return -1;
    }
}

#define W4G_N_CFG 8

/** @brief Device facts, cached per thread and device (a thread binds one). */
struct w4_dev_info {
    int dev;
    int major;
    int sms;
    int blocks_per_sm[W4G_N_CFG];    /* 0 = not yet set up */
};

w4_dev_info* w4_info(void) {
    static thread_local w4_dev_info info = { -1, 0, 0, { 0 } };
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return NULL;
    if (dev != info.dev) {
        info.dev = dev;
        for (int i = 0; i < W4G_N_CFG; i++) info.blocks_per_sm[i] = 0;
        if (cudaDeviceGetAttribute(&info.major,
                cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&info.sms,
                cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            info.dev = -1;
            return NULL;
        }
    }
    return &info;
}

/** @brief Integer environment knob, read once per thread. */
int w4_env_int(const char* name, int* cache, int dflt) {
    if (*cache == -2) {
        const char* e = getenv(name);
        *cache = (e && e[0]) ? atoi(e) : dflt;
    }
    return *cache;
}

/** @brief VV_W4A16_MMA=0 forces the dequant + dense GEMM fallback. */
bool w4_mma_allowed(void) {
    static thread_local int v = -2;
    return w4_env_int("VV_W4A16_MMA", &v, 1) != 0;
}

template <int LPR, int R>
vv_status_t launch_gemv_lpr(const half* x, w4v_segs segs, int n_segs,
                            int K, int gshift, cudaStream_t st)
{
    constexpr int RPB = W4V_THREADS / LPR * R;
    int blocks = 0;
    for (int i = 0; i < W4V_MAX_SEGS; i++) {
        if (i < n_segs) {
            segs.first_block[i] = blocks;
            blocks += (segs.n[i] + RPB - 1) / RPB;
        } else {
            segs.first_block[i] = INT_MAX;
        }
    }
    w4a16_gemv_kernel<LPR, R><<<blocks, W4V_THREADS, 0, st>>>(
        x, segs, K, gshift);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

template <int R>
vv_status_t launch_gemv_r(const half* x, const w4v_segs& segs, int n_segs,
                          int K, int gshift, int lpr, cudaStream_t st)
{
    switch (lpr) {
        case 16:  return launch_gemv_lpr<16,  R>(x, segs, n_segs, K, gshift, st);
        case 32:  return launch_gemv_lpr<32,  R>(x, segs, n_segs, K, gshift, st);
        case 64:  return launch_gemv_lpr<64,  R>(x, segs, n_segs, K, gshift, st);
        default:  return launch_gemv_lpr<128, R>(x, segs, n_segs, K, gshift, st);
    }
}

/**
 * @brief One GEMV launch over `n_segs` projections of the same x.
 *
 * At least 16 lanes per row, so every step reads 256 contiguous bytes of it
 * (8 lanes, 128 bytes, measured 10-15% slower); more while the grid is short
 * of ~48K threads, so a lone 512- or 256-row projection still reaches every
 * SM; never more than twice the row's chunk count. (Capping at one chunk
 * per lane, so none idles, was measured slower on the lone 512-row k/v:
 * 3.8 us at 64 lanes against 3.4 us at 128.)
 * One chunk in flight per lane: at 72 registers the occupancy that buys
 * hides the latency better than unrolling does (swept on a 3090, one chunk
 * beat two and four on every shape). Long rows (the down projection) go two
 * to a lane group, so each x chunk fetched serves two weight chunks:
 * 3584x18944 moves 750 -> 805 GB/s that way, while the short-K shapes lose
 * from the halved grid. VV_W4A16_GEMV_LPR (16..128) and VV_W4A16_GEMV_R
 * (1, 2) override the choice for tuning.
 */
/**
 * @brief Rows per lane group and lanes per row of a one-token launch over
 *        these projections. The multi-row kernel takes the same `lpr`, so a
 *        row of it sums exactly as the one-token GEMV does.
 */
void gemv_shape(const w4v_segs& segs, int n_segs, int K, int* rows_out,
                int* lpr_out)
{
    long rows_total = 0;
    for (int i = 0; i < n_segs; i++) rows_total += segs.n[i];
    const int nch = K >> 5;
    static thread_local int lpr_env = -2, r_env = -2;
    const int r_req = w4_env_int("VV_W4A16_GEMV_R", &r_env, 0);
    const int rows = r_req > 0 ? (r_req >= 2 ? 2 : 1) : (K >= 8192 ? 2 : 1);
    int lpr = w4_env_int("VV_W4A16_GEMV_LPR", &lpr_env, 0);
    if (lpr <= 0) {
        lpr = 16;
        while (lpr < 128 && rows_total / rows * lpr < 49152L) lpr *= 2;
        while (lpr > 16 && lpr > 2 * nch) lpr >>= 1;
    }
    *rows_out = rows;
    *lpr_out = lpr;
}

vv_status_t launch_gemv(const void* x, const w4v_segs& segs, int n_segs,
                        int K, int gshift, cudaStream_t st)
{
    int rows = 1, lpr = 16;
    gemv_shape(segs, n_segs, K, &rows, &lpr);
    const half* xh = (const half*)x;
    if (rows == 2) return launch_gemv_r<2>(xh, segs, n_segs, K, gshift, lpr, st);
    return launch_gemv_r<1>(xh, segs, n_segs, K, gshift, lpr, st);
}

template <int LPR, int R>
vv_status_t launch_gemv_rows_lpr(const half* x, int M, w4v_segs segs,
                                 int n_segs, int K, int gshift,
                                 cudaStream_t st)
{
    constexpr int RPB = W4R_THREADS / LPR * R;
    /* Tiles of whole LPR-chunk strides: a lane's chunks stay in order. */
    constexpr int TILE_MAX = W4R_TILE_MAX > LPR ? W4R_TILE_MAX / LPR * LPR : LPR;
    constexpr int SMEM_MAX = W4R_MX * 4 * TILE_MAX * (int)sizeof(uint4);
    const int nch = K >> 5;
    const int tile = TILE_MAX < nch ? TILE_MAX : nch;
    const int SMEM = W4R_MX * 4 * tile * (int)sizeof(uint4);
    int blocks = 0;
    for (int i = 0; i < W4V_MAX_SEGS; i++) {
        if (i < n_segs) {
            segs.first_block[i] = blocks;
            blocks += (segs.n[i] + RPB - 1) / RPB;
        } else {
            segs.first_block[i] = INT_MAX;
        }
    }
    auto kern = w4a16_gemv_rows_kernel<LPR, R>;
    if (SMEM > 48 * 1024) {
        static thread_local int set_dev = -1;
        int dev = 0;
        cudaGetDevice(&dev);
        if (set_dev != dev) {
            if (cudaFuncSetAttribute(kern,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX)
                    != cudaSuccess)
                return VV_ERR_CUDA_LAUNCH;
            set_dev = dev;
        }
    }
    kern<<<blocks, W4R_THREADS, SMEM, st>>>(x, M, segs, K, gshift, tile);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

template <int R>
vv_status_t launch_gemv_rows_r(const half* x, int M, const w4v_segs& segs,
                               int n_segs, int K, int gshift, int lpr,
                               cudaStream_t st)
{
    switch (lpr) {
        case 16:  return launch_gemv_rows_lpr<16,  R>(x, M, segs, n_segs, K, gshift, st);
        case 32:  return launch_gemv_rows_lpr<32,  R>(x, M, segs, n_segs, K, gshift, st);
        case 64:  return launch_gemv_rows_lpr<64,  R>(x, M, segs, n_segs, K, gshift, st);
        default:  return launch_gemv_rows_lpr<128, R>(x, M, segs, n_segs, K, gshift, st);
    }
}

/*
 * Weight rows per lane group: as many as keep a wave of blocks on the card,
 * up to 4. More rows per block is less x traffic per weight byte; the bits
 * do not depend on it.
 */
int gemv_rows_r(long rows_total, int lpr, int sms) {
    const int gpb = W4R_THREADS / lpr;
    for (int r = 4; r > 1; r >>= 1)
        if (rows_total / ((long)gpb * r) >= (long)sms) return r;
    return 1;
}

vv_status_t gemv_one(const void* x, const void* packed, const void* sz,
                     const void* bias, void* y, int N, int K, int gshift,
                     cudaStream_t st)
{
    w4v_segs segs;
    memset(&segs, 0, sizeof(segs));
    segs.w[0] = (const uint4*)packed;
    segs.sz[0] = (const half2*)sz;
    segs.bias[0] = (const half*)bias;
    segs.y[0] = (half*)y;
    segs.n[0] = N;
    return launch_gemv(x, segs, 1, K, gshift, st);
}

/**
 * @brief Split-K factor for one launch.
 *
 * Up to M ~ 64 the layer is bound by how fast the weights stream, so the
 * grid has to cover every SM: K is split until it does (a 16-row prefill
 * against a 3584-wide projection is 28 tiles on 82 SMs otherwise). Above
 * that it is compute bound and the split only has to fix wave
 * quantization: 84 tiles on 82 one-block SMs is two waves for 1.02 waves
 * of work, and splitting that K four ways makes it 1.25.
 */
int w4_pick_splits(int tiles, int slots, int sms, int kt_total, int M, int N,
                   size_t scratch_bytes)
{
    const size_t per_split = (size_t)M * N * sizeof(float);
    int best = 1;
    if (M <= 64) {
        int s = 1;
        while (tiles * s < 2 * sms && kt_total / (s * 2) >= 4 &&
               (size_t)(s * 2) * per_split <= scratch_bytes)
            s *= 2;
        return s;
    }
    double best_cost = 1e30;
    for (int s = 1; s <= 8; s++) {
        if (s > 1 && (kt_total / s < 4 || (size_t)s * per_split > scratch_bytes))
            break;
        const int waves = (tiles * s + slots - 1) / slots;
        const double cost = (double)waves / s + 0.05 * (s - 1);
        if (cost < best_cost - 1e-9) { best_cost = cost; best = s; }
    }
    return best;
}

template <int BM, int BN, int BK, int WM, int WN, int MT, int STAGES>
vv_status_t launch_gemm_cfg(w4_dev_info* info, int cfg,
                            const void* A, const void* W, const void* sz,
                            const void* bias, void* C, void* scratch,
                            size_t scratch_bytes, int M, int N, int K,
                            int gshift, cudaStream_t st)
{
    constexpr int SMEM = w4g_shape<BM, BN, BK>::STAGE * STAGES;
    constexpr int THREADS = WM * WN * 32;
    if ((K % BK) != 0) return VV_ERR_UNSUPPORTED;
    auto kern = w4a16_gemm_kernel<BM, BN, BK, WM, WN, MT, STAGES>;
    if (info->blocks_per_sm[cfg] == 0) {
        int nb = 0;
        if (cudaFuncSetAttribute(kern,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)
                != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, THREADS,
                                                          SMEM) != cudaSuccess)
            return VV_ERR_CUDA_LAUNCH;
        info->blocks_per_sm[cfg] = nb > 0 ? nb : 1;
    }

    const int tiles_n = (N + BN - 1) / BN;
    const int tiles_m = (M + BM - 1) / BM;
    const int kt_total = K / BK;
    int splits = w4_pick_splits(tiles_n * tiles_m,
                                info->blocks_per_sm[cfg] * info->sms,
                                info->sms, kt_total, M, N,
                                scratch ? scratch_bytes : 0);
    const int kt_per = (kt_total + splits - 1) / splits;
    splits = (kt_total + kt_per - 1) / kt_per;

    float* partial = (splits > 1) ? (float*)scratch : NULL;
    dim3 grid(tiles_n, tiles_m, splits);
    kern<<<grid, THREADS, SMEM, st>>>(
        (const half*)A, (const uint8_t*)W, (const half2*)sz,
        (const half*)bias, (half*)C, partial, M, N, K, gshift, kt_per);
    if (cudaGetLastError() != cudaSuccess) return VV_ERR_CUDA_LAUNCH;

    if (splits > 1) {
        const size_t total2 = (size_t)M * N / 2;
        const int threads = 256;
        const unsigned blocks = (unsigned)((total2 + threads - 1) / threads);
        w4a16_splitk_reduce_kernel<<<blocks, threads, 0, st>>>(
            partial, (const half*)bias, (half*)C, M, N, splits);
        if (cudaGetLastError() != cudaSuccess) return VV_ERR_CUDA_LAUNCH;
    }
    return VV_OK;
}

/** @brief Pick a tile configuration for M (VV_W4A16_CFG overrides). */
int w4_pick_cfg(int M) {
    static thread_local int v = -2;
    const int forced = w4_env_int("VV_W4A16_CFG", &v, -1);
    if (forced >= 0 && forced < W4G_N_CFG) return forced;
    /*
     * Swept on a 3090 over the 7B projections: 16x64x128 is best up to 16
     * rows (the narrow N tile keeps 2-4x more blocks in flight while the
     * layer is weight-bound), 32x128x128 at 17-32, 64x128x128 from there on
     * and 128x128x64 only once the grid is several waves deep.
     *
     * The square 3584x3584 q/o projection would be ~5% faster at 1024-2048
     * rows on 64x128x128 (918 against 970 us at 2048); that is ~0.5% of a
     * prefill, and its different split-K changes the FP32 summation order
     * enough to move 126 of the 168 segment boundaries of the 32-minute
     * test file by up to 50 ms (text and speakers unchanged), so it is not
     * taken.
     */
    if (M <= 16) return 2;
    if (M <= 32) return 3;
    if (M <= 512) return 6;
    return 7;
}

vv_status_t launch_gemm(w4_dev_info* info, int cfg, const void* A,
                        const void* W, const void* sz, const void* bias,
                        void* C, void* scratch, size_t scratch_bytes,
                        int M, int N, int K, int gs, cudaStream_t st)
{
#define W4G_CASE(i, BM, BN, BK, WM, WN, MT, ST)                              \
    case i: return launch_gemm_cfg<BM, BN, BK, WM, WN, MT, ST>(             \
                info, i, A, W, sz, bias, C, scratch, scratch_bytes,         \
                M, N, K, gs, st);
    switch (cfg) {
        W4G_CASE(0,  16, 128, 128, 1, 4, 1, 4)
        W4G_CASE(1,  16,  64, 256, 1, 4, 1, 3)
        W4G_CASE(2,  16,  64, 128, 1, 4, 1, 4)
        W4G_CASE(3,  32, 128, 128, 1, 4, 2, 3)
        W4G_CASE(4,  32,  64, 128, 1, 4, 2, 4)
        W4G_CASE(5,  64, 128,  64, 2, 4, 2, 4)
        W4G_CASE(6,  64, 128, 128, 2, 4, 2, 3)
        default:
        W4G_CASE(7, 128, 128,  64, 2, 4, 4, 3)
    }
#undef W4G_CASE
}

} /* namespace */

extern "C" {

vv_status_t vv_w4a16_dequant_dev(const void* packed, const void* sz,
                                 void* out_fp16, int N, int K, int group_size,
                                 void* stream)
{
    if (!packed || !sz || !out_fp16) return VV_ERR_NULL_PTR;
    const int gs = w4_gshift(group_size);
    if (gs < 0 || (K & 31) != 0) return VV_ERR_UNSUPPORTED;
    const size_t total = (size_t)N * (K >> 3);
    const int threads = 256;
    w4a16_dequant_kernel<<<(unsigned)((total + threads - 1) / threads),
                           threads, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)packed, (const half2*)sz, (half*)out_fp16, N, K, gs);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_w4a16_gemv_dev(const void* x, const void* packed,
                              const void* sz, const void* bias, void* y,
                              int N, int K, int group_size, void* stream)
{
    if (!x || !packed || !sz || !y) return VV_ERR_NULL_PTR;
    const int gs = w4_gshift(group_size);
    if (gs < 0 || (K & 31) != 0 || N <= 0) return VV_ERR_UNSUPPORTED;
    return gemv_one(x, packed, sz, bias, y, N, K, gs, (cudaStream_t)stream);
}

vv_status_t vv_w4a16_gemv_multi_dev(const void* x,
                                    const vv_w4a16_proj_t* projs, int n_proj,
                                    int K, int group_size, void* stream)
{
    if (!x || !projs) return VV_ERR_NULL_PTR;
    if (n_proj < 1 || n_proj > W4V_MAX_SEGS) return VV_ERR_INVALID_ARG;
    const int gs = w4_gshift(group_size);
    if (gs < 0 || (K & 31) != 0) return VV_ERR_UNSUPPORTED;
    w4v_segs segs;
    memset(&segs, 0, sizeof(segs));
    for (int i = 0; i < n_proj; i++) {
        if (!projs[i].packed || !projs[i].sz || !projs[i].y)
            return VV_ERR_NULL_PTR;
        if (projs[i].N <= 0) return VV_ERR_UNSUPPORTED;
        segs.w[i] = (const uint4*)projs[i].packed;
        segs.sz[i] = (const half2*)projs[i].sz;
        segs.bias[i] = (const half*)projs[i].bias;
        segs.y[i] = (half*)projs[i].y;
        segs.n[i] = projs[i].N;
    }
    return launch_gemv(x, segs, n_proj, K, gs, (cudaStream_t)stream);
}

vv_status_t vv_w4a16_gemv_rows_dev(const void* x, int M,
                                   const vv_w4a16_proj_t* projs, int n_proj,
                                   int K, int group_size, void* stream)
{
    if (!x || !projs) return VV_ERR_NULL_PTR;
    if (n_proj < 1 || n_proj > W4V_MAX_SEGS || M < 1 || M > 16)
        return VV_ERR_INVALID_ARG;
    const int gs = w4_gshift(group_size);
    if (gs < 0 || (K & 31) != 0) return VV_ERR_UNSUPPORTED;
    w4v_segs segs;
    memset(&segs, 0, sizeof(segs));
    for (int i = 0; i < n_proj; i++) {
        if (!projs[i].packed || !projs[i].sz || !projs[i].y)
            return VV_ERR_NULL_PTR;
        if (projs[i].N <= 0) return VV_ERR_UNSUPPORTED;
        segs.w[i] = (const uint4*)projs[i].packed;
        segs.sz[i] = (const half2*)projs[i].sz;
        segs.bias[i] = (const half*)projs[i].bias;
        segs.y[i] = (half*)projs[i].y;
        segs.n[i] = projs[i].N;
    }
    int rows = 1, lpr = 16;
    gemv_shape(segs, n_proj, K, &rows, &lpr);
    w4_dev_info* info = w4_info();
    if (!info) return VV_ERR_CUDA_LAUNCH;
    long rows_total = 0;
    for (int i = 0; i < n_proj; i++) rows_total += segs.n[i];
    static thread_local int r_env = -2;
    const int r_req = w4_env_int("VV_W4A16_ROWS_R", &r_env, 0);
    const int R = r_req == 1 || r_req == 2 || r_req == 4
                ? r_req : gemv_rows_r(rows_total, lpr, info->sms);
    cudaStream_t st = (cudaStream_t)stream;
    /* Eight rows a launch: a 16-row block reads the weights twice. */
    for (int m0 = 0; m0 < M; m0 += W4R_MX) {
        const int mm = M - m0 < W4R_MX ? M - m0 : W4R_MX;
        w4v_segs sg = segs;
        for (int i = 0; i < n_proj; i++)
            sg.y[i] = segs.y[i] + (size_t)m0 * segs.n[i];
        const half* xh = (const half*)x + (size_t)m0 * K;
        vv_status_t s;
        if (R == 4)      s = launch_gemv_rows_r<4>(xh, mm, sg, n_proj, K, gs, lpr, st);
        else if (R == 2) s = launch_gemv_rows_r<2>(xh, mm, sg, n_proj, K, gs, lpr, st);
        else             s = launch_gemv_rows_r<1>(xh, mm, sg, n_proj, K, gs, lpr, st);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

vv_status_t vv_w4a16_gather_dev(const void* x, const int32_t* perm, void* out,
                                int M, int K, void* stream)
{
    if (!x || !perm || !out) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || (K & 7) != 0) return VV_ERR_INVALID_ARG;
    const size_t total8 = (size_t)M * K / 8;
    const int threads = 256;
    w4a16_gather_kernel<<<(unsigned)((total8 + threads - 1) / threads),
                          threads, 0, (cudaStream_t)stream>>>(
        (const half*)x, perm, (half*)out, M, K);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_w4a16_gemm_dev(const void* A, const void* packed,
                              const void* sz, const void* bias, void* C,
                              void* scratch, size_t scratch_bytes,
                              int M, int N, int K, int group_size,
                              void* stream)
{
    if (!A || !packed || !sz || !C) return VV_ERR_NULL_PTR;
    const int gs = w4_gshift(group_size);
    if (gs < 0 || (K & 31) != 0 || M <= 0 || N <= 0 || (N & 7) != 0)
        return VV_ERR_UNSUPPORTED;
    cudaStream_t st = (cudaStream_t)stream;

    if (M == 1) return gemv_one(A, packed, sz, bias, C, N, K, gs, st);

    w4_dev_info* info = w4_info();
    if (!info) return VV_ERR_CUDA_LAUNCH;

    if (info->major >= 8 && w4_mma_allowed()) {
        vv_status_t s = launch_gemm(info, w4_pick_cfg(M), A, packed, sz, bias,
                                    C, scratch, scratch_bytes, M, N, K, gs,
                                    st);
        if (s != VV_ERR_UNSUPPORTED) return s;
        /* K not a multiple of that tile's BK: the 64-deep tile takes it */
        s = launch_gemm(info, 5, A, packed, sz, bias, C, scratch,
                        scratch_bytes, M, N, K, gs, st);
        if (s != VV_ERR_UNSUPPORTED) return s;
    }

    /* Turing, or forced off: expand into the scratch, then dense GEMM. */
    if (!scratch) return VV_ERR_NULL_PTR;
    if (scratch_bytes < (size_t)N * K * 2) return VV_ERR_OVERFLOW;
    vv_status_t s = vv_w4a16_dequant_dev(packed, sz, scratch, N, K,
                                         group_size, stream);
    if (s != VV_OK) return s;
    s = vv_gemm_fp16_dev(A, scratch, C, M, N, K, 1.0f, 0.0f, stream);
    if (s != VV_OK || !bias) return s;
    return vv_bias_add_dev(C, bias, M, N, stream);
}

} /* extern "C" */
