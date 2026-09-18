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
 * GEMV (and tiny-M GEMM): y[M,N] = x[M,K] . W^T + bias
 *
 * LPR lanes share R output rows (LPR 8..128; above 32 the rows span several
 * warps and the partial sums meet in shared memory). Each lane walks the
 * rows' 16-byte chunks with a stride of LPR and keeps U chunk columns in
 * flight; the x values of a chunk are loaded once and used for all R rows.
 * Per chunk, row and query row the 32 products are formed as four HFMA2
 * chains of four and flushed to FP32, so no FP16 partial ever sums more
 * than four terms.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define W4V_THREADS 128

template <int LPR, int M, int U, int R>
__global__ void __launch_bounds__(W4V_THREADS)
w4a16_gemv_kernel(const half*  __restrict__ x,
                  const uint4* __restrict__ w,
                  const half2* __restrict__ sz,
                  const half*  __restrict__ bias,
                  half*        __restrict__ y,
                  int N, int K, int gshift)
{
    constexpr int GPB = W4V_THREADS / LPR;            /* lane groups/block */
    constexpr int WIDTH = LPR < 32 ? LPR : 32;
    const int row0 = (blockIdx.x * GPB + (int)threadIdx.x / LPR) * R;
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

    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = 0.0f;

    for (int c0 = lr; c0 < nch; c0 += LPR * U) {
        uint4 wv[U][R];
        half2 sv[U][R];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int c = c0 + u * LPR;
            if (c < nch) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    wv[u][r] = w4_ld_stream(wrow[r] + c);
                    sv[u][r] = szrow[r][(c << 5) >> gshift];
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int c = c0 + u * LPR;
            if (c >= nch) break;
#pragma unroll
            for (int m = 0; m < M; m++) {
                const uint4* xp = (const uint4*)(x + (size_t)m * K + (c << 5));
                uint4 xv[4];
#pragma unroll
                for (int i = 0; i < 4; i++) xv[i] = __ldg(xp + i);
                const half2* xh = (const half2*)xv;     /* pairs 0..15 */
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const w4_group g = w4_make_group(sv[u][r]);
                    const uint32_t wd[4] = { wv[u][r].x, wv[u][r].y,
                                             wv[u][r].z, wv[u][r].w };
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
                        acc[r][m] += f.x + f.y;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
#pragma unroll
            for (int off = WIDTH / 2; off > 0; off >>= 1)
                acc[r][m] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][m], off, WIDTH);
        }
    if (LPR > 32) {
        constexpr int WPG = LPR / 32;                  /* warps per group */
        __shared__ float red[R * M][W4V_THREADS / 32];
        const int warp = (int)threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int m = 0; m < M; m++) red[r * M + m][warp] = acc[r][m];
        }
        __syncthreads();
        if (lr == 0) {
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int m = 0; m < M; m++) {
                    float v = 0.0f;
#pragma unroll
                    for (int i = 0; i < WPG; i++) v += red[r * M + m][warp + i];
                    acc[r][m] = v;
                }
        }
    }
    if (lr == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= N) break;
            const float b = bias ? __half2float(bias[row]) : 0.0f;
#pragma unroll
            for (int m = 0; m < M; m++)
                y[(size_t)m * N + row] = __float2half(acc[r][m] + b);
        }
    }
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

/** @brief Largest M the GEMV kernel takes before the tensor-core GEMM. */
int w4_gemv_max_m(void) {
    static thread_local int v = -2;
    int m = w4_env_int("VV_W4A16_GEMV_MAX_M", &v, 1);
    return m < 1 ? 1 : (m > 4 ? 4 : m);
}

template <int LPR, int M, int U, int R>
vv_status_t launch_gemv_lpr(const half* x, const uint4* w, const half2* sz,
                            const half* bias, half* y, int N, int K,
                            int gshift, cudaStream_t st)
{
    constexpr int RPB = W4V_THREADS / LPR * R;
    const int blocks = (N + RPB - 1) / RPB;
    w4a16_gemv_kernel<LPR, M, U, R><<<blocks, W4V_THREADS, 0, st>>>(
        x, w, sz, bias, y, N, K, gshift);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

template <int M, int U, int R>
vv_status_t launch_gemv_u(const half* x, const uint4* w, const half2* sz,
                          const half* bias, half* y, int N, int K,
                          int gshift, int lpr, cudaStream_t st)
{
    switch (lpr) {
        case 8:   return launch_gemv_lpr<8,   M, U, R>(x, w, sz, bias, y, N, K, gshift, st);
        case 16:  return launch_gemv_lpr<16,  M, U, R>(x, w, sz, bias, y, N, K, gshift, st);
        case 32:  return launch_gemv_lpr<32,  M, U, R>(x, w, sz, bias, y, N, K, gshift, st);
        case 64:  return launch_gemv_lpr<64,  M, U, R>(x, w, sz, bias, y, N, K, gshift, st);
        default:  return launch_gemv_lpr<128, M, U, R>(x, w, sz, bias, y, N, K, gshift, st);
    }
}

template <int M>
vv_status_t launch_gemv(const void* x, const void* packed, const void* sz,
                        const void* bias, void* y, int N, int K, int gshift,
                        cudaStream_t st)
{
    const int nch = K >> 5;
    /*
     * At least 16 lanes per row, so every step reads 256 contiguous bytes of
     * it (8 lanes, 128 bytes, measured 10-15% slower); more while the grid is
     * short of ~48K threads, so the 512- and 256-row k/v projections still
     * reach every SM; never more than twice the row's chunk count. One
     * chunk in flight per lane: at 72 registers the occupancy that buys
     * hides the latency better than unrolling does (swept on a 3090, U=1
     * beat U=2/4 on every shape). Long rows (the down projection) go two
     * to a lane group, so each x chunk fetched serves two weight chunks:
     * 3584x18944 moves 750 -> 805 GB/s that way, while the short-K shapes
     * lose from the halved grid.
     */
    static thread_local int lpr_env = -2, u_env = -2, r_env = -2;
    const int r_req = w4_env_int("VV_W4A16_GEMV_R", &r_env, 0);
    const int rows = r_req > 0 ? (r_req >= 2 ? 2 : 1) : (K >= 8192 ? 2 : 1);
    int lpr = w4_env_int("VV_W4A16_GEMV_LPR", &lpr_env, 0);
    if (lpr <= 0) {
        lpr = 16;
        while (lpr < 128 && (long)N / rows * lpr < 49152L) lpr *= 2;
        while (lpr > 16 && lpr > 2 * nch) lpr >>= 1;
    }
    const int u = w4_env_int("VV_W4A16_GEMV_U", &u_env, 1);
    const half* xh = (const half*)x;
    const uint4* w = (const uint4*)packed;
    const half2* s = (const half2*)sz;
    const half* b = (const half*)bias;
    half* yh = (half*)y;
    if (M == 1 && rows == 2) {
        if (u <= 1) return launch_gemv_u<M, 1, 2>(xh, w, s, b, yh, N, K, gshift, lpr, st);
        return launch_gemv_u<M, 2, 2>(xh, w, s, b, yh, N, K, gshift, lpr, st);
    }
    if (u <= 1 || M > 1) return launch_gemv_u<M, 1, 1>(xh, w, s, b, yh, N, K, gshift, lpr, st);
    return launch_gemv_u<M, 2, 1>(xh, w, s, b, yh, N, K, gshift, lpr, st);
}

vv_status_t gemv_any_m(const void* x, const void* packed, const void* sz,
                       const void* bias, void* y, int M, int N, int K,
                       int gshift, cudaStream_t st)
{
    switch (M) {
        case 1: return launch_gemv<1>(x, packed, sz, bias, y, N, K, gshift, st);
        case 2: return launch_gemv<2>(x, packed, sz, bias, y, N, K, gshift, st);
        case 3: return launch_gemv<3>(x, packed, sz, bias, y, N, K, gshift, st);
        case 4: return launch_gemv<4>(x, packed, sz, bias, y, N, K, gshift, st);
        default: return VV_ERR_UNSUPPORTED;
    }
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
    return launch_gemv<1>(x, packed, sz, bias, y, N, K, gs,
                          (cudaStream_t)stream);
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

    if (M <= w4_gemv_max_m())
        return gemv_any_m(A, packed, sz, bias, C, M, N, K, gs, st);

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
    if (!scratch || scratch_bytes < (size_t)N * K * 2) return VV_ERR_NULL_PTR;
    vv_status_t s = vv_w4a16_dequant_dev(packed, sz, scratch, N, K,
                                         group_size, stream);
    if (s != VV_OK) return s;
    s = vv_gemm_fp16_dev(A, scratch, C, M, N, K, 1.0f, 0.0f, stream);
    if (s != VV_OK || !bias) return s;
    return vv_bias_add_dev(C, bias, M, N, stream);
}

} /* extern "C" */
