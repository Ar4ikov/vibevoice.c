/**
 * @file gemm_i8.cu
 * @brief Int8-activation GEMM for prefill: W8A8 and W4A8 on tensor cores.
 *
 * C[M,N] = A[M,K] . B[N,K]^T with A the per-token int8 activations and B the
 * weights, both K-contiguous — exactly the row.col operand order of
 * `mma.sync.m16n8k32.s32.s8.s8.s32`, so neither side is transposed.
 *
 * Tensor-core path (sm_80+):
 *   - a CTA computes BM x BN, eight warps in a WARPS_M x WARPS_N grid;
 *   - K advances 64 bytes per stage through a STAGES-deep cp.async ring, so
 *     global loads for stage k+STAGES-1 are in flight while stage k computes;
 *   - A (and the W8 B) rows are 64 bytes in shared memory with the 16-byte
 *     chunk index XORed by (row >> 1) & 3, which makes every ldmatrix phase
 *     hit eight distinct bank groups;
 *   - W8A8 loads B with ldmatrix.x4 (two n8 fragments per instruction);
 *   - W4A8 keeps B packed in shared memory (rows padded to 48 bytes, which is
 *     conflict-free for the 16-bit reads below) and expands each fragment in
 *     registers: two nibble masks and a byte permute give four k-ordered
 *     codes, and one __vsub4 subtracts the replicated zero point. The result
 *     is in [-15, 15], so the MMA accumulates the exact integer
 *     sum (q - z) x of the group; at every group boundary it is scaled by
 *     s[n, g] into an FP32 accumulator.
 *   - epilogue: acc * sx[m] (* sw[n]) + bias[n] (+ residual), FP16 out.
 *
 * No FP16 copy of the weight is ever made: the old path dequantized every
 * projection into a 136 MB scratch and read it back, which is most of what a
 * mid-sized prefill chunk used to cost.
 *
 * SIMT path: 64x64 tiles of dp4a for sm_61..sm_75, or any shape the MMA
 * kernel does not take. Same arithmetic, same results.
 */

#include "i8_common.cuh"
#include "vibevoice/device.h"

#include <stdlib.h>

extern "C" vv_status_t vv_i8_gemv_launch(const vv_i8_args* ap, int w4,
                                         void* stream);

/* ═══════════════════════════════════════════════════════════════════════════
 * Tensor-core kernel
 * ═══════════════════════════════════════════════════════════════════════════ */

#define I8_BK 64                  /* bytes of K per stage (two k32 MMA steps) */
#define I8_W4_LDB 48              /* padded row of a packed W4 B tile          */

__device__ __forceinline__ void i8_cp_async16(void* smem, const void* gmem,
                                              bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(n));
#else
    (void)smem; (void)gmem; (void)valid;
#endif
}

__device__ __forceinline__ void i8_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void i8_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void i8_ldsm_x4(uint32_t r[4], const void* smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
    (void)smem; r[0] = r[1] = r[2] = r[3] = 0u;
#endif
}

/** @brief d += a[16x32] . b[32x8], s8 in, s32 accumulate. */
__device__ __forceinline__ void i8_mma(int d[4], const uint32_t a[4],
                                       uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

/** Byte offset of 16-byte chunk `c` of row `r` in a swizzled 64-byte-row tile. */
__device__ __forceinline__ int i8_swz(int r, int c) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, bool W4>
struct i8_mma_cfg {
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WM = BM / WARPS_M;
    static constexpr int WN = BN / WARPS_N;
    static constexpr int MI = WM / 16;
    static constexpr int NI = WN / 8;
    static constexpr int A_BYTES = BM * I8_BK;
    static constexpr int B_BYTES = W4 ? BN * I8_W4_LDB : BN * I8_BK;
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int SMEM = STAGES * STAGE_BYTES;
};

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, bool W4>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
i8_mma_kernel(const vv_i8_args p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    typedef i8_mma_cfg<BM, BN, WARPS_M, WARPS_N, STAGES, W4> C;
    constexpr int MI = C::MI, NI = C::NI;
    static_assert(W4 || (NI % 2) == 0, "ldmatrix.x4 loads n8 fragments in pairs");

    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * C::WM;
    const int wn0 = (warp % WARPS_N) * C::WN;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int K = p.K, M = p.M, N = p.N;
    const int ktiles = K / I8_BK;
    const size_t ldb_g = W4 ? (size_t)(K >> 1) : (size_t)K;
    const uint8_t* wbytes = (const uint8_t*)p.w;

    auto load_stage = [&](int stage, int kt) {
        unsigned char* sA = smem + stage * C::STAGE_BYTES;
        unsigned char* sB = sA + C::A_BYTES;
        const int k0 = kt * I8_BK;
        for (int ch = tid; ch < BM * 4; ch += C::THREADS) {
            const int r = ch >> 2, c = ch & 3;
            const int gm = m0 + r;
            const bool ok = gm < M;
            const int8_t* src = p.xq + (size_t)(ok ? gm : 0) * K + k0 + c * 16;
            i8_cp_async16(sA + i8_swz(r, c), src, ok);
        }
        if (W4) {
            for (int ch = tid; ch < BN * 2; ch += C::THREADS) {
                const int r = ch >> 1, c = ch & 1;
                const int gn = n0 + r;
                const bool ok = gn < N;
                const uint8_t* src = wbytes + (size_t)(ok ? gn : 0) * ldb_g +
                                     (k0 >> 1) + c * 16;
                i8_cp_async16(sB + r * I8_W4_LDB + c * 16, src, ok);
            }
        } else {
            for (int ch = tid; ch < BN * 4; ch += C::THREADS) {
                const int r = ch >> 2, c = ch & 3;
                const int gn = n0 + r;
                const bool ok = gn < N;
                const uint8_t* src = wbytes + (size_t)(ok ? gn : 0) * ldb_g +
                                     k0 + c * 16;
                i8_cp_async16(sB + i8_swz(r, c), src, ok);
            }
        }
    };

    int acc[MI][NI][4];
    float facc[W4 ? MI : 1][W4 ? NI : 1][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0;
    if (W4) {
#pragma unroll
        for (int i = 0; i < (W4 ? MI : 1); i++)
#pragma unroll
            for (int j = 0; j < (W4 ? NI : 1); j++)
#pragma unroll
                for (int c = 0; c < 4; c++) facc[i][j][c] = 0.0f;
    }

    /*
     * W4: zero points of this thread's B rows for the current group, and its
     * accumulator columns' scales. Both are read one group ahead of use (the
     * zeros) or a whole group before (the scales), so the global loads are
     * off the MMA's critical path.
     */
    uint32_t zrep[W4 ? NI : 1], znext[W4 ? NI : 1];
    float sc[W4 ? NI : 1][2];
    const int ng = W4 ? K / p.G : 1;
    const int steps_per_group = W4 ? p.G / 32 : 1;
    if (W4) {
#pragma unroll
        for (int j = 0; j < (W4 ? NI : 1); j++) {
            const int gn = n0 + wn0 + j * 8 + (lane >> 2);
            znext[j] = (gn < N) ? (uint32_t)__ldg(p.zeros + (size_t)gn * ng) : 0u;
        }
    }

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktiles) load_stage(s, s);
        i8_cp_commit();
    }

    for (int kt = 0; kt < ktiles; kt++) {
        i8_cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < ktiles) load_stage(nk % STAGES, nk);
            i8_cp_commit();
        }

        const unsigned char* sA = smem + (kt % STAGES) * C::STAGE_BYTES;
        const unsigned char* sB = sA + C::A_BYTES;

#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int ks = kt * 2 + kk;              /* global k32 step */

            if (W4 && (ks % steps_per_group) == 0) {
                const int g = ks / steps_per_group;
#pragma unroll
                for (int j = 0; j < (W4 ? NI : 1); j++) {
                    zrep[j] = znext[j] * 0x01010101u;
                    const int gn = n0 + wn0 + j * 8 + (lane >> 2);
                    znext[j] = (gn < N && g + 1 < ng)
                        ? (uint32_t)__ldg(p.zeros + (size_t)gn * ng + g + 1) : 0u;
                    const int cn = n0 + wn0 + j * 8 + (lane & 3) * 2;
                    sc[j][0] = (cn < N) ? __half2float(__ldg(p.scales + (size_t)cn * ng + g)) : 0.0f;
                    sc[j][1] = (cn + 1 < N) ? __half2float(__ldg(p.scales + (size_t)(cn + 1) * ng + g)) : 0.0f;
                }
            }

            uint32_t af[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const int r = wm0 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int c = kk * 2 + (lane >> 4);
                i8_ldsm_x4(af[i], sA + i8_swz(r, c));
            }

            uint32_t bf[NI][2];
            if (W4) {
#pragma unroll
                for (int j = 0; j < NI; j++) {
                    const int r = wn0 + j * 8 + (lane >> 2);
                    const unsigned char* rowp = sB + r * I8_W4_LDB + kk * 16 +
                                                (lane & 3) * 2;
                    const uint32_t w0 = *(const uint16_t*)rowp;
                    const uint32_t w1 = *(const uint16_t*)(rowp + 8);
                    bf[j][0] = i8_unpack4_sub(w0, zrep[W4 ? j : 0]);
                    bf[j][1] = i8_unpack4_sub(w1, zrep[W4 ? j : 0]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < NI; j += 2) {
                    const int r = wn0 + j * 8 + (lane & 7) + (lane >> 4) * 8;
                    const int c = kk * 2 + ((lane >> 3) & 1);
                    uint32_t t[4];
                    i8_ldsm_x4(t, sB + i8_swz(r, c));
                    bf[j][0] = t[0]; bf[j][1] = t[1];
                    bf[j + 1][0] = t[2]; bf[j + 1][1] = t[3];
                }
            }

#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
                    i8_mma(acc[i][j], af[i], bf[j][0], bf[j][1]);

            if (W4 && ((ks + 1) % steps_per_group) == 0) {
#pragma unroll
                for (int j = 0; j < (W4 ? NI : 1); j++) {
                    const float s0 = sc[j][0], s1 = sc[j][1];
#pragma unroll
                    for (int i = 0; i < MI; i++) {
                        facc[W4 ? i : 0][j][0] = fmaf((float)acc[i][j][0], s0, facc[W4 ? i : 0][j][0]);
                        facc[W4 ? i : 0][j][1] = fmaf((float)acc[i][j][1], s1, facc[W4 ? i : 0][j][1]);
                        facc[W4 ? i : 0][j][2] = fmaf((float)acc[i][j][2], s0, facc[W4 ? i : 0][j][2]);
                        facc[W4 ? i : 0][j][3] = fmaf((float)acc[i][j][3], s1, facc[W4 ? i : 0][j][3]);
                        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
                    }
                }
            }
        }
    }
    i8_cp_wait<0>();

    /* Epilogue: c0,c1 at row lane/4, cols 2(lane%4)+{0,1}; c2,c3 at row+8. */
#pragma unroll
    for (int i = 0; i < MI; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int gm = m0 + wm0 + i * 16 + (lane >> 2) + h * 8;
            if (gm >= M) continue;
            const float sxm = p.sx[gm];
#pragma unroll
            for (int j = 0; j < NI; j++) {
                const int gn = n0 + wn0 + j * 8 + (lane & 3) * 2;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (gn + e >= N) continue;
                    float v;
                    if (W4) v = facc[W4 ? i : 0][W4 ? j : 0][h * 2 + e] * sxm;
                    else    v = (float)acc[i][j][h * 2 + e] * sxm * p.sw[gn + e];
                    i8_store(p, gm, gn + e, v);
                }
            }
        }
    }
#else
    (void)p;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SIMT dp4a kernel (any sm_61+, any K % 32 == 0)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define I8S_BM 64
#define I8S_BN 64
#define I8S_BKW 8                  /* 32 bytes of K = 8 words per tile */

template <bool W4>
__global__ void __launch_bounds__(256)
i8_simt_kernel(const vv_i8_args p)
{
    __shared__ uint32_t sA[I8S_BM][I8S_BKW + 1];
    __shared__ uint32_t sB[I8S_BN][I8S_BKW + 1];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * I8S_BM, n0 = blockIdx.y * I8S_BN;
    const int K = p.K, M = p.M, N = p.N;
    const int ng = W4 ? K / p.G : 1;
    const int ktiles = K >> 5;
    const int tiles_per_group = W4 ? p.G >> 5 : 1;

    int acc[4][4];
    float facc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) { acc[i][j] = 0; facc[i][j] = 0.0f; }

    for (int kt = 0; kt < ktiles; kt++) {
        const int k0 = kt << 5;
        if (tid < 128) {
            const int r = tid >> 1, h = tid & 1;
            const int gm = m0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gm < M) v = *(const uint4*)(p.xq + (size_t)gm * K + k0 + h * 16);
            sA[r][h * 4 + 0] = v.x; sA[r][h * 4 + 1] = v.y;
            sA[r][h * 4 + 2] = v.z; sA[r][h * 4 + 3] = v.w;
        } else if (!W4) {
            const int t = tid - 128, r = t >> 1, h = t & 1;
            const int gn = n0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gn < N) v = *(const uint4*)((const int8_t*)p.w + (size_t)gn * K + k0 + h * 16);
            sB[r][h * 4 + 0] = v.x; sB[r][h * 4 + 1] = v.y;
            sB[r][h * 4 + 2] = v.z; sB[r][h * 4 + 3] = v.w;
        } else if (tid < 192) {
            const int r = tid - 128;
            const int gn = n0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            uint32_t zr = 0;
            if (gn < N) {
                v = *(const uint4*)((const uint8_t*)p.w + (size_t)gn * (K >> 1) + (k0 >> 1));
                zr = (uint32_t)p.zeros[(size_t)gn * ng + kt / tiles_per_group] * 0x01010101u;
            }
            const uint32_t words[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int q = 0; q < 4; q++) {
                sB[r][2 * q + 0] = (gn < N) ? i8_unpack4_sub(words[q] & 0xFFFFu, zr) : 0u;
                sB[r][2 * q + 1] = (gn < N) ? i8_unpack4_sub(words[q] >> 16, zr) : 0u;
            }
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < I8S_BKW; w++) {
            int a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = (int)sA[ty + 16 * i][w];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = (int)sB[tx + 16 * j][w];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = i8_dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();

        if (W4 && ((kt + 1) % tiles_per_group) == 0) {
            const int g = kt / tiles_per_group;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int gn = n0 + tx + 16 * j;
                const float s = (gn < N) ? __half2float(p.scales[(size_t)gn * ng + g]) : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    facc[i][j] = fmaf((float)acc[i][j], s, facc[i][j]);
                    acc[i][j] = 0;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
        const float sxm = p.sx[gm];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            const float v = W4 ? facc[i][j] * sxm
                               : (float)acc[i][j] * sxm * p.sw[gn];
            i8_store(p, gm, gn, v);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Dispatch
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Tensor cores need sm_80 for the s8 m16n8k32 shape, cp.async and this use of
 * ldmatrix. Cached per thread and device, the same way the attention kernels
 * cache theirs, because a request thread binds one device and a process may
 * serve from two. VV_I8_MMA=0 forces the SIMT kernel for comparisons.
 */
static bool i8_mma_usable(int* sm_count) {
    static thread_local int cached_dev = -1;
    static thread_local int usable = 0;
    static thread_local int sms = 0;
    static thread_local int allowed = -1;
    if (allowed < 0) {
        const char* e = getenv("VV_I8_MMA");
        allowed = (e && e[0] == '0') ? 0 : 1;
    }
    if (!allowed) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev != cached_dev) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess)
            return false;
        usable = (major >= 8);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   dev) != cudaSuccess)
            sms = 0;
        cached_dev = dev;
    }
    if (sm_count) *sm_count = sms;
    return usable != 0;
}

/*
 * Tile shape for this problem, measured on a 3090 (82 SMs) over the 7B and
 * 1.5B projections at M = 256 and 2048:
 *   - W8A8: 128x128 with three stages, not four. Three are 48 KB of shared
 *     memory, so two CTAs share an SM and hide each other's barriers; that
 *     is 183 against 137 TOPS on gate/up at M = 2048. When 128x128 would
 *     leave SMs idle (down/o at M = 256: 56 CTAs), 64x128 at one CTA per SM
 *     or better is faster (127 against 85 TOPS), and below that 32x64.
 *   - W4A8: the register-heavy 128x128 kernel runs one CTA per SM whatever
 *     the stage count, and it beats the smaller tiles even at 56 CTAs
 *     (75 against 44 TOPS on down at M = 256), so it is used down to half
 *     the SMs.
 * VV_I8_TILE=0..3 forces one.
 */
enum { I8_TILE_128x128 = 0, I8_TILE_64x128 = 1, I8_TILE_32x64 = 2,
       I8_TILE_128x128_S3 = 3, I8_TILE_COUNT };

static int i8_pick_tile(int M, int N, int sms, bool w4) {
    static thread_local int forced = -2;
    if (forced == -2) {
        const char* e = getenv("VV_I8_TILE");
        forced = (e && e[0] >= '0' && e[0] < '0' + I8_TILE_COUNT) ? e[0] - '0' : -1;
    }
    if (forced >= 0) return forced;
    const long nsm = sms > 0 ? sms : 80;
    const long t128 = (long)((M + 127) / 128) * ((N + 127) / 128);
    const long t64 = (long)((M + 63) / 64) * ((N + 127) / 128);
    if (w4) return (2 * t128 >= nsm) ? I8_TILE_128x128 : I8_TILE_32x64;
    if (t128 >= nsm) return I8_TILE_128x128_S3;
    if (t64 >= nsm) return I8_TILE_64x128;
    return I8_TILE_32x64;
}

/*
 * Kernels whose shared memory passes 48 KB need an opt-in per device. It is
 * set on first use, which is always an eager prefill (decode, the captured
 * part, is M <= 8 and never gets here), and remembered per thread and device.
 */
template <int BM, int BN, int WM_, int WN_, int ST, bool W4>
static vv_status_t i8_mma_launch(const vv_i8_args& p, cudaStream_t st) {
    typedef i8_mma_cfg<BM, BN, WM_, WN_, ST, W4> C;
    if (C::SMEM > 48 * 1024) {
        static thread_local int done_dev = -1;
        int dev = 0;
        cudaGetDevice(&dev);
        if (done_dev != dev) {
            if (cudaFuncSetAttribute(i8_mma_kernel<BM, BN, WM_, WN_, ST, W4>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     C::SMEM) != cudaSuccess)
                return VV_ERR_CUDA_LAUNCH;
            done_dev = dev;
        }
    }
    const dim3 grid((p.M + BM - 1) / BM, (p.N + BN - 1) / BN);
    i8_mma_kernel<BM, BN, WM_, WN_, ST, W4><<<grid, C::THREADS, C::SMEM, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

static vv_status_t i8_simt_launch(const vv_i8_args& p, bool w4, cudaStream_t st) {
    const dim3 grid((p.M + I8S_BM - 1) / I8S_BM, (p.N + I8S_BN - 1) / I8S_BN);
    if (w4) i8_simt_kernel<true><<<grid, 256, 0, st>>>(p);
    else    i8_simt_kernel<false><<<grid, 256, 0, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

static vv_status_t i8_dispatch(const vv_i8_args& p, bool w4, int path,
                               void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (p.M <= 0 || p.N <= 0 || p.K <= 0) return VV_ERR_INVALID_ARG;
    if ((p.K & 31) != 0) return VV_ERR_UNSUPPORTED;
    if (w4 && (p.G <= 0 || (p.G & 31) != 0 || (p.K % p.G) != 0))
        return VV_ERR_UNSUPPORTED;

    if (path == VV_I8_PATH_AUTO) {
        if (p.M <= 8 && (!w4 || p.xsum)) path = VV_I8_PATH_GEMV;
        else if (p.x_nibble) return VV_ERR_UNSUPPORTED;   /* GEMMs: natural */
        else if (i8_mma_usable(NULL) && (p.K % I8_BK) == 0) path = VV_I8_PATH_MMA;
        else path = VV_I8_PATH_SIMT;
    }
    switch (path) {
    case VV_I8_PATH_GEMV:
        return vv_i8_gemv_launch(&p, w4 ? 1 : 0, stream);
    case VV_I8_PATH_MMA: {
        if (p.x_nibble) return VV_ERR_UNSUPPORTED;
        int sms = 0;
        if (!i8_mma_usable(&sms) || (p.K % I8_BK) != 0) return VV_ERR_UNSUPPORTED;
        switch (i8_pick_tile(p.M, p.N, sms, w4)) {
        case I8_TILE_128x128:
            return w4 ? i8_mma_launch<128, 128, 2, 4, 4, true>(p, st)
                      : i8_mma_launch<128, 128, 2, 4, 4, false>(p, st);
        case I8_TILE_128x128_S3:
            return w4 ? i8_mma_launch<128, 128, 2, 4, 3, true>(p, st)
                      : i8_mma_launch<128, 128, 2, 4, 3, false>(p, st);
        case I8_TILE_64x128:
            return w4 ? i8_mma_launch<64, 128, 2, 4, 4, true>(p, st)
                      : i8_mma_launch<64, 128, 2, 4, 4, false>(p, st);
        default:
            return w4 ? i8_mma_launch<32, 64, 2, 2, 4, true>(p, st)
                      : i8_mma_launch<32, 64, 2, 2, 4, false>(p, st);
        }
    }
    case VV_I8_PATH_SIMT:
        if (p.x_nibble) return VV_ERR_UNSUPPORTED;
        return i8_simt_launch(p, w4, st);
    default:
        return VV_ERR_INVALID_ARG;
    }
}

extern "C" {

vv_status_t vv_w8a8_linear_dev(
    const int8_t* xq, const float* sx, const int8_t* w, const float* sw,
    const void* bias, const void* residual, void* y, int y_f32,
    int M, int N, int K, int path, void* stream)
{
    if (!xq || !sx || !w || !sw || !y) return VV_ERR_NULL_PTR;
    vv_i8_args p;
    p.xq = xq; p.sx = sx; p.xsum = NULL; p.w = w; p.sw = sw;
    p.scales = NULL; p.zeros = NULL;
    p.bias = (const half*)bias; p.res = (const half*)residual;
    p.y = y; p.y_f32 = y_f32; p.M = M; p.N = N; p.K = K; p.G = K;
    p.x_nibble = 0;
    return i8_dispatch(p, false, path, stream);
}

vv_status_t vv_w4a8_linear_dev(
    const int8_t* xq, int x_layout, const float* sx, const int32_t* xsum,
    const uint8_t* packed, const void* scales, const uint8_t* zeros,
    int group_size, const void* bias, const void* residual, void* y,
    int y_f32, int M, int N, int K, int path, void* stream)
{
    if (!xq || !sx || !packed || !scales || !zeros || !y)
        return VV_ERR_NULL_PTR;
    vv_i8_args p;
    p.xq = xq; p.sx = sx; p.xsum = xsum; p.w = packed; p.sw = NULL;
    p.scales = (const half*)scales; p.zeros = zeros;
    p.bias = (const half*)bias; p.res = (const half*)residual;
    p.y = y; p.y_f32 = y_f32; p.M = M; p.N = N; p.K = K; p.G = group_size;
    p.x_nibble = (x_layout == VV_Q8_NIBBLE);
    return i8_dispatch(p, true, path, stream);
}

} /* extern "C" */
