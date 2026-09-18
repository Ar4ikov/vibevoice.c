/**
 * @file bitnet.cu
 * @brief Integer kernels for VibeVoice-ASR-BitNet on NVIDIA GPUs.
 *
 * The same products as src/cpu/bitnet_cpu.c, with the same numerics (see
 * include/vibevoice/bitnet.h), so a GPU result can be compared to the CPU one
 * bit for bit:
 *
 *   ternary x int8   decode: dp4a GEMV, one warp per weight row. The 2-bit
 *                    codes of one 32-bit word line up with four contiguous
 *                    activation words, so the unpack is a shift and a mask.
 *                    prefill: mma.m16n8k32.s8 with the codes kept 2-bit in
 *                    shared memory and unpacked into B fragments in
 *                    registers: 4x less shared memory and global traffic for
 *                    the weights than an int8 tile.
 *   int8 x int8      the same GEMM with an int8 B tile (the I8_S encoder)
 *   int8 head        dp4a GEMV fused with a two-stage argmax
 *
 * The codes stay unsigned (0, 1, 2) inside the dot products and the row sum
 * of the activations is subtracted once at the end:
 *     sum_k (c - 1) q = sum_k c q - sum_k q.
 *
 * Every per-token scalar (activation scale, row sum, absmax) is read from
 * device memory, and nothing here allocates or sets attributes at launch, so
 * the decode path can be captured into a CUDA graph.
 *
 * Release builds use --use_fast_math, which turns `/` into an approximate
 * divide and lets the compiler contract a*b+c into an FMA. The epilogues use
 * the __f*_rn intrinsics so they round exactly like the CPU code.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#define TERN_MASK 0x03030303u

/* ─── Activation quantization ───────────────────────────────────────────── */

#define AQ_THREADS 256

__device__ __forceinline__ float aq_load(const void* x, int f16, size_t i) {
    return f16 ? __half2float(((const half*)x)[i]) : ((const float*)x)[i];
}

__global__ void act_quant_kernel(const void* __restrict__ x, int f16, int K,
                                 int8_t* __restrict__ q,
                                 float* __restrict__ scale,
                                 int32_t* __restrict__ sum) {
    __shared__ float s_max[AQ_THREADS / 32];
    __shared__ int s_sum[AQ_THREADS / 32];
    __shared__ float s_scale;
    const int m = blockIdx.x;
    const size_t base = (size_t)m * K;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += AQ_THREADS)
        amax = fmaxf(amax, fabsf(aq_load(x, f16, base + k)));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_max[warp] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = 0.0f;
        for (int w = 0; w < AQ_THREADS / 32; w++) mx = fmaxf(mx, s_max[w]);
        /* same as the CPU: 127 / max(amax, 1e-5) in double, rounded once */
        const double d = mx > 0.00001 ? (double)mx : 0.00001;
        s_scale = (float)(127.0 / d);
    }
    __syncthreads();
    const float s = s_scale;

    int acc = 0;
    for (int k = threadIdx.x; k < K; k += AQ_THREADS) {
        int v = __float2int_rn(__fmul_rn(aq_load(x, f16, base + k), s));
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        q[base + k] = (int8_t)v;
        acc += v;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) s_sum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < AQ_THREADS / 32; w++) t += s_sum[w];
        scale[m] = s;
        if (sum) sum[m] = t;
    }
}

/* ─── Shared epilogue ───────────────────────────────────────────────────── */

struct Epi {
    const int32_t* xsum;    /* ternary: row sums of q, subtracted          */
    const float* x_scale;   /* ternary: per-token multiplier               */
    const float* a_scale;   /* int8: device scalar multiplier              */
    float w_scale;
    const float* bias;
    int32_t* acc;
    void* y;
    int y_f16;
    float* absmax;          /* int8 only                                    */
    int tern;
};

/** Finish one output element; returns |y| for the absmax fold (0 if none). */
__device__ __forceinline__ float epi_store(const Epi& e, int m, int n, int N,
                                           int32_t v) {
    if (e.tern) v -= e.xsum[m];
    const size_t o = (size_t)m * N + n;
    if (e.acc) e.acc[o] = v;
    if (!e.y) return 0.0f;
    float r;
    if (e.tern) {
        r = __fmul_rn(__fdiv_rn((float)v, e.x_scale[m]), e.w_scale);
    } else {
        r = __fmul_rn((float)v, __fdiv_rn(e.w_scale, *e.a_scale));
    }
    if (e.bias) r = __fadd_rn(r, e.bias[n]);
    if (e.y_f16) ((half*)e.y)[o] = __float2half_rn(r);
    else ((float*)e.y)[o] = r;
    return fabsf(r);
}

__device__ __forceinline__ void absmax_fold(float* absmax, float v) {
    /* non-negative floats order like their bit patterns */
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0 && v > 0.0f) atomicMax((int*)absmax, __float_as_int(v));
}

/* ─── GEMV (M <= 8): one warp per weight row ────────────────────────────── */

#define GEMV_WARPS 4
#define GEMV_MAXM 8

/* ternary: row = K/16 words; word i covers block i/8, bytes (i%8)*4..+3 */
__global__ void __launch_bounds__(32 * GEMV_WARPS)
tern_gemv_kernel(const int8_t* __restrict__ q, const uint8_t* __restrict__ codes,
                 Epi e, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    const uint32_t* wrow = (const uint32_t*)(codes + (size_t)n * (K / 4));
    const int words = K / 16;
    int a[GEMV_MAXM];
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; m++) a[m] = 0;

    for (int wi = lane; wi < words; wi += 32) {
        const uint32_t w = __ldg(wrow + wi);
        const int c0 = (int)((w >> 6) & TERN_MASK), c1 = (int)((w >> 4) & TERN_MASK);
        const int c2 = (int)((w >> 2) & TERN_MASK), c3 = (int)(w & TERN_MASK);
        const int xo = (wi >> 3) * 128 + (wi & 7) * 4; /* byte offset in the row */
#pragma unroll
        for (int m = 0; m < GEMV_MAXM; m++) {
            if (m < M) {
                const int* xp = (const int*)(q + (size_t)m * K + xo);
                a[m] = __dp4a(c0, __ldg(xp), a[m]);
                a[m] = __dp4a(c1, __ldg(xp + 8), a[m]);
                a[m] = __dp4a(c2, __ldg(xp + 16), a[m]);
                a[m] = __dp4a(c3, __ldg(xp + 24), a[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; m++) {
        if (m < M) {
            int v = a[m];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) epi_store(e, m, n, N, v);
        }
    }
}

/* int8: dp4a over aligned words, bytes when K % 4 != 0 */
__global__ void __launch_bounds__(32 * GEMV_WARPS)
i8_gemv_kernel(const int8_t* __restrict__ a, const int8_t* __restrict__ w,
               Epi e, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    int s[GEMV_MAXM];
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; m++) s[m] = 0;
    const int8_t* wr = w + (size_t)n * K;
    if ((K & 3) == 0) {
        const int* w4 = (const int*)wr;
        for (int i = lane; i < K / 4; i += 32) {
            const int wv = __ldg(w4 + i);
#pragma unroll
            for (int m = 0; m < GEMV_MAXM; m++)
                if (m < M) s[m] = __dp4a(wv, __ldg((const int*)(a + (size_t)m * K) + i), s[m]);
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            const int wv = wr[k];
#pragma unroll
            for (int m = 0; m < GEMV_MAXM; m++)
                if (m < M) s[m] += wv * (int)a[(size_t)m * K + k];
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; m++) {
        if (m < M) {
            int v = s[m];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) amax = fmaxf(amax, epi_store(e, m, n, N, v));
        }
    }
    if (e.absmax) absmax_fold(e.absmax, amax);
}

/* ─── Tensor-core GEMM (sm_80+) ─────────────────────────────────────────── */

#define MM_BM 64
#define MM_BN 64
#define MM_BK 128
#define MM_THREADS 128
/* 144-byte smem rows: row r starts on bank 4r, so the eight rows a fragment
 * load touches land on disjoint banks */
#define MM_LD 144

__device__ __forceinline__ void mma_s8(int (&c)[4], const int (&a)[4], int b0, int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;
#endif
}

/** Copy 16 bytes of row `row` at column `col` (bytes), zero outside [R, K). */
template <bool ALIGNED>
__device__ __forceinline__ uint4 ld16(const int8_t* base, int row, int R,
                                      int col, int K, size_t ld) {
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row >= R) return v;
    const int8_t* p = base + (size_t)row * ld + col;
    if (ALIGNED) {
        if (col < K) v = __ldg((const uint4*)p);
    } else {
        int8_t tmp[16];
#pragma unroll
        for (int i = 0; i < 16; i++) tmp[i] = (col + i < K) ? p[i] : (int8_t)0;
        v = *(const uint4*)tmp;
    }
    return v;
}

/*
 * C[M, N] = A[M, K] . B[N, K]^T with A int8 row-major and B either int8
 * rows (TERN = false) or 2-bit ternary codes (TERN = true, K % 128 == 0).
 * 64x64 block tile, 4 warps of 32x32, K step 128, register-staged double
 * buffering.
 */
template <bool TERN, bool ALIGNED>
__global__ void __launch_bounds__(MM_THREADS)
i8_mma_kernel(const int8_t* __restrict__ A, const void* __restrict__ Bv,
              Epi e, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int B_LD = TERN ? 32 : MM_LD;
    __shared__ __align__(16) int8_t As[2][MM_BM * MM_LD];
    __shared__ __align__(16) int8_t Bs[2][MM_BN * B_LD];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * MM_BM, n0 = blockIdx.x * MM_BN;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int g = lane >> 2, t4 = lane & 3;
    const int nk = (K + MM_BK - 1) / MM_BK;
    const int8_t* B8 = (const int8_t*)Bv;
    const size_t tern_ld = (size_t)K / 4;

    uint4 ra[4], rb[4];
    auto load = [&](int kt) {
        const int k0 = kt * MM_BK;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = tid + i * MM_THREADS; /* 512 chunks of 16 B */
            ra[i] = ld16<ALIGNED>(A, m0 + c / 8, M, k0 + (c % 8) * 16, K, (size_t)K);
        }
        if (TERN) {
            const int row = tid >> 1, part = tid & 1;
            rb[0] = make_uint4(0, 0, 0, 0);
            if (n0 + row < N)
                rb[0] = __ldg((const uint4*)(B8 + (size_t)(n0 + row) * tern_ld +
                                            (size_t)kt * 32 + part * 16));
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int c = tid + i * MM_THREADS;
                rb[i] = ld16<ALIGNED>(B8, n0 + c / 8, N, k0 + (c % 8) * 16, K, (size_t)K);
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = tid + i * MM_THREADS;
            *(uint4*)&As[buf][(c / 8) * MM_LD + (c % 8) * 16] = ra[i];
        }
        if (TERN) {
            *(uint4*)&Bs[buf][(tid >> 1) * 32 + (tid & 1) * 16] = rb[0];
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int c = tid + i * MM_THREADS;
                *(uint4*)&Bs[buf][(c / 8) * MM_LD + (c % 8) * 16] = rb[i];
            }
        }
    };

    int acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    load(0);
    store(0);
    __syncthreads();

    for (int kt = 0; kt < nk; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < nk) load(kt + 1);

        const int8_t* as = As[buf];
        const int8_t* bs = Bs[buf];
        /* ternary: the two code words of each n8 tile hold all four k32
         * steps of this K block (one 2-bit field each) */
        uint32_t tw[4][2];
        if (TERN) {
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int n = wn + nt * 8 + g;
                tw[nt][0] = *(const uint32_t*)&bs[n * 32 + t4 * 4];
                tw[nt][1] = *(const uint32_t*)&bs[n * 32 + 16 + t4 * 4];
            }
        }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            int af[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const int r = wm + mt * 16 + g;
                const int c = kk * 32 + t4 * 4;
                af[mt][0] = *(const int*)&as[r * MM_LD + c];
                af[mt][1] = *(const int*)&as[(r + 8) * MM_LD + c];
                af[mt][2] = *(const int*)&as[r * MM_LD + c + 16];
                af[mt][3] = *(const int*)&as[(r + 8) * MM_LD + c + 16];
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                int b0, b1;
                if (TERN) {
                    const int sh = 6 - 2 * kk;
                    b0 = (int)((tw[nt][0] >> sh) & TERN_MASK);
                    b1 = (int)((tw[nt][1] >> sh) & TERN_MASK);
                } else {
                    const int n = wn + nt * 8 + g;
                    b0 = *(const int*)&bs[n * MM_LD + kk * 32 + t4 * 4];
                    b1 = *(const int*)&bs[n * MM_LD + kk * 32 + 16 + t4 * 4];
                }
#pragma unroll
                for (int mt = 0; mt < 2; mt++) mma_s8(acc[mt][nt], af[mt], b0, b1);
            }
        }
        if (kt + 1 < nk) store(buf ^ 1);
        __syncthreads();
    }

    float amax = 0.0f;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int m = m0 + wm + mt * 16 + g + (r >= 2 ? 8 : 0);
                const int n = n0 + wn + nt * 8 + t4 * 2 + (r & 1);
                if (m < M && n < N) amax = fmaxf(amax, epi_store(e, m, n, N, acc[mt][nt][r]));
            }
    if (!TERN && e.absmax) absmax_fold(e.absmax, amax);
#else
    (void)A; (void)Bv; (void)e; (void)M; (void)N; (void)K;
#endif
}

/* ─── Requantization ────────────────────────────────────────────────────── */

__global__ void requant_kernel(const float* __restrict__ y, int64_t n,
                               const float* __restrict__ absmax, int relu,
                               int8_t* __restrict__ q, float* __restrict__ out_scale) {
    const float am = *absmax;
    const float inv = am != 0.0f ? __fdiv_rn(127.0f, am) : 0.0f;
    const float lo = relu ? 0.0f : -127.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float v = __fmul_rn(y[i], inv);
        v = fminf(fmaxf(v, lo), 127.0f);
        q[i] = (int8_t)__float2int_rn(v);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && out_scale) *out_scale = inv;
}

/* ─── Int8 head + argmax ────────────────────────────────────────────────── */

#define HEAD_WARPS 8
#define HEAD_ROWS_PER_WARP 8
#define HEAD_ROWS (HEAD_WARPS * HEAD_ROWS_PER_WARP)

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

__global__ void __launch_bounds__(32 * HEAD_WARPS)
i8_head_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
               const int8_t* __restrict__ w, const float* __restrict__ w_scale,
               int V, int K, float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float s_v[HEAD_WARPS];
    __shared__ int s_i[HEAD_WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float s = *scale;
    const int* x4 = (const int*)q;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    const int r0 = blockIdx.x * HEAD_ROWS + warp * HEAD_ROWS_PER_WARP;
    for (int r = 0; r < HEAD_ROWS_PER_WARP; r++) {
        const int n = r0 + r;
        if (n >= V) break;
        const int* w4 = (const int*)(w + (size_t)n * K);
        int acc = 0;
        for (int i = lane; i < K / 4; i += 32) acc = __dp4a(__ldg(w4 + i), __ldg(x4 + i), acc);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        /* same order as the CPU: ((float)acc * w_scale) / s */
        const float v = __fdiv_rn(__fmul_rn((float)acc, w_scale[n]), s);
        if (better(v, n, bv, bi)) { bv = v; bi = n; }
    }
    if (lane == 0) { s_v[warp] = bv; s_i[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = s_v[0];
        int i = s_i[0];
        for (int k = 1; k < HEAD_WARPS; k++)
            if (better(s_v[k], s_i[k], v, i)) { v = s_v[k]; i = s_i[k]; }
        pv[blockIdx.x] = v;
        pi[blockIdx.x] = i;
    }
}

__global__ void head_final_kernel(const float* __restrict__ pv, const int* __restrict__ pi,
                                  int n, int32_t* __restrict__ token,
                                  float* __restrict__ value) {
    __shared__ float s_v[256];
    __shared__ int s_i[256];
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 256)
        if (better(pv[i], pi[i], bv, bi)) { bv = pv[i]; bi = pi[i]; }
    s_v[threadIdx.x] = bv;
    s_i[threadIdx.x] = bi;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st &&
            better(s_v[threadIdx.x + st], s_i[threadIdx.x + st], s_v[threadIdx.x], s_i[threadIdx.x])) {
            s_v[threadIdx.x] = s_v[threadIdx.x + st];
            s_i[threadIdx.x] = s_i[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *token = s_i[0];
        if (value) *value = s_v[0];
    }
}

/* ─── Host side ─────────────────────────────────────────────────────────── */

extern "C" {

#include "vibevoice/device.h"
#include "vibevoice/types.h"

/**
 * @brief Whether this device runs the tensor-core path. Cached per thread
 * and per device, as prefill_use_mma in attention.cu; VV_BITNET_MMA=0
 * forces the dp4a fallback, which is how the two are compared on one card.
 */
static bool bitnet_use_mma(void) {
    static thread_local int cached_dev = -1;
    static thread_local int usable = 0;
    static thread_local int allowed = -1;
    if (allowed < 0) {
        const char* e = getenv("VV_BITNET_MMA");
        allowed = (e && e[0] == '0') ? 0 : 1;
    }
    if (!allowed) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev != cached_dev) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        usable = major >= 8;
        cached_dev = dev;
    }
    return usable != 0;
}

static vv_status_t launch_status(void) {
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_act_quant_i8_dev(const void* x, int x_f16, int M, int K,
                                int8_t* q, float* scale, int32_t* sum,
                                void* stream) {
    if (!x || !q || !scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    act_quant_kernel<<<M, AQ_THREADS, 0, (cudaStream_t)stream>>>(x, x_f16, K, q, scale, sum);
    return launch_status();
}

/** GEMV over row blocks of GEMV_MAXM: the M <= 8 path and the pre-sm_80 GEMM. */
static void gemv_chunks(bool tern, const int8_t* a, const void* w, Epi e,
                        int M, int N, int K, cudaStream_t st) {
    const dim3 grid((N + GEMV_WARPS - 1) / GEMV_WARPS);
    for (int m0 = 0; m0 < M; m0 += GEMV_MAXM) {
        const int mb = M - m0 < GEMV_MAXM ? M - m0 : GEMV_MAXM;
        Epi ec = e;
        if (ec.xsum) ec.xsum += m0;
        if (ec.x_scale) ec.x_scale += m0;
        if (ec.acc) ec.acc += (size_t)m0 * N;
        if (ec.y) ec.y = (char*)ec.y + (size_t)m0 * N * (ec.y_f16 ? 2 : 4);
        if (tern)
            tern_gemv_kernel<<<grid, 32 * GEMV_WARPS, 0, st>>>(
                a + (size_t)m0 * K, (const uint8_t*)w, ec, mb, N, K);
        else
            i8_gemv_kernel<<<grid, 32 * GEMV_WARPS, 0, st>>>(
                a + (size_t)m0 * K, (const int8_t*)w, ec, mb, N, K);
    }
}

vv_status_t vv_ternary_gemm_dev(const int8_t* q, const int32_t* xsum,
                                const float* x_scale, const uint8_t* codes,
                                float w_scale, const float* bias,
                                int32_t* acc, void* y, int y_f16,
                                int M, int N, int K, void* stream) {
    if (!q || !xsum || !codes) return VV_ERR_NULL_PTR;
    if (y && !x_scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0 || K % 128) return VV_ERR_INVALID_ARG;
    Epi e = {};
    e.xsum = xsum;
    e.x_scale = x_scale;
    e.w_scale = w_scale;
    e.bias = bias;
    e.acc = acc;
    e.y = y;
    e.y_f16 = y_f16;
    e.tern = 1;
    cudaStream_t st = (cudaStream_t)stream;
    if (M > GEMV_MAXM && bitnet_use_mma()) {
        const dim3 grid((N + MM_BN - 1) / MM_BN, (M + MM_BM - 1) / MM_BM);
        i8_mma_kernel<true, true><<<grid, MM_THREADS, 0, st>>>(q, codes, e, M, N, K);
    } else {
        gemv_chunks(true, q, codes, e, M, N, K, st);
    }
    return launch_status();
}

vv_status_t vv_i8_gemm_dev(const int8_t* a, const float* a_scale,
                           const int8_t* w, float w_scale, const float* bias,
                           int32_t* acc, float* y, float* absmax,
                           int M, int N, int K, void* stream) {
    if (!a || !w) return VV_ERR_NULL_PTR;
    if (y && !a_scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    Epi e = {};
    e.a_scale = a_scale;
    e.w_scale = w_scale;
    e.bias = bias;
    e.acc = acc;
    e.y = y;
    e.absmax = y ? absmax : NULL;
    cudaStream_t st = (cudaStream_t)stream;
    if (M > GEMV_MAXM && bitnet_use_mma()) {
        const dim3 grid((N + MM_BN - 1) / MM_BN, (M + MM_BM - 1) / MM_BM);
        const bool aligned = (K % 16) == 0 && ((uintptr_t)a % 16) == 0 &&
                             ((uintptr_t)w % 16) == 0;
        if (aligned)
            i8_mma_kernel<false, true><<<grid, MM_THREADS, 0, st>>>(a, w, e, M, N, K);
        else
            i8_mma_kernel<false, false><<<grid, MM_THREADS, 0, st>>>(a, w, e, M, N, K);
    } else {
        gemv_chunks(false, a, w, e, M, N, K, st);
    }
    return launch_status();
}

vv_status_t vv_i8s_requant_dev(const float* y, int64_t n, const float* absmax,
                               int relu, int8_t* q, float* out_scale,
                               void* stream) {
    if (!y || !absmax || !q) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_ERR_INVALID_ARG;
    int64_t blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    requant_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(y, n, absmax, relu, q, out_scale);
    return launch_status();
}

size_t vv_i8_head_argmax_scratch_bytes(int V) {
    const size_t nb = (size_t)(V + HEAD_ROWS - 1) / HEAD_ROWS;
    return nb * (sizeof(float) + sizeof(int));
}

vv_status_t vv_i8_head_argmax_dev(const int8_t* q, const float* scale,
                                  const int8_t* w, const float* w_scale,
                                  int V, int K, int32_t* token, float* value,
                                  void* scratch, void* stream) {
    if (!q || !scale || !w || !w_scale || !token || !scratch) return VV_ERR_NULL_PTR;
    if (V <= 0 || K <= 0 || K % 4) return VV_ERR_INVALID_ARG;
    const int nb = (V + HEAD_ROWS - 1) / HEAD_ROWS;
    float* pv = (float*)scratch;
    int* pi = (int*)(pv + nb);
    cudaStream_t st = (cudaStream_t)stream;
    i8_head_kernel<<<nb, 32 * HEAD_WARPS, 0, st>>>(q, scale, w, w_scale, V, K, pv, pi);
    head_final_kernel<<<1, 256, 0, st>>>(pv, pi, nb, token, value);
    return launch_status();
}

} /* extern "C" */
