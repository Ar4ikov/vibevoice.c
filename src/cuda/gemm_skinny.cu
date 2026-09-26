/**
 * @file gemm_skinny.cu
 * @brief Tensor-core linear layers for up to 64 rows (a prefill's 9..64, a
 *        fast check's 2..8) on dense FP16, per-channel INT8 and NF4 weights,
 *        bit-identical to the dequant + tile GEMM path they replace.
 *
 * A Streaming-7B chunk prefills 29 rows. The tile GEMM (gemm.cu) pads them
 * to a 128-row tile and covers N with 128-wide blocks: 4 blocks for the
 * 512-row k and v projections, 28 for the 3584-row ones, on an 82-SM card,
 * each doing four times the tensor work the rows need. INT8 and NF4 first
 * write the whole weight out as FP16 and read it back. Here a block covers
 * the rows in 16-row MMA tiles and 64 output columns, the projections of one
 * input (q/k/v, gate/up) share a launch so every SM gets blocks, and the
 * weight goes from memory to the B fragments in the form it is stored,
 * dequantized in registers.
 *
 * Same bits as the old path, by construction:
 *   - every output is one FP32 accumulator fed by mma.m16n8k16 over k in
 *     ascending 16-wide steps, starting from zero. That is what the tile
 *     kernel's wmma m16n16k16 compiles to on sm_80+ (two HMMA.16816.F32 per
 *     fragment, one per 8 columns), and there is no split-K, so there is no
 *     other summation order;
 *   - INT8 and NF4 dequantize with the float expressions of
 *     int8_dequant_kernel and vv_dequant_nf4_kernel, rounded to FP16 the
 *     same way, so the MMA sees the FP16 weight the scratch copy held;
 *   - the epilogue is alpha * acc rounded to FP16, then the bias added the
 *     way vv_bias_add_dev adds it.
 * So whatever row count a prefill chunk lands on, the output does not move;
 * tests/test_skinny.c holds the kernels to that byte for byte. sm_80+ only
 * (mma.m16n8k16, cp.async): elsewhere the entry point declines and the
 * caller runs the old path.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "vibevoice/device.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SK_SM80 1
#endif

#define SK_WARPS    4
#define SK_THREADS  (SK_WARPS * 32)
#define SK_MAX_SEGS 3

/** NF4 code book: the values vv_dequant_nf4_kernel multiplies by. */
__constant__ float c_sk_nf4[16] = {
    -1.0f,              -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f,  0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};

/* ─── PTX helpers (sm_80+) ───────────────────────────────────────────────── */

/** @brief 16 bytes global -> shared, zero-filled when !valid. */
__device__ __forceinline__ void sk_cp16(void* smem, const void* gmem,
                                        bool valid) {
#ifdef SK_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(valid ? 16 : 0));
#else
    (void)smem; (void)gmem; (void)valid;
#endif
}

/** @brief A few bytes global -> shared (the NF4 scales of one row). */
template <int BYTES>
__device__ __forceinline__ void sk_cp_small(void* smem, const void* gmem,
                                            bool valid) {
#ifdef SK_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                 :: "r"(s), "l"(gmem), "n"(BYTES), "r"(valid ? BYTES : 0));
#else
    (void)smem; (void)gmem; (void)valid;
#endif
}

__device__ __forceinline__ void sk_commit(void) {
#ifdef SK_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N_PENDING>
__device__ __forceinline__ void sk_wait(void) {
#ifdef SK_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_PENDING));
#endif
}

__device__ __forceinline__ void sk_ldsm_x4(uint32_t r[4], const void* p) {
#ifdef SK_SM80
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#else
    (void)p; r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

/** @brief d += a . b, m16n8k16, FP16 in, FP32 accumulate. */
__device__ __forceinline__ void sk_mma(float d[4], const uint32_t a[4],
                                       uint32_t b0, uint32_t b1) {
#ifdef SK_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ uint32_t sk_h2u(half2 v) {
    return *reinterpret_cast<uint32_t*>(&v);
}

/**
 * @brief Two consecutive int8 weights (low byte = lower k) as the half2 the
 *        dequant kernel would have stored: fp16((float)q * s).
 *
 * q + 128 goes into the low mantissa byte of 2^23, and subtracting
 * 2^23 + 128 leaves exactly q: the same float as an I2F, from a PRMT and an
 * FADD instead of the quarter-rate conversion unit.
 */
__device__ __forceinline__ uint32_t sk_i8_pair(uint32_t v, float s) {
    const uint32_t x = v ^ 0x8080u;
    const float f0 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7650))
                   - 8388736.0f;
    const float f1 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7651))
                   - 8388736.0f;
    return sk_h2u(__floats2half2_rn(f0 * s, f1 * s));
}

/**
 * @brief One NF4 byte (high nibble = lower k) as the half2 the dequant
 *        kernel would have stored: fp16(lut[q] * absmax).
 */
__device__ __forceinline__ uint32_t sk_nf4_pair(uint32_t byte, float sc,
                                                const float* lut) {
    return sk_h2u(__floats2half2_rn(lut[byte >> 4] * sc,
                                    lut[byte & 0xFu] * sc));
}

/* ─── Kernel ─────────────────────────────────────────────────────────────── */

/** @brief Up to three projections of one input, passed by value. */
struct sk_segs {
    const uint8_t* w[SK_MAX_SEGS];
    const void*    sc[SK_MAX_SEGS];
    const half*    bias[SK_MAX_SEGS];
    half*          y[SK_MAX_SEGS];
    int            n[SK_MAX_SEGS];
    int            first_block[SK_MAX_SEGS];   /* unused: INT_MAX */
};

/**
 * @brief Shared-memory layout of one pipeline stage.
 *
 * A: MT*16 rows of BK halves. W: BN rows of the weight's own bytes for the
 * same BK columns (FP16 BK*2, INT8 BK, NF4 BK/2), then for NF4 the BK/64
 * absmax values of each row. Row strides are padded so the ldmatrix rows and
 * the per-lane 8- and 16-bit weight reads of one warp hit distinct banks;
 * every part stays 16-byte aligned for cp.async.
 */
template <int WT, int MT, int NT, int BK>
struct sk_shape {
    static constexpr int BN = SK_WARPS * NT * 8;
    static constexpr int ASTR = BK + 8;                          /* halves */
    static constexpr int A_BYTES = MT * 16 * ASTR * 2;
    static constexpr int W_ROW = WT == VV_SKINNY_F16 ? BK * 2
                               : WT == VV_SKINNY_INT8 ? BK : BK / 2;
    static constexpr int WSTR = W_ROW + 16;                      /* bytes  */
    static constexpr int W_BYTES = BN * WSTR;
    static constexpr int S_ROW = WT == VV_SKINNY_NF4 ? BK / 64 * 2 : 0;
    static constexpr int S_BYTES = (BN * S_ROW + 15) / 16 * 16;
    static constexpr int STAGE = A_BYTES + W_BYTES + S_BYTES;
    static constexpr int LUT_BYTES = WT == VV_SKINNY_NF4 ? 64 : 0;
};

/**
 * @brief y_seg[M, N_seg] = alpha * A[M, K] . W_seg^T (+ bias_seg).
 *
 * Grid: the concatenated BN-wide column blocks of every projection. Warp w
 * of a block owns columns w*NT*8 .. +NT*8 and every row; the stage loop is
 * the usual multi-stage cp.async pipeline, and inside a stage the k16 steps
 * run in order, so each accumulator sees k ascending.
 */
template <int WT, int MT, int NT, int BK, int STAGES>
__global__ void __launch_bounds__(SK_THREADS)
skinny_kernel(const half* __restrict__ A, const sk_segs segs,
              int M, int K, float alpha)
{
    typedef sk_shape<WT, MT, NT, BK> S;
    constexpr int BN = S::BN;
    constexpr int KS = BK / 16;
    static_assert(BK % 64 == 0 || WT != VV_SKINNY_NF4, "NF4 scale blocks");
    static_assert(WT != VV_SKINNY_NF4 || BK >= 128, "cp.async of the scales");
    static_assert(NT % 2 == 0, "FP16 B fragments load in pairs");

    extern __shared__ __align__(16) uint8_t sk_smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;

    /* Which projection this block serves: selects, not indexed loads. */
    const int bx = (int)blockIdx.x;
    const int seg = bx >= segs.first_block[2] ? 2
                  : (bx >= segs.first_block[1] ? 1 : 0);
    const uint8_t* W = seg == 0 ? segs.w[0] : (seg == 1 ? segs.w[1] : segs.w[2]);
    [[maybe_unused]] const void* SC = seg == 0 ? segs.sc[0]
                                    : (seg == 1 ? segs.sc[1] : segs.sc[2]);
    const half* bias = seg == 0 ? segs.bias[0]
                     : (seg == 1 ? segs.bias[1] : segs.bias[2]);
    half* Y = seg == 0 ? segs.y[0] : (seg == 1 ? segs.y[1] : segs.y[2]);
    const int N = seg == 0 ? segs.n[0] : (seg == 1 ? segs.n[1] : segs.n[2]);
    const int blk = bx - (seg == 0 ? 0 : (seg == 1 ? segs.first_block[1]
                                                   : segs.first_block[2]));
    const int n0 = blk * BN;
    const int KT = K / BK;

    /* NF4: the code book, in shared memory past the stages. */
    [[maybe_unused]] float* lut = (float*)(sk_smem + STAGES * S::STAGE);
    if constexpr (WT == VV_SKINNY_NF4) {
        if (tid < 16) lut[tid] = c_sk_nf4[tid];
    }

    /* INT8: one scale per output row, for the rows this lane's B holds. */
    [[maybe_unused]] float srow[NT];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        srow[nt] = 0.0f;
        if constexpr (WT == VV_SKINNY_INT8) {
            const int n = n0 + warp * NT * 8 + nt * 8 + g;
            if (n < N) srow[nt] = ((const float*)SC)[n];
        }
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    const size_t w_ld = WT == VV_SKINNY_F16 ? (size_t)K * 2
                      : WT == VV_SKINNY_INT8 ? (size_t)K : (size_t)K / 2;

    auto load_stage = [&](int s, int kt) {
        uint8_t* base = sk_smem + s * S::STAGE;
        half* As = (half*)base;
        const int k0 = kt * BK;
        constexpr int A_CH = BK / 8;                  /* 16-byte chunks/row */
#pragma unroll
        for (int i = tid; i < MT * 16 * A_CH; i += SK_THREADS) {
            const int r = i / A_CH, c = i % A_CH;
            const bool ok = r < M;
            const half* src = ok ? A + (size_t)r * K + k0 + c * 8 : A;
            sk_cp16(As + r * S::ASTR + c * 8, src, ok);
        }
        uint8_t* Ws = base + S::A_BYTES;
        constexpr int W_CH = S::W_ROW / 16;
        const size_t wk = (size_t)kt * S::W_ROW;
#pragma unroll
        for (int i = tid; i < BN * W_CH; i += SK_THREADS) {
            const int r = i / W_CH, c = i % W_CH;
            const bool ok = n0 + r < N;
            const uint8_t* src = ok ? W + (size_t)(n0 + r) * w_ld + wk + c * 16
                                    : W;
            sk_cp16(Ws + r * S::WSTR + c * 16, src, ok);
        }
        if constexpr (WT == VV_SKINNY_NF4) {
            uint8_t* Ss = Ws + S::W_BYTES;
            const size_t s_ld = (size_t)(K / 64) * 2;
            for (int i = tid; i < BN; i += SK_THREADS) {
                const bool ok = n0 + i < N;
                const uint8_t* src = ok
                    ? (const uint8_t*)SC + (size_t)(n0 + i) * s_ld
                      + (size_t)kt * S::S_ROW
                    : (const uint8_t*)SC;
                sk_cp_small<S::S_ROW>(Ss + i * S::S_ROW, src, ok);
            }
        }
    };

    /* Prologue */
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load_stage(s, s);
        sk_commit();
    }

    for (int kt = 0; kt < KT; kt++) {
        sk_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk % STAGES, nk);
            sk_commit();
        }
        const uint8_t* base = sk_smem + (kt % STAGES) * S::STAGE;
        const half* As = (const half*)base;
        const uint8_t* Ws = base + S::A_BYTES;

#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
                sk_ldsm_x4(a[mt], As + (mt * 16 + (lane & 15)) * S::ASTR
                                     + ks * 16 + (lane >> 4) * 8);

            uint32_t b[NT][2];
            if constexpr (WT == VV_SKINNY_F16) {
                /* Two n8 tiles per ldmatrix: rows n..n+7 at k and k+8, then
                 * rows n+8..n+15 -- b0/b1 of one tile, then of the next. */
                const half* Wh = (const half*)Ws;
                constexpr int WSTRH = S::WSTR / 2;
#pragma unroll
                for (int np = 0; np < NT / 2; np++) {
                    uint32_t r[4];
                    const int row = warp * NT * 8 + np * 16 + (lane & 7)
                                  + ((lane >> 4) << 3);
                    sk_ldsm_x4(r, Wh + row * WSTRH + ks * 16
                                     + ((lane >> 3) & 1) * 8);
                    b[2 * np][0] = r[0];     b[2 * np][1] = r[1];
                    b[2 * np + 1][0] = r[2]; b[2 * np + 1][1] = r[3];
                }
            } else if constexpr (WT == VV_SKINNY_INT8) {
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int r = warp * NT * 8 + nt * 8 + g;
                    const uint8_t* wr = Ws + r * S::WSTR + ks * 16 + 2 * t;
                    b[nt][0] = sk_i8_pair(*(const uint16_t*)wr, srow[nt]);
                    b[nt][1] = sk_i8_pair(*(const uint16_t*)(wr + 8), srow[nt]);
                }
            } else {
                const half* Ss = (const half*)(Ws + S::W_BYTES);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int r = warp * NT * 8 + nt * 8 + g;
                    const uint8_t* wr = Ws + r * S::WSTR + ks * 8;
                    const float sc = __half2float(Ss[r * (BK / 64) + (ks >> 2)]);
                    b[nt][0] = sk_nf4_pair(wr[t], sc, lut);
                    b[nt][1] = sk_nf4_pair(wr[4 + t], sc, lut);
                }
            }

#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
                    sk_mma(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
        }
    }
    sk_wait<0>();

    /* Epilogue: what the tile kernel stores, then what vv_bias_add_dev adds. */
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int col = n0 + warp * NT * 8 + nt * 8 + 2 * t;
        if (col >= N) continue;
        const float b0 = bias ? __half2float(bias[col]) : 0.0f;
        const float b1 = bias ? __half2float(bias[col + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                const int row = mt * 16 + g + hr * 8;
                if (row >= M) continue;
                half h0 = __float2half(alpha * acc[mt][nt][2 * hr + 0]);
                half h1 = __float2half(alpha * acc[mt][nt][2 * hr + 1]);
                if (bias) {
                    h0 = __float2half(__half2float(h0) + b0);
                    h1 = __float2half(__half2float(h1) + b1);
                }
                *(half2*)(Y + (size_t)row * N + col) = __halves2half2(h0, h1);
            }
        }
    }
}

/* ─── Host side ──────────────────────────────────────────────────────────── */

namespace {

/*
 * One tile shape per format and row tile count, swept on a 3090 over the
 * launches of a 7B layer (tests/test_skinny.c, VV_SKINNY_BENCH=3): NT n8
 * tiles per warp (64 output columns per block), BK columns of k per stage,
 * ST stages.
 *   FP16  BK 128, 3 stages: bound by the weight stream, which wants long
 *         runs per row; at 64 rows that is 104 KB, so BK 64 there.
 *   INT8  BK 128, 2 stages: two blocks per SM hide the dequant better than
 *         deeper stages in one.
 *   NF4   BK 128 (the scales go 4 bytes a row), 2 stages: up to three
 *         blocks per SM.
 * Wider or narrower tiles moved no projection by more than ~10%.
 */
template <int WT, int MT> struct sk_tile;
template <int MT> struct sk_tile<VV_SKINNY_F16, MT> {
    static constexpr int NT = 2, BK = MT < 4 ? 128 : 64, ST = 3;
};
template <int MT> struct sk_tile<VV_SKINNY_INT8, MT> {
    static constexpr int NT = 2, BK = 128, ST = 2;
};
template <int MT> struct sk_tile<VV_SKINNY_NF4, MT> {
    static constexpr int NT = 2, BK = 128, ST = 2;
};

/** @brief Device facts, cached per thread and device (a thread binds one). */
struct sk_dev_info {
    int dev;
    int major;
    int smem_optin;                   /* dynamic shared memory per block */
    bool set_up[3][4];                /* that limit raised for the kernel */
};

sk_dev_info* sk_info(void) {
    static thread_local sk_dev_info info = { -1, 0, 0, { { false } } };
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return NULL;
    if (dev != info.dev) {
        memset(&info, 0, sizeof(info));
        info.dev = dev;
        if (cudaDeviceGetAttribute(&info.major,
                cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&info.smem_optin,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
            info.dev = -1;
            return NULL;
        }
    }
    return &info;
}

/** @brief VV_SKINNY=0 sends every caller back to the dequant + tile path. */
bool sk_enabled(void) {
    static thread_local int v = -2;
    if (v == -2) {
        const char* e = getenv("VV_SKINNY");
        v = (e && e[0]) ? atoi(e) : 1;
    }
    return v != 0;
}

template <int WT, int MT>
vv_status_t sk_launch(sk_dev_info* info, const half* A, sk_segs segs,
                      int n_segs, int M, int K, float alpha, cudaStream_t st)
{
    typedef sk_tile<WT, MT> T;
    typedef sk_shape<WT, MT, T::NT, T::BK> S;
    constexpr int SMEM = S::STAGE * T::ST + S::LUT_BYTES;
    if (K % T::BK != 0 || SMEM > info->smem_optin) return VV_ERR_UNSUPPORTED;
    auto kern = skinny_kernel<WT, MT, T::NT, T::BK, T::ST>;
    if (!info->set_up[WT][MT - 1]) {
        if (cudaFuncSetAttribute(kern,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)
                != cudaSuccess)
            return VV_ERR_CUDA_LAUNCH;
        info->set_up[WT][MT - 1] = true;
    }
    int blocks = 0;
    for (int i = 0; i < SK_MAX_SEGS; i++) {
        if (i < n_segs) {
            segs.first_block[i] = blocks;
            blocks += (segs.n[i] + S::BN - 1) / S::BN;
        } else {
            segs.first_block[i] = INT_MAX;
        }
    }
    kern<<<blocks, SK_THREADS, SMEM, st>>>(A, segs, M, K, alpha);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

template <int WT>
vv_status_t sk_launch_wt(sk_dev_info* info, const half* A,
                         const sk_segs& segs, int n_segs, int M, int K,
                         float alpha, cudaStream_t st)
{
    switch ((M + 15) / 16) {
        case 1:  return sk_launch<WT, 1>(info, A, segs, n_segs, M, K, alpha, st);
        case 2:  return sk_launch<WT, 2>(info, A, segs, n_segs, M, K, alpha, st);
        case 3:  return sk_launch<WT, 3>(info, A, segs, n_segs, M, K, alpha, st);
        default: return sk_launch<WT, 4>(info, A, segs, n_segs, M, K, alpha, st);
    }
}

} /* namespace */

extern "C" {

vv_status_t vv_skinny_linear_dev(const void* A, int format,
                                 const vv_skinny_proj_t* projs, int n_proj,
                                 int M, int K, float alpha, void* stream)
{
    if (!A || !projs) return VV_ERR_NULL_PTR;
    if (n_proj < 1 || n_proj > SK_MAX_SEGS ||
        format < VV_SKINNY_F16 || format > VV_SKINNY_NF4)
        return VV_ERR_INVALID_ARG;
    if (M < 1 || M > VV_SKINNY_M_MAX || K <= 0 ||
        ((uintptr_t)A & 15) != 0 || !sk_enabled())
        return VV_ERR_UNSUPPORTED;

    sk_segs segs;
    memset(&segs, 0, sizeof(segs));
    for (int i = 0; i < n_proj; i++) {
        const vv_skinny_proj_t* p = &projs[i];
        if (!p->w || !p->y || (format != VV_SKINNY_F16 && !p->scales))
            return VV_ERR_NULL_PTR;
        /* Rows in 16-byte chunks (the BK checks cover K), half2 stores,
         * the NF4 scales in 4-byte cp.async pieces. */
        if (p->N <= 0 || (p->N & 7) != 0 || ((uintptr_t)p->w & 15) != 0 ||
            ((uintptr_t)p->y & 3) != 0 ||
            (format != VV_SKINNY_F16 && ((uintptr_t)p->scales & 3) != 0))
            return VV_ERR_UNSUPPORTED;
        segs.w[i] = (const uint8_t*)p->w;
        segs.sc[i] = p->scales;
        segs.bias[i] = (const half*)p->bias;
        segs.y[i] = (half*)p->y;
        segs.n[i] = p->N;
    }

    sk_dev_info* info = sk_info();
    if (!info || info->major < 8) return VV_ERR_UNSUPPORTED;
    const half* a = (const half*)A;
    cudaStream_t st = (cudaStream_t)stream;
    switch (format) {
        case VV_SKINNY_F16:
            return sk_launch_wt<VV_SKINNY_F16>(info, a, segs, n_proj, M, K,
                                               alpha, st);
        case VV_SKINNY_INT8:
            return sk_launch_wt<VV_SKINNY_INT8>(info, a, segs, n_proj, M, K,
                                                alpha, st);
        default:
            return sk_launch_wt<VV_SKINNY_NF4>(info, a, segs, n_proj, M, K,
                                               alpha, st);
    }
}

} /* extern "C" */
