/**
 * @file vae_kernels.cu
 * @brief Batched, streaming Conv-VAE encoder kernels.
 *
 * Every activation is channel-first and packed along time: the items of one
 * launch (segments of a file, several requests, a stateless window) sit side
 * by side in one [C][T_total] buffer with no padding between them. Anything
 * pointwise in time — RMSNorm, the FFN GEMMs, GELU, the residual adds — then
 * runs over the concatenated axis without knowing items exist. Only the
 * convolutions look back in time, and they read the item boundaries from a
 * descriptor table passed by value (vv_vae_conv_desc_t): where the item's
 * input and output start, how many columns of left context sit in its
 * streaming state, and how many to leave there for the next chunk.
 *
 * No kernel's arithmetic depends on where an item sits in the packed axis or
 * on how a stream was cut into chunks: every output sums its products in a
 * fixed order, and an FP16 rounding happens at every point the old
 * multi-kernel sequence stored an intermediate. So batching, chunking and
 * fusing change no bit of the output, which is what lets the tests demand
 * bit-exact equality rather than a tolerance. The one change against the
 * kernels this replaces is the downsample: a tensor-core GEMM over im2col
 * tiles instead of a scalar loop, which sums in WMMA order. Against the
 * PyTorch reference that moves test30's latents from 0.149 / 0.261 % to
 * 0.147 / 0.265 % relative error (acoustic / semantic); transcripts are
 * unchanged.
 *
 * All indexing into packed buffers is 64-bit: a packed stage-0 FFN hidden
 * buffer passes INT_MAX elements at about eleven 60 s segments.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#include "vibevoice/device.h"

using namespace nvcuda;

#define VAE_THREADS 256

static inline vv_status_t launch_status(void) {
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* ─── Convolution (stem, downsample, head) ──────────────────────────────── */

/**
 * One output (channel, time) per thread, accumulated over (in channel, tap)
 * in that order — the order conv1d_kernel used, so results are identical.
 * Position p of the item's input is column p of [context ++ input]; context
 * columns come from the state (zeros when it is fresh), columns past the end
 * are the ceil-alignment padding of a final chunk and contribute nothing.
 */
template <bool TRANSPOSED>
__global__ void __launch_bounds__(VAE_THREADS) vae_conv_kernel(
    vv_vae_conv_desc_t d, const half* __restrict__ x, int64_t ld_in,
    const half* __restrict__ w, const half* __restrict__ b,
    half* __restrict__ y, int64_t ld_out, int in_ch, int k, int stride)
{
    const vv_vae_conv_item_t& it = d.it[blockIdx.z];
    const int oc = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= it.out_len) return;

    const half* xb = it.in_ptr ? (const half*)it.in_ptr : x + it.in_off;
    const half* tl = (const half*)it.tail_in;
    const int have = it.have;
    const int total = have + it.in_len;
    const half* wr = w + (size_t)oc * in_ch * k;

    float sum = 0.0f;
    for (int ic = 0; ic < in_ch; ic++) {
        for (int kk = 0; kk < k; kk++) {
            const int p = t * stride + kk;
            if (p < total) {
                float xv;
                if (p < have)
                    xv = tl ? __half2float(tl[(size_t)ic * d.cap + p]) : 0.0f;
                else
                    xv = __half2float(xb[(size_t)ic * ld_in + (p - have)]);
                const float wv = __half2float(wr[ic * k + kk]);
                sum += wv * xv;
            }
        }
    }
    if (b) sum += __half2float(b[oc]);

    if (TRANSPOSED) {
        if (t >= it.skip)
            ((half*)it.out_ptr)[(size_t)(t - it.skip) * it.out_ld + oc] =
                __float2half(sum);
    } else {
        y[(size_t)oc * ld_out + it.out_off + t] = __float2half(sum);
    }
}

/**
 * The next chunk's left context: the last `keep` columns of
 * [context ++ input], written to the state's other buffer so that reading
 * the old context and writing the new one never touch the same memory.
 */
__global__ void vae_tail_kernel(vv_vae_conv_desc_t d, const half* __restrict__ x,
                                int64_t ld_in, int in_ch)
{
    const vv_vae_conv_item_t& it = d.it[blockIdx.y];
    if (!it.tail_out || it.keep <= 0) return;
    const int keep = it.keep;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_ch * keep) return;
    const int c = idx / keep, j = idx - c * keep;
    const int p = it.have + it.in_len - keep + j;

    half v;
    if (p < it.have) {
        const half* tl = (const half*)it.tail_in;
        v = tl ? tl[(size_t)c * d.cap + p] : __float2half(0.0f);
    } else {
        const half* xb = it.in_ptr ? (const half*)it.in_ptr : x + it.in_off;
        v = xb[(size_t)c * ld_in + (p - it.have)];
    }
    ((half*)it.tail_out)[(size_t)c * d.cap + j] = v;
}

/**
 * The downsample convolutions as a GEMM: column q of the tile is packed
 * output column p0 + q, row ic * k + kk the input it reads through tap kk —
 * the weight's own [out][in][k] order, so the weight is the GEMM's A as it
 * lies. Positions are resolved exactly as vae_conv_kernel resolves them:
 * context from the state (zeros when fresh), then the item's input, then the
 * zero right padding of a final chunk.
 */
__global__ void vae_im2col_kernel(vv_vae_conv_desc_t d, const half* __restrict__ x,
                                  int64_t ld_in, int k, int stride,
                                  int64_t p0, int pc, half* __restrict__ col,
                                  int64_t ldcol)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= pc) return;
    const int kr = blockIdx.y;
    const int ic = kr / k, kk = kr - ic * k;
    const int64_t p = p0 + q;
    int i = 0;
    while (i + 1 < d.n && p >= d.it[i + 1].out_off) i++;
    const vv_vae_conv_item_t& it = d.it[i];
    const int t = (int)(p - it.out_off);
    const int pos = t * stride + kk;
    half v = __float2half(0.0f);
    if (t < it.out_len && pos < it.have + it.in_len) {
        if (pos < it.have) {
            const half* tl = (const half*)it.tail_in;
            if (tl) v = tl[(size_t)ic * d.cap + pos];
        } else {
            const half* xb = it.in_ptr ? (const half*)it.in_ptr : x + it.in_off;
            v = xb[(size_t)ic * ld_in + (pos - it.have)];
        }
    }
    col[(size_t)kr * ldcol + q] = v;
}

/* ─── RMSNorm statistics ───────────────────────────────────────────────── */

/**
 * 1/rms of every packed column, then of every context column an item carries
 * in its state (at T + item * cap + j). One thread per column summing the
 * channels in order, exactly as rmsnorm_channel_first_kernel did.
 */
__global__ void vae_rms_kernel(const half* __restrict__ x, int64_t ld,
                               int64_t T, int C, float eps,
                               float* __restrict__ rinv,
                               vv_vae_conv_desc_t d)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const half* col;
    int64_t stride;
    int64_t dst;
    if (i < T) {
        col = x + i;
        stride = ld;
        dst = i;
    } else {
        const int64_t j = i - T;
        if (d.cap <= 0) return;
        const int item = (int)(j / d.cap), q = (int)(j - (int64_t)item * d.cap);
        if (item >= d.n) return;
        const vv_vae_conv_item_t& it = d.it[item];
        if (!it.tail_in || q >= it.have) return;
        col = (const half*)it.tail_in + q;
        stride = d.cap;
        dst = i;
    }
    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        const float v = __half2float(col[(size_t)c * stride]);
        sum_sq += v * v;
    }
    rinv[dst] = rsqrtf(sum_sq / (float)C + eps);
}

/* ─── Mixer: RMSNorm -> depthwise conv -> gamma residual, fused ─────────── */

#define MIX_TT 64      /* output columns per block */
#define MIX_CG 32      /* channels per block       */
#define MIX_MAXK 16

/**
 * One block owns MIX_CG channels by MIX_TT columns of one item. It first
 * normalises the k-1 column halo plus its own columns into shared memory
 * (rounded to FP16, as the separate norm kernel stored them), then runs the
 * depthwise taps and the gamma residual. Output goes to a second buffer: a
 * block's halo is another block's output, so updating in place would race.
 * The halo left of the item comes from the state, which holds the raw
 * residual columns; normalising them here gives the same bits the old path
 * cached, because the norm of a column depends on nothing else.
 */
__global__ void __launch_bounds__(VAE_THREADS) vae_mixer_kernel(
    vv_vae_conv_desc_t d, const half* __restrict__ xin, half* __restrict__ xout,
    int64_t ld, const float* __restrict__ rinv, int64_t T_total,
    const half* __restrict__ nw, const half* __restrict__ cw,
    const half* __restrict__ cb, const half* __restrict__ gamma,
    int C, int k)
{
    const int item = blockIdx.z;
    const vv_vae_conv_item_t& it = d.it[item];
    const int t0 = blockIdx.x * MIX_TT;
    if (t0 >= it.in_len) return;
    const int c0 = blockIdx.y * MIX_CG;
    const int H = k - 1;
    const int W = MIX_TT + H;

    __shared__ half nv[MIX_CG][MIX_TT + MIX_MAXK];

    const half* tl = (const half*)it.tail_in;
    for (int e = threadIdx.x; e < MIX_CG * W; e += blockDim.x) {
        const int cl = e / W, q = e - cl * W;
        const int c = c0 + cl;
        half out = __float2half(0.0f);
        if (c < C) {
            const int p = t0 - H + q;
            if (p >= 0) {
                if (p < it.in_len) {
                    const int64_t col = it.in_off + p;
                    const float v = __half2float(xin[(size_t)c * ld + col]);
                    const float r = rinv[col];
                    out = __float2half(v * r * __half2float(nw[c]));
                }
            } else {
                const int tp = it.have + p;
                if (tl && tp >= 0) {
                    const float v = __half2float(tl[(size_t)c * d.cap + tp]);
                    const float r = rinv[T_total + (int64_t)item * d.cap + tp];
                    out = __float2half(v * r * __half2float(nw[c]));
                }
            }
        }
        nv[cl][q] = out;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < MIX_CG * MIX_TT; e += blockDim.x) {
        const int cl = e / MIX_TT, tt = e - cl * MIX_TT;
        const int c = c0 + cl, t = t0 + tt;
        if (c >= C || t >= it.in_len) continue;
        float sum = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            const float wv = __half2float(cw[c * k + kk]);
            sum += wv * __half2float(nv[cl][tt + kk]);
        }
        if (cb) sum += __half2float(cb[c]);
        const float yv = __half2float(__float2half(sum));
        const size_t off = (size_t)c * ld + it.in_off + t;
        const float xv = __half2float(xin[off]);
        xout[off] = gamma ? __float2half(xv + yv * __half2float(gamma[c]))
                          : __float2half(xv + yv);
    }
}

/* ─── GEMM with epilogues ───────────────────────────────────────────────── */

/*
 * The same 128x128x32 block tile, 8 warps and double buffering as the plain
 * kernels in gemm.cu, so every output element sums its K products in the
 * same order and the results match bit for bit. What is new: row strides,
 * a column window, an RMSNorm applied while B is staged, and epilogues that
 * do what used to be two or three more kernels.
 */
#define BM        128
#define BN        128
#define BK        32
#define WARPS     8
#define THREADS   (WARPS * 32)
#define WARP_M    64
#define WARP_N    32
#define FRAG_M    (WARP_M / 16)
#define FRAG_N    (WARP_N / 16)
#define KSTEPS    (BK / 16)
#define LDK       (BK + 8)
#define LDN       (BN + 8)

template <int ROWS>
__device__ __forceinline__ void vae_load_k_major(
    const half* __restrict__ src, int64_t lds, int row_base, int row_limit,
    int K, int k0, bool k_aligned, half* __restrict__ dst)
{
    const int tid = threadIdx.x;
    #pragma unroll
    for (int p = 0; p < ROWS / 64; ++p) {
        const int r  = (tid >> 2) + p * 64;
        const int c  = (tid & 3) * 8;
        const int gr = row_base + r;
        half* dd = dst + r * LDK + c;
        if (gr < row_limit && k_aligned && k0 + c + 8 <= K) {
            *(float4*)dd = *(const float4*)(src + (size_t)gr * lds + k0 + c);
        } else {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int kk = k0 + c + t;
                dd[t] = (gr < row_limit && kk < K)
                      ? src[(size_t)gr * lds + kk] : __float2half(0.f);
            }
        }
    }
}

/**
 * B of the NN shape, [K][P] with row stride ldb. With `nw` set, each value
 * is normalised on the way in: half(x * rinv[col] * nw[row]), the value the
 * standalone RMSNorm would have stored.
 */
__device__ __forceinline__ void vae_load_n_major(
    const half* __restrict__ src, int64_t ldb, int k0, int K,
    int col_base, int P, bool p_aligned,
    const float* __restrict__ rinv, const half* __restrict__ nw,
    half* __restrict__ dst)
{
    const int tid = threadIdx.x;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r  = (tid >> 4) + p * 16;
        const int c  = (tid & 15) * 8;
        const int gk = k0 + r;
        const int gc = col_base + c;
        half* dd = dst + r * LDN + c;
        if (gk < K && p_aligned && gc + 8 <= P) {
            float4 raw = *(const float4*)(src + (size_t)gk * ldb + gc);
            if (nw) {
                half* h = (half*)&raw;
                const float wn = __half2float(nw[gk]);
                #pragma unroll
                for (int t = 0; t < 8; ++t)
                    h[t] = __float2half(__half2float(h[t]) * rinv[gc + t] * wn);
            }
            *(float4*)dd = raw;
        } else {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                half v = __float2half(0.f);
                if (gk < K && gc + t < P) {
                    v = src[(size_t)gk * ldb + gc + t];
                    if (nw)
                        v = __float2half(__half2float(v) * rinv[gc + t]
                                         * __half2float(nw[gk]));
                }
                dd[t] = v;
            }
        }
    }
}

/** @brief Epilogue of the FFN GEMMs. */
enum {
    VAE_EPI_BIAS_GELU  = 0,   /* h = gelu(half(half(acc) + b))              */
    VAE_EPI_BIAS_RESID = 1,   /* x = half(x + half(half(acc) + b) * gamma)  */
    VAE_EPI_BIAS       = 2    /* y = half(acc + b), one rounding (convs)    */
};

__device__ __forceinline__ float vae_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

template <int EPI>
__global__ __launch_bounds__(THREADS) void vae_gemm_nn_kernel(
    const half* __restrict__ A, const half* __restrict__ B, int64_t ldb,
    half* __restrict__ C, int64_t ldc, int M, int K, int P,
    const half* __restrict__ bias, const half* __restrict__ gamma,
    const float* __restrict__ rinv, const half* __restrict__ nw)
{
    extern __shared__ char smem_raw[];
    half* As = (half*)smem_raw;
    half* Bs = As + 2 * BM * LDK;

    const int warp   = threadIdx.x >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const bool k_aligned = (K & 7) == 0;
    const bool p_aligned = (ldb & 7) == 0 && (((uintptr_t)B) & 15) == 0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

    vae_load_k_major<BM>(A, K, row_base, M, K, 0, k_aligned, As);
    vae_load_n_major(B, ldb, 0, K, col_base, P, p_aligned, rinv, nw, Bs);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const half* Ac = As + stage * BM * LDK;
        const half* Bc = Bs + stage * BK * LDN;
        #pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[FRAG_M];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[FRAG_N];
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
                wmma::load_matrix_sync(af[i],
                    Ac + (warp_m * WARP_M + i * 16) * LDK + ks * 16, LDK);
            #pragma unroll
            for (int j = 0; j < FRAG_N; ++j)
                wmma::load_matrix_sync(bf[j],
                    Bc + (ks * 16) * LDN + warp_n * WARP_N + j * 16, LDN);
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
                #pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        if (k0 + BK < K) {
            const int nx = stage ^ 1;
            vae_load_k_major<BM>(A, K, row_base, M, K, k0 + BK, k_aligned,
                                 As + nx * BM * LDK);
            vae_load_n_major(B, ldb, k0 + BK, K, col_base, P, p_aligned,
                             rinv, nw, Bs + nx * BK * LDN);
            __syncthreads();
            stage = nx;
        }
    }

    __syncthreads();
    float* stg = (float*)smem_raw + warp * 256;
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::store_matrix_sync(stg, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int row0 = row_base + warp_m * WARP_M + i * 16;
            const int col0 = col_base + warp_n * WARP_N + j * 16;
            for (int e = lane; e < 256; e += 32) {
                const int r = row0 + (e >> 4);
                const int c = col0 + (e & 15);
                if (r < M && c < P) {
                    const size_t off = (size_t)r * ldc + c;
                    if (EPI == VAE_EPI_BIAS) {
                        const float b = bias ? __half2float(bias[r]) : 0.0f;
                        C[off] = __float2half(stg[e] + b);
                        continue;
                    }
                    float v = __half2float(__float2half(stg[e]));
                    if (bias)
                        v = __half2float(__float2half(v + __half2float(bias[r])));
                    if (EPI == VAE_EPI_BIAS_GELU) {
                        C[off] = __float2half(vae_gelu(v));
                    } else {
                        const float xv = __half2float(C[off]);
                        C[off] = gamma
                            ? __float2half(xv + v * __half2float(gamma[r]))
                            : __float2half(xv + v);
                    }
                }
            }
            __syncwarp();
        }
    }
}

/* ─── TN GEMM for the speech connectors ─────────────────────────────────── */

/**
 * The pipeline used to add the two connector outputs on the host in FP32 and
 * truncate the sum to FP16, flushing subnormals. The fused epilogue keeps
 * that rounding so the prompt embedding stays bit-identical.
 */
__device__ __forceinline__ half vae_trunc_half(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t  e    = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    const uint32_t frac = (u >> 13) & 0x3FFu;
    unsigned short h;
    if (e <= 0)        h = (unsigned short)sign;
    else if (e >= 31)  h = (unsigned short)(sign | 0x7C00u);
    else               h = (unsigned short)(sign | ((uint32_t)e << 10) | frac);
    return __ushort_as_half(h);
}

__global__ __launch_bounds__(THREADS) void vae_gemm_tn_kernel(
    const half* __restrict__ A, int64_t lda, const half* __restrict__ B,
    half* __restrict__ C, int64_t ldc, int M, int N, int K,
    const half* __restrict__ bias, vv_vae_rows_t rows)
{
    extern __shared__ char smem_raw[];
    half* As = (half*)smem_raw;
    half* Bs = As + 2 * BM * LDK;

    const int warp   = threadIdx.x >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const bool k_aligned = (K & 7) == 0 && (lda & 7) == 0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

    vae_load_k_major<BM>(A, lda, row_base, M, K, 0, k_aligned, As);
    vae_load_k_major<BN>(B, K, col_base, N, K, 0, (K & 7) == 0, Bs);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const half* Ac = As + stage * BM * LDK;
        const half* Bc = Bs + stage * BN * LDK;
        #pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[FRAG_M];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[FRAG_N];
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
                wmma::load_matrix_sync(af[i],
                    Ac + (warp_m * WARP_M + i * 16) * LDK + ks * 16, LDK);
            #pragma unroll
            for (int j = 0; j < FRAG_N; ++j)
                wmma::load_matrix_sync(bf[j],
                    Bc + (warp_n * WARP_N + j * 16) * LDK + ks * 16, LDK);
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
                #pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        if (k0 + BK < K) {
            const int nx = stage ^ 1;
            vae_load_k_major<BM>(A, lda, row_base, M, K, k0 + BK, k_aligned,
                                 As + nx * BM * LDK);
            vae_load_k_major<BN>(B, K, col_base, N, K, k0 + BK, (K & 7) == 0,
                                 Bs + nx * BN * LDK);
            __syncthreads();
            stage = nx;
        }
    }

    __syncthreads();
    float* stg = (float*)smem_raw + warp * 256;
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::store_matrix_sync(stg, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int row0 = row_base + warp_m * WARP_M + i * 16;
            const int col0 = col_base + warp_n * WARP_N + j * 16;
            for (int e = lane; e < 256; e += 32) {
                const int r = row0 + (e >> 4);
                const int c = col0 + (e & 15);
                if (r >= M || c >= N) continue;
                half y = __float2half(stg[e]);
                if (bias)
                    y = __float2half(__half2float(y) + __half2float(bias[c]));
                if (rows.mode == VV_VAE_ROWS_PLAIN) {
                    C[(size_t)r * ldc + c] = y;
                    continue;
                }
                int s = 0;
                while (s + 1 < rows.n && r >= rows.seg[s + 1].row0) s++;
                half* dst = (half*)rows.seg[s].dst
                          + (size_t)(r - rows.seg[s].row0) * rows.seg[s].ld + c;
                if (rows.mode == VV_VAE_ROWS_STORE)
                    *dst = y;
                else if (rows.mode == VV_VAE_ROWS_STORE_TRUNC)
                    *dst = vae_trunc_half(__half2float(y));
                else
                    *dst = vae_trunc_half(__half2float(*dst) + __half2float(y));
            }
            __syncwarp();
        }
    }
}

__global__ void vae_f32_to_f16_kernel(const float* __restrict__ in,
                                      half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

/* ─── Public entry points ───────────────────────────────────────────────── */

extern "C" {

static int desc_max(const vv_vae_conv_desc_t* d, bool out) {
    int m = 0;
    for (int i = 0; i < d->n; i++) {
        const int v = out ? d->it[i].out_len : d->it[i].in_len;
        if (v > m) m = v;
    }
    return m;
}

vv_status_t vv_vae_conv_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, const void* w, const void* b,
                            void* y, int64_t ld_out, int in_ch, int out_ch,
                            int k, int stride, bool transposed, void* stream) {
    if (!d || !w) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->n > VV_VAE_MAX_ITEMS) return VV_ERR_INVALID_ARG;
    const int m = desc_max(d, true);
    if (m <= 0) return VV_OK;
    dim3 grid((m + VAE_THREADS - 1) / VAE_THREADS, out_ch, d->n);
    if (transposed)
        vae_conv_kernel<true><<<grid, VAE_THREADS, 0, (cudaStream_t)stream>>>(
            *d, (const half*)x, ld_in, (const half*)w, (const half*)b,
            (half*)y, ld_out, in_ch, k, stride);
    else
        vae_conv_kernel<false><<<grid, VAE_THREADS, 0, (cudaStream_t)stream>>>(
            *d, (const half*)x, ld_in, (const half*)w, (const half*)b,
            (half*)y, ld_out, in_ch, k, stride);
    return launch_status();
}

vv_status_t vv_vae_tail_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, int in_ch, void* stream) {
    if (!d) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->cap <= 0) return VV_OK;
    const int n = in_ch * d->cap;
    dim3 grid((n + VAE_THREADS - 1) / VAE_THREADS, d->n);
    vae_tail_kernel<<<grid, VAE_THREADS, 0, (cudaStream_t)stream>>>(
        *d, (const half*)x, ld_in, in_ch);
    return launch_status();
}

vv_status_t vv_vae_rms_dev(const void* x, int64_t ld, int64_t T, int C,
                           float eps, float* rinv,
                           const vv_vae_conv_desc_t* d, void* stream) {
    if (!x || !rinv) return VV_ERR_NULL_PTR;
    vv_vae_conv_desc_t none;
    none.n = 0;
    none.cap = 0;
    const vv_vae_conv_desc_t* dd = d ? d : &none;
    const int64_t n = T + (int64_t)dd->n * (dd->cap > 0 ? dd->cap : 0);
    if (n <= 0) return VV_OK;
    const int64_t blocks = (n + 127) / 128;
    vae_rms_kernel<<<(unsigned)blocks, 128, 0, (cudaStream_t)stream>>>(
        (const half*)x, ld, T, C, eps, rinv, *dd);
    return launch_status();
}

vv_status_t vv_vae_mixer_dev(const vv_vae_conv_desc_t* d, const void* xin,
                             void* xout, int64_t ld, const float* rinv,
                             int64_t T_total, const void* norm_w,
                             const void* conv_w, const void* conv_b,
                             const void* gamma, int C, int k, void* stream) {
    if (!d || !xin || !xout || !rinv || !norm_w || !conv_w)
        return VV_ERR_NULL_PTR;
    if (k < 1 || k > MIX_MAXK || d->n <= 0 || d->n > VV_VAE_MAX_ITEMS)
        return VV_ERR_INVALID_ARG;
    const int m = desc_max(d, false);
    if (m <= 0) return VV_OK;
    dim3 grid((m + MIX_TT - 1) / MIX_TT, (C + MIX_CG - 1) / MIX_CG, d->n);
    vae_mixer_kernel<<<grid, VAE_THREADS, 0, (cudaStream_t)stream>>>(
        *d, (const half*)xin, (half*)xout, ld, rinv, T_total,
        (const half*)norm_w, (const half*)conv_w, (const half*)conv_b,
        (const half*)gamma, C, k);
    return launch_status();
}

vv_status_t vv_vae_gemm_nn_dev(int epilogue, const void* A, const void* B,
                               int64_t ldb, void* C, int64_t ldc,
                               int M, int K, int P, const void* bias,
                               const void* gamma, const float* rinv,
                               const void* norm_w, void* stream) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || P <= 0) return VV_ERR_INVALID_ARG;
    if ((norm_w != NULL) != (rinv != NULL)) return VV_ERR_INVALID_ARG;
    const size_t shmem = (size_t)2 * (BM * LDK + BK * LDN) * sizeof(half);
    dim3 grid((P + BN - 1) / BN, (M + BM - 1) / BM);
    if (epilogue == VAE_EPI_BIAS_GELU)
        vae_gemm_nn_kernel<VAE_EPI_BIAS_GELU>
            <<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
            (const half*)A, (const half*)B, ldb, (half*)C, ldc, M, K, P,
            (const half*)bias, (const half*)gamma, rinv, (const half*)norm_w);
    else if (epilogue == VAE_EPI_BIAS_RESID)
        vae_gemm_nn_kernel<VAE_EPI_BIAS_RESID>
            <<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
            (const half*)A, (const half*)B, ldb, (half*)C, ldc, M, K, P,
            (const half*)bias, (const half*)gamma, rinv, (const half*)norm_w);
    else if (epilogue == VAE_EPI_BIAS)
        vae_gemm_nn_kernel<VAE_EPI_BIAS>
            <<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
            (const half*)A, (const half*)B, ldb, (half*)C, ldc, M, K, P,
            (const half*)bias, (const half*)gamma, rinv, (const half*)norm_w);
    else
        return VV_ERR_INVALID_ARG;
    return launch_status();
}

vv_status_t vv_vae_im2col_dev(const vv_vae_conv_desc_t* d, const void* x,
                              int64_t ld_in, int in_ch, int k, int stride,
                              int64_t p0, int pc, void* col, int64_t ldcol,
                              void* stream) {
    if (!d || !col) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->n > VV_VAE_MAX_ITEMS || in_ch * k > 65535)
        return VV_ERR_INVALID_ARG;
    if (pc <= 0) return VV_OK;
    dim3 grid((pc + VAE_THREADS - 1) / VAE_THREADS, in_ch * k);
    vae_im2col_kernel<<<grid, VAE_THREADS, 0, (cudaStream_t)stream>>>(
        *d, (const half*)x, ld_in, k, stride, p0, pc, (half*)col, ldcol);
    return launch_status();
}

vv_status_t vv_vae_gemm_tn_dev(const void* A, int64_t lda, const void* B,
                               void* C, int64_t ldc, int M, int N, int K,
                               const void* bias, const vv_vae_rows_t* rows,
                               void* stream) {
    if (!A || !B) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    vv_vae_rows_t plain;
    plain.mode = VV_VAE_ROWS_PLAIN;
    plain.n = 0;
    const vv_vae_rows_t* r = rows ? rows : &plain;
    if (r->mode == VV_VAE_ROWS_PLAIN && !C) return VV_ERR_NULL_PTR;
    if (r->mode != VV_VAE_ROWS_PLAIN &&
        (r->n <= 0 || r->n > VV_VAE_MAX_ITEMS || r->seg[0].row0 != 0))
        return VV_ERR_INVALID_ARG;
    const size_t shmem = (size_t)2 * (BM + BN) * LDK * sizeof(half);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    vae_gemm_tn_kernel<<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
        (const half*)A, lda, (const half*)B, (half*)C, ldc, M, N, K,
        (const half*)bias, *r);
    return launch_status();
}

vv_status_t vv_vae_f32_to_f16_dev(const float* in, void* out, int64_t n,
                                  void* stream) {
    if (!in || !out) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    const int64_t blocks = (n + VAE_THREADS - 1) / VAE_THREADS;
    vae_f32_to_f16_kernel<<<(unsigned)blocks, VAE_THREADS, 0,
                            (cudaStream_t)stream>>>(in, (half*)out, n);
    return launch_status();
}

} /* extern "C" */
