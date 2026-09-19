/**
 * @file gemm.cu
 * @brief FP16 GEMM on tensor cores — no cuBLAS.
 *
 * The runtime ships as a single self-contained binary, so it cannot depend on
 * libcublas (600 MB of redistributables, or a toolkit install on the target
 * machine). These kernels use the WMMA API directly, which compiles into the
 * same `mma.sync` instructions cuBLAS issues on Ampere.
 *
 * Two shapes are needed:
 *   TN:  C[M,N] = A[M,K] @ B[N,K]^T   — a linear layer, weight in [out, in]
 *   NN:  C[M,P] = A[M,K] @ B[K,P]     — Conv-VAE FFN, weight @ activations
 *
 * Both use a 128x128x32 block tile split over 8 warps (each warp owns a 64x32
 * quadrant = 4x2 WMMA fragments), double-buffered through shared memory so the
 * global loads for step k+1 are in flight while step k is on the tensor cores.
 * A TN call of 9..64 rows goes to gemm_skinny.cu instead, which gives the same
 * bits without padding the rows to 128 or leaving most SMs idle.
 *
 * IMPORTANT: All linear-layer callers pass weight in [N, K] layout
 * (i.e., [out_features, in_features], the standard PyTorch convention).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#include "vibevoice/device.h"

using namespace nvcuda;

extern "C" {
    #include "vibevoice/types.h"

    vv_status_t vv_dequant_nf4_dev(
        const uint8_t* packed, const void* scales_fp16,
        void* output_fp16, int n_elements, int block_size, void* stream);
}

/* ─── Tile geometry ──────────────────────────────────────────────────────── */

#define BM        128
#define BN        128
#define BK        32
#define WARPS     8
#define THREADS   (WARPS * 32)

#define WARP_M    64            /* BM / 2 warp rows */
#define WARP_N    32            /* BN / 4 warp cols */
#define FRAG_M    (WARP_M / 16) /* 4 */
#define FRAG_N    (WARP_N / 16) /* 2 */
#define KSTEPS    (BK / 16)     /* 2 */

/* Row stride of a K-major shared tile. +8 keeps the 16-byte stores aligned
 * (80 bytes per row) while breaking the power-of-two bank pattern. */
#define LDK       (BK + 8)      /* 40 */
/* Row stride of an N-major shared tile (NN kernel's B). */
#define LDN       (BN + 8)      /* 136 */

/* ─── Shared-memory tile loaders ─────────────────────────────────────────── */

/**
 * Load a [rows x BK] slab of a K-major matrix into shared memory.
 * Four threads cover one row (8 halves each), two passes cover 128 rows.
 */
template <int ROWS>
__device__ __forceinline__ void load_k_major(
    const half* __restrict__ src, int row_base, int row_limit,
    int K, int k0, bool k_aligned, half* __restrict__ dst)
{
    const int tid = threadIdx.x;
    #pragma unroll
    for (int p = 0; p < ROWS / 64; ++p) {
        const int r  = (tid >> 2) + p * 64;
        const int c  = (tid & 3) * 8;
        const int gr = row_base + r;
        half* d = dst + r * LDK + c;

        if (gr < row_limit && k_aligned && k0 + c + 8 <= K) {
            *(float4*)d = *(const float4*)(src + (size_t)gr * K + k0 + c);
        } else {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int kk = k0 + c + t;
                d[t] = (gr < row_limit && kk < K)
                     ? src[(size_t)gr * K + kk] : __float2half(0.f);
            }
        }
    }
}

/**
 * Load a [BK x BN] slab of an N-major matrix (B of the NN shape) into shared.
 * Sixteen threads cover one row of 128 halves; two passes cover 32 rows.
 */
__device__ __forceinline__ void load_n_major(
    const half* __restrict__ src, int k0, int K,
    int col_base, int P, bool p_aligned, half* __restrict__ dst)
{
    const int tid = threadIdx.x;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r  = (tid >> 4) + p * 16;
        const int c  = (tid & 15) * 8;
        const int gk = k0 + r;
        const int gc = col_base + c;
        half* d = dst + r * LDN + c;

        if (gk < K && p_aligned && gc + 8 <= P) {
            *(float4*)d = *(const float4*)(src + (size_t)gk * P + gc);
        } else {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                d[t] = (gk < K && gc + t < P)
                     ? src[(size_t)gk * P + gc + t] : __float2half(0.f);
            }
        }
    }
}

/* ─── Epilogue ───────────────────────────────────────────────────────────── */

/**
 * Spill one 16x16 accumulator through shared memory and write it out with
 * bounds checks. WMMA fragment layout is implementation-defined, so going
 * through store_matrix_sync is the only portable way to index the results.
 */
__device__ __forceinline__ void store_frag(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& c_frag,
    float* stage, half* __restrict__ C, int ldc,
    int row0, int col0, int rows, int cols,
    float alpha, float beta)
{
    wmma::store_matrix_sync(stage, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = lane; i < 256; i += 32) {
        const int r = row0 + (i >> 4);
        const int c = col0 + (i & 15);
        if (r < rows && c < cols) {
            const size_t off = (size_t)r * ldc + c;
            float v = alpha * stage[i];
            if (beta != 0.f) v += beta * __half2float(C[off]);
            C[off] = __float2half(v);
        }
    }
    __syncwarp();
}

/* ─── TN kernel: C[M,N] = A[M,K] @ B[N,K]^T ──────────────────────────────── */

__global__ __launch_bounds__(THREADS) void gemm_tn_kernel(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K,
    float alpha, float beta)
{
    extern __shared__ char smem_raw[];
    half* As = (half*)smem_raw;                 /* [2][BM][LDK] */
    half* Bs = As + 2 * BM * LDK;               /* [2][BN][LDK] */

    const int warp   = threadIdx.x >> 5;
    const int warp_m = warp >> 2;               /* 0..1 */
    const int warp_n = warp & 3;                /* 0..3 */
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const bool k_aligned = (K & 7) == 0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

    load_k_major<BM>(A, row_base, M, K, 0, k_aligned, As);
    load_k_major<BN>(B, col_base, N, K, 0, k_aligned, Bs);
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
            load_k_major<BM>(A, row_base, M, K, k0 + BK, k_aligned,
                             As + nx * BM * LDK);
            load_k_major<BN>(B, col_base, N, K, k0 + BK, k_aligned,
                             Bs + nx * BN * LDK);
            __syncthreads();
            stage = nx;
        }
    }

    __syncthreads();
    float* stg = (float*)smem_raw + warp * 256;
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            store_frag(acc[i][j], stg, C, N,
                       row_base + warp_m * WARP_M + i * 16,
                       col_base + warp_n * WARP_N + j * 16,
                       M, N, alpha, beta);
}

/* ─── NN kernel: C[M,P] = A[M,K] @ B[K,P] ────────────────────────────────── */

__global__ __launch_bounds__(THREADS) void gemm_nn_kernel(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int K, int P,
    float alpha, float beta)
{
    extern __shared__ char smem_raw[];
    half* As = (half*)smem_raw;                 /* [2][BM][LDK] */
    half* Bs = As + 2 * BM * LDK;               /* [2][BK][LDN] */

    const int warp   = threadIdx.x >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const bool k_aligned = (K & 7) == 0;
    const bool p_aligned = (P & 7) == 0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

    load_k_major<BM>(A, row_base, M, K, 0, k_aligned, As);
    load_n_major(B, 0, K, col_base, P, p_aligned, Bs);
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
            load_k_major<BM>(A, row_base, M, K, k0 + BK, k_aligned,
                             As + nx * BM * LDK);
            load_n_major(B, k0 + BK, K, col_base, P, p_aligned,
                         Bs + nx * BK * LDN);
            __syncthreads();
            stage = nx;
        }
    }

    __syncthreads();
    float* stg = (float*)smem_raw + warp * 256;
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            store_frag(acc[i][j], stg, C, P,
                       row_base + warp_m * WARP_M + i * 16,
                       col_base + warp_n * WARP_N + j * 16,
                       M, P, alpha, beta);
}

/* ─── Skinny-M path ──────────────────────────────────────────────────────── */

/**
 * For M <= 8 the tensor-core tile is 94% padding, so fall back to a
 * bandwidth-bound dot-product kernel: one warp streams one row of B and
 * accumulates against every row of A, which stays resident in L2.
 */
#define SMALL_M_MAX 8

__global__ void gemm_tn_skinny_kernel(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K,
    float alpha, float beta)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (n >= N) return;

    float acc[SMALL_M_MAX];
    #pragma unroll
    for (int m = 0; m < SMALL_M_MAX; ++m) acc[m] = 0.f;

    const half* brow = B + (size_t)n * K;
    const int k_vec = (K / 8) * 8;

    for (int k = lane * 8; k < k_vec; k += 32 * 8) {
        const float4 bv = *(const float4*)(brow + k);
        const half2* bh = (const half2*)&bv;
        for (int m = 0; m < M; ++m) {
            const float4 av = *(const float4*)(A + (size_t)m * K + k);
            const half2* ah = (const half2*)&av;
            float s = 0.f;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float2 a = __half22float2(ah[t]);
                const float2 b = __half22float2(bh[t]);
                s += a.x * b.x + a.y * b.y;
            }
            acc[m] += s;
        }
    }
    for (int k = k_vec + lane; k < K; k += 32) {
        const float b = __half2float(brow[k]);
        for (int m = 0; m < M; ++m)
            acc[m] += __half2float(A[(size_t)m * K + k]) * b;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        for (int m = 0; m < M; ++m)
            acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);

    if (lane == 0) {
        for (int m = 0; m < M; ++m) {
            const size_t o = (size_t)m * N + n;
            float v = alpha * acc[m];
            if (beta != 0.f) v += beta * __half2float(C[o]);
            C[o] = __float2half(v);
        }
    }
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

extern "C" {

/**
 * @brief FP16 GEMM on tensor cores:  C = alpha * A @ B^T + beta * C
 *
 * A: [M, K] FP16 row-major
 * B: [N, K] FP16 row-major — weight in [out_features, in_features] layout
 * C: [M, N] FP16 row-major
 *
 * FP32 accumulation throughout (critical at K = 3584).
 */
vv_status_t vv_gemm_fp16_dev(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    void* stream)
{
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;

    cudaStream_t st = (cudaStream_t)stream;

    if (M <= SMALL_M_MAX && (K & 7) == 0) {
        const int warps_per_block = 8;
        dim3 grid((N + warps_per_block - 1) / warps_per_block);
        gemm_tn_skinny_kernel<<<grid, warps_per_block * 32, 0, st>>>(
            (const half*)A, (const half*)B, (half*)C, M, N, K, alpha, beta);
        return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
    }

    /* 9..64 rows: the small-M kernels, same bits as the tile below. */
    if (M <= VV_SKINNY_M_MAX && beta == 0.0f) {
        const vv_skinny_proj_t p = { B, NULL, NULL, C, N };
        const vv_status_t s = vv_skinny_linear_dev(A, VV_SKINNY_F16, &p, 1,
                                                   M, K, alpha, stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
    }

    return vv_gemm_fp16_tile_dev(A, B, C, M, N, K, alpha, beta, stream);
}

/**
 * @brief The 128x128 tile kernel whatever M:  C = alpha * A @ B^T + beta * C
 *
 * Above 64 rows this is what vv_gemm_fp16_dev runs; below, the small-M
 * kernels are held to it bit for bit (tests/test_skinny.c).
 */
vv_status_t vv_gemm_fp16_tile_dev(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    void* stream)
{
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;

    const size_t shmem = (size_t)2 * (BM + BN) * LDK * sizeof(half);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tn_kernel<<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
        (const half*)A, (const half*)B, (half*)C, M, N, K, alpha, beta);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief FP16 GEMM on tensor cores:  C = alpha * A @ B + beta * C
 *
 * A: [M, K] FP16 row-major
 * B: [K, P] FP16 row-major
 * C: [M, P] FP16 row-major
 *
 * Used by the Conv-VAE FFN: output[out_ch, len] = weight[out_ch, in_ch] @ input[in_ch, len].
 */
vv_status_t vv_gemm_fp16_nn_dev(
    const void* A, const void* B, void* C,
    int M, int K, int P,
    float alpha, float beta,
    void* stream)
{
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || P <= 0) return VV_ERR_INVALID_ARG;

    const size_t shmem = (size_t)2 * (BM * LDK + BK * LDN) * sizeof(half);
    dim3 grid((P + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_nn_kernel<<<grid, THREADS, shmem, (cudaStream_t)stream>>>(
        (const half*)A, (const half*)B, (half*)C, M, K, P, alpha, beta);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief NF4 dequant + FP16 GEMM (9..64 rows: vv_skinny_linear_dev).
 *
 * input:  [M, K] FP16
 * weight: [N, K/2] uint8 (NF4 packed, row-major [out, in/2])
 * scales: [N*K/block_size] FP16
 * output: [M, N] FP16
 * temp:   [N, K] FP16 (pre-allocated scratch for the dequantized weight)
 */
vv_status_t vv_nf4_gemm_dev(
    const void* input_fp16,
    const uint8_t* weight_packed,
    const void* weight_scales_fp16,
    void* output_fp16,
    void* temp_weight_fp16,
    int M, int N, int K,
    int block_size,
    void* stream)
{
    if (!input_fp16 || !weight_packed || !weight_scales_fp16 ||
        !output_fp16) {
        return VV_ERR_NULL_PTR;
    }

    /* 9..64 rows: dequantized in registers, no scratch, same bits. */
    if (M > SMALL_M_MAX && M <= VV_SKINNY_M_MAX && block_size == 64) {
        const vv_skinny_proj_t p = { weight_packed, weight_scales_fp16, NULL,
                                     output_fp16, N };
        vv_status_t s = vv_skinny_linear_dev(input_fp16, VV_SKINNY_NF4, &p, 1,
                                             M, K, 1.0f, stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
    }
    if (!temp_weight_fp16) return VV_ERR_NULL_PTR;

    vv_status_t s = vv_dequant_nf4_dev(
        weight_packed, weight_scales_fp16,
        temp_weight_fp16, N * K, block_size, stream);
    if (s != VV_OK) return s;

    return vv_gemm_fp16_dev(input_fp16, temp_weight_fp16, output_fp16,
                             M, N, K, 1.0f, 0.0f, stream);
}

/** @brief Retained for API compatibility; there is no handle to release. */
void vv_gemm_cleanup(void) {}

} /* extern "C" */
