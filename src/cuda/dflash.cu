/**
 * @file dflash.cu
 * @brief Kernels of speculative decoding with a DFlash 2 drafter: the
 *        drafter's own ops, and the multi-row LM head and argmax that check
 *        a drafted block in one pass.
 *
 * The head and the argmax here compute every row exactly as the one-row
 * kernels of the decode step do (lm_head.cu): the same lanes take the same
 * elements, the same FMA chain, the same butterfly, the same tie rule. The
 * weight is read once for all rows instead of once per row. That is what
 * lets a verified block produce the tokens a token-by-token decode would
 * have, bit for bit (docs/DFLASH.md).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

#include "vibevoice/device.h"

/* ─── Multi-row LM head ──────────────────────────────────────────────────── */

#define LMR_WARPS 8

__device__ __forceinline__ float lmr_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

/* A float4 of a weight read once: past L1, so the x rows every warp of the
 * SM rereads stay there. */
__device__ __forceinline__ float4 lmr_ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return *(const float4*)&v;
}

/*
 * logits[m][n] = sum_k W[n,k] * x[m][k] for m < M: lm_head_gemv_kernel's
 * arithmetic per row (lane l takes k = 8l + 256i, two FMAs per half2 in
 * order, then the xor butterfly). W streams past L1 (lmr_ld_stream) so
 * the x rows stay there for every warp of the SM -- read through L1, W
 * evicted them and the x traffic went to L2, 1.75x the one-row time for 8
 * rows. A warp takes R weight rows, so each x float4 is also loaded and
 * converted once for R of them.
 */
template <int MX, int R>
__global__ void __launch_bounds__(LMR_WARPS * 32)
lm_head_rows_kernel(const half* __restrict__ x, const half* __restrict__ W,
                    float* __restrict__ logits, int M, int V, int K)
{
    const int row0 = (blockIdx.x * LMR_WARPS + threadIdx.y) * R;
    if (row0 >= V) return;
    const int lane = threadIdx.x;

    float acc[R][MX];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MX; m++) acc[r][m] = 0.0f;

    int k = lane * 8;
    for (; k + 8 <= K; k += 256) {
        float2 a[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
            float4 wv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (row0 + r < V) wv = lmr_ld_stream(W + (size_t)(row0 + r) * K + k);
            const half2* wh = (const half2*)&wv;
#pragma unroll
            for (int b = 0; b < 4; b++) a[r][b] = __half22float2(wh[b]);
        }
#pragma unroll
        for (int m = 0; m < MX; m++) {
            if (m < M) {
                const float4 xv = __ldg((const float4*)(x + (size_t)m * K + k));
                const half2* xh = (const half2*)&xv;
                float2 c[4];
#pragma unroll
                for (int b = 0; b < 4; b++) c[b] = __half22float2(xh[b]);
#pragma unroll
                for (int r = 0; r < R; r++)
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        acc[r][m] = fmaf(a[r][b].x, c[b].x, acc[r][m]);
                        acc[r][m] = fmaf(a[r][b].y, c[b].y, acc[r][m]);
                    }
            }
        }
    }
    for (int t = (K & ~255) + lane; t < K; t += 32) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float w = row0 + r < V
                          ? __half2float(W[(size_t)(row0 + r) * K + t]) : 0.0f;
#pragma unroll
            for (int m = 0; m < MX; m++)
                if (m < M)
                    acc[r][m] = fmaf(w, __half2float(x[(size_t)m * K + t]),
                                     acc[r][m]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MX; m++) {
            if (m < M) {
                const float v = lmr_warp_sum(acc[r][m]);
                if (lane == 0 && row0 + r < V) logits[(size_t)m * V + row0 + r] = v;
            }
        }
}

/* ─── Multi-row argmax ───────────────────────────────────────────────────── */

#define AMR_BLK   256
#define AMR_PARTS 256   /* = VV_ARGMAX_PARTIALS, as vv_argmax_dev */

/* argmax_partial_kernel of lm_head.cu, row blockIdx.y. */
__global__ void argmax_rows_partial_kernel(const float* __restrict__ logits,
                                           int V, float* __restrict__ pmax,
                                           int* __restrict__ pidx)
{
    __shared__ float sv[AMR_BLK];
    __shared__ int   si[AMR_BLK];
    const float* l = logits + (size_t)blockIdx.y * V;
    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int i = blockIdx.x * AMR_BLK + tid; i < V; i += gridDim.x * AMR_BLK) {
        const float v = l[i];
        if (v > best) { best = v; bidx = i; }
    }
    sv[tid] = best;
    si[tid] = bidx;
    __syncthreads();
    for (int s = AMR_BLK / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        pmax[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = sv[0];
        pidx[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = si[0];
    }
}

/* argmax_final_kernel of lm_head.cu, row blockIdx.x. */
__global__ void argmax_rows_final_kernel(const float* __restrict__ pmax,
                                         const int* __restrict__ pidx, int n,
                                         int* __restrict__ out_token)
{
    __shared__ float sv[AMR_BLK];
    __shared__ int   si[AMR_BLK];
    const float* pm = pmax + (size_t)blockIdx.x * n;
    const int* pi = pidx + (size_t)blockIdx.x * n;
    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int i = tid; i < n; i += AMR_BLK)
        if (pm[i] > best) { best = pm[i]; bidx = pi[i]; }
    sv[tid] = best;
    si[tid] = bidx;
    __syncthreads();
    for (int s = AMR_BLK / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out_token[blockIdx.x] = si[0];
}

/* ─── Top-k per row ──────────────────────────────────────────────────────── */

#define TK_THREADS 256

__device__ __forceinline__ bool tk_better(float v, int i, float w, int j) {
    return v > w || (v == w && i < j);
}

/* Largest `k` (<= KK) of each row, descending, ties to the lower id. */
template <int KK>
__global__ void __launch_bounds__(TK_THREADS)
topk_rows_kernel(const float* __restrict__ logits, int V, int k,
                 float* __restrict__ out_v, int* __restrict__ out_i)
{
    const float* l = logits + (size_t)blockIdx.x * V;
    float lv[KK];
    int li[KK];
#pragma unroll
    for (int j = 0; j < KK; j++) { lv[j] = -FLT_MAX; li[j] = INT_MAX; }
    for (int i = threadIdx.x; i < V; i += TK_THREADS) {
        const float v = l[i];
        if (!tk_better(v, i, lv[KK - 1], li[KK - 1])) continue;
        lv[KK - 1] = v;
        li[KK - 1] = i;
#pragma unroll
        for (int j = KK - 1; j > 0; j--) {
            if (tk_better(lv[j], li[j], lv[j - 1], li[j - 1])) {
                const float tv = lv[j]; lv[j] = lv[j - 1]; lv[j - 1] = tv;
                const int ti = li[j]; li[j] = li[j - 1]; li[j - 1] = ti;
            }
        }
    }
    /* Merge: k rounds, each takes the best head over all threads' lists. */
    __shared__ float hv[TK_THREADS];
    __shared__ int   hi[TK_THREADS];
    __shared__ int   ht[TK_THREADS];
    int head = 0;
    for (int r = 0; r < k; r++) {
        float v = -FLT_MAX;
        int i = INT_MAX;
#pragma unroll
        for (int j = 0; j < KK; j++)
            if (j == head) { v = lv[j]; i = li[j]; }
        hv[threadIdx.x] = v;
        hi[threadIdx.x] = i;
        ht[threadIdx.x] = threadIdx.x;
        __syncthreads();
        for (int s = TK_THREADS / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s &&
                tk_better(hv[threadIdx.x + s], hi[threadIdx.x + s],
                          hv[threadIdx.x], hi[threadIdx.x])) {
                hv[threadIdx.x] = hv[threadIdx.x + s];
                hi[threadIdx.x] = hi[threadIdx.x + s];
                ht[threadIdx.x] = ht[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_v[(size_t)blockIdx.x * k + r] = hv[0];
            out_i[(size_t)blockIdx.x * k + r] = hi[0];
        }
        const int winner = ht[0];
        __syncthreads();
        if (threadIdx.x == winner) head++;
    }
}

/* ─── The drafter's dynamic convolution ──────────────────────────────────── */

/*
 * out[t][c] = sum_{o < taps, t - o in t's block} (base[o][c] +
 *             dyn[t][o * G + c / gs]) * x[t - o][c]
 * in the order the trainer adds them (base term, then the dynamic one).
 */
__global__ void dflash_conv_kernel(const half* __restrict__ x,
                                   const half* __restrict__ dyn, int dyn_ld,
                                   const float* __restrict__ base,
                                   half* __restrict__ out, int H, int taps,
                                   int gs, int block)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= H) return;
    const int t0 = (t / block) * block;
    const int G = H / gs;
    float acc = 0.0f;
    for (int o = 0; o < taps && t - o >= t0; o++) {
        const float xv = __half2float(x[(size_t)(t - o) * H + c]);
        acc += base[(size_t)o * H + c] * xv;
        acc += __half2float(dyn[(size_t)t * dyn_ld + o * G + c / gs]) * xv;
    }
    out[(size_t)t * H + c] = __float2half(acc);
}

/* ─── The drafter's path selector ────────────────────────────────────────── */

#define WALK_MAX_K 32

/*
 * One block, `rank` threads (a multiple of 32, <= 1024). Position t scores
 * candidate j as unary[t][j] + <pred[prev] * hproj[t], succ[cand[t][j]]>
 * and moves on from the best (lowest j on a tie).
 */
__global__ void dflash_walk_kernel(const half* __restrict__ hproj,
                                   const float* __restrict__ unary,
                                   const int* __restrict__ cand,
                                   const half* __restrict__ pred_cb,
                                   const half* __restrict__ succ_cb,
                                   const int* __restrict__ anchor,
                                   int M, int k, int rank,
                                   int* __restrict__ out)
{
    __shared__ float red[32][WALK_MAX_K];
    __shared__ int prev_s;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    if (tid == 0) prev_s = *anchor;
    __syncthreads();
    for (int t = 0; t < M; t++) {
        const int prev = prev_s;
        const float g = __half2float(pred_cb[(size_t)prev * rank + tid]) *
                        __half2float(hproj[(size_t)t * rank + tid]);
        for (int j = 0; j < k; j++) {
            const int cj = cand[(size_t)t * k + j];
            float p = g * __half2float(succ_cb[(size_t)cj * rank + tid]);
            p = lmr_warp_sum(p);
            if (lane == 0) red[warp][j] = p;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            float bv = -FLT_MAX;
            for (int j = 0; j < k; j++) {
                float s = unary[(size_t)t * k + j];
                for (int w = 0; w < n_warps; w++) s += red[w][j];
                if (s > bv) { bv = s; best = j; }
            }
            prev_s = cand[(size_t)t * k + best];
            out[t] = prev_s;
        }
        __syncthreads();
    }
}

/* ─── The drafter's FP32 residual stream ─────────────────────────────────── */

/* y = x * rsqrt(mean(x^2) + eps) * w, one block per row, x in FP32. */
__global__ void rmsnorm_f32_kernel(const float* __restrict__ x,
                                   const half* __restrict__ w,
                                   half* __restrict__ y, int H, float eps)
{
    const float* xr = x + (size_t)blockIdx.x * H;
    half* yr = y + (size_t)blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float red[32];
    ss = lmr_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        v = lmr_warp_sum(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float)H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yr[i] = __float2half(xr[i] * r * __half2float(w[i]));
}

/* h[i] += scale * y[i] (y FP16), or h[i] = scale * y[i] with `set`. */
__global__ void add_scaled_kernel(float* __restrict__ h,
                                  const half* __restrict__ y, float scale,
                                  int n, int set)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = scale * __half2float(y[i]);
    h[i] = set ? v : h[i] + v;
}

/* ids[0] = *anchor, the rest the mask token. */
__global__ void dflash_block_ids_kernel(const int* __restrict__ anchor,
                                       int mask, int n, int* __restrict__ ids)
{
    const int i = threadIdx.x;
    if (i < n) ids[i] = i == 0 ? *anchor : mask;
}

/*
 * The block verified: row i of `post` is the target's token after position
 * p + i. accepted = the leading drafts it agrees with; out[0..a] = those
 * drafts then the target's own token after them; *n_out = a + 1.
 */
__global__ void dflash_accept_kernel(const int* __restrict__ draft,
                                     const int* __restrict__ post, int n_draft,
                                     int* __restrict__ out,
                                     int* __restrict__ n_out)
{
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a < n_draft && draft[a] == post[a]) {
        out[a] = draft[a];
        a++;
    }
    out[a] = post[a];
    *n_out = a + 1;
}

__global__ void gather_i32_kernel(const int* __restrict__ map,
                                  int* __restrict__ idx, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[i] = map[idx[i]];
}

extern "C" {

vv_status_t vv_lm_head_rows_dev(const void* x, const void* W, void* logits,
                                int M, int V, int K, void* stream)
{
    if (!x || !W || !logits) return VV_ERR_NULL_PTR;
    if (M <= 0 || M > 16 || V <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    const dim3 block(32, LMR_WARPS);
    cudaStream_t st = (cudaStream_t)stream;
    /* 3090, 8 rows: R = 1 took 1.75x the one-row head, R = 4 1.08x. */
    if (M <= 8) {
        constexpr int R = 4;
        const dim3 grid((V + LMR_WARPS * R - 1) / (LMR_WARPS * R));
        lm_head_rows_kernel<8, R><<<grid, block, 0, st>>>(
            (const half*)x, (const half*)W, (float*)logits, M, V, K);
    } else {
        constexpr int R = 2;
        const dim3 grid((V + LMR_WARPS * R - 1) / (LMR_WARPS * R));
        lm_head_rows_kernel<16, R><<<grid, block, 0, st>>>(
            (const half*)x, (const half*)W, (float*)logits, M, V, K);
    }
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

size_t vv_argmax_rows_scratch_bytes(int M) {
    return (size_t)M * AMR_PARTS * (sizeof(float) + sizeof(int));
}

vv_status_t vv_argmax_rows_dev(const void* logits, int M, int V,
                               void* scratch, int32_t* out_tokens,
                               void* stream)
{
    if (!logits || !scratch || !out_tokens) return VV_ERR_NULL_PTR;
    if (M <= 0) return VV_ERR_INVALID_ARG;
    float* pm = (float*)scratch;
    int* pi = (int*)(pm + (size_t)M * AMR_PARTS);
    cudaStream_t st = (cudaStream_t)stream;
    argmax_rows_partial_kernel<<<dim3(AMR_PARTS, M), AMR_BLK, 0, st>>>(
        (const float*)logits, V, pm, pi);
    argmax_rows_final_kernel<<<M, AMR_BLK, 0, st>>>(pm, pi, AMR_PARTS,
                                                    (int*)out_tokens);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_topk_rows_dev(const float* logits, int M, int V, int k,
                             float* out_vals, int32_t* out_ids, void* stream)
{
    if (!logits || !out_vals || !out_ids) return VV_ERR_NULL_PTR;
    if (M <= 0 || k <= 0 || k > 32 || V < k) return VV_ERR_INVALID_ARG;
    cudaStream_t st = (cudaStream_t)stream;
    if (k <= 16)
        topk_rows_kernel<16><<<M, TK_THREADS, 0, st>>>(logits, V, k, out_vals,
                                                      (int*)out_ids);
    else
        topk_rows_kernel<32><<<M, TK_THREADS, 0, st>>>(logits, V, k, out_vals,
                                                      (int*)out_ids);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dflash_conv_dev(const void* x, const void* dyn, int dyn_ld,
                               const float* base, void* out, int M, int H,
                               int taps, int group_size, int block,
                               void* stream)
{
    if (!x || !dyn || !base || !out) return VV_ERR_NULL_PTR;
    if (M <= 0 || H <= 0 || taps <= 0 || group_size <= 0 ||
        H % group_size || block <= 0)
        return VV_ERR_INVALID_ARG;
    const int threads = 256;
    const dim3 grid((H + threads - 1) / threads, M);
    dflash_conv_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const half*)x, (const half*)dyn, dyn_ld, base, (half*)out, H, taps,
        group_size, block);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dflash_walk_dev(const void* hproj, const float* unary,
                               const int32_t* cand, const void* pred_cb,
                               const void* succ_cb, const int32_t* anchor,
                               int M, int k, int rank, int32_t* out,
                               void* stream)
{
    if (!hproj || !unary || !cand || !pred_cb || !succ_cb || !anchor || !out)
        return VV_ERR_NULL_PTR;
    if (M <= 0 || k <= 0 || k > WALK_MAX_K || rank < 32 || rank > 1024 ||
        rank % 32)
        return VV_ERR_INVALID_ARG;
    dflash_walk_kernel<<<1, rank, 0, (cudaStream_t)stream>>>(
        (const half*)hproj, unary, (const int*)cand, (const half*)pred_cb,
        (const half*)succ_cb, (const int*)anchor, M, k, rank, (int*)out);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_rmsnorm_f32in_dev(const float* x, const void* w, void* y,
                                 int rows, int H, float eps, void* stream)
{
    if (!x || !w || !y) return VV_ERR_NULL_PTR;
    if (rows <= 0 || H <= 0) return VV_ERR_INVALID_ARG;
    rmsnorm_f32_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        x, (const half*)w, (half*)y, H, eps);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_add_scaled_f16_dev(float* h, const void* y, float scale, int n,
                                  bool set, void* stream)
{
    if (!h || !y) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    const int threads = 256;
    add_scaled_kernel<<<(n + threads - 1) / threads, threads, 0,
                        (cudaStream_t)stream>>>(h, (const half*)y, scale, n,
                                                set ? 1 : 0);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gather_i32_dev(const int32_t* map, int32_t* idx, int n,
                              void* stream)
{
    if (!map || !idx) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    gather_i32_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const int*)map, (int*)idx, n);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dflash_block_ids_dev(const int32_t* anchor, int mask_id,
                                    int n, int32_t* ids, void* stream)
{
    if (!anchor || !ids) return VV_ERR_NULL_PTR;
    if (n <= 0 || n > 1024) return VV_ERR_INVALID_ARG;
    dflash_block_ids_kernel<<<1, n, 0, (cudaStream_t)stream>>>(
        (const int*)anchor, mask_id, n, (int*)ids);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dflash_accept_dev(const int32_t* draft, const int32_t* post,
                                 int n_draft, int32_t* out, int32_t* n_out,
                                 void* stream)
{
    if (!draft || !post || !out || !n_out) return VV_ERR_NULL_PTR;
    dflash_accept_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
        (const int*)draft, (const int*)post, n_draft, (int*)out, (int*)n_out);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
