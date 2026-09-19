/**
 * @file gemv_i8.cu
 * @brief Decode-shaped int8 linear layers (M <= 8) on dp4a.
 *
 * Laid out like the W4A16 GEMV (w4a16.cu), which it has to keep up with:
 * LPR lanes share one output row (16..128; above 32 the row spans several
 * warps and the partial sums meet in shared memory), each lane walks the
 * row's 16-byte chunks with a stride of LPR, and up to eight activation rows
 * share every weight load, so the weights are read once however many tokens
 * are in flight. Weights are streamed past L1 (read once per token, no
 * reuse). Several projections of the same input -- q/k/v, gate/up -- go in
 * one launch: the grid is the concatenation of their row blocks, so the
 * 512-row k and v ride along behind q instead of each paying a ramp-up.
 *
 * W8A8: a chunk is 16 int8 weights, four dp4a per activation row; the dot
 * product is an exact int32 until the one multiply by sx * sw at the end.
 *
 * W4A8: a chunk is 32 weights (16 bytes), all inside one group because G is
 * a multiple of 32. The weights are the W4A16 GPU layout, and the
 * activations come in the matching VV_Q8_W4_GEMV order, so the nibbles are
 * used as unsigned bytes (0..15) straight from the packed word:
 * `w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F` each meet one activation word
 * with no shuffle. The zero point is applied once per 32 weights through the
 * quantizer's xsum:
 *
 *   sum (q - z) x = sum q x - z * sum x
 *
 * which keeps the inner loop to one AND, one shift and two dp4a per 8
 * weights per row.
 *
 * Launch geometry depends only on the shapes, and nothing is read from the
 * host, so the decode step that contains these stays capturable.
 */

#include "i8_common.cuh"
#include "vibevoice/device.h"

#include <limits.h>
#include <string.h>

#define I8V_THREADS 128
#define I8V_MAX_SEGS 3

/** @brief The projections of one launch, passed by value. */
struct i8v_segs {
    const uint4*  w[I8V_MAX_SEGS];
    const float*  sw[I8V_MAX_SEGS];      /* W8: FP32 [N]                    */
    const half2*  sz[I8V_MAX_SEGS];      /* W4: {scale, zero} [N][K/G]      */
    const half*   bias[I8V_MAX_SEGS];
    const half*   res[I8V_MAX_SEGS];
    void*         y[I8V_MAX_SEGS];
    int           n[I8V_MAX_SEGS];
    int           first_block[I8V_MAX_SEGS];  /* unused: INT_MAX */
};

__device__ __forceinline__ uint4 i8v_ld_stream(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <bool W4, int LPR, int MR>
__global__ void __launch_bounds__(I8V_THREADS)
i8_gemv_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx,
               const int32_t* __restrict__ xsum, const i8v_segs segs,
               int K, int G, int y_f32)
{
    constexpr int GPB = I8V_THREADS / LPR;           /* rows per block */
    constexpr int WIDTH = LPR < 32 ? LPR : 32;

    /* Which projection this block serves: selects, not indexed loads, so
     * the parameter arrays stay in the constant bank. */
    const int bx = (int)blockIdx.x;
    const int seg = bx >= segs.first_block[2] ? 2
                  : (bx >= segs.first_block[1] ? 1 : 0);
#define I8V_SEL(f) (seg == 0 ? segs.f[0] : (seg == 1 ? segs.f[1] : segs.f[2]))
    const uint4* w = I8V_SEL(w);
    const int N = I8V_SEL(n);
    const int blk = bx - (seg == 0 ? 0 : (seg == 1 ? segs.first_block[1]
                                                   : segs.first_block[2]));
    const int row = blk * GPB + (int)threadIdx.x / LPR;
    const int lr = (int)threadIdx.x % LPR;

    /* A chunk is 16 bytes: 16 int8 weights, or 32 int4 ones. */
    const int nch = row < N ? (W4 ? K >> 5 : K >> 4) : 0;
    const int rr = row < N ? row : N - 1;
    const uint4* wrow = w + (size_t)rr * (W4 ? K >> 5 : K >> 4);
    const int ng = W4 ? K / G : 1;
    const half2* szrow = W4 ? I8V_SEL(sz) + (size_t)rr * ng : NULL;

    float facc[MR];
    int iacc[MR];
#pragma unroll
    for (int m = 0; m < MR; m++) { facc[m] = 0.0f; iacc[m] = 0; }

    for (int c = lr; c < nch; c += LPR) {
        const uint4 wv = i8v_ld_stream(wrow + c);
        if (W4) {
            const float2 sz = __half22float2(szrow[(c << 5) / G]);
            const int z = (int)sz.y;
            const uint32_t wa[4] = { wv.x, wv.y, wv.z, wv.w };
            uint32_t ev[4], od[4];
#pragma unroll
            for (int t = 0; t < 4; t++) {
                ev[t] = wa[t] & 0x0F0F0F0Fu;          /* slots 0, 2, 4, 6 */
                od[t] = (wa[t] >> 4) & 0x0F0F0F0Fu;   /* slots 1, 3, 5, 7 */
            }
#pragma unroll
            for (int m = 0; m < MR; m++) {
                /* VV_Q8_W4_GEMV: bytes 8t..8t+3 are word t's even slots,
                 * 8t+4..8t+7 its odd ones. */
                const int4* xp = (const int4*)(xq + (size_t)m * K + (c << 5));
                const int4 xa = __ldg(xp), xb = __ldg(xp + 1);
                int dot = 0;
                dot = i8_dp4a((int)ev[0], xa.x, dot);
                dot = i8_dp4a((int)od[0], xa.y, dot);
                dot = i8_dp4a((int)ev[1], xa.z, dot);
                dot = i8_dp4a((int)od[1], xa.w, dot);
                dot = i8_dp4a((int)ev[2], xb.x, dot);
                dot = i8_dp4a((int)od[2], xb.y, dot);
                dot = i8_dp4a((int)ev[3], xb.z, dot);
                dot = i8_dp4a((int)od[3], xb.w, dot);
                const int xs = __ldg(xsum + (size_t)m * (K >> 5) + c);
                facc[m] = fmaf(sz.x, (float)(dot - z * xs), facc[m]);
            }
        } else {
#pragma unroll
            for (int m = 0; m < MR; m++) {
                const int4 xv = __ldg((const int4*)(xq + (size_t)m * K) + c);
                int a = iacc[m];
                a = i8_dp4a((int)wv.x, xv.x, a);
                a = i8_dp4a((int)wv.y, xv.y, a);
                a = i8_dp4a((int)wv.z, xv.z, a);
                a = i8_dp4a((int)wv.w, xv.w, a);
                iacc[m] = a;
            }
        }
    }

    /* W8 sums are exact int32 until here; reduce them as integers. */
#pragma unroll
    for (int m = 0; m < MR; m++) {
#pragma unroll
        for (int off = WIDTH / 2; off > 0; off >>= 1) {
            if (W4) facc[m] += __shfl_xor_sync(0xFFFFFFFFu, facc[m], off, WIDTH);
            else    iacc[m] += __shfl_xor_sync(0xFFFFFFFFu, iacc[m], off, WIDTH);
        }
    }
    if (LPR > 32) {
        constexpr int WPG = LPR > 32 ? LPR / 32 : 1;  /* warps per row */
        __shared__ int red[MR][I8V_THREADS / 32];
        const int warp = (int)threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int m = 0; m < MR; m++)
                red[m][warp] = W4 ? __float_as_int(facc[m]) : iacc[m];
        }
        __syncthreads();
        if (lr == 0) {
#pragma unroll
            for (int m = 0; m < MR; m++) {
                float f = 0.0f;
                int v = 0;
#pragma unroll
                for (int i = 0; i < WPG; i++) {
                    if (W4) f += __int_as_float(red[m][warp + i]);
                    else    v += red[m][warp + i];
                }
                facc[m] = f;
                iacc[m] = v;
            }
        }
    }
    if (lr != 0 || row >= N) return;

    const float swn = W4 ? 1.0f : I8V_SEL(sw)[row];
    const half* bias = I8V_SEL(bias);
    const half* res = I8V_SEL(res);
    void* y = I8V_SEL(y);
#undef I8V_SEL
    const float b = bias ? __half2float(bias[row]) : 0.0f;
#pragma unroll
    for (int m = 0; m < MR; m++) {
        const size_t idx = (size_t)m * N + row;
        float v = W4 ? facc[m] * sx[m] : (float)iacc[m] * sx[m] * swn;
        v += b;
        if (res) v += __half2float(res[idx]);
        if (y_f32) ((float*)y)[idx] = v;
        else       ((half*)y)[idx] = __float2half(v);
    }
}

template <bool W4, int LPR>
static void i8v_launch_mr(int M, int blocks, const int8_t* xq, const float* sx,
                          const int32_t* xsum, const i8v_segs& segs, int K,
                          int G, int y_f32, cudaStream_t st)
{
#define I8V_CASE(MR) case MR: i8_gemv_kernel<W4, LPR, MR><<<blocks,           \
        I8V_THREADS, 0, st>>>(xq, sx, xsum, segs, K, G, y_f32); break;
    switch (M) {
        I8V_CASE(1) I8V_CASE(2) I8V_CASE(3) I8V_CASE(4)
        I8V_CASE(5) I8V_CASE(6) I8V_CASE(7) I8V_CASE(8)
    }
#undef I8V_CASE
}

template <bool W4>
static void i8v_launch_lpr(int lpr, int M, int blocks_of[], const int8_t* xq,
                           const float* sx, const int32_t* xsum, i8v_segs segs,
                           int n_segs, int K, int G, int y_f32, cudaStream_t st)
{
    const int rpb = I8V_THREADS / lpr;
    int blocks = 0;
    for (int i = 0; i < I8V_MAX_SEGS; i++) {
        if (i < n_segs) {
            segs.first_block[i] = blocks;
            blocks += (segs.n[i] + rpb - 1) / rpb;
        } else {
            segs.first_block[i] = INT_MAX;
        }
    }
    (void)blocks_of;
    switch (lpr) {
    case 16: i8v_launch_mr<W4, 16>(M, blocks, xq, sx, xsum, segs, K, G, y_f32, st); break;
    case 32: i8v_launch_mr<W4, 32>(M, blocks, xq, sx, xsum, segs, K, G, y_f32, st); break;
    case 64: i8v_launch_mr<W4, 64>(M, blocks, xq, sx, xsum, segs, K, G, y_f32, st); break;
    default: i8v_launch_mr<W4, 128>(M, blocks, xq, sx, xsum, segs, K, G, y_f32, st); break;
    }
}

/**
 * @brief Lanes per row: at least 16 (256 contiguous bytes per step), more
 *        while the grid is short of ~48K threads so a lone 512-row
 *        projection still reaches every SM, never more than twice the row's
 *        chunk count -- the rule the W4A16 GEMV measured.
 */
static int i8v_pick_lpr(long rows_total, int nch) {
    int lpr = 16;
    while (lpr < 128 && rows_total * lpr < 49152L) lpr *= 2;
    while (lpr > 16 && lpr > 2 * nch) lpr >>= 1;
    return lpr;
}

/**
 * @brief Launch the M <= 8 kernels over `n` projections of the same xq.
 *
 * `ap` carries the shared operands (xq, sx, xsum, K, G, M, y_f32, layout);
 * the per-projection ones come from `projs` (or, for n = 1 with projs NULL,
 * from `ap` itself).
 */
extern "C" vv_status_t vv_i8_gemv_launch_multi(const vv_i8_args* ap, int w4,
                                               const vv_i8_proj_t* projs,
                                               int n, void* stream)
{
    const vv_i8_args p = *ap;
    if (p.M < 1 || p.M > 8 || n < 1 || n > I8V_MAX_SEGS)
        return VV_ERR_UNSUPPORTED;
    if (w4) {
        if ((p.K & 31) != 0 || !p.xsum || p.G <= 0 || (p.G & 31) != 0 ||
            (p.K % p.G) != 0 || p.x_layout != VV_Q8_W4_GEMV)
            return VV_ERR_UNSUPPORTED;
    } else if ((p.K & 15) != 0 || p.x_layout != VV_Q8_NATURAL) {
        return VV_ERR_UNSUPPORTED;
    }
    i8v_segs segs;
    memset(&segs, 0, sizeof(segs));
    long rows_total = 0;
    for (int i = 0; i < n; i++) {
        const vv_i8_proj_t* q = projs ? &projs[i] : NULL;
        segs.w[i]    = (const uint4*)(q ? q->w : p.w);
        segs.sw[i]   = q ? (const float*)q->sw : p.sw;
        segs.sz[i]   = q ? (const half2*)q->sz : p.sz;
        segs.bias[i] = q ? (const half*)q->bias : p.bias;
        segs.res[i]  = q ? NULL : p.res;
        segs.y[i]    = q ? q->y : p.y;
        segs.n[i]    = q ? q->N : p.N;
        if (!segs.w[i] || !segs.y[i] || segs.n[i] <= 0 ||
            (w4 ? !segs.sz[i] : !segs.sw[i]))
            return VV_ERR_NULL_PTR;
        rows_total += segs.n[i];
    }
    const int lpr = i8v_pick_lpr(rows_total, w4 ? p.K >> 5 : p.K >> 4);
    cudaStream_t st = (cudaStream_t)stream;
    if (w4) i8v_launch_lpr<true>(lpr, p.M, NULL, p.xq, p.sx, p.xsum, segs, n,
                                 p.K, p.G, p.y_f32, st);
    else    i8v_launch_lpr<false>(lpr, p.M, NULL, p.xq, p.sx, p.xsum, segs, n,
                                  p.K, p.G, p.y_f32, st);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/** @brief One projection: the operands are all in `ap`. */
extern "C" vv_status_t vv_i8_gemv_launch(const vv_i8_args* ap, int w4,
                                         void* stream)
{
    return vv_i8_gemv_launch_multi(ap, w4, NULL, 1, stream);
}
