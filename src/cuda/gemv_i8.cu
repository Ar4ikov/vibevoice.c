/**
 * @file gemv_i8.cu
 * @brief Decode-shaped int8 linear layers (M <= 8) on dp4a.
 *
 * One warp per output row, four rows per block (so the 512-row k/v
 * projections still launch 128 blocks), and up to eight activation rows
 * sharing each weight load: the weights are read once however many tokens
 * are in flight, which is the whole cost at this shape.
 *
 * W8A8: each lane reads 16 int8 weights per step and does four dp4a per
 * activation row; the dot product is an exact int32 until the one multiply
 * by sx * sw at the end.
 *
 * W4A8: each lane reads 32 weights (16 bytes) per step, all inside one group
 * because G is a multiple of 32. The nibbles are used as unsigned bytes
 * (0..15) straight from the packed word: `w & 0x0F0F0F0F` gives the odd
 * columns and `(w >> 4) & 0x0F0F0F0F` the even ones, and the activation word
 * pair is split the same way with two byte permutes. The zero point is
 * applied once per 32 weights through the quantizer's xsum:
 *
 *   sum (q - z) x = sum q x - z * sum x
 *
 * which keeps the inner loop to one AND, one shift and two dp4a per 8
 * weights per row.
 */

#include "i8_common.cuh"
#include "vibevoice/device.h"

#define I8_GEMV_WARPS 4

__device__ __forceinline__ int i8_warp_isum(int v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

__device__ __forceinline__ float i8_warp_fsum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

template <int MR>
__global__ void __launch_bounds__(32 * I8_GEMV_WARPS)
w8a8_gemv_kernel(const vv_i8_args p)
{
    const int n = blockIdx.x * I8_GEMV_WARPS + threadIdx.y;
    if (n >= p.N) return;
    const int lane = threadIdx.x;
    const int chunks = p.K >> 4;
    const int4* wr = (const int4*)((const int8_t*)p.w + (size_t)n * p.K);

    int acc[MR];
#pragma unroll
    for (int m = 0; m < MR; m++) acc[m] = 0;

#pragma unroll 2
    for (int c = lane; c < chunks; c += 32) {
        const int4 wv = __ldg(wr + c);
#pragma unroll
        for (int m = 0; m < MR; m++) {
            const int4 xv = __ldg((const int4*)(p.xq + (size_t)m * p.K) + c);
            acc[m] = i8_dp4a(wv.x, xv.x, acc[m]);
            acc[m] = i8_dp4a(wv.y, xv.y, acc[m]);
            acc[m] = i8_dp4a(wv.z, xv.z, acc[m]);
            acc[m] = i8_dp4a(wv.w, xv.w, acc[m]);
        }
    }

    const float swn = p.sw[n];
#pragma unroll
    for (int m = 0; m < MR; m++) {
        const int a = i8_warp_isum(acc[m]);
        if (lane == 0) i8_store(p, m, n, (float)a * p.sx[m] * swn);
    }
}

template <int MR, bool NIB>
__global__ void __launch_bounds__(32 * I8_GEMV_WARPS)
w4a8_gemv_kernel(const vv_i8_args p)
{
    const int n = blockIdx.x * I8_GEMV_WARPS + threadIdx.y;
    if (n >= p.N) return;
    const int lane = threadIdx.x;
    const int chunks = p.K >> 5;               /* 32 weights = 16 bytes */
    const int ng = p.K / p.G;
    const uint4* wr = (const uint4*)((const uint8_t*)p.w + (size_t)n * (p.K >> 1));
    const half* srow = p.scales + (size_t)n * ng;
    const uint8_t* zrow = p.zeros + (size_t)n * ng;

    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; m++) acc[m] = 0.0f;

#pragma unroll 2
    for (int c = lane; c < chunks; c += 32) {
        const uint4 wv = __ldg(wr + c);
        const int g = (c << 5) / p.G;
        const float s = __half2float(srow[g]);
        const int z = (int)zrow[g];
        const uint32_t wa[4] = { wv.x, wv.y, wv.z, wv.w };
        uint32_t hi[4], lo[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            hi[i] = (wa[i] >> 4) & 0x0F0F0F0Fu;   /* k = 8i + 0, 2, 4, 6 */
            lo[i] = wa[i] & 0x0F0F0F0Fu;          /* k = 8i + 1, 3, 5, 7 */
        }
#pragma unroll
        for (int m = 0; m < MR; m++) {
            const int4* xp = (const int4*)(p.xq + (size_t)m * p.K + (c << 5));
            const int4 xa = __ldg(xp), xb = __ldg(xp + 1);
            const uint32_t xw[8] = { (uint32_t)xa.x, (uint32_t)xa.y,
                                     (uint32_t)xa.z, (uint32_t)xa.w,
                                     (uint32_t)xb.x, (uint32_t)xb.y,
                                     (uint32_t)xb.z, (uint32_t)xb.w };
            int dot = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                /* nibble layout: evens are words 0-3, odds words 4-7 */
                const uint32_t ev = NIB ? xw[i]
                    : __byte_perm(xw[2 * i], xw[2 * i + 1], 0x6420);
                const uint32_t od = NIB ? xw[4 + i]
                    : __byte_perm(xw[2 * i], xw[2 * i + 1], 0x7531);
                dot = i8_dp4a((int)hi[i], (int)ev, dot);
                dot = i8_dp4a((int)lo[i], (int)od, dot);
            }
            const int xs = __ldg(p.xsum + (size_t)m * chunks + c);
            acc[m] = fmaf(s, (float)(dot - z * xs), acc[m]);
        }
    }

#pragma unroll
    for (int m = 0; m < MR; m++) {
        const float a = i8_warp_fsum(acc[m]);
        if (lane == 0) i8_store(p, m, n, a * p.sx[m]);
    }
}

#define I8_GEMV_CASE(KERN, MR)                                               \
    case MR: KERN<MR><<<grid, block, 0, st>>>(p); break;
#define I8_W4_CASE(MR)                                                       \
    case MR: if (p.x_nibble) w4a8_gemv_kernel<MR, true><<<grid, block, 0, st>>>(p); \
             else w4a8_gemv_kernel<MR, false><<<grid, block, 0, st>>>(p);   \
             break;

/** @brief Launch the M <= 8 kernels; `w4` picks the weight format. */
extern "C" vv_status_t vv_i8_gemv_launch(const vv_i8_args* ap, int w4,
                                         void* stream)
{
    const vv_i8_args p = *ap;
    if (p.M < 1 || p.M > 8) return VV_ERR_UNSUPPORTED;
    if (w4) {
        if ((p.K & 31) != 0 || !p.xsum || p.G <= 0 || (p.G & 31) != 0 ||
            (p.K % p.G) != 0)
            return VV_ERR_UNSUPPORTED;
    } else if ((p.K & 15) != 0) {
        return VV_ERR_UNSUPPORTED;
    }
    const dim3 block(32, I8_GEMV_WARPS);
    const dim3 grid((p.N + I8_GEMV_WARPS - 1) / I8_GEMV_WARPS);
    cudaStream_t st = (cudaStream_t)stream;
    if (w4) {
        switch (p.M) {
            I8_W4_CASE(1) I8_W4_CASE(2) I8_W4_CASE(3) I8_W4_CASE(4)
            I8_W4_CASE(5) I8_W4_CASE(6) I8_W4_CASE(7) I8_W4_CASE(8)
        }
    } else {
        switch (p.M) {
            I8_GEMV_CASE(w8a8_gemv_kernel, 1) I8_GEMV_CASE(w8a8_gemv_kernel, 2)
            I8_GEMV_CASE(w8a8_gemv_kernel, 3) I8_GEMV_CASE(w8a8_gemv_kernel, 4)
            I8_GEMV_CASE(w8a8_gemv_kernel, 5) I8_GEMV_CASE(w8a8_gemv_kernel, 6)
            I8_GEMV_CASE(w8a8_gemv_kernel, 7) I8_GEMV_CASE(w8a8_gemv_kernel, 8)
        }
    }
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}
