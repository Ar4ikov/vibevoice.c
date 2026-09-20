#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct vv_gemm_p { int M, N, K; float alpha, beta; };
#define VV_GRID_ARGS                                                         \
    uint3 blockIdx  [[threadgroup_position_in_grid]],                        \
    uint3 threadIdx [[thread_position_in_threadgroup]],                      \
    uint3 blockDim  [[threads_per_threadgroup]],                             \
    uint  lane      [[thread_index_in_simdgroup]],                           \
    uint  warp      [[simdgroup_index_in_threadgroup]]

#define G_BM  64
#define G_BN  64
#define G_BK  32
#define G_LDS (G_BK + 8)        /* K-major tile row stride, halves   */
#define G_LDN (G_BN + 8)        /* N-major tile row stride, halves   */
#define G_LDC (32 + 4)          /* epilogue staging stride, floats   */

/* rows x 32 halves of a K-major matrix, zero outside [row_limit, K) */
static inline void g_load_kmajor(threadgroup half* dst, device const half* src,
                                 int row_base, int row_limit, int K, int k0,
                                 uint tid) {
    for (uint e = tid; e < 64u * 4u; e += 128u) {
        const int r = (int)(e >> 2), c = (int)(e & 3u) * 8;
        const int gr = row_base + r, gk = k0 + c;
        threadgroup half* d = dst + r * G_LDS + c;
        if (gr < row_limit && gk + 8 <= K && (K & 7) == 0) {
            device const half4* s4 = (device const half4*)(src + (ulong)gr * (ulong)K + gk);
            *(threadgroup half4*)(d) = s4[0];
            *(threadgroup half4*)(d + 4) = s4[1];
        } else {
            for (int t = 0; t < 8; t++)
                d[t] = (gr < row_limit && gk + t < K)
                     ? src[(ulong)gr * (ulong)K + gk + t] : (half)0.0f;
        }
    }
}

static inline void g_epilogue(threadgroup float* stage,
                              thread simdgroup_float8x8 (&acc)[4][4],
                              device half* C, int ldc, int row0, int col0,
                              int rows, int cols, float alpha, float beta,
                              uint lane) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], stage + (i * 8) * G_LDC + j * 8, G_LDC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lane; e < 32 * 32; e += 32) {
        const int r = e >> 5, c = e & 31;
        const int gr = row0 + r, gc = col0 + c;
        if (gr < rows && gc < cols) {
            const ulong off = (ulong)gr * (ulong)ldc + (ulong)gc;
            float v = alpha * stage[r * G_LDC + c];
            if (beta != 0.0f) v += beta * (float)C[off];
            C[off] = (half)v;
        }
    }
}

/* C[M,N] = alpha * A[M,K] . B[N,K]^T + beta * C */
kernel void vv_gemm_tn(constant vv_gemm_p& p [[buffer(0)]],
                       device const half* A [[buffer(1)]],
                       device const half* B [[buffer(2)]],
                       device half* C [[buffer(3)]],
                       threadgroup uchar* smem [[threadgroup(0)]],
                       VV_GRID_ARGS) {
    threadgroup half* As = (threadgroup half*)smem;
    threadgroup half* Bs = As + G_BM * G_LDS;
    const uint tid = threadIdx.x;
    const int row0 = (int)blockIdx.y * G_BM, col0 = (int)blockIdx.x * G_BN;
    const int wm = (int)warp >> 1, wn = (int)warp & 1;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += G_BK) {
        g_load_kmajor(As, A, row0, p.M, p.K, k0, tid);
        g_load_kmajor(Bs, B, col0, p.N, p.K, k0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < G_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a[i], As + (wm * 32 + i * 8) * G_LDS + kk, G_LDS);
            for (int j = 0; j < 4; j++)
                simdgroup_load(b[j], Bs + (wn * 32 + j * 8) * G_LDS + kk, G_LDS,
                               ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* stage = (threadgroup float*)smem + (int)warp * 32 * G_LDC;
    g_epilogue(stage, acc, C, p.N, row0 + wm * 32, col0 + wn * 32, p.M, p.N,
               p.alpha, p.beta, lane);
}


/* The same C = A . B^T, on Metal 4 tensor operations. */
struct gp4 { uint M, N, K; };
template <int BM, int BN>
kernel void t4_gemm(device half* A [[buffer(0)]],
                    device half* B [[buffer(1)]],
                    device float* C [[buffer(2)]],
                    constant gp4& p [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    const int M = (int)p.M, N = (int)p.N, K = (int)p.K;
    auto tA = tensor(A, dextents<int32_t, 2>(K, M));
    auto tB = tensor(B, dextents<int32_t, 2>(K, N));
    auto tC = tensor(C, dextents<int32_t, 2>(N, M));  /* float out: half is rejected */
    constexpr auto d = matmul2d_descriptor(BM, BN,
                                           static_cast<int>(dynamic_extent),
                                           false, true, false);
    matmul2d<d, execution_simdgroups<4>> op;
    auto sA = tA.slice(0, (int)tg.x * BM);
    auto sB = tB.slice(0, (int)tg.y * BN);
    auto sC = tC.slice((int)tg.y * BN, (int)tg.x * BM);
    op.run(sA, sB, sC);
}
template [[host_name("t4_64x32")]] kernel decltype(t4_gemm<64,32>) t4_gemm<64,32>;
template [[host_name("t4_64x64")]] kernel decltype(t4_gemm<64,64>) t4_gemm<64,64>;
template [[host_name("t4_32x32")]] kernel decltype(t4_gemm<32,32>) t4_gemm<32,32>;
