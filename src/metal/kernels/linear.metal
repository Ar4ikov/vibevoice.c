/*
 * linear.metal -- NF4 GEMV and dequantisation, dense FP16 GEMM, and the LM
 * head. Ports of src/cuda/{nf4_gemv,dequant_nf4,gemm,lm_head}.cu.
 *
 * The GEMVs keep the CUDA kernels' arithmetic exactly: one simdgroup per
 * row, each lane summing its 8-element slices in the same order, then the
 * same xor butterfly. A decode step therefore produces the bits the CUDA
 * build does. The tiled GEMM runs on simdgroup matrices instead of mma.sync;
 * its sums are FP32 as there, in a different order.
 */

/* ─── NF4 ───────────────────────────────────────────────────────────────── */

struct vv_nf4_gemv_p { int N; int K; int has_bias; };

/* y[n] = sum_k dequant(W)[n,k] x[k] (+ bias[n]). 8 simdgroups = 8 rows per
 * group. packed [N][K/2] (high nibble first), scales [N][K/64]. */
kernel void vv_nf4_gemv(constant vv_nf4_gemv_p& p [[buffer(0)]],
                        device const half* x [[buffer(1)]],
                        device const uint* w [[buffer(2)]],
                        device const half* scales [[buffer(3)]],
                        device const half* bias [[buffer(4)]],
                        device half* y [[buffer(5)]],
                        VV_GRID_ARGS) {
    threadgroup float lut[16];
    if (warp == 0 && lane < 16) lut[lane] = vv_nf4_lut[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.N) return;
    device const uint* wrow = w + (ulong)row * (ulong)(p.K >> 3);
    device const half* srow = scales + (ulong)row * (ulong)(p.K >> 6);

    float acc = 0.0f;
    for (int base = 0; base < p.K; base += 256) {
        const int e0 = base + (int)lane * 8;
        const uint bits = wrow[e0 >> 3];
        const float sc = (float)srow[e0 >> 6];
        const half4 xa = *(device const half4*)(x + e0);
        const half4 xb = *(device const half4*)(x + e0 + 4);
        const float xs[8] = { (float)xa.x, (float)xa.y, (float)xa.z, (float)xa.w,
                              (float)xb.x, (float)xb.y, (float)xb.z, (float)xb.w };
        for (int b = 0; b < 4; b++) {
            const uint byte = (bits >> (b * 8)) & 0xFFu;
            const float w0 = lut[byte >> 4] * sc;
            const float w1 = lut[byte & 0xFu] * sc;
            acc = fma(w0, xs[2 * b], acc);
            acc = fma(w1, xs[2 * b + 1], acc);
        }
    }
    acc = vv_warp_sum(acc);
    if (lane == 0) {
        if (p.has_bias) acc += (float)bias[row];
        y[row] = (half)acc;
    }
}

struct vv_nf4_deq_p { int n_elements; };

/* 8 values per thread: one uint of packed nibbles, one scale. */
kernel void vv_dequant_nf4(constant vv_nf4_deq_p& p [[buffer(0)]],
                           device const uint* packed [[buffer(1)]],
                           device const half* scales [[buffer(2)]],
                           device half4* out [[buffer(3)]],
                           uint idx [[thread_position_in_grid]]) {
    const int e0 = (int)idx * 8;
    if (e0 >= p.n_elements) return;
    const uint bits = packed[idx];
    const float sc = (float)scales[e0 >> 6];
    half v[8];
    for (int b = 0; b < 4; b++) {
        const uint byte = (bits >> (b * 8)) & 0xFFu;
        v[2 * b]     = (half)(vv_nf4_lut[byte >> 4] * sc);
        v[2 * b + 1] = (half)(vv_nf4_lut[byte & 0xFu] * sc);
    }
    out[idx * 2]     = half4(v[0], v[1], v[2], v[3]);
    out[idx * 2 + 1] = half4(v[4], v[5], v[6], v[7]);
}

/* ─── Dense GEMM ────────────────────────────────────────────────────────── */

struct vv_gemm_p { int M; int N; int K; float alpha; float beta; };

/* M <= 8: one simdgroup streams one row of B against every row of A.
 * The same partial sums and shfl_down reduction as gemm.cu. */
kernel void vv_gemm_tn_skinny(constant vv_gemm_p& p [[buffer(0)]],
                              device const half* A [[buffer(1)]],
                              device const half* B [[buffer(2)]],
                              device half* C [[buffer(3)]],
                              VV_GRID_ARGS) {
    const int n = (int)blockIdx.x * (int)(blockDim.x >> 5) + (int)warp;
    if (n >= p.N) return;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    device const half* brow = B + (ulong)n * (ulong)p.K;
    const int k_vec = (p.K / 8) * 8;
    for (int k = (int)lane * 8; k < k_vec; k += 32 * 8) {
        const half4 b0 = *(device const half4*)(brow + k);
        const half4 b1 = *(device const half4*)(brow + k + 4);
        const float bf[8] = { (float)b0.x, (float)b0.y, (float)b0.z, (float)b0.w,
                              (float)b1.x, (float)b1.y, (float)b1.z, (float)b1.w };
        for (int m = 0; m < p.M; ++m) {
            device const half* arow = A + (ulong)m * (ulong)p.K + k;
            const half4 a0 = *(device const half4*)(arow);
            const half4 a1 = *(device const half4*)(arow + 4);
            const float af[8] = { (float)a0.x, (float)a0.y, (float)a0.z, (float)a0.w,
                                  (float)a1.x, (float)a1.y, (float)a1.z, (float)a1.w };
            float s = 0.0f;
            for (int t = 0; t < 4; ++t)
                s += af[2 * t] * bf[2 * t] + af[2 * t + 1] * bf[2 * t + 1];
            acc[m] += s;
        }
    }
    for (int k = k_vec + (int)lane; k < p.K; k += 32) {
        const float b = (float)brow[k];
        for (int m = 0; m < p.M; ++m)
            acc[m] += (float)A[(ulong)m * (ulong)p.K + k] * b;
    }
    for (int m = 0; m < 8; ++m) acc[m] = vv_warp_sum_down(acc[m]);
    if (lane == 0) {
        for (int m = 0; m < p.M; ++m) {
            const ulong o = (ulong)m * (ulong)p.N + (ulong)n;
            float v = p.alpha * acc[m];
            if (p.beta != 0.0f) v += p.beta * (float)C[o];
            C[o] = (half)v;
        }
    }
}

/*
 * Tiled GEMM on simdgroup matrices: a 64x64 block of C per threadgroup of
 * four simdgroups, each owning a 32x32 quadrant (4x4 accumulators of 8x8
 * FP32), K in steps of 32 staged through threadgroup memory.
 */
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

/* C[M,P] = alpha * A[M,K] . B[K,P] + beta * C  (p.N holds P) */
kernel void vv_gemm_nn(constant vv_gemm_p& p [[buffer(0)]],
                       device const half* A [[buffer(1)]],
                       device const half* B [[buffer(2)]],
                       device half* C [[buffer(3)]],
                       threadgroup uchar* smem [[threadgroup(0)]],
                       VV_GRID_ARGS) {
    threadgroup half* As = (threadgroup half*)smem;
    threadgroup half* Bs = As + G_BM * G_LDS;          /* [G_BK][G_LDN] */
    const uint tid = threadIdx.x;
    const int P = p.N;
    const int row0 = (int)blockIdx.y * G_BM, col0 = (int)blockIdx.x * G_BN;
    const int wm = (int)warp >> 1, wn = (int)warp & 1;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += G_BK) {
        g_load_kmajor(As, A, row0, p.M, p.K, k0, tid);
        for (uint e = tid; e < (uint)G_BK * 8u; e += 128u) {
            const int r = (int)(e >> 3), c = (int)(e & 7u) * 8;
            const int gk = k0 + r, gc = col0 + c;
            threadgroup half* d = Bs + r * G_LDN + c;
            if (gk < p.K && gc + 8 <= P && (P & 7) == 0) {
                device const half4* s4 = (device const half4*)(B + (ulong)gk * (ulong)P + gc);
                *(threadgroup half4*)(d) = s4[0];
                *(threadgroup half4*)(d + 4) = s4[1];
            } else {
                for (int t = 0; t < 8; t++)
                    d[t] = (gk < p.K && gc + t < P)
                         ? B[(ulong)gk * (ulong)P + gc + t] : (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < G_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a[i], As + (wm * 32 + i * 8) * G_LDS + kk, G_LDS);
            for (int j = 0; j < 4; j++)
                simdgroup_load(b[j], Bs + kk * G_LDN + wn * 32 + j * 8, G_LDN);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* stage = (threadgroup float*)smem + (int)warp * 32 * G_LDC;
    g_epilogue(stage, acc, C, P, row0 + wm * 32, col0 + wn * 32, p.M, P,
               p.alpha, p.beta, lane);
}

/* ─── LM head ───────────────────────────────────────────────────────────── */

struct vv_lm_head_p { int V; int K; };

/* logits[n] = W[n,:] . x, FP32, one simdgroup per row as in lm_head.cu. */
kernel void vv_lm_head_gemv(constant vv_lm_head_p& p [[buffer(0)]],
                            device const half* x [[buffer(1)]],
                            device const half* W [[buffer(2)]],
                            device float* logits [[buffer(3)]],
                            VV_GRID_ARGS) {
    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.V) return;
    device const half* wrow = W + (ulong)row * (ulong)p.K;
    float acc = 0.0f;
    int k = (int)lane * 8;
    for (; k + 8 <= p.K; k += 256) {
        const half4 w0 = *(device const half4*)(wrow + k);
        const half4 w1 = *(device const half4*)(wrow + k + 4);
        const half4 x0 = *(device const half4*)(x + k);
        const half4 x1 = *(device const half4*)(x + k + 4);
        acc = fma((float)w0.x, (float)x0.x, acc);
        acc = fma((float)w0.y, (float)x0.y, acc);
        acc = fma((float)w0.z, (float)x0.z, acc);
        acc = fma((float)w0.w, (float)x0.w, acc);
        acc = fma((float)w1.x, (float)x1.x, acc);
        acc = fma((float)w1.y, (float)x1.y, acc);
        acc = fma((float)w1.z, (float)x1.z, acc);
        acc = fma((float)w1.w, (float)x1.w, acc);
    }
    for (int t = (p.K & ~255) + (int)lane; t < p.K; t += 32)
        acc = fma((float)wrow[t], (float)x[t], acc);
    acc = vv_warp_sum(acc);
    if (lane == 0) logits[row] = acc;
}

/* ─── Argmax ────────────────────────────────────────────────────────────── */

struct vv_argmax_p { int V; int n; int has_value; };

/* Stage 1: 256 groups of 256, strided over V; the tree keeps the lower
 * thread on ties, as lm_head.cu does. */
kernel void vv_argmax_partial(constant vv_argmax_p& p [[buffer(0)]],
                              device const float* logits [[buffer(1)]],
                              device float* pmax [[buffer(2)]],
                              device int* pidx [[buffer(3)]],
                              uint3 blockIdx [[threadgroup_position_in_grid]],
                              uint3 threadIdx [[thread_position_in_threadgroup]],
                              uint3 gridDim [[threadgroups_per_grid]]) {
    threadgroup float sv[256];
    threadgroup int si[256];
    const int tid = (int)threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int i = (int)blockIdx.x * 256 + tid; i < p.V; i += (int)gridDim.x * 256) {
        const float v = logits[i];
        if (v > best) { best = v; bidx = i; }
    }
    sv[tid] = best; si[tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}

kernel void vv_argmax_final(constant vv_argmax_p& p [[buffer(0)]],
                            device const float* pmax [[buffer(1)]],
                            device const int* pidx [[buffer(2)]],
                            device int* out_token [[buffer(3)]],
                            device float* out_value [[buffer(4)]],
                            uint3 threadIdx [[thread_position_in_threadgroup]]) {
    threadgroup float sv[256];
    threadgroup int si[256];
    const int tid = (int)threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int i = tid; i < p.n; i += 256)
        if (pmax[i] > best) { best = pmax[i]; bidx = pidx[i]; }
    sv[tid] = best; si[tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        *out_token = si[0];
        if (p.has_value) *out_value = sv[0];
    }
}
