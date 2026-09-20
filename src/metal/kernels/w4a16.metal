/*
 * w4a16.metal -- INT4 group-affine weights (AWQ / GPTQ) in the GPU layout
 * (vv_int4g_to_gpu_layout), on FP16 activations.
 *
 * Layout, as w4a16.cu reads it: a row is 16-byte chunks of 32 weights;
 * inside a chunk, 32-bit word t holds k-pairs t, t+4, t+8, t+12 (pair p is
 * k = 2p, 2p+1), nibble slot s being pair t + 4*(s & 3), element s >> 2.
 * sz is half2 {scale, integer zero point} per group of G >= 32, so a chunk
 * never straddles two groups.
 *
 * Dequantization is the reference's formula, w = (q - z) * s in FP16: q - z
 * is exact and the multiply rounds once, like PyTorch's
 * (qweight - qzeros) * scales. The products are summed in FP32 (the CUDA
 * GEMV sums chains of four in FP16 first; the difference is below FP16
 * resolution of the output).
 */

/* The 32 weights of one chunk, dequantized, in k order. */
static inline void w4_chunk(uint4 wv, half2 sz, thread half w[32]) {
    const half s = sz.x, z = sz.y;
    const uint words[4] = { wv.x, wv.y, wv.z, wv.w };
    for (int t = 0; t < 4; t++) {
        const uint word = words[t];
        for (int j = 0; j < 4; j++) {
            const int k = 2 * (t + 4 * j);
            w[k]     = ((half)((word >> (4 * j)) & 0xFu) - z) * s;
            w[k + 1] = ((half)((word >> (4 * (j + 4))) & 0xFu) - z) * s;
        }
    }
}

static inline float w4_chunk_dot(uint4 wv, half2 sz, device const half* xc) {
    half w[32];
    w4_chunk(wv, sz, w);
    float acc = 0.0f;
    for (int i = 0; i < 32; i += 4) {
        const half4 xv = *(device const half4*)(xc + i);
        acc = fma((float)w[i],     (float)xv.x, acc);
        acc = fma((float)w[i + 1], (float)xv.y, acc);
        acc = fma((float)w[i + 2], (float)xv.z, acc);
        acc = fma((float)w[i + 3], (float)xv.w, acc);
    }
    return acc;
}

/*
 * GEMV for up to three projections of the same x (q/k/v, gate/up): the
 * grid is their row blocks one after another, 4 rows (simdgroups) a block,
 * and each block finds its projection from first_block. Lane l of a row
 * takes chunks l, l + 32, ...
 */
struct vv_w4v_p {
    int K; int gshift;
    int n[3]; int first_block[3]; int has_bias[3];
};

kernel void vv_w4a16_gemv(constant vv_w4v_p& p [[buffer(0)]],
                          device const half* x [[buffer(1)]],
                          device const uint4* w0 [[buffer(2)]],
                          device const half2* sz0 [[buffer(3)]],
                          device const half* b0 [[buffer(4)]],
                          device half* y0 [[buffer(5)]],
                          device const uint4* w1 [[buffer(6)]],
                          device const half2* sz1 [[buffer(7)]],
                          device const half* b1 [[buffer(8)]],
                          device half* y1 [[buffer(9)]],
                          device const uint4* w2 [[buffer(10)]],
                          device const half2* sz2 [[buffer(11)]],
                          device const half* b2 [[buffer(12)]],
                          device half* y2 [[buffer(13)]],
                          VV_GRID_ARGS) {
    const int bx = (int)blockIdx.x;
    const int seg = bx >= p.first_block[2] ? 2 : (bx >= p.first_block[1] ? 1 : 0);
    device const uint4* w = seg == 0 ? w0 : (seg == 1 ? w1 : w2);
    device const half2* sz = seg == 0 ? sz0 : (seg == 1 ? sz1 : sz2);
    device const half* bias = seg == 0 ? b0 : (seg == 1 ? b1 : b2);
    device half* y = seg == 0 ? y0 : (seg == 1 ? y1 : y2);
    const int N = p.n[seg];
    const int row = (bx - p.first_block[seg]) * 4 + (int)warp;
    if (row >= N) return;

    const int nch = p.K >> 5, ngroups = p.K >> p.gshift;
    device const uint4* wrow = w + (ulong)row * (ulong)nch;
    device const half2* srow = sz + (ulong)row * (ulong)ngroups;
    float acc = 0.0f;
    for (int c = (int)lane; c < nch; c += 32)
        acc += w4_chunk_dot(wrow[c], srow[(c << 5) >> p.gshift], x + (c << 5));
    acc = vv_warp_sum(acc);
    if (lane == 0) {
        const float b = p.has_bias[seg] ? (float)bias[row] : 0.0f;
        y[row] = (half)(acc + b);
    }
}

struct vv_w4_gather_p { int M; int K; };

/* out[m][j] = x[m][perm[j]], 8 columns a thread. */
kernel void vv_w4a16_gather(constant vv_w4_gather_p& p [[buffer(0)]],
                            device const half* x [[buffer(1)]],
                            device const int* perm [[buffer(2)]],
                            device half* out [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    const ulong i = (ulong)tid * 8;
    if (i >= (ulong)p.M * (ulong)p.K) return;
    const int m = (int)(i / (ulong)p.K), j = (int)(i % (ulong)p.K);
    device const half* xr = x + (ulong)m * (ulong)p.K;
    for (int t = 0; t < 8; t++) out[i + t] = xr[perm[j + t]];
}

struct vv_w4_deq_p { int N; int K; int gshift; };

/* GPU layout -> dense FP16 [N][K], one chunk (32 weights) a thread. */
kernel void vv_w4a16_dequant(constant vv_w4_deq_p& p [[buffer(0)]],
                             device const uint4* w [[buffer(1)]],
                             device const half2* sz [[buffer(2)]],
                             device half* out [[buffer(3)]],
                             uint tid [[thread_position_in_grid]]) {
    const int nch = p.K >> 5;
    if ((ulong)tid >= (ulong)p.N * (ulong)nch) return;
    const int row = (int)(tid / (uint)nch), c = (int)(tid % (uint)nch);
    half v[32];
    w4_chunk(w[tid], sz[(ulong)row * (ulong)(p.K >> p.gshift) + (ulong)((c << 5) >> p.gshift)], v);
    device half4* dst = (device half4*)(out + (ulong)row * (ulong)p.K + (ulong)(c << 5));
    for (int i = 0; i < 8; i++)
        dst[i] = half4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}

/*
 * GEMM without unrolling the weight: C[M,N] = A[M,K] . W[N,K]^T.
 *
 * The tile is linear.metal's (64x64 of C per threadgroup, four simdgroups of
 * 32x32, K in steps of 32) -- these files are one library, so its helpers and
 * strides are reused rather than repeated. What changes is where B comes from:
 * a K step of 32 is exactly one 16-byte chunk of a row, so each step
 * dequantizes one chunk per row straight into threadgroup memory and the
 * weight is read once, in four bits, instead of being written out as 136 MB
 * of FP16 first.
 *
 * The values are the same halves `vv_w4a16_dequant` would have written and
 * the k order is the same, so this is bit for bit what dequant + vv_gemm_tn
 * produced. Bias stays with the caller's vv_bias_add, added after the output
 * has been rounded to FP16, as it was before.
 */
struct vv_w4_gemm_p { int M; int N; int K; int gshift; };

kernel void vv_w4a16_gemm(constant vv_w4_gemm_p& p [[buffer(0)]],
                          device const half* A [[buffer(1)]],
                          device const uint4* w [[buffer(2)]],
                          device const half2* sz [[buffer(3)]],
                          device half* C [[buffer(4)]],
                          threadgroup uchar* smem [[threadgroup(0)]],
                          VV_GRID_ARGS) {
    threadgroup half* As = (threadgroup half*)smem;
    threadgroup half* Bs = As + G_BM * G_LDS;
    const uint tid = threadIdx.x;
    const int row0 = (int)blockIdx.y * G_BM, col0 = (int)blockIdx.x * G_BN;
    const int wm = (int)warp >> 1, wn = (int)warp & 1;
    const int nch = p.K >> 5, ngroups = p.K >> p.gshift;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    /* Two threads a row: one takes the chunk's words 0,1 and the other 2,3. */
    const int br = (int)(tid >> 1), wsel = (int)(tid & 1) * 2;
    const int gn = col0 + br;
    const bool live = gn < p.N;
    device const uint4* wrow = w + (ulong)(live ? gn : 0) * (ulong)nch;
    device const half2* srow = sz + (ulong)(live ? gn : 0) * (ulong)ngroups;

    for (int k0 = 0; k0 < p.K; k0 += G_BK) {
        g_load_kmajor(As, A, row0, p.M, p.K, k0, tid);
        {
            const int c = k0 >> 5;
            threadgroup half* d = Bs + br * G_LDS;
            const uint4 wv = live ? wrow[c] : uint4(0u);
            const half2 szv = live ? srow[(c << 5) >> p.gshift] : half2(0.0h);
            const half s = szv.x, z = szv.y;
            const uint w01[2] = { wsel == 0 ? wv.x : wv.z,
                                  wsel == 0 ? wv.y : wv.w };
            for (int t = 0; t < 2; t++) {
                const uint word = w01[t];
                const int tt = wsel + t;
                for (int j = 0; j < 4; j++) {
                    const int k = 2 * (tt + 4 * j);
                    d[k]     = live ? ((half)((word >> (4 * j)) & 0xFu) - z) * s
                                    : (half)0.0h;
                    d[k + 1] = live ? ((half)((word >> (4 * (j + 4))) & 0xFu) - z) * s
                                    : (half)0.0h;
                }
            }
        }
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
               1.0f, 0.0f, lane);
}
