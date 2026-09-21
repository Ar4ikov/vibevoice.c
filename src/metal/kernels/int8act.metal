/*
 * int8act.metal -- per-token int8 activations and the W8A8 / W4A8 linear
 * layers that read them (act_quant.cu, gemv_i8.cu, gemm_i8.cu).
 *
 * Quantizers: sx = amax / 127, q = round-half-even(v * (127 / amax)), with
 * v the FP16 value the unfused op writes -- bit-identical to
 * vv_quant_act_q8_cpu on the same values. The CUDA kernels hold the row in
 * shared memory; a 18944-wide row is 37 KB, past Metal's 32 KB, so these
 * produce each value twice instead (once for the absmax, once to quantize),
 * by the same expression, which gives the same bits.
 *
 * Apple GPUs have no dp4a and no integer matrix unit, so the dot products
 * are int32 multiply-adds of unpacked bytes: exact, as the tests demand.
 */

/* Where column j of a 32-column run is stored (vv_q8_pos). */
static inline int q8_pos(int layout, int j) {
    const int e = j & 1, t = (j >> 1) & 3, s = (j >> 3) + 4 * e;
    switch (layout) {
    case 1:  return e ? 16 + (j >> 1) : (j >> 1);           /* NIBBLE  */
    case 2:  return 8 * t + 4 * (s & 1) + (s >> 1);         /* W4_GEMV */
    case 3:  return 16 * e + 4 * t + (s & 3);               /* W4_MMA  */
    default: return j;
    }
}

struct vv_q8_p { int K; int layout; int op; float eps; int has_xsum; };

#define Q8_OP_COPY   0
#define Q8_OP_NORM   1
#define Q8_OP_SWIGLU 2

/* Column i of the row as the unfused op stores it (FP16). */
static inline half q8_value(constant vv_q8_p& p, device const half* a,
                            device const half* b, int i, float rms) {
    if (p.op == Q8_OP_NORM)
        return (half)((float)a[i] * rms * (float)b[i]);
    if (p.op == Q8_OP_SWIGLU) {
        const float g = (float)a[i], u = (float)b[i];
        return (half)(g / (1.0f + exp(-g)) * u);
    }
    return a[i];
}

/* One group of 256 per row. a/b: x and the norm weight (NORM), gate and up
 * (SWIGLU), or x alone (COPY). */
kernel void vv_q8_quant(constant vv_q8_p& p [[buffer(0)]],
                        device const half* a [[buffer(1)]],
                        device const half* b [[buffer(2)]],
                        device char* xq [[buffer(3)]],
                        device float* sx [[buffer(4)]],
                        device int* xsum [[buffer(5)]],
                        VV_GRID_ARGS) {
    threadgroup float sdata[256];
    threadgroup float red[8];
    const int tid = (int)threadIdx.x;
    const ulong r = blockIdx.x;
    const int K = p.K;
    device const half* ar = a + r * (ulong)K;
    device const half* br = p.op == Q8_OP_SWIGLU ? b + r * (ulong)K : b;

    /* rmsnorm.metal's statistics, operation for operation */
    float rms = 0.0f;
    if (p.op == Q8_OP_NORM) {
        float sum_sq = 0.0f;
        for (int i = tid; i < K; i += 256) {
            const float v = (float)ar[i];
            sum_sq += v * v;
        }
        sdata[tid] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        rms = rsqrt(sdata[0] / (float)K + p.eps);
    }

    float amax = 0.0f;
    for (int i = tid; i < K; i += 256)
        amax = max(amax, fabs((float)q8_value(p, ar, br, i, rms)));
    amax = vv_warp_max(amax);
    if (lane == 0) red[warp] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < 8; w++) m = max(m, red[w]);
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];

    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (tid == 0) sx[r] = amax / 127.0f;

    device char* xr = xq + r * (ulong)K;
    const int chunks = K >> 3;
    for (int base = 0; base < chunks; base += 256) {
        const int c = base + tid;
        int s = 0;
        if (c < chunks) {
            device char* run = xr + ((c >> 2) << 5);
            const int j0 = (c & 3) << 3;
            for (int j = 0; j < 8; j++) {
                const float v = (float)q8_value(p, ar, br, (c << 3) + j, rms);
                int q = (int)rint(v * inv);
                q = clamp(q, -127, 127);
                s += q;
                run[q8_pos(p.layout, j0 + j)] = (char)q;
            }
        }
        if (p.has_xsum) {
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            if (c < chunks && (tid & 3) == 0) xsum[r * (ulong)(K >> 5) + (c >> 2)] = s;
        }
    }
}

struct vv_colmax_p { int M; int K; };

/* acc[k] = max(acc[k], max_m |x[m,k]|); non-negative floats order as ints. */
kernel void vv_col_absmax(constant vv_colmax_p& p [[buffer(0)]],
                          device const half* x [[buffer(1)]],
                          device atomic_int* acc [[buffer(2)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint3 threadIdx [[thread_position_in_threadgroup]]) {
    const int k = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (k >= p.K) return;
    const int m0 = (int)blockIdx.y * 64;
    const int m1 = min(p.M, m0 + 64);
    float v = 0.0f;
    for (int m = m0; m < m1; m++) v = max(v, fabs((float)x[(ulong)m * (ulong)p.K + k]));
    if (v == v) atomic_fetch_max_explicit(acc + k, as_type<int>(v), memory_order_relaxed);
}

/* ─── Linear layers ─────────────────────────────────────────────────────── */

static inline int dp4a(uint a, uint b, int c) {
    const int4 x = int4(as_type<char4>(a)), y = int4(as_type<char4>(b));
    return c + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
}

/* Four nibbles (slots 0..3 in the low 16 bits) -> four bytes q - z. */
static inline uint unpack4_sub(uint h, int z) {
    const int q0 = (int)(h & 0xFu) - z, q1 = (int)((h >> 4) & 0xFu) - z;
    const int q2 = (int)((h >> 8) & 0xFu) - z, q3 = (int)((h >> 12) & 0xFu) - z;
    return ((uint)q0 & 0xFFu) | (((uint)q1 & 0xFFu) << 8) |
           (((uint)q2 & 0xFFu) << 16) | (((uint)q3 & 0xFFu) << 24);
}

struct vv_i8_p {
    int M; int K; int G; int y_f32;
    int n[3]; int first_block[3]; int has_bias[3]; int has_res[3];
};

/*
 * GEMV for M <= 8 over up to three projections of the same activations,
 * four rows (simdgroups) a group. W8: a lane takes 16-byte chunks (16
 * weights), the sum stays int32. W4: a chunk is 32 weights in one group,
 * nibbles used as unsigned bytes against the VV_Q8_W4_GEMV order, the zero
 * point applied once per chunk through xsum: sum (q - z) x = sum q x - z sum x.
 */
template <bool W4>
kernel void vv_i8_gemv(constant vv_i8_p& p [[buffer(0)]],
                       device const char* xq [[buffer(1)]],
                       device const float* sx [[buffer(2)]],
                       device const int* xsum [[buffer(3)]],
                       device const uint4* w0 [[buffer(4)]],
                       device const float* sw0 [[buffer(5)]],
                       device const half2* sz0 [[buffer(6)]],
                       device const half* b0 [[buffer(7)]],
                       device const half* r0 [[buffer(8)]],
                       device void* y0 [[buffer(9)]],
                       device const uint4* w1 [[buffer(10)]],
                       device const float* sw1 [[buffer(11)]],
                       device const half2* sz1 [[buffer(12)]],
                       device const half* b1 [[buffer(13)]],
                       device void* y1 [[buffer(14)]],
                       device const uint4* w2 [[buffer(15)]],
                       device const float* sw2 [[buffer(16)]],
                       device const half2* sz2 [[buffer(17)]],
                       device const half* b2 [[buffer(18)]],
                       device void* y2 [[buffer(19)]],
                       VV_GRID_ARGS) {
    const int bx = (int)blockIdx.x;
    const int seg = bx >= p.first_block[2] ? 2 : (bx >= p.first_block[1] ? 1 : 0);
    device const uint4* w = seg == 0 ? w0 : (seg == 1 ? w1 : w2);
    const int N = p.n[seg];
    const int row = (bx - p.first_block[seg]) * 4 + (int)warp;
    if (row >= N) return;
    const int K = p.K;
    const int nch = W4 ? K >> 5 : K >> 4;
    device const uint4* wrow = w + (ulong)row * (ulong)nch;

    float facc[8];
    int iacc[8];
    for (int m = 0; m < 8; m++) { facc[m] = 0.0f; iacc[m] = 0; }

    if (W4) {
        device const half2* sz = seg == 0 ? sz0 : (seg == 1 ? sz1 : sz2);
        device const half2* szrow = sz + (ulong)row * (ulong)(K / p.G);
        for (int c = (int)lane; c < nch; c += 32) {
            const uint4 wv = wrow[c];
            const float2 szv = float2(szrow[(c << 5) / p.G]);
            const int z = (int)szv.y;
            const uint wa[4] = { wv.x, wv.y, wv.z, wv.w };
            for (int m = 0; m < 8; m++) {
                if (m >= p.M) break;
                device const uint* xp = (device const uint*)(xq + (ulong)m * (ulong)K + (ulong)(c << 5));
                int dot = 0;
                for (int t = 0; t < 4; t++) {
                    dot = dp4a(wa[t] & 0x0F0F0F0Fu, xp[2 * t], dot);
                    dot = dp4a((wa[t] >> 4) & 0x0F0F0F0Fu, xp[2 * t + 1], dot);
                }
                const int xs = xsum[(ulong)m * (ulong)(K >> 5) + (ulong)c];
                facc[m] = fma(szv.x, (float)(dot - z * xs), facc[m]);
            }
        }
    } else {
        for (int c = (int)lane; c < nch; c += 32) {
            const uint4 wv = wrow[c];
            for (int m = 0; m < 8; m++) {
                if (m >= p.M) break;
                const uint4 xv = ((device const uint4*)(xq + (ulong)m * (ulong)K))[c];
                int a = iacc[m];
                a = dp4a(wv.x, xv.x, a); a = dp4a(wv.y, xv.y, a);
                a = dp4a(wv.z, xv.z, a); a = dp4a(wv.w, xv.w, a);
                iacc[m] = a;
            }
        }
    }
    for (int m = 0; m < 8; m++) {
        if (W4) facc[m] = vv_warp_sum(facc[m]);
        else {
            int v = iacc[m];
            v += simd_shuffle_xor(v, 16); v += simd_shuffle_xor(v, 8);
            v += simd_shuffle_xor(v, 4);  v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 1);
            iacc[m] = v;
        }
    }
    if (lane != 0) return;
    device const float* sw = seg == 0 ? sw0 : (seg == 1 ? sw1 : sw2);
    device const half* bias = seg == 0 ? b0 : (seg == 1 ? b1 : b2);
    device void* y = seg == 0 ? y0 : (seg == 1 ? y1 : y2);
    const float swn = W4 ? 1.0f : sw[row];
    const float bv = p.has_bias[seg] ? (float)bias[row] : 0.0f;
    for (int m = 0; m < 8; m++) {
        if (m >= p.M) break;
        const ulong idx = (ulong)m * (ulong)N + (ulong)row;
        float v = W4 ? facc[m] * sx[m] : (float)iacc[m] * sx[m] * swn;
        v += bv;
        if (seg == 0 && p.has_res[0]) v += (float)r0[idx];
        if (p.y_f32) ((device float*)y)[idx] = v;
        else         ((device half*)y)[idx] = (half)v;
    }
}

typedef decltype(vv_i8_gemv<false>) vv_i8_gemv_t;
template [[host_name("vv_i8_gemv_w8")]] kernel vv_i8_gemv_t vv_i8_gemv<false>;
template [[host_name("vv_i8_gemv_w4")]] kernel vv_i8_gemv_t vv_i8_gemv<true>;

/*
 * Any M: 64x64 tiles, 256 threads with 4x4 outputs each, K in steps of 32
 * bytes of activations. W4 reads the VV_Q8_W4_MMA order and subtracts the
 * zero point in the unpacked weight bytes (q - z fits a signed byte), and
 * folds each group's integer sum into FP32 with its scale.
 */
struct vv_i8g_p { int M; int N; int K; int G; int y_f32; int has_bias; int has_res; int pad; };

template <bool W4>
kernel void vv_i8_gemm(constant vv_i8g_p& p [[buffer(0)]],
                       device const char* xq [[buffer(1)]],
                       device const float* sx [[buffer(2)]],
                       device const uchar* wb [[buffer(3)]],
                       device const float* sw [[buffer(4)]],
                       device const half2* sz [[buffer(5)]],
                       device const half* bias [[buffer(6)]],
                       device const half* res [[buffer(7)]],
                       device void* y [[buffer(8)]],
                       VV_GRID_ARGS) {
    threadgroup uint sA[64][9];
    threadgroup uint sB[64][9];
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = (int)blockIdx.x * 64, n0 = (int)blockIdx.y * 64;
    const int K = p.K, M = p.M, N = p.N;
    const int ng = W4 ? K / p.G : 1;
    const int ktiles = K >> 5;
    const int tpg = W4 ? p.G >> 5 : 1;

    int acc[4][4];
    float facc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) { acc[i][j] = 0; facc[i][j] = 0.0f; }

    for (int kt = 0; kt < ktiles; kt++) {
        const int k0 = kt << 5;
        if (tid < 128) {
            const int r = tid >> 1, h = tid & 1;
            const int gm = m0 + r;
            uint4 v = uint4(0);
            if (gm < M) v = *(device const uint4*)(xq + (ulong)gm * (ulong)K + (ulong)(k0 + h * 16));
            sA[r][h * 4 + 0] = v.x; sA[r][h * 4 + 1] = v.y;
            sA[r][h * 4 + 2] = v.z; sA[r][h * 4 + 3] = v.w;
        } else if (!W4) {
            const int t = tid - 128, r = t >> 1, h = t & 1;
            const int gn = n0 + r;
            uint4 v = uint4(0);
            if (gn < N) v = *(device const uint4*)(wb + (ulong)gn * (ulong)K + (ulong)(k0 + h * 16));
            sB[r][h * 4 + 0] = v.x; sB[r][h * 4 + 1] = v.y;
            sB[r][h * 4 + 2] = v.z; sB[r][h * 4 + 3] = v.w;
        } else if (tid < 192) {
            const int r = tid - 128;
            const int gn = n0 + r;
            uint4 v = uint4(0);
            int z = 0;
            if (gn < N) {
                v = *(device const uint4*)(wb + (ulong)gn * (ulong)(K >> 1) + (ulong)(k0 >> 1));
                z = (int)(float)sz[(ulong)gn * (ulong)ng + (ulong)(kt / tpg)].y;
            }
            const uint words[4] = { v.x, v.y, v.z, v.w };
            for (int q = 0; q < 4; q++) {
                sB[r][q]     = gn < N ? unpack4_sub(words[q], z) : 0u;
                sB[r][4 + q] = gn < N ? unpack4_sub(words[q] >> 16, z) : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int wi = 0; wi < 8; wi++) {
            uint a[4], b[4];
            for (int i = 0; i < 4; i++) a[i] = sA[ty + 16 * i][wi];
            for (int j = 0; j < 4; j++) b[j] = sB[tx + 16 * j][wi];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = dp4a(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (W4 && ((kt + 1) % tpg) == 0) {
            const int g = kt / tpg;
            for (int j = 0; j < 4; j++) {
                const int gn = n0 + tx + 16 * j;
                const float s = gn < N ? (float)sz[(ulong)gn * (ulong)ng + (ulong)g].x : 0.0f;
                for (int i = 0; i < 4; i++) {
                    facc[i][j] = fma((float)acc[i][j], s, facc[i][j]);
                    acc[i][j] = 0;
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        const int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
        const float sxm = sx[gm];
        for (int j = 0; j < 4; j++) {
            const int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            float v = W4 ? facc[i][j] * sxm : (float)acc[i][j] * sxm * sw[gn];
            const ulong idx = (ulong)gm * (ulong)N + (ulong)gn;
            if (p.has_bias) v += (float)bias[gn];
            if (p.has_res) v += (float)res[idx];
            if (p.y_f32) ((device float*)y)[idx] = v;
            else         ((device half*)y)[idx] = (half)v;
        }
    }
}

typedef decltype(vv_i8_gemm<false>) vv_i8_gemm_t;
template [[host_name("vv_i8_gemm_w8")]] kernel vv_i8_gemm_t vv_i8_gemm<false>;
template [[host_name("vv_i8_gemm_w4")]] kernel vv_i8_gemm_t vv_i8_gemm<true>;
