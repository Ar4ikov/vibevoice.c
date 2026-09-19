/*
 * bitnet.metal -- the integer kernels of VibeVoice-ASR-BitNet (bitnet.cu),
 * with the numerics of src/cpu/bitnet_cpu.c so a GPU result equals the CPU
 * one bit for bit:
 *
 *   ternary x int8   sum_k (c - 1) q = sum_k c q - sum_k q, codes unsigned
 *   int8 x int8      the I8_S encoder's GEMMs, absmax folded for requant
 *   int8 head        GEMV fused with a two-stage argmax
 *
 * The CPU file is built with -ffp-contract=off; every function here that
 * does float arithmetic says the same, since the library otherwise fuses
 * a*b+c inside an expression. Dot products are exact int32.
 */

#define BN_TERN_MASK 0x03030303u

static inline int bn_dp4a(uint a, uint b, int c) {
    const int4 x = int4(as_type<char4>(a)), y = int4(as_type<char4>(b));
    return c + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
}

/* ─── Activation quantization ───────────────────────────────────────────── */

struct vv_bn_aq_p { int K; int x_f16; int has_sum; };

kernel void vv_bn_act_quant(constant vv_bn_aq_p& p [[buffer(0)]],
                            device const void* x [[buffer(1)]],
                            device char* q [[buffer(2)]],
                            device float* scale [[buffer(3)]],
                            device int* sum [[buffer(4)]],
                            VV_GRID_ARGS) {
    #pragma METAL fp contract(off)
    threadgroup float s_max[8];
    threadgroup int s_sum[8];
    threadgroup float s_scale;
    const ulong m = blockIdx.x;
    const ulong base = m * (ulong)p.K;
    device const half* xh = (device const half*)x;
    device const float* xf = (device const float*)x;

    float amax = 0.0f;
    for (int k = (int)threadIdx.x; k < p.K; k += 256)
        amax = max(amax, fabs(p.x_f16 ? (float)xh[base + k] : xf[base + k]));
    amax = vv_warp_max(amax);
    if (lane == 0) s_max[warp] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIdx.x == 0) {
        float mx = 0.0f;
        for (int w = 0; w < 8; w++) mx = max(mx, s_max[w]);
        /*
         * The CPU computes (float)(127.0 / max(amax, 1e-5)) in double. For
         * a float amax that is the correctly rounded float quotient (a
         * division rounded to 53 bits and then to 24 rounds as if once),
         * which is what `/` gives here; at the floor it is 12700000.
         */
        s_scale = mx > 0.00001f ? 127.0f / mx : 12700000.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = s_scale;

    int acc = 0;
    for (int k = (int)threadIdx.x; k < p.K; k += 256) {
        const float v = p.x_f16 ? (float)xh[base + k] : xf[base + k];
        /* ggml's nearest_int(x * s), contracted as the reference build is:
         * one FMA with 1.5 * 2^23 */
        const float t = fma(v, s, 12582912.0f);
        int iv = (as_type<int>(t) & 0x007fffff) - 0x00400000;
        iv = iv > 127 ? 127 : (iv < -128 ? -128 : iv);
        q[base + k] = (char)iv;
        acc += iv;
    }
    acc += simd_shuffle_xor(acc, 16); acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);  acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) s_sum[warp] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < 8; w++) t += s_sum[w];
        scale[m] = s;
        if (p.has_sum) sum[m] = t;
    }
}

/* ─── Shared epilogue ───────────────────────────────────────────────────── */

struct vv_bn_epi {
    int M; int N; int K; int tern;
    float w_scale; int y_f16;
    int has_acc; int has_y; int has_bias; int has_absmax;
};

/* One output: returns |y| for the absmax fold (0 when there is no y). */
static inline float bn_store(constant vv_bn_epi& e, int m, int n, int v,
                             device const int* xsum, device const float* x_scale,
                             device const float* a_scale, device const float* bias,
                             device int* acc, device void* y) {
    #pragma METAL fp contract(off)
    if (e.tern) v -= xsum[m];
    const ulong o = (ulong)m * (ulong)e.N + (ulong)n;
    if (e.has_acc) acc[o] = v;
    if (!e.has_y) return 0.0f;
    float r;
    if (e.tern) {
        const float d = (float)v / x_scale[m];
        r = d * e.w_scale;
    } else {
        const float f = e.w_scale / *a_scale;
        r = (float)v * f;
    }
    if (e.has_bias) r = r + bias[n];
    if (e.y_f16) ((device half*)y)[o] = (half)r;
    else ((device float*)y)[o] = r;
    return fabs(r);
}

/* ─── GEMV (M <= 8), one simdgroup per weight row ───────────────────────── */

/* Ternary rows are K/16 words; word i covers block i/8, bytes (i%8)*4..+3,
 * and its four 2-bit fields meet x at +0, +32, +64 and +96. */
kernel void vv_bn_tern_gemv(constant vv_bn_epi& e [[buffer(0)]],
                            device const char* q [[buffer(1)]],
                            device const uint* codes [[buffer(2)]],
                            device const int* xsum [[buffer(3)]],
                            device const float* x_scale [[buffer(4)]],
                            device const float* bias [[buffer(5)]],
                            device int* acc [[buffer(6)]],
                            device void* y [[buffer(7)]],
                            VV_GRID_ARGS) {
    const int n = (int)blockIdx.x * 4 + (int)warp;
    if (n >= e.N) return;
    device const uint* wrow = codes + (ulong)n * (ulong)(e.K / 16);
    const int words = e.K / 16;
    int a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int wi = (int)lane; wi < words; wi += 32) {
        const uint w = wrow[wi];
        const uint c0 = (w >> 6) & BN_TERN_MASK, c1 = (w >> 4) & BN_TERN_MASK;
        const uint c2 = (w >> 2) & BN_TERN_MASK, c3 = w & BN_TERN_MASK;
        const int xo = (wi >> 3) * 128 + (wi & 7) * 4;
        for (int m = 0; m < 8; m++) {
            if (m >= e.M) break;
            device const uint* xp = (device const uint*)(q + (ulong)m * (ulong)e.K + (ulong)xo);
            a[m] = bn_dp4a(c0, xp[0], a[m]);
            a[m] = bn_dp4a(c1, xp[8], a[m]);
            a[m] = bn_dp4a(c2, xp[16], a[m]);
            a[m] = bn_dp4a(c3, xp[24], a[m]);
        }
    }
    for (int m = 0; m < 8; m++) {
        if (m >= e.M) break;
        int v = a[m];
        v += simd_shuffle_xor(v, 16); v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);  v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0) bn_store(e, m, n, v, xsum, x_scale, nullptr, bias, acc, y);
    }
}

kernel void vv_bn_i8_gemv(constant vv_bn_epi& e [[buffer(0)]],
                          device const char* a [[buffer(1)]],
                          device const char* w [[buffer(2)]],
                          device const float* a_scale [[buffer(3)]],
                          device const float* bias [[buffer(4)]],
                          device int* acc [[buffer(5)]],
                          device void* y [[buffer(6)]],
                          device atomic_int* absmax [[buffer(7)]],
                          VV_GRID_ARGS) {
    const int n = (int)blockIdx.x * 4 + (int)warp;
    if (n >= e.N) return;
    int s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    device const char* wr = w + (ulong)n * (ulong)e.K;
    if ((e.K & 3) == 0) {
        device const uint* w4 = (device const uint*)wr;
        for (int i = (int)lane; i < e.K / 4; i += 32) {
            const uint wv = w4[i];
            for (int m = 0; m < 8; m++) {
                if (m >= e.M) break;
                s[m] = bn_dp4a(wv, ((device const uint*)(a + (ulong)m * (ulong)e.K))[i], s[m]);
            }
        }
    } else {
        for (int k = (int)lane; k < e.K; k += 32) {
            const int wv = wr[k];
            for (int m = 0; m < 8; m++) {
                if (m >= e.M) break;
                s[m] += wv * (int)a[(ulong)m * (ulong)e.K + (ulong)k];
            }
        }
    }
    float amax = 0.0f;
    for (int m = 0; m < 8; m++) {
        if (m >= e.M) break;
        int v = s[m];
        v += simd_shuffle_xor(v, 16); v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);  v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0) amax = max(amax, bn_store(e, m, n, v, nullptr, nullptr, a_scale, bias, acc, y));
    }
    if (e.has_absmax) {
        amax = vv_warp_max(amax);
        if (lane == 0 && amax > 0.0f)
            atomic_fetch_max_explicit(absmax, as_type<int>(amax), memory_order_relaxed);
    }
}

/*
 * Any M: 64x64 tiles, 256 threads with 4x4 outputs, K in steps of 32.
 * Ternary B is unpacked from its 2-bit codes into bytes 0..2 in threadgroup
 * memory; the row sum of q is subtracted in the epilogue as on the CPU.
 */
template <bool TERN>
kernel void vv_bn_gemm(constant vv_bn_epi& e [[buffer(0)]],
                       device const char* A [[buffer(1)]],
                       device const uchar* B [[buffer(2)]],
                       device const int* xsum [[buffer(3)]],
                       device const float* x_scale [[buffer(4)]],
                       device const float* a_scale [[buffer(5)]],
                       device const float* bias [[buffer(6)]],
                       device int* acc_out [[buffer(7)]],
                       device void* y [[buffer(8)]],
                       device atomic_int* absmax [[buffer(9)]],
                       VV_GRID_ARGS) {
    threadgroup uint sA[64][9];
    threadgroup uint sB[64][9];
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = (int)blockIdx.x * 64, n0 = (int)blockIdx.y * 64;
    const int M = e.M, N = e.N, K = e.K;
    int acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < K; k0 += 32) {
        if (tid < 128) {
            const int r = tid >> 1, h = tid & 1;
            const int gm = m0 + r;
            for (int wv = 0; wv < 4; wv++) {
                uint word = 0;
                for (int b = 0; b < 4; b++) {
                    const int k = k0 + h * 16 + wv * 4 + b;
                    const int v = (gm < M && k < K) ? (int)A[(ulong)gm * (ulong)K + (ulong)k] : 0;
                    word |= ((uint)v & 0xFFu) << (8 * b);
                }
                sA[r][h * 4 + wv] = word;
            }
        } else {
            const int t = tid - 128, r = t >> 1, h = t & 1;
            const int gn = n0 + r;
            for (int wv = 0; wv < 4; wv++) {
                uint word = 0;
                for (int b = 0; b < 4; b++) {
                    const int k = k0 + h * 16 + wv * 4 + b;
                    int v = 0;
                    if (gn < N && k < K) {
                        if (TERN) {
                            /* byte k of the row: block k/128, word j of it
                             * holds x offsets j*4 + {0..3} + {0,32,64,96} */
                            const int blk = k >> 7, kk = k & 127;
                            const int field = kk >> 5, j = (kk & 31) >> 2, byte = kk & 3;
                            const uint w = ((device const uint*)(B + (ulong)gn * (ulong)(K / 4)))[blk * 8 + j];
                            v = (int)((w >> (6 - 2 * field + 8 * byte)) & 3u);
                        } else {
                            v = (int)((device const char*)B)[(ulong)gn * (ulong)K + (ulong)k];
                        }
                    }
                    word |= ((uint)v & 0xFFu) << (8 * b);
                }
                sB[r][h * 4 + wv] = word;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int wi = 0; wi < 8; wi++) {
            uint a[4], b[4];
            for (int i = 0; i < 4; i++) a[i] = sA[ty + 16 * i][wi];
            for (int j = 0; j < 4; j++) b[j] = sB[tx + 16 * j][wi];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = bn_dp4a(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float amax = 0.0f;
    for (int i = 0; i < 4; i++) {
        const int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
        for (int j = 0; j < 4; j++) {
            const int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            amax = max(amax, bn_store(e, gm, gn, acc[i][j], xsum, x_scale, a_scale,
                                      bias, acc_out, y));
        }
    }
    if (!TERN && e.has_absmax) {
        amax = vv_warp_max(amax);
        if (lane == 0 && amax > 0.0f)
            atomic_fetch_max_explicit(absmax, as_type<int>(amax), memory_order_relaxed);
    }
}

typedef decltype(vv_bn_gemm<true>) vv_bn_gemm_t;
template [[host_name("vv_bn_gemm_tern")]] kernel vv_bn_gemm_t vv_bn_gemm<true>;
template [[host_name("vv_bn_gemm_i8")]] kernel vv_bn_gemm_t vv_bn_gemm<false>;

/* ─── Requantization ────────────────────────────────────────────────────── */

struct vv_bn_rq_p { ulong n; int relu; int has_out_scale; };

kernel void vv_bn_requant(constant vv_bn_rq_p& p [[buffer(0)]],
                          device const float* y [[buffer(1)]],
                          device const float* absmax [[buffer(2)]],
                          device char* q [[buffer(3)]],
                          device float* out_scale [[buffer(4)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint3 t3 [[thread_position_in_threadgroup]],
                          uint3 ntg [[threadgroups_per_grid]]) {
    #pragma METAL fp contract(off)
    const float am = *absmax;
    const float inv = am != 0.0f ? 127.0f / am : 0.0f;
    const float lo = p.relu ? 0.0f : -127.0f;
    for (ulong i = (ulong)tg.x * 256ul + t3.x; i < p.n; i += (ulong)ntg.x * 256ul) {
        float v = y[i] * inv;
        v = min(max(v, lo), 127.0f);
        q[i] = (char)(int)rint(v);
    }
    if (tg.x == 0 && t3.x == 0 && p.has_out_scale) *out_scale = inv;
}

/* ─── Int8 head + argmax ────────────────────────────────────────────────── */

static inline bool bn_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

struct vv_bn_head_p { int V; int K; int nb; int has_value; };

/* 8 simdgroups, 8 rows each; ((float)acc * w_scale) / s, the CPU order. */
kernel void vv_bn_head(constant vv_bn_head_p& p [[buffer(0)]],
                       device const char* q [[buffer(1)]],
                       device const float* scale [[buffer(2)]],
                       device const char* w [[buffer(3)]],
                       device const float* w_scale [[buffer(4)]],
                       device float* pv [[buffer(5)]],
                       device int* pi [[buffer(6)]],
                       VV_GRID_ARGS) {
    #pragma METAL fp contract(off)
    threadgroup float s_v[8];
    threadgroup int s_i[8];
    const float s = *scale;
    device const uint* x4 = (device const uint*)q;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    const int r0 = (int)blockIdx.x * 64 + (int)warp * 8;
    for (int r = 0; r < 8; r++) {
        const int n = r0 + r;
        if (n >= p.V) break;
        device const uint* w4 = (device const uint*)(w + (ulong)n * (ulong)p.K);
        int acc = 0;
        for (int i = (int)lane; i < p.K / 4; i += 32) acc = bn_dp4a(w4[i], x4[i], acc);
        acc += simd_shuffle_xor(acc, 16); acc += simd_shuffle_xor(acc, 8);
        acc += simd_shuffle_xor(acc, 4);  acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 1);
        const float prod = (float)acc * w_scale[n];
        const float v = prod / s;
        if (bn_better(v, n, bv, bi)) { bv = v; bi = n; }
    }
    if (lane == 0) { s_v[warp] = bv; s_i[warp] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIdx.x == 0) {
        float v = s_v[0];
        int i = s_i[0];
        for (int k = 1; k < 8; k++)
            if (bn_better(s_v[k], s_i[k], v, i)) { v = s_v[k]; i = s_i[k]; }
        pv[blockIdx.x] = v;
        pi[blockIdx.x] = i;
    }
}

kernel void vv_bn_head_final(constant vv_bn_head_p& p [[buffer(0)]],
                             device const float* pv [[buffer(1)]],
                             device const int* pi [[buffer(2)]],
                             device int* token [[buffer(3)]],
                             device float* value [[buffer(4)]],
                             uint3 threadIdx [[thread_position_in_threadgroup]]) {
    threadgroup float s_v[256];
    threadgroup int s_i[256];
    const int t = (int)threadIdx.x;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = t; i < p.nb; i += 256)
        if (bn_better(pv[i], pi[i], bv, bi)) { bv = pv[i]; bi = pi[i]; }
    s_v[t] = bv; s_i[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st && bn_better(s_v[t + st], s_i[t + st], s_v[t], s_i[t])) {
            s_v[t] = s_v[t + st];
            s_i[t] = s_i[t + st];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        *token = s_i[0];
        if (p.has_value) *value = s_v[0];
    }
}
