/*
 * vae.metal -- the batched, streaming Conv-VAE encoder (vae_kernels.cu).
 *
 * Activations are channel-first and packed along time; convolutions find
 * item boundaries in the descriptor table, which arrives in the parameter
 * struct with GPU addresses where the CUDA table has device pointers.
 *
 * The same invariants as the CUDA kernels: every output sums its products in
 * one fixed order whatever the tile, batch or chunking, so batched ==
 * one by one, chunked == whole, and conv GEMM == direct conv, bit for bit.
 * The convolutions fuse each product into the sum (fma), as nvcc does to
 * `sum += w * x`; the FFN GEMMs accumulate 8 at a time in K order on
 * simdgroup matrices.
 */

#define VAE_MAX_ITEMS 32

struct vv_vae_item {
    long in_off;
    long out_off;
    device const half* in_ptr;
    device const half* tail_in;
    device half* tail_out;
    device half* out_ptr;
    int in_len; int out_len; int have; int keep; int out_ld; int skip;
};

struct vv_vae_desc {
    int n; int cap;
    vv_vae_item it[VAE_MAX_ITEMS];
};

/* ─── Direct convolution ────────────────────────────────────────────────── */

struct vv_vae_conv_p {
    vv_vae_desc d;
    long ld_in; long ld_out;
    int in_ch; int k; int stride; int has_bias;
};

template <bool TR>
kernel void vv_vae_conv(constant vv_vae_conv_p& p [[buffer(0)]],
                        device const half* x [[buffer(1)]],
                        device const half* w [[buffer(2)]],
                        device const half* b [[buffer(3)]],
                        device half* y [[buffer(4)]],
                        VV_GRID_ARGS) {
    constant vv_vae_item& it = p.d.it[blockIdx.z];
    const int oc = (int)blockIdx.y;
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= it.out_len) return;
    device const half* xb = it.in_ptr ? it.in_ptr : x + it.in_off;
    device const half* tl = it.tail_in;
    const int have = it.have, total = have + it.in_len;
    device const half* wr = w + (ulong)oc * (ulong)p.in_ch * (ulong)p.k;
    float sum = 0.0f;
    for (int ic = 0; ic < p.in_ch; ic++) {
        for (int kk = 0; kk < p.k; kk++) {
            const int pos = t * p.stride + kk;
            if (pos < total) {
                float xv;
                if (pos < have)
                    xv = tl ? (float)tl[(ulong)ic * (ulong)p.d.cap + pos] : 0.0f;
                else
                    xv = (float)xb[(ulong)ic * (ulong)p.ld_in + (ulong)(pos - have)];
                sum = fma((float)wr[ic * p.k + kk], xv, sum);
            }
        }
    }
    if (p.has_bias) sum += (float)b[oc];
    if (TR) {
        if (t >= it.skip)
            it.out_ptr[(ulong)(t - it.skip) * (ulong)it.out_ld + oc] = (half)sum;
    } else {
        y[(ulong)oc * (ulong)p.ld_out + (ulong)it.out_off + t] = (half)sum;
    }
}

typedef decltype(vv_vae_conv<false>) vv_vae_conv_t;
template [[host_name("vv_vae_conv")]] kernel vv_vae_conv_t vv_vae_conv<false>;
template [[host_name("vv_vae_conv_tr")]] kernel vv_vae_conv_t vv_vae_conv<true>;

/* ─── Streaming state ───────────────────────────────────────────────────── */

struct vv_vae_tail_p { vv_vae_desc d; long ld_in; int in_ch; int pad; };

kernel void vv_vae_tail(constant vv_vae_tail_p& p [[buffer(0)]],
                        device const half* x [[buffer(1)]],
                        VV_GRID_ARGS) {
    constant vv_vae_item& it = p.d.it[blockIdx.y];
    if (!it.tail_out || it.keep <= 0) return;
    const int keep = it.keep;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= p.in_ch * keep) return;
    const int c = idx / keep, j = idx - c * keep;
    const int pos = it.have + it.in_len - keep + j;
    half v;
    if (pos < it.have) {
        v = it.tail_in ? it.tail_in[(ulong)c * (ulong)p.d.cap + pos] : (half)0.0f;
    } else {
        device const half* xb = it.in_ptr ? it.in_ptr : x + it.in_off;
        v = xb[(ulong)c * (ulong)p.ld_in + (ulong)(pos - it.have)];
    }
    it.tail_out[(ulong)c * (ulong)p.d.cap + j] = v;
}

/* ─── im2col and the conv GEMM ──────────────────────────────────────────── */

struct vv_vae_im2col_p { vv_vae_desc d; long ld_in; long p0; long ldcol;
                         int k; int stride; int pc; int pad; };

kernel void vv_vae_im2col(constant vv_vae_im2col_p& p [[buffer(0)]],
                          device const half* x [[buffer(1)]],
                          device half* col [[buffer(2)]],
                          VV_GRID_ARGS) {
    const int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= p.pc) return;
    const int kr = (int)blockIdx.y;
    const int ic = kr / p.k, kk = kr - ic * p.k;
    const long pp = p.p0 + q;
    int i = 0;
    while (i + 1 < p.d.n && pp >= p.d.it[i + 1].out_off) i++;
    constant vv_vae_item& it = p.d.it[i];
    const int t = (int)(pp - it.out_off);
    const int pos = t * p.stride + kk;
    half v = (half)0.0f;
    if (t < it.out_len && pos < it.have + it.in_len) {
        if (pos < it.have) {
            if (it.tail_in) v = it.tail_in[(ulong)ic * (ulong)p.d.cap + pos];
        } else {
            device const half* xb = it.in_ptr ? it.in_ptr : x + it.in_off;
            v = xb[(ulong)ic * (ulong)p.ld_in + (ulong)(pos - it.have)];
        }
    }
    col[(ulong)kr * (ulong)p.ldcol + (ulong)q] = v;
}

struct vv_vae_cgemm_p { vv_vae_desc d; long ldb; long ldc; long p0;
                        int M; int K; int P; int has_bias; };

/*
 * FP32 tiled GEMM of the [out][in * k] weight against im2col columns: each
 * output fuses its products into one sum in (in channel, tap) order from
 * zero, then adds the bias -- vv_vae_conv's arithmetic, so the bits are the
 * same whatever the tile. TM x TN outputs per group of 256, RM x RN each.
 */
template <bool TR, int TM, int TN, int RM, int RN, int TBK>
kernel void vv_vae_conv_gemm(constant vv_vae_cgemm_p& p [[buffer(0)]],
                             device const half* A [[buffer(1)]],
                             device const half* B [[buffer(2)]],
                             device half* C [[buffer(3)]],
                             device const half* bias [[buffer(4)]],
                             VV_GRID_ARGS) {
    threadgroup float As[TBK][TM];
    threadgroup float Bs[TBK][TN];
    const int tid = (int)threadIdx.x;
    const int tx = tid % (TN / RN), ty = tid / (TN / RN);
    const int row_base = (int)blockIdx.y * TM, col_base = (int)blockIdx.x * TN;
    const int ar = tid / (TBK / 4), ac = (tid % (TBK / 4)) * 4;
    const int br = tid / (TN / 4), bc = (tid % (TN / 4)) * 4;

    float acc[RM][RN];
    for (int i = 0; i < RM; i++)
        for (int j = 0; j < RN; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < p.K; k0 += TBK) {
        {
            const int gr = row_base + ar, gk = k0 + ac;
            for (int t = 0; t < 4; t++)
                As[ac + t][ar] = (gr < p.M && gk + t < p.K)
                               ? (float)A[(ulong)gr * (ulong)p.K + (ulong)(gk + t)] : 0.0f;
            const int bk = k0 + br, gc = col_base + bc;
            for (int t = 0; t < 4; t++)
                Bs[br][bc + t] = (bk < p.K && gc + t < p.P)
                               ? (float)B[(ulong)bk * (ulong)p.ldb + (ulong)(gc + t)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TBK; kk++) {
            float av[RM], bv[RN];
            for (int i = 0; i < RM; i++) av[i] = As[kk][ty * RM + i];
            for (int j = 0; j < RN; j++) bv[j] = Bs[kk][tx * RN + j];
            for (int i = 0; i < RM; i++)
                for (int j = 0; j < RN; j++) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int j = 0; j < RN; j++) {
        const int c = col_base + tx * RN + j;
        if (c >= p.P) continue;
        device half* dst;
        ulong step;
        if (TR) {
            const long pp = p.p0 + c;
            int it_i = 0;
            while (it_i + 1 < p.d.n && pp >= p.d.it[it_i + 1].out_off) it_i++;
            constant vv_vae_item& it = p.d.it[it_i];
            const int t = (int)(pp - it.out_off);
            if (t >= it.out_len || t < it.skip) continue;
            dst = it.out_ptr + (ulong)(t - it.skip) * (ulong)it.out_ld;
            step = 1;
        } else {
            dst = C + c;
            step = (ulong)p.ldc;
        }
        for (int i = 0; i < RM; i++) {
            const int r = row_base + ty * RM + i;
            if (r >= p.M) continue;
            float v = acc[i][j];
            if (p.has_bias) v += (float)bias[r];
            dst[(ulong)r * step] = (half)v;
        }
    }
}

typedef decltype(vv_vae_conv_gemm<false, 64, 64, 4, 4, 16>) vv_vae_conv_gemm_t;
template [[host_name("vv_vae_conv_gemm_64")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<false, 64, 64, 4, 4, 16>;
template [[host_name("vv_vae_conv_gemm_32")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<false, 32, 32, 2, 2, 32>;
template [[host_name("vv_vae_conv_gemm_16")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<false, 16, 16, 1, 1, 64>;
template [[host_name("vv_vae_conv_gemm_tr_64")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<true, 64, 64, 4, 4, 16>;
template [[host_name("vv_vae_conv_gemm_tr_32")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<true, 32, 32, 2, 2, 32>;
template [[host_name("vv_vae_conv_gemm_tr_16")]] kernel vv_vae_conv_gemm_t vv_vae_conv_gemm<true, 16, 16, 1, 1, 64>;

/* ─── RMSNorm statistics ────────────────────────────────────────────────── */

struct vv_vae_rms_p { vv_vae_desc d; long ld; long T; int C; float eps; };

kernel void vv_vae_rms(constant vv_vae_rms_p& p [[buffer(0)]],
                       device const half* x [[buffer(1)]],
                       device float* rinv [[buffer(2)]],
                       VV_GRID_ARGS) {
    const long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    device const half* col;
    long stride;
    if (i < p.T) {
        col = x + i;
        stride = p.ld;
    } else {
        const long j = i - p.T;
        if (p.d.cap <= 0) return;
        const int item = (int)(j / p.d.cap), q = (int)(j - (long)item * p.d.cap);
        if (item >= p.d.n) return;
        constant vv_vae_item& it = p.d.it[item];
        if (!it.tail_in || q >= it.have) return;
        col = it.tail_in + q;
        stride = p.d.cap;
    }
    float sum_sq = 0.0f;
    for (int c = 0; c < p.C; c++) {
        const float v = (float)col[(ulong)c * (ulong)stride];
        sum_sq += v * v;
    }
    rinv[i] = rsqrt(sum_sq / (float)p.C + p.eps);
}

/* ─── Mixer: RMSNorm -> depthwise conv -> gamma residual ────────────────── */

#define MIX_TT 64
#define MIX_CG 32
#define MIX_MAXK 16

struct vv_vae_mixer_p { vv_vae_desc d; long ld; long T_total; int C; int k;
                        int has_cb; int has_gamma; };

kernel void vv_vae_mixer(constant vv_vae_mixer_p& p [[buffer(0)]],
                         device const half* xin [[buffer(1)]],
                         device half* xout [[buffer(2)]],
                         device const float* rinv [[buffer(3)]],
                         device const half* nw [[buffer(4)]],
                         device const half* cw [[buffer(5)]],
                         device const half* cb [[buffer(6)]],
                         device const half* gamma [[buffer(7)]],
                         VV_GRID_ARGS) {
    threadgroup half nv[MIX_CG][MIX_TT + MIX_MAXK];
    const int item = (int)blockIdx.z;
    constant vv_vae_item& it = p.d.it[item];
    const int t0 = (int)blockIdx.x * MIX_TT;
    if (t0 >= it.in_len) return;                   /* whole group */
    const int c0 = (int)blockIdx.y * MIX_CG;
    const int H = p.k - 1, W = MIX_TT + H;
    device const half* tl = it.tail_in;

    for (int e = (int)threadIdx.x; e < MIX_CG * W; e += (int)blockDim.x) {
        const int cl = e / W, q = e - cl * W;
        const int c = c0 + cl;
        half out = (half)0.0f;
        if (c < p.C) {
            const int pos = t0 - H + q;
            if (pos >= 0) {
                if (pos < it.in_len) {
                    const long colx = it.in_off + pos;
                    const float v = (float)xin[(ulong)c * (ulong)p.ld + (ulong)colx];
                    out = (half)(v * rinv[colx] * (float)nw[c]);
                }
            } else {
                const int tp = it.have + pos;
                if (tl && tp >= 0) {
                    const float v = (float)tl[(ulong)c * (ulong)p.d.cap + tp];
                    const float r = rinv[p.T_total + (long)item * p.d.cap + tp];
                    out = (half)(v * r * (float)nw[c]);
                }
            }
        }
        nv[cl][q] = out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = (int)threadIdx.x; e < MIX_CG * MIX_TT; e += (int)blockDim.x) {
        const int cl = e / MIX_TT, tt = e - cl * MIX_TT;
        const int c = c0 + cl, t = t0 + tt;
        if (c >= p.C || t >= it.in_len) continue;
        float sum = 0.0f;
        for (int kk = 0; kk < p.k; kk++)
            sum = fma((float)cw[c * p.k + kk], (float)nv[cl][tt + kk], sum);
        if (p.has_cb) sum += (float)cb[c];
        const float yv = (float)(half)sum;
        const ulong off = (ulong)c * (ulong)p.ld + (ulong)it.in_off + (ulong)t;
        const float xv = (float)xin[off];
        xout[off] = p.has_gamma ? (half)(xv + yv * (float)gamma[c]) : (half)(xv + yv);
    }
}

/* ─── FFN GEMM with epilogues ───────────────────────────────────────────── */

#define VE_BIAS_GELU  0
#define VE_BIAS_RESID 1
#define VE_BIAS       2

#define VG_BM  64
#define VG_BN  64
#define VG_BK  32
#define VG_LDS (VG_BK + 8)
#define VG_LDN (VG_BN + 8)
#define VG_LDC (32 + 4)

struct vv_vae_gemm_p { long ldb; long ldc; int M; int K; int P;
                       int has_bias; int has_gamma; int has_norm; };

template <int EPI>
static inline void vg_epilogue(float acc, device half* C, ulong off, int r,
                               device const half* bias, device const half* gamma,
                               int has_bias, int has_gamma) {
    if (EPI == VE_BIAS) {
        const float b = has_bias ? (float)bias[r] : 0.0f;
        C[off] = (half)(acc + b);
        return;
    }
    float v = (float)(half)acc;
    if (has_bias) v = (float)(half)(v + (float)bias[r]);
    if (EPI == VE_BIAS_GELU) {
        C[off] = (half)vv_gelu(v);
    } else {
        const float xv = (float)C[off];
        C[off] = has_gamma ? (half)(xv + v * (float)gamma[r]) : (half)(xv + v);
    }
}

/*
 * C[M,P] (ldc) from A[M,K] . B[K,P] (ldb): 64x64 per group of four
 * simdgroups, each a 32x32 quadrant of 8x8 FP32 accumulators, K in steps
 * of 32 through threadgroup memory. With has_norm, B is RMS-normalised as it
 * is staged: half(x * rinv[col] * nw[row]), the value the separate norm
 * stored. Every element accumulates in K order 8 at a time, so the tile
 * (and how the columns are split into calls) does not change its bits.
 */
template <int EPI>
kernel void vv_vae_gemm_nn(constant vv_vae_gemm_p& p [[buffer(0)]],
                           device const half* A [[buffer(1)]],
                           device const half* B [[buffer(2)]],
                           device half* C [[buffer(3)]],
                           device const half* bias [[buffer(4)]],
                           device const half* gamma [[buffer(5)]],
                           device const float* rinv [[buffer(6)]],
                           device const half* nw [[buffer(7)]],
                           threadgroup uchar* smem [[threadgroup(0)]],
                           VV_GRID_ARGS) {
    threadgroup half* As = (threadgroup half*)smem;          /* [64][LDS]   */
    threadgroup half* Bs = As + VG_BM * VG_LDS;               /* [32][LDN]   */
    const uint tid = threadIdx.x;
    const int row0 = (int)blockIdx.y * VG_BM, col0 = (int)blockIdx.x * VG_BN;
    const int wm = (int)warp >> 1, wn = (int)warp & 1;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += VG_BK) {
        for (uint e = tid; e < 64u * 4u; e += 128u) {
            const int r = (int)(e >> 2), c = (int)(e & 3u) * 8;
            const int gr = row0 + r, gk = k0 + c;
            threadgroup half* d = As + r * VG_LDS + c;
            for (int t = 0; t < 8; t++)
                d[t] = (gr < p.M && gk + t < p.K) ? A[(ulong)gr * (ulong)p.K + (ulong)(gk + t)] : (half)0.0f;
        }
        for (uint e = tid; e < (uint)VG_BK * 8u; e += 128u) {
            const int r = (int)(e >> 3), c = (int)(e & 7u) * 8;
            const int gk = k0 + r, gc = col0 + c;
            threadgroup half* d = Bs + r * VG_LDN + c;
            const float wn_ = (p.has_norm && gk < p.K) ? (float)nw[gk] : 0.0f;
            for (int t = 0; t < 8; t++) {
                half v = (half)0.0f;
                if (gk < p.K && gc + t < p.P) {
                    v = B[(ulong)gk * (ulong)p.ldb + (ulong)(gc + t)];
                    if (p.has_norm) v = (half)((float)v * rinv[gc + t] * wn_);
                }
                d[t] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < VG_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a[i], As + (wm * 32 + i * 8) * VG_LDS + kk, VG_LDS);
            for (int j = 0; j < 4; j++)
                simdgroup_load(b[j], Bs + kk * VG_LDN + wn * 32 + j * 8, VG_LDN);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* stage = (threadgroup float*)smem + (int)warp * 32 * VG_LDC;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], stage + (i * 8) * VG_LDC + j * 8, VG_LDC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int rb = row0 + wm * 32, cb0 = col0 + wn * 32;
    for (int e = (int)lane; e < 32 * 32; e += 32) {
        const int r = rb + (e >> 5), c = cb0 + (e & 31);
        if (r < p.M && c < p.P)
            vg_epilogue<EPI>(stage[(e >> 5) * VG_LDC + (e & 31)], C,
                             (ulong)r * (ulong)p.ldc + (ulong)c, r, bias, gamma,
                             p.has_bias, p.has_gamma);
    }
}

typedef decltype(vv_vae_gemm_nn<VE_BIAS>) vv_vae_gemm_nn_t;
template [[host_name("vv_vae_gemm_nn_gelu")]] kernel vv_vae_gemm_nn_t vv_vae_gemm_nn<VE_BIAS_GELU>;
template [[host_name("vv_vae_gemm_nn_resid")]] kernel vv_vae_gemm_nn_t vv_vae_gemm_nn<VE_BIAS_RESID>;
template [[host_name("vv_vae_gemm_nn_bias")]] kernel vv_vae_gemm_nn_t vv_vae_gemm_nn<VE_BIAS>;

/* ─── TN GEMM for the speech connectors ─────────────────────────────────── */

#define VV_ROWS_PLAIN       0
#define VV_ROWS_STORE       1
#define VV_ROWS_STORE_TRUNC 2
#define VV_ROWS_ACC_TRUNC   3

struct vv_vae_row_seg { int row0; int ld; device half* dst; };
struct vv_vae_rows { int mode; int n; vv_vae_row_seg seg[VAE_MAX_ITEMS]; };

struct vv_vae_gemm_tn_p { vv_vae_rows rows; long lda; long ldc;
                          int M; int N; int K; int has_bias; };

/* The FP32 sum truncated to FP16, subnormals flushed: the rounding the
 * connectors' host-side sum used to apply. */
static inline half vae_trunc_half(float f) {
    const uint u = as_type<uint>(f);
    const uint sign = (u >> 16) & 0x8000u;
    const int e = (int)((u >> 23) & 0xFFu) - 127 + 15;
    const uint frac = (u >> 13) & 0x3FFu;
    ushort h;
    if (e <= 0) h = (ushort)sign;
    else if (e >= 31) h = (ushort)(sign | 0x7C00u);
    else h = (ushort)(sign | ((uint)e << 10) | frac);
    return as_type<half>(h);
}

kernel void vv_vae_gemm_tn(constant vv_vae_gemm_tn_p& p [[buffer(0)]],
                           device const half* A [[buffer(1)]],
                           device const half* B [[buffer(2)]],
                           device half* C [[buffer(3)]],
                           device const half* bias [[buffer(4)]],
                           threadgroup uchar* smem [[threadgroup(0)]],
                           VV_GRID_ARGS) {
    threadgroup half* As = (threadgroup half*)smem;          /* [64][LDS] */
    threadgroup half* Bs = As + VG_BM * VG_LDS;               /* [64][LDS] */
    const uint tid = threadIdx.x;
    const int row0 = (int)blockIdx.y * VG_BM, col0 = (int)blockIdx.x * VG_BN;
    const int wm = (int)warp >> 1, wn = (int)warp & 1;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += VG_BK) {
        for (uint e = tid; e < 64u * 4u; e += 128u) {
            const int r = (int)(e >> 2), c = (int)(e & 3u) * 8;
            const int ga = row0 + r, gb = col0 + r, gk = k0 + c;
            threadgroup half* da = As + r * VG_LDS + c;
            threadgroup half* db = Bs + r * VG_LDS + c;
            for (int t = 0; t < 8; t++) {
                da[t] = (ga < p.M && gk + t < p.K) ? A[(ulong)ga * (ulong)p.lda + (ulong)(gk + t)] : (half)0.0f;
                db[t] = (gb < p.N && gk + t < p.K) ? B[(ulong)gb * (ulong)p.K + (ulong)(gk + t)] : (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < VG_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a[i], As + (wm * 32 + i * 8) * VG_LDS + kk, VG_LDS);
            for (int j = 0; j < 4; j++)
                simdgroup_load(b[j], Bs + (wn * 32 + j * 8) * VG_LDS + kk, VG_LDS,
                               ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* stage = (threadgroup float*)smem + (int)warp * 32 * VG_LDC;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], stage + (i * 8) * VG_LDC + j * 8, VG_LDC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int rb = row0 + wm * 32, cb0 = col0 + wn * 32;
    for (int e = (int)lane; e < 32 * 32; e += 32) {
        const int r = rb + (e >> 5), c = cb0 + (e & 31);
        if (r >= p.M || c >= p.N) continue;
        half yv = (half)stage[(e >> 5) * VG_LDC + (e & 31)];
        if (p.has_bias) yv = (half)((float)yv + (float)bias[c]);
        if (p.rows.mode == VV_ROWS_PLAIN) {
            C[(ulong)r * (ulong)p.ldc + (ulong)c] = yv;
            continue;
        }
        int s = 0;
        while (s + 1 < p.rows.n && r >= p.rows.seg[s + 1].row0) s++;
        device half* dst = p.rows.seg[s].dst
                         + (ulong)(r - p.rows.seg[s].row0) * (ulong)p.rows.seg[s].ld + (ulong)c;
        if (p.rows.mode == VV_ROWS_STORE)
            *dst = yv;
        else if (p.rows.mode == VV_ROWS_STORE_TRUNC)
            *dst = vae_trunc_half((float)yv);
        else
            *dst = vae_trunc_half((float)*dst + (float)yv);
    }
}
