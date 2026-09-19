/*
 * attention.metal -- the KV cache formats and the attention kernels.
 *
 * Ports of src/cuda/{kv_codec.cuh,kv_quant.cu,attention.cu,
 * attention_decode.cu}: the same stored bytes for every format (FP16, FP8
 * E4M3/E5M2, TurboQuant 4/3/2/1.5 bit), the same page-table addressing,
 * the same split-KV decode with its merge. Lane L of a simdgroup owns dims
 * 4L..4L+3 of a 128-dim head, as a CUDA warp lane does, so the packed codes
 * are read and written with the same shuffles.
 *
 * Prefill has two kernels: the scalar one (fa1, one simdgroup per query
 * row) and a simdgroup-matrix one that packs the query heads of a KV group
 * into the rows of one tile, so K and V are read once for the group.
 */

#define KV_FP16     0
#define KV_FP8_E4M3 1
#define KV_FP8_E5M2 2
#define KV_TQ4      3
#define KV_TQ3      4
#define KV_TQ2      5
#define KV_TQ1_5    6
#define KV_PAGE     64
#define ATT_D       128

constant float kv_lev4[16] = {
    -2.725112f, -2.065280f, -1.613787f, -1.251845f,
    -0.938657f, -0.654379f, -0.386970f, -0.127960f,
    +0.127960f, +0.386970f, +0.654379f, +0.938657f,
    +1.251845f, +1.613787f, +2.065280f, +2.725112f
};
constant float kv_lev3[8] = {
    -2.150300f, -1.343258f, -0.755588f, -0.244756f,
    +0.244756f, +0.755588f, +1.343258f, +2.150300f
};
constant float kv_lev2[4] = { -1.511418f, -0.452456f, +0.452456f, +1.511418f };
constant float kv_vq15[16] = {
    -1.500251f, +0.674678f,  -1.127883f, -0.787015f,
    -0.283251f, +0.390133f,  -0.103102f, +1.624620f,
    +0.100120f, -1.618563f,  +0.286482f, -0.379555f,
    +1.134227f, +0.790368f,  +1.496572f, -0.666578f
};

#define WHT_NORM 0.0883883476f

static inline int kv_row(device const int* pt, bool use_pt, int t) {
    return use_pt ? pt[t / KV_PAGE] * KV_PAGE + (t % KV_PAGE) : t;
}

/* ─── FP8, emulated (bit for bit as kv_codec.cuh) ───────────────────────── */

static inline uint f2e4m3(float f) {
    const uint b = as_type<uint>(f);
    const uint s = (b >> 24) & 0x80u;
    const float a = fabs(f);
    if (!(a > 0.0f)) return s;
    if (a >= 464.0f) return s | 0x7Eu;
    int e = (int)((b >> 23) & 0xFFu) - 127;
    if (e < -6) {
        int q = (int)(a * 512.0f + 0.5f);
        if (q > 8) q = 8;
        return s | (uint)q;
    }
    uint mm = (b & 0x7FFFFFu) >> 20;
    const uint rem = b & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (mm & 1u))) {
        if (++mm == 8u) { mm = 0; ++e; }
    }
    if (e > 8) return s | 0x7Eu;
    return s | ((uint)(e + 7) << 3) | mm;
}

static inline float e4m32f(uint v) {
    const uint e = (v >> 3) & 0xFu, m = v & 7u;
    const float r = (e == 0u) ? (float)m * 0.001953125f
                              : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (v & 0x80u) ? -r : r;
}

static inline uint f2e5m2(float f) {
    const uint b = as_type<uint>(f);
    const uint s = (b >> 24) & 0x80u;
    const float a = fabs(f);
    if (!(a > 0.0f)) return s;
    if (a >= 61440.0f) return s | 0x7Bu;
    int e = (int)((b >> 23) & 0xFFu) - 127;
    if (e < -14) {
        int q = (int)(a * 65536.0f + 0.5f);
        if (q > 4) q = 4;
        return s | (uint)q;
    }
    uint mm = (b & 0x7FFFFFu) >> 21;
    const uint rem = b & 0x1FFFFFu;
    if (rem > 0x100000u || (rem == 0x100000u && (mm & 1u))) {
        if (++mm == 4u) { mm = 0; ++e; }
    }
    if (e > 15) return s | 0x7Bu;
    return s | ((uint)(e + 15) << 2) | mm;
}

static inline float e5m22f(uint v) {
    const uint e = (v >> 2) & 0x1Fu, m = v & 3u;
    const float r = (e == 0u) ? (float)m * 1.52587890625e-05f
                              : as_type<float>(((e + 112u) << 23) | (m << 21));
    return (v & 0x80u) ? -r : r;
}

/* ─── Randomized Hadamard transform over one simdgroup ──────────────────── */

static inline float kv_sign(int d) {
    uint h = (uint)d * 2654435761u + 0x9E3779B9u;
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

static inline void simd_wht128(thread float v[4], uint lane) {
    float a;
    a = v[0] + v[1]; v[1] = v[0] - v[1]; v[0] = a;
    a = v[2] + v[3]; v[3] = v[2] - v[3]; v[2] = a;
    a = v[0] + v[2]; v[2] = v[0] - v[2]; v[0] = a;
    a = v[1] + v[3]; v[3] = v[1] - v[3]; v[1] = a;
    for (ushort m = 1; m < 32; m <<= 1) {
        const float o0 = simd_shuffle_xor(v[0], m);
        const float o1 = simd_shuffle_xor(v[1], m);
        const float o2 = simd_shuffle_xor(v[2], m);
        const float o3 = simd_shuffle_xor(v[3], m);
        if (lane & m) {
            v[0] = o0 - v[0]; v[1] = o1 - v[1]; v[2] = o2 - v[2]; v[3] = o3 - v[3];
        } else {
            v[0] += o0; v[1] += o1; v[2] += o2; v[3] += o3;
        }
    }
}

/* ─── Packed codes ──────────────────────────────────────────────────────── */

/* Bits a lane owns in a TurboQuant vector (4 values), 0 otherwise. */
template <int FMT> constexpr int kv_nbits() {
    return FMT == KV_TQ4 ? 16 : FMT == KV_TQ3 ? 12 : FMT == KV_TQ2 ? 8
         : FMT == KV_TQ1_5 ? 6 : 0;
}

/* Lane's NB bits out of a packed vector the simdgroup reads word by word. */
template <int NB>
static inline uint simd_bits_of(uint mine, uint lane) {
    const int bit = NB * (int)lane;
    const int wi = bit >> 5, sh = bit & 31;
    const uint lo = simd_shuffle(mine, (ushort)wi);
    const uint hi = simd_shuffle(mine, (ushort)((wi + 1) & 31));
    const uint v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
    return v & ((1u << NB) - 1u);
}

template <int FMT>
static inline void unpack4(uint bits, float sigma, thread float out[4]) {
    if (FMT == KV_TQ4) {
        for (int i = 0; i < 4; ++i) out[i] = sigma * kv_lev4[(bits >> (4 * i)) & 15u];
    } else if (FMT == KV_TQ3) {
        for (int i = 0; i < 4; ++i) out[i] = sigma * kv_lev3[(bits >> (3 * i)) & 7u];
    } else if (FMT == KV_TQ2) {
        for (int i = 0; i < 4; ++i) out[i] = sigma * kv_lev2[(bits >> (2 * i)) & 3u];
    } else {
        for (int pp = 0; pp < 2; ++pp) {
            const uint c = (bits >> (3 * pp)) & 7u;
            out[2 * pp + 0] = sigma * kv_vq15[2 * c + 0];
            out[2 * pp + 1] = sigma * kv_vq15[2 * c + 1];
        }
    }
}

template <int FMT>
static inline uint pack4(thread const float y[4], float inv_sigma) {
    uint bits = 0;
    if (FMT == KV_TQ4) {
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint c = 0;
            for (int j = 0; j < 15; ++j) c += (t > 0.5f * (kv_lev4[j] + kv_lev4[j + 1])) ? 1u : 0u;
            bits |= c << (4 * i);
        }
    } else if (FMT == KV_TQ3) {
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint c = 0;
            for (int j = 0; j < 7; ++j) c += (t > 0.5f * (kv_lev3[j] + kv_lev3[j + 1])) ? 1u : 0u;
            bits |= c << (3 * i);
        }
    } else if (FMT == KV_TQ2) {
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint c = 0;
            for (int j = 0; j < 3; ++j) c += (t > 0.5f * (kv_lev2[j] + kv_lev2[j + 1])) ? 1u : 0u;
            bits |= c << (2 * i);
        }
    } else {
        for (int pp = 0; pp < 2; ++pp) {
            const float a = y[2 * pp + 0] * inv_sigma;
            const float b = y[2 * pp + 1] * inv_sigma;
            uint best = 0;
            float bd = FLT_MAX;
            for (int c = 0; c < 8; ++c) {
                const float da = a - kv_vq15[2 * c + 0];
                const float db = b - kv_vq15[2 * c + 1];
                const float d = da * da + db * db;
                if (d < bd) { bd = d; best = (uint)c; }
            }
            bits |= best << (3 * pp);
        }
    }
    return bits;
}

/* A position's raw words: what one lane loads for its 4 values. */
template <int FMT> struct kv_raw { uint w; half sigma; };
template <> struct kv_raw<KV_FP16> { uint2 w; half sigma; };

template <int FMT>
static inline kv_raw<FMT> kv_fetch(device const uchar* store, device const half* meta,
                                   ulong vi, int bpv, uint lane) {
    kv_raw<FMT> r;
    if (FMT == KV_FP8_E4M3 || FMT == KV_FP8_E5M2) {
        r.w = ((device const uint*)(store + vi * (ulong)bpv))[lane];
        r.sigma = 0.0h;
    } else {
        const int nwords = kv_nbits<FMT>();          /* N * 32 bits / 32 */
        device const uint* w = (device const uint*)(store + vi * (ulong)bpv);
        r.w = ((int)lane < nwords) ? w[lane] : 0u;
        r.sigma = meta[vi];
    }
    return r;
}

template <>
inline kv_raw<KV_FP16> kv_fetch<KV_FP16>(device const uchar* store, device const half* meta,
                                        ulong vi, int bpv, uint lane) {
    kv_raw<KV_FP16> r;
    r.w = *(device const uint2*)((device const half*)store + vi * ATT_D + lane * 4);
    r.sigma = 0.0h;
    return r;
}

template <int FMT>
static inline void kv_decode(kv_raw<FMT> r, uint lane, thread float out[4]) {
    if (FMT == KV_FP8_E4M3 || FMT == KV_FP8_E5M2) {
        for (int i = 0; i < 4; ++i) {
            const uint b = (r.w >> (8 * i)) & 0xFFu;
            out[i] = (FMT == KV_FP8_E4M3) ? e4m32f(b) : e5m22f(b);
        }
    } else {
        const uint v = simd_bits_of<kv_nbits<FMT>()>(r.w, lane);
        unpack4<FMT>(v, (float)r.sigma, out);
    }
}

template <>
inline void kv_decode<KV_FP16>(kv_raw<KV_FP16> r, uint lane, thread float out[4]) {
    const half2 a = as_type<half2>(r.w.x), b = as_type<half2>(r.w.y);
    out[0] = (float)a.x; out[1] = (float)a.y; out[2] = (float)b.x; out[3] = (float)b.y;
}

/* ─── Store ─────────────────────────────────────────────────────────────── */

struct vv_kv_store_p {
    int n_kv_heads; int head_dim; int pos0; int use_dev_pos; int n_pos;
    int bpv; int use_pt; int has_meta; int has_ref; int chunks;
};

/* FP16: 16-byte chunks, each landing on its position's (paged) row. */
kernel void vv_kv_store_raw(constant vv_kv_store_p& p [[buffer(0)]],
                            device const uint4* k_src [[buffer(1)]],
                            device const uint4* v_src [[buffer(2)]],
                            device uint4* k_dst [[buffer(3)]],
                            device uint4* v_dst [[buffer(4)]],
                            device const int* d_pos [[buffer(5)]],
                            device const int* pt [[buffer(6)]],
                            uint i [[thread_position_in_grid]]) {
    const ulong per_pos = (ulong)p.n_kv_heads * (ulong)p.chunks;
    if ((ulong)i >= per_pos * (ulong)p.n_pos) return;
    const int pos0 = p.use_dev_pos ? *d_pos : p.pos0;
    const int pp = (int)((ulong)i / per_pos);
    const ulong r = (ulong)i - (ulong)pp * per_pos;
    const ulong dst = (ulong)kv_row(pt, p.use_pt, pos0 + pp) * per_pos + r;
    k_dst[dst] = k_src[i];
    v_dst[dst] = v_src[i];
}

/* One simdgroup per (position, kv head) vector; 4 simdgroups a group. */
template <int FMT>
kernel void vv_kv_store_q(constant vv_kv_store_p& p [[buffer(0)]],
                          device const half* src [[buffer(1)]],
                          device uchar* store [[buffer(2)]],
                          device half* meta [[buffer(3)]],
                          device const half* ref [[buffer(4)]],
                          device const int* d_pos [[buffer(5)]],
                          device const int* pt [[buffer(6)]],
                          VV_GRID_ARGS) {
    threadgroup uint codes[4][32];
    const int wg = (int)blockIdx.x * 4 + (int)warp;
    const int total = p.n_pos * p.n_kv_heads;
    if (wg >= total) return;                       /* whole simdgroup */
    const int pos0 = p.use_dev_pos ? *d_pos : p.pos0;
    const int pp = wg / p.n_kv_heads, h = wg % p.n_kv_heads;
    const int d0 = (int)lane * 4;
    device const half* s = src + ((ulong)pp * (ulong)p.n_kv_heads + (ulong)h) * (ulong)p.head_dim + d0;
    const int row = kv_row(pt, p.use_pt, pos0 + pp);
    device uchar* dst = store + ((ulong)row * (ulong)p.n_kv_heads + (ulong)h) * (ulong)p.bpv;

    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = (float)s[i];
    if (p.has_ref) {
        device const half* r = ref + (ulong)h * (ulong)p.head_dim + d0;
        for (int i = 0; i < 4; ++i) v[i] -= (float)r[i];
    }

    if (FMT == KV_FP8_E4M3 || FMT == KV_FP8_E5M2) {
        uint packed = 0;
        for (int i = 0; i < 4; ++i) {
            const uint b = (FMT == KV_FP8_E4M3) ? f2e4m3(v[i]) : f2e5m2(v[i]);
            packed |= b << (8 * i);
        }
        ((device uint*)dst)[lane] = packed;
        return;
    }

    for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
    simd_wht128(v, lane);
    for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;

    float ss = 0.0f;
    for (int i = 0; i < 4; ++i) ss = fma(v[i], v[i], ss);
    const float sigma = sqrt(max(vv_warp_sum(ss) / (float)p.head_dim, 1e-12f));
    const uint bits = pack4<FMT>(v, 1.0f / sigma);

    /* Words overlap lanes; each word gathers the codes that land in it. */
    const int nb = max(kv_nbits<FMT>(), 1);
    codes[warp][lane] = bits;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int nwords = p.bpv / 4;
    if ((int)lane < nwords) {
        const int w0 = (int)lane * 32;
        uint word = 0;
        for (int L = w0 / nb; L < 32 && L * nb < w0 + 32; ++L) {
            const int sh = L * nb - w0;
            const uint c = codes[warp][L];
            word |= sh >= 0 ? (c << sh) : (c >> (-sh));
        }
        ((device uint*)dst)[lane] = word;
    }
    if (lane == 0 && p.has_meta)
        meta[(ulong)row * (ulong)p.n_kv_heads + (ulong)h] = (half)sigma;
}

typedef decltype(vv_kv_store_q<KV_TQ4>) vv_kv_store_q_t;
template [[host_name("vv_kv_store_fp8_e4m3")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_FP8_E4M3>;
template [[host_name("vv_kv_store_fp8_e5m2")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_FP8_E5M2>;
template [[host_name("vv_kv_store_tq4")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_TQ4>;
template [[host_name("vv_kv_store_tq3")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_TQ3>;
template [[host_name("vv_kv_store_tq2")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_TQ2>;
template [[host_name("vv_kv_store_tq1_5")]] kernel vv_kv_store_q_t vv_kv_store_q<KV_TQ1_5>;

struct vv_kv_ref_p { int n_kv_heads; int head_dim; int n_pos; };

/* Mean key over the first chunk, one group per head (kv_build_ref_kernel). */
kernel void vv_kv_build_ref(constant vv_kv_ref_p& p [[buffer(0)]],
                            device const half* src [[buffer(1)]],
                            device half* ref [[buffer(2)]],
                            VV_GRID_ARGS) {
    const int h = (int)blockIdx.x, d = (int)threadIdx.x;
    if (d >= p.head_dim) return;
    float acc = 0.0f;
    for (int pp = 0; pp < p.n_pos; ++pp)
        acc += (float)src[((ulong)pp * (ulong)p.n_kv_heads + (ulong)h) * (ulong)p.head_dim + d];
    ref[(ulong)h * (ulong)p.head_dim + d] = (half)(acc / (float)p.n_pos);
}

struct vv_kv_map_p { int n; int id[256]; };

kernel void vv_kv_page_map(constant vv_kv_map_p& p [[buffer(0)]],
                           device int* table [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if ((int)i < p.n) table[i] = p.id[i];
}

struct vv_kv_rot_p { int n_heads; int head_dim; int rows; int inverse; };

kernel void vv_kv_rotate(constant vv_kv_rot_p& p [[buffer(0)]],
                         device half* x [[buffer(1)]],
                         VV_GRID_ARGS) {
    const int idx = (int)blockIdx.x * 4 + (int)warp;
    if (idx >= p.rows * p.n_heads) return;
    const int d0 = (int)lane * 4;
    device half* ptr = x + (ulong)idx * (ulong)p.head_dim + d0;
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = (float)ptr[i];
    if (!p.inverse) {
        for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
        simd_wht128(v, lane);
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;
    } else {
        simd_wht128(v, lane);
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }
    for (int i = 0; i < 4; ++i) ptr[i] = (half)v[i];
}

struct vv_kv_deq_p { int n_kv_heads; int head_dim; int n_pos; int bpv; int has_meta; int has_ref; };

template <int FMT>
kernel void vv_kv_dequant(constant vv_kv_deq_p& p [[buffer(0)]],
                          device const uchar* store [[buffer(1)]],
                          device const half* meta [[buffer(2)]],
                          device const half* ref [[buffer(3)]],
                          device half* out [[buffer(4)]],
                          VV_GRID_ARGS) {
    const int idx = (int)blockIdx.x * 4 + (int)warp;
    if (idx >= p.n_pos * p.n_kv_heads) return;
    const int d0 = (int)lane * 4;
    float v[4];
    const kv_raw<FMT> r = kv_fetch<FMT>(store, meta, (ulong)idx, p.bpv, lane);
    kv_decode<FMT>(r, lane, v);
    if (FMT >= KV_TQ4) {
        simd_wht128(v, lane);
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }
    if (p.has_ref) {
        device const half* rr = ref + (ulong)(idx % p.n_kv_heads) * (ulong)p.head_dim + d0;
        for (int i = 0; i < 4; ++i) v[i] += (float)rr[i];
    }
    device half* o = out + (ulong)idx * (ulong)p.head_dim + d0;
    for (int i = 0; i < 4; ++i) o[i] = (half)v[i];
}

typedef decltype(vv_kv_dequant<KV_TQ4>) vv_kv_dequant_t;
template [[host_name("vv_kv_dequant_fp8_e4m3")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_FP8_E4M3>;
template [[host_name("vv_kv_dequant_fp8_e5m2")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_FP8_E5M2>;
template [[host_name("vv_kv_dequant_tq4")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_TQ4>;
template [[host_name("vv_kv_dequant_tq3")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_TQ3>;
template [[host_name("vv_kv_dequant_tq2")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_TQ2>;
template [[host_name("vv_kv_dequant_tq1_5")]] kernel vv_kv_dequant_t vv_kv_dequant<KV_TQ1_5>;

/* ─── Decode: split KV, GQA packed or spread (attention_decode.cu) ──────── */

struct vv_attn_dec_p {
    int n_kv_heads; int G; int cache_len; int use_dev_len;
    int n_parts; int bpv; int use_pt; float scale;
};

/*
 * H == 1 (spread): grid (n_kv, n_parts), G simdgroups, one head each.
 * H == G (packed): grid (n_kv, n_parts / 8), 8 simdgroups, one slice each
 * for every head of the group. Same FP32 arithmetic either way.
 */
template <int FMT, int H>
kernel void vv_attn_decode(constant vv_attn_dec_p& p [[buffer(0)]],
                           device const half* q [[buffer(1)]],
                           device const uchar* k_store [[buffer(2)]],
                           device const uchar* v_store [[buffer(3)]],
                           device const half* k_meta [[buffer(4)]],
                           device const half* v_meta [[buffer(5)]],
                           device const int* pt [[buffer(6)]],
                           device float* part_o [[buffer(7)]],
                           device float* part_m [[buffer(8)]],
                           device float* part_l [[buffer(9)]],
                           device const int* d_len [[buffer(10)]],
                           VV_GRID_ARGS) {
    const int kv_head = (int)blockIdx.x;
    const int part = (H == 1) ? (int)blockIdx.y : (int)(blockIdx.y * 8 + warp);
    const int head0 = kv_head * p.G + ((H == 1) ? (int)warp : 0);
    if (part >= p.n_parts) return;
    const int cache_len = p.use_dev_len ? *d_len : p.cache_len;
    const int chunk = (cache_len + p.n_parts - 1) / p.n_parts;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > cache_len) end = cache_len;

    float qr[H][4], o[H][4], m[H], l[H];
    for (int h = 0; h < H; ++h) {
        device const half* qh = q + (ulong)(head0 + h) * ATT_D + lane * 4;
        for (int i = 0; i < 4; ++i) { qr[h][i] = (float)qh[i]; o[h][i] = 0.0f; }
        m[h] = -FLT_MAX; l[h] = 0.0f;
    }

    kv_raw<FMT> rk, rv;
    if (begin < end) {
        const ulong vi = (ulong)kv_row(pt, p.use_pt, begin) * (ulong)p.n_kv_heads + kv_head;
        rk = kv_fetch<FMT>(k_store, k_meta, vi, p.bpv, lane);
        rv = kv_fetch<FMT>(v_store, v_meta, vi, p.bpv, lane);
    }
    for (int t = begin; t < end; ++t) {
        const int tn = (t + 1 < end) ? t + 1 : t;
        const ulong vn = (ulong)kv_row(pt, p.use_pt, tn) * (ulong)p.n_kv_heads + kv_head;
        const kv_raw<FMT> nk = kv_fetch<FMT>(k_store, k_meta, vn, p.bpv, lane);
        const kv_raw<FMT> nv = kv_fetch<FMT>(v_store, v_meta, vn, p.bpv, lane);

        float kv[4];
        kv_decode<FMT>(rk, lane, kv);
        float pr[H], alpha[H];
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
            for (int i = 0; i < 4; ++i) dot = fma(qr[h][i], kv[i], dot);
            dot = vv_warp_sum(dot) * p.scale;
            const float m_new = max(m[h], dot);
            alpha[h] = (m[h] > -FLT_MAX) ? fast::exp(m[h] - m_new) : 0.0f;
            pr[h] = fast::exp(dot - m_new);
            m[h] = m_new;
        }
        kv_decode<FMT>(rv, lane, kv);
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < 4; ++i) o[h][i] = o[h][i] * alpha[h] + pr[h] * kv[i];
            l[h] = l[h] * alpha[h] + pr[h];
        }
        rk = nk; rv = nv;
    }

    for (int h = 0; h < H; ++h) {
        const ulong row = (ulong)(head0 + h) * (ulong)p.n_parts + (ulong)part;
        device float* po = part_o + row * ATT_D + lane * 4;
        for (int i = 0; i < 4; ++i) po[i] = o[h][i];
        if (lane == 0) {
            part_m[row] = (end > begin) ? m[h] : -FLT_MAX;
            part_l[row] = l[h];
        }
    }
}

typedef decltype(vv_attn_decode<KV_FP16, 1>) vv_attn_decode_t;
#define ATT_DEC_INST(F, FN, HH)                                                \
template [[host_name("vv_attn_decode_" FN "_h" #HH)]]                          \
kernel vv_attn_decode_t vv_attn_decode<F, HH>;
#define ATT_DEC_FMT(F, FN)                                                     \
    ATT_DEC_INST(F, FN, 1) ATT_DEC_INST(F, FN, 2) ATT_DEC_INST(F, FN, 4)       \
    ATT_DEC_INST(F, FN, 6) ATT_DEC_INST(F, FN, 7) ATT_DEC_INST(F, FN, 8)
ATT_DEC_FMT(KV_FP16, "fp16")
ATT_DEC_FMT(KV_FP8_E4M3, "fp8_e4m3")
ATT_DEC_FMT(KV_FP8_E5M2, "fp8_e5m2")
ATT_DEC_FMT(KV_TQ4, "tq4")
ATT_DEC_FMT(KV_TQ3, "tq3")
ATT_DEC_FMT(KV_TQ2, "tq2")
ATT_DEC_FMT(KV_TQ1_5, "tq1_5")

struct vv_attn_comb_p { int n_parts; };

/* Merge the slices: one group per output row, one thread per dim. */
kernel void vv_attn_combine(constant vv_attn_comb_p& p [[buffer(0)]],
                            device const float* part_o [[buffer(1)]],
                            device const float* part_m [[buffer(2)]],
                            device const float* part_l [[buffer(3)]],
                            device half* out [[buffer(4)]],
                            VV_GRID_ARGS) {
    const int row = (int)blockIdx.x, d = (int)threadIdx.x;
    device const float* pm = part_m + (ulong)row * (ulong)p.n_parts;
    device const float* pl = part_l + (ulong)row * (ulong)p.n_parts;
    float gmax = -FLT_MAX;
    for (int i = 0; i < p.n_parts; ++i) gmax = max(gmax, pm[i]);
    float num = 0.0f, den = 0.0f;
    if (gmax > -FLT_MAX) {
        for (int i = 0; i < p.n_parts; ++i) {
            const float mi = pm[i];
            if (mi <= -FLT_MAX) continue;
            const float w = fast::exp(mi - gmax);
            num += w * part_o[((ulong)row * (ulong)p.n_parts + (ulong)i) * ATT_D + d];
            den += w * pl[i];
        }
    }
    out[(ulong)row * ATT_D + d] = (half)(den > 1e-20f ? num / den : 0.0f);
}

/* ─── Prefill, scalar (fa1): one simdgroup per query row ────────────────── */

struct vv_attn_pre_p {
    int n_q_heads; int n_kv_heads; int G; int q_len; int q_offset; int kv_len;
    int causal; int bpv; int use_pt; float scale;
};

#define FA1_BR 8
#define FA1_BC 32
#define FA1_LD (ATT_D + 2)

template <int FMT>
kernel void vv_attn_prefill_fa1(constant vv_attn_pre_p& p [[buffer(0)]],
                                device const half* Q [[buffer(1)]],
                                device const uchar* k_store [[buffer(2)]],
                                device const uchar* v_store [[buffer(3)]],
                                device const half* k_meta [[buffer(4)]],
                                device const half* v_meta [[buffer(5)]],
                                device const int* pt [[buffer(6)]],
                                device half* O [[buffer(7)]],
                                VV_GRID_ARGS) {
    threadgroup half K_tile[FA1_BC * FA1_LD];
    threadgroup half V_tile[FA1_BC * FA1_LD];
    threadgroup half Q_s[FA1_BR * ATT_D];

    const int head = (int)blockIdx.x, tile_row = (int)blockIdx.y;
    const int kv_head = head / p.G;
    const int q_row = tile_row * FA1_BR + (int)warp;
    const bool alive = q_row < p.q_len;
    const int q_abs = p.q_offset + q_row;
    const int tile_last = tile_row * FA1_BR + FA1_BR - 1;
    int block_max_kv = p.causal ? (p.q_offset + tile_last + 1) : p.kv_len;
    if (block_max_kv > p.kv_len) block_max_kv = p.kv_len;
    const int q_stride = p.n_q_heads * ATT_D;

    {
        const int q_base = q_row * q_stride + head * ATT_D;
        for (int d = (int)lane; d < ATT_D; d += 32)
            Q_s[warp * ATT_D + d] = alive ? Q[q_base + d] : (half)0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o_reg[4] = { 0.f, 0.f, 0.f, 0.f };
    float m_i = -FLT_MAX, l_i = 0.0f;
    const int d0 = (int)lane * 4;
    const int n_tiles = (block_max_kv + FA1_BC - 1) / FA1_BC;

    for (int tj = 0; tj < n_tiles; ++tj) {
        const int kv_start = tj * FA1_BC;
        for (int r = (int)warp; r < FA1_BC; r += FA1_BR) {
            const int gp = kv_start + r;
            float kk[4] = { 0.f, 0.f, 0.f, 0.f }, vv[4] = { 0.f, 0.f, 0.f, 0.f };
            if (gp < p.kv_len) {                          /* simdgroup-uniform */
                const ulong vi = (ulong)kv_row(pt, p.use_pt, gp) * (ulong)p.n_kv_heads + kv_head;
                kv_decode<FMT>(kv_fetch<FMT>(k_store, k_meta, vi, p.bpv, lane), lane, kk);
                kv_decode<FMT>(kv_fetch<FMT>(v_store, v_meta, vi, p.bpv, lane), lane, vv);
            }
            for (int i = 0; i < 4; ++i) {
                K_tile[r * FA1_LD + d0 + i] = (half)kk[i];
                V_tile[r * FA1_LD + d0 + i] = (half)vv[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kv_pos = kv_start + (int)lane;
        float score = -FLT_MAX;
        if (alive && kv_pos < p.kv_len && (!p.causal || kv_pos <= q_abs)) {
            float dot = 0.0f;
            for (int d = 0; d < ATT_D; d += 2) {
                dot = fma((float)Q_s[warp * ATT_D + d], (float)K_tile[lane * FA1_LD + d], dot);
                dot = fma((float)Q_s[warp * ATT_D + d + 1], (float)K_tile[lane * FA1_LD + d + 1], dot);
            }
            score = dot * p.scale;
        }
        const float tile_max = vv_warp_max(score);
        const float m_new = max(m_i, tile_max);
        const bool have_any = m_new > -FLT_MAX;
        const float alpha = (have_any && m_i > -FLT_MAX) ? fast::exp(m_i - m_new) : 1.0f;
        for (int i = 0; i < 4; ++i) o_reg[i] *= alpha;
        l_i *= alpha;
        const float pk_own = (score > -FLT_MAX && have_any) ? fast::exp(score - m_new) : 0.0f;
        l_i += vv_warp_sum(pk_own);
        const int tile_valid = min(FA1_BC, p.kv_len - kv_start);
        for (int k = 0; k < tile_valid; ++k) {
            const float pk = simd_shuffle(pk_own, (ushort)k);
            if (pk != 0.0f)
                for (int i = 0; i < 4; ++i)
                    o_reg[i] += pk * (float)V_tile[k * FA1_LD + d0 + i];
        }
        if (have_any) m_i = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (alive) {
        const float inv_l = (l_i > 1e-20f) ? (1.0f / l_i) : 0.0f;
        const int out_offset = q_row * q_stride + head * ATT_D;
        for (int i = 0; i < 4; ++i) O[out_offset + d0 + i] = (half)(o_reg[i] * inv_l);
    }
}

typedef decltype(vv_attn_prefill_fa1<KV_FP16>) vv_attn_prefill_fa1_t;
#define ATT_FA1_INST(F, FN)                                                    \
template [[host_name("vv_attn_prefill_fa1_" FN)]]                              \
kernel vv_attn_prefill_fa1_t vv_attn_prefill_fa1<F>;
ATT_FA1_INST(KV_FP16, "fp16")
ATT_FA1_INST(KV_FP8_E4M3, "fp8_e4m3")
ATT_FA1_INST(KV_FP8_E5M2, "fp8_e5m2")
ATT_FA1_INST(KV_TQ4, "tq4")
ATT_FA1_INST(KV_TQ3, "tq3")
ATT_FA1_INST(KV_TQ2, "tq2")
ATT_FA1_INST(KV_TQ1_5, "tq1_5")

/* ─── Prefill on simdgroup matrices, GQA-packed ─────────────────────────── */

/*
 * A threadgroup takes PA_ROWS consecutive packed rows of one KV group --
 * row R is (position R / G, head R % G) -- as eight simdgroups of 8 rows.
 * Each PA_BC-position tile of K and V is decoded to FP16 in threadgroup
 * memory once for all of them, so a wider group re-reads a long cache less
 * often.
 *
 * What makes the width affordable is where Q lives. With Q in registers,
 * each simdgroup holds 16 FP32 accumulators and 16 Q fragments, and eight
 * such simdgroups leave too few registers to keep the cores busy: that
 * arrangement measured 0.2 TFLOP/s at 1024 positions against 0.46 for four
 * simdgroups of the same shape. Staging Q in threadgroup memory instead
 * (below) frees the registers the width needs.
 *
 * S = Q K^T and the online softmax run on the fragments' own elements:
 * lane l holds row fm = (l/4 & 4) + (l/2 % 4), columns fn, fn+1 with
 * fn = (l/4 & 2) * 2 + (l % 2) * 2, so a row's max and sum reduce over lanes
 * l, l^1, l^8, l^9. P goes into P V as an FP16 pair (value and remainder),
 * which keeps about 22 bits of each probability.
 */
#define PA_ROWS 64
#define PA_BC   16
#define PA_LD   (ATT_D + 8)

/*
 * SPLIT keeps about 22 bits of each probability by feeding P V an FP16 pair
 * (the value and what rounding dropped), which is what `fa2` means here;
 * `flashinfer` rounds P once, as its CUDA counterpart does, and halves the
 * P V matrix multiplies.
 */
template <int FMT, bool SPLIT>
kernel void vv_attn_prefill_mma(constant vv_attn_pre_p& p [[buffer(0)]],
                                device const half* Q [[buffer(1)]],
                                device const uchar* k_store [[buffer(2)]],
                                device const uchar* v_store [[buffer(3)]],
                                device const half* k_meta [[buffer(4)]],
                                device const half* v_meta [[buffer(5)]],
                                device const int* pt [[buffer(6)]],
                                device half* O [[buffer(7)]],
                                VV_GRID_ARGS) {
    threadgroup half Ks[PA_BC * PA_LD];
    threadgroup half Vs[PA_BC * PA_LD];
    threadgroup half Qs[PA_ROWS * PA_LD];
    const uint tid = threadIdx.x;
    const int kv_head = (int)blockIdx.y;
    const int G = p.G;
    const int R0 = (int)blockIdx.x * PA_ROWS;
    const int total_rows = p.q_len * G;

    /* This lane's elements of an 8x8 fragment: row fm, columns fn, fn+1. */
    const int qid = (int)lane / 4;
    const int fm = (qid & 4) + (((int)lane / 2) % 4);
    const int fn = (qid & 2) * 2 + ((int)lane % 2) * 2;
    const int Rm = R0 + (int)warp * 8 + fm;
    const bool alive = Rm < total_rows;
    const int pos_m = alive ? Rm / G : 0;
    const int head_m = kv_head * G + (alive ? Rm % G : 0);
    const int qa = p.q_offset + pos_m;

    /*
     * Q stays in threadgroup memory and its fragments are loaded per tile.
     * Keeping all 16 in registers next to the 16 FP32 accumulators leaves
     * too few registers for a second threadgroup per core, and this kernel
     * is bound by how much of the KV re-read it can hide, not by the loads.
     * The rows are (position, head) pairs, so each is staged by hand.
     */
    for (uint e = tid; e < (uint)PA_ROWS * 16u; e += 256u) {
        const int r = (int)(e >> 4), c = (int)(e & 15u) * 8;
        const int R = R0 + r;
        threadgroup half* d = Qs + r * PA_LD + c;
        if (R < total_rows) {
            const int pos = R / G, h = kv_head * G + R % G;
            device const half4* s4 = (device const half4*)
                (Q + ((ulong)pos * (ulong)p.n_q_heads + (ulong)h) * ATT_D + c);
            *(threadgroup half4*)d = s4[0];
            *(threadgroup half4*)(d + 4) = s4[1];
        } else {
            *(threadgroup half4*)d = half4(0.0h);
            *(threadgroup half4*)(d + 4) = half4(0.0h);
        }
    }

    const int last = min(R0 + PA_ROWS, total_rows) - 1;
    int kv_end = p.causal ? p.q_offset + last / G + 1 : p.kv_len;
    if (kv_end > p.kv_len) kv_end = p.kv_len;
    const int n_tiles = (kv_end + PA_BC - 1) / PA_BC;

    simdgroup_float8x8 o[ATT_D / 8];
    for (int d = 0; d < ATT_D / 8; ++d) o[d] = simdgroup_float8x8(0.0f);
    float m_i = -FLT_MAX, l_i = 0.0f;

    for (int t = 0; t < n_tiles; ++t) {
        const int kv0 = t * PA_BC;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (FMT == KV_FP16) {
            for (uint e = tid; e < (uint)PA_BC * 16u; e += 256u) {
                const int r = (int)(e >> 4), c = (int)(e & 15u) * 8;
                const int gp = kv0 + r;
                threadgroup half* dk = Ks + r * PA_LD + c;
                threadgroup half* dv = Vs + r * PA_LD + c;
                if (gp < p.kv_len) {
                    const ulong vi = (ulong)kv_row(pt, p.use_pt, gp) * (ulong)p.n_kv_heads + kv_head;
                    device const half4* sk = (device const half4*)((device const half*)k_store + vi * ATT_D + c);
                    device const half4* sv = (device const half4*)((device const half*)v_store + vi * ATT_D + c);
                    *(threadgroup half4*)dk = sk[0]; *(threadgroup half4*)(dk + 4) = sk[1];
                    *(threadgroup half4*)dv = sv[0]; *(threadgroup half4*)(dv + 4) = sv[1];
                } else {
                    *(threadgroup half4*)dk = half4(0.0h); *(threadgroup half4*)(dk + 4) = half4(0.0h);
                    *(threadgroup half4*)dv = half4(0.0h); *(threadgroup half4*)(dv + 4) = half4(0.0h);
                }
            }
        } else {
            for (int r = (int)warp; r < PA_BC; r += 8) {
                const int gp = kv0 + r;
                float kk[4] = { 0.f, 0.f, 0.f, 0.f }, vv[4] = { 0.f, 0.f, 0.f, 0.f };
                if (gp < p.kv_len) {                      /* simdgroup-uniform */
                    const ulong vi = (ulong)kv_row(pt, p.use_pt, gp) * (ulong)p.n_kv_heads + kv_head;
                    kv_decode<FMT>(kv_fetch<FMT>(k_store, k_meta, vi, p.bpv, lane), lane, kk);
                    kv_decode<FMT>(kv_fetch<FMT>(v_store, v_meta, vi, p.bpv, lane), lane, vv);
                }
                *(threadgroup half4*)(Ks + r * PA_LD + lane * 4) = half4((half)kk[0], (half)kk[1], (half)kk[2], (half)kk[3]);
                *(threadgroup half4*)(Vs + r * PA_LD + lane * 4) = half4((half)vv[0], (half)vv[1], (half)vv[2], (half)vv[3]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* S = Q K^T, 8 x 32 per simdgroup */
        simdgroup_float8x8 s[PA_BC / 8];
        for (int j = 0; j < PA_BC / 8; ++j) s[j] = simdgroup_float8x8(0.0f);
        for (int kk = 0; kk < ATT_D / 8; ++kk) {
            simdgroup_half8x8 qf;
            simdgroup_load(qf, Qs + ((int)warp * 8) * PA_LD + kk * 8, PA_LD);
            for (int j = 0; j < PA_BC / 8; ++j) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, Ks + (j * 8) * PA_LD + kk * 8, PA_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qf, kf, s[j]);
            }
        }

        /* online softmax on this lane's 8 elements of row fm */
        float sv[PA_BC / 4];
        float mx = -FLT_MAX;
        for (int j = 0; j < PA_BC / 8; ++j) {
            thread auto& e = s[j].thread_elements();
            for (int i = 0; i < 2; ++i) {
                const int kp = kv0 + j * 8 + fn + i;
                const bool ok = alive && kp < p.kv_len && (!p.causal || kp <= qa);
                const float v = ok ? e[i] * p.scale : -FLT_MAX;
                sv[2 * j + i] = v;
                mx = max(mx, v);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m_new = max(m_i, mx);
        const float alpha = (m_new > -FLT_MAX && m_i > -FLT_MAX) ? fast::exp(m_i - m_new) : 1.0f;
        float sum = 0.0f;
        simdgroup_half8x8 ph[PA_BC / 8], pl[PA_BC / 8];
        for (int j = 0; j < PA_BC / 8; ++j) {
            thread auto& eh = ph[j].thread_elements();
            thread auto& el = pl[j].thread_elements();
            for (int i = 0; i < 2; ++i) {
                const float v = sv[2 * j + i];
                const float pr = (v > -FLT_MAX && m_new > -FLT_MAX) ? fast::exp(v - m_new) : 0.0f;
                sum += pr;
                const half hi = (half)pr;
                eh[i] = hi;
                if (SPLIT) el[i] = (half)(pr - (float)hi);
            }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l_i = l_i * alpha + sum;
        m_i = m_new;
        for (int d = 0; d < ATT_D / 8; ++d) {
            thread auto& e = o[d].thread_elements();
            e[0] *= alpha; e[1] *= alpha;
        }

        /* O += P V */
        for (int j = 0; j < PA_BC / 8; ++j) {
            for (int d = 0; d < ATT_D / 8; ++d) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, Vs + (j * 8) * PA_LD + d * 8, PA_LD);
                simdgroup_multiply_accumulate(o[d], ph[j], vf, o[d]);
                if (SPLIT) simdgroup_multiply_accumulate(o[d], pl[j], vf, o[d]);
            }
        }
    }

    if (alive) {
        const float inv_l = (l_i > 1e-20f) ? 1.0f / l_i : 0.0f;
        device half* orow = O + ((ulong)pos_m * (ulong)p.n_q_heads + (ulong)head_m) * ATT_D;
        for (int d = 0; d < ATT_D / 8; ++d) {
            thread auto& e = o[d].thread_elements();
            *(device half2*)(orow + d * 8 + fn) = half2((half)(e[0] * inv_l), (half)(e[1] * inv_l));
        }
    }
}

typedef decltype(vv_attn_prefill_mma<KV_FP16, true>) vv_attn_prefill_mma_t;
#define ATT_MMA_INST(F, FN)                                                    \
template [[host_name("vv_attn_prefill_mma_" FN)]]                              \
kernel vv_attn_prefill_mma_t vv_attn_prefill_mma<F, true>;                     \
template [[host_name("vv_attn_prefill_fi_" FN)]]                               \
kernel vv_attn_prefill_mma_t vv_attn_prefill_mma<F, false>;
ATT_MMA_INST(KV_FP16, "fp16")
ATT_MMA_INST(KV_FP8_E4M3, "fp8_e4m3")
ATT_MMA_INST(KV_FP8_E5M2, "fp8_e5m2")
ATT_MMA_INST(KV_TQ4, "tq4")
ATT_MMA_INST(KV_TQ3, "tq3")
ATT_MMA_INST(KV_TQ2, "tq2")
ATT_MMA_INST(KV_TQ1_5, "tq1_5")
