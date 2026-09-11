/**
 * @file kv_quant.cu
 * @brief Quantized KV-cache storage and the attention kernels that read it.
 *
 * See include/vibevoice/kv_quant.h for the format definitions and for why the
 * Hadamard rotation lets attention work directly on stored values.
 *
 * Layout, per layer:
 *   store: [max_seq_len][n_kv_heads][bytes_per_vec]  bytes
 *   meta:  [max_seq_len][n_kv_heads]                 FP16 RMS (TurboQuant only)
 *
 * Sub-byte codes are laid out so that lane L of a warp owns dims 4L..4L+3 —
 * the same split the FP16 decode kernel uses — which means its codes are a
 * contiguous run of 4*bits bits at offset 4*bits*L. The warp loads the whole
 * vector as uint32 words (one per lane, fully coalesced) and each lane then
 * shuffles the one or two words its bits land in.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

/* ─── Quantizer codebooks ────────────────────────────────────────────────── */
/*
 * Lloyd-Max levels for a unit Gaussian, computed offline (tools/lloyd_max.py).
 * SQNR: 4-bit 20.2 dB, 3-bit 14.6 dB, 2-bit 9.3 dB, 1.5-bit 7.0 dB.
 */
__device__ __constant__ float c_lev4[16] = {
    -2.725112f, -2.065280f, -1.613787f, -1.251845f,
    -0.938657f, -0.654379f, -0.386970f, -0.127960f,
    +0.127960f, +0.386970f, +0.654379f, +0.938657f,
    +1.251845f, +1.613787f, +2.065280f, +2.725112f
};
__device__ __constant__ float c_lev3[8] = {
    -2.150300f, -1.343258f, -0.755588f, -0.244756f,
    +0.244756f, +0.755588f, +1.343258f, +2.150300f
};
__device__ __constant__ float c_lev2[4] = {
    -1.511418f, -0.452456f, +0.452456f, +1.511418f
};
/* 8-entry 2-D codebook; a pair of coordinates shares one 3-bit index. */
__device__ __constant__ float c_vq15[16] = {
    -1.500251f, +0.674678f,  -1.127883f, -0.787015f,
    -0.283251f, +0.390133f,  -0.103102f, +1.624620f,
    +0.100120f, -1.618563f,  +0.286482f, -0.379555f,
    +1.134227f, +0.790368f,  +1.496572f, -0.666578f
};

#define WHT_NORM 0.0883883476f      /* 1 / sqrt(128) */

/* ─── FP8, emulated ──────────────────────────────────────────────────────── */

__device__ __forceinline__ uint32_t f2e4m3(float f) {
    const uint32_t b = __float_as_uint(f);
    const uint32_t s = (b >> 24) & 0x80u;
    const float a = fabsf(f);
    if (!(a > 0.0f)) return s;                    /* zero, or NaN -> zero */
    if (a >= 464.0f) return s | 0x7Eu;            /* saturate at max normal */

    int e = (int)((b >> 23) & 0xFFu) - 127;
    if (e < -6) {                                  /* subnormal: m * 2^-9 */
        int q = (int)(a * 512.0f + 0.5f);
        if (q > 8) q = 8;                          /* 8 rolls into min normal */
        return s | (uint32_t)q;
    }
    uint32_t m   = b & 0x7FFFFFu;
    uint32_t mm  = m >> 20;
    uint32_t rem = m & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (mm & 1u))) {
        if (++mm == 8u) { mm = 0; ++e; }
    }
    if (e > 8) return s | 0x7Eu;
    return s | ((uint32_t)(e + 7) << 3) | mm;
}

__device__ __forceinline__ float e4m32f(uint32_t v) {
    const uint32_t e = (v >> 3) & 0xFu;
    const uint32_t m = v & 7u;
    const float r = (e == 0u) ? (float)m * 0.001953125f
                              : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (v & 0x80u) ? -r : r;
}

__device__ __forceinline__ uint32_t f2e5m2(float f) {
    const uint32_t b = __float_as_uint(f);
    const uint32_t s = (b >> 24) & 0x80u;
    const float a = fabsf(f);
    if (!(a > 0.0f)) return s;
    if (a >= 61440.0f) return s | 0x7Bu;           /* max normal = 57344 */

    int e = (int)((b >> 23) & 0xFFu) - 127;
    if (e < -14) {                                 /* subnormal: m * 2^-16 */
        int q = (int)(a * 65536.0f + 0.5f);
        if (q > 4) q = 4;
        return s | (uint32_t)q;
    }
    uint32_t m   = b & 0x7FFFFFu;
    uint32_t mm  = m >> 21;
    uint32_t rem = m & 0x1FFFFFu;
    if (rem > 0x100000u || (rem == 0x100000u && (mm & 1u))) {
        if (++mm == 4u) { mm = 0; ++e; }
    }
    if (e > 15) return s | 0x7Bu;
    return s | ((uint32_t)(e + 15) << 2) | mm;
}

__device__ __forceinline__ float e5m22f(uint32_t v) {
    const uint32_t e = (v >> 2) & 0x1Fu;
    const uint32_t m = v & 3u;
    const float r = (e == 0u) ? (float)m * 1.52587890625e-05f
                              : __uint_as_float(((e + 112u) << 23) | (m << 21));
    return (v & 0x80u) ? -r : r;
}

/* ─── Randomized Hadamard transform over one warp (head_dim = 128) ───────── */

/** @brief Fixed +-1 sign pattern; derived from the index, so nothing is stored. */
__device__ __forceinline__ float kv_sign(int d) {
    uint32_t h = (uint32_t)d * 2654435761u + 0x9E3779B9u;
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

/**
 * @brief In-place Walsh-Hadamard transform of 128 values held as 4 per lane.
 *
 * Strides 1 and 2 pair elements inside a lane; strides 4..64 pair lane L with
 * L^1 .. L^16, which is exactly one __shfl_xor per stage.
 */
__device__ __forceinline__ void warp_wht128(float v[4], int lane) {
    float a;
    a = v[0] + v[1]; v[1] = v[0] - v[1]; v[0] = a;
    a = v[2] + v[3]; v[3] = v[2] - v[3]; v[2] = a;
    a = v[0] + v[2]; v[2] = v[0] - v[2]; v[0] = a;
    a = v[1] + v[3]; v[3] = v[1] - v[3]; v[1] = a;

    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float o0 = __shfl_xor_sync(0xFFFFFFFFu, v[0], m);
        const float o1 = __shfl_xor_sync(0xFFFFFFFFu, v[1], m);
        const float o2 = __shfl_xor_sync(0xFFFFFFFFu, v[2], m);
        const float o3 = __shfl_xor_sync(0xFFFFFFFFu, v[3], m);
        if (lane & m) {
            v[0] = o0 - v[0]; v[1] = o1 - v[1];
            v[2] = o2 - v[2]; v[3] = o3 - v[3];
        } else {
            v[0] += o0; v[1] += o1; v[2] += o2; v[3] += o3;
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

/* ─── Bit-level access to a packed vector ────────────────────────────────── */

/** @brief Bits a lane owns for a format (4 values x bits-per-value). */
template <int FMT> struct KVBits { enum { N = 0 }; };
template <> struct KVBits<VV_KV_TQ4>   { enum { N = 16 }; };
template <> struct KVBits<VV_KV_TQ3>   { enum { N = 12 }; };
template <> struct KVBits<VV_KV_TQ2>   { enum { N =  8 }; };
template <> struct KVBits<VV_KV_TQ1_5> { enum { N =  6 }; };

/**
 * @brief Cooperatively read lane `lane`'s slice of a packed 128-value vector.
 *
 * One coalesced uint32 load per lane covers the whole vector; each lane then
 * pulls the word(s) holding its own bits out of the warp with a shuffle.
 */
template <int NBITS>
__device__ __forceinline__ uint32_t warp_fetch_bits(
    const uint8_t* __restrict__ base, int lane)
{
    const int nwords = (NBITS * 32 + 31) / 32;
    const uint32_t* w = (const uint32_t*)base;
    const uint32_t mine = (lane < nwords) ? w[lane] : 0u;

    const int bit = NBITS * lane;
    const int wi  = bit >> 5;
    const int sh  = bit & 31;
    const uint32_t lo = __shfl_sync(0xFFFFFFFFu, mine, wi);
    const uint32_t hi = __shfl_sync(0xFFFFFFFFu, mine, (wi + 1) & 31);
    const uint32_t v  = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
    return v & ((1u << NBITS) - 1u);
}

/** @brief Decode a lane's 4 values from its packed bits. */
template <int FMT>
__device__ __forceinline__ void unpack4(uint32_t bits, float sigma, float out[4])
{
    if (FMT == VV_KV_TQ4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = sigma * c_lev4[(bits >> (4 * i)) & 15u];
    } else if (FMT == VV_KV_TQ3) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = sigma * c_lev3[(bits >> (3 * i)) & 7u];
    } else if (FMT == VV_KV_TQ2) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = sigma * c_lev2[(bits >> (2 * i)) & 3u];
    } else {  /* VV_KV_TQ1_5: two pairs, 3 bits each */
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t c = (bits >> (3 * p)) & 7u;
            out[2 * p + 0] = sigma * c_vq15[2 * c + 0];
            out[2 * p + 1] = sigma * c_vq15[2 * c + 1];
        }
    }
}

/** @brief Nearest-level search; returns the packed code for 4 values. */
template <int FMT>
__device__ __forceinline__ uint32_t pack4(const float y[4], float inv_sigma)
{
    uint32_t bits = 0;
    if (FMT == VV_KV_TQ4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint32_t c = 0;
            #pragma unroll
            for (int j = 0; j < 15; ++j)
                c += (t > 0.5f * (c_lev4[j] + c_lev4[j + 1])) ? 1u : 0u;
            bits |= c << (4 * i);
        }
    } else if (FMT == VV_KV_TQ3) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint32_t c = 0;
            #pragma unroll
            for (int j = 0; j < 7; ++j)
                c += (t > 0.5f * (c_lev3[j] + c_lev3[j + 1])) ? 1u : 0u;
            bits |= c << (3 * i);
        }
    } else if (FMT == VV_KV_TQ2) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float t = y[i] * inv_sigma;
            uint32_t c = 0;
            #pragma unroll
            for (int j = 0; j < 3; ++j)
                c += (t > 0.5f * (c_lev2[j] + c_lev2[j + 1])) ? 1u : 0u;
            bits |= c << (2 * i);
        }
    } else {  /* VV_KV_TQ1_5 */
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float a = y[2 * p + 0] * inv_sigma;
            const float b = y[2 * p + 1] * inv_sigma;
            uint32_t best = 0;
            float bd = FLT_MAX;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float da = a - c_vq15[2 * c + 0];
                const float db = b - c_vq15[2 * c + 1];
                const float d = da * da + db * db;
                if (d < bd) { bd = d; best = (uint32_t)c; }
            }
            bits |= best << (3 * p);
        }
    }
    return bits;
}

/* ─── Store kernel ───────────────────────────────────────────────────────── */

/**
 * One warp handles one (position, kv_head) vector of both K and V.
 * Grid is flattened over positions x heads x {K, V}.
 */
template <int FMT>
__global__ void kv_store_kernel(
    const half* __restrict__ src,      /* [n_pos, n_kv_heads, 128]  */
    uint8_t* __restrict__ store,       /* [max_pos, n_kv_heads, BPV] */
    half* __restrict__ meta,           /* [max_pos, n_kv_heads] or NULL */
    const half* __restrict__ ref,      /* [n_kv_heads, 128] or NULL  */
    int n_kv_heads, int head_dim, int pos0, int n_pos, int bpv)
{
    const int warp_global = blockIdx.x * (blockDim.y) + threadIdx.y;
    const int total = n_pos * n_kv_heads;
    if (warp_global >= total) return;

    const int p    = warp_global / n_kv_heads;
    const int h    = warp_global % n_kv_heads;
    const int lane = threadIdx.x;
    const int d0   = lane * 4;

    const half* s = src + ((size_t)p * n_kv_heads + h) * head_dim + d0;
    uint8_t* dst = store
        + ((size_t)(pos0 + p) * n_kv_heads + h) * (size_t)bpv;

    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __half2float(s[i]);

    if (ref) {
        const half* r = ref + (size_t)h * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] -= __half2float(r[i]);
    }

    if (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2) {
        uint32_t packed = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (FMT == VV_KV_FP8_E4M3) ? f2e4m3(v[i]) : f2e5m2(v[i]);
            packed |= b << (8 * i);
        }
        ((uint32_t*)dst)[lane] = packed;
        return;
    }

    /* TurboQuant: rotate, then quantize against the vector's own RMS. */
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
    warp_wht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;

    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) ss = fmaf(v[i], v[i], ss);
    const float sigma = sqrtf(fmaxf(warp_sum(ss) / (float)head_dim, 1e-12f));

    const uint32_t bits = pack4<FMT>(v, 1.0f / sigma);

    /* Lanes write overlapping words, so assemble through shared memory. */
    extern __shared__ uint32_t s_pack[];
    uint32_t* buf = s_pack + threadIdx.y * 16;
    if (lane < 16) buf[lane] = 0u;
    __syncwarp();

    const int nb = KVBits<FMT>::N;
    const int bit = nb * lane;
    atomicOr(&buf[bit >> 5], bits << (bit & 31));
    if ((bit & 31) + nb > 32)
        atomicOr(&buf[(bit >> 5) + 1], bits >> (32 - (bit & 31)));
    __syncwarp();

    const int nwords = bpv / 4;
    if (lane < nwords) ((uint32_t*)dst)[lane] = buf[lane];
    if (lane == 0 && meta)
        meta[(size_t)(pos0 + p) * n_kv_heads + h] = __float2half(sigma);
}

/**
 * @brief Mean key over a run of positions, one block per head.
 *
 * Called once per layer on the first prefill chunk. Any fixed vector would be
 * correct — softmax cancels it — so a mean over the first chunk is plenty.
 */
__global__ void kv_build_ref_kernel(
    const half* __restrict__ src, half* __restrict__ ref,
    int n_kv_heads, int head_dim, int n_pos)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    float acc = 0.0f;
    for (int p = 0; p < n_pos; ++p)
        acc += __half2float(src[((size_t)p * n_kv_heads + h) * head_dim + d]);
    ref[(size_t)h * head_dim + d] = __float2half(acc / (float)n_pos);
}

/* ─── Rotation kernels (applied to Q before, O after, attention) ─────────── */

__global__ void kv_rotate_kernel(half* __restrict__ x, int n_heads,
                                 int head_dim, int rows, bool inverse)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= rows * n_heads) return;
    const int lane = threadIdx.x;
    const int d0 = lane * 4;

    half* p = x + (size_t)idx * head_dim + d0;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __half2float(p[i]);

    if (!inverse) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= kv_sign(d0 + i);
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM;
    } else {
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __float2half(v[i]);
}

/* ─── Flash decode over a quantized cache ────────────────────────────────── */

#define QD_WARPS   8
#define QD_POS_PER_WARP 128
#define QD_MAX_PARTS 256

/** @brief Read lane's 4 dims of one stored vector. */
template <int FMT>
__device__ __forceinline__ void kv_read4(
    const uint8_t* __restrict__ vec, const half* __restrict__ meta_ptr,
    int lane, float out[4])
{
    if (FMT == VV_KV_FP8_E4M3 || FMT == VV_KV_FP8_E5M2) {
        const uint32_t w = ((const uint32_t*)vec)[lane];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (w >> (8 * i)) & 0xFFu;
            out[i] = (FMT == VV_KV_FP8_E4M3) ? e4m32f(b) : e5m22f(b);
        }
    } else {
        const uint32_t bits = warp_fetch_bits<KVBits<FMT>::N>(vec, lane);
        unpack4<FMT>(bits, __half2float(*meta_ptr), out);
    }
}

template <int FMT>
__global__ void q_decode_split_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ k_store,
    const uint8_t* __restrict__ v_store,
    const half* __restrict__ k_meta,
    const half* __restrict__ v_meta,
    float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, int n_parts, int bpv, float scale)
{
    const int q_head  = blockIdx.x;
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int part    = blockIdx.y * QD_WARPS + warp_id;
    if (part >= n_parts) return;

    const int kv_head = q_head / (n_q_heads / n_kv_heads);
    const int d0 = lane * 4;

    const int chunk = (cache_len + n_parts - 1) / n_parts;
    const int begin = part * chunk;
    int end = begin + chunk;
    if (end > cache_len) end = cache_len;

    float o_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.0f;

    float qreg[4];
    {
        const half* qh = q + (size_t)q_head * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) qreg[i] = __half2float(qh[i]);
    }

    for (int t = begin; t < end; ++t) {
        const size_t vo = ((size_t)t * n_kv_heads + kv_head) * (size_t)bpv;
        const size_t mo = (size_t)t * n_kv_heads + kv_head;

        float kv[4];
        kv_read4<FMT>(k_store + vo, k_meta ? k_meta + mo : NULL, lane, kv);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dot = fmaf(qreg[i], kv[i], dot);
        dot = warp_sum(dot) * scale;

        const float m_new = fmaxf(m_i, dot);
        const float alpha = (m_i > -FLT_MAX) ? __expf(m_i - m_new) : 0.0f;
        const float p = __expf(dot - m_new);

        kv_read4<FMT>(v_store + vo, v_meta ? v_meta + mo : NULL, lane, kv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[i] = o_acc[i] * alpha + p * kv[i];
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    float* po = part_o + ((size_t)q_head * n_parts + part) * head_dim + d0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) po[i] = o_acc[i];
    if (lane == 0) {
        part_m[q_head * n_parts + part] = (end > begin) ? m_i : -FLT_MAX;
        part_l[q_head * n_parts + part] = l_i;
    }
}

__global__ void q_decode_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, half* __restrict__ output,
    int n_parts, int head_dim)
{
    const int q_head = blockIdx.x;
    const int d = threadIdx.x;

    extern __shared__ float sh[];
    float* s_m = sh;
    float* s_l = sh + n_parts;
    for (int i = d; i < n_parts; i += blockDim.x) {
        s_m[i] = part_m[q_head * n_parts + i];
        s_l[i] = part_l[q_head * n_parts + i];
    }
    __syncthreads();
    if (d >= head_dim) return;

    float gmax = -FLT_MAX;
    for (int i = 0; i < n_parts; ++i) gmax = fmaxf(gmax, s_m[i]);

    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n_parts; ++i) {
        if (s_m[i] <= -FLT_MAX) continue;
        const float w = __expf(s_m[i] - gmax);
        num += w * part_o[((size_t)q_head * n_parts + i) * head_dim + d];
        den += w * s_l[i];
    }
    output[q_head * head_dim + d] = __float2half(den > 1e-20f ? num / den : 0.0f);
}

/**
 * @brief Unpack a stored cache back to FP16 — used by tests and by the CPU
 *        offload path, never on the GPU hot path.
 */
template <int FMT>
__global__ void kv_dequant_kernel(
    const uint8_t* __restrict__ store, const half* __restrict__ meta,
    const half* __restrict__ ref,
    half* __restrict__ out, int n_kv_heads, int head_dim, int n_pos, int bpv)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= n_pos * n_kv_heads) return;
    const int lane = threadIdx.x;
    const int d0 = lane * 4;

    float v[4];
    kv_read4<FMT>(store + (size_t)idx * bpv, meta ? meta + idx : NULL, lane, v);

    if (FMT >= VV_KV_TQ4) {          /* undo the rotation for inspection */
        warp_wht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= WHT_NORM * kv_sign(d0 + i);
    }
    if (ref) {
        const half* r = ref + (size_t)(idx % n_kv_heads) * head_dim + d0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += __half2float(r[i]);
    }
    half* p = out + (size_t)idx * head_dim + d0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __float2half(v[i]);
}

/* ─── Flash prefill over a quantized cache ───────────────────────────────── */

#define QP_BR      8
#define QP_BC      32
#define QP_THREADS 256
#define QP_KV_PAD  2

template <int FMT>
__global__ void q_prefill_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ k_store,
    const uint8_t* __restrict__ v_store,
    const half* __restrict__ k_meta,
    const half* __restrict__ v_meta,
    half* __restrict__ O,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, int bpv, float scale, bool causal)
{
    const int head     = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;
    const int kv_head  = head / (n_q_heads / n_kv_heads);

    const int q_row  = tile_row * QP_BR + warp_id;
    const bool alive = (q_row < q_len);
    const int q_abs  = q_offset + q_row;

    const int tile_last = tile_row * QP_BR + QP_BR - 1;
    int block_max_kv = causal ? (q_offset + tile_last + 1) : kv_len;
    if (block_max_kv > kv_len) block_max_kv = kv_len;

    const int q_stride = n_q_heads * head_dim;
    const int kv_row   = head_dim + QP_KV_PAD;

    extern __shared__ char smem_raw[];
    half* K_tile = (half*)smem_raw;
    half* V_tile = K_tile + QP_BC * kv_row;
    half* Q_smem = V_tile + QP_BC * kv_row;

    {
        const int q_base = q_row * q_stride + head * head_dim;
        for (int d = lane; d < head_dim; d += 32)
            Q_smem[warp_id * head_dim + d] =
                alive ? Q[q_base + d] : __float2half(0.0f);
    }
    __syncthreads();

    float o_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.0f;
    const int d0 = lane * 4;

    const int num_kv_tiles = (block_max_kv + QP_BC - 1) / QP_BC;

    for (int tj = 0; tj < num_kv_tiles; ++tj) {
        const int kv_start = tj * QP_BC;

        /* One warp per cached position: the packed layout is warp-shaped. */
        for (int r = warp_id; r < QP_BC; r += QP_BR) {
            const int gp = kv_start + r;
            float kk[4], vv[4];
            if (gp < kv_len) {
                const size_t vo = ((size_t)gp * n_kv_heads + kv_head) * (size_t)bpv;
                const size_t mo = (size_t)gp * n_kv_heads + kv_head;
                kv_read4<FMT>(k_store + vo, k_meta ? k_meta + mo : NULL, lane, kk);
                kv_read4<FMT>(v_store + vo, v_meta ? v_meta + mo : NULL, lane, vv);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) { kk[i] = 0.f; vv[i] = 0.f; }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                K_tile[r * kv_row + d0 + i] = __float2half(kk[i]);
                V_tile[r * kv_row + d0 + i] = __float2half(vv[i]);
            }
        }
        __syncthreads();

        const int kv_pos = kv_start + lane;
        float score = -FLT_MAX;
        if (alive && kv_pos < kv_len && (!causal || kv_pos <= q_abs)) {
            const half2* q_ptr = (const half2*)(Q_smem + warp_id * head_dim);
            const half2* k_ptr = (const half2*)(K_tile + lane * kv_row);
            float dot = 0.0f;
            for (int d = 0; d < head_dim / 2; ++d) {
                const float2 a = __half22float2(q_ptr[d]);
                const float2 b = __half22float2(k_ptr[d]);
                dot = fmaf(a.x, b.x, dot);
                dot = fmaf(a.y, b.y, dot);
            }
            score = dot * scale;
        }

        const float tile_max = warp_max(score);
        const float m_new = fmaxf(m_i, tile_max);
        const bool have_any = (m_new > -FLT_MAX);
        const float alpha = (have_any && m_i > -FLT_MAX) ? __expf(m_i - m_new) : 1.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_reg[i] *= alpha;
        l_i *= alpha;

        const float p = (score > -FLT_MAX && have_any) ? __expf(score - m_new) : 0.0f;
        l_i += warp_sum(p);

        const int tile_valid = min(QP_BC, kv_len - kv_start);
        for (int k = 0; k < tile_valid; ++k) {
            const float pk = __shfl_sync(0xFFFFFFFFu, p, k);
            if (pk != 0.0f) {
                const half* v_row = V_tile + k * kv_row + d0;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    o_reg[i] += pk * __half2float(v_row[i]);
            }
        }
        if (have_any) m_i = m_new;
        __syncthreads();
    }

    if (alive) {
        const float inv_l = (l_i > 1e-20f) ? (1.0f / l_i) : 0.0f;
        const int out_offset = q_row * q_stride + head * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            O[out_offset + d0 + i] = __float2half(o_reg[i] * inv_l);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C entry points
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

static float* s_qpart_o = NULL;
static float* s_qpart_m = NULL;
static float* s_qpart_l = NULL;
static int    s_qheads = 0, s_qdim = 0;

static vv_status_t ensure_qscratch(int n_heads, int head_dim) {
    if (s_qpart_o && n_heads <= s_qheads && head_dim <= s_qdim) return VV_OK;
    if (s_qpart_o) { cudaFree(s_qpart_o); cudaFree(s_qpart_m); cudaFree(s_qpart_l); }
    s_qheads = n_heads; s_qdim = head_dim;
    const size_t no = (size_t)n_heads * QD_MAX_PARTS * head_dim * sizeof(float);
    const size_t nm = (size_t)n_heads * QD_MAX_PARTS * sizeof(float);
    if (cudaMalloc((void**)&s_qpart_o, no) != cudaSuccess) return VV_ERR_CUDA_OOM;
    if (cudaMalloc((void**)&s_qpart_m, nm) != cudaSuccess) return VV_ERR_CUDA_OOM;
    if (cudaMalloc((void**)&s_qpart_l, nm) != cudaSuccess) return VV_ERR_CUDA_OOM;
    return VV_OK;
}

void vv_kv_quant_cleanup(void) {
    if (s_qpart_o) { cudaFree(s_qpart_o); s_qpart_o = NULL; }
    if (s_qpart_m) { cudaFree(s_qpart_m); s_qpart_m = NULL; }
    if (s_qpart_l) { cudaFree(s_qpart_l); s_qpart_l = NULL; }
    s_qheads = s_qdim = 0;
}

vv_status_t vv_kv_rotate_dev(void* x, int n_heads, int head_dim,
                             int rows, void* stream)
{
    if (!x) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int total = rows * n_heads;
    dim3 block(32, 4);
    kv_rotate_kernel<<<(total + 3) / 4, block, 0, (cudaStream_t)stream>>>(
        (half*)x, n_heads, head_dim, rows, false);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_kv_unrotate_dev(void* x, int n_heads, int head_dim,
                               int rows, void* stream)
{
    if (!x) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int total = rows * n_heads;
    dim3 block(32, 4);
    kv_rotate_kernel<<<(total + 3) / 4, block, 0, (cudaStream_t)stream>>>(
        (half*)x, n_heads, head_dim, rows, true);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

#define DISPATCH_Q(fmt, CALL)                                   \
    switch (fmt) {                                              \
        case VV_KV_FP8_E4M3: CALL(VV_KV_FP8_E4M3); break;       \
        case VV_KV_FP8_E5M2: CALL(VV_KV_FP8_E5M2); break;       \
        case VV_KV_TQ4:      CALL(VV_KV_TQ4);      break;       \
        case VV_KV_TQ3:      CALL(VV_KV_TQ3);      break;       \
        case VV_KV_TQ2:      CALL(VV_KV_TQ2);      break;       \
        case VV_KV_TQ1_5:    CALL(VV_KV_TQ1_5);    break;       \
        default: return VV_ERR_UNSUPPORTED;                     \
    }

vv_status_t vv_kv_quant_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos,
    int n_positions, int kv_format, void* stream)
{
    if (!k_fp16 || !v_fp16 || !k_store || !v_store) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    if (n_positions <= 0) return VV_OK;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const int total = n_positions * n_kv_heads;
    const int wpb = 4;                       /* warps per block */
    dim3 block(32, wpb);
    const int blocks = (total + wpb - 1) / wpb;
    const size_t sh = (size_t)wpb * 16 * sizeof(uint32_t);
    cudaStream_t st = (cudaStream_t)stream;

    if (k_ref && build_ref) {
        kv_build_ref_kernel<<<n_kv_heads, head_dim, 0, st>>>(
            (const half*)k_fp16, (half*)k_ref, n_kv_heads, head_dim,
            n_positions);
    }

#define STORE_CALL(F)                                                        \
    kv_store_kernel<F><<<blocks, block, sh, st>>>(                           \
        (const half*)k_fp16, (uint8_t*)k_store, (half*)k_meta,               \
        (const half*)k_ref, n_kv_heads, head_dim, pos, n_positions, bpv);    \
    kv_store_kernel<F><<<blocks, block, sh, st>>>(                           \
        (const half*)v_fp16, (uint8_t*)v_store, (half*)v_meta,               \
        NULL, n_kv_heads, head_dim, pos, n_positions, bpv);

    DISPATCH_Q(kv_format, STORE_CALL)
#undef STORE_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_kv_dequant_dev(
    const void* store, const void* meta, const void* ref, void* out_fp16,
    int n_kv_heads, int head_dim, int n_pos, int kv_format, void* stream)
{
    if (!store || !out_fp16) return VV_ERR_NULL_PTR;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const int total = n_pos * n_kv_heads;
    dim3 block(32, 4);
    const int blocks = (total + 3) / 4;
    cudaStream_t s = (cudaStream_t)stream;

#define DEQ_CALL(F)                                                              kv_dequant_kernel<F><<<blocks, block, 0, s>>>(                                   (const uint8_t*)store, (const half*)meta, (const half*)ref,                  (half*)out_fp16, n_kv_heads, head_dim, n_pos, bpv);

    DISPATCH_Q(kv_format, DEQ_CALL)
#undef DEQ_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_decode_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    int kv_format, void* stream)
{
    if (!q || !k_store || !v_store || !output) return VV_ERR_NULL_PTR;
    if (cache_len == 0) return VV_OK;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;

    vv_status_t st = ensure_qscratch(n_q_heads, head_dim);
    if (st != VV_OK) return st;

    int n_parts = (cache_len + QD_POS_PER_WARP - 1) / QD_POS_PER_WARP;
    if (n_parts < 1) n_parts = 1;
    if (n_parts > QD_MAX_PARTS) n_parts = QD_MAX_PARTS;
    n_parts = ((n_parts + QD_WARPS - 1) / QD_WARPS) * QD_WARPS;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(n_q_heads, n_parts / QD_WARPS);
    dim3 block(32, QD_WARPS);
    cudaStream_t s = (cudaStream_t)stream;

#define DEC_CALL(F)                                                          \
    q_decode_split_kernel<F><<<grid, block, 0, s>>>(                         \
        (const half*)q, (const uint8_t*)k_store, (const uint8_t*)v_store,    \
        (const half*)k_meta, (const half*)v_meta,                            \
        s_qpart_o, s_qpart_m, s_qpart_l,                                     \
        n_q_heads, n_kv_heads, head_dim, cache_len, n_parts, bpv, scale);

    DISPATCH_Q(kv_format, DEC_CALL)
#undef DEC_CALL

    const size_t shb = (size_t)n_parts * 2 * sizeof(float);
    const int cthreads = head_dim > n_parts ? head_dim : n_parts;
    q_decode_combine_kernel<<<n_q_heads, cthreads, shb, s>>>(
        s_qpart_o, s_qpart_m, s_qpart_l, (half*)output, n_parts, head_dim);

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_gqa_attention_prefill_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal,
    int kv_format, void* stream)
{
    if (!q || !k_store || !v_store || !output) return VV_ERR_NULL_PTR;
    if (q_len == 0 || kv_len == 0) return VV_OK;
    if (head_dim != 128) return VV_ERR_UNSUPPORTED;

    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)kv_format, head_dim);
    const float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(n_q_heads, (q_len + QP_BR - 1) / QP_BR);
    dim3 block(32, QP_BR);
    const size_t shb = (size_t)(2 * QP_BC * (head_dim + QP_KV_PAD)
                                + QP_BR * head_dim) * sizeof(half);
    cudaStream_t s = (cudaStream_t)stream;

#define PRE_CALL(F)                                                          \
    q_prefill_kernel<F><<<grid, block, shb, s>>>(                            \
        (const half*)q, (const uint8_t*)k_store, (const uint8_t*)v_store,    \
        (const half*)k_meta, (const half*)v_meta, (half*)output,             \
        n_q_heads, n_kv_heads, head_dim, q_len, q_offset, kv_len,            \
        bpv, scale, causal);

    DISPATCH_Q(kv_format, PRE_CALL)
#undef PRE_CALL

    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
