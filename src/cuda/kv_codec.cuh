/**
 * @file kv_codec.cuh
 * @brief Encoding and decoding of one stored KV vector, shared by the
 *        quantizing store and every kernel that reads a quantized cache.
 *
 * The tables are `static __constant__`: each translation unit that includes
 * this gets its own copy with a compile-time initializer, which is what the
 * build (no relocatable device code) and graph capture (no runtime symbol
 * copies) both need.
 */
#ifndef VV_CUDA_KV_CODEC_CUH
#define VV_CUDA_KV_CODEC_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#include "vibevoice/kv_quant.h"

/**
 * @brief Row of the store that holds logical position `t`.
 *
 * A contiguous cache (no table) stores position t at row t. A paged one maps
 * each run of VV_KV_PAGE_SIZE positions to a page of the pool through the
 * context's table, so only the page index moves and the offset inside the
 * page stays put.
 */
__device__ __forceinline__ int kv_phys_row(const int* __restrict__ page_table,
                                           int t)
{
    return page_table
        ? page_table[t / VV_KV_PAGE_SIZE] * VV_KV_PAGE_SIZE
          + (t % VV_KV_PAGE_SIZE)
        : t;
}

/* ─── Quantizer codebooks ────────────────────────────────────────────────── */
/*
 * Lloyd-Max levels for a unit Gaussian, computed offline (tools/lloyd_max.py).
 * SQNR: 4-bit 20.2 dB, 3-bit 14.6 dB, 2-bit 9.3 dB, 1.5-bit 7.0 dB.
 */
static __constant__ float c_lev4[16] = {
    -2.725112f, -2.065280f, -1.613787f, -1.251845f,
    -0.938657f, -0.654379f, -0.386970f, -0.127960f,
    +0.127960f, +0.386970f, +0.654379f, +0.938657f,
    +1.251845f, +1.613787f, +2.065280f, +2.725112f
};
static __constant__ float c_lev3[8] = {
    -2.150300f, -1.343258f, -0.755588f, -0.244756f,
    +0.244756f, +0.755588f, +1.343258f, +2.150300f
};
static __constant__ float c_lev2[4] = {
    -1.511418f, -0.452456f, +0.452456f, +1.511418f
};
/* 8-entry 2-D codebook; a pair of coordinates shares one 3-bit index. */
static __constant__ float c_vq15[16] = {
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

#endif /* VV_CUDA_KV_CODEC_CUH */
