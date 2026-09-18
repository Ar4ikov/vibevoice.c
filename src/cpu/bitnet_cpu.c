/**
 * @file bitnet_cpu.c
 * @brief Integer CPU kernels for VibeVoice-ASR-BitNet. See bitnet.h.
 *
 * Two products live here, and everything else is quantization around them:
 *
 *   ternary x int8   the 196 BitNet projections of the 1.5B LM
 *   int8 x int8      the I8_S speech encoder, and the quantized LM head
 *
 * Both are exact in int32, so every ISA below computes the same bits and the
 * tests compare them to a scalar loop with ==, not with a tolerance.
 *
 * x86: AVX2 is the baseline (`vpmaddubsw` on unsigned codes x signed
 * activations, the same instruction sequence VibeASR.cpp uses). AVX-VNNI and
 * AVX512-VNNI replace the maddubs + madd pair with one `vpdpbusd`, which also
 * removes the int16 intermediate, so they need no periodic widening. The
 * family is picked once at runtime from CPUID, so one binary runs anywhere.
 * Every SIMD function carries its own target attribute on GCC/Clang, and MSVC
 * accepts the intrinsics without one, so the MSVC build gets the same code.
 *
 * ARM: `sdot` when the compiler targets it (__ARM_FEATURE_DOTPROD), else
 * `smull` + `sadalp`, which is exact as well.
 */

#include "vibevoice/bitnet.h"
#include "vibevoice/vibevoice.h"

#include "vv_thread.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define VV_BN_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define VV_BN_NEON 1
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VV_TGT(x) __attribute__((target(x)))
#else
#define VV_TGT(x)
#endif

/* AVX-VNNI intrinsics arrived in GCC 11, Clang 12 and MSVC 17.0. */
#if defined(VV_BN_X86) &&                                                  \
    ((defined(__clang__) && __clang_major__ >= 12) ||                      \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11) ||       \
     (defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1930))
#define VV_BN_HAVE_AVXVNNI 1
#endif
#if defined(VV_BN_X86) &&                                                  \
    ((defined(__clang__) && __clang_major__ >= 8) ||                       \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8) ||        \
     (defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1920))
#define VV_BN_HAVE_AVX512VNNI 1
#endif

#define TGT_AVX2 VV_TGT("avx2")

#if defined(__GNUC__) || defined(__clang__)
#define VV_AI inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define VV_AI __forceinline
#else
#define VV_AI inline
#endif
#define TGT_AVXVNNI VV_TGT("avx2,avxvnni")
#define TGT_AVX512 VV_TGT("avx2,avx512f,avx512bw,avx512vnni")

/* ─── ISA selection ─────────────────────────────────────────────────────── */

enum {
    ISA_AUTO = 0,
    ISA_SCALAR = 1,
    ISA_AVX2 = 2,
    ISA_AVXVNNI = 3,
    ISA_AVX512VNNI = 4,
    ISA_NEON = 5,
};

#ifdef VV_BN_X86
static void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
    int v[4];
    __cpuidex(v, leaf, sub);
    for (int i = 0; i < 4; i++) r[i] = (unsigned)v[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static unsigned long long xgetbv0(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#endif
}
#endif

/** What the CPU and OS support: bit (1 << ISA_*). */
static unsigned detect_isa_mask(void) {
    unsigned m = 1u << ISA_SCALAR;
#ifdef VV_BN_X86
    unsigned r[4];
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    cpuid(1, 0, r);
    const int osxsave = (r[2] >> 27) & 1;
    if (!osxsave || max_leaf < 7) return m;
    const unsigned long long xcr0 = xgetbv0();
    const int ymm_os = (xcr0 & 0x6) == 0x6;
    const int zmm_os = (xcr0 & 0xE6) == 0xE6;
    cpuid(7, 0, r);
    const int avx2 = (r[1] >> 5) & 1;
    const int avx512f = (r[1] >> 16) & 1;
    const int avx512bw = (r[1] >> 30) & 1;
    const int avx512vnni = (r[2] >> 11) & 1;
    unsigned r1[4];
    cpuid(7, 1, r1);
    const int avxvnni = (r1[0] >> 4) & 1;
    if (ymm_os && avx2) m |= 1u << ISA_AVX2;
#ifdef VV_BN_HAVE_AVXVNNI
    if (ymm_os && avx2 && avxvnni) m |= 1u << ISA_AVXVNNI;
#endif
#ifdef VV_BN_HAVE_AVX512VNNI
    if (zmm_os && avx2 && avx512f && avx512bw && avx512vnni)
        m |= 1u << ISA_AVX512VNNI;
#endif
    (void)avxvnni;
    (void)avx512vnni;
#endif
#ifdef VV_BN_NEON
    m |= 1u << ISA_NEON;
#endif
    return m;
}

static int best_isa(unsigned mask) {
    if (mask & (1u << ISA_AVX512VNNI)) return ISA_AVX512VNNI;
    if (mask & (1u << ISA_AVXVNNI)) return ISA_AVXVNNI;
    if (mask & (1u << ISA_AVX2)) return ISA_AVX2;
    if (mask & (1u << ISA_NEON)) return ISA_NEON;
    return ISA_SCALAR;
}

/* The chosen family. Written once by vv_once (or by the test hook before any
 * kernel runs), read-only afterwards. */
static int g_isa = ISA_AUTO;
static unsigned g_isa_mask = 0;
static vv_once_t g_isa_once = VV_ONCE_INIT;

static void init_isa(void) {
    g_isa_mask = detect_isa_mask();
    g_isa = best_isa(g_isa_mask);
    const char* env = getenv("VV_BITNET_ISA");
    if (env && *env) {
        const int want = atoi(env);
        if (want > 0 && (g_isa_mask & (1u << want))) g_isa = want;
    }
}

static int isa(void) {
    vv_once(&g_isa_once, init_isa);
    return g_isa;
}

vv_status_t vv_bitnet_cpu_force_isa(int want) {
    isa();
    if (want == ISA_AUTO) {
        g_isa = best_isa(g_isa_mask);
        return VV_OK;
    }
    if (want < 0 || want > ISA_NEON || !(g_isa_mask & (1u << want)))
        return VV_ERR_UNSUPPORTED;
    g_isa = want;
    return VV_OK;
}

const char* vv_bitnet_cpu_isa(void) {
    switch (isa()) {
        case ISA_AVX512VNNI: return "AVX512-VNNI";
        case ISA_AVXVNNI: return "AVX-VNNI";
        case ISA_AVX2: return "AVX2";
#if defined(__ARM_FEATURE_DOTPROD)
        case ISA_NEON: return "NEON-dotprod";
#else
        case ISA_NEON: return "NEON";
#endif
        default: return "scalar";
    }
}

/* ─── Weight preparation ────────────────────────────────────────────────── */

vv_status_t vv_ternary_pack(const int8_t* t, int64_t N, int64_t K, uint8_t* codes) {
    if (!t || !codes) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || K % VV_TERNARY_BLOCK) return VV_ERR_INVALID_ARG;
    const int64_t nblk = N * K / VV_TERNARY_BLOCK;
    for (int64_t b = 0; b < nblk; b++) {
        const int8_t* src = t + b * VV_TERNARY_BLOCK;
        uint8_t* dst = codes + b * 32;
        for (int j = 0; j < 32; j++) {
            dst[j] = (uint8_t)(((src[j] + 1) << 6) | ((src[j + 32] + 1) << 4) |
                               ((src[j + 64] + 1) << 2) | (src[j + 96] + 1));
        }
    }
    return VV_OK;
}

void vv_ternary_unpack_row(const uint8_t* codes, int64_t K, int8_t* t) {
    for (int64_t b = 0; b < K / VV_TERNARY_BLOCK; b++) {
        const uint8_t* src = codes + b * 32;
        int8_t* dst = t + b * VV_TERNARY_BLOCK;
        for (int j = 0; j < 32; j++) {
            dst[j] = (int8_t)(((src[j] >> 6) & 3) - 1);
            dst[j + 32] = (int8_t)(((src[j] >> 4) & 3) - 1);
            dst[j + 64] = (int8_t)(((src[j] >> 2) & 3) - 1);
            dst[j + 96] = (int8_t)((src[j] & 3) - 1);
        }
    }
}

vv_status_t vv_ternarize_f32(const float* w, int64_t N, int64_t K,
                             uint8_t* codes, float* scale) {
    if (!w || !codes || !scale) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || K % VV_TERNARY_BLOCK) return VV_ERR_INVALID_ARG;
    const int64_t n = N * K;

    double acc = 0.0;
    int64_t i;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
    for (i = 0; i < n; i++) acc += fabs((double)w[i]);
    float mean = (float)(acc / (double)n);
    if (mean < 1e-5f) mean = 1e-5f;
    const float s = 1.0f / mean;
    /* max|W'| over a tensor with at least one nonzero value is exactly
     * fl(1/s); an all-zero tensor gets scale 0 like the reference. */
    const float inv = 1.0f / s;
    int any = 0;

    int64_t b;
#ifdef _OPENMP
#pragma omp parallel for reduction(| : any) schedule(static)
#endif
    for (b = 0; b < n / VV_TERNARY_BLOCK; b++) {
        uint8_t* dst = codes + b * 32;
        memset(dst, 0, 32);
        for (int j = 0; j < VV_TERNARY_BLOCK; j++) {
            /* torch.round is round-half-to-even, as is nearbyintf in the
             * default rounding mode */
            float r = nearbyintf(w[b * VV_TERNARY_BLOCK + j] * s);
            if (r > 1.0f) r = 1.0f;
            if (r < -1.0f) r = -1.0f;
            const float wq = r / s;
            uint8_t c;
            if (fabsf(wq) < 1e-6f) c = 1;
            else { c = wq > 0 ? 2 : 0; any = 1; }
            dst[j & 31] |= (uint8_t)(c << (6 - 2 * (j >> 5)));
        }
    }
    *scale = any ? inv : 0.0f;
    return VV_OK;
}

vv_status_t vv_i8s_quantize_f32(const float* w, int64_t N, int64_t K,
                                int8_t* q, float* scale) {
    if (!w || !q || !scale) return VV_ERR_NULL_PTR;
    const int64_t n = N * K;
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        const float a = fabsf(w[i]);
        if (a > amax) amax = a;
    }
    const float s = amax > 0.0f ? 127.0f / amax : 1.0f;
    for (int64_t i = 0; i < n; i++) {
        int32_t v = (int32_t)roundf(w[i] * s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    *scale = 1.0f / s;
    return VV_OK;
}

vv_status_t vv_i8_rowquant_f32(const float* w, int64_t N, int64_t K,
                               int8_t* q, float* scales) {
    if (!w || !q || !scales) return VV_ERR_NULL_PTR;
    int64_t r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (r = 0; r < N; r++) {
        const float* row = w + r * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            const float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float v = nearbyintf(row[k] * id);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[r * K + k] = (int8_t)v;
        }
        scales[r] = d;
    }
    return VV_OK;
}

/* ─── Activation quantization ───────────────────────────────────────────── */

/* round-half-to-even via the 1.5 * 2^23 trick, as ggml's nearest_int */
static inline int nearest_int(float f) {
    float v = f + 12582912.0f;
    int32_t i;
    memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}

vv_status_t vv_act_quant_i8_cpu(const float* x, int M, int K, int8_t* q,
                                float* scale, int32_t* sum) {
    if (!x || !q || !scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    int m;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M > 1)
#endif
    for (m = 0; m < M; m++) {
        const float* xr = x + (size_t)m * K;
        int8_t* qr = q + (size_t)m * K;
        /* max|x| is order-independent, so the float max is the same number
         * as ggml's double max of fabs((double)x) */
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = fabsf(xr[k]);
            if (a > amax) amax = a;
        }
        const double dmax = amax > 0.00001 ? (double)amax : 0.00001;
        const float s = (float)(127.0 / dmax);
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            int v = nearest_int(xr[k] * s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            acc += v;
            qr[k] = (int8_t)v;
        }
        scale[m] = s;
        if (sum) sum[m] = acc;
    }
    return VV_OK;
}

/* ─── Ternary x int8: scalar ────────────────────────────────────────────── */

/** acc[t] for one weight row against `nt` activation rows (stride ldq). */
static void tern_row_scalar(const uint8_t* codes, const int8_t* q, int ldq,
                            int nt, int K, int32_t* acc) {
    for (int t = 0; t < nt; t++) {
        const int8_t* x = q + (size_t)t * ldq;
        int32_t s = 0;
        for (int b = 0; b < K / VV_TERNARY_BLOCK; b++) {
            const uint8_t* w = codes + b * 32;
            const int8_t* xb = x + b * VV_TERNARY_BLOCK;
            for (int j = 0; j < 32; j++) {
                s += (((w[j] >> 6) & 3) - 1) * xb[j];
                s += (((w[j] >> 4) & 3) - 1) * xb[j + 32];
                s += (((w[j] >> 2) & 3) - 1) * xb[j + 64];
                s += ((w[j] & 3) - 1) * xb[j + 96];
            }
        }
        acc[t] = s;
    }
}

/* ─── Ternary x int8: AVX2 ──────────────────────────────────────────────── */

#ifdef VV_BN_X86

TGT_AVX2 static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/*
 * One weight row against up to 4 activation rows. The codes are unpacked
 * once per block and reused for every activation row, which is what makes
 * the prefill case compute-bound rather than unpack-bound.
 *
 * maddubs(code u8 in [0,2], x s8) gives pairs of at most 2*2*128 = 512 in
 * magnitude, four per block, so an int16 lane holds 16 blocks (-32768 ..
 * 32512) before it has to be widened.
 */
TGT_AVX2 static VV_AI void tern_rows_avx2_n(const uint8_t* codes, const int8_t* q,
                                            int ldq, int K, const int32_t* xsum,
                                            int32_t* acc, const int NT) {
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const int nb = K / VV_TERNARY_BLOCK;
    __m256i a32[4], a16[4];
    for (int t = 0; t < NT; t++) a32[t] = a16[t] = _mm256_setzero_si256();

    for (int b = 0; b < nb; b++) {
        const __m256i w = _mm256_loadu_si256((const __m256i*)(codes + b * 32));
        const __m256i c3 = _mm256_and_si256(w, mask);
        const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(w, 2), mask);
        const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
        const __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(w, 6), mask);
        for (int t = 0; t < NT; t++) {
            const int8_t* x = q + (size_t)t * ldq + b * VV_TERNARY_BLOCK;
            const __m256i p0 = _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i*)(x)));
            const __m256i p1 = _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i*)(x + 32)));
            const __m256i p2 = _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i*)(x + 64)));
            const __m256i p3 = _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i*)(x + 96)));
            a16[t] = _mm256_add_epi16(a16[t], _mm256_add_epi16(_mm256_add_epi16(p0, p1),
                                                              _mm256_add_epi16(p2, p3)));
        }
        if ((b & 15) == 15 || b == nb - 1) {
            for (int t = 0; t < NT; t++) {
                a32[t] = _mm256_add_epi32(a32[t], _mm256_madd_epi16(a16[t], ones));
                a16[t] = _mm256_setzero_si256();
            }
        }
    }
    for (int t = 0; t < NT; t++) acc[t] = hsum_epi32_avx2(a32[t]) - xsum[t];
}

/* The token count is a compile-time constant in each case, so the
 * accumulators live in registers instead of a spilled array. */
TGT_AVX2 static void tern_row_avx2(const uint8_t* codes, const int8_t* q,
                                   int ldq, int nt, int K, const int32_t* xsum,
                                   int32_t* acc) {
    switch (nt) {
        case 1: tern_rows_avx2_n(codes, q, ldq, K, xsum, acc, 1); break;
        case 2: tern_rows_avx2_n(codes, q, ldq, K, xsum, acc, 2); break;
        case 3: tern_rows_avx2_n(codes, q, ldq, K, xsum, acc, 3); break;
        default: tern_rows_avx2_n(codes, q, ldq, K, xsum, acc, 4); break;
    }
}

#ifdef VV_BN_HAVE_AVXVNNI
/* vpdpbusd accumulates u8 x s8 quads straight into int32: no int16 stage. */
TGT_AVXVNNI static VV_AI void tern_row_avxvnni_n(const uint8_t* codes, const int8_t* q,
                                         int ldq, int nt_unused, int K,
                                         const int32_t* xsum, int32_t* acc, const int NT) {
    (void)nt_unused;
    const __m256i mask = _mm256_set1_epi8(3);
    const int nb = K / VV_TERNARY_BLOCK;
    __m256i a32[4];
    for (int t = 0; t < 4; t++) a32[t] = _mm256_setzero_si256();
    for (int b = 0; b < nb; b++) {
        const __m256i w = _mm256_loadu_si256((const __m256i*)(codes + b * 32));
        const __m256i c3 = _mm256_and_si256(w, mask);
        const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(w, 2), mask);
        const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
        const __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(w, 6), mask);
        for (int t = 0; t < NT; t++) {
            const int8_t* x = q + (size_t)t * ldq + b * VV_TERNARY_BLOCK;
            __m256i a = a32[t];
            a = _mm256_dpbusd_avx_epi32(a, c0, _mm256_loadu_si256((const __m256i*)(x)));
            a = _mm256_dpbusd_avx_epi32(a, c1, _mm256_loadu_si256((const __m256i*)(x + 32)));
            a = _mm256_dpbusd_avx_epi32(a, c2, _mm256_loadu_si256((const __m256i*)(x + 64)));
            a = _mm256_dpbusd_avx_epi32(a, c3, _mm256_loadu_si256((const __m256i*)(x + 96)));
            a32[t] = a;
        }
    }
    for (int t = 0; t < NT; t++) acc[t] = hsum_epi32_avx2(a32[t]) - xsum[t];
}

TGT_AVXVNNI static void tern_row_avxvnni(const uint8_t* codes, const int8_t* q,
                                         int ldq, int nt, int K,
                                         const int32_t* xsum, int32_t* acc) {
    switch (nt) {
        case 1: tern_row_avxvnni_n(codes, q, ldq, nt, K, xsum, acc, 1); break;
        case 2: tern_row_avxvnni_n(codes, q, ldq, nt, K, xsum, acc, 2); break;
        case 3: tern_row_avxvnni_n(codes, q, ldq, nt, K, xsum, acc, 3); break;
        default: tern_row_avxvnni_n(codes, q, ldq, nt, K, xsum, acc, 4); break;
    }
}
#endif

#ifdef VV_BN_HAVE_AVX512VNNI
/*
 * Two blocks per 512-bit step: the low half of every code vector comes from
 * block b and the high half from b + 1, so the activation vectors are the
 * matching 32-byte runs of both blocks side by side.
 */
TGT_AVX512 static VV_AI void tern_row_avx512_n(const uint8_t* codes, const int8_t* q,
                                       int ldq, int nt_unused, int K,
                                       const int32_t* xsum, int32_t* acc, const int NT) {
    (void)nt_unused;
    const __m512i mask = _mm512_set1_epi8(3);
    const int nb = K / VV_TERNARY_BLOCK;
    __m512i a32[4];
    for (int t = 0; t < 4; t++) a32[t] = _mm512_setzero_si512();
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const __m512i w = _mm512_loadu_si512((const void*)(codes + b * 32));
        const __m512i c3 = _mm512_and_si512(w, mask);
        const __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(w, 2), mask);
        const __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(w, 4), mask);
        const __m512i c0 = _mm512_and_si512(_mm512_srli_epi16(w, 6), mask);
        for (int t = 0; t < NT; t++) {
            const int8_t* x = q + (size_t)t * ldq + b * VV_TERNARY_BLOCK;
#define VV_X2(off) _mm512_inserti64x4(                                          \
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(x + (off)))), \
        _mm256_loadu_si256((const __m256i*)(x + VV_TERNARY_BLOCK + (off))), 1)
            __m512i a = a32[t];
            a = _mm512_dpbusd_epi32(a, c0, VV_X2(0));
            a = _mm512_dpbusd_epi32(a, c1, VV_X2(32));
            a = _mm512_dpbusd_epi32(a, c2, VV_X2(64));
            a = _mm512_dpbusd_epi32(a, c3, VV_X2(96));
#undef VV_X2
            a32[t] = a;
        }
    }
    int32_t tail[4] = { 0, 0, 0, 0 };
    if (b < nb) {
        /* odd block count: finish the last block with the AVX2 kernel
         * (xsum is applied once, below) */
        const int32_t zero[4] = { 0, 0, 0, 0 };
        tern_row_avx2(codes + b * 32, q + b * VV_TERNARY_BLOCK, ldq, NT,
                      VV_TERNARY_BLOCK, zero, tail);
    }
    for (int t = 0; t < NT; t++)
        acc[t] = _mm512_reduce_add_epi32(a32[t]) + tail[t] - xsum[t];
}

TGT_AVX512 static void tern_row_avx512(const uint8_t* codes, const int8_t* q,
                                       int ldq, int nt, int K,
                                       const int32_t* xsum, int32_t* acc) {
    switch (nt) {
        case 1: tern_row_avx512_n(codes, q, ldq, nt, K, xsum, acc, 1); break;
        case 2: tern_row_avx512_n(codes, q, ldq, nt, K, xsum, acc, 2); break;
        case 3: tern_row_avx512_n(codes, q, ldq, nt, K, xsum, acc, 3); break;
        default: tern_row_avx512_n(codes, q, ldq, nt, K, xsum, acc, 4); break;
    }
}
#endif

#endif /* VV_BN_X86 */

/* ─── Ternary x int8: NEON ──────────────────────────────────────────────── */

#ifdef VV_BN_NEON
/*
 * On ARM the codes are turned into signed ternary values (c - 1) before the
 * dot product, so no activation-sum correction is needed: sdot takes s8 x s8.
 */
static VV_AI void tern_row_neon_n(const uint8_t* codes, const int8_t* q, int ldq,
                          int nt_unused, int K, const int32_t* xsum, int32_t* acc, const int NT) {
    (void)xsum;
    (void)nt_unused;
    const uint8x16_t mask = vdupq_n_u8(3);
    const int8x16_t one = vdupq_n_s8(1);
    const int nb = K / VV_TERNARY_BLOCK;
    int32x4_t a32[4];
    for (int t = 0; t < 4; t++) a32[t] = vdupq_n_s32(0);
    for (int b = 0; b < nb; b++) {
        for (int h = 0; h < 2; h++) { /* two 16-byte halves of the block */
            const uint8x16_t w = vld1q_u8(codes + b * 32 + h * 16);
            const int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w, 6)), one);
            const int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w, 4), mask)), one);
            const int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w, 2), mask)), one);
            const int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w, mask)), one);
            for (int t = 0; t < NT; t++) {
                const int8_t* x = q + (size_t)t * ldq + b * VV_TERNARY_BLOCK + h * 16;
                const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 32);
                const int8x16_t x2 = vld1q_s8(x + 64), x3 = vld1q_s8(x + 96);
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t a = a32[t];
                a = vdotq_s32(a, t0, x0);
                a = vdotq_s32(a, t1, x1);
                a = vdotq_s32(a, t2, x2);
                a = vdotq_s32(a, t3, x3);
                a32[t] = a;
#else
                /* |t * x| <= 128, so two products fit an int16 lane before
                 * sadalp widens them */
                int16x8_t p = vmull_s8(vget_low_s8(t0), vget_low_s8(x0));
                p = vmlal_s8(p, vget_high_s8(t0), vget_high_s8(x0));
                a32[t] = vpadalq_s16(a32[t], p);
                p = vmull_s8(vget_low_s8(t1), vget_low_s8(x1));
                p = vmlal_s8(p, vget_high_s8(t1), vget_high_s8(x1));
                a32[t] = vpadalq_s16(a32[t], p);
                p = vmull_s8(vget_low_s8(t2), vget_low_s8(x2));
                p = vmlal_s8(p, vget_high_s8(t2), vget_high_s8(x2));
                a32[t] = vpadalq_s16(a32[t], p);
                p = vmull_s8(vget_low_s8(t3), vget_low_s8(x3));
                p = vmlal_s8(p, vget_high_s8(t3), vget_high_s8(x3));
                a32[t] = vpadalq_s16(a32[t], p);
#endif
            }
        }
    }
    for (int t = 0; t < NT; t++) acc[t] = vaddvq_s32(a32[t]);
}

static void tern_row_neon(const uint8_t* codes, const int8_t* q, int ldq,
                          int nt, int K, const int32_t* xsum, int32_t* acc) {
    switch (nt) {
        case 1: tern_row_neon_n(codes, q, ldq, nt, K, xsum, acc, 1); break;
        case 2: tern_row_neon_n(codes, q, ldq, nt, K, xsum, acc, 2); break;
        case 3: tern_row_neon_n(codes, q, ldq, nt, K, xsum, acc, 3); break;
        default: tern_row_neon_n(codes, q, ldq, nt, K, xsum, acc, 4); break;
    }
}
#endif

typedef void (*tern_row_fn)(const uint8_t*, const int8_t*, int, int, int,
                            const int32_t*, int32_t*);

static void tern_row_scalar_fn(const uint8_t* codes, const int8_t* q, int ldq,
                               int nt, int K, const int32_t* xsum, int32_t* acc) {
    (void)xsum;
    tern_row_scalar(codes, q, ldq, nt, K, acc);
}

static tern_row_fn pick_tern(void) {
    switch (isa()) {
#ifdef VV_BN_X86
#ifdef VV_BN_HAVE_AVX512VNNI
        case ISA_AVX512VNNI: return tern_row_avx512;
#endif
#ifdef VV_BN_HAVE_AVXVNNI
        case ISA_AVXVNNI: return tern_row_avxvnni;
#endif
        case ISA_AVX2: return tern_row_avx2;
#endif
#ifdef VV_BN_NEON
        case ISA_NEON: return tern_row_neon;
#endif
        default: return tern_row_scalar_fn;
    }
}

/** Whether the picked row kernel needs the activation row sums. */
static int tern_needs_xsum(void) {
    const int i = isa();
    return i == ISA_AVX2 || i == ISA_AVXVNNI || i == ISA_AVX512VNNI;
}

/* Activation rows per M block: 64 rows of K = 8960 is 560 KB, about what
 * one Zen 3 core has in L2, and it is reused across every weight row. */
#define TERN_MBLOCK 64

/*
 * The shared driver. Threads split N; each walks its rows against the
 * activations in M blocks, four rows at a time, so one weight row (at most
 * 2.2 KB) stays in L1 across the whole block.
 */
static vv_status_t ternary_drive(const int8_t* q, const uint8_t* codes,
                                 int M, int N, int K, int32_t* acc_out,
                                 const float* scale, float w_scale,
                                 const float* bias, float* y) {
    if (!q || !codes) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0 || K % VV_TERNARY_BLOCK) return VV_ERR_INVALID_ARG;
    const tern_row_fn fn = pick_tern();
    const int need_sum = tern_needs_xsum();

    /* Row sums of the activations, for the unsigned-code kernels. Small
     * (M ints); on the stack up to 1024 rows. */
    int32_t xs_stack[1024];
    int32_t* xs = M <= 1024 ? xs_stack : (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)M);
    if (!xs) return VV_ERR_OUT_OF_MEMORY;
    if (need_sum) {
        for (int m = 0; m < M; m++) {
            int32_t s = 0;
            const int8_t* r = q + (size_t)m * K;
            for (int k = 0; k < K; k++) s += r[k];
            xs[m] = s;
        }
    } else {
        memset(xs, 0, sizeof(int32_t) * (size_t)M);
    }

    const size_t row_bytes = (size_t)K / 4;
    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((int64_t)M * N * K > 200000)
#endif
    for (n = 0; n < N; n++) {
        const uint8_t* wrow = codes + (size_t)n * row_bytes;
        const float b = bias ? bias[n] : 0.0f;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int nt = M - m0 < 4 ? M - m0 : 4;
            int32_t a[4];
            fn(wrow, q + (size_t)m0 * K, K, nt, K, xs + m0, a);
            for (int t = 0; t < nt; t++) {
                const size_t o = (size_t)(m0 + t) * N + n;
                if (acc_out) acc_out[o] = a[t];
                if (y) {
                    float v = (float)a[t] / scale[m0 + t] * w_scale;
                    if (bias) v += b;
                    y[o] = v;
                }
            }
        }
    }
    if (xs != xs_stack) vv_free(xs);
    return VV_OK;
}

/*
 * The M-blocked variant for prefill: weight rows outer, activation blocks
 * inner would stream the whole activation matrix once per row, so block M
 * first. Chosen when there are enough rows for it to matter.
 */
static vv_status_t ternary_drive_blocked(const int8_t* q, const uint8_t* codes,
                                         int M, int N, int K, int32_t* acc_out,
                                         const float* scale, float w_scale,
                                         const float* bias, float* y) {
    if (M <= TERN_MBLOCK)
        return ternary_drive(q, codes, M, N, K, acc_out, scale, w_scale, bias, y);
    for (int m0 = 0; m0 < M; m0 += TERN_MBLOCK) {
        const int mb = M - m0 < TERN_MBLOCK ? M - m0 : TERN_MBLOCK;
        /* each block writes its own rows of the [M, N] outputs; the driver
         * indexes as if M = mb, so offset the output pointers */
        vv_status_t s = ternary_drive(q + (size_t)m0 * K, codes, mb, N, K,
                                      acc_out ? acc_out + (size_t)m0 * N : NULL,
                                      scale ? scale + m0 : NULL, w_scale, bias,
                                      y ? y + (size_t)m0 * N : NULL);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

vv_status_t vv_ternary_gemm_i32_cpu(const int8_t* q, const uint8_t* codes,
                                    int32_t* acc, int M, int N, int K) {
    if (!acc) return VV_ERR_NULL_PTR;
    return ternary_drive_blocked(q, codes, M, N, K, acc, NULL, 0.0f, NULL, NULL);
}

vv_status_t vv_ternary_linear_cpu(const int8_t* q, const float* scale,
                                  const uint8_t* codes, float w_scale,
                                  const float* bias, float* y,
                                  int M, int N, int K) {
    if (!scale || !y) return VV_ERR_NULL_PTR;
    return ternary_drive_blocked(q, codes, M, N, K, NULL, scale, w_scale, bias, y);
}

/* ─── Int8 x int8 ───────────────────────────────────────────────────────── */

/* Up to 2 activation rows x 4 weight rows per call; out[t*4 + r]. */
typedef void (*i8_tile_fn)(const int8_t* a, int lda, int nt, const int8_t* w,
                           int ldw, int nr, int K, int32_t* out);

static void i8_tile_scalar(const int8_t* a, int lda, int nt, const int8_t* w,
                           int ldw, int nr, int K, int32_t* out) {
    for (int t = 0; t < nt; t++)
        for (int r = 0; r < nr; r++) {
            const int8_t* x = a + (size_t)t * lda;
            const int8_t* v = w + (size_t)r * ldw;
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)x[k] * v[k];
            out[t * 4 + r] = s;
        }
}

#ifdef VV_BN_X86
/*
 * Sign trick: maddubs needs one unsigned operand, so use |a| and move a's
 * sign onto the weight. That way round, a = -128 is harmless (|a| = 128 is a
 * valid u8) and only the weight is negated, which stays in range because
 * weights are clamped to [-127, 127]. Every pair is at most 2 * 128 * 127 =
 * 32512, so the int16 stage cannot saturate, and madd widens it right away.
 */
TGT_AVX2 static VV_AI void i8_tile_avx2_n(const int8_t* a, int lda, int nt_unused,
                                  const int8_t* w, int ldw, int nr_unused, int K,
                                  int32_t* out, const int NT, const int NR) {
    (void)nt_unused;
    (void)nr_unused;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc[2][4];
    for (int t = 0; t < 2; t++)
        for (int r = 0; r < 4; r++) acc[t][r] = _mm256_setzero_si256();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m256i x[2], ax[2], wv[4];
        for (int t = 0; t < NT; t++) {
            x[t] = _mm256_loadu_si256((const __m256i*)(a + (size_t)t * lda + k));
            ax[t] = _mm256_abs_epi8(x[t]); /* |-128| = 128 is fine as u8 */
        }
        for (int r = 0; r < NR; r++)
            wv[r] = _mm256_loadu_si256((const __m256i*)(w + (size_t)r * ldw + k));
        for (int t = 0; t < NT; t++)
            for (int r = 0; r < NR; r++) {
                const __m256i p = _mm256_maddubs_epi16(ax[t], _mm256_sign_epi8(wv[r], x[t]));
                acc[t][r] = _mm256_add_epi32(acc[t][r], _mm256_madd_epi16(p, ones));
            }
    }
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < NR; r++) {
            int32_t s = hsum_epi32_avx2(acc[t][r]);
            for (int kk = k; kk < K; kk++)
                s += (int32_t)a[(size_t)t * lda + kk] * w[(size_t)r * ldw + kk];
            out[t * 4 + r] = s;
        }
}

TGT_AVX2 static void i8_tile_avx2(const int8_t* a, int lda, int nt,
                                  const int8_t* w, int ldw, int nr, int K,
                                  int32_t* out) {
    if (nt == 2 && nr == 4) i8_tile_avx2_n(a, lda, nt, w, ldw, nr, K, out, 2, 4);
    else if (nt == 1 && nr == 4) i8_tile_avx2_n(a, lda, nt, w, ldw, nr, K, out, 1, 4);
    else i8_tile_avx2_n(a, lda, nt, w, ldw, nr, K, out, nt, nr);
}

#ifdef VV_BN_HAVE_AVXVNNI
TGT_AVXVNNI static VV_AI void i8_tile_avxvnni_n(const int8_t* a, int lda, int nt_unused,
                                        const int8_t* w, int ldw, int nr_unused, int K,
                                        int32_t* out, const int NT, const int NR) {
    (void)nt_unused;
    (void)nr_unused;
    __m256i acc[2][4];
    for (int t = 0; t < 2; t++)
        for (int r = 0; r < 4; r++) acc[t][r] = _mm256_setzero_si256();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m256i x[2], ax[2], wv[4];
        for (int t = 0; t < NT; t++) {
            x[t] = _mm256_loadu_si256((const __m256i*)(a + (size_t)t * lda + k));
            ax[t] = _mm256_abs_epi8(x[t]);
        }
        for (int r = 0; r < NR; r++)
            wv[r] = _mm256_loadu_si256((const __m256i*)(w + (size_t)r * ldw + k));
        for (int t = 0; t < NT; t++)
            for (int r = 0; r < NR; r++)
                acc[t][r] = _mm256_dpbusd_avx_epi32(acc[t][r], ax[t], _mm256_sign_epi8(wv[r], x[t]));
    }
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < NR; r++) {
            int32_t s = hsum_epi32_avx2(acc[t][r]);
            for (int kk = k; kk < K; kk++)
                s += (int32_t)a[(size_t)t * lda + kk] * w[(size_t)r * ldw + kk];
            out[t * 4 + r] = s;
        }
}

TGT_AVXVNNI static void i8_tile_avxvnni(const int8_t* a, int lda, int nt,
                                        const int8_t* w, int ldw, int nr, int K,
                                        int32_t* out) {
    if (nt == 2 && nr == 4) i8_tile_avxvnni_n(a, lda, nt, w, ldw, nr, K, out, 2, 4);
    else if (nt == 1 && nr == 4) i8_tile_avxvnni_n(a, lda, nt, w, ldw, nr, K, out, 1, 4);
    else i8_tile_avxvnni_n(a, lda, nt, w, ldw, nr, K, out, nt, nr);
}
#endif

#ifdef VV_BN_HAVE_AVX512VNNI
TGT_AVX512 static VV_AI void i8_tile_avx512_n(const int8_t* a, int lda, int nt_unused,
                                      const int8_t* w, int ldw, int nr_unused, int K,
                                      int32_t* out, const int NT, const int NR) {
    (void)nt_unused;
    (void)nr_unused;
    __m512i acc[2][4];
    for (int t = 0; t < 2; t++)
        for (int r = 0; r < 4; r++) acc[t][r] = _mm512_setzero_si512();
    int k = 0;
    for (; k + 64 <= K; k += 64) {
        __m512i x[2], ax[2], wv[4];
        __mmask64 neg[2], nz[2];
        for (int t = 0; t < NT; t++) {
            x[t] = _mm512_loadu_si512((const void*)(a + (size_t)t * lda + k));
            ax[t] = _mm512_abs_epi8(x[t]);
            neg[t] = _mm512_movepi8_mask(x[t]);
            nz[t] = _mm512_test_epi8_mask(x[t], x[t]);
        }
        for (int r = 0; r < NR; r++)
            wv[r] = _mm512_loadu_si512((const void*)(w + (size_t)r * ldw + k));
        for (int t = 0; t < NT; t++)
            for (int r = 0; r < NR; r++) {
                /* no vpsignb at 512 bits: negate w where x < 0, zero where x == 0 */
                __m512i sw = _mm512_mask_sub_epi8(wv[r], neg[t], _mm512_setzero_si512(), wv[r]);
                sw = _mm512_maskz_mov_epi8(nz[t], sw);
                acc[t][r] = _mm512_dpbusd_epi32(acc[t][r], ax[t], sw);
            }
    }
    int32_t tail[8] = { 0 };
    if (k < K)
        i8_tile_avx2(a + k, lda, NT, w + k, ldw, NR, K - k, tail);
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < NR; r++)
            out[t * 4 + r] = _mm512_reduce_add_epi32(acc[t][r]) + tail[t * 4 + r];
}

TGT_AVX512 static void i8_tile_avx512(const int8_t* a, int lda, int nt,
                                      const int8_t* w, int ldw, int nr, int K,
                                      int32_t* out) {
    if (nt == 2 && nr == 4) i8_tile_avx512_n(a, lda, nt, w, ldw, nr, K, out, 2, 4);
    else if (nt == 1 && nr == 4) i8_tile_avx512_n(a, lda, nt, w, ldw, nr, K, out, 1, 4);
    else i8_tile_avx512_n(a, lda, nt, w, ldw, nr, K, out, nt, nr);
}
#endif
#endif /* VV_BN_X86 */

#ifdef VV_BN_NEON
static VV_AI void i8_tile_neon_n(const int8_t* a, int lda, int nt_unused, const int8_t* w,
                         int ldw, int nr_unused, int K, int32_t* out, const int NT, const int NR) {
    (void)nt_unused;
    (void)nr_unused;
    int32x4_t acc[2][4];
    for (int t = 0; t < 2; t++)
        for (int r = 0; r < 4; r++) acc[t][r] = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        int8x16_t x[2], wv[4];
        for (int t = 0; t < NT; t++) x[t] = vld1q_s8(a + (size_t)t * lda + k);
        for (int r = 0; r < NR; r++) wv[r] = vld1q_s8(w + (size_t)r * ldw + k);
        for (int t = 0; t < NT; t++)
            for (int r = 0; r < NR; r++) {
#if defined(__ARM_FEATURE_DOTPROD)
                acc[t][r] = vdotq_s32(acc[t][r], x[t], wv[r]);
#else
                /* one s8*s8 product fits int16, a pair does not: widen each */
                acc[t][r] = vpadalq_s16(acc[t][r], vmull_s8(vget_low_s8(x[t]), vget_low_s8(wv[r])));
                acc[t][r] = vpadalq_s16(acc[t][r], vmull_high_s8(x[t], wv[r]));
#endif
            }
    }
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < NR; r++) {
            int32_t s = vaddvq_s32(acc[t][r]);
            for (int kk = k; kk < K; kk++)
                s += (int32_t)a[(size_t)t * lda + kk] * w[(size_t)r * ldw + kk];
            out[t * 4 + r] = s;
        }
}

static void i8_tile_neon(const int8_t* a, int lda, int nt, const int8_t* w,
                         int ldw, int nr, int K, int32_t* out) {
    if (nt == 2 && nr == 4) i8_tile_neon_n(a, lda, nt, w, ldw, nr, K, out, 2, 4);
    else if (nt == 1 && nr == 4) i8_tile_neon_n(a, lda, nt, w, ldw, nr, K, out, 1, 4);
    else i8_tile_neon_n(a, lda, nt, w, ldw, nr, K, out, nt, nr);
}
#endif

static i8_tile_fn pick_i8(void) {
    switch (isa()) {
#ifdef VV_BN_X86
#ifdef VV_BN_HAVE_AVX512VNNI
        case ISA_AVX512VNNI: return i8_tile_avx512;
#endif
#ifdef VV_BN_HAVE_AVXVNNI
        case ISA_AVXVNNI: return i8_tile_avxvnni;
#endif
        case ISA_AVX2: return i8_tile_avx2;
#endif
#ifdef VV_BN_NEON
        case ISA_NEON: return i8_tile_neon;
#endif
        default: return i8_tile_scalar;
    }
}

/*
 * acc/y are [M, N]. Threads split N in groups of 4 rows; for small N and
 * large M (the VAE's first stages: N = 128, M = 264 000) they split M.
 */
static vv_status_t i8_drive(const int8_t* a, const int8_t* w, int M, int N,
                            int K, int32_t* acc_out, float d,
                            const float* bias, float* y, float* absmax) {
    if (!a || !w) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    const i8_tile_fn fn = pick_i8();
    const int ngroups = (N + 3) / 4;
    const int mtiles = (M + 1) / 2;
    const int64_t work = (int64_t)ngroups * mtiles;

    /* per-thread max, combined below: `reduction(max:)` is OpenMP 3.1 and
     * MSVC only has 2.0 */
    float tmax[256];
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads > 256) nthreads = 256;
#endif
    for (int t = 0; t < nthreads; t++) tmax[t] = 0.0f;
    const int par = (int64_t)M * N * K > 200000;

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) if (par)
#endif
    {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    float amax = 0.0f;
    int64_t it;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (it = 0; it < work; it++) {
        /* M-major when the weight matrix is small enough to stay in cache,
         * N-major otherwise: either way a thread owns a contiguous range */
        int g, mt;
        if (N <= 512) { mt = (int)(it / ngroups); g = (int)(it % ngroups); }
        else          { g = (int)(it / mtiles);   mt = (int)(it % mtiles); }
        const int n0 = g * 4, m0 = mt * 2;
        const int nr = N - n0 < 4 ? N - n0 : 4;
        const int nt = M - m0 < 2 ? M - m0 : 2;
        int32_t o[8];
        fn(a + (size_t)m0 * K, K, nt, w + (size_t)n0 * K, K, nr, K, o);
        for (int t = 0; t < nt; t++)
            for (int r = 0; r < nr; r++) {
                const size_t idx = (size_t)(m0 + t) * N + n0 + r;
                if (acc_out) acc_out[idx] = o[t * 4 + r];
                if (y) {
                    float v = (float)o[t * 4 + r] * d;
                    if (bias) v += bias[n0 + r];
                    y[idx] = v;
                    const float av = fabsf(v);
                    if (av > amax) amax = av;
                }
            }
    }
    tmax[tid] = amax;
    }
    if (absmax) {
        float m = 0.0f;
        for (int t = 0; t < nthreads; t++) m = tmax[t] > m ? tmax[t] : m;
        *absmax = m;
    }
    (void)par;
    return VV_OK;
}

vv_status_t vv_i8_gemm_i32_cpu(const int8_t* a, const int8_t* w, int32_t* acc,
                               int M, int N, int K) {
    if (!acc) return VV_ERR_NULL_PTR;
    return i8_drive(a, w, M, N, K, acc, 0.0f, NULL, NULL, NULL);
}

vv_status_t vv_i8_linear_cpu(const int8_t* a, float a_scale, const int8_t* w,
                             float w_scale, const float* bias, float* y,
                             float* absmax, int M, int N, int K) {
    if (!y) return VV_ERR_NULL_PTR;
    if (!(a_scale > 0.0f)) return VV_ERR_INVALID_ARG;
    /* same expression as the reference: combined = w_scale / inp_scale */
    const float d = w_scale / a_scale;
    return i8_drive(a, w, M, N, K, NULL, d, bias, y, absmax);
}

vv_status_t vv_i8s_requant_cpu(const float* y, int64_t n, float absmax,
                               int relu, int8_t* q, float* out_scale) {
    if (!y || !q) return VV_ERR_NULL_PTR;
    const float inv = absmax != 0.0f ? 127.0f / absmax : 0.0f;
    const float lo = relu ? 0.0f : -127.0f;
    int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (i = 0; i < n; i++) {
        float v = y[i] * inv;
        v = v < lo ? lo : (v > 127.0f ? 127.0f : v);
        q[i] = (int8_t)nearbyintf(v);
    }
    if (out_scale) *out_scale = inv;
    return VV_OK;
}

/* ─── Quantized LM head ─────────────────────────────────────────────────── */

vv_status_t vv_i8_head_argmax_cpu(const int8_t* q, float scale, const int8_t* w,
                                  const float* w_scale, int V, int K,
                                  int32_t* token, float* value) {
    if (!q || !w || !w_scale || !token) return VV_ERR_NULL_PTR;
    if (V <= 0 || K <= 0 || !(scale > 0.0f)) return VV_ERR_INVALID_ARG;
    const i8_tile_fn fn = pick_i8();
    const int ngroups = (V + 3) / 4;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    /* one (value, index) per thread, merged in thread order so the result
     * does not depend on scheduling: ties resolve to the lowest index */
    float best_v[256];
    int32_t best_i[256];
    if (nthreads > 256) nthreads = 256;
    for (int t = 0; t < nthreads; t++) { best_v[t] = -FLT_MAX; best_i[t] = INT32_MAX; }

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float bv = -FLT_MAX;
        int32_t bi = INT32_MAX;
        int g;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (g = 0; g < ngroups; g++) {
            const int n0 = g * 4;
            const int nr = V - n0 < 4 ? V - n0 : 4;
            int32_t o[8];
            fn(q, K, 1, w + (size_t)n0 * K, K, nr, K, o);
            for (int r = 0; r < nr; r++) {
                const float v = (float)o[r] * w_scale[n0 + r] / scale;
                if (v > bv || (v == bv && n0 + r < bi)) { bv = v; bi = n0 + r; }
            }
        }
        best_v[tid] = bv;
        best_i[tid] = bi;
    }
    float bv = -FLT_MAX;
    int32_t bi = 0;
    int have = 0;
    for (int t = 0; t < nthreads; t++) {
        if (best_i[t] == INT32_MAX) continue;
        if (!have || best_v[t] > bv || (best_v[t] == bv && best_i[t] < bi)) {
            bv = best_v[t];
            bi = best_i[t];
            have = 1;
        }
    }
    *token = bi;
    if (value) *value = bv;
    return VV_OK;
}
