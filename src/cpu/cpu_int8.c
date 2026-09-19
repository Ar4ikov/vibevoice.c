/**
 * @file cpu_int8.c
 * @brief Int8-activation linear layers on the CPU: W8A8 and W4A8.
 *
 * Activations are quantized per token (vv_quant_act_q8_cpu), and the weight
 * row is dotted against up to four activation rows at once in exact int32:
 *
 *   W8A8   y = sx * sw * sum_k w x                         (one scale per row)
 *   W4A8   y = sx * sum_g s_g * (sum_k q x - z_g * sum_k x)  (per group)
 *
 * Instruction sets, chosen once at runtime:
 *   AVX-VNNI / AVX512-VNNI  vpdpbusd: u8 x s8 -> s32 in one instruction
 *   AVX2                    vpmaddubsw + vpmaddwd
 *   NEON dotprod            sdot (compiled in when __ARM_FEATURE_DOTPROD)
 *   scalar                  everything else
 *
 * Both x86 forms multiply an unsigned byte by a signed one. The W4 codes are
 * already unsigned (0..15, the zero point is applied through the per-32 sums
 * of x). For W8 the "sign trick" moves the weight's sign onto the activation:
 * w * x = |w| * (sign(w) x). Neither side is ever -128 (both quantizers clamp
 * to +-127), so a vpmaddubsw pair is at most 2 * 127 * 127 = 32258 and never
 * saturates.
 *
 * The W4A8 path wants each 32-column run of the activations as its 16 even
 * columns followed by its 16 odd ones (VV_Q8_NIBBLE): that is how the packed
 * weight splits into high nibbles and low nibbles, so a run of 16 weight
 * bytes becomes one 32-byte vector with no shuffle in the inner loop.
 */

#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define VV_I8_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VV_TGT(x) __attribute__((target(x)))
#else
#define VV_TGT(x)
#endif

/* vpdpbusd needs GCC 11 / clang 12 for the AVX-VNNI spelling. */
#if defined(VV_I8_X86) && ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || \
                           (defined(__clang__) && __clang_major__ >= 12))
#define VV_I8_VNNI 1
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define VV_I8_SDOT 1
#include <arm_neon.h>
#endif

/* ─── ISA selection ─────────────────────────────────────────────────────── */

typedef enum {
    I8_SCALAR = 0, I8_AVX2, I8_AVXVNNI, I8_AVX512VNNI, I8_SDOT
} i8_isa_t;

#ifdef VV_I8_X86
static void i8_cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
    int t[4];
    __cpuidex(t, leaf, sub);
    r[0] = (unsigned)t[0]; r[1] = (unsigned)t[1];
    r[2] = (unsigned)t[2]; r[3] = (unsigned)t[3];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static unsigned long long i8_xgetbv0(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#endif
}
#endif

static i8_isa_t i8_detect(void) {
    const char* e = getenv("VV_CPU_I8");
    const int force_scalar = e && strcmp(e, "scalar") == 0;
    const int no_vnni = e && strcmp(e, "avx2") == 0;
    if (force_scalar) return I8_SCALAR;
#ifdef VV_I8_X86
    unsigned r[4];
    i8_cpuid(0, 0, r);
    const unsigned maxleaf = r[0];
    if (maxleaf < 7) return I8_SCALAR;
    i8_cpuid(1, 0, r);
    const int osxsave = (r[2] >> 27) & 1;
    if (!osxsave) return I8_SCALAR;
    const unsigned long long xcr0 = i8_xgetbv0();
    if ((xcr0 & 6) != 6) return I8_SCALAR;              /* XMM + YMM state */
    i8_cpuid(7, 0, r);
    const int avx2 = (r[1] >> 5) & 1;
    const int avx512f = (r[1] >> 16) & 1, avx512vl = (r[1] >> 31) & 1;
    const int avx512vnni = (r[2] >> 11) & 1;
    i8_cpuid(7, 1, r);
    const int avxvnni = (r[0] >> 4) & 1;
    if (!avx2) return I8_SCALAR;
#ifdef VV_I8_VNNI
    if (!no_vnni) {
        if (avxvnni) return I8_AVXVNNI;
        if (avx512f && avx512vl && avx512vnni && (xcr0 & 0xE0) == 0xE0)
            return I8_AVX512VNNI;
    }
#else
    (void)avx512f; (void)avx512vl; (void)avx512vnni; (void)avxvnni;
    (void)no_vnni;
#endif
    return I8_AVX2;
#elif defined(VV_I8_SDOT)
    (void)no_vnni;
    return I8_SDOT;
#else
    (void)no_vnni;
    return I8_SCALAR;
#endif
}

static i8_isa_t i8_isa(void) {
    static int cached = -1;              /* benign race: same value */
    if (cached < 0) cached = (int)i8_detect();
    return (i8_isa_t)cached;
}

const char* vv_cpu_int8_isa(void) {
    switch (i8_isa()) {
        case I8_AVX2:       return "AVX2";
        case I8_AVXVNNI:    return "AVX-VNNI";
        case I8_AVX512VNNI: return "AVX512-VNNI";
        case I8_SDOT:       return "NEON-dotprod";
        default:            return "scalar";
    }
}

/* ─── Activation quantizer ──────────────────────────────────────────────── */

static void quant_row(const float* x, int K, int layout, int8_t* q,
                      float* sx, int32_t* xsum) {
    float amax = 0.0f;
    for (int k = 0; k < K; k++) {
        const float a = fabsf(x[k]);
        amax = a > amax ? a : amax;
    }
    const float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    *sx = amax / 127.0f;
    int pos[32];
    for (int j = 0; j < 32; j++) pos[j] = vv_q8_pos(layout, j);
    for (int c = 0; c < K / 32; c++) {
        int s = 0;
        for (int j = 0; j < 32; j++) {
            float v = rintf(x[c * 32 + j] * inv);
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            const int qi = (int)v;
            s += qi;
            q[c * 32 + pos[j]] = (int8_t)qi;
        }
        if (xsum) xsum[c] = s;
    }
}

vv_status_t vv_quant_act_q8_cpu(const float* x, int M, int K, int layout,
                                int8_t* xq, float* sx, int32_t* xsum) {
    if (!x || !xq || !sx) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || (K % 32) != 0) return VV_ERR_INVALID_ARG;
    if (layout < 0 || layout >= VV_Q8_LAYOUT_COUNT) return VV_ERR_INVALID_ARG;
    int m;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M > 4)
#endif
    for (m = 0; m < M; m++)
        quant_row(x + (size_t)m * K, K, layout, xq + (size_t)m * K, sx + m,
                  xsum ? xsum + (size_t)m * (K / 32) : NULL);
    return VV_OK;
}

/* ─── Dot kernels: one weight row against NR <= 4 activation rows ───────── */

/*
 * W8: out[r] = sum_k w[k] * x[r][k] as int32. W4: out[r] = sum over groups of
 * s_g * (sum_k q x - z_g * sum_k x), as float; the caller multiplies by sx.
 */

static void w8_dot_scalar(const int8_t* w, const int8_t* const* x, int nr,
                          int K, int32_t* out) {
    for (int r = 0; r < nr; r++) {
        int32_t a = 0;
        for (int k = 0; k < K; k++) a += (int32_t)w[k] * (int32_t)x[r][k];
        out[r] = a;
    }
}

static void w4_dot_scalar(const uint8_t* wp, const uint16_t* srow,
                          const uint8_t* zrow, int G,
                          const int8_t* const* x, const int32_t* const* gs,
                          int nr, int K, float* out) {
    for (int r = 0; r < nr; r++) {
        float acc = 0.0f;
        for (int g0 = 0; g0 < K; g0 += G) {
            int32_t dot = 0;
            for (int c = g0; c < g0 + G; c += 32) {
                const uint8_t* b = wp + c / 2;
                const int8_t* xv = x[r] + c;
                for (int j = 0; j < 16; j++) {
                    dot += (int32_t)(b[j] >> 4) * xv[j];        /* even k */
                    dot += (int32_t)(b[j] & 15) * xv[16 + j];   /* odd k  */
                }
            }
            const float s = vv_half_to_float(srow[g0 / G]);
            acc += s * (float)(dot - (int32_t)zrow[g0 / G] * gs[r][g0 / G]);
        }
        out[r] = acc;
    }
}

#ifdef VV_I8_X86

static inline VV_TGT("avx2") int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

static inline VV_TGT("avx2") float hsum_f32(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/*
 * One generator for three instruction sets: DOTU8(acc, u, s) accumulates
 * u8 x s8 products of 32 bytes into eight int32 lanes.
 */
#define DOTU8_AVX2(acc, u, s) \
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), ones))
#define DOTU8_AVXVNNI(acc, u, s)    acc = _mm256_dpbusd_avx_epi32(acc, u, s)
#define DOTU8_AVX512VNNI(acc, u, s) acc = _mm256_dpbusd_epi32(acc, u, s)

/*
 * The W4 kernel accumulates a run of chunks in `h` before FLUSHing it into
 * the int32 group sum. On AVX2 `h` holds int16 pair sums: a 4-bit code
 * times an int8 activation pairs to at most 2 * 15 * 127 = 3810, so eight
 * chunks fit an int16 lane (30480) and seven of every eight vpmaddwd go
 * away. With VNNI `h` is already int32 and the flush is a plain add.
 */
#define ACC16_AVX2(h, u, s) h = _mm256_add_epi16(h, _mm256_maddubs_epi16(u, s))
#define FLUSH16_AVX2(a, h)                                                     \
    do { a = _mm256_add_epi32(a, _mm256_madd_epi16(h, ones));                  \
         h = _mm256_setzero_si256(); } while (0)
#define FLUSH32(a, h)                                                          \
    do { a = _mm256_add_epi32(a, h); h = _mm256_setzero_si256(); } while (0)

#define DEFINE_I8_KERNELS(SUF, TGT, DOTU8, ACC, FLUSH, ACC_RUN)                \
static VV_TGT(TGT) void w8_dot_##SUF(const int8_t* w,                          \
        const int8_t* const* x, int nr, int K, int32_t* out) {                 \
    const __m256i ones = _mm256_set1_epi16(1);                                 \
    (void)ones;                                                                \
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;           \
    int k = 0;                                                                 \
    for (; k + 32 <= K; k += 32) {                                             \
        const __m256i wv = _mm256_loadu_si256((const __m256i*)(w + k));       \
        const __m256i aw = _mm256_abs_epi8(wv);                                \
        DOTU8(a0, aw, _mm256_sign_epi8(                                        \
            _mm256_loadu_si256((const __m256i*)(x[0] + k)), wv));             \
        if (nr > 1) DOTU8(a1, aw, _mm256_sign_epi8(                            \
            _mm256_loadu_si256((const __m256i*)(x[1] + k)), wv));             \
        if (nr > 2) DOTU8(a2, aw, _mm256_sign_epi8(                            \
            _mm256_loadu_si256((const __m256i*)(x[2] + k)), wv));             \
        if (nr > 3) DOTU8(a3, aw, _mm256_sign_epi8(                            \
            _mm256_loadu_si256((const __m256i*)(x[3] + k)), wv));             \
    }                                                                          \
    int32_t t[4] = { hsum_i32(a0), hsum_i32(a1), hsum_i32(a2), hsum_i32(a3) }; \
    for (; k < K; k++)                                                         \
        for (int r = 0; r < nr; r++) t[r] += (int32_t)w[k] * x[r][k];          \
    for (int r = 0; r < nr; r++) out[r] = t[r];                                \
}                                                                              \
static VV_TGT(TGT) void w4_dot_##SUF(const uint8_t* wp,                       \
        const uint16_t* srow, const uint8_t* zrow, int G,                      \
        const int8_t* const* x, const int32_t* const* gs, int nr, int K,       \
        float* out) {                                                          \
    const __m256i ones = _mm256_set1_epi16(1);                                 \
    const __m128i m4 = _mm_set1_epi8(0x0F);                                    \
    (void)ones;                                                                \
    __m256 f0 = _mm256_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;               \
    float corr[4] = { 0.0f, 0.0f, 0.0f, 0.0f };                                \
    for (int g0 = 0, g = 0; g0 < K; g0 += G, g++) {                            \
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;        \
        __m256i h0 = a0, h1 = a0, h2 = a0, h3 = a0;                            \
        int pend = 0;                                                          \
        for (int c = g0; c < g0 + G; c += 32) {                                \
            const __m128i b = _mm_loadu_si128((const __m128i*)(wp + c / 2));   \
            const __m256i q = _mm256_set_m128i(_mm_and_si128(b, m4),           \
                _mm_and_si128(_mm_srli_epi16(b, 4), m4));                      \
            ACC(h0, q, _mm256_loadu_si256((const __m256i*)(x[0] + c)));        \
            if (nr > 1) { ACC(h1, q,                                           \
                _mm256_loadu_si256((const __m256i*)(x[1] + c)));               \
                }                                                              \
            if (nr > 2) { ACC(h2, q,                                           \
                _mm256_loadu_si256((const __m256i*)(x[2] + c)));               \
                }                                                              \
            if (nr > 3) { ACC(h3, q,                                           \
                _mm256_loadu_si256((const __m256i*)(x[3] + c)));               \
                }                                                              \
            if (++pend == ACC_RUN) {                                           \
                FLUSH(a0, h0); FLUSH(a1, h1); FLUSH(a2, h2); FLUSH(a3, h3);    \
                pend = 0;                                                      \
            }                                                                  \
        }                                                                      \
        if (pend) {                                                            \
            FLUSH(a0, h0); FLUSH(a1, h1); FLUSH(a2, h2); FLUSH(a3, h3);        \
        }                                                                      \
        const float s = vv_half_to_float(srow[g]);                             \
        const float zs = s * (float)zrow[g];                                   \
        const __m256 sv = _mm256_set1_ps(s);                                   \
        f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), sv, f0);                  \
        corr[0] += zs * (float)gs[0][g];                                       \
        if (nr > 1) { f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), sv, f1);   \
                      corr[1] += zs * (float)gs[1][g]; }                       \
        if (nr > 2) { f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a2), sv, f2);   \
                      corr[2] += zs * (float)gs[2][g]; }                       \
        if (nr > 3) { f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a3), sv, f3);   \
                      corr[3] += zs * (float)gs[3][g]; }                       \
    }                                                                          \
    const float t[4] = { hsum_f32(f0), hsum_f32(f1), hsum_f32(f2),             \
                         hsum_f32(f3) };                                       \
    for (int r = 0; r < nr; r++) out[r] = t[r] - corr[r];                      \
}

DEFINE_I8_KERNELS(avx2, "avx2,fma", DOTU8_AVX2, ACC16_AVX2, FLUSH16_AVX2, 8)
#ifdef VV_I8_VNNI
DEFINE_I8_KERNELS(avxvnni, "avx2,fma,avxvnni", DOTU8_AVXVNNI, DOTU8_AVXVNNI,
                  FLUSH32, 64)
DEFINE_I8_KERNELS(avx512vnni, "avx2,fma,avx512f,avx512vl,avx512vnni",
                  DOTU8_AVX512VNNI, DOTU8_AVX512VNNI, FLUSH32, 64)
#endif

#endif /* VV_I8_X86 */

#ifdef VV_I8_SDOT
static void w8_dot_sdot(const int8_t* w, const int8_t* const* x, int nr,
                        int K, int32_t* out) {
    int32x4_t a[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                       vdupq_n_s32(0) };
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const int8x16_t wv = vld1q_s8(w + k);
        for (int r = 0; r < nr; r++)
            a[r] = vdotq_s32(a[r], wv, vld1q_s8(x[r] + k));
    }
    for (int r = 0; r < nr; r++) {
        int32_t t = vaddvq_s32(a[r]);
        for (int kk = k; kk < K; kk++) t += (int32_t)w[kk] * x[r][kk];
        out[r] = t;
    }
}

static void w4_dot_sdot(const uint8_t* wp, const uint16_t* srow,
                        const uint8_t* zrow, int G, const int8_t* const* x,
                        const int32_t* const* gs, int nr, int K, float* out) {
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g0 = 0, g = 0; g0 < K; g0 += G, g++) {
        int32x4_t a[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                           vdupq_n_s32(0) };
        for (int c = g0; c < g0 + G; c += 32) {
            const uint8x16_t b = vld1q_u8(wp + c / 2);
            const int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
            const int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(b, m4));
            for (int r = 0; r < nr; r++) {
                a[r] = vdotq_s32(a[r], hi, vld1q_s8(x[r] + c));
                a[r] = vdotq_s32(a[r], lo, vld1q_s8(x[r] + c + 16));
            }
        }
        const float s = vv_half_to_float(srow[g]);
        for (int r = 0; r < nr; r++)
            acc[r] += s * (float)(vaddvq_s32(a[r]) - (int32_t)zrow[g] * gs[r][g]);
    }
    for (int r = 0; r < nr; r++) out[r] = acc[r];
}
#endif

typedef void (*w8_dot_fn)(const int8_t*, const int8_t* const*, int, int,
                          int32_t*);
typedef void (*w4_dot_fn)(const uint8_t*, const uint16_t*, const uint8_t*,
                          int, const int8_t* const*, const int32_t* const*,
                          int, int, float*);

static w8_dot_fn pick_w8(void) {
    switch (i8_isa()) {
#ifdef VV_I8_X86
    case I8_AVX2:       return w8_dot_avx2;
#ifdef VV_I8_VNNI
    case I8_AVXVNNI:    return w8_dot_avxvnni;
    case I8_AVX512VNNI: return w8_dot_avx512vnni;
#endif
#endif
#ifdef VV_I8_SDOT
    case I8_SDOT:       return w8_dot_sdot;
#endif
    default:            return w8_dot_scalar;
    }
}

static w4_dot_fn pick_w4(void) {
    switch (i8_isa()) {
#ifdef VV_I8_X86
    case I8_AVX2:       return w4_dot_avx2;
#ifdef VV_I8_VNNI
    case I8_AVXVNNI:    return w4_dot_avxvnni;
    case I8_AVX512VNNI: return w4_dot_avx512vnni;
#endif
#endif
#ifdef VV_I8_SDOT
    case I8_SDOT:       return w4_dot_sdot;
#endif
    default:            return w4_dot_scalar;
    }
}

/* ─── GEMM drivers ──────────────────────────────────────────────────────── */

/*
 * Parallel over blocks of NB output rows. Inside a block the activation rows
 * go four at a time: those four rows (4 x K bytes) stay in L1 while the block's
 * weight rows stream past them from L2, so a weight row is fetched from DRAM
 * once per call and the four-row dot shares each weight load and unpack.
 */
#define I8_NB 16
/** Groups per row the W4A8 driver sums on the stack (K / G). */
#define I8_MAX_GROUPS 1024

vv_status_t vv_w8a8_gemm_cpu(const int8_t* xq, const float* sx,
                             const int8_t* w, const float* sw,
                             const void* bias_f16, float* out,
                             int M, int N, int K) {
    if (!xq || !sx || !w || !sw || !out) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    const w8_dot_fn dot = pick_w8();
    const uint16_t* bias = (const uint16_t*)bias_f16;
    const int nblocks = (N + I8_NB - 1) / I8_NB;
    int nb;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (nb = 0; nb < nblocks; nb++) {
        const int n0 = nb * I8_NB;
        const int n1 = n0 + I8_NB < N ? n0 + I8_NB : N;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int nr = (M - m0) < 4 ? (M - m0) : 4;
            const int8_t* xr[4];
            for (int r = 0; r < 4; r++)
                xr[r] = xq + (size_t)(m0 + (r < nr ? r : 0)) * K;
            for (int n = n0; n < n1; n++) {
                int32_t t[4];
                dot(w + (size_t)n * K, xr, nr, K, t);
                const float b = bias ? vv_half_to_float(bias[n]) : 0.0f;
                for (int r = 0; r < nr; r++)
                    out[(size_t)(m0 + r) * N + n] =
                        (float)t[r] * sx[m0 + r] * sw[n] + b;
            }
        }
    }
    return VV_OK;
}

vv_status_t vv_w4a8_gemm_cpu(const int8_t* xq, const float* sx,
                             const int32_t* xsum, const uint8_t* packed,
                             const void* scales_f16, const uint8_t* zeros,
                             int G, const void* bias_f16, float* out,
                             int M, int N, int K) {
    if (!xq || !sx || !xsum || !packed || !scales_f16 || !zeros || !out)
        return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    if (G <= 0 || (G % 32) != 0 || (K % G) != 0) return VV_ERR_UNSUPPORTED;
    if (K / G > I8_MAX_GROUPS) return VV_ERR_UNSUPPORTED;
    const w4_dot_fn dot = pick_w4();
    const uint16_t* bias = (const uint16_t*)bias_f16;
    const uint16_t* sc = (const uint16_t*)scales_f16;
    const int ng = K / G, nc = K / 32;
    const int nblocks = (N + I8_NB - 1) / I8_NB;
    int nb;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (nb = 0; nb < nblocks; nb++) {
        const int n0 = nb * I8_NB;
        const int n1 = n0 + I8_NB < N ? n0 + I8_NB : N;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int nr = (M - m0) < 4 ? (M - m0) : 4;
            const int8_t* xr[4];
            /*
             * Sum of xq over each weight group, per activation row. The
             * zero-point term needs it once per group, the same for every
             * weight row, so it is summed here instead of in the dot.
             */
            int32_t gbuf[4][I8_MAX_GROUPS];
            const int32_t* gs[4];
            for (int r = 0; r < 4; r++) {
                const int m = m0 + (r < nr ? r : 0);
                xr[r] = xq + (size_t)m * K;
                const int32_t* xs = xsum + (size_t)m * nc;
                for (int g = 0; g < ng; g++) {
                    int32_t t = 0;
                    for (int c = 0; c < G / 32; c++) t += xs[g * (G / 32) + c];
                    gbuf[r][g] = t;
                }
                gs[r] = gbuf[r];
            }
            for (int n = n0; n < n1; n++) {
                float t[4];
                dot(packed + (size_t)n * (K / 2), sc + (size_t)n * ng,
                    zeros + (size_t)n * ng, G, xr, gs, nr, K, t);
                const float b = bias ? vv_half_to_float(bias[n]) : 0.0f;
                for (int r = 0; r < nr; r++)
                    out[(size_t)(m0 + r) * N + n] = t[r] * sx[m0 + r] + b;
            }
        }
    }
    return VV_OK;
}
