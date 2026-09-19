/**
 * @file vae_i8.c
 * @brief VibeASR.cpp's int8 speech encoder, on the CPU. See vae_i8.h.
 *
 * Every op of the reference (ggml fork, `ggml.c`: mul_mat_add[_relu],
 * rms_norm_scaled, add_scaled, im2col_asym) is two passes here, as it is
 * there: compute the FP32 result and its max |y| over the whole tensor,
 * then requantize with `id = 127 / max|y|`:
 *
 *   linear / conv   y = fma((float)acc, w_scale / x_scale, bias)
 *   rms_norm        y = ((float)x * (1 / sqrtf(sumsq / C + eps * s^2))) * g
 *   add_scaled      y = fma((float)b, 1 / s_b, ((float)a * (1 / s_a)) * g)
 *   requant         q = rne(clamp(y * id, relu ? 0 : -127, 127))
 *
 * acc is exact in int32, and the float expressions are what the reference's
 * AVX2 kernels compile to -- with the fused multiply-adds GCC makes of them
 * at -march=native (see I8_DEF_EPI). add_scaled also repeats the
 * reference's dependence on its thread partition: the scalar head and tail
 * of each thread's range divide by the scales and round ties away from zero.
 * With the same thread count as the reference's -t, every int8 tensor, every
 * scale and so the features equal VibeASR.cpp's bit for bit (checked on
 * jfk, test30 and test120 at 12 threads).
 *
 * The first pass keeps the FP32 results when they fit a scratch budget and
 * recomputes them otherwise, so memory stays bounded by the int8 tensors.
 */

#include "vibevoice/vae_i8.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/cpu_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define VV_I8_X86 1
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__) && defined(VV_I8_X86)
#include <intrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define I8_TGT_AVX2 __attribute__((target("avx2")))
#define I8_TGT_FMA  __attribute__((target("avx2,fma")))
#else
#define I8_TGT_AVX2
#define I8_TGT_FMA
#endif

/* ─── Ownership ─────────────────────────────────────────────────────────── */

vv_status_t vv_i8vae_create(vv_i8vae_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    vv_i8vae_t* v = (vv_i8vae_t*)vv_alloc(sizeof(*v));
    if (!v) return VV_ERR_OUT_OF_MEMORY;
    memset(v, 0, sizeof(*v));
    v->block_eps = 1e-5f;
    v->conn_eps = 1e-5f;
    *out = v;
    return VV_OK;
}

void* vv_i8vae_own(vv_i8vae_t* v, size_t bytes) {
    if (!v) return NULL;
    if (v->n_owned == v->cap_owned) {
        const int cap = v->cap_owned ? v->cap_owned * 2 : 256;
        void** n = (void**)vv_realloc(v->owned, (size_t)cap * sizeof(void*));
        if (!n) return NULL;
        v->owned = n;
        v->cap_owned = cap;
    }
    void* p = vv_alloc(bytes ? bytes : 1);
    if (!p) return NULL;
    v->owned[v->n_owned++] = p;
    v->bytes += bytes;
    return p;
}

void vv_i8vae_free(vv_i8vae_t* v) {
    if (!v) return;
    for (int i = 0; i < v->n_owned; i++) vv_free(v->owned[i]);
    vv_free(v->owned);
    vv_free(v);
}

/* ─── Integer GEMM ──────────────────────────────────────────────────────── */

/*
 * Packed weights for the AVX2 kernel: panels of 8 output channels, and in a
 * panel the K dimension in quads, [N/8][K/4][8][4]. A 32-byte load is then
 * 8 channels x 4 consecutive k, which pairs with 4 bytes of an activation
 * row broadcast to every lane.
 */
static size_t packed_bytes(int N, int K) {
    return (size_t)((N + 7) / 8) * 8 * (size_t)((K + 3) / 4) * 4;
}

static void pack_weights(const int8_t* w, int N, int K, int8_t* p) {
    const int np = (N + 7) / 8, kq = (K + 3) / 4;
    for (int pi = 0; pi < np; pi++)
        for (int q = 0; q < kq; q++)
            for (int l = 0; l < 8; l++)
                for (int b = 0; b < 4; b++) {
                    const int n = pi * 8 + l, k = q * 4 + b;
                    p[(((size_t)pi * kq + q) * 8 + l) * 4 + b] =
                        (n < N && k < K) ? w[(size_t)n * K + k] : 0;
                }
}

#ifdef VV_I8_X86
static int cpu_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return 0;
    __cpuidex(r, 7, 0);
    if (!(r[1] & (1 << 5))) return 0;
    __cpuid(r, 1);
    /* OSXSAVE + AVX, and the OS saves the YMM state */
    if ((r[2] & (1 << 27)) == 0 || (r[2] & (1 << 28)) == 0) return 0;
    return (_xgetbv(0) & 6) == 6;
#else
    return 0;
#endif
}

/*
 * 4 activation rows x 16 channels. The sign trick keeps maddubs exact:
 * |a| is the unsigned operand (|-128| = 128 is a valid u8) and a's sign
 * moves onto the weight, which stays in range because I8_S weights are
 * clamped to [-127, 127]. A pair is at most 2 * 128 * 127 = 32512, and madd
 * widens it to int32 at once.
 */
I8_TGT_AVX2 static void gemm_4x16_avx2(const int8_t* a, size_t lda, int mr,
                                        const int8_t* p0, const int8_t* p1,
                                        int kq, int32_t* out /* [4][16] */) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    const int8_t* a0 = a;
    const int8_t* a1 = mr > 1 ? a + lda : a;
    const int8_t* a2 = mr > 2 ? a + 2 * lda : a;
    const int8_t* a3 = mr > 3 ? a + 3 * lda : a;
    for (int q = 0; q < kq; q++) {
        const __m256i b0 = _mm256_loadu_si256((const __m256i*)(p0 + (size_t)q * 32));
        const __m256i b1 = _mm256_loadu_si256((const __m256i*)(p1 + (size_t)q * 32));
        int32_t s;
        __m256i av, aa;
#define I8_ROW(ap, c0, c1)                                                     \
        memcpy(&s, (ap) + (size_t)q * 4, 4);                                  \
        av = _mm256_set1_epi32(s);                                            \
        aa = _mm256_abs_epi8(av);                                             \
        c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(                          \
                 _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b0, av)), ones));  \
        c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(                          \
                 _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b1, av)), ones));
        I8_ROW(a0, c00, c01)
        I8_ROW(a1, c10, c11)
        I8_ROW(a2, c20, c21)
        I8_ROW(a3, c30, c31)
#undef I8_ROW
    }
    _mm256_storeu_si256((__m256i*)(out + 0), c00);
    _mm256_storeu_si256((__m256i*)(out + 8), c01);
    _mm256_storeu_si256((__m256i*)(out + 16), c10);
    _mm256_storeu_si256((__m256i*)(out + 24), c11);
    _mm256_storeu_si256((__m256i*)(out + 32), c20);
    _mm256_storeu_si256((__m256i*)(out + 40), c21);
    _mm256_storeu_si256((__m256i*)(out + 48), c30);
    _mm256_storeu_si256((__m256i*)(out + 56), c31);
}
#endif

/* Same tile from the packed layout, any ISA. */
static void gemm_4x16_scalar(const int8_t* a, size_t lda, int mr,
                             const int8_t* p0, const int8_t* p1, int kq,
                             int32_t* out) {
    for (int r = 0; r < 4; r++) {
        const int8_t* ar = a + (size_t)(r < mr ? r : 0) * lda;
        for (int h = 0; h < 2; h++) {
            const int8_t* p = h ? p1 : p0;
            int32_t acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int q = 0; q < kq; q++)
                for (int l = 0; l < 8; l++)
                    for (int b = 0; b < 4; b++)
                        acc[l] += (int32_t)ar[q * 4 + b] *
                                  (int32_t)p[((size_t)q * 8 + l) * 4 + b];
            for (int l = 0; l < 8; l++) out[r * 16 + h * 8 + l] = acc[l];
        }
    }
}

typedef void (*gemm_tile_fn)(const int8_t*, size_t, int, const int8_t*,
                             const int8_t*, int, int32_t*);

static gemm_tile_fn pick_gemm(void) {
#ifdef VV_I8_X86
    if (cpu_has_avx2()) return gemm_4x16_avx2;
#endif
    return gemm_4x16_scalar;
}

#ifdef VV_I8_X86
static int cpu_has_fma(void) {
    if (!cpu_has_avx2()) return 0;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 1);
    return (r[2] >> 12) & 1;
#else
    return 0;
#endif
}
#endif

/* ─── Requantization ────────────────────────────────────────────────────── */

/* round half to even for |f| < 2^22, as cvtps / rintf in the default mode */
static inline int rne(float f) {
    float v = f + 12582912.0f;
    int32_t i;
    memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}

static inline int8_t requant(float y, float id, float lo) {
    float v = y * id;
    v = v > lo ? v : lo;           /* max(v, lo) */
    v = v < 127.0f ? v : 127.0f;   /* min(v, 127) */
    return (int8_t)rne(v);
}

static inline float fabs_f(float x) { return x < 0.0f ? -x : x; }

/*
 * The reference is built with -march=native and GCC's default
 * -ffp-contract=fast, so its "multiply, then add" epilogues -- intrinsics
 * included, GCC contracts _mm256_add_ps(_mm256_mul_ps()) too -- run as
 * fused multiply-adds. Its disassembly (libggml.so) shows exactly these:
 *
 *   mul_mat_add (linear, conv, depthwise), every element:
 *       y = fma((float)acc, w_scale / x_scale, bias)
 *   add_scaled, 8-wide body:      y = fma((float)b, 1/s_b, ((float)a * (1/s_a)) * g)
 *   add_scaled, scalar head/tail: y = fma((float)a / s_a, g, (float)b / s_b)
 *
 * so they are written as fmaf here; this file itself is built with
 * -ffp-contract=off, and the FMA variants are compiled for avx2+fma so the
 * fmaf calls become instructions. Without FMA hardware the same functions
 * call libm's fmaf, which is exact too, only slower.
 */
typedef void (*epi_fn)(const int32_t* acc, int n, float d, const float* bias,
                       float* out);

#define I8_DEF_EPI(name, attr)                                                 \
    attr static void name(const int32_t* acc, int n, float d,                 \
                          const float* bias, float* out) {                    \
        for (int c = 0; c < n; c++) out[c] = fmaf((float)acc[c], d, bias[c]);  \
    }
I8_DEF_EPI(epi_generic, )
#ifdef VV_I8_X86
I8_DEF_EPI(epi_fma, I8_TGT_FMA)
#endif

/*
 * add_scaled over one thread's range [i0, i1) of the reference's partition
 * (ggml splits the flat tensor into n_threads equal ranges): the scalar head
 * up to the next row start, 8-wide chunks, a scalar tail -- each with its
 * own expression, see above. With id < 0 it returns the range's max |y|;
 * otherwise it requantizes into b in place, rounding half to even except in
 * the range's last (i1 - i0) % 8 elements, where the reference's scalar tail
 * uses roundf. C must be a multiple of 8, so no chunk crosses a row.
 */
#define I8_DEF_ADD(name, attr)                                                 \
    attr static float name(const int8_t* a, int8_t* b, const float* g, int C, \
                           int64_t i0, int64_t i1, float sa, float sb,        \
                           float id) {                                        \
        const float ia = 1.0f / sa, ib = 1.0f / sb;                           \
        int64_t hend = i0;                                                    \
        if (i0 % C) {                                                         \
            hend = i0 + (C - i0 % C);                                         \
            if (hend > i1) hend = i1;                                         \
        }                                                                     \
        const int64_t vend = hend + (i1 - hend) / 8 * 8;                      \
        const int64_t qend = i0 + (i1 - i0) / 8 * 8;                          \
        float m = 0.0f;                                                       \
        for (int64_t i = i0; i < i1; i++) {                                   \
            const float af = (float)a[i], bf = (float)b[i];                   \
            const float gg = g[i % C];                                        \
            const float v = (i >= hend && i < vend)                           \
                          ? fmaf(bf, ib, (af * ia) * gg)                       \
                          : fmaf(af / sa, gg, bf / sb);                       \
            if (id < 0.0f) {                                                  \
                const float av = fabs_f(v);                                   \
                if (av > m) m = av;                                           \
            } else {                                                          \
                float x = v * id;                                             \
                if (i < qend) {                                               \
                    x = x > -127.0f ? x : -127.0f;                            \
                    x = x < 127.0f ? x : 127.0f;                              \
                    b[i] = (int8_t)rne(x);                                    \
                } else {                                                      \
                    x = x < -127.0f ? -127.0f : (x > 127.0f ? 127.0f : x);    \
                    b[i] = (int8_t)roundf(x);                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
        return m;                                                             \
    }

typedef float (*add_fn)(const int8_t*, int8_t*, const float*, int, int64_t,
                        int64_t, float, float, float);
I8_DEF_ADD(add_generic, )

#ifdef VV_I8_X86
/*
 * The same with the 8-wide body in AVX2 -- the reference's own expression on
 * the same 8 elements, so the result is that of add_generic bit for bit.
 * Requantization walks [i0, qend) with round-half-even and the last
 * (i1 - i0) % 8 elements with roundf; those are computed first because the
 * requantization overwrites b in place.
 */
I8_TGT_FMA static inline float add_val(const int8_t* a, const int8_t* b,
                                       const float* g, int C, int64_t i,
                                       int64_t hend, int64_t vend, float sa,
                                       float sb, float ia, float ib) {
    const float af = (float)a[i], bf = (float)b[i], gg = g[i % C];
    return (i >= hend && i < vend) ? fmaf(bf, ib, (af * ia) * gg)
                                   : fmaf(af / sa, gg, bf / sb);
}

I8_TGT_FMA static inline __m256 add_val8(const int8_t* a, const int8_t* b,
                                         const float* g, __m256 via,
                                         __m256 vib) {
    const __m256 af = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i*)a)));
    const __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i*)b)));
    return _mm256_fmadd_ps(bf, vib,
                           _mm256_mul_ps(_mm256_mul_ps(af, via), _mm256_loadu_ps(g)));
}

I8_TGT_FMA static float add_fma(const int8_t* a, int8_t* b, const float* g,
                                int C, int64_t i0, int64_t i1, float sa,
                                float sb, float id) {
    const float ia = 1.0f / sa, ib = 1.0f / sb;
    const __m256 via = _mm256_set1_ps(ia), vib = _mm256_set1_ps(ib);
    int64_t hend = i0;
    if (i0 % C) {
        hend = i0 + (C - i0 % C);
        if (hend > i1) hend = i1;
    }
    const int64_t vend = hend + (i1 - hend) / 8 * 8;
    const int64_t qend = i0 + (i1 - i0) / 8 * 8;

    if (id < 0.0f) {
        float m = 0.0f;
        __m256 vm = _mm256_setzero_ps();
        const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        int64_t i = i0;
        for (; i < hend; i++) {
            const float v = fabs_f(add_val(a, b, g, C, i, hend, vend, sa, sb, ia, ib));
            if (v > m) m = v;
        }
        for (; i < vend; i += 8)
            vm = _mm256_max_ps(vm, _mm256_and_ps(absm,
                     add_val8(a + i, b + i, g + i % C, via, vib)));
        for (; i < i1; i++) {
            const float v = fabs_f(add_val(a, b, g, C, i, hend, vend, sa, sb, ia, ib));
            if (v > m) m = v;
        }
        float l[8];
        _mm256_storeu_ps(l, vm);
        for (int k = 0; k < 8; k++) if (l[k] > m) m = l[k];
        return m;
    }

    int8_t tail[8];
    for (int64_t i = qend; i < i1; i++) {
        float x = add_val(a, b, g, C, i, hend, vend, sa, sb, ia, ib) * id;
        x = x < -127.0f ? -127.0f : (x > 127.0f ? 127.0f : x);
        tail[i - qend] = (int8_t)roundf(x);
    }
    const __m256 vid = _mm256_set1_ps(id);
    const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    int64_t i = i0;
    const int64_t e1 = hend < qend ? hend : qend;
    for (; i < e1; i++) {
        float x = add_val(a, b, g, C, i, hend, vend, sa, sb, ia, ib) * id;
        x = x > -127.0f ? x : -127.0f;
        x = x < 127.0f ? x : 127.0f;
        b[i] = (int8_t)rne(x);
    }
    const int64_t e2 = vend < qend ? vend : qend;
    for (; i + 8 <= e2; i += 8) {
        __m256 x = _mm256_mul_ps(add_val8(a + i, b + i, g + i % C, via, vib), vid);
        x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
        const __m256i q32 = _mm256_cvtps_epi32(x);
        const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q32),
                                            _mm256_extracti128_si256(q32, 1));
        _mm_storel_epi64((__m128i*)(b + i), _mm_packs_epi16(q16, q16));
    }
    for (; i < qend; i++) {
        float x = add_val(a, b, g, C, i, hend, vend, sa, sb, ia, ib) * id;
        x = x > -127.0f ? x : -127.0f;
        x = x < 127.0f ? x : 127.0f;
        b[i] = (int8_t)rne(x);
    }
    for (i = qend; i < i1; i++) b[i] = tail[i - qend];
    return 0.0f;
}
#endif

/* ─── Row kernels: RMSNorm, requantization, max |y| ─────────────────────── */

/* One depthwise output row (all channels) from tap 0's input row xr. */
typedef void (*dw_fn)(const int8_t* wk, int k, const int8_t* xr, int C,
                      float d, const float* bias, float* out);

static void dw_generic(const int8_t* wk, int k, const int8_t* xr, int C,
                       float d, const float* bias, float* out) {
    for (int c = 0; c < C; c++) {
        int32_t acc = 0;
        for (int j = 0; j < k; j++)
            acc += (int32_t)wk[(size_t)j * C + c] * (int32_t)xr[(size_t)j * C + c];
        out[c] = fmaf((float)acc, d, bias[c]);
    }
}

/* One RMSNorm row: 1 / rms into *r, returns max |x * r * g|. Activations are
 * requantized to [-127, 127], so a squared pair fits int16 (the reference
 * sums with maddubs(|x|, |x|) too). */
typedef float (*rms_max_fn)(const int8_t* x, int C, const float* g, float es,
                            float* r);
typedef void (*rms_q_fn)(const int8_t* x, int C, const float* g, float r,
                         float id, int8_t* q);
/* q = rne(clamp(y * id, lo, 127)) */
typedef void (*quant_fn)(const float* y, int64_t n, float id, float lo,
                         int8_t* q);
typedef float (*absmax_fn)(const float* y, int64_t n);

static float rms_max_generic(const int8_t* x, int C, const float* g, float es,
                             float* r) {
    int32_t ss = 0;
    for (int c = 0; c < C; c++) ss += (int32_t)x[c] * (int32_t)x[c];
    const float rr = 1.0f / sqrtf((float)ss / (float)C + es);
    *r = rr;
    float m = 0.0f;
    for (int c = 0; c < C; c++) {
        const float a = fabs_f(((float)x[c] * rr) * g[c]);
        if (a > m) m = a;
    }
    return m;
}

static void rms_q_generic(const int8_t* x, int C, const float* g, float r,
                          float id, int8_t* q) {
    for (int c = 0; c < C; c++) q[c] = requant(((float)x[c] * r) * g[c], id, -127.0f);
}

static void quant_generic(const float* y, int64_t n, float id, float lo,
                          int8_t* q) {
    for (int64_t i = 0; i < n; i++) q[i] = requant(y[i], id, lo);
}

static float absmax_generic(const float* y, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        const float a = fabs_f(y[i]);
        if (a > m) m = a;
    }
    return m;
}

#ifdef VV_I8_X86
I8_TGT_AVX2 static inline __m256 cvt8(const int8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}

I8_TGT_AVX2 static inline float hmax8(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

I8_TGT_AVX2 static inline void store_q8(__m256 x, __m256 lo, __m256 hi, int8_t* q) {
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    const __m256i q32 = _mm256_cvtps_epi32(x);
    const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q32),
                                        _mm256_extracti128_si256(q32, 1));
    _mm_storel_epi64((__m128i*)q, _mm_packs_epi16(q16, q16));
}

I8_TGT_AVX2 static float rms_max_avx2(const int8_t* x, int C, const float* g,
                                      float es, float* r) {
    int c = 0;
    __m256i acc = _mm256_setzero_si256();
    for (; c + 32 <= C; c += 32) {
        const __m256i ax = _mm256_abs_epi8(_mm256_loadu_si256((const __m256i*)(x + c)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, ax),
                                                      _mm256_set1_epi16(1)));
    }
    int32_t l[8];
    _mm256_storeu_si256((__m256i*)l, acc);
    int32_t ss = l[0] + l[1] + l[2] + l[3] + l[4] + l[5] + l[6] + l[7];
    for (; c < C; c++) ss += (int32_t)x[c] * (int32_t)x[c];
    const float rr = 1.0f / sqrtf((float)ss / (float)C + es);
    *r = rr;
    const __m256 vr = _mm256_set1_ps(rr);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 vm = _mm256_setzero_ps();
    for (c = 0; c + 8 <= C; c += 8)
        vm = _mm256_max_ps(vm, _mm256_and_ps(absm, _mm256_mul_ps(
                 _mm256_mul_ps(cvt8(x + c), vr), _mm256_loadu_ps(g + c))));
    float m = hmax8(vm);
    for (; c < C; c++) {
        const float a = fabs_f(((float)x[c] * rr) * g[c]);
        if (a > m) m = a;
    }
    return m;
}

I8_TGT_AVX2 static void rms_q_avx2(const int8_t* x, int C, const float* g,
                                   float r, float id, int8_t* q) {
    const __m256 vr = _mm256_set1_ps(r), vid = _mm256_set1_ps(id);
    const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    int c = 0;
    for (; c + 8 <= C; c += 8)
        store_q8(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(cvt8(x + c), vr),
                                             _mm256_loadu_ps(g + c)), vid),
                 lo, hi, q + c);
    for (; c < C; c++) q[c] = requant(((float)x[c] * r) * g[c], id, -127.0f);
}

I8_TGT_AVX2 static void quant_avx2(const float* y, int64_t n, float id, float lo,
                                   int8_t* q) {
    const __m256 vid = _mm256_set1_ps(id), vlo = _mm256_set1_ps(lo),
                 hi = _mm256_set1_ps(127.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        store_q8(_mm256_mul_ps(_mm256_loadu_ps(y + i), vid), vlo, hi, q + i);
    for (; i < n; i++) q[i] = requant(y[i], id, lo);
}

/*
 * Depthwise causal conv of one output row, 16 channels at a time: taps in
 * int16 pairs (|w|, |x| <= 127, so two products fit), widened to int32,
 * then the reference's fused epilogue. xr points at tap 0's row.
 */
I8_TGT_FMA static void dw_row_avx2(const int8_t* wk, int k, const int8_t* xr,
                                   int C, float d, const float* bias,
                                   float* out) {
    const __m256 vd = _mm256_set1_ps(d);
    int c = 0;
    for (; c + 16 <= C; c += 16) {
        __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
        for (int j = 0; j < k; j += 2) {
            __m256i p = _mm256_mullo_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wk + (size_t)j * C + c))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xr + (size_t)j * C + c))));
            if (j + 1 < k)
                p = _mm256_add_epi16(p, _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wk + (size_t)(j + 1) * C + c))),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xr + (size_t)(j + 1) * C + c)))));
            lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p)));
            hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p, 1)));
        }
        _mm256_storeu_ps(out + c, _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), vd,
                                                  _mm256_loadu_ps(bias + c)));
        _mm256_storeu_ps(out + c + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), vd,
                                                      _mm256_loadu_ps(bias + c + 8)));
    }
    for (; c < C; c++) {
        int32_t acc = 0;
        for (int j = 0; j < k; j++)
            acc += (int32_t)wk[(size_t)j * C + c] * (int32_t)xr[(size_t)j * C + c];
        out[c] = fmaf((float)acc, d, bias[c]);
    }
}

I8_TGT_AVX2 static float absmax_avx2(const float* y, int64_t n) {
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 vm = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        vm = _mm256_max_ps(vm, _mm256_and_ps(absm, _mm256_loadu_ps(y + i)));
    float m = hmax8(vm);
    for (; i < n; i++) {
        const float a = fabs_f(y[i]);
        if (a > m) m = a;
    }
    return m;
}
#endif

/* ─── Activation tensors ────────────────────────────────────────────────── */

/* Zero rows in front of every activation buffer: the causal left padding of
 * the next convolution is read from there. */
#define I8_MARGIN 65536

typedef struct {
    int8_t* q;       /* base (after the margin)       */
    int64_t T;
    int     C;
    float   s;       /* multiplier: real = q / s     */
} act_t;

typedef struct {
    int8_t* buf[3];  /* margin + data each           */
    int8_t* hid;     /* FFN hidden, no margin         */
    float*  ybuf;    /* FP32 results of one op        */
    size_t  ybuf_n;  /* floats                        */
    float*  rowv;    /* one float per time step       */
    gemm_tile_fn gemm;
    epi_fn  epi;
    add_fn  add;
    dw_fn   dw;
    rms_max_fn rms_max;
    rms_q_fn   rms_q;
    quant_fn   quant;
    absmax_fn  absmax;
    int     nt;
    const char* dump;   /* VV_I8VAE_DUMP: every op's output, for diffs   */
    int*    dump_seq;
    int     dump_n;
    /* Keep a GEMM's FP32 results between the two passes only when its
     * contraction is long enough that recomputing costs more than writing
     * and reading them back; the depthwise conv likewise. */
    int     keep_k;
    int     keep_dw;
} work_t;

/* Write one op's int8 output and its multiplier (debugging aid). */
static void dump_act(const work_t* w, const char* what, const act_t* a) {
    if (!w->dump || !w->dump_seq) return;
    char fn[1024];
    const int i = (*w->dump_seq)++;
    snprintf(fn, sizeof(fn), "%s/o_%03d.bin", w->dump, i);
    FILE* f = fopen(fn, "wb");
    if (f) {
        fwrite(a->q, 1, (size_t)a->T * (size_t)a->C, f);
        fclose(f);
    }
    snprintf(fn, sizeof(fn), "%s/index.txt", w->dump);
    f = fopen(fn, i == 0 ? "w" : "a");
    if (f) {
        fprintf(f, "%d %s %d %lld %.9g\n", i, what, a->C, (long long)a->T,
                (double)a->s);
        fclose(f);
    }
}

static float reduce_max(const float* v, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

/*
 * One thread per physical core unless OMP_NUM_THREADS says otherwise: the
 * CPU path has already settled on that (vv_cpu_threads), a GPU run has not,
 * and SMT siblings only add barrier spinning to these two-pass ops. The
 * count also fixes add_scaled's partition, as the reference's -t does.
 */
static int n_threads(void) {
#ifdef _OPENMP
    int n = omp_get_max_threads();
    if (!getenv("OMP_NUM_THREADS")) {
        const int cores = vv_cpu_physical_cores();
        if (cores > 0 && cores < n) n = cores;
    }
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

static int thread_id(void) {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/* ─── Linear / convolution ──────────────────────────────────────────────── */

#define TILE_M 32
#define TILE_N 64

/*
 * y = acc * d + bias for rows [m0, m0+mb) x channels [n0, n0+nb), where
 * row t of the im2col matrix starts at x + t * stride * C - lp * C and is
 * k * C bytes long: the activation rows t*s - lp .. t*s - lp + k - 1, with
 * the weight reordered to [out][tap][in] so the product runs straight over
 * them. `emit` gets each value.
 */
typedef struct {
    const vv_i8_layer_t* L;
    const int8_t* x;
    int64_t lda;       /* bytes between im2col rows      */
    int64_t off;       /* bytes from row 0 to window 0   */
    int     K;
    float   d;
} gemm_job_t;

/* The tile [m0, m0+mb) x [n0, n0+nb) into out[(m - m0) * ldo + (n - n0)]. */
static void gemm_tile(const work_t* w, const gemm_job_t* j, int64_t m0, int mb,
                      int n0, int nb, float* out, int64_t ldo) {
    const vv_i8_layer_t* L = j->L;
    const int kq = j->K / 4;
    const int8_t* base = j->x + j->off + m0 * j->lda;
    int32_t acc[64];
    for (int n = n0; n < n0 + nb; n += 16) {
        const int8_t* p0 = L->packed + (size_t)(n / 8) * kq * 32;
        const int8_t* p1 = (n + 8 < L->out_ch) ? p0 + (size_t)kq * 32 : p0;
        const int nn = (n0 + nb - n) < 16 ? (n0 + nb - n) : 16;
        for (int m = 0; m < mb; m += 4) {
            const int mr = (mb - m) < 4 ? (mb - m) : 4;
            w->gemm(base + (int64_t)m * j->lda, (size_t)j->lda, mr, p0, p1,
                    kq, acc);
            for (int r = 0; r < mr; r++)
                w->epi(acc + r * 16, nn, j->d, L->bias + n,
                       out + (int64_t)(m + r) * ldo + (n - n0));
        }
    }
}

/* Largest y the op would produce, and y itself when `keep`. */
static float gemm_pass1(const work_t* w, const gemm_job_t* j, int64_t M, int N,
                        float* ybuf, bool keep) {
    const int64_t tm = (M + TILE_M - 1) / TILE_M;
    const int tn = (N + TILE_N - 1) / TILE_N;
    const int64_t n_tiles = tm * tn;
    float tmax[256];
    const int nt = w->nt < 256 ? w->nt : 256;
    for (int i = 0; i < nt; i++) tmax[i] = 0.0f;
    int64_t ti;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float scratch[TILE_M * TILE_N];
        float lm = 0.0f;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (ti = 0; ti < n_tiles; ti++) {
            const int64_t m0 = (ti / tn) * TILE_M;
            const int n0 = (int)(ti % tn) * TILE_N;
            const int mb = (int)((M - m0) < TILE_M ? (M - m0) : TILE_M);
            const int nb = (N - n0) < TILE_N ? (N - n0) : TILE_N;
            float* o;
            int64_t ldo;
            if (keep) { o = ybuf + m0 * N + n0; ldo = N; }
            else      { o = scratch; ldo = TILE_N; }
            gemm_tile(w, j, m0, mb, n0, nb, o, ldo);
            for (int r = 0; r < mb; r++) {
                const float a = w->absmax(o + (int64_t)r * ldo, nb);
                if (a > lm) lm = a;
            }
        }
        tmax[thread_id() % nt] = lm > tmax[thread_id() % nt]
                               ? lm : tmax[thread_id() % nt];
    }
    return reduce_max(tmax, nt);
}

static void gemm_pass2(const work_t* w, const gemm_job_t* j, int64_t M, int N,
                       const float* ybuf, bool kept, float id, float lo,
                       int8_t* q) {
    if (kept) {
        const int64_t n = M * N;
        int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(w->nt)
#endif
        for (i = 0; i < n; i += 4096)
            w->quant(ybuf + i, n - i < 4096 ? n - i : 4096, id, lo, q + i);
        return;
    }
    const int64_t tm = (M + TILE_M - 1) / TILE_M;
    const int tn = (N + TILE_N - 1) / TILE_N;
    const int64_t n_tiles = tm * tn;
    int64_t ti;
#ifdef _OPENMP
#pragma omp parallel num_threads(w->nt)
#endif
    {
        float scratch[TILE_M * TILE_N];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (ti = 0; ti < n_tiles; ti++) {
            const int64_t m0 = (ti / tn) * TILE_M;
            const int n0 = (int)(ti % tn) * TILE_N;
            const int mb = (int)((M - m0) < TILE_M ? (M - m0) : TILE_M);
            const int nb = (N - n0) < TILE_N ? (N - n0) : TILE_N;
            gemm_tile(w, j, m0, mb, n0, nb, scratch, TILE_N);
            for (int r = 0; r < mb; r++)
                w->quant(scratch + (int64_t)r * TILE_N, nb, id, lo,
                         q + (m0 + r) * N + n0);
        }
    }
}

/* Output length of a causal conv with left padding k - stride, no right
 * padding: what ggml's im2col yields. */
static int64_t conv_out_len(int64_t T, int k, int s) {
    const int64_t lp = k - s;
    if (T + lp < k) return 0;
    return (T + lp - k) / s + 1;
}

/* A conv or linear layer (not depthwise, K % 4 == 0). */
static vv_status_t op_linear(const work_t* w, const vv_i8_layer_t* L,
                             const act_t* x, act_t* y, int8_t* ybase,
                             int relu) {
    const int lp = L->k - L->stride;
    const int64_t M = conv_out_len(x->T, L->k, L->stride);
    const int N = L->out_ch;
    gemm_job_t j;
    j.L = L;
    j.x = x->q;
    j.lda = (int64_t)L->stride * x->C;
    j.off = -(int64_t)lp * x->C;
    j.K = L->k * x->C;
    j.d = L->w_scale / x->s;
    y->q = ybase;
    y->T = M;
    y->C = N;
    if (M == 0) { y->s = 0.0f; return VV_OK; }
    const bool keep = j.K > w->keep_k && (size_t)M * (size_t)N <= w->ybuf_n;
    const float gmax = gemm_pass1(w, &j, M, N, w->ybuf, keep);
    const float id = gmax != 0.0f ? 127.0f / gmax : 0.0f;
    gemm_pass2(w, &j, M, N, w->ybuf, keep, id, relu ? 0.0f : -127.0f, y->q);
    y->s = id;
    return VV_OK;
}

/* The stem: one input channel, K = k (not a multiple of 4), scalar. */
static vv_status_t op_stem(const work_t* w, const vv_i8_layer_t* L,
                           const act_t* x, act_t* y, int8_t* ybase) {
    const int k = L->k, lp = k - L->stride, N = L->out_ch;
    const int64_t M = conv_out_len(x->T, k, L->stride);
    const float d = L->w_scale / x->s;
    if (N > 256) return VV_ERR_UNSUPPORTED;
    y->q = ybase;
    y->T = M;
    y->C = N;
    float tmax[256];
    const int nt = w->nt < 256 ? w->nt : 256;
    for (int i = 0; i < nt; i++) tmax[i] = 0.0f;
    int64_t t;
    /* pass 1: max only, the stem is cheap to redo */
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float lm = 0.0f;
        int32_t acc[256];
        float yr[256];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (t = 0; t < M; t++) {
            const int8_t* xr = x->q + t * L->stride - lp;
            for (int n = 0; n < N; n++) {
                const int8_t* wr = L->w + (size_t)n * k;
                int32_t a = 0;
                for (int i = 0; i < k; i++) a += (int32_t)wr[i] * xr[i];
                acc[n] = a;
            }
            w->epi(acc, N, d, L->bias, yr);
            for (int n = 0; n < N; n++) {
                const float a = fabs_f(yr[n]);
                if (a > lm) lm = a;
            }
        }
        if (lm > tmax[thread_id() % nt]) tmax[thread_id() % nt] = lm;
    }
    const float gmax = reduce_max(tmax, nt);
    const float id = gmax != 0.0f ? 127.0f / gmax : 0.0f;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        int32_t acc[256];
        float yr[256];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (t = 0; t < M; t++) {
            const int8_t* xr = x->q + t * L->stride - lp;
            int8_t* o = y->q + t * N;
            for (int n = 0; n < N; n++) {
                const int8_t* wr = L->w + (size_t)n * k;
                int32_t a = 0;
                for (int i = 0; i < k; i++) a += (int32_t)wr[i] * xr[i];
                acc[n] = a;
            }
            w->epi(acc, N, d, L->bias, yr);
            for (int n = 0; n < N; n++) o[n] = requant(yr[n], id, -127.0f);
        }
    }
    y->s = id;
    return VV_OK;
}

/* Depthwise causal conv, stride 1: weights [k][C]. */
static void dw_row(const work_t* w, const vv_i8_layer_t* L, const int8_t* x,
                   int C, int64_t t, float d, float* out) {
    w->dw(L->w, L->k, x + (t - (L->k - 1)) * C, C, d, L->bias, out);
}

static vv_status_t op_dwconv(const work_t* w, const vv_i8_layer_t* L,
                             const act_t* x, act_t* y, int8_t* ybase) {
    const int C = x->C;
    if (C > 2048) return VV_ERR_UNSUPPORTED;
    const int64_t M = x->T;
    const float d = L->w_scale / x->s;
    y->q = ybase;
    y->T = M;
    y->C = C;
    float tmax[256];
    const int nt = w->nt < 256 ? w->nt : 256;
    for (int i = 0; i < nt; i++) tmax[i] = 0.0f;
    const bool keep = w->keep_dw && (size_t)M * (size_t)C <= w->ybuf_n;
    int64_t t;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float row[2048];
        float lm = 0.0f;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (t = 0; t < M; t++) {
            float* o = keep ? w->ybuf + t * C : row;
            dw_row(w, L, x->q, C, t, d, o);
            const float a = w->absmax(o, C);
            if (a > lm) lm = a;
        }
        if (lm > tmax[thread_id() % nt]) tmax[thread_id() % nt] = lm;
    }
    const float gmax = reduce_max(tmax, nt);
    const float id = gmax != 0.0f ? 127.0f / gmax : 0.0f;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float row[2048];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (t = 0; t < M; t++) {
            const float* o = keep ? w->ybuf + t * C : row;
            if (!keep) dw_row(w, L, x->q, C, t, d, row);
            w->quant(o, C, id, -127.0f, y->q + t * C);
        }
    }
    y->s = id;
    return VV_OK;
}

/* ─── RMSNorm and scaled add ────────────────────────────────────────────── */

static vv_status_t op_rmsnorm(const work_t* w, const float* g, float eps,
                              const act_t* x, act_t* y, int8_t* ybase) {
    const int C = x->C;
    const int64_t M = x->T;
    /* eps * scale^2, in double, then stored as a float, as the reference */
    const float eps_scaled = (float)((double)eps * (double)x->s * (double)x->s);
    float* rinv = w->rowv;
    y->q = ybase;
    y->T = M;
    y->C = C;
    float tmax[256];
    const int nt = w->nt < 256 ? w->nt : 256;
    for (int i = 0; i < nt; i++) tmax[i] = 0.0f;
    int64_t t;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float lm = 0.0f;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (t = 0; t < M; t++) {
            const float a = w->rms_max(x->q + t * C, C, g, eps_scaled, &rinv[t]);
            if (a > lm) lm = a;
        }
        if (lm > tmax[thread_id() % nt]) tmax[thread_id() % nt] = lm;
    }
    const float gmax = reduce_max(tmax, nt);
    const float id = gmax != 0.0f ? 127.0f / gmax : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (t = 0; t < M; t++) {
        w->rms_q(x->q + t * C, C, g, rinv[t], id, y->q + t * C);
    }
    y->s = id;
    return VV_OK;
}

/* y = a / s_a * g + b / s_b, written over b (element-wise, so in place). */
static vv_status_t op_add_scaled(const work_t* w, const act_t* a,
                                 const float* g, act_t* b) {
    const int C = a->C;
    const int64_t n = a->T * (int64_t)C;
    if (a->T != b->T || a->C != b->C) return VV_ERR_SHAPE_MISMATCH;
    if (C % 8) return VV_ERR_UNSUPPORTED;
    /* The reference's partition: n_threads equal flat ranges. Its result
     * depends on it (see I8_DEF_ADD), so the range count is the thread
     * count this encoder runs with, as it is the reference's. */
    const int nr = w->nt < 256 ? (w->nt > 0 ? w->nt : 1) : 256;
    const int64_t dr = (n + nr - 1) / nr;
    float tmax[256];
    int k;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nr)
#endif
    for (k = 0; k < nr; k++) {
        const int64_t i0 = dr * k < n ? dr * k : n;
        const int64_t i1 = i0 + dr < n ? i0 + dr : n;
        tmax[k] = w->add(a->q, b->q, g, C, i0, i1, a->s, b->s, -1.0f);
    }
    const float gmax = reduce_max(tmax, nr);
    const float id = gmax != 0.0f ? 127.0f / gmax : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nr)
#endif
    for (k = 0; k < nr; k++) {
        const int64_t i0 = dr * k < n ? dr * k : n;
        const int64_t i1 = i0 + dr < n ? i0 + dr : n;
        w->add(a->q, b->q, g, C, i0, i1, a->s, b->s, id);
    }
    b->s = id;
    return VV_OK;
}

/* ─── Tower ─────────────────────────────────────────────────────────────── */

static int64_t tower_frames(const vv_i8_tower_t* tw, int64_t n) {
    int64_t T = conv_out_len(n, tw->ds[0].k, tw->ds[0].stride);
    for (int i = 1; i < tw->n_stages; i++)
        T = conv_out_len(T, tw->ds[i].k, tw->ds[i].stride);
    return conv_out_len(T, tw->head.k, 1);
}

int vv_i8vae_frames(const vv_i8vae_t* v, int64_t n_samples) {
    if (!v || n_samples <= 0) return 0;
    const int64_t a = tower_frames(&v->tower[0], n_samples);
    const int64_t s = tower_frames(&v->tower[1], n_samples);
    return (int)(a < s ? a : s);
}

/* Largest [T][C] of any activation, and of any FFN hidden. */
static void tower_sizes(const vv_i8_tower_t* tw, int64_t n, size_t* act,
                        size_t* hid) {
    size_t a = (size_t)n, h = 0;
    int64_t T = conv_out_len(n, tw->ds[0].k, tw->ds[0].stride);
    for (int i = 0; i < tw->n_stages; i++) {
        if (i > 0) T = conv_out_len(T, tw->ds[i].k, tw->ds[i].stride);
        const int C = tw->ds[i].out_ch;
        if ((size_t)T * C > a) a = (size_t)T * C;
        if (tw->depth[i] > 0) {
            const size_t hh = (size_t)T * tw->blocks[i][0].fc1.out_ch;
            if (hh > h) h = hh;
        }
    }
    const int64_t F = conv_out_len(T, tw->head.k, 1);
    if ((size_t)F * tw->hidden > a) a = (size_t)F * tw->hidden;
    if ((size_t)F * tw->cfc1.out_ch > a) a = (size_t)F * tw->cfc1.out_ch;
    *act = a;
    *hid = h;
}

static vv_status_t run_block(const work_t* w, const vv_i8_block_t* B,
                             float eps, act_t* x) {
    /* x in buf[0]; norm -> buf[1]; mixer -> buf[2]; add into buf[0] */
    act_t h, m, f1, f2;
    vv_status_t s = op_rmsnorm(w, B->norm, eps, x, &h, w->buf[1]);
    if (s == VV_OK) { dump_act(w, "norm", &h); s = op_dwconv(w, &B->mixer, &h, &m, w->buf[2]); }
    if (s == VV_OK) { dump_act(w, "mixer", &m); s = op_add_scaled(w, &m, B->gamma, x); }
    if (s == VV_OK) { dump_act(w, "add", x); s = op_rmsnorm(w, B->ffn_norm, eps, x, &h, w->buf[1]); }
    if (s == VV_OK) { dump_act(w, "ffn_norm", &h); s = op_linear(w, &B->fc1, &h, &f1, w->hid, 1); }
    if (s == VV_OK) { dump_act(w, "fc1", &f1); s = op_linear(w, &B->fc2, &f1, &f2, w->buf[2], 0); }
    if (s == VV_OK) { dump_act(w, "fc2", &f2); s = op_add_scaled(w, &f2, B->ffn_gamma, x); }
    if (s == VV_OK) dump_act(w, "add", x);
    return s;
}

static vv_status_t run_tower(const vv_i8vae_t* v, const vv_i8_tower_t* tw,
                             work_t* w, const float* audio, int64_t n,
                             act_t* out) {
    /* The audio itself: s = 127 / max(|a|, 1e-5), q = roundf(a * s). */
    float amax = 0.00001f;
    for (int64_t i = 0; i < n; i++) {
        const float a = fabs_f(audio[i]);
        if (a > amax) amax = a;
    }
    const float sc = 127.0f / amax;
    act_t x;
    x.q = w->buf[1];
    x.T = n;
    x.C = 1;
    x.s = sc;
    {
        int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(w->nt)
#endif
        for (i = 0; i < n; i++) {
            int q = (int)roundf(audio[i] * sc);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            x.q[i] = (int8_t)q;
        }
    }
    act_t y;
    dump_act(w, "audio", &x);
    vv_status_t s = op_stem(w, &tw->ds[0], &x, &y, w->buf[0]);
    if (s != VV_OK) return s;
    dump_act(w, "stem", &y);
    x = y;
    for (int i = 0; i < tw->n_stages && s == VV_OK; i++) {
        if (i > 0) {
            /* downsample buf[0] -> buf[1], then back to buf[0] */
            s = op_linear(w, &tw->ds[i], &x, &y, w->buf[1], 0);
            if (s != VV_OK) break;
            dump_act(w, "downsample", &y);
            memcpy(w->buf[0], y.q, (size_t)y.T * (size_t)y.C);
            x = y;
            x.q = w->buf[0];
        }
        for (int b = 0; b < tw->depth[i] && s == VV_OK; b++)
            s = run_block(w, &tw->blocks[i][b], v->block_eps, &x);
    }
    if (s != VV_OK) return s;
    act_t h, c1, cn, c2;
    s = op_linear(w, &tw->head, &x, &h, w->buf[1], 0);
    if (s == VV_OK) { dump_act(w, "head", &h); s = op_linear(w, &tw->cfc1, &h, &c1, w->buf[2], 0); }
    if (s == VV_OK) { dump_act(w, "cfc1", &c1); s = op_rmsnorm(w, tw->cnorm, v->conn_eps, &c1, &cn, w->buf[0]); }
    if (s == VV_OK) { dump_act(w, "cnorm", &cn); s = op_linear(w, &tw->cfc2, &cn, &c2, w->buf[1], 0); }
    if (s == VV_OK) { dump_act(w, "cfc2", &c2); *out = c2; }
    return s;
}

static void work_free(work_t* w) {
    for (int i = 0; i < 3; i++)
        if (w->buf[i]) vv_free(w->buf[i] - I8_MARGIN);
    vv_free(w->hid);
    vv_free(w->ybuf);
    vv_free(w->rowv);
    memset(w, 0, sizeof(*w));
}

/* FP32 scratch the first pass may keep; past it the op is recomputed. */
#define I8_YBUF_BYTES ((size_t)512 << 20)

static vv_status_t work_alloc(const vv_i8vae_t* v, int64_t n, work_t* w) {
    memset(w, 0, sizeof(*w));
    size_t act = 0, hid = 0;
    for (int t = 0; t < 2; t++) {
        size_t a, h;
        tower_sizes(&v->tower[t], n, &a, &h);
        if (a > act) act = a;
        if (h > hid) hid = h;
    }
    for (int i = 0; i < 3; i++) {
        int8_t* p = (int8_t*)vv_alloc(I8_MARGIN + act + 64);
        if (!p) { work_free(w); return VV_ERR_OUT_OF_MEMORY; }
        memset(p, 0, I8_MARGIN);
        w->buf[i] = p + I8_MARGIN;
    }
    w->hid = (int8_t*)vv_alloc(hid + 64);
    size_t yn = hid > act ? hid : act;
    if (yn * sizeof(float) > I8_YBUF_BYTES) yn = I8_YBUF_BYTES / sizeof(float);
    w->ybuf = (float*)vv_alloc(yn * sizeof(float));
    w->ybuf_n = yn;
    w->rowv = (float*)vv_alloc((size_t)(n + 1) * sizeof(float));
    if (!w->hid || !w->ybuf || !w->rowv) { work_free(w); return VV_ERR_OUT_OF_MEMORY; }
    w->gemm = pick_gemm();
    w->epi = epi_generic;
    w->add = add_generic;
    w->dw = dw_generic;
    w->rms_max = rms_max_generic;
    w->rms_q = rms_q_generic;
    w->quant = quant_generic;
    w->absmax = absmax_generic;
#ifdef VV_I8_X86
    if (cpu_has_avx2()) {
        w->rms_max = rms_max_avx2;
        w->rms_q = rms_q_avx2;
        w->quant = quant_avx2;
        w->absmax = absmax_avx2;
    }
    if (cpu_has_fma()) {
        w->epi = epi_fma;
        w->add = add_fma;
        w->dw = dw_row_avx2;
    }
#endif
    w->nt = v->n_threads > 0 ? v->n_threads : n_threads();
    {
        const char* e = getenv("VV_I8VAE_KEEP_K");
        w->keep_k = (e && e[0]) ? atoi(e) : 0;
        e = getenv("VV_I8VAE_KEEP_DW");
        w->keep_dw = (e && e[0]) ? atoi(e) : 1;
    }
    w->dump = getenv("VV_I8VAE_DUMP");
    if (w->dump && !w->dump[0]) w->dump = NULL;
    w->dump_seq = &w->dump_n;
    return VV_OK;
}

vv_status_t vv_i8vae_encode_tower(const vv_i8vae_t* v, int tower,
                                  const float* audio, int64_t n_samples,
                                  float* out, int* n_frames) {
    if (!v || !audio || !out || !n_frames) return VV_ERR_NULL_PTR;
    if (tower < 0 || tower > 1 || n_samples <= 0) return VV_ERR_INVALID_ARG;
    work_t w;
    vv_status_t s = work_alloc(v, n_samples, &w);
    if (s != VV_OK) return s;
    act_t y;
    s = run_tower(v, &v->tower[tower], &w, audio, n_samples, &y);
    if (s == VV_OK) {
        const float dq = 1.0f / y.s;
        const int64_t n = y.T * y.C;
        for (int64_t i = 0; i < n; i++) out[i] = (float)y.q[i] * dq;
        *n_frames = (int)y.T;
    }
    work_free(&w);
    return s;
}

vv_status_t vv_i8vae_encode(const vv_i8vae_t* v, const float* audio,
                            int64_t n_samples, int64_t window_samples,
                            float* out, int* n_frames) {
    if (!v || !audio || !out || !n_frames) return VV_ERR_NULL_PTR;
    if (n_samples <= 0) return VV_ERR_INVALID_ARG;
    const int hs = v->tower[0].hidden;
    if (v->tower[1].hidden != hs) return VV_ERR_SHAPE_MISMATCH;
    const int64_t win = (window_samples > 0 && window_samples < n_samples)
                      ? window_samples : n_samples;
    work_t w;
    vv_status_t s = work_alloc(v, win, &w);
    if (s != VV_OK) return s;

    int total = 0;
    for (int64_t off = 0; off < n_samples && s == VV_OK; off += win) {
        const int64_t len = (n_samples - off) < win ? (n_samples - off) : win;
        act_t ya, ys;
        float* dst = out + (size_t)total * hs;
        s = run_tower(v, &v->tower[0], &w, audio + off, len, &ya);
        if (s != VV_OK) break;
        const int fa = (int)ya.T;
        {
            const float dq = 1.0f / ya.s;
            const int64_t n = (int64_t)fa * hs;
            for (int64_t i = 0; i < n; i++) dst[i] = (float)ya.q[i] * dq;
        }
        s = run_tower(v, &v->tower[1], &w, audio + off, len, &ys);
        if (s != VV_OK) break;
        const int fs = (int)ys.T;
        const int f = fa < fs ? fa : fs;
        {
            const float dq = 1.0f / ys.s;
            const int64_t n = (int64_t)f * hs;
            for (int64_t i = 0; i < n; i++)
                dst[i] = dst[i] + (float)ys.q[i] * dq;
        }
        total += f;
    }
    work_free(&w);
    if (s == VV_OK) *n_frames = total;
    return s;
}

/* ─── Preparation ───────────────────────────────────────────────────────── */

static vv_status_t prep_layer(vv_i8vae_t* v, vv_i8_layer_t* L, int in_ch,
                              int want_out, const char* what) {
    if (!L->w || !L->bias) {
        VV_LOG_E("vae_i8: %s has no weights", what);
        return VV_ERR_WEIGHT_MISSING;
    }
    if (L->in_ch != in_ch || (want_out > 0 && L->out_ch != want_out) ||
        L->k < 1 || L->stride < 1 || L->stride > L->k) {
        VV_LOG_E("vae_i8: %s is %dx%d k%d s%d, expected %d inputs", what,
                 L->out_ch, L->in_ch, L->k, L->stride, in_ch);
        return VV_ERR_SHAPE_MISMATCH;
    }
    if (L->depthwise || (L->k * L->in_ch) % 4 != 0) return VV_OK;
    L->packed = (int8_t*)vv_i8vae_own(v, packed_bytes(L->out_ch, L->k * L->in_ch));
    if (!L->packed) return VV_ERR_OUT_OF_MEMORY;
    pack_weights(L->w, L->out_ch, L->k * L->in_ch, L->packed);
    return VV_OK;
}

vv_status_t vv_i8vae_prepare(vv_i8vae_t* v) {
    if (!v) return VV_ERR_NULL_PTR;
    for (int t = 0; t < 2; t++) {
        vv_i8_tower_t* tw = &v->tower[t];
        if (tw->n_stages < 1 || tw->n_stages > VV_I8VAE_MAX_STAGES)
            return VV_ERR_MODEL_FORMAT;
        vv_status_t s = prep_layer(v, &tw->ds[0], 1, 0, "stem");
        if (s != VV_OK) return s;
        int C = tw->ds[0].out_ch;
        for (int i = 0; i < tw->n_stages; i++) {
            if (i > 0) {
                s = prep_layer(v, &tw->ds[i], C, 0, "downsample");
                if (s != VV_OK) return s;
                C = tw->ds[i].out_ch;
            }
            if (C > 2048) return VV_ERR_UNSUPPORTED;
            for (int b = 0; b < tw->depth[i]; b++) {
                vv_i8_block_t* B = &tw->blocks[i][b];
                if (!B->norm || !B->gamma || !B->ffn_norm || !B->ffn_gamma)
                    return VV_ERR_WEIGHT_MISSING;
                if (!B->mixer.depthwise || B->mixer.stride != 1 ||
                    B->mixer.out_ch != C)
                    return VV_ERR_SHAPE_MISMATCH;
                s = prep_layer(v, &B->mixer, C, C, "mixer");
                if (s == VV_OK) s = prep_layer(v, &B->fc1, C, 0, "ffn.linear1");
                if (s == VV_OK) s = prep_layer(v, &B->fc2, B->fc1.out_ch, C,
                                               "ffn.linear2");
                if (s != VV_OK) return s;
            }
        }
        vv_status_t s2 = prep_layer(v, &tw->head, C, 0, "head");
        if (s2 == VV_OK) s2 = prep_layer(v, &tw->cfc1, tw->head.out_ch, 0,
                                         "connector.fc1");
        if (s2 == VV_OK) s2 = prep_layer(v, &tw->cfc2, tw->cfc1.out_ch,
                                         tw->cfc1.out_ch, "connector.fc2");
        if (s2 != VV_OK) return s2;
        if (!tw->cnorm) return VV_ERR_WEIGHT_MISSING;
        tw->vae_dim = tw->head.out_ch;
        tw->hidden = tw->cfc2.out_ch;
    }
    return VV_OK;
}
