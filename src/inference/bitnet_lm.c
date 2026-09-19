/**
 * @file bitnet_lm.c
 * @brief One VibeVoice-ASR-BitNet transformer layer on the CPU, computed the
 *        way VibeASR.cpp's ggml graph computes it.
 *
 * The reference runs llama.cpp's qwen2 graph (build_qwen2 + llm_build_kqv,
 * flash attention off) on the CPU backend. What that graph does, and what
 * this file repeats:
 *
 *   norm      ggml_rms_norm then ggml_mul: sum of x*x in double,
 *             scale = 1 / sqrtf(mean + eps), y = (x * scale) * w
 *   q/k/v/o,  per-token int8 activations, ternary weights, exact int32,
 *   gate/up,  y = (float)acc / s * w_scale, then + bias (bitnet.h)
 *   down
 *   rope      NEOX, the ggml rope cache: theta *= powf(base, -2/128) per
 *             pair, cosf / sinf of it
 *   KV cache  F16: K and V are rounded to half on the way in
 *   KQ        Q rounded to F16 too, ggml_vec_dot_f16's order (4 x 8 FMA
 *             lanes, then its reduction)
 *   softmax   scale, causal -inf mask, ggml_v_expf (a polynomial, not expf)
 *             in chunks of 8 whose sums are added in double
 *   KQV       llamafile's tinyBLAS: F16 V against F32 probabilities, 8 FMA
 *             lanes over the positions, then its horizontal sum
 *   ffn       ggml_v_silu(gate) * up
 *
 * and where the reference's GCC build (-march=native, contraction on) fuses
 * a multiply-add in plain C -- the rope rotation -- this file uses fmaf in
 * the same place (read off libggml.so's disassembly). The ternary epilogue
 * has no multiply-add to fuse. The logits then match VibeASR.cpp's to the
 * printed digit on every token checked (docs/BITNET.md).
 *
 * The SIMD paths need AVX2 + FMA + F16C (what the reference was built with
 * on the same box); anything else runs the same operations lane by lane
 * with fmaf, bit-identical and slower.
 */

#include "vibevoice/inference.h"
#include "vibevoice/bitnet.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define VV_BL_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BL_TGT __attribute__((target("avx2,fma,f16c")))
#else
#define BL_TGT
#endif

/* ─── CPU features ──────────────────────────────────────────────────────── */

#ifdef VV_BL_X86
static int bl_simd(void) {
    unsigned a = 0, b = 0, c = 0, d = 0;
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuid(r, 1);
    a = (unsigned)r[0]; b = (unsigned)r[1]; c = (unsigned)r[2]; d = (unsigned)r[3];
#else
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
#endif
    const int fma = (c >> 12) & 1, osx = (c >> 27) & 1, avx = (c >> 28) & 1,
              f16c = (c >> 29) & 1;
    if (!(fma && osx && avx && f16c)) return 0;
#if defined(_MSC_VER) && !defined(__clang__)
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuidex(r, 7, 0);
    b = (unsigned)r[1];
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6) return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
#endif
    return (b >> 5) & 1;   /* AVX2 */
}
#else
static int bl_simd(void) { return 0; }
#endif

static inline float f16r(float x) {
    return vv_half_to_float(vv_float_to_half_rne(x));
}

static inline float asf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t asu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* ─── ggml_v_expf, one lane ─────────────────────────────────────────────── */

/* The AVX2 routine lane by lane: every lane's result depends only on that
 * lane (the slow-path branch picks per-lane masks), so this is exact. */
static float gexpf1(float x) {
    const float r = 0x1.8p23f;
    const float z = fmaf(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fmaf(-n, 0x1.7f7d1cp-20f, fmaf(-n, 0x1.62e4p-1f, x));
    const uint32_t e = asu(z) << 23;
    const float k = asf(e + asu(1.0f));
    const float an = fabsf(n);
    const float u = b * b;
    const float j = fmaf(fmaf(fmaf(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
                              fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                         u, 0x1.ffffecp-1f * b);
    if (!(an > 126.0f)) return fmaf(k, j, k);
    const uint32_t g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = asf(g + 0x7f000000u);
    const float s2 = asf(e - g);
    if (an > 192.0f) return s1 * s1;
    return fmaf(s2, j, s2) * s1;
}

#ifdef VV_BL_X86
/* ggml.c, verbatim. */
BL_TGT static inline __m256 gexpf8(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

/* llamafile / ggml_vec_soft_max_f32 horizontal sum: ((0+4)+(2+6))+((1+5)+(3+7)) */
BL_TGT static inline float hsum8_tb(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
#endif

static inline float hsum8_tb_s(const float* l) {
    const float h0 = l[4] + l[0], h1 = l[5] + l[1], h2 = l[6] + l[2], h3 = l[7] + l[3];
    const float x0 = h0 + h2, x1 = h1 + h3;
    return x0 + x1;
}

/* ─── Norm, rope, silu ──────────────────────────────────────────────────── */

void vv_bitnet_rmsnorm_cpu(const float* x, const float* w, float* y, int T,
                           int n, float eps) {
    int t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (T > 4)
#endif
    for (t = 0; t < T; t++) {
        const float* xr = x + (size_t)t * n;
        float* yr = y + (size_t)t * n;
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += (double)(xr[i] * xr[i]);
        const float mean = (float)(sum / n);
        const float scale = 1.0f / sqrtf(mean + eps);
        for (int i = 0; i < n; i++) yr[i] = (xr[i] * scale) * w[i];
    }
}

/*
 * One head's NEOX rotation. The reference's GCC build contracts ggml's
 * `x0*cos - x1*sin` and `x0*sin + x1*cos` into one FMA each (libggml.so:
 * vfmsub231ss / vfmadd132ss in ggml_compute_forward_rope_f32), which is
 * what the fmaf calls below repeat. The BL_TGT copy inlines them.
 */
#define BL_DEF_ROPE(name, attr)                                               \
    attr static void name(float* r, const float* cache, int hd) {            \
        for (int i0 = 0; i0 < hd; i0 += 2) {                                 \
            const float c = cache[i0], s = cache[i0 + 1];                    \
            const int ic = i0 / 2;                                           \
            const float x0 = r[ic], x1 = r[ic + hd / 2];                     \
            r[ic] = fmaf(x0, c, -(x1 * s));                                  \
            r[ic + hd / 2] = fmaf(x0, s, x1 * c);                            \
        }                                                                    \
    }
BL_DEF_ROPE(rope_head_s, )
#ifdef VV_BL_X86
BL_DEF_ROPE(rope_head_fma, BL_TGT)
#endif

/* NEOX rope at positions pos0 + t, the ggml way. */
static void rope_ggml(float* x, int T, int n_heads, int hd, int pos0,
                      float base, int simd) {
    const float theta_scale = powf(base, -2.0f / (float)hd);
    int t;
    (void)simd;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (T > 4)
#endif
    for (t = 0; t < T; t++) {
        float cache[512];
        float theta = (float)(pos0 + t);
        for (int i0 = 0; i0 < hd; i0 += 2) {
            cache[i0 + 0] = cosf(theta);
            cache[i0 + 1] = sinf(theta);
            theta *= theta_scale;
        }
        for (int h = 0; h < n_heads; h++) {
            float* r = x + ((size_t)t * n_heads + h) * hd;
#ifdef VV_BL_X86
            if (simd) { rope_head_fma(r, cache, hd); continue; }
#endif
            rope_head_s(r, cache, hd);
        }
    }
}

#ifdef VV_BL_X86
BL_TGT static void silu_mul_avx2(float* g, const float* u, size_t n) {
    const __m256 one = _mm256_set1_ps(1), zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(g + i);
        const __m256 s = _mm256_div_ps(x, _mm256_add_ps(one,
                             gexpf8(_mm256_sub_ps(zero, x))));
        _mm256_storeu_ps(g + i, _mm256_mul_ps(s, _mm256_loadu_ps(u + i)));
    }
    for (; i < n; i++) {
        const float s = g[i] / (1.0f + expf(-g[i]));
        g[i] = s * u[i];
    }
}
#endif

/* ggml_vec_silu_f32 (8-lane body; rows here are multiples of 8) then mul */
static void silu_mul(float* g, const float* u, int T, int n, int simd) {
    int t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (T > 1)
#endif
    for (t = 0; t < T; t++) {
        float* gr = g + (size_t)t * n;
        const float* ur = u + (size_t)t * n;
#ifdef VV_BL_X86
        if (simd) { silu_mul_avx2(gr, ur, (size_t)n); continue; }
#endif
        (void)simd;
        const int nv = n / 8 * 8;
        int i = 0;
        for (; i < nv; i++) {
            const float s = gr[i] / (1.0f + gexpf1(0.0f - gr[i]));
            gr[i] = s * ur[i];
        }
        for (; i < n; i++) {
            const float s = gr[i] / (1.0f + expf(-gr[i]));
            gr[i] = s * ur[i];
        }
    }
}

/* ─── Attention ─────────────────────────────────────────────────────────── */

/* ggml_vec_dot_f16 for n % 32 == 0 on F16-valued floats */
static float dot_ggml_s(const float* a, const float* b, int n) {
    float acc[4][8];
    memset(acc, 0, sizeof(acc));
    for (int i = 0; i < n; i += 32)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 8; l++)
                acc[j][l] = fmaf(a[i + 8 * j + l], b[i + 8 * j + l], acc[j][l]);
    float x0[8], t0[4];
    for (int l = 0; l < 8; l++) {
        const float s02 = acc[0][l] + acc[2][l], s13 = acc[1][l] + acc[3][l];
        x0[l] = s02 + s13;
    }
    for (int l = 0; l < 4; l++) t0[l] = x0[l] + x0[l + 4];
    return (t0[0] + t0[1]) + (t0[2] + t0[3]);
}

#ifdef VV_BL_X86
BL_TGT static float dot_ggml_avx2(const float* a, const float* b, int n) {
    __m256 s[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < n; i += 32)
        for (int j = 0; j < 4; j++)
            s[j] = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8 * j),
                                   _mm256_loadu_ps(b + i + 8 * j), s[j]);
    s[0] = _mm256_add_ps(s[0], s[2]);
    s[1] = _mm256_add_ps(s[1], s[3]);
    s[0] = _mm256_add_ps(s[0], s[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(s[0]),
                                 _mm256_extractf128_ps(s[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

/* softmax over wp[0..nc) (nc % 8 == 0), in place into p; returns nothing */
BL_TGT static void softmax_avx2(float* wp, int nc, float max) {
    double sum = 0.0;
    const __m256 vm = _mm256_set1_ps(max);
    for (int i = 0; i < nc; i += 8) {
        const __m256 v = gexpf8(_mm256_sub_ps(_mm256_loadu_ps(wp + i), vm));
        _mm256_storeu_ps(wp + i, v);
        sum += (double)hsum8_tb(v);
    }
    sum = 1.0 / sum;
    const float f = (float)sum;
    for (int i = 0; i < nc; i++) wp[i] *= f;
}

/* out[d] = tinyBLAS dot over positions j of V[j][d] * p[j] */
BL_TGT static void kqv_avx2(const float* V, size_t ldv, const float* p, int nc,
                            int hd, float* acc /* [8][hd] */, float* out) {
    for (int l = 0; l < 8; l++)
        for (int d = 0; d < hd; d += 8)
            _mm256_storeu_ps(acc + l * hd + d, _mm256_setzero_ps());
    for (int j = 0; j < nc; j++) {
        const float pj = p[j];
        if (pj == 0.0f) continue;          /* fma(v, 0, a) == a */
        const __m256 vp = _mm256_set1_ps(pj);
        const float* vr = V + (size_t)j * ldv;
        float* a = acc + (j & 7) * hd;
        for (int d = 0; d < hd; d += 8)
            _mm256_storeu_ps(a + d, _mm256_fmadd_ps(_mm256_loadu_ps(vr + d), vp,
                                                    _mm256_loadu_ps(a + d)));
    }
    for (int d = 0; d < hd; d += 8) {
        __m256 h0 = _mm256_add_ps(_mm256_loadu_ps(acc + 4 * hd + d), _mm256_loadu_ps(acc + 0 * hd + d));
        __m256 h1 = _mm256_add_ps(_mm256_loadu_ps(acc + 5 * hd + d), _mm256_loadu_ps(acc + 1 * hd + d));
        __m256 h2 = _mm256_add_ps(_mm256_loadu_ps(acc + 6 * hd + d), _mm256_loadu_ps(acc + 2 * hd + d));
        __m256 h3 = _mm256_add_ps(_mm256_loadu_ps(acc + 7 * hd + d), _mm256_loadu_ps(acc + 3 * hd + d));
        const __m256 x0 = _mm256_add_ps(h0, h2), x1 = _mm256_add_ps(h1, h3);
        _mm256_storeu_ps(out + d, _mm256_add_ps(x0, x1));
    }
}
#endif

static void softmax_s(float* wp, int nc, float max) {
    double sum = 0.0;
    for (int i = 0; i < nc; i += 8) {
        float v[8];
        for (int l = 0; l < 8; l++) v[l] = wp[i + l] = gexpf1(wp[i + l] - max);
        sum += (double)hsum8_tb_s(v);
    }
    sum = 1.0 / sum;
    const float f = (float)sum;
    for (int i = 0; i < nc; i++) wp[i] *= f;
}

static void kqv_s(const float* V, size_t ldv, const float* p, int nc, int hd,
                  float* acc, float* out) {
    memset(acc, 0, sizeof(float) * 8 * (size_t)hd);
    for (int j = 0; j < nc; j++) {
        if (p[j] == 0.0f) continue;
        const float* vr = V + (size_t)j * ldv;
        float* a = acc + (j & 7) * hd;
        for (int d = 0; d < hd; d++) a[d] = fmaf(vr[d], p[j], a[d]);
    }
    for (int d = 0; d < hd; d++) {
        float l[8];
        for (int k = 0; k < 8; k++) l[k] = acc[k * hd + d];
        out[d] = hsum8_tb_s(l);
    }
}

/* Scratch floats one attention worker needs for a cache of kv positions. */
static size_t attn_scratch(int kv, int hd) {
    return 2 * (size_t)((kv + 31) / 32 * 32) + (size_t)hd * 9 + 16;
}

/*
 * KQV for a single query, as ggml computes it when the batch has one token:
 * llamafile's sgemm only takes n >= 2, so the product falls back to
 * ggml_vec_dot_f16 over the transposed F16 V rows with the probabilities
 * converted to F16. p is already F16-valued and zero past the cache;
 * nc % 32 == 0. col is nc floats of scratch.
 */
static void kqv_vec(const float* V, size_t ldv, const float* p, int nc, int hd,
                    float* col, float* out, int simd) {
    (void)simd;
    for (int d = 0; d < hd; d++) {
        for (int j = 0; j < nc; j++) col[j] = p[j] != 0.0f ? V[(size_t)j * ldv + d] : 0.0f;
#ifdef VV_BL_X86
        out[d] = simd ? dot_ggml_avx2(col, p, nc) : dot_ggml_s(col, p, nc);
#else
        out[d] = dot_ggml_s(col, p, nc);
#endif
    }
}

/*
 * Causal attention of T queries at positions pos0.. over the cache,
 * Q rounded to F16 as the reference's KQ product does.
 */
static void attention(const float* q, const float* K, const float* V,
                      float* out, int T, int n_heads, int n_kv, int hd,
                      int pos0, float* scratch, size_t per_thread, int simd) {
    const int grp = n_heads / n_kv;
    const size_t ldkv = (size_t)n_kv * hd;
    const float kq_scale = 1.0f / sqrtf((float)hd);
    const int total = T * n_heads;
    int w;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (total > 1)
#endif
    for (w = 0; w < total; w++) {
        const int t = w / n_heads, h = w % n_heads, hk = h / grp;
#ifdef _OPENMP
        float* sc = scratch + per_thread * (size_t)omp_get_thread_num();
#else
        float* sc = scratch;
#endif
        const int kvl = pos0 + t + 1;
        const int nc = T == 1 ? (kvl + 31) / 32 * 32 : (kvl + 7) / 8 * 8;
        float* wp = sc;
        float* qh = sc + nc;               /* hd */
        float* acc = qh + hd;              /* 8 * hd, or nc for kqv_vec */
        const float* qr = q + ((size_t)t * n_heads + h) * hd;
        for (int d = 0; d < hd; d++) qh[d] = f16r(qr[d]);
        float max = -INFINITY;
        for (int j = 0; j < kvl; j++) {
            const float* kr = K + (size_t)j * ldkv + (size_t)hk * hd;
#ifdef VV_BL_X86
            const float s = simd ? dot_ggml_avx2(kr, qh, hd) : dot_ggml_s(kr, qh, hd);
#else
            const float s = dot_ggml_s(kr, qh, hd);
#endif
            wp[j] = s * kq_scale;
            if (wp[j] > max) max = wp[j];
        }
        for (int j = kvl; j < nc; j++) wp[j] = -INFINITY;
        float* o = out + ((size_t)t * n_heads + h) * hd;
        if (T == 1) {
#ifdef VV_BL_X86
            if (simd) softmax_avx2(wp, nc, max);
            else
#endif
            softmax_s(wp, nc, max);
            for (int j = 0; j < nc; j++) wp[j] = f16r(wp[j]);
            kqv_vec(V + (size_t)hk * hd, ldkv, wp, nc, hd, acc, o, simd);
            continue;
        }
#ifdef VV_BL_X86
        if (simd) {
            softmax_avx2(wp, nc, max);
            kqv_avx2(V + (size_t)hk * hd, ldkv, wp, nc, hd, acc, o);
            continue;
        }
#endif
        softmax_s(wp, nc, max);
        kqv_s(V + (size_t)hk * hd, ldkv, wp, nc, hd, acc, o);
    }
}

/* ─── The layer ─────────────────────────────────────────────────────────── */

size_t vv_bitnet_layer_cpu_bytes(const vv_llm_config_t* c, int T, int kv_max) {
    const size_t hs = (size_t)c->hidden_size;
    const size_t qd = (size_t)c->num_attention_heads * c->head_dim;
    const size_t kvd = (size_t)c->num_key_value_heads * c->head_dim;
    const size_t in = (size_t)c->intermediate_size;
    const size_t wide = in > qd ? in : qd;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    const size_t fl = (size_t)T * (hs + qd + 2 * kvd + qd + 2 * in)
                    + (size_t)T * 2            /* scales, sums */
                    + (size_t)nt * attn_scratch(kv_max, c->head_dim);
    return fl * sizeof(float) + (size_t)T * (wide > hs ? wide : hs) + 256;
}

/*
 * VV_BITNET_LM_DUMP=<dir>: every intermediate of a layer call, named after
 * the reference graph's tensors (l<layer>_p<pos0>_<name>.bin, FP32 rows),
 * for a diff against VibeASR.cpp's eval-callback dump.
 */
static void lm_dump(int layer, int pos0, const char* name, const float* x,
                    size_t n) {
    const char* d = getenv("VV_BITNET_LM_DUMP");
    if (!d || !d[0] || pos0 > 4096) return;
    char fn[1024];
    snprintf(fn, sizeof(fn), "%s/l%d_p%d_%s.bin", d, layer, pos0, name);
    FILE* f = fopen(fn, "wb");
    if (!f) return;
    fwrite(x, sizeof(float), n, f);
    fclose(f);
}

/** One ternary projection on pre-quantized rows. */
static vv_status_t tproj(const vv_weight_t* w, const int8_t* q, const float* s,
                         float* y, int T, int N, int K) {
    if (w->quant_kind != VV_QUANT_TERNARY) return VV_ERR_UNSUPPORTED;
    return vv_ternary_linear_cpu(q, s, (const uint8_t*)w->tensor.data,
                                 w->tscale, (const float*)w->bias.data, y,
                                 T, N, K);
}

vv_status_t vv_bitnet_layer_cpu(const vv_layer_weights_t* L,
                                const vv_llm_config_t* c, float* h,
                                vv_kv_cache_t* kv, int layer, int pos0, int T,
                                void* ws, size_t ws_bytes) {
    const int hs = c->hidden_size, nh = c->num_attention_heads;
    const int nkv = c->num_key_value_heads, hd = c->head_dim;
    const int qd = nh * hd, kvd = nkv * hd, in = c->intermediate_size;
    if (hd % 32 || hd > 512 || nh % nkv) return VV_ERR_UNSUPPORTED;
    const int kv_len = pos0 + T;
    if (ws_bytes < vv_bitnet_layer_cpu_bytes(c, T, kv_len))
        return VV_ERR_OUT_OF_MEMORY;
    const int simd = bl_simd();

    float* f = (float*)ws;
    float* norm = f;             f += (size_t)T * hs;
    float* qb = f;               f += (size_t)T * qd;
    float* kb = f;               f += (size_t)T * kvd;
    float* vb = f;               f += (size_t)T * kvd;
    float* att = f;              f += (size_t)T * qd;
    float* gate = f;             f += (size_t)T * in;
    float* up = f;               f += (size_t)T * in;
    float* sc = f;               f += (size_t)T;
    f += (size_t)T;              /* spare */
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    const size_t per = attn_scratch(kv_len, hd);
    float* ascr = f;             f += (size_t)nt * per;
    int8_t* q8 = (int8_t*)f;
    vv_status_t s;

    /* attention */
    lm_dump(layer, pos0, "inp", h, (size_t)T * hs);
    vv_bitnet_rmsnorm_cpu(h, (const float*)L->input_layernorm.data, norm, T,
                          hs, c->rms_norm_eps);
    lm_dump(layer, pos0, "attn_norm", norm, (size_t)T * hs);
    if ((s = vv_act_quant_i8_cpu(norm, T, hs, q8, sc, NULL)) != VV_OK) return s;
    if ((s = tproj(&L->attn.q_proj, q8, sc, qb, T, qd, hs)) != VV_OK ||
        (s = tproj(&L->attn.k_proj, q8, sc, kb, T, kvd, hs)) != VV_OK ||
        (s = tproj(&L->attn.v_proj, q8, sc, vb, T, kvd, hs)) != VV_OK)
        return s;
    lm_dump(layer, pos0, "Qpre", qb, (size_t)T * qd);
    lm_dump(layer, pos0, "Vcur", vb, (size_t)T * kvd);
    rope_ggml(qb, T, nh, hd, pos0 + kv->rope_gap, c->rope_theta, simd);
    rope_ggml(kb, T, nkv, hd, pos0 + kv->rope_gap, c->rope_theta, simd);
    lm_dump(layer, pos0, "Qcur", qb, (size_t)T * qd);
    lm_dump(layer, pos0, "Kcur", kb, (size_t)T * kvd);
    /* the reference's cache is F16 */
    for (size_t i = 0; i < (size_t)T * kvd; i++) {
        kb[i] = f16r(kb[i]);
        vb[i] = f16r(vb[i]);
    }
    if ((s = vv_kv_cache_append(kv, layer, kb, vb, T, false, NULL)) != VV_OK)
        return s;
    {
        const void *kc, *vc;
        int cl;
        vv_kv_cache_get(kv, layer, &kc, &vc, &cl);
        attention(qb, (const float*)kc, (const float*)vc, att, T, nh, nkv, hd,
                  pos0, ascr, per, simd);
    }
    lm_dump(layer, pos0, "kqv_merged_cont", att, (size_t)T * qd);
    if ((s = vv_act_quant_i8_cpu(att, T, qd, q8, sc, NULL)) != VV_OK) return s;
    if ((s = tproj(&L->attn.o_proj, q8, sc, norm, T, hs, qd)) != VV_OK) return s;
    lm_dump(layer, pos0, "kqv_out", norm, (size_t)T * hs);
    for (size_t i = 0; i < (size_t)T * hs; i++) h[i] = norm[i] + h[i];
    lm_dump(layer, pos0, "ffn_inp", h, (size_t)T * hs);

    /* feed-forward */
    vv_bitnet_rmsnorm_cpu(h, (const float*)L->post_attn_layernorm.data, norm,
                          T, hs, c->rms_norm_eps);
    if ((s = vv_act_quant_i8_cpu(norm, T, hs, q8, sc, NULL)) != VV_OK) return s;
    if ((s = tproj(&L->mlp.up_proj, q8, sc, up, T, in, hs)) != VV_OK ||
        (s = tproj(&L->mlp.gate_proj, q8, sc, gate, T, in, hs)) != VV_OK)
        return s;
    lm_dump(layer, pos0, "ffn_norm", norm, (size_t)T * hs);
    lm_dump(layer, pos0, "ffn_up", up, (size_t)T * in);
    lm_dump(layer, pos0, "ffn_gate", gate, (size_t)T * in);
    silu_mul(gate, up, T, in, simd);
    lm_dump(layer, pos0, "ffn_gate_par", gate, (size_t)T * in);
    if ((s = vv_act_quant_i8_cpu(gate, T, in, q8, sc, NULL)) != VV_OK) return s;
    if ((s = tproj(&L->mlp.down_proj, q8, sc, norm, T, hs, in)) != VV_OK) return s;
    lm_dump(layer, pos0, "ffn_out", norm, (size_t)T * hs);
    for (size_t i = 0; i < (size_t)T * hs; i++) h[i] = norm[i] + h[i];
    lm_dump(layer, pos0, "l_out", h, (size_t)T * hs);
    return VV_OK;
}

/* ─── Head ──────────────────────────────────────────────────────────────── */

/*
 * One head row. The reference's head is a plain F16 mul_mat on the CPU: the
 * activation is converted to F16 (vec_dot_type) and ggml_vec_dot_f16 runs
 * its 4 x 8 FMA lanes, the same dot the KQ product uses. The caller rounds
 * x to F16 once; this is dot_ggml over the F16 row.
 */
#ifdef VV_BL_X86
BL_TGT static float head_row_avx2(const uint16_t* w, const float* xh, int K) {
    __m256 s[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < K; i += 32)
        for (int j = 0; j < 4; j++)
            s[j] = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + i + 8 * j))),
                _mm256_loadu_ps(xh + i + 8 * j), s[j]);
    s[0] = _mm256_add_ps(s[0], s[2]);
    s[1] = _mm256_add_ps(s[1], s[3]);
    s[0] = _mm256_add_ps(s[0], s[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(s[0]),
                                 _mm256_extractf128_ps(s[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}
#endif

static float head_row_s(const uint16_t* w, const float* xh, int K) {
    float wr[32];
    float acc[4][8];
    memset(acc, 0, sizeof(acc));
    for (int i = 0; i < K; i += 32) {
        for (int l = 0; l < 32; l++) wr[l] = vv_half_to_float(w[i + l]);
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 8; l++)
                acc[j][l] = fmaf(wr[8 * j + l], xh[i + 8 * j + l], acc[j][l]);
    }
    float x0[8], t0[4];
    for (int l = 0; l < 8; l++) {
        const float s02 = acc[0][l] + acc[2][l], s13 = acc[1][l] + acc[3][l];
        x0[l] = s02 + s13;
    }
    for (int l = 0; l < 4; l++) t0[l] = x0[l] + x0[l + 4];
    return (t0[0] + t0[1]) + (t0[2] + t0[3]);
}

/* The largest hidden size the head helpers take (x rounded on the stack). */
#define BL_HEAD_MAX_K 8192

vv_status_t vv_bitnet_head_f16_argmax_cpu(const float* x, const uint16_t* w,
                                          int V, int K, int32_t* token,
                                          float* value) {
    if (!x || !w || !token) return VV_ERR_NULL_PTR;
    if (K % 32 || K > BL_HEAD_MAX_K) return VV_ERR_UNSUPPORTED;
    const int simd = bl_simd();
    float xh[BL_HEAD_MAX_K];
    for (int k = 0; k < K; k++) xh[k] = f16r(x[k]);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    float bv[256];
    int bi[256];
    if (nt > 256) nt = 256;
    for (int i = 0; i < nt; i++) { bv[i] = -INFINITY; bi[i] = -1; }
    /* Each thread takes one contiguous range, so "first maximum" within a
     * range plus lowest range index across threads is the global first
     * maximum -- llama.cpp's greedy sampler keeps the first one too. */
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
#ifdef _OPENMP
        const int id = omp_get_thread_num(), n = omp_get_num_threads();
#else
        const int id = 0, n = 1;
#endif
        const int r0 = (int)((int64_t)V * id / n), r1 = (int)((int64_t)V * (id + 1) / n);
        float best = -INFINITY;
        int arg = -1;
        for (int r = r0; r < r1; r++) {
            const uint16_t* wr = w + (size_t)r * K;
#ifdef VV_BL_X86
            const float v = simd ? head_row_avx2(wr, xh, K) : head_row_s(wr, xh, K);
#else
            const float v = head_row_s(wr, xh, K);
#endif
            if (v > best || arg < 0) { best = v; arg = r; }
        }
        bv[id] = best;
        bi[id] = arg;
    }
    float best = -INFINITY;
    int arg = -1;
    for (int i = 0; i < nt; i++)
        if (bi[i] >= 0 && (arg < 0 || bv[i] > best)) { best = bv[i]; arg = bi[i]; }
    (void)simd;
    *token = arg;
    if (value) *value = best;
    return arg >= 0 ? VV_OK : VV_ERR_INVALID_ARG;
}

/* ─── The same argmax through an int8 filter ────────────────────────────── */

vv_status_t vv_bitnet_head_filter_build(const uint16_t* w, int V, int K,
                                        int8_t* q, float* scale, float* bound) {
    if (!w || !q || !scale || !bound) return VV_ERR_NULL_PTR;
    if (V <= 0 || K <= 0 || K > 16384) return VV_ERR_INVALID_ARG;
    int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (r = 0; r < V; r++) {
        const uint16_t* wr = w + (size_t)r * K;
        int8_t* qr = q + (size_t)r * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = fabsf(vv_half_to_float(wr[k]));
            if (a > amax) amax = a;
        }
        /* vv_i8_rowquant_f32's rounding: q = rint(w / d), d = amax / 127 */
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        double ew = 0.0, qq = 0.0, ww = 0.0;
        for (int k = 0; k < K; k++) {
            const float v = vv_half_to_float(wr[k]);
            float t = nearbyintf(v * id);
            if (t > 127.0f) t = 127.0f;
            if (t < -127.0f) t = -127.0f;
            qr[k] = (int8_t)t;
            const double e = (double)v - (double)d * (double)t;
            ew += e * e;
            qq += (double)t * (double)t;
            ww += (double)v * (double)v;
        }
        scale[r] = d;
        /* rounded up: the bounds only ever have to be too large */
        bound[3 * (size_t)r + 0] = (float)(sqrt(ew) * (1.0 + 1e-6)) + 1e-30f;
        bound[3 * (size_t)r + 1] = (float)((double)d * sqrt(qq) * (1.0 + 1e-6));
        bound[3 * (size_t)r + 2] = (float)(sqrt(ww) * (1.0 + 1e-6));
    }
    return VV_OK;
}

size_t vv_bitnet_head_filter_scratch(int V, int K) {
    /* int32 accumulators, reused as FP32 upper bounds; the int8 activation */
    return (size_t)V * sizeof(int32_t) + (size_t)K + 256;
}

vv_status_t vv_bitnet_head_filtered_argmax_cpu(
    const float* x, const uint16_t* w, const int8_t* q, const float* scale,
    const float* bound, int V, int K, void* scratch, size_t scratch_bytes,
    int32_t* token, float* value, int* n_exact) {
    if (!x || !w || !q || !scale || !bound || !scratch || !token)
        return VV_ERR_NULL_PTR;
    if (K % 32 || K > BL_HEAD_MAX_K) return VV_ERR_UNSUPPORTED;
    if (scratch_bytes < vv_bitnet_head_filter_scratch(V, K)) return VV_ERR_OVERFLOW;
    /* the scan's input: x rounded to F16, as the reference's mul_mat does;
     * the bounds below are all taken against this vector */
    float xh[BL_HEAD_MAX_K];
    for (int k = 0; k < K; k++) xh[k] = f16r(x[k]);
    int32_t* acc = (int32_t*)scratch;
    float* upper = (float*)scratch;           /* overwrites acc row by row */
    int8_t* xq = (int8_t*)scratch + (size_t)V * sizeof(int32_t);

    /* x as int8 (the ternary layers' per-token quantizer) and the norms of
     * x and of its rounding error, in double, rounded up */
    float s = 0.0f;
    vv_status_t st = vv_act_quant_i8_cpu(xh, 1, K, xq, &s, NULL);
    if (st != VV_OK) return st;
    double xx = 0.0, ee = 0.0;
    for (int k = 0; k < K; k++) {
        const double e = (double)xh[k] - (double)xq[k] / (double)s;
        xx += (double)xh[k] * (double)xh[k];
        ee += e * e;
    }
    const float xn = (float)(sqrt(xx) * (1.0 + 1e-6));
    const float en = (float)(sqrt(ee) * (1.0 + 1e-6));
    st = vv_i8_gemm_i32_cpu(xq, q, acc, 1, V, K);
    if (st != VV_OK) return st;

    /*
     * F16 logit L (what the full scan computes, x meaning xh from here on)
     * vs the int8 estimate a = acc * d / s:  L - a = sum e_w x + d sum q e_x,
     * so |L - a| <= ||e_w|| ||x|| + d ||q|| ||e_x||, and the scan's own
     * float error (32 FMA lanes of K/32 terms, a 5-level sum) is below
     * (K/32 + 5) 2^-24 ||w|| ||x||; the (K/8 + 3) used here is looser. The
     * bound adds a margin to all of it.
     */
    const float fp = (float)((double)(K / 8 + 3) * 5.9604644775390625e-8 * 2.0);
    float lower_t[256];
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (nt > 256) nt = 256;
#endif
    for (int i = 0; i < nt; i++) lower_t[i] = -INFINITY;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
#ifdef _OPENMP
        const int id = omp_get_thread_num();
#else
        const int id = 0;
#endif
        float lo = -INFINITY;
        int r;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (r = 0; r < V; r++) {
            const float a = (float)acc[r] * (scale[r] / s);
            const float* b = bound + 3 * (size_t)r;
            const float B = (b[0] * xn + b[1] * en + fp * b[2] * xn +
                             fabsf(a) * 1e-6f) * 1.01f + 1e-30f;
            upper[r] = a + B;
            if (a - B > lo) lo = a - B;
        }
        lower_t[id] = lo;
    }
    float best_lower = -INFINITY;
    for (int i = 0; i < nt; i++) if (lower_t[i] > best_lower) best_lower = lower_t[i];

    const int simd = bl_simd();
    float best = -INFINITY;
    int arg = -1, n = 0;
    for (int r = 0; r < V; r++) {
        if (!(upper[r] >= best_lower)) continue;
        const uint16_t* wr = w + (size_t)r * K;
#ifdef VV_BL_X86
        const float v = simd ? head_row_avx2(wr, xh, K) : head_row_s(wr, xh, K);
#else
        const float v = head_row_s(wr, xh, K);
#endif
        n++;
        if (arg < 0 || v > best) { best = v; arg = r; }
    }
    (void)simd;
    if (n_exact) *n_exact = n;
    *token = arg;
    if (value) *value = best;
    return arg >= 0 ? VV_OK : VV_ERR_INVALID_ARG;
}
