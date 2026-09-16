/**
 * @file cpu_kernels.c
 * @brief CPU compute path. See cpu_kernels.h for the contract.
 *
 * The shape of every quantized GEMM here is the same: walk one output row's
 * weights, dequantize them into a K-float scratch that stays in L1/L2, and
 * dot it against the M input rows. Materialising the whole matrix instead
 * would move 13 GB of FP32 per forward pass at 7B; this moves the 3.2 GB of
 * packed weights once and nothing else, which is the floor for a decode
 * step and makes the CPU path bandwidth-bound rather than absurd.
 *
 * The dequantize step is where the SIMD matters, so it has an AVX2
 * specialisation picked at runtime. Everything else is written so the
 * compiler can vectorise it on its own.
 */

#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"

#include "vv_thread.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define VV_X86 1
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#define VV_NEON 1
#include <arm_neon.h>
#endif

/** bitsandbytes NF4 code book. */
static const float NF4[16] = {
    -1.0f,                 -0.6961928009986877f,  -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
    -0.09105003625154495f,  0.0f,                  0.07958029955625534f,
     0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,
     1.0f
};

/*
 * byte -> its two values, high nibble first. A pair table rather than a
 * 16-entry code book because NEON has no permute across 16 floats: two
 * 8-byte loads and a combine beat any select chain, and the table is 2 KB.
 */
static float g_nf4_pair[256][2];
static float g_i4_pair[256][2];
static vv_once_t g_tables_once = VV_ONCE_INIT;

static void fill_tables(void) {
    for (int b = 0; b < 256; b++) {
        g_nf4_pair[b][0] = NF4[b >> 4];
        g_nf4_pair[b][1] = NF4[b & 0xF];
        g_i4_pair[b][0] = (float)(b >> 4);
        g_i4_pair[b][1] = (float)(b & 0xF);
    }
}

/** @brief Build the nibble tables once, however many threads ask for them. */
static void build_tables(void) { vv_once(&g_tables_once, fill_tables); }

/* ─── Runtime SIMD selection ────────────────────────────────────────────── */

typedef enum { SIMD_SCALAR = 0, SIMD_AVX2, SIMD_NEON } simd_kind_t;

static simd_kind_t detect_simd(void) {
#if defined(VV_X86) && defined(__GNUC__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c"))
        return SIMD_AVX2;
#endif
#ifdef VV_NEON
    return SIMD_NEON;
#endif
    return SIMD_SCALAR;
}

static simd_kind_t simd_kind(void) {
    static int cached = -1;
    if (cached < 0) cached = (int)detect_simd();
    return (simd_kind_t)cached;
}

const char* vv_cpu_simd_name(void) {
    switch (simd_kind()) {
        case SIMD_AVX2: return "AVX2";
        case SIMD_NEON: return "NEON";
        default:        return "scalar";
    }
}

/**
 * @brief Physical cores, or 0 when the topology cannot be read.
 *
 * These kernels are bandwidth bound, and a second thread on the same core
 * adds contention rather than throughput: on a 5900X, decode runs at 7.8
 * tok/s across the 12 cores and 4.3 across the 24 hardware threads.
 */
static int physical_cores(void) {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (!len) return 0;
    char* buf = (char*)malloc(len);
    if (!buf) return 0;
    int n = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf, &len)) {
        for (DWORD off = 0; off < len;) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf + off);
            if (p->Relationship == RelationProcessorCore) n++;
            off += p->Size;
        }
    }
    free(buf);
    return n;
#elif defined(__APPLE__)
    int n = 0;
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0)
        return n;                       /* performance cores only */
    sz = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0) == 0) return n;
    return 0;
#elif defined(__linux__)
    /* One entry per core: the CPU that leads its own sibling list. */
    int n = 0;
    for (int cpu = 0; cpu < 1024; cpu++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                 cpu);
        FILE* f = fopen(path, "r");
        if (!f) break;
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == cpu) n++;
        fclose(f);
    }
    return n;
#else
    return 0;
#endif
}

int vv_cpu_threads(void) {
#ifdef _OPENMP
    static int set = 0;
    if (!set) {
        set = 1;
        /* An explicit OMP_NUM_THREADS always wins. */
        if (!getenv("OMP_NUM_THREADS")) {
            const int cores = physical_cores();
            if (cores > 0 && cores < omp_get_max_threads())
                omp_set_num_threads(cores);
        }
    }
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* ─── Dequantize one weight row ─────────────────────────────────────────── */

static void nf4_row_scalar(const uint8_t* __restrict w,
                           const uint16_t* __restrict scales,
                           float* __restrict out, int K) {
    for (int base = 0; base < K; base += 64) {
        const float s = vv_half_to_float(scales[base >> 6]);
        const uint8_t* wb = w + (base >> 1);
        float* o = out + base;
        for (int j = 0; j < 32; j++) {
            const float* p = g_nf4_pair[wb[j]];
            o[2 * j + 0] = p[0] * s;
            o[2 * j + 1] = p[1] * s;
        }
    }
}

#if defined(VV_X86) && defined(__GNUC__)
__attribute__((target("avx2,fma,f16c")))
static void nf4_row_avx2(const uint8_t* __restrict w,
                         const uint16_t* __restrict scales,
                         float* __restrict out, int K) {
    /*
     * Element e lives in nibble (e & 1 ? low : high) of byte e/2, so the
     * eight values covered by one uint32 need shifts 4,0,12,8,20,16,28,24 —
     * one variable shift instead of a byte shuffle plus an unpack.
     */
    const __m256i shifts = _mm256_setr_epi32(4, 0, 12, 8, 20, 16, 28, 24);
    const __m256i mask = _mm256_set1_epi32(0xF);
    const __m256 lut_lo = _mm256_loadu_ps(NF4);
    const __m256 lut_hi = _mm256_loadu_ps(NF4 + 8);

    for (int base = 0; base < K; base += 64) {
        const __m256 s = _mm256_set1_ps(vv_half_to_float(scales[base >> 6]));
        const uint32_t* wp = (const uint32_t*)(w + (base >> 1));
        for (int j = 0; j < 8; j++) {
            const __m256i bits = _mm256_set1_epi32((int)wp[j]);
            const __m256i idx =
                _mm256_and_si256(_mm256_srlv_epi32(bits, shifts), mask);
            const __m256 a = _mm256_permutevar8x32_ps(lut_lo, idx);
            const __m256 b = _mm256_permutevar8x32_ps(lut_hi, idx);
            /* bit 3 of the index selects the upper half of the code book. */
            const __m256 sel =
                _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28));
            const __m256 v = _mm256_blendv_ps(a, b, sel);
            _mm256_storeu_ps(out + base + j * 8, _mm256_mul_ps(v, s));
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void int4g_row_avx2(const uint8_t* __restrict w,
                           const uint16_t* __restrict scales,
                           const uint16_t* __restrict mins,
                           float* __restrict out, int K, int group) {
    const __m256i shifts = _mm256_setr_epi32(4, 0, 12, 8, 20, 16, 28, 24);
    const __m256i mask = _mm256_set1_epi32(0xF);

    for (int base = 0; base < K; base += group) {
        const int g = base / group;
        const __m256 s = _mm256_set1_ps(vv_half_to_float(scales[g]));
        const __m256 m = _mm256_set1_ps(vv_half_to_float(mins[g]));
        const uint32_t* wp = (const uint32_t*)(w + (base >> 1));
        for (int j = 0; j < group / 8; j++) {
            const __m256i bits = _mm256_set1_epi32((int)wp[j]);
            const __m256i idx =
                _mm256_and_si256(_mm256_srlv_epi32(bits, shifts), mask);
            const __m256 q = _mm256_cvtepi32_ps(idx);
            _mm256_storeu_ps(out + base + j * 8, _mm256_fmadd_ps(q, s, m));
        }
    }
}

/*
 * Four accumulators, not one. An FMA has four cycles of latency and the core
 * can start two per cycle, so a single dependency chain runs at an eighth of
 * the throughput the unit is capable of; four independent chains keep it fed.
 */
__attribute__((target("avx2,fma")))
static float dot_avx2(const float* __restrict a, const float* __restrict b,
                      int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                               _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                               _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                               _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx2,f16c")))
static void f16_row_avx2(const uint16_t* __restrict src,
                         float* __restrict dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i,
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(src + i))));
    for (; i < n; i++) dst[i] = vv_half_to_float(src[i]);
}
#endif /* VV_X86 && __GNUC__ */

#ifdef VV_NEON
static void nf4_row_neon(const uint8_t* __restrict w,
                         const uint16_t* __restrict scales,
                         float* __restrict out, int K) {
    for (int base = 0; base < K; base += 64) {
        const float32x4_t s = vdupq_n_f32(vv_half_to_float(scales[base >> 6]));
        const uint8_t* wb = w + (base >> 1);
        float* o = out + base;
        for (int j = 0; j < 32; j += 2) {
            const float32x4_t v = vcombine_f32(vld1_f32(g_nf4_pair[wb[j]]),
                                               vld1_f32(g_nf4_pair[wb[j + 1]]));
            vst1q_f32(o + 2 * j, vmulq_f32(v, s));
        }
    }
}

static void int4g_row_neon(const uint8_t* __restrict w,
                           const uint16_t* __restrict scales,
                           const uint16_t* __restrict mins,
                           float* __restrict out, int K, int group) {
    for (int base = 0; base < K; base += group) {
        const int g = base / group;
        const float32x4_t s = vdupq_n_f32(vv_half_to_float(scales[g]));
        const float32x4_t m = vdupq_n_f32(vv_half_to_float(mins[g]));
        const uint8_t* wb = w + (base >> 1);
        float* o = out + base;
        for (int j = 0; j < group / 2; j += 2) {
            const float32x4_t q = vcombine_f32(vld1_f32(g_i4_pair[wb[j]]),
                                               vld1_f32(g_i4_pair[wb[j + 1]]));
            vst1q_f32(o + 2 * j, vfmaq_f32(m, q, s));
        }
    }
}

static float dot_neon(const float* __restrict a, const float* __restrict b,
                      int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    const float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1),
                                      vaddq_f32(acc2, acc3));
#ifdef __aarch64__
    float sum = vaddvq_f32(acc);
#else
    float32x2_t h = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float sum = vget_lane_f32(vpadd_f32(h, h), 0);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
static void f16_row_neon(const uint16_t* __restrict src,
                         float* __restrict dst, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i,
                  vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(src + i))));
    for (; i < n; i++) dst[i] = vv_half_to_float(src[i]);
}
#endif
#endif /* VV_NEON */

static void int4g_row_scalar(const uint8_t* __restrict w,
                             const uint16_t* __restrict scales,
                             const uint16_t* __restrict mins,
                             float* __restrict out, int K, int group) {
    for (int base = 0; base < K; base += group) {
        const int g = base / group;
        const float s = vv_half_to_float(scales[g]);
        const float m = vv_half_to_float(mins[g]);
        const uint8_t* wb = w + (base >> 1);
        float* o = out + base;
        for (int j = 0; j < group / 2; j++) {
            o[2 * j + 0] = (float)(wb[j] >> 4) * s + m;
            o[2 * j + 1] = (float)(wb[j] & 0xF) * s + m;
        }
    }
}

static void f16_row(const uint16_t* src, float* dst, int n) {
#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2) { f16_row_avx2(src, dst, n); return; }
#endif
#if defined(VV_NEON) && (defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__))
    f16_row_neon(src, dst, n);
    return;
#endif
    for (int i = 0; i < n; i++) dst[i] = vv_half_to_float(src[i]);
}

static float dot_f32(const float* a, const float* b, int n) {
#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2) return dot_avx2(a, b, n);
#endif
#ifdef VV_NEON
    return dot_neon(a, b, n);
#endif
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];         s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

#if defined(VV_X86) && defined(__GNUC__)
/**
 * @brief Fused NF4 dequantize + dot for one output row.
 *
 * The decode step is M = 1, where writing the dequantized row to scratch and
 * reading it back costs more instructions than the multiply-accumulate it
 * feeds. Here the values never leave registers.
 */
__attribute__((target("avx2,fma,f16c")))
static float nf4_dot_avx2(const uint8_t* __restrict w,
                          const uint16_t* __restrict scales,
                          const float* __restrict x, int K) {
    const __m256i shifts = _mm256_setr_epi32(4, 0, 12, 8, 20, 16, 28, 24);
    const __m256i mask = _mm256_set1_epi32(0xF);
    const __m256 lut_lo = _mm256_loadu_ps(NF4);
    const __m256 lut_hi = _mm256_loadu_ps(NF4 + 8);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

    for (int base = 0; base < K; base += 64) {
        const __m256 s = _mm256_set1_ps(vv_half_to_float(scales[base >> 6]));
        const uint32_t* wp = (const uint32_t*)(w + (base >> 1));
        for (int j = 0; j < 8; j += 2) {
            const __m256i i0 = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)wp[j]), shifts), mask);
            const __m256i i1 = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)wp[j + 1]), shifts), mask);
            const __m256 v0 = _mm256_blendv_ps(
                _mm256_permutevar8x32_ps(lut_lo, i0),
                _mm256_permutevar8x32_ps(lut_hi, i0),
                _mm256_castsi256_ps(_mm256_slli_epi32(i0, 28)));
            const __m256 v1 = _mm256_blendv_ps(
                _mm256_permutevar8x32_ps(lut_lo, i1),
                _mm256_permutevar8x32_ps(lut_hi, i1),
                _mm256_castsi256_ps(_mm256_slli_epi32(i1, 28)));
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(v0, s),
                                   _mm256_loadu_ps(x + base + j * 8), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(v1, s),
                                   _mm256_loadu_ps(x + base + j * 8 + 8), acc1);
        }
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma,f16c")))
static float int4g_dot_avx2(const uint8_t* __restrict w,
                            const uint16_t* __restrict scales,
                            const uint16_t* __restrict mins,
                            const float* __restrict x, int K, int group) {
    const __m256i shifts = _mm256_setr_epi32(4, 0, 12, 8, 20, 16, 28, 24);
    const __m256i mask = _mm256_set1_epi32(0xF);
    __m256 acc = _mm256_setzero_ps();

    for (int base = 0; base < K; base += group) {
        const int g = base / group;
        const __m256 s = _mm256_set1_ps(vv_half_to_float(scales[g]));
        const __m256 m = _mm256_set1_ps(vv_half_to_float(mins[g]));
        const uint32_t* wp = (const uint32_t*)(w + (base >> 1));
        for (int j = 0; j < group / 8; j++) {
            const __m256i idx = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)wp[j]), shifts), mask);
            const __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(idx), s, m);
            acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(x + base + j * 8), acc);
        }
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif /* VV_X86 && __GNUC__ */

/* ─── Quantized GEMM ────────────────────────────────────────────────────── */

/*
 * Input rows are processed in blocks so that the block stays in L2 while
 * every weight row is dotted against it. Without the block, a 143-token
 * prefill reads the whole 2 MB activation matrix once per output row: 7 GB
 * out of L3 for one projection, against 58 MB of weight re-reads this way.
 */
/**
 * Row scratch is allocated once per thread per call, not per row: at
 * K = 18944 it is 74 KB, which stays in L2 while the M input rows are dotted
 * against it.
 *
 * Blocking the M and K loops to keep the activation tile in L1 was tried and
 * measured slower here (19 tok/s against 25): at M = 143 the extra weight
 * re-reads and the parallel regions cost more than the L3 traffic they save.
 * A real packed micro-kernel would be the way to beat this, not more tiling.
 *
 * DEQUANT_SLICE fills `row[0..kc)` with weight row `n` from column `kp`; the
 * whole row, in this arrangement.
 */
#define VV_QGEMM_BODY(DEQUANT_SLICE)                                             build_tables();                                                              const uint16_t* bs = (const uint16_t*)bias;                                  const int kp = 0, kc = K;                                                    _Pragma("omp parallel")                                                      {                                                                                float* row = (float*)malloc((size_t)K * sizeof(float));                      if (row) {                                                                       int n;                             _Pragma("omp for schedule(static)")                                          for (n = 0; n < N; n++)      {                                                    DEQUANT_SLICE;                                                               const float b = bs ? vv_half_to_float(bs[n]) : 0.0f;                         for (int m = 0; m < M; m++)                                                      output[(size_t)m * N + n] =                                                      dot_f32(input + (size_t)m * K + kp, row, kc) + b;                }                                                                            free(row);                                                               }                                                                        }

vv_status_t vv_nf4_gemm_cpu(const float* input, const uint8_t* packed,
                            const void* scales, const void* bias,
                            float* output, int M, int N, int K) {
    if (!input || !packed || !scales || !output) return VV_ERR_NULL_PTR;
    if ((K & 63) != 0) return VV_ERR_UNSUPPORTED;

    const uint16_t* sc = (const uint16_t*)scales;
    const uint16_t* bsv = (const uint16_t*)bias;
#if defined(VV_X86) && defined(__GNUC__)
    const bool use_avx2 = (simd_kind() == SIMD_AVX2);
    if (use_avx2 && M == 1) {
        build_tables();
        int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (n = 0; n < N; n++)
            output[n] = nf4_dot_avx2(packed + (size_t)n * (K >> 1),
                                     sc + (size_t)n * (K >> 6), input, K)
                        + (bsv ? vv_half_to_float(bsv[n]) : 0.0f);
        return VV_OK;
    }
    if (use_avx2) {
        VV_QGEMM_BODY(nf4_row_avx2(packed + (size_t)n * (K >> 1) + (kp >> 1),
                                   sc + (size_t)n * (K >> 6) + (kp >> 6),
                                   row, kc))
        return VV_OK;
    }
#endif
#ifdef VV_NEON
    VV_QGEMM_BODY(nf4_row_neon(packed + (size_t)n * (K >> 1) + (kp >> 1),
                               sc + (size_t)n * (K >> 6) + (kp >> 6), row, kc))
#else
    VV_QGEMM_BODY(nf4_row_scalar(packed + (size_t)n * (K >> 1) + (kp >> 1),
                                 sc + (size_t)n * (K >> 6) + (kp >> 6),
                                 row, kc))
#endif
    return VV_OK;
}

vv_status_t vv_int4g_gemm_cpu(const float* input, const uint8_t* packed,
                              const void* scales, const void* mins,
                              const void* bias, float* output,
                              int M, int N, int K, int group) {
    if (!input || !packed || !scales || !mins || !output) return VV_ERR_NULL_PTR;
    if (group <= 0 || (K % group) != 0 || (group & 7) != 0)
        return VV_ERR_UNSUPPORTED;

    const uint16_t* sc = (const uint16_t*)scales;
    const uint16_t* mn = (const uint16_t*)mins;
    const uint16_t* bsv = (const uint16_t*)bias;
    const int ng = K / group;

#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2 && M == 1) {
        int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (n = 0; n < N; n++)
            output[n] = int4g_dot_avx2(packed + (size_t)n * (K >> 1),
                                       sc + (size_t)n * ng,
                                       mn + (size_t)n * ng, input, K, group)
                        + (bsv ? vv_half_to_float(bsv[n]) : 0.0f);
        return VV_OK;
    }
    if (simd_kind() == SIMD_AVX2) {
        VV_QGEMM_BODY(int4g_row_avx2(packed + (size_t)n * (K >> 1) + (kp >> 1),
                                     sc + (size_t)n * ng + kp / group,
                                     mn + (size_t)n * ng + kp / group,
                                     row, kc, group))
        return VV_OK;
    }
#endif
#ifdef VV_NEON
    VV_QGEMM_BODY(int4g_row_neon(packed + (size_t)n * (K >> 1) + (kp >> 1),
                                 sc + (size_t)n * ng + kp / group,
                                 mn + (size_t)n * ng + kp / group,
                                 row, kc, group))
#else
    VV_QGEMM_BODY(int4g_row_scalar(packed + (size_t)n * (K >> 1) + (kp >> 1),
                                   sc + (size_t)n * ng + kp / group,
                                   mn + (size_t)n * ng + kp / group,
                                   row, kc, group))
#endif
    return VV_OK;
}

vv_status_t vv_gemm_f16w_cpu(const float* input, const void* w_fp16,
                             const void* bias_fp16, float* output,
                             int M, int N, int K) {
    if (!input || !w_fp16 || !output) return VV_ERR_NULL_PTR;
    const uint16_t* w = (const uint16_t*)w_fp16;
    const uint16_t* bs = (const uint16_t*)bias_fp16;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float* row = (float*)malloc((size_t)K * sizeof(float));
        if (row) {
            int n;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (n = 0; n < N; n++) {
                f16_row(w + (size_t)n * K, row, K);
                const float b = bs ? vv_half_to_float(bs[n]) : 0.0f;
                for (int m = 0; m < M; m++)
                    output[(size_t)m * N + n] =
                        dot_f32(input + (size_t)m * K, row, K) + b;
            }
            free(row);
        }
    }
    return VV_OK;
}

vv_status_t vv_gemm_f32_cpu(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < M; i++) {
        float* crow = C + (size_t)i * N;
        if (beta == 0.0f) memset(crow, 0, (size_t)N * sizeof(float));
        else for (int j = 0; j < N; j++) crow[j] *= beta;
        for (int k = 0; k < K; k++) {
            const float a = alpha * A[(size_t)i * K + k];
            if (a == 0.0f) continue;
            const float* brow = B + (size_t)k * N;
            for (int j = 0; j < N; j++) crow[j] += a * brow[j];
        }
    }
    return VV_OK;
}

/* ─── Transformer ops ───────────────────────────────────────────────────── */

vv_status_t vv_rmsnorm_cpu(const float* input, const void* weight_fp16,
                           float* output, int rows, int n, float eps) {
    if (!input || !weight_fp16 || !output) return VV_ERR_NULL_PTR;
    const uint16_t* w = (const uint16_t*)weight_fp16;

    int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (r = 0; r < rows; r++) {
        const float* x = input + (size_t)r * n;
        float* y = output + (size_t)r * n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        const float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++)
            y[i] = x[i] * inv * vv_half_to_float(w[i]);
    }
    return VV_OK;
}

vv_status_t vv_rope_cpu(float* x, int rows, int n_heads, int head_dim,
                        int position_offset, float theta) {
    if (!x) return VV_ERR_NULL_PTR;
    const int half = head_dim / 2;

    int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (r = 0; r < rows; r++) {
        const int pos = position_offset + r;
        for (int h = 0; h < n_heads; h++) {
            float* v = x + ((size_t)r * n_heads + h) * head_dim;
            for (int i = 0; i < half; i++) {
                /* Half-split layout: pairs (i, i + head_dim/2). */
                const float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
                const float ang = (float)pos * freq;
                const float c = cosf(ang), s = sinf(ang);
                const float a = v[i], b = v[i + half];
                v[i]        = a * c - b * s;
                v[i + half] = a * s + b * c;
            }
        }
    }
    return VV_OK;
}

vv_status_t vv_attention_prefill_cpu(
    const float* q, const float* k, const float* v, float* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int rows, int q_offset, int kv_len, bool causal) {
    if (!q || !k || !v || !output) return VV_ERR_NULL_PTR;

    const int group = n_q_heads / n_kv_heads;
    const int q_stride = n_q_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    int r;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (r = 0; r < rows; r++) {
        for (int h = 0; h < n_q_heads; h++) {
            const int kv_head = h / group;
            const int q_abs = q_offset + r;
            const int limit = causal ? (q_abs + 1 < kv_len ? q_abs + 1 : kv_len)
                                     : kv_len;
            const float* qv = q + (size_t)r * q_stride + (size_t)h * head_dim;
            float* ov = output + (size_t)r * q_stride + (size_t)h * head_dim;

            /* Online softmax: one pass, no score buffer. */
            float m_i = -INFINITY, l_i = 0.0f;
            for (int d = 0; d < head_dim; d++) ov[d] = 0.0f;

            for (int t = 0; t < limit; t++) {
                const float* kv = k + (size_t)t * kv_stride
                                    + (size_t)kv_head * head_dim;
                const float sc = dot_f32(qv, kv, head_dim) * scale;
                const float m_new = sc > m_i ? sc : m_i;
                const float a = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
                const float p = expf(sc - m_new);
                const float* vv_ = v + (size_t)t * kv_stride
                                     + (size_t)kv_head * head_dim;
                for (int d = 0; d < head_dim; d++) ov[d] = ov[d] * a + p * vv_[d];
                l_i = l_i * a + p;
                m_i = m_new;
            }
            const float inv = l_i > 0.0f ? 1.0f / l_i : 0.0f;
            for (int d = 0; d < head_dim; d++) ov[d] *= inv;
        }
    }
    return VV_OK;
}

vv_status_t vv_attention_decode_cpu(
    const float* q, const float* k_cache, const float* v_cache, float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len) {
    return vv_attention_prefill_cpu(q, k_cache, v_cache, output,
                                    n_q_heads, n_kv_heads, head_dim,
                                    1, cache_len - 1, cache_len, true);
}

vv_status_t vv_swiglu_cpu(const float* gate, const float* up,
                          float* output, int n) {
    if (!gate || !up || !output) return VV_ERR_NULL_PTR;
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < n; i++) {
        const float g = gate[i];
        output[i] = (g / (1.0f + expf(-g))) * up[i];
    }
    return VV_OK;
}

vv_status_t vv_residual_add_cpu(float* x, const float* y, int n) {
    if (!x || !y) return VV_ERR_NULL_PTR;
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < n; i++) x[i] += y[i];
    return VV_OK;
}

vv_status_t vv_embedding_cpu(const void* table_fp16, const int32_t* ids,
                             float* output, int rows, int hidden) {
    if (!table_fp16 || !ids || !output) return VV_ERR_NULL_PTR;
    const uint16_t* t = (const uint16_t*)table_fp16;
    for (int r = 0; r < rows; r++)
        f16_row(t + (size_t)ids[r] * hidden, output + (size_t)r * hidden,
                hidden);
    return VV_OK;
}

vv_status_t vv_lm_head_argmax_cpu(const float* x, const void* w_fp16,
                                  int V, int K, int32_t* token_id,
                                  float* out_value) {
    if (!x || !w_fp16 || !token_id) return VV_ERR_NULL_PTR;
    const uint16_t* w = (const uint16_t*)w_fp16;

    int best = 0;
    float best_v = -INFINITY;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int lb = 0;
        float lv = -INFINITY;
        float* row = (float*)malloc((size_t)K * sizeof(float));
        if (row) {
            int t;
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (t = 0; t < V; t++) {
                f16_row(w + (size_t)t * K, row, K);
                const float s = dot_f32(x, row, K);
                if (s > lv) { lv = s; lb = t; }
            }
            free(row);
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (lv > best_v) { best_v = lv; best = lb; }
        }
    }

    *token_id = best;
    if (out_value) *out_value = best_v;
    return VV_OK;
}

vv_status_t vv_sample_greedy_cpu(const float* logits, int vocab_size,
                                 int32_t* token_id) {
    if (!logits || !token_id) return VV_ERR_NULL_PTR;
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    *token_id = best;
    return VV_OK;
}
