/**
 * @file cpu_kernels.c
 * @brief CPU compute path. See cpu_kernels.h for the contract.
 *
 * There are two quantized GEMMs here and which one runs depends on M.
 *
 * A decode step is M = 1 and reads all 3.2 GB of packed weights for one
 * token, so it is bound by DRAM and nothing else: it walks one output row's
 * weights, dequantizes them straight into registers and dots them against
 * the single input row. Materialising the dequantized matrix instead would
 * move 13 GB of FP32 per forward pass, which is the thing to avoid.
 *
 * Prefill is M in the hundreds and has enough arithmetic to be bound by the
 * FMA units instead, but only if the operands are laid out for them. That is
 * the packed path further down — see the comment on it.
 *
 * The dequantize step is where the SIMD matters, so it has an AVX2
 * specialisation picked at runtime. Everything else is written so the
 * compiler can vectorise it on its own.
 */

/* sched_getaffinity and CPU_SET are GNU extensions. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

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
#elif defined(__linux__)
#include <sched.h>
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

/* ─── Core topology and thread placement ───────────────────────────────── */

/*
 * Two things are needed from the machine: how many physical cores this
 * process may use, and where to put a worker so it gets one to itself.
 *
 * The count matters because these kernels are bandwidth bound and a second
 * thread on the same core adds contention rather than throughput: on a
 * 5900X, decode runs at 7.8 tok/s across the 12 cores and 4.3 across the 24
 * hardware threads.
 *
 * The placement matters for the same reason, and is easy to miss because it
 * is not wrong on average — it is wrong at random. Nothing stops the OS from
 * putting two of the twelve workers on the two hyperthreads of one core and
 * leaving another core idle, and it does: the same prefill measured 650 and
 * 1120 GFLOP/s on consecutive runs until the threads were pinned. libgomp
 * ignores a `proc_bind` clause unless OMP_PROC_BIND is set in the
 * environment, which is not something a library can arrange for itself, so
 * the pinning is done here.
 */

#define VV_MAX_CORES 256

#if defined(_WIN32)
/* A CPU index means nothing on its own past 64 processors: it is an index
   within a group, and a thread is placed by naming both. */
static GROUP_AFFINITY g_core_cpu[VV_MAX_CORES];
#else
static int g_core_cpu[VV_MAX_CORES];  /**< one logical CPU per usable core  */
#endif
static int g_n_cores = -1;            /**< -1 = topology not read yet       */
static vv_once_t g_topo_once = VV_ONCE_INIT;

#if defined(__linux__)
/** @brief Mark every CPU named in a "0-1,12-13" sibling list. */
static void mark_siblings(const char* list, cpu_set_t* set) {
    const char* p = list;
    while (*p) {
        char* end;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); }
        for (long c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET((int)c, set);
        if (*end != ',') break;
        p = end + 1;
    }
}
#endif

/**
 * @brief Fill g_core_cpu with one CPU per physical core this process may use.
 *
 * "May use" rather than "exists": a cpuset or a taskset is the whole reason
 * a container gets four cores out of a host's sixty-four, and counting the
 * host's would oversubscribe every one of them.
 */
static void detect_topology(void) {
    g_n_cores = 0;
#if defined(__linux__)
    cpu_set_t allowed, taken;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    CPU_ZERO(&taken);
    for (int cpu = 0; cpu < CPU_SETSIZE && g_n_cores < VV_MAX_CORES; cpu++) {
        if (!CPU_ISSET(cpu, &allowed) || CPU_ISSET(cpu, &taken)) continue;
        char path[160], list[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                 cpu);
        FILE* f = fopen(path, "r");
        if (f && fgets(list, sizeof(list), f)) mark_siblings(list, &taken);
        else CPU_SET(cpu, &taken);
        if (f) fclose(f);
        g_core_cpu[g_n_cores++] = cpu;
    }
#elif defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (!len) return;
    char* buf = (char*)malloc(len);
    if (!buf) return;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf, &len)) {
        for (DWORD off = 0; off < len && g_n_cores < VV_MAX_CORES;) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf + off);
            /* The low bit of the core's mask: one hardware thread of it. */
            if (p->Relationship == RelationProcessorCore &&
                p->Processor.GroupCount >= 1) {
                const GROUP_AFFINITY* g = &p->Processor.GroupMask[0];
                for (int b = 0; b < (int)(sizeof(KAFFINITY) * 8); b++)
                    if (g->Mask & ((KAFFINITY)1 << b)) {
                        GROUP_AFFINITY* dst = &g_core_cpu[g_n_cores++];
                        memset(dst, 0, sizeof(*dst));
                        dst->Group = g->Group;
                        dst->Mask = (KAFFINITY)1 << b;
                        break;
                    }
            }
            off += p->Size;
        }
    }
    free(buf);
#elif defined(__APPLE__)
    int n = 0;
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) != 0 || n <= 0) {
        sz = sizeof(n);
        if (sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0) != 0) n = 0;
    }
    if (n > VV_MAX_CORES) n = VV_MAX_CORES;
    g_n_cores = n;                                   /* placement: see bind */
#endif
}

static int physical_cores(void) {
    vv_once(&g_topo_once, detect_topology);
    return g_n_cores;
}

/** @brief True unless the user has said where threads go, or said not to. */
static bool binding_wanted(void) {
    const char* off = getenv("VV_CPU_BIND");
    if (off && off[0] == '0') return false;
    return !getenv("OMP_PROC_BIND") && !getenv("OMP_PLACES") &&
           !getenv("GOMP_CPU_AFFINITY") && !getenv("KMP_AFFINITY");
}

/**
 * @brief Pin the calling worker to core `slot`, once per thread.
 *
 * Darwin has no way to ask for this — thread_policy_set's affinity tags are
 * a hint the scheduler may ignore, and are unimplemented on Apple silicon —
 * so there it is a no-op and the P-core count has to be enough.
 */
static void bind_worker(int slot) {
    static VV_TLS int bound = 0;
    if (bound) return;
    bound = 1;
    if (!binding_wanted()) return;
    const int n = physical_cores();
    if (n <= 0 || slot < 0 || slot >= n) return;
#if defined(__linux__)
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(g_core_cpu[slot], &one);
    sched_setaffinity(0, sizeof(one), &one);
#elif defined(_WIN32)
    GROUP_AFFINITY prev;
    SetThreadGroupAffinity(GetCurrentThread(), &g_core_cpu[slot], &prev);
#endif
}

/** @brief Pin every worker of the current team. Call inside a parallel region. */
static void bind_team(void) {
#ifdef _OPENMP
    bind_worker(omp_get_thread_num());
#else
    bind_worker(0);
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
        /*
         * libgomp keeps one pool per host thread and reuses it, so pinning
         * the team here settles every `omp parallel for` below as well.
         */
        if (!omp_in_parallel()) {
#pragma omp parallel
            { bind_team(); }
        }
    }
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int vv_cpu_physical_cores(void) {
    const int n = physical_cores();
    return n > 0 ? n : 0;
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

__attribute__((target("avx2")))
static void int8_row_avx2(const int8_t* __restrict q, float scale,
                          float* __restrict dst, int n) {
    const __m256 s = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(s, _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i))))));
    for (; i < n; i++) dst[i] = (float)q[i] * scale;
}

/**
 * @brief sum_k q[k] * x[k] for one int8 row; the caller applies the scale.
 *
 * Two independent chains, sixteen weights per step: the row is read once,
 * straight into the FMA, which is what decode (M = 1) is bound by.
 */
__attribute__((target("avx2,fma")))
static float int8_dot_avx2(const int8_t* __restrict q,
                           const float* __restrict x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i b = _mm_loadu_si128((const __m128i*)(q + i));
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),
                             _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))),
            _mm256_loadu_ps(x + i + 8), a1);
    }
    const __m256 acc = _mm256_add_ps(a0, a1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += (float)q[i] * x[i];
    return sum;
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

static void int8_row(const int8_t* q, float scale, float* dst, int n) {
#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2) { int8_row_avx2(q, scale, dst, n); return; }
#endif
    for (int i = 0; i < n; i++) dst[i] = (float)q[i] * scale;
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

/* ─── Packed GEMM ───────────────────────────────────────────────────────── */

/*
 * C[M,N] = A[M,K] . B[N,K]^T, B arriving 4-bit packed or as FP16.
 *
 * Both operands are stored K-contiguous, which is the layout an outer
 * product does not want. Dotting one dequantized weight row against each of
 * the M input rows — what this file did before — reads the whole activation
 * matrix once per output column: 2.0 MB per column and 7.3 GB per
 * projection at M = 285, K = 3584, none of which fits anywhere useful. The
 * FMA units then idle at a fifth of their rate waiting on L3.
 *
 * So both operands are copied into k-major panels, A as [KC][MR] and B as
 * [KC][NR], and the MR x NR result stays in registers for a whole k-block.
 * Each A element now meets NR weights while it sits in a register and each
 * B element meets MR inputs, so bytes touched per FMA fall by
 * 2*MR*NR/(MR+NR) — about 11 at 6x16.
 *
 * The block sizes come straight off the cache hierarchy. The B panel is
 * KC*NR*4 bytes, read once per m-panel, so it belongs in L1: 256*16*4 is
 * 16 KB of a 32 KB L1. The A block is MC*KC*4, read once per n-panel and
 * therefore N/NR times, so it belongs in L2: 318*256*4 is 318 KB of 512 KB.
 * MC also sets how many times the weights are streamed from DRAM —
 * ceil(M/MC) — which is why it is as large as L2 allows.
 *
 * Only whole NR-column panels go through here; a ragged tail at the end of N
 * falls back to the row-at-a-time loop below, which is exact for any width.
 */

#define VV_KC     256   /* k-block: multiple of 64, so quant groups align   */
#define VV_L2_A   (320 * 1024)  /* bytes of L2 the A block may claim        */

/** @brief Where one panel of weights comes from, and in what encoding. */
typedef struct {
    const uint8_t*  w;        /**< packed 4-bit codes, row-major over K     */
    const uint16_t* scales;   /**< per-group scale, FP16                    */
    const uint16_t* mins;     /**< INT4G only: -zero * scale                */
    const uint16_t* wf16;     /**< FP16 weights, when there are no codes    */
    int             group;    /**< weights per scale (64 for NF4)           */
    int             K;
    const int8_t*   w8;       /**< INT8 codes [N][K]                        */
    const float*    rscale;   /**< INT8 only: one FP32 scale per row        */
} bpanel_src_t;

/**
 * @brief One micro-kernel: MR x NR of C, accumulating over a k-block.
 * @param mr  Rows of C to write back; the rest of the tile is padding.
 */
typedef void (*micro_fn)(const float* ap, const float* bp, float* c,
                         int ldc, int kc, int mr);
/** @brief Fill bp[kc][NR] with NR weight rows from n0, starting at column kb. */
typedef void (*packb_fn)(const bpanel_src_t* src, int n0, int kb, int kc,
                         float* bp, float* tmp);

typedef struct {
    int      mr, nr;
    micro_fn micro;
    packb_fn packb;
} ukernel_t;

/* ── Portable micro-kernel ── */

/*
 * 4x8 rather than 6x16: without a register count to target, the accumulator
 * tile has to be small enough that any vectorizer keeps it in registers.
 */
#define VV_REF_MR 4
#define VV_REF_NR 8

static void micro_ref(const float* __restrict ap, const float* __restrict bp,
                      float* __restrict c, int ldc, int kc, int mr) {
    float t[VV_REF_MR][VV_REF_NR];
    for (int i = 0; i < VV_REF_MR; i++)
        for (int j = 0; j < VV_REF_NR; j++)
            t[i][j] = (i < mr) ? c[(size_t)i * ldc + j] : 0.0f;

    for (int k = 0; k < kc; k++) {
        const float* __restrict b = bp + (size_t)k * VV_REF_NR;
        for (int i = 0; i < VV_REF_MR; i++) {
            const float a = ap[(size_t)k * VV_REF_MR + i];
            for (int j = 0; j < VV_REF_NR; j++) t[i][j] += a * b[j];
        }
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < VV_REF_NR; j++) c[(size_t)i * ldc + j] = t[i][j];
}

/*
 * The portable packers dequantize a row into `tmp` with the existing row
 * kernels and then scatter it down the panel. That is two passes over the
 * panel instead of one, but it is 4 KB of L1 traffic against a k-block's
 * worth of FMAs, and it keeps one code path for every ISA without one.
 */
#define VV_PACKB_REF(DEQUANT_ROW)                                             \
    for (int r = 0; r < VV_REF_NR; r++) {                                     \
        DEQUANT_ROW;                                                          \
        for (int k = 0; k < kc; k++) bp[(size_t)k * VV_REF_NR + r] = tmp[k];  \
    }

static void packb_nf4_ref(const bpanel_src_t* s, int n0, int kb, int kc,
                          float* bp, float* tmp) {
    const size_t wstr = (size_t)s->K >> 1, sstr = (size_t)s->K >> 6;
#ifdef VV_NEON
    VV_PACKB_REF(nf4_row_neon(s->w + (size_t)(n0 + r) * wstr + (kb >> 1),
                              s->scales + (size_t)(n0 + r) * sstr + (kb >> 6),
                              tmp, kc))
#else
    VV_PACKB_REF(nf4_row_scalar(s->w + (size_t)(n0 + r) * wstr + (kb >> 1),
                                s->scales + (size_t)(n0 + r) * sstr + (kb >> 6),
                                tmp, kc))
#endif
}

static void packb_int4g_ref(const bpanel_src_t* s, int n0, int kb, int kc,
                            float* bp, float* tmp) {
    const size_t wstr = (size_t)s->K >> 1;
    const size_t gstr = (size_t)(s->K / s->group), goff = (size_t)(kb / s->group);
#ifdef VV_NEON
    VV_PACKB_REF(int4g_row_neon(s->w + (size_t)(n0 + r) * wstr + (kb >> 1),
                                s->scales + (size_t)(n0 + r) * gstr + goff,
                                s->mins + (size_t)(n0 + r) * gstr + goff,
                                tmp, kc, s->group))
#else
    VV_PACKB_REF(int4g_row_scalar(s->w + (size_t)(n0 + r) * wstr + (kb >> 1),
                                  s->scales + (size_t)(n0 + r) * gstr + goff,
                                  s->mins + (size_t)(n0 + r) * gstr + goff,
                                  tmp, kc, s->group))
#endif
}

static void packb_f16_ref(const bpanel_src_t* s, int n0, int kb, int kc,
                          float* bp, float* tmp) {
    VV_PACKB_REF(f16_row(s->wf16 + (size_t)(n0 + r) * s->K + kb, tmp, kc))
}

static void packb_int8_ref(const bpanel_src_t* s, int n0, int kb, int kc,
                           float* bp, float* tmp) {
    VV_PACKB_REF(int8_row(s->w8 + (size_t)(n0 + r) * s->K + kb,
                          s->rscale[n0 + r], tmp, kc))
}

static const ukernel_t UK_REF_INT8  = { VV_REF_MR, VV_REF_NR, micro_ref,
                                        packb_int8_ref };
static const ukernel_t UK_REF_NF4   = { VV_REF_MR, VV_REF_NR, micro_ref,
                                        packb_nf4_ref };
static const ukernel_t UK_REF_INT4G = { VV_REF_MR, VV_REF_NR, micro_ref,
                                        packb_int4g_ref };
static const ukernel_t UK_REF_F16   = { VV_REF_MR, VV_REF_NR, micro_ref,
                                        packb_f16_ref };

/* ── AVX2 micro-kernel ── */

#if defined(VV_X86) && defined(__GNUC__)

/*
 * 6 rows by 16 columns: 12 accumulators, two vectors of B and one broadcast
 * of A leave one of the sixteen ymm registers spare. Each k-step issues 12
 * FMAs against 8 loads, so at two FMA ports and two load ports the FMAs are
 * what the loop waits on, which is the point.
 */
#define VV_MR 6
#define VV_NR 16

__attribute__((target("avx2,fma")))
static void micro_6x16_avx2(const float* __restrict ap,
                            const float* __restrict bp,
                            float* __restrict c, int ldc, int kc, int mr) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
    __m256 c8 = _mm256_setzero_ps(), c9 = _mm256_setzero_ps();
    __m256 cA = _mm256_setzero_ps(), cB = _mm256_setzero_ps();

    for (int k = 0; k < kc; k++) {
        const __m256 b0 = _mm256_loadu_ps(bp);
        const __m256 b1 = _mm256_loadu_ps(bp + 8);
        __m256 a;
        a = _mm256_broadcast_ss(ap + 0);
        c0 = _mm256_fmadd_ps(a, b0, c0); c1 = _mm256_fmadd_ps(a, b1, c1);
        a = _mm256_broadcast_ss(ap + 1);
        c2 = _mm256_fmadd_ps(a, b0, c2); c3 = _mm256_fmadd_ps(a, b1, c3);
        a = _mm256_broadcast_ss(ap + 2);
        c4 = _mm256_fmadd_ps(a, b0, c4); c5 = _mm256_fmadd_ps(a, b1, c5);
        a = _mm256_broadcast_ss(ap + 3);
        c6 = _mm256_fmadd_ps(a, b0, c6); c7 = _mm256_fmadd_ps(a, b1, c7);
        a = _mm256_broadcast_ss(ap + 4);
        c8 = _mm256_fmadd_ps(a, b0, c8); c9 = _mm256_fmadd_ps(a, b1, c9);
        a = _mm256_broadcast_ss(ap + 5);
        cA = _mm256_fmadd_ps(a, b0, cA); cB = _mm256_fmadd_ps(a, b1, cB);
        ap += VV_MR; bp += VV_NR;
    }

    /* `mr` is 6 for every panel but the last, so these predict away. */
    #define VV_ST(i, lo, hi) if (mr > (i)) {                                  \
        float* p = c + (size_t)(i) * ldc;                                     \
        _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), lo));           \
        _mm256_storeu_ps(p + 8, _mm256_add_ps(_mm256_loadu_ps(p + 8), hi)); }
    VV_ST(0, c0, c1) VV_ST(1, c2, c3) VV_ST(2, c4, c5)
    VV_ST(3, c6, c7) VV_ST(4, c8, c9) VV_ST(5, cA, cB)
    #undef VV_ST
}

/** @brief The usual two-stage 8x8 float transpose, in registers. */
__attribute__((target("avx2"), always_inline))
static inline void transpose8_avx2(__m256 v[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]);
    const __m256 t1 = _mm256_unpackhi_ps(v[0], v[1]);
    const __m256 t2 = _mm256_unpacklo_ps(v[2], v[3]);
    const __m256 t3 = _mm256_unpackhi_ps(v[2], v[3]);
    const __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]);
    const __m256 t5 = _mm256_unpackhi_ps(v[4], v[5]);
    const __m256 t6 = _mm256_unpacklo_ps(v[6], v[7]);
    const __m256 t7 = _mm256_unpackhi_ps(v[6], v[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    v[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    v[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    v[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    v[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    v[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    v[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    v[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    v[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/*
 * Eight rows are dequantized into eight registers and transposed there, so
 * the panel is written once, contiguously, with no scratch in between. The
 * eight values one uint32 covers never straddle a quant group: k0 is a
 * multiple of 8 and every group size is too — which is also why the scales
 * are converted once per group rather than once per eight values.
 * vv_half_to_float is branchy software, and at one call per eight values it
 * cost more than the dequantize it was feeding.
 */
#define VV_PACKB_AVX2(GRP, GROUP_SETUP, DEQUANT_8)                            \
    const int grp = (GRP);                                                    \
    for (int jb = 0; jb < VV_NR; jb += 8) {                                   \
        for (int kg = 0; kg < kc; kg += grp) {                                \
            float gs[8], gm[8];                                               \
            for (int r = 0; r < 8; r++) {                                     \
                const int n = n0 + jb + r; (void)n; GROUP_SETUP;              \
            }                                                                 \
            for (int k0 = kg; k0 < kg + grp; k0 += 8) {                       \
                const int kk = kb + k0;                                       \
                __m256 v[8];                                                  \
                for (int r = 0; r < 8; r++) {                                 \
                    const int n = n0 + jb + r; (void)n; v[r] = (DEQUANT_8);   \
                }                                                             \
                transpose8_avx2(v);                                           \
                for (int t = 0; t < 8; t++)                                   \
                    _mm256_storeu_ps(bp + (size_t)(k0 + t) * VV_NR + jb,      \
                                     v[t]);                                   \
            }                                                                 \
        }                                                                     \
    }

/** @brief The eight 4-bit codes of row `n` at column `kk`, as float indices. */
__attribute__((target("avx2"), always_inline))
static inline __m256 codes_8_avx2(const uint8_t* w, int kk) {
    const __m256i shifts = _mm256_setr_epi32(4, 0, 12, 8, 20, 16, 28, 24);
    uint32_t bits;
    memcpy(&bits, w + (kk >> 1), sizeof(bits));
    return _mm256_castsi256_ps(_mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32((int)bits), shifts),
        _mm256_set1_epi32(0xF)));
}

__attribute__((target("avx2,fma"), always_inline))
static inline __m256 nf4_8_avx2(const bpanel_src_t* s, int n, int kk,
                                float scale) {
    const __m256i idx = _mm256_castps_si256(
        codes_8_avx2(s->w + (size_t)n * ((size_t)s->K >> 1), kk));
    const __m256 v = _mm256_blendv_ps(
        _mm256_permutevar8x32_ps(_mm256_loadu_ps(NF4), idx),
        _mm256_permutevar8x32_ps(_mm256_loadu_ps(NF4 + 8), idx),
        _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28)));
    return _mm256_mul_ps(v, _mm256_set1_ps(scale));
}

__attribute__((target("avx2,fma"), always_inline))
static inline __m256 int4g_8_avx2(const bpanel_src_t* s, int n, int kk,
                                  float scale, float min) {
    const __m256i idx = _mm256_castps_si256(
        codes_8_avx2(s->w + (size_t)n * ((size_t)s->K >> 1), kk));
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(idx), _mm256_set1_ps(scale),
                           _mm256_set1_ps(min));
}

__attribute__((target("avx2,f16c"), always_inline))
static inline __m256 f16_8_avx2(const bpanel_src_t* s, int n, int kk) {
    return _mm256_cvtph_ps(_mm_loadu_si128(
        (const __m128i*)(s->wf16 + (size_t)n * s->K + kk)));
}

__attribute__((target("avx2,fma,f16c")))
static void packb_nf4_avx2(const bpanel_src_t* s, int n0, int kb, int kc,
                           float* bp, float* tmp) {
    (void)tmp;
    VV_PACKB_AVX2(64,
        gs[r] = vv_half_to_float(
            s->scales[(size_t)n * ((size_t)s->K >> 6) + ((kb + kg) >> 6)]),
        nf4_8_avx2(s, n, kk, gs[r]))
}

__attribute__((target("avx2,fma,f16c")))
static void packb_int4g_avx2(const bpanel_src_t* s, int n0, int kb, int kc,
                             float* bp, float* tmp) {
    (void)tmp;
    const size_t gstr = (size_t)(s->K / s->group);
    VV_PACKB_AVX2(s->group,
        do { const size_t g = (size_t)n * gstr + (size_t)(kb + kg) / s->group;
             gs[r] = vv_half_to_float(s->scales[g]);
             gm[r] = vv_half_to_float(s->mins[g]); } while (0),
        int4g_8_avx2(s, n, kk, gs[r], gm[r]))
}

__attribute__((target("avx2,fma,f16c")))
static void packb_f16_avx2(const bpanel_src_t* s, int n0, int kb, int kc,
                           float* bp, float* tmp) {
    (void)tmp;
    VV_PACKB_AVX2(kc, gs[r] = gm[r] = 0.0f, f16_8_avx2(s, n, kk))
}

__attribute__((target("avx2"), always_inline))
static inline __m256 int8_8_avx2(const bpanel_src_t* s, int n, int kk,
                                 float scale) {
    return _mm256_mul_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_loadl_epi64(
            (const __m128i*)(s->w8 + (size_t)n * s->K + kk)))));
}

__attribute__((target("avx2,fma")))
static void packb_int8_avx2(const bpanel_src_t* s, int n0, int kb, int kc,
                            float* bp, float* tmp) {
    (void)tmp;
    VV_PACKB_AVX2(kc, gs[r] = s->rscale[n]; gm[r] = 0.0f,
                  int8_8_avx2(s, n, kk, gs[r]))
}

static const ukernel_t UK_AVX2_INT8  = { VV_MR, VV_NR, micro_6x16_avx2,
                                         packb_int8_avx2 };
static const ukernel_t UK_AVX2_NF4   = { VV_MR, VV_NR, micro_6x16_avx2,
                                         packb_nf4_avx2 };
static const ukernel_t UK_AVX2_INT4G = { VV_MR, VV_NR, micro_6x16_avx2,
                                         packb_int4g_avx2 };
static const ukernel_t UK_AVX2_F16   = { VV_MR, VV_NR, micro_6x16_avx2,
                                         packb_f16_avx2 };
#endif /* VV_X86 && __GNUC__ */

/* ── Blocking driver ── */

/** @brief Copy MR rows of A, k-major, zero-padded past the last real row. */
static void pack_a_panel(const float* A, int M, int K, int row0, int kb,
                         int kc, int mr_max, float* ap) {
    for (int r = 0; r < mr_max; r++) {
        const int row = row0 + r;
        if (row < M) {
            const float* src = A + (size_t)row * K + kb;
            for (int k = 0; k < kc; k++) ap[(size_t)k * mr_max + r] = src[k];
        } else {
            for (int k = 0; k < kc; k++) ap[(size_t)k * mr_max + r] = 0.0f;
        }
    }
}

/**
 * @brief C[M,N] += A . B^T for the first `n_packed` columns of C.
 *
 * C must already hold the bias (or zero); every panel accumulates into it.
 * Returns VV_ERR_OUT_OF_MEMORY if the panels cannot be allocated, in which
 * case the caller still has its untouched fallback.
 */
static vv_status_t gemm_packed(const float* A, const bpanel_src_t* src,
                               float* C, int M, int N, int K, int n_packed,
                               const ukernel_t* uk) {
    const int MR = uk->mr, NR = uk->nr;
    const int nt = vv_cpu_threads();

    /* A k-block is a whole number of quant groups, so the packers can index
       a scale per eight values without a boundary case. */
    int kc_max = ((VV_KC + src->group - 1) / src->group) * src->group;
    if (kc_max > K) kc_max = K;

    int mc = VV_L2_A / (kc_max * (int)sizeof(float));
    if (mc < MR) mc = MR;
    mc = mc / MR * MR;
    const int m_padded = (M + MR - 1) / MR * MR;
    if (mc > m_padded) mc = m_padded;

    /* One allocation for the shared A block and every thread's B panel, so
       a failure is seen before any thread has written to C. */
    const size_t ap_n = (size_t)mc * kc_max;
    const size_t per_thread = (size_t)kc_max * NR + (size_t)kc_max;
    float* ap = (float*)malloc((ap_n + per_thread * (size_t)nt) * sizeof(float));
    if (!ap) return VV_ERR_OUT_OF_MEMORY;

#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        bind_team();
        float* bp = ap + ap_n + per_thread * (size_t)tid;
        float* tmp = bp + (size_t)kc_max * NR;

        for (int mb = 0; mb < M; mb += mc) {
            const int m_left = M - mb;
            const int m_rows = m_left < mc ? m_left : mc;
            const int panels = (m_rows + MR - 1) / MR;

            for (int kb = 0; kb < K; kb += kc_max) {
                const int k_left = K - kb;
                const int kc = k_left < kc_max ? k_left : kc_max;
                int i;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (i = 0; i < panels; i++)
                    pack_a_panel(A, M, K, mb + i * MR, kb, kc, MR,
                                 ap + (size_t)i * kc_max * MR);
                /* Barrier here: every n-panel below reads all of `ap`. */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (i = 0; i < n_packed / NR; i++) {
                    const int n0 = i * NR;
                    uk->packb(src, n0, kb, kc, bp, tmp);
                    for (int p = 0; p < panels; p++) {
                        const int mr = m_rows - p * MR;
                        uk->micro(ap + (size_t)p * kc_max * MR, bp,
                                  C + (size_t)(mb + p * MR) * N + n0,
                                  N, kc, mr < MR ? mr : MR);
                    }
                }
            }
        }
    }
    free(ap);
    return VV_OK;
}

/** @brief Pick the packed kernel for this encoding, or NULL if there is none. */
static const ukernel_t* pick_ukernel(int kind) {
#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2)
        return kind == 0 ? &UK_AVX2_NF4
             : kind == 1 ? &UK_AVX2_INT4G
             : kind == 3 ? &UK_AVX2_INT8 : &UK_AVX2_F16;
#endif
    return kind == 0 ? &UK_REF_NF4 : kind == 1 ? &UK_REF_INT4G
         : kind == 3 ? &UK_REF_INT8 : &UK_REF_F16;
}

/**
 * @brief Seed C with the bias, which every panel then accumulates into.
 */
static void fill_bias(float* C, const uint16_t* bias, int M, int N) {
    if (!bias) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        return;
    }
    int m;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[(size_t)m * N + n] = vv_half_to_float(bias[n]);
}

/*
 * Below this many rows the packing does not pay for itself: the weights
 * still have to be read in full, so the kernel is bandwidth bound either
 * way, and the row-at-a-time path skips two copies.
 */
#define VV_PACK_MIN_M 4

/**
 * @brief Run the packed path over whole NR-column panels of C.
 * @return Columns computed — 0 when the packed path declined, in which case
 *         C is untouched and the caller's row loop must cover all of N.
 */
static int gemm_packed_try(const float* input, const bpanel_src_t* src,
                           const uint16_t* bias, float* output,
                           int M, int N, int K, int kind) {
    /* Eight values of a row are dequantized at a time, so K has to divide. */
    if (M < VV_PACK_MIN_M || (K & 7) != 0) return 0;
    const ukernel_t* uk = pick_ukernel(kind);
    const int n_packed = N / uk->nr * uk->nr;
    if (n_packed == 0) return 0;

    build_tables();
    fill_bias(output, bias, M, N);
    if (gemm_packed(input, src, output, M, N, K, n_packed, uk) != VV_OK) {
        /* The row loop assigns rather than accumulates, so the bias this
           wrote is harmlessly overwritten. */
        return 0;
    }
    return n_packed;
}

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
#define VV_QGEMM_BODY(DEQUANT_SLICE)                                             build_tables();                                                              const uint16_t* bs = (const uint16_t*)bias;                                  const int kp = 0, kc = K;                                                    _Pragma("omp parallel")                                                      {                                                                                float* row = (float*)malloc((size_t)K * sizeof(float));                      if (row) {                                                                       int n;                             _Pragma("omp for schedule(static)")                                          for (n = n_lo; n < N; n++) {                                                    DEQUANT_SLICE;                                                               const float b = bs ? vv_half_to_float(bs[n]) : 0.0f;                         for (int m = 0; m < M; m++)                                                      output[(size_t)m * N + n] =                                                      dot_f32(input + (size_t)m * K + kp, row, kc) + b;                }                                                                            free(row);                                                               }                                                                        }

vv_status_t vv_nf4_gemm_cpu(const float* input, const uint8_t* packed,
                            const void* scales, const void* bias,
                            float* output, int M, int N, int K) {
    if (!input || !packed || !scales || !output) return VV_ERR_NULL_PTR;
    if ((K & 63) != 0) return VV_ERR_UNSUPPORTED;

    const uint16_t* sc = (const uint16_t*)scales;
    const uint16_t* bsv = (const uint16_t*)bias;

    const bpanel_src_t bsrc = { packed, sc, NULL, NULL, 64, K };
    const int n_lo = gemm_packed_try(input, &bsrc, bsv, output, M, N, K, 0);
    if (n_lo == N) return VV_OK;

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

    const bpanel_src_t bsrc = { packed, sc, mn, NULL, group, K };
    const int n_lo = gemm_packed_try(input, &bsrc, bsv, output, M, N, K, 1);
    if (n_lo == N) return VV_OK;

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

    const bpanel_src_t bsrc = { NULL, NULL, NULL, w, 8, K };
    const int n_lo = gemm_packed_try(input, &bsrc, bs, output, M, N, K, 2);
    if (n_lo == N) return VV_OK;

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
            for (n = n_lo; n < N; n++) {
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

vv_status_t vv_int8_gemm_cpu(const float* input, const int8_t* q,
                             const float* scales, const void* bias,
                             float* output, int M, int N, int K) {
    if (!input || !q || !scales || !output) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    const uint16_t* bs = (const uint16_t*)bias;

    const bpanel_src_t bsrc = { NULL, NULL, NULL, NULL, 8, K, q, scales };
    const int n_lo = gemm_packed_try(input, &bsrc, bs, output, M, N, K, 3);
    if (n_lo == N) return VV_OK;

#if defined(VV_X86) && defined(__GNUC__)
    if (simd_kind() == SIMD_AVX2 && M == 1) {
        int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (n = n_lo; n < N; n++)
            output[n] = int8_dot_avx2(q + (size_t)n * K, input, K) * scales[n]
                        + (bs ? vv_half_to_float(bs[n]) : 0.0f);
        return VV_OK;
    }
#endif

    vv_status_t st = VV_OK;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float* row = (float*)malloc((size_t)K * sizeof(float));
        /* A worker without its row would leave its share unwritten; it
           still takes part in the loop, which every thread must reach. */
        if (!row) st = VV_ERR_OUT_OF_MEMORY;
        int n;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (n = n_lo; n < N; n++) {
            if (!row) continue;
            int8_row(q + (size_t)n * K, scales[n], row, K);
            const float b = bs ? vv_half_to_float(bs[n]) : 0.0f;
            for (int m = 0; m < M; m++)
                output[(size_t)m * N + n] =
                    dot_f32(input + (size_t)m * K, row, K) + b;
        }
        free(row);
    }
    return st;
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
