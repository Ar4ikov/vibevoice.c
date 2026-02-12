/**
 * @file benchmark.c
 * @brief Performance benchmarking for vibevoice.c components.
 *
 * Measures:
 * - Audio preprocessing throughput
 * - NF4 dequantization speed (CPU)
 * - Memory allocation overhead
 * - (GPU benchmarks require CUDA device)
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

/* External */
extern const float VV_NF4_TABLE[16];
extern vv_status_t vv_dequant_nf4_cpu(const uint8_t* packed,
                                        const float* scales,
                                        float* output,
                                        int n_elements,
                                        int block_size);

/* ─── Benchmarks ────────────────────────────────────────────────────────── */

static void bench_normalize(void) {
    printf("bench_normalize:\n");

    int n = 24000 * 60;  /* 60 seconds at 24kHz */
    float* audio = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        audio[i] = 0.5f * sinf(2.0f * 3.14159f * 440.0f * (float)i / 24000.0f);
    }

    int runs = 100;
    double t0 = get_time_ms();
    for (int r = 0; r < runs; r++) {
        vv_audio_normalize(audio, n, -25.0f, 1e-6f);
    }
    double t1 = get_time_ms();

    double ms_per_run = (t1 - t0) / runs;
    double audio_sec = (double)n / 24000.0;
    double rtf = ms_per_run / 1000.0 / audio_sec;

    printf("  %d samples (%.1f sec): %.2f ms/run (RTF: %.6f)\n",
           n, audio_sec, ms_per_run, rtf);

    vv_free(audio);
}

static void bench_nf4_dequant_cpu(void) {
    printf("bench_nf4_dequant_cpu:\n");

    /* Simulate a large weight tensor: 3584 * 3584 = ~12.8M elements */
    int n_elements = 3584 * 3584;
    int n_packed = n_elements / 2;
    int n_blocks = n_elements / 64;

    uint8_t* packed = (uint8_t*)vv_alloc(n_packed);
    float* scales = (float*)vv_alloc((size_t)n_blocks * sizeof(float));
    float* output = (float*)vv_alloc((size_t)n_elements * sizeof(float));

    /* Fill with random-ish data */
    for (int i = 0; i < n_packed; i++) packed[i] = (uint8_t)(i & 0xFF);
    for (int i = 0; i < n_blocks; i++) scales[i] = 0.01f * (float)(i % 100);

    int runs = 10;
    double t0 = get_time_ms();
    for (int r = 0; r < runs; r++) {
        vv_dequant_nf4_cpu(packed, scales, output, n_elements, 64);
    }
    double t1 = get_time_ms();

    double ms_per_run = (t1 - t0) / runs;
    double gb_per_sec = ((double)n_elements * sizeof(float) / 1e9) /
                         (ms_per_run / 1000.0);

    printf("  %d elements (%.1f MB output): %.2f ms/run (%.2f GB/s output)\n",
           n_elements,
           (double)n_elements * sizeof(float) / (1024.0 * 1024.0),
           ms_per_run, gb_per_sec);

    vv_free(packed);
    vv_free(scales);
    vv_free(output);
}

static void bench_resample(void) {
    printf("bench_resample:\n");

    int sr_in = 44100;
    int n = sr_in * 10;  /* 10 seconds at 44.1kHz */
    float* audio = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        audio[i] = sinf(2.0f * 3.14159f * 1000.0f * (float)i / (float)sr_in);
    }

    int runs = 5;
    double total_ms = 0;

    for (int r = 0; r < runs; r++) {
        float* out = NULL;
        int out_len = 0;
        double t0 = get_time_ms();
        vv_audio_resample(audio, sr_in, n, &out, 24000, &out_len);
        double t1 = get_time_ms();
        total_ms += (t1 - t0);
        if (out) vv_free(out);
    }

    double ms_per_run = total_ms / runs;
    printf("  44.1kHz -> 24kHz, 10 sec: %.2f ms/run\n", ms_per_run);

    vv_free(audio);
}

static void bench_alloc(void) {
    printf("bench_alloc:\n");

    int n = 100000;
    double t0 = get_time_ms();
    for (int i = 0; i < n; i++) {
        void* p = vv_alloc(1024);
        vv_free(p);
    }
    double t1 = get_time_ms();

    printf("  %d alloc+free cycles (1KB): %.2f ms (%.0f ns/op)\n",
           n, t1 - t0, (t1 - t0) / n * 1e6);
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    printf("=== vibevoice.c Benchmarks ===\n");
    printf("=== CPU-only (no GPU required) ===\n\n");

    (void)argc;
    (void)argv;

    bench_normalize();
    bench_nf4_dequant_cpu();
    bench_resample();
    bench_alloc();

    printf("\n=== Done ===\n");
    return 0;
}
