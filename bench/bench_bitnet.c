/**
 * @file bench_bitnet.c
 * @brief Throughput of the BitNet integer kernels on the 1.5B shapes.
 *
 *   vv_bench_bitnet [cpu|gpu|all] [prefill_M]
 *
 * Decode (M = 1) is reported as GB/s of weight bytes read, since that is
 * what bounds it; prefill as integer GOP/s (2*M*N*K per call). Each number
 * is the best of several timed repetitions after a warm-up. The CPU uses
 * however many threads OpenMP is given (OMP_NUM_THREADS).
 */

#include "vibevoice/bitnet.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { const char* name; int N, K; } shape_t;

/* one Qwen2-1.5B layer: q, k, v, o, gate, up, down */
static const shape_t SHAPES[] = {
    { "q/o   1536x1536", 1536, 1536 }, { "k/v    256x1536", 256, 1536 },
    { "gate/up 8960x1536", 8960, 1536 }, { "down  1536x8960", 1536, 8960 },
};
/* per layer: 2 of the first, 2 of the second, 2 of the third, 1 of the last */
static const int PER_LAYER[] = { 2, 2, 2, 1 };

static uint32_t rng = 1u;
static uint32_t urand(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void fill(void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)urand();
}
/* ternary codes must stay in 0..2 */
static void fill_codes(uint8_t* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = 0x55 ^ (uint8_t)(urand() & 0x55) ^ (uint8_t)((urand() & 1) ? 0x14 : 0);
    for (size_t i = 0; i < n; i++) {
        uint8_t b = c[i], o = 0;
        for (int f = 0; f < 4; f++) {
            uint8_t v = (b >> (2 * f)) & 3;
            o |= (uint8_t)((v == 3 ? 2 : v) << (2 * f));
        }
        c[i] = o;
    }
}

static double best_ms(double* t, int n) {
    double b = t[0];
    for (int i = 1; i < n; i++) b = t[i] < b ? t[i] : b;
    return b;
}

static void bench_cpu(int PM) {
    /* one thread per physical core, pinned: what the runtime does at init */
    int threads = vv_cpu_threads();
    printf("CPU (%s, %d threads)\n", vv_bitnet_cpu_isa(), threads);
    printf("  %-18s %10s %9s %12s %9s\n", "shape", "M=1 us", "GB/s", "M=PM ms", "GOP/s");
    double layer_us = 0;
    for (size_t s = 0; s < sizeof(SHAPES) / sizeof(SHAPES[0]); s++) {
        const int N = SHAPES[s].N, K = SHAPES[s].K;
        uint8_t* codes = vv_alloc((size_t)N * K / 4);
        int8_t* q = vv_alloc((size_t)PM * K);
        float* y = vv_alloc(sizeof(float) * (size_t)PM * N);
        float* sc = vv_alloc(sizeof(float) * (size_t)PM);
        fill_codes(codes, (size_t)N * K / 4);
        fill(q, (size_t)PM * K);
        for (int m = 0; m < PM; m++) sc[m] = 10.0f;
        double t[20];
        for (int r = 0; r < 3; r++) vv_ternary_linear_cpu(q, sc, codes, 0.05f, NULL, y, 1, N, K);
        for (int r = 0; r < 20; r++) {
            const double t0 = vv_time_ms();
            for (int i = 0; i < 20; i++) vv_ternary_linear_cpu(q, sc, codes, 0.05f, NULL, y, 1, N, K);
            t[r] = (vv_time_ms() - t0) / 20.0;
        }
        const double gemv = best_ms(t, 20);
        vv_ternary_linear_cpu(q, sc, codes, 0.05f, NULL, y, PM, N, K);
        for (int r = 0; r < 5; r++) {
            const double t0 = vv_time_ms();
            vv_ternary_linear_cpu(q, sc, codes, 0.05f, NULL, y, PM, N, K);
            t[r] = vv_time_ms() - t0;
        }
        const double gemm = best_ms(t, 5);
        printf("  %-18s %10.1f %9.1f %12.2f %9.0f\n", SHAPES[s].name, gemv * 1000.0,
               (double)N * K / 4 / (gemv * 1e6), gemm, 2.0 * PM * N * K / (gemm * 1e6));
        layer_us += gemv * 1000.0 * PER_LAYER[s];
        vv_free(codes); vv_free(q); vv_free(y); vv_free(sc);
    }
    printf("  one layer at M=1: %.1f us -> 28 layers %.2f ms/token\n", layer_us, layer_us * 28 / 1000.0);

    /* int8 head */
    {
        const int V = 151936, K = 1536;
        int8_t* w = vv_alloc((size_t)V * K);
        float* ws = vv_alloc(sizeof(float) * V);
        int8_t q[1536];
        fill(w, (size_t)V * K);
        fill(q, K);
        for (int i = 0; i < V; i++) ws[i] = 0.01f;
        int32_t tok;
        double t[10];
        vv_i8_head_argmax_cpu(q, 3.0f, w, ws, V, K, &tok, NULL);
        for (int r = 0; r < 10; r++) {
            const double t0 = vv_time_ms();
            vv_i8_head_argmax_cpu(q, 3.0f, w, ws, V, K, &tok, NULL);
            t[r] = vv_time_ms() - t0;
        }
        const double b = best_ms(t, 10);
        printf("  int8 head 151936x1536: %.2f ms, %.1f GB/s\n", b, (double)V * K / (b * 1e6));
        vv_free(w); vv_free(ws);
    }
    /* int8 x int8 (encoder FFN shapes) */
    {
        static const int VS[][3] = { { 2048, 8192, 2048 }, { 1051, 2048, 512 }, { 8250, 128, 32 } };
        for (int v = 0; v < 3; v++) {
            const int M = VS[v][0], N = VS[v][1], K = VS[v][2];
            int8_t* a = vv_alloc((size_t)M * K);
            int8_t* w = vv_alloc((size_t)N * K);
            float* y = vv_alloc(sizeof(float) * (size_t)M * N);
            fill(a, (size_t)M * K);
            for (size_t i = 0; i < (size_t)N * K; i++) w[i] = (int8_t)((int)(urand() % 255) - 127);
            double t[5];
            float am;
            vv_i8_linear_cpu(a, 2.0f, w, 0.01f, NULL, y, &am, M, N, K);
            for (int r = 0; r < 5; r++) {
                const double t0 = vv_time_ms();
                vv_i8_linear_cpu(a, 2.0f, w, 0.01f, NULL, y, &am, M, N, K);
                t[r] = vv_time_ms() - t0;
            }
            const double b = best_ms(t, 5);
            printf("  int8 GEMM M=%d N=%d K=%d: %.2f ms, %.0f GOP/s\n", M, N, K, b,
                   2.0 * M * N * K / (b * 1e6));
            vv_free(a); vv_free(w); vv_free(y);
        }
    }
}

#ifdef VV_HAS_ACCEL
static void bench_gpu(int PM) {
    if (vv_dev_device_count() <= 0) { printf("GPU: no device\n"); return; }
    char name[256] = "?";
    vv_dev_get_device_name(0, name, sizeof(name));
    printf("GPU (%s)\n", name);
    void* st = NULL;
    vv_dev_stream_create(&st);
    printf("  %-18s %10s %9s %12s %9s\n", "shape", "M=1 us", "GB/s", "M=PM ms", "TOP/s");
    double layer_us = 0;
    for (size_t s = 0; s < sizeof(SHAPES) / sizeof(SHAPES[0]); s++) {
        const int N = SHAPES[s].N, K = SHAPES[s].K;
        uint8_t* codes = vv_alloc((size_t)N * K / 4);
        fill_codes(codes, (size_t)N * K / 4);
        void *dc, *dq, *dy, *ds, *dsum;
        vv_dev_alloc(&dc, (size_t)N * K / 4);
        vv_dev_alloc(&dq, (size_t)PM * K);
        vv_dev_alloc(&dy, 2 * (size_t)PM * N);
        vv_dev_alloc(&ds, 4 * (size_t)PM);
        vv_dev_alloc(&dsum, 4 * (size_t)PM);
        vv_dev_memcpy_h2d(dc, codes, (size_t)N * K / 4, st);
        vv_dev_memset_async(dq, 1, (size_t)PM * K, st);
        vv_dev_memset_async(ds, 0x41, 4 * (size_t)PM, st);
        vv_dev_memset_async(dsum, 0, 4 * (size_t)PM, st);
        double t[10];
        const int reps = 200;
        for (int i = 0; i < 20; i++)
            vv_ternary_gemm_dev(dq, dsum, ds, dc, 0.05f, NULL, NULL, dy, 1, 1, N, K, st);
        vv_dev_stream_sync(st);
        for (int r = 0; r < 10; r++) {
            const double t0 = vv_time_ms();
            for (int i = 0; i < reps; i++)
                vv_ternary_gemm_dev(dq, dsum, ds, dc, 0.05f, NULL, NULL, dy, 1, 1, N, K, st);
            vv_dev_stream_sync(st);
            t[r] = (vv_time_ms() - t0) / reps;
        }
        const double gemv = best_ms(t, 10);
        for (int r = 0; r < 10; r++) {
            const double t0 = vv_time_ms();
            for (int i = 0; i < 10; i++)
                vv_ternary_gemm_dev(dq, dsum, ds, dc, 0.05f, NULL, NULL, dy, 1, PM, N, K, st);
            vv_dev_stream_sync(st);
            t[r] = (vv_time_ms() - t0) / 10;
        }
        const double gemm = best_ms(t, 10);
        printf("  %-18s %10.1f %9.1f %12.3f %9.1f\n", SHAPES[s].name, gemv * 1000.0,
               (double)N * K / 4 / (gemv * 1e6), gemm, 2.0 * PM * N * K / (gemm * 1e9));
        layer_us += gemv * 1000.0 * PER_LAYER[s];
        vv_dev_free(dc); vv_dev_free(dq); vv_dev_free(dy); vv_dev_free(ds); vv_dev_free(dsum);
        vv_free(codes);
    }
    printf("  one layer at M=1: %.1f us (launch-bound when small) -> 28 layers %.2f ms/token\n",
           layer_us, layer_us * 28 / 1000.0);
    {
        const int V = 151936, K = 1536;
        void *dw, *dws, *dq, *ds, *dt, *scr;
        vv_dev_alloc(&dw, (size_t)V * K);
        vv_dev_alloc(&dws, 4 * (size_t)V);
        vv_dev_alloc(&dq, K);
        vv_dev_alloc(&ds, 4);
        vv_dev_alloc(&dt, 4);
        vv_dev_alloc(&scr, vv_i8_head_argmax_scratch_bytes(V));
        vv_dev_memset_async(dw, 3, (size_t)V * K, st);
        vv_dev_memset_async(dws, 0x3c, 4 * (size_t)V, st);
        vv_dev_memset_async(dq, 1, K, st);
        vv_dev_memset_async(ds, 0x40, 4, st);
        double t[10];
        for (int r = 0; r < 10; r++) {
            const double t0 = vv_time_ms();
            for (int i = 0; i < 50; i++)
                vv_i8_head_argmax_dev(dq, ds, dw, dws, V, K, dt, NULL, scr, st);
            vv_dev_stream_sync(st);
            t[r] = (vv_time_ms() - t0) / 50;
        }
        const double b = best_ms(t, 10);
        printf("  int8 head 151936x1536: %.3f ms, %.1f GB/s\n", b, (double)V * K / (b * 1e6));
        vv_dev_free(dw); vv_dev_free(dws); vv_dev_free(dq); vv_dev_free(ds);
        vv_dev_free(dt); vv_dev_free(scr);
    }
    {
        static const int VS[][3] = { { 2048, 8192, 2048 }, { 16384, 2048, 512 }, { 264000, 128, 32 } };
        for (int v = 0; v < 3; v++) {
            const int M = VS[v][0], N = VS[v][1], K = VS[v][2];
            void *da, *dw, *dy, *das, *dmax;
            vv_dev_alloc(&da, (size_t)M * K);
            vv_dev_alloc(&dw, (size_t)N * K);
            vv_dev_alloc(&dy, 4 * (size_t)M * N);
            vv_dev_alloc(&das, 4);
            vv_dev_alloc(&dmax, 4);
            vv_dev_memset_async(da, 1, (size_t)M * K, st);
            vv_dev_memset_async(dw, 2, (size_t)N * K, st);
            vv_dev_memset_async(das, 0x40, 4, st);
            double t[5];
            vv_i8_gemm_dev(da, das, dw, 0.01f, NULL, NULL, dy, dmax, M, N, K, st);
            for (int r = 0; r < 5; r++) {
                vv_dev_stream_sync(st);
                const double t0 = vv_time_ms();
                for (int i = 0; i < 5; i++)
                    vv_i8_gemm_dev(da, das, dw, 0.01f, NULL, NULL, dy, dmax, M, N, K, st);
                vv_dev_stream_sync(st);
                t[r] = (vv_time_ms() - t0) / 5;
            }
            const double b = best_ms(t, 5);
            printf("  int8 GEMM M=%d N=%d K=%d: %.3f ms, %.1f TOP/s\n", M, N, K, b,
                   2.0 * M * N * K / (b * 1e9));
            vv_dev_free(da); vv_dev_free(dw); vv_dev_free(dy); vv_dev_free(das); vv_dev_free(dmax);
        }
    }
    vv_dev_stream_destroy(st);
}
#endif

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "all";
    const int pm = argc > 2 ? atoi(argv[2]) : 512;
    if (!strcmp(what, "cpu") || !strcmp(what, "all")) bench_cpu(pm);
#ifdef VV_HAS_ACCEL
    if (!strcmp(what, "gpu") || !strcmp(what, "all")) bench_gpu(pm > 0 ? pm * 4 : 2048);
#endif
    return 0;
}
