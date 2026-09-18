/**
 * @file bench_encoder.c
 * @brief Speech front end throughput: one clip, then N at once.
 *
 *   vv_bench_encoder --model DIR --audio WAV [--conc 1,4,8] [--reps 5]
 *                    [--cpu-encoder]
 *
 * Loads the model on device 0, then times the front end alone — both
 * Conv-VAE encoders and both connectors, audio in, prompt rows out — for
 * one clip, and for N threads submitting the same clip at once, with the
 * batching service and without it. Prints milliseconds per second of audio,
 * the best of --reps runs (the box is shared; the best is the signal).
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"
#include "vibevoice/inference.h"
#include "vibevoice/frontend.h"
#include "vibevoice/device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vv_thread.h"

typedef struct {
    vv_frontend_t*        fe;
    vv_frontend_stream_t* st;
    const float*          audio;
    int                   n;
    void*                 rows;
    int                   hs;
    void*                 done;
    vv_status_t           status;
    double                t_end;
} worker_t;

static VV_THREAD_RET worker_main(void* arg) {
    worker_t* w = (worker_t*)arg;
    vv_dev_set_device(0);
    vv_frontend_job_t job;
    memset(&job, 0, sizeof(job));
    vv_frontend_stream_reset(w->st);
    job.audio = w->audio;
    job.n_samples = w->n;
    job.stream = w->st;
    job.is_final = true;
    job.rows = w->rows;
    job.rows_ld = w->hs;
    job.done_event = w->done;
    w->status = vv_frontend_submit(w->fe, &job);
    if (w->status == VV_OK) w->status = vv_dev_event_sync(w->done);
    w->t_end = vv_time_ms();
    VV_THREAD_RETURN;
}

static double run_conc(worker_t* w, int n) {
    vv_thread_t tid[64];
    const double t0 = vv_time_ms();
    for (int i = 0; i < n; i++) vv_thread_start(&tid[i], worker_main, &w[i]);
    double t_end = t0;
    for (int i = 0; i < n; i++) {
        vv_thread_join(tid[i]);
        if (w[i].status != VV_OK) return -1.0;
        if (w[i].t_end > t_end) t_end = w[i].t_end;
    }
    return t_end - t0;
}

int main(int argc, char** argv) {
    const char* model = NULL;
    const char* wav = NULL;
    const char* conc_list = "1,4,8";
    int reps = 5;
    bool cpu_enc = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--audio") && i + 1 < argc) wav = argv[++i];
        else if (!strcmp(argv[i], "--conc") && i + 1 < argc) conc_list = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu-encoder")) cpu_enc = true;
    }
    if (!model || !wav) {
        fprintf(stderr, "usage: %s --model DIR --audio WAV [--conc 1,4,8] "
                        "[--reps N] [--cpu-encoder]\n", argv[0]);
        return 2;
    }
    vv_log_set_level(VV_LOG_WARN);

    float* audio = NULL;
    int n = 0;
    if (vv_audio_preprocess(wav, &audio, &n) != VV_OK) {
        fprintf(stderr, "cannot read %s\n", wav);
        return 1;
    }
    const double sec = (double)n / 24000.0;

    vv_init_params_t ip = vv_init_params_default();
    memset(&ip.gpus, 0, sizeof(ip.gpus));
    if (vv_gpu_set_resolve(&ip.gpus, 0, false) != VV_OK) {
        fprintf(stderr, "no usable gpu 0\n");
        return 1;
    }
    vv_inference_ctx_t* ctx = NULL;
    if (vv_inference_init(model, 0, &ip, &ctx) != VV_OK || !ctx->frontend) {
        fprintf(stderr, "cannot bring the model up on gpu 0\n");
        return 1;
    }
    vv_frontend_t* fe = ctx->frontend;
    const int hs = ctx->model->config.llm.hidden_size;
    const int frames = vv_frontend_frames(fe, n);
    printf("clip %.2f s, %d frames; front end %.1f MB on the device\n",
           sec, frames, (double)vv_frontend_bytes(fe) / (1024.0 * 1024.0));

    enum { MAXW = 64 };
    worker_t w[MAXW];
    memset(w, 0, sizeof(w));
    for (int i = 0; i < MAXW; i++) {
        w[i].fe = fe;
        w[i].audio = audio;
        w[i].n = n;
        w[i].hs = hs;
    }

    /* Warm up: first launches pay for module loading. */
    {
        vv_frontend_stream_create(fe, &w[0].st);
        vv_dev_alloc(&w[0].rows, (size_t)frames * hs * 2);
        vv_dev_event_create(&w[0].done);
        run_conc(w, 1);
        run_conc(w, 1);
    }

    for (int pass = 0; pass < 2; pass++) {
        const bool svc = pass == 1;
        if (svc) vv_frontend_service_start(fe, 0);
        const char* p = conc_list;
        while (*p) {
            const int c = atoi(p);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
            if (c < 1 || c > MAXW) continue;
            for (int i = 0; i < c; i++) {
                if (!w[i].st) vv_frontend_stream_create(fe, &w[i].st);
                if (!w[i].rows) vv_dev_alloc(&w[i].rows, (size_t)frames * hs * 2);
                if (!w[i].done) vv_dev_event_create(&w[i].done);
            }
            double best = 1e30;
            uint64_t l0 = 0, l1 = 0;
            vv_frontend_stats(fe, &l0, NULL);
            for (int r = 0; r < reps; r++) {
                const double ms = run_conc(w, c);
                if (ms < 0) { fprintf(stderr, "run failed\n"); return 1; }
                if (ms < best) best = ms;
            }
            vv_frontend_stats(fe, &l1, NULL);
            printf("%-8s conc=%-2d  wall %8.1f ms  %6.2f ms per audio-second  "
                   "(%.1f launches per run)\n", svc ? "service" : "direct",
                   c, best, best / (sec * c), (double)(l1 - l0) / reps);
        }
        if (svc) vv_frontend_service_stop(fe);
    }

    if (cpu_enc && ctx->acoustic_encoder && ctx->semantic_encoder) {
        for (int e = 0; e < 2; e++) {
            float* lat = NULL;
            int fr = 0;
            double best = 1e30;
            for (int r = 0; r < (reps < 3 ? reps : 3); r++) {
                const double t0 = vv_time_ms();
                vv_conv_vae_encode_cpu(e == 0 ? ctx->acoustic_encoder
                                              : ctx->semantic_encoder,
                                       audio, n, &lat, &fr);
                const double ms = vv_time_ms() - t0;
                if (ms < best) best = ms;
                vv_free(lat);
                lat = NULL;
            }
            printf("cpu %-8s %8.1f ms  RTF %.3f\n", e == 0 ? "acoustic" : "semantic",
                   best, best / 1000.0 / sec);
        }
    }

    for (int i = 0; i < MAXW; i++) {
        if (w[i].st) vv_frontend_stream_free(w[i].st);
        if (w[i].rows) vv_dev_free(w[i].rows);
        if (w[i].done) vv_dev_event_destroy(w[i].done);
    }
    vv_inference_free(ctx);
    vv_free(audio);
    return 0;
}
