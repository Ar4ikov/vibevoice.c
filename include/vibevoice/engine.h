/**
 * @file engine.h
 * @brief One loaded model, several concurrent transcriptions.
 *
 * The weights are 3.2 GB and read-only, so they are loaded once and shared;
 * what a request actually needs for itself is a KV cache, a workspace and a
 * pair of streams. The engine keeps `n_slots` of those and hands one to each
 * caller, blocking while all are busy.
 *
 * This buys real concurrency for the parts of a request that are not weight
 * reads — audio decoding, the Conv-VAE encoder, prompt building, JSON
 * post-processing — which is most of the wall clock on short clips. Decode
 * itself is bandwidth-bound on the weights, so two simultaneous decodes on
 * one card do not go twice as fast; they interleave.
 *
 * Give it several devices and it puts a replica on each — its own weights,
 * its own slots — and then they do, because nothing is shared between them.
 *
 * Every entry point is thread-safe.
 */
#ifndef VV_ENGINE_H
#define VV_ENGINE_H

#include "vibevoice/types.h"
#include "vibevoice/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vv_engine vv_engine_t;

typedef struct vv_engine_params {
    const char* model_dir;
    int         gpu_id;
    int         n_slots;      /**< Concurrent requests. Default 1.          */
    int         max_seq_len;  /**< Per-slot KV window. Default 32768.       */
    int         kv_format;    /**< vv_kv_format_t                           */
    float       vram_budget;  /**< Fraction of free VRAM. Default 1.0       */
    int         gpu_layers;   /**< Layers on the GPU. -1 = fit to VRAM      */
    bool        cpu_only;
    /**
     * Devices to spread the slots over, each with its own copy of the
     * weights. Empty means the single device named by `gpu_id`.
     */
    vv_gpu_set_t gpus;
} vv_engine_params_t;

static inline vv_engine_params_t vv_engine_params_default(void) {
    vv_engine_params_t p;
    p.model_dir = NULL;
    p.gpu_id = 0;
    p.n_slots = 1;
    p.max_seq_len = 32768;
    p.kv_format = 0;
    p.vram_budget = 1.0f;
    p.gpu_layers = -1;
    p.cpu_only = false;
    p.gpus.n = 0;
    return p;
}

/** @brief Load the model and bring up `n_slots` inference contexts. */
vv_status_t vv_engine_create(const vv_engine_params_t* params,
                             vv_engine_t** out);

void vv_engine_free(vv_engine_t* e);

/**
 * @brief Transcribe one clip. Blocks until a slot frees up, then runs.
 *
 * @param pcm         mono float samples, any sample rate
 * @param n_samples   sample count
 * @param sample_rate source rate; resampled to 24 kHz internally
 * @param perf        optional; filled with this request's metrics
 */
vv_status_t vv_engine_transcribe(vv_engine_t* e,
                                 const float* pcm, int n_samples,
                                 int sample_rate,
                                 const vv_inference_params_t* params,
                                 vv_transcription_t** out,
                                 vv_perf_metrics_t* perf);

/** @brief Slot count, for logging and for sizing a server's thread pool. */
int vv_engine_slots(const vv_engine_t* e);

/** @brief Requests completed and requests currently running. */
void vv_engine_stats(const vv_engine_t* e, uint64_t* completed, int* busy);

/** @brief The model directory this engine was created from. */
const char* vv_engine_model_dir(const vv_engine_t* e);

#ifdef __cplusplus
}
#endif

#endif /* VV_ENGINE_H */
