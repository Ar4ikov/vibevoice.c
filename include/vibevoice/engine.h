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
#include "vibevoice/stream.h"

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
    int          split_mode;  /**< vv_split_mode_t across those devices     */
    int          weight_quant; /**< vv_load_quant_t; 0 = keep the checkpoint's */
    int          attn_backend; /**< vv_attn_backend_t. Default: auto        */
    int          kv_paging;   /**< vv_kv_paging_t. Default: auto (paged when
                                   a device holds more than one slot)       */
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
    p.split_mode = VV_SPLIT_AUTO;
    p.weight_quant = 0;
    p.attn_backend = VV_ATTN_AUTO;
    p.kv_paging = VV_KV_PAGED_AUTO;
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

/* ─── Streaming sessions (chunked models) ───────────────────────────────── */

/** @brief Whether the loaded model transcribes chunk by chunk (Streaming-7B),
 *         so vv_engine_stream_open() applies. */
bool vv_engine_is_streaming(const vv_engine_t* e);

/**
 * @brief A streaming session holding one slot for its whole life.
 *
 * Counts against the slots like a request does: it blocks in open while
 * all are busy, and the slot is back when it closes. PCM arrives at
 * `sample_rate` and is resampled on the fly (sample for sample what a
 * whole-file resample gives); the streaming model is never
 * loudness-normalized.
 */
typedef struct vv_engine_stream vv_engine_stream_t;

/**
 * @param params  hotwords, token cap, callbacks; the geometry and ids come
 *                from the model
 */
vv_status_t vv_engine_stream_open(vv_engine_t* e,
                                  const vv_stream_params_t* params,
                                  int sample_rate, vv_engine_stream_t** out);

/** @brief Push mono PCM at the session's rate; runs every ready chunk. */
vv_status_t vv_engine_stream_push(vv_engine_stream_t* s, const float* pcm,
                                  size_t n);

/** @brief End of audio: the resampler's tail, the padded windows, DONE. */
vv_status_t vv_engine_stream_finish(vv_engine_stream_t* s);

/** @brief Stop at the next token (any thread); see vv_stream_cancel(). */
void vv_engine_stream_cancel(vv_engine_stream_t* s);

/** @brief The underlying session (transcript, stats). */
vv_stream_t* vv_engine_stream_session(vv_engine_stream_t* s);

/** @brief Close the session and give the slot back. NULL is fine. */
void vv_engine_stream_close(vv_engine_stream_t* s);

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
