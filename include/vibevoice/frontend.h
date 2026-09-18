/**
 * @file frontend.h
 * @brief The speech front end of one device: both Conv-VAE encoders, both
 * connectors, one arena, and an optional batching service.
 *
 * Everything here is per device, not per request. The FP16 weights (1.4 GB
 * of encoders, 0.1 GB of connectors) are uploaded once and shared by every
 * slot on the device; the scratch is one arena sized from the largest batch
 * rather than one allocation per call. A request owns only its streaming
 * state (vv_frontend_stream_t, ~1.4 MB) and two events.
 *
 * One launch takes a batch of jobs — whole files, chunks of live streams,
 * stateless windows — packed along time. The acoustic and semantic encoders
 * run concurrently on two streams; the connectors follow on the same
 * streams and write the prompt's hidden-state rows in place, the semantic
 * one accumulating onto the acoustic one in its GEMM epilogue. Nothing on
 * the way allocates, frees or synchronises the device.
 *
 * With the service started (serve --slots > 1), requests from all slots of
 * the device go through one worker that waits a few milliseconds for more
 * work before launching, so concurrent requests share launches.
 */
#ifndef VV_FRONTEND_H
#define VV_FRONTEND_H

#include "vibevoice/types.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vv_frontend        vv_frontend_t;
typedef struct vv_frontend_stream vv_frontend_stream_t;

typedef struct vv_frontend_params {
    int     max_items;    /**< jobs per launch (<= VV_VAE_MAX_ITEMS)      */
    int64_t max_samples;  /**< 24 kHz samples per launch, all jobs summed */
    int     gather_us;    /**< service: wait this long for more jobs      */
} vv_frontend_params_t;

/**
 * @brief Defaults: 16 jobs, 30 s of audio per launch, 2 ms gather window.
 * VV_ENC_BATCH_SEC overrides the seconds per launch.
 */
vv_frontend_params_t vv_frontend_params_default(void);

/**
 * @brief Device bytes the front end will take, before creating it: weights,
 * connector weights and the arena. What the placement budget reserves.
 */
size_t vv_frontend_device_bytes(const vv_conv_vae_encoder_t* acoustic,
                                const vv_conv_vae_encoder_t* semantic,
                                const vv_connector_t* ac_conn,
                                const vv_connector_t* sem_conn,
                                const vv_frontend_params_t* p);

/** @brief Device bytes of one vv_frontend_stream_t (per slot). */
size_t vv_frontend_stream_bytes(const vv_conv_vae_encoder_t* acoustic,
                                const vv_conv_vae_encoder_t* semantic);

/**
 * @brief Bring the front end up on the current device. Init-time: uploads,
 * allocates, synchronises. Any of the four parts may be NULL; the encoder
 * descriptions and connectors are borrowed and must outlive the front end.
 */
vv_status_t vv_frontend_create(const vv_conv_vae_encoder_t* acoustic,
                               const vv_conv_vae_encoder_t* semantic,
                               const vv_connector_t* ac_conn,
                               const vv_connector_t* sem_conn,
                               const vv_frontend_params_t* p,
                               vv_frontend_t** out);
void vv_frontend_free(vv_frontend_t* fe);

/** @brief Device bytes the front end holds. */
size_t vv_frontend_bytes(const vv_frontend_t* fe);

/** @brief Frames `n_samples` of a fresh stream produce (both encoders agree). */
int vv_frontend_frames(const vv_frontend_t* fe, int64_t n_samples);

/** @brief Per-request streaming state for both encoders. */
vv_status_t vv_frontend_stream_create(vv_frontend_t* fe,
                                      vv_frontend_stream_t** out);
void vv_frontend_stream_reset(vv_frontend_stream_t* st);
void vv_frontend_stream_free(vv_frontend_stream_t* st);

/** @brief One piece of work. */
typedef struct vv_frontend_job {
    const float*          audio;      /**< host, 24 kHz mono               */
    int64_t               n_samples;
    /** Carried context; NULL encodes a stateless window, which must fit
        in one launch. */
    vv_frontend_stream_t* stream;
    bool                  is_final;   /**< last chunk of `stream`          */
    int                   skip_frames;/**< leading frames not written      */

    /* Destination: hidden-state rows (connectors run), or raw latents. */
    void*                 rows;       /**< FP16 [frame][rows_ld] or NULL   */
    int                   rows_ld;
    void*                 ac_latents; /**< FP16 [frame][64] or NULL        */
    void*                 sem_latents;/**< FP16 [frame][128] or NULL       */

    void*                 wait_event; /**< destination ready (may be NULL) */
    void*                 done_event; /**< recorded when all is written    */

    int                   n_frames;   /**< out: frames written             */
    vv_status_t           status;     /**< out                             */
} vv_frontend_job_t;

/**
 * @brief Run jobs now on the calling thread; returns once everything is
 * enqueued (not finished — wait on each job's done_event). Jobs larger than
 * a launch are split, their state carried across the pieces.
 */
vv_status_t vv_frontend_run(vv_frontend_t* fe, vv_frontend_job_t* jobs, int n);

/**
 * @brief Submit one job: through the service when it runs, else directly.
 * Returns once the job is enqueued on the device.
 */
vv_status_t vv_frontend_submit(vv_frontend_t* fe, vv_frontend_job_t* job);

/**
 * @brief Start the batching worker for this device. `device` is the CUDA
 * device the worker binds; the front end must have been created on it.
 */
vv_status_t vv_frontend_service_start(vv_frontend_t* fe, int device);
void vv_frontend_service_stop(vv_frontend_t* fe);

/** @brief Launches and jobs since creation, for logs and benchmarks. */
void vv_frontend_stats(const vv_frontend_t* fe, uint64_t* launches,
                       uint64_t* jobs);

/**
 * @brief Connectors only, over latents the caller already holds (the
 * acoustic draw path): `frames` rows into `rows`. Either latent may be NULL.
 */
vv_status_t vv_frontend_connect(vv_frontend_t* fe, const void* ac_latents,
                                const void* sem_latents, int frames,
                                void* rows, int rows_ld,
                                void* wait_event, void* done_event);

#ifdef __cplusplus
}
#endif

#endif /* VV_FRONTEND_H */
