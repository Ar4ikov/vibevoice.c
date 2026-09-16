/**
 * @file server.h
 * @brief OpenAI-compatible HTTP front end for the engine.
 *
 * Endpoints (the set GPUStack's speech-to-text backend contract expects):
 *
 *   GET  /health            liveness, also /healthz and /v1/health
 *   GET  /v1/models         the loaded model, as an OpenAI model list
 *   GET  /v1/models/{id}    one entry of the same
 *   POST /v1/audio/transcriptions   multipart/form-data, OpenAI audio API
 *   POST /v1/audio/translations     same handler (this model outputs English)
 *   GET  /metrics           Prometheus-style counters
 *
 * The transcription endpoint takes `file` plus the optional `model`,
 * `language`, `prompt` (used as hotwords), `temperature` and
 * `response_format` fields, and answers with json, verbose_json, text, srt
 * or vtt.
 */
#ifndef VV_SERVER_H
#define VV_SERVER_H

#include "vibevoice/types.h"
#include "vibevoice/engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vv_server vv_server_t;

typedef struct vv_server_params {
    const char* host;        /**< Bind address. Default "0.0.0.0".         */
    int         port;        /**< Default 8080.                            */
    int         max_conns;   /**< Connection threads; 0 = slots + queue + 8. */
    int         queue_size;  /**< Waiting transcriptions. Default 16; 0 = reject when busy. */
    const char* model_name;  /**< Name reported by /v1/models.             */
    const char* api_key;     /**< If set, require this bearer token.       */
    size_t      max_body;    /**< Upload cap in bytes. Default 512 MB.     */
    /**
     * vv_acoustic_sampling_t applied to every request. 0 = the mode, which
     * is deterministic; anything else draws a fresh latent per request, so
     * the same audio can come back worded differently.
     */
    int         acoustic_sampling;
} vv_server_params_t;

static inline vv_server_params_t vv_server_params_default(void) {
    vv_server_params_t p;
    p.host = "0.0.0.0";
    p.port = 8080;
    p.max_conns = 0;
    p.queue_size = 16;
    p.model_name = NULL;
    p.api_key = NULL;
    p.max_body = (size_t)512 * 1024 * 1024;
    p.acoustic_sampling = 0;
    return p;
}

/** @brief Bind, listen, and serve until vv_server_stop(). Blocks. */
vv_status_t vv_server_run(vv_engine_t* engine, const vv_server_params_t* params,
                          vv_server_t** handle);

/** @brief Ask a running server to stop; safe from a signal handler. */
void vv_server_stop(vv_server_t* s);

#ifdef __cplusplus
}
#endif

#endif /* VV_SERVER_H */
