/**
 * @file engine.c
 * @brief Slot pool over one shared set of model weights. See engine.h.
 */

#include "vibevoice/engine.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"
#include "vibevoice/kv_quant.h"

#include <string.h>

#include "vv_thread.h"

struct vv_engine {
    vv_inference_ctx_t** slots;
    bool*                busy;
    int                  n_slots;
    int                  n_busy;
    uint64_t             completed;

    vv_mutex_t           lock;
    vv_cond_t            slot_free;
    char                 model_dir[512];
};

vv_status_t vv_engine_create(const vv_engine_params_t* params,
                             vv_engine_t** out) {
    if (!params || !params->model_dir || !out) return VV_ERR_NULL_PTR;

    int n = params->n_slots > 0 ? params->n_slots : 1;

    vv_engine_t* e = (vv_engine_t*)vv_alloc(sizeof(vv_engine_t));
    if (!e) return VV_ERR_OUT_OF_MEMORY;
    memset(e, 0, sizeof(*e));
    strncpy(e->model_dir, params->model_dir, sizeof(e->model_dir) - 1);

    vv_init_params_t ip = vv_init_params_default();
    ip.vram_budget = params->vram_budget;
    ip.cpu_only    = params->cpu_only;
    ip.max_seq_len = params->max_seq_len;
    ip.kv_format   = params->kv_format;
    ip.gpu_layers  = params->gpu_layers;

    e->slots = (vv_inference_ctx_t**)vv_alloc((size_t)n * sizeof(void*));
    e->busy  = (bool*)vv_alloc((size_t)n * sizeof(bool));
    if (!e->slots || !e->busy) { vv_engine_free(e); return VV_ERR_OUT_OF_MEMORY; }
    memset(e->slots, 0, (size_t)n * sizeof(void*));
    memset(e->busy, 0, (size_t)n * sizeof(bool));

    vv_status_t s = vv_inference_init(params->model_dir, params->gpu_id,
                                      &ip, &e->slots[0]);
    if (s != VV_OK) { vv_engine_free(e); return s; }
    e->n_slots = 1;

    for (int i = 1; i < n; i++) {
        s = vv_inference_clone(e->slots[0], &ip, &e->slots[i]);
        if (s != VV_OK) {
            /*
             * Running with fewer slots is better than not running: a clone
             * usually fails because the KV caches no longer fit, and the
             * caller only loses concurrency.
             */
            VV_LOG_W("engine: only %d of %d slots available (%s)",
                     i, n, vv_status_str(s));
            break;
        }
        e->n_slots = i + 1;
    }

    /*
     * Two slots writing their token streams to the same stderr produce
     * unreadable interleaved text, so the echo goes quiet as soon as there is
     * more than one. A single slot still streams, which is what makes a long
     * transcription watchable.
     */
    if (e->n_slots > 1)
        for (int i = 0; i < e->n_slots; i++) e->slots[i]->quiet = true;

    vv_mutex_init(&e->lock);
    vv_cond_init(&e->slot_free);

    VV_LOG_I("engine: ready with %d slot(s), kv=%s, max_seq=%d",
             e->n_slots, vv_kv_format_name((vv_kv_format_t)params->kv_format),
             ip.max_seq_len);

    *out = e;
    return VV_OK;
}

void vv_engine_free(vv_engine_t* e) {
    if (!e) return;
    if (e->slots) {
        /* Clones borrow slot 0's weights, so they must go first. */
        for (int i = e->n_slots - 1; i >= 0; i--)
            if (e->slots[i]) vv_inference_free(e->slots[i]);
        vv_free(e->slots);
    }
    if (e->busy) vv_free(e->busy);
    if (e->n_slots > 0) {
        vv_cond_destroy(&e->slot_free);
        vv_mutex_destroy(&e->lock);
    }
    vv_free(e);
}

int vv_engine_slots(const vv_engine_t* e) { return e ? e->n_slots : 0; }

const char* vv_engine_model_dir(const vv_engine_t* e) {
    return e ? e->model_dir : "";
}

void vv_engine_stats(const vv_engine_t* e, uint64_t* completed, int* busy) {
    if (!e) return;
    vv_mutex_lock((vv_mutex_t*)&e->lock);
    if (completed) *completed = e->completed;
    if (busy) *busy = e->n_busy;
    vv_mutex_unlock((vv_mutex_t*)&e->lock);
}

static int acquire_slot(vv_engine_t* e) {
    vv_mutex_lock(&e->lock);
    for (;;) {
        for (int i = 0; i < e->n_slots; i++) {
            if (!e->busy[i]) {
                e->busy[i] = true;
                e->n_busy++;
                vv_mutex_unlock(&e->lock);
                return i;
            }
        }
        vv_cond_wait(&e->slot_free, &e->lock);
    }
}

static void release_slot(vv_engine_t* e, int idx, bool ok) {
    vv_mutex_lock(&e->lock);
    e->busy[idx] = false;
    e->n_busy--;
    if (ok) e->completed++;
    vv_cond_signal(&e->slot_free);
    vv_mutex_unlock(&e->lock);
}

vv_status_t vv_engine_transcribe(vv_engine_t* e,
                                 const float* pcm, int n_samples,
                                 int sample_rate,
                                 const vv_inference_params_t* params,
                                 vv_transcription_t** out,
                                 vv_perf_metrics_t* perf) {
    if (!e || !pcm || !out) return VV_ERR_NULL_PTR;
    if (n_samples <= 0) return VV_ERR_INVALID_ARG;

    /* Resample and normalize outside the slot: it needs no GPU. */
    float* audio = NULL;
    int n_out = 0;
    vv_status_t s = vv_audio_prepare(pcm, n_samples, sample_rate,
                                     &audio, &n_out);
    if (s != VV_OK) return s;

    const int slot = acquire_slot(e);
    s = vv_inference_transcribe(e->slots[slot], audio, n_out, params, out);
    if (perf) *perf = e->slots[slot]->last_perf;
    release_slot(e, slot, s == VV_OK);

    vv_free(audio);
    return s;
}
