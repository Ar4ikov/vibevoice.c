/**
 * @file engine.c
 * @brief Slot pool over one shared set of model weights. See engine.h.
 */

#include "vibevoice/engine.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/frontend.h"

#include <string.h>

#include "vv_thread.h"

struct vv_engine {
    vv_inference_ctx_t** slots;
    bool*                busy;
    int                  n_slots;
    int                  n_busy;
    uint64_t             completed;
    int                  n_replicas;  /**< devices holding a copy of the weights */

    vv_mutex_t           lock;
    vv_cond_t            slot_free;
    char                 model_dir[512];
};

/**
 * @brief Bring up `want` slots on one device, the first owning the weights.
 *
 * Returns how many were actually made. Running with fewer slots beats not
 * running: a clone usually fails because the KV caches stopped fitting, and
 * the caller only loses concurrency.
 */
static int fill_device(vv_engine_t* e, const char* model_dir, int gpu_id,
                       vv_init_params_t* ip, int want) {
    const int base = e->n_slots;
    /* The first context sizes a shared KV pool for every slot it will
     * lend it to. */
    ip->n_slots = want;
    vv_status_t s = vv_inference_init(model_dir, gpu_id, ip, &e->slots[base]);
    if (s != VV_OK) {
        VV_LOG_W("engine: gpu %d unusable (%s)", gpu_id, vv_status_str(s));
        return 0;
    }
    e->n_slots = base + 1;

    int made = want;
    for (int i = 1; i < want; i++) {
        s = vv_inference_clone(e->slots[base], ip, &e->slots[base + i]);
        if (s != VV_OK) {
            VV_LOG_W("engine: gpu %d holds %d of %d slot(s) (%s)",
                     gpu_id, i, want, vv_status_str(s));
            made = i;
            break;
        }
        e->n_slots = base + i + 1;
    }

    /*
     * Every slot on the device shares one speech front end. With more than
     * one slot, its work goes through a worker that plans one launch at a
     * time from whatever has arrived: concurrent requests share the
     * encoder's kernel launches, a short request is not stuck behind a long
     * file's whole encode, and the arena is sized once per device, not per
     * slot.
     */
    if (made > 1 && e->slots[base]->frontend)
        vv_frontend_service_start(e->slots[base]->frontend, gpu_id);
    return made;
}

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
    ip.weight_quant = params->weight_quant;
    ip.smooth      = params->smooth;
    ip.weights_source = params->weights_source;
    ip.vae_numerics = params->vae_numerics;
    ip.head_format = params->head_format;
    ip.gpus        = params->gpus;
    if (ip.gpus.n == 0) { ip.gpus.n = 1; ip.gpus.id[0] = params->gpu_id; }

    e->slots = (vv_inference_ctx_t**)vv_alloc((size_t)n * sizeof(void*));
    e->busy  = (bool*)vv_alloc((size_t)n * sizeof(bool));
    if (!e->slots || !e->busy) { vv_engine_free(e); return VV_ERR_OUT_OF_MEMORY; }
    memset(e->slots, 0, (size_t)n * sizeof(void*));
    memset(e->busy, 0, (size_t)n * sizeof(bool));

    /*
     * One replica per device, each with its own copy of the weights and the
     * slots spread evenly over them. The model is 3.2 GB against 24 GB of
     * card, so a second device is not there for capacity — it is there
     * because a slot's KV cache is 1.8 GB and because two decodes on one
     * card interleave on the same weight bandwidth while two on separate
     * cards do not. Nothing crosses between them, which is why this scales
     * and layer sharding (#7) does not.
     */
    int devices = params->cpu_only ? 1 : ip.gpus.n;

    /*
     * Auto: replicate while there is a slot for every device, shard when
     * there is not. More devices than slots means replicas that would sit
     * idle, and a single-request caller (n_slots == 1) is exactly that case
     * — the only way a second card helps it is by holding some of the model.
     */
    vv_split_mode_t mode = (vv_split_mode_t)params->split_mode;
    if (devices > 1 && mode == VV_SPLIT_AUTO)
        mode = (n >= devices) ? VV_SPLIT_REPLICA : VV_SPLIT_LAYER;
    ip.split_mode = (int)mode;
    ip.attn_backend = params->attn_backend;
    ip.kv_paging = params->kv_paging;
    if (devices > 1)
        VV_LOG_I("engine: %d devices, %s split", devices,
                 vv_split_mode_name(mode));

    if (devices > 1 && mode == VV_SPLIT_LAYER) {
        /* One model spread over every device; the slots clone it. */
        if (fill_device(e, params->model_dir, ip.gpus.id[0], &ip, n) > 0)
            e->n_replicas = 1;
    } else {
        for (int d = 0; d < devices && e->n_slots < n; d++) {
            /* Spread the remainder over the first devices, not the last. */
            int want = n / devices + (d < n % devices ? 1 : 0);
            if (want <= 0) continue;
            if (e->n_slots + want > n) want = n - e->n_slots;
            /* A replica owns one device, so it is capped by that one. */
            ip.gpus.n = 1;
            ip.gpus.id[0] = params->gpus.n ? params->gpus.id[d] : params->gpu_id;
            ip.gpus.cap[0] = params->gpus.n ? params->gpus.cap[d]
                                            : params->gpus.cap[0];
            if (fill_device(e, params->model_dir, ip.gpus.id[0], &ip, want) > 0)
                e->n_replicas++;
        }
        ip.gpus = params->gpus;
    }
    if (e->n_slots == 0) { vv_engine_free(e); return VV_ERR_CUDA; }

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

    if (e->n_replicas > 1)
        VV_LOG_I("engine: ready with %d slot(s) over %d device(s), kv=%s, "
                 "max_seq=%d", e->n_slots, e->n_replicas,
                 vv_kv_format_name((vv_kv_format_t)params->kv_format),
                 ip.max_seq_len);
    else
        VV_LOG_I("engine: ready with %d slot(s), kv=%s, max_seq=%d",
                 e->n_slots,
                 vv_kv_format_name((vv_kv_format_t)params->kv_format),
                 ip.max_seq_len);

    *out = e;
    return VV_OK;
}

void vv_engine_free(vv_engine_t* e) {
    if (!e) return;
    if (e->slots) {
        /* A clone borrows its device's first slot, so it must go first. */
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

/* A free slot, waiting for one for at most `wait_ms` (< 0: for as long as it
 * takes). -1 when none freed up in time. */
static int acquire_slot_wait(vv_engine_t* e, int wait_ms) {
    const double deadline = wait_ms >= 0 ? vv_time_ms() + wait_ms : 0.0;
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
        if (wait_ms < 0) {
            vv_cond_wait(&e->slot_free, &e->lock);
            continue;
        }
        const double left = deadline - vv_time_ms();
        if (left <= 0.0) {
            vv_mutex_unlock(&e->lock);
            return -1;
        }
        vv_cond_timedwait(&e->slot_free, &e->lock, (int)left + 1);
    }
}

static int acquire_slot(vv_engine_t* e) { return acquire_slot_wait(e, -1); }

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
    /* Every slot runs the same model, so any of them knows its family. */
    const bool normalize = e->slots[0]->family_ok
                           ? e->slots[0]->family.normalize_audio : true;
    const bool vibeasr = e->slots[0]->family_ok &&
                         e->slots[0]->family.vibeasr_audio;
    vv_status_t s = vibeasr
        ? vv_audio_prepare_vibeasr(pcm, n_samples, sample_rate, normalize,
                                   &audio, &n_out)
        : vv_audio_prepare_ex(pcm, n_samples, sample_rate, normalize,
                              &audio, &n_out);
    if (s != VV_OK) return s;

    const int slot = acquire_slot(e);
    s = vv_inference_transcribe(e->slots[slot], audio, n_out, params, out);
    if (perf) *perf = e->slots[slot]->last_perf;
    release_slot(e, slot, s == VV_OK);

    vv_free(audio);
    return s;
}

vv_status_t vv_engine_generate(vv_engine_t* e,
                               const vv_generate_params_t* params,
                               vv_generation_t** out,
                               vv_perf_metrics_t* perf) {
    if (!e || !params || !out) return VV_ERR_NULL_PTR;

    const int slot = acquire_slot(e);
    const vv_status_t s = vv_inference_generate(e->slots[slot], params, out);
    if (perf) *perf = e->slots[slot]->last_perf;
    release_slot(e, slot, s == VV_OK);
    return s;
}

/* ─── Streaming sessions ────────────────────────────────────────────────── */

struct vv_engine_stream {
    vv_engine_t*    e;
    int             slot;
    vv_stream_t*    st;
    vv_resampler_t* rs;
    bool            ok;
};

bool vv_engine_is_streaming(const vv_engine_t* e) {
    return e && e->n_slots > 0 && e->slots[0]->family_ok &&
           e->slots[0]->family.mode == VV_GEN_CHUNKED;
}

vv_status_t vv_engine_stream_open(vv_engine_t* e,
                                  const vv_stream_params_t* params,
                                  int sample_rate, vv_engine_stream_t** out) {
    return vv_engine_stream_open_ex(e, params, sample_rate, -1, out);
}

vv_status_t vv_engine_stream_open_ex(vv_engine_t* e,
                                     const vv_stream_params_t* params,
                                     int sample_rate, int wait_ms,
                                     vv_engine_stream_t** out) {
    if (!e || !params || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!vv_engine_is_streaming(e)) return VV_ERR_UNSUPPORTED;
    if (sample_rate <= 0) return VV_ERR_INVALID_ARG;

    vv_engine_stream_t* s = (vv_engine_stream_t*)vv_alloc(sizeof(*s));
    if (!s) return VV_ERR_OUT_OF_MEMORY;
    memset(s, 0, sizeof(*s));
    s->e = e;
    s->slot = -1;

    const int target = e->slots[0]->family.sample_rate > 0
                       ? e->slots[0]->family.sample_rate : 24000;
    vv_status_t st = vv_resampler_create(sample_rate, target, &s->rs);
    if (st != VV_OK) { vv_free(s); return st; }

    s->slot = acquire_slot_wait(e, wait_ms);
    if (s->slot < 0) {
        vv_engine_stream_close(s);
        return VV_ERR_BUSY;
    }
    vv_inference_ctx_t* ctx = e->slots[s->slot];
    memset(&ctx->last_perf, 0, sizeof(ctx->last_perf));
    if (ctx->use_gpu) st = vv_dev_set_device(ctx->gpu_id);
    if (st == VV_OK) st = vv_stream_open(ctx, params, &s->st);
    if (st != VV_OK) {
        vv_engine_stream_close(s);
        return st;
    }
    *out = s;
    return VV_OK;
}

vv_status_t vv_engine_stream_push(vv_engine_stream_t* s, const float* pcm,
                                  size_t n) {
    if (!s) return VV_ERR_NULL_PTR;
    const float* o = NULL;
    size_t no = 0;
    vv_status_t st = vv_resampler_push(s->rs, pcm, n, &o, &no);
    if (st != VV_OK) return st;
    return vv_stream_push(s->st, o, no);
}

vv_status_t vv_engine_stream_finish(vv_engine_stream_t* s) {
    if (!s) return VV_ERR_NULL_PTR;
    const float* o = NULL;
    size_t no = 0;
    vv_status_t st = vv_resampler_finish(s->rs, &o, &no);
    if (st == VV_OK && no) st = vv_stream_push(s->st, o, no);
    if (st == VV_OK) st = vv_stream_finish(s->st);
    if (st == VV_OK) s->ok = true;
    return st;
}

void vv_engine_stream_cancel(vv_engine_stream_t* s) {
    if (s) vv_stream_cancel(s->st);
}

vv_stream_t* vv_engine_stream_session(vv_engine_stream_t* s) {
    return s ? s->st : NULL;
}

void vv_engine_stream_close(vv_engine_stream_t* s) {
    if (!s) return;
    vv_stream_close(s->st);
    vv_resampler_free(s->rs);
    if (s->slot >= 0) release_slot(s->e, s->slot, s->ok);
    vv_free(s);
}
