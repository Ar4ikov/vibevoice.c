/**
 * @file frontend.c
 * @brief Per-device speech front end: encoders, connectors, batching.
 *        See frontend.h.
 */

#include "vibevoice/frontend.h"
#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"

#include <stdlib.h>
#include <string.h>

#include "vv_thread.h"

#define FE_ALIGN 256

static size_t fe_al(size_t v) { return (v + FE_ALIGN - 1) / FE_ALIGN * FE_ALIGN; }

typedef struct fe_req {
    vv_frontend_job_t* job;
    bool               done;
    struct fe_req*     next;
} fe_req_t;

struct vv_frontend_stream {
    vv_frontend_t*  fe;
    vv_vae_state_t* st[2];
};

struct vv_frontend {
    const vv_conv_vae_encoder_t* enc[2];     /**< acoustic, semantic      */
    vv_vae_weights_t*   w[2];
    vv_vae_arena_t*     arena[2];
    vv_connector_dev_t* conn[2];
    int                 vae[2];
    int                 hs;
    vv_frontend_params_t p;
    int                 f_max;               /**< latent rows per launch  */

    void*   s_main;
    void*   s_enc[2];
    void*   ev_in;
    void*   ev_enc[2];
    void*   ev_ac_rows;
    float*  pin[2];                          /**< pinned audio staging    */
    void*   ev_pin[2];
    int     pin_i;
    void*   misc;                            /**< one device allocation   */
    float*  in_f32;
    void*   in_f16;
    void*   lat[2];
    void*   cscr[2];
    size_t  bytes;

    vv_mutex_t lock;
    uint64_t   launches;
    uint64_t   n_jobs;

    /* Service */
    bool        svc_on;
    bool        svc_stop;
    int         svc_device;
    vv_thread_t svc_thread;
    vv_mutex_t  q_lock;
    vv_cond_t   q_cond;
    vv_cond_t   q_done;
    fe_req_t*   q_head;
    fe_req_t*   q_tail;
};

/* ─── Parameters and sizes ──────────────────────────────────────────────── */

vv_frontend_params_t vv_frontend_params_default(void) {
    vv_frontend_params_t p;
    p.max_items = 16;
    p.max_samples = (int64_t)30 * 24000;
    const char* e = getenv("VV_ENC_BATCH_SEC");
    if (e && e[0]) {
        const double sec = atof(e);
        if (sec >= 1.0 && sec <= 3600.0) p.max_samples = (int64_t)(sec * 24000.0);
    }
    return p;
}

static void clamp_params(vv_frontend_params_t* p) {
    if (p->max_items < 1) p->max_items = 1;
    if (p->max_items > VV_VAE_MAX_ITEMS) p->max_items = VV_VAE_MAX_ITEMS;
    if (p->max_samples < 3200) p->max_samples = 3200;
}

static int frames_bound(const vv_conv_vae_encoder_t* e,
                        const vv_frontend_params_t* p) {
    return vv_conv_vae_frames(e, p->max_samples) + 8 * p->max_items;
}

static size_t misc_bytes(const vv_conv_vae_encoder_t* const enc[2],
                         const vv_connector_t* const conn[2],
                         const vv_frontend_params_t* p, int f_max) {
    size_t b = fe_al((size_t)p->max_samples * 4) + fe_al((size_t)p->max_samples * 2);
    for (int e = 0; e < 2; e++) {
        if (!enc[e]) continue;
        b += fe_al((size_t)f_max * (size_t)enc[e]->vae_dim * 2);
        if (conn[e])
            b += 2 * fe_al((size_t)f_max * (size_t)conn[e]->hidden_size * 2);
    }
    return b;
}

size_t vv_frontend_device_bytes(const vv_conv_vae_encoder_t* acoustic,
                                const vv_conv_vae_encoder_t* semantic,
                                const vv_connector_t* ac_conn,
                                const vv_connector_t* sem_conn,
                                const vv_frontend_params_t* params) {
    vv_frontend_params_t p = params ? *params : vv_frontend_params_default();
    clamp_params(&p);
    const vv_conv_vae_encoder_t* enc[2] = { acoustic, semantic };
    const vv_connector_t* conn[2] = { ac_conn, sem_conn };
    size_t b = 0;
    int f_max = 0;
    for (int e = 0; e < 2; e++) {
        if (!enc[e]) continue;
        b += vv_vae_weights_bytes(enc[e]);
        b += vv_vae_arena_bytes(enc[e], p.max_items, p.max_samples);
        if (conn[e]) b += vv_connector_dev_bytes(conn[e]);
        const int f = frames_bound(enc[e], &p);
        if (f > f_max) f_max = f;
    }
    return b + misc_bytes(enc, conn, &p, f_max);
}

size_t vv_frontend_stream_bytes(const vv_conv_vae_encoder_t* acoustic,
                                const vv_conv_vae_encoder_t* semantic) {
    return (acoustic ? vv_vae_state_bytes(acoustic) : 0)
         + (semantic ? vv_vae_state_bytes(semantic) : 0);
}

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

void vv_frontend_free(vv_frontend_t* fe) {
    if (!fe) return;
    vv_frontend_service_stop(fe);
    if (fe->s_main) vv_dev_stream_sync(fe->s_main);
    for (int e = 0; e < 2; e++) {
        if (fe->s_enc[e]) vv_dev_stream_sync(fe->s_enc[e]);
        vv_vae_arena_free(fe->arena[e]);
        vv_vae_weights_free(fe->w[e]);
        vv_connector_dev_free(fe->conn[e]);
        if (fe->ev_enc[e]) vv_dev_event_destroy(fe->ev_enc[e]);
        if (fe->s_enc[e]) vv_dev_stream_destroy(fe->s_enc[e]);
        if (fe->ev_pin[e]) vv_dev_event_destroy(fe->ev_pin[e]);
        if (fe->pin[e]) vv_dev_free_pinned(fe->pin[e]);
    }
    if (fe->misc) vv_dev_free(fe->misc);
    if (fe->ev_in) vv_dev_event_destroy(fe->ev_in);
    if (fe->ev_ac_rows) vv_dev_event_destroy(fe->ev_ac_rows);
    if (fe->s_main) vv_dev_stream_destroy(fe->s_main);
    vv_mutex_destroy(&fe->lock);
    vv_mutex_destroy(&fe->q_lock);
    vv_cond_destroy(&fe->q_cond);
    vv_cond_destroy(&fe->q_done);
    vv_free(fe);
}

vv_status_t vv_frontend_create(const vv_conv_vae_encoder_t* acoustic,
                               const vv_conv_vae_encoder_t* semantic,
                               const vv_connector_t* ac_conn,
                               const vv_connector_t* sem_conn,
                               const vv_frontend_params_t* params,
                               vv_frontend_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!acoustic && !semantic) return VV_ERR_INVALID_ARG;

    vv_frontend_t* fe = (vv_frontend_t*)vv_alloc(sizeof(vv_frontend_t));
    if (!fe) return VV_ERR_OUT_OF_MEMORY;
    memset(fe, 0, sizeof(*fe));
    vv_mutex_init(&fe->lock);
    vv_mutex_init(&fe->q_lock);
    vv_cond_init(&fe->q_cond);
    vv_cond_init(&fe->q_done);
    fe->p = params ? *params : vv_frontend_params_default();
    clamp_params(&fe->p);
    fe->enc[0] = acoustic;
    fe->enc[1] = semantic;
    const vv_connector_t* conn[2] = { ac_conn, sem_conn };

    /* Background priority: a long file's encoder launches must not hold
       up other slots' decode steps, which are latency, not throughput. */
    vv_status_t s = vv_dev_stream_create_background(&fe->s_main);
    if (s == VV_OK) s = vv_dev_event_create(&fe->ev_in);
    if (s == VV_OK) s = vv_dev_event_create(&fe->ev_ac_rows);

    for (int e = 0; e < 2 && s == VV_OK; e++) {
        s = vv_dev_alloc_pinned((void**)&fe->pin[e],
                                (size_t)fe->p.max_samples * sizeof(float));
        if (s == VV_OK) s = vv_dev_event_create(&fe->ev_pin[e]);
        if (s != VV_OK || !fe->enc[e]) continue;
        s = vv_dev_stream_create_background(&fe->s_enc[e]);
        if (s == VV_OK) s = vv_dev_event_create(&fe->ev_enc[e]);
        if (s == VV_OK) s = vv_vae_weights_upload(fe->enc[e], fe->s_main, &fe->w[e]);
        if (s == VV_OK)
            s = vv_vae_arena_create(fe->enc[e], fe->p.max_items,
                                    fe->p.max_samples, &fe->arena[e]);
        if (s == VV_OK && conn[e])
            s = vv_connector_upload(conn[e], fe->s_main, &fe->conn[e]);
        fe->vae[e] = fe->enc[e]->vae_dim;
        if (conn[e]) fe->hs = conn[e]->hidden_size;
        const int f = frames_bound(fe->enc[e], &fe->p);
        if (f > fe->f_max) fe->f_max = f;
    }

    if (s == VV_OK) {
        const size_t mb = misc_bytes(fe->enc, conn, &fe->p, fe->f_max);
        s = vv_dev_alloc(&fe->misc, mb);
        if (s == VV_OK) {
            char* p = (char*)fe->misc;
            fe->in_f32 = (float*)p; p += fe_al((size_t)fe->p.max_samples * 4);
            fe->in_f16 = p;         p += fe_al((size_t)fe->p.max_samples * 2);
            for (int e = 0; e < 2; e++) {
                if (!fe->enc[e]) continue;
                fe->lat[e] = p;
                p += fe_al((size_t)fe->f_max * (size_t)fe->vae[e] * 2);
                if (conn[e]) {
                    fe->cscr[e] = p;
                    p += 2 * fe_al((size_t)fe->f_max * (size_t)conn[e]->hidden_size * 2);
                }
            }
            fe->bytes = mb;
            for (int e = 0; e < 2; e++) {
                if (fe->w[e]) fe->bytes += vv_vae_weights_bytes(fe->enc[e]);
                if (fe->enc[e])
                    fe->bytes += vv_vae_arena_bytes(fe->enc[e], fe->p.max_items,
                                                    fe->p.max_samples);
                if (fe->conn[e]) fe->bytes += fe->conn[e]->bytes;
            }
        }
    }
    if (s == VV_OK) s = vv_dev_stream_sync(fe->s_main);
    if (s != VV_OK) {
        vv_frontend_free(fe);
        return s;
    }
    VV_LOG_I("frontend: speech encoders resident (%.1f MB incl. %.1f MB arena "
             "for %d jobs / %.0f s per launch)",
             (double)fe->bytes / (1024.0 * 1024.0),
             (double)((fe->enc[0] ? vv_vae_arena_bytes(fe->enc[0], fe->p.max_items,
                                                       fe->p.max_samples) : 0) +
                      (fe->enc[1] ? vv_vae_arena_bytes(fe->enc[1], fe->p.max_items,
                                                       fe->p.max_samples) : 0))
                 / (1024.0 * 1024.0),
             fe->p.max_items, (double)fe->p.max_samples / 24000.0);
    *out = fe;
    return VV_OK;
}

size_t vv_frontend_bytes(const vv_frontend_t* fe) { return fe ? fe->bytes : 0; }

int vv_frontend_frames(const vv_frontend_t* fe, int64_t n_samples) {
    if (!fe) return 0;
    const vv_conv_vae_encoder_t* e = fe->enc[0] ? fe->enc[0] : fe->enc[1];
    return vv_conv_vae_frames(e, n_samples);
}

void vv_frontend_stats(const vv_frontend_t* fe, uint64_t* launches,
                       uint64_t* jobs) {
    if (!fe) return;
    vv_mutex_lock((vv_mutex_t*)&fe->lock);
    if (launches) *launches = fe->launches;
    if (jobs) *jobs = fe->n_jobs;
    vv_mutex_unlock((vv_mutex_t*)&fe->lock);
}

vv_status_t vv_frontend_stream_create(vv_frontend_t* fe,
                                      vv_frontend_stream_t** out) {
    if (!fe || !out) return VV_ERR_NULL_PTR;
    vv_frontend_stream_t* st =
        (vv_frontend_stream_t*)vv_alloc(sizeof(vv_frontend_stream_t));
    if (!st) return VV_ERR_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));
    st->fe = fe;
    for (int e = 0; e < 2; e++) {
        if (!fe->w[e]) continue;
        const vv_status_t s = vv_vae_state_create(fe->w[e], &st->st[e]);
        if (s != VV_OK) { vv_frontend_stream_free(st); return s; }
    }
    *out = st;
    return VV_OK;
}

void vv_frontend_stream_reset(vv_frontend_stream_t* st) {
    if (!st) return;
    vv_vae_state_reset(st->st[0]);
    vv_vae_state_reset(st->st[1]);
}

void vv_frontend_stream_free(vv_frontend_stream_t* st) {
    if (!st) return;
    vv_vae_state_free(st->st[0]);
    vv_vae_state_free(st->st[1]);
    vv_free(st);
}

/* ─── One launch ────────────────────────────────────────────────────────── */

typedef struct {
    vv_frontend_job_t* job;
    int64_t off;        /**< first sample of the job in this piece        */
    int64_t len;
    bool    fin;
    int     frames;     /**< produced, skipped ones included              */
    int     skip;
    int     written0;   /**< frames of the job written before this piece  */
    int     row0;       /**< packed latent row (rows jobs)                */
} fe_piece_t;

static vv_status_t connectors(vv_frontend_t* fe, const void* const lat[2],
                              const int lat_ld[2], int n_rows,
                              const vv_vae_rows_t* rows_ac,
                              const vv_vae_rows_t* rows_sem,
                              void* const* waits, int n_waits) {
    vv_status_t s = VV_OK;
    const bool has_ac = fe->conn[0] && lat[0];
    const bool has_sem = fe->conn[1] && lat[1];
    for (int e = 0; e < 2; e++) {
        if (!fe->s_enc[e]) continue;
        for (int i = 0; i < n_waits && s == VV_OK; i++)
            if (waits[i]) s = vv_dev_stream_wait_event(fe->s_enc[e], waits[i]);
    }
    if (s == VV_OK && has_ac) {
        s = vv_connector_forward_dev(fe->conn[0], lat[0], lat_ld[0], n_rows,
                                     fe->cscr[0], rows_ac, fe->s_enc[0]);
        if (s == VV_OK && has_sem)
            s = vv_dev_event_record(fe->ev_ac_rows, fe->s_enc[0]);
    }
    if (s == VV_OK && has_sem) {
        if (has_ac) s = vv_dev_stream_wait_event(fe->s_enc[1], fe->ev_ac_rows);
        if (s == VV_OK)
            s = vv_connector_forward_dev(fe->conn[1], lat[1], lat_ld[1], n_rows,
                                         fe->cscr[1], rows_sem, fe->s_enc[1]);
    }
    return s;
}

/** Record the end of each encoder stream and make s_main wait for both. */
static vv_status_t join(vv_frontend_t* fe) {
    vv_status_t s = VV_OK;
    for (int e = 0; e < 2 && s == VV_OK; e++) {
        if (!fe->s_enc[e]) continue;
        s = vv_dev_event_record(fe->ev_enc[e], fe->s_enc[e]);
        if (s == VV_OK) s = vv_dev_stream_wait_event(fe->s_main, fe->ev_enc[e]);
    }
    return s;
}

static int row_mode(const vv_frontend_t* fe, int e) {
    const bool both = fe->conn[0] && fe->conn[1];
    if (!both) return VV_VAE_ROWS_STORE_TRUNC;
    return e == 0 ? VV_VAE_ROWS_STORE : VV_VAE_ROWS_ACC_TRUNC;
}

static vv_status_t launch(vv_frontend_t* fe, fe_piece_t* pc, int np) {
    /* Everything that can be refused is checked before anything is
       enqueued: a refused launch leaves the device and the states alone. */
    int n_rows = 0;
    int64_t total = 0;
    int64_t poff[VV_VAE_MAX_ITEMS];
    if (np < 1 || np > VV_VAE_MAX_ITEMS) return VV_ERR_INVALID_ARG;
    for (int i = 0; i < np; i++) {
        pc[i].row0 = -1;
        if (pc[i].job->rows) {
            pc[i].row0 = n_rows;
            n_rows += pc[i].frames - pc[i].skip;
        }
        poff[i] = total;
        total += pc[i].len;
    }
    if (n_rows > fe->f_max || total > fe->p.max_samples) return VV_ERR_OVERFLOW;

    /* ── Audio in: host -> pinned -> device FP32 -> FP16, on s_main ── */
    float* pin = fe->pin[fe->pin_i];
    vv_status_t s = vv_dev_event_sync(fe->ev_pin[fe->pin_i]);
    if (s != VV_OK) return s;
    for (int i = 0; i < np; i++)
        if (pc[i].len > 0)
            memcpy(pin + poff[i], pc[i].job->audio + pc[i].off,
                   (size_t)pc[i].len * sizeof(float));
    if (total > 0) {
        s = vv_dev_memcpy_h2d(fe->in_f32, pin, (size_t)total * sizeof(float),
                              fe->s_main);
        if (s == VV_OK) s = vv_dev_event_record(fe->ev_pin[fe->pin_i], fe->s_main);
        if (s == VV_OK)
            s = vv_vae_f32_to_f16_dev(fe->in_f32, fe->in_f16, total, fe->s_main);
        fe->pin_i ^= 1;
    }
    if (s == VV_OK) s = vv_dev_event_record(fe->ev_in, fe->s_main);
    if (s != VV_OK) return s;

    /* ── Both encoders, concurrently ── */
    for (int e = 0; e < 2 && s == VV_OK; e++) {
        if (!fe->w[e]) continue;
        s = vv_dev_stream_wait_event(fe->s_enc[e], fe->ev_in);
        vv_vae_item_t items[VV_VAE_MAX_ITEMS];
        memset(items, 0, sizeof(items));
        for (int i = 0; i < np && s == VV_OK; i++) {
            vv_frontend_job_t* j = pc[i].job;
            void* ljob = e == 0 ? j->ac_latents : j->sem_latents;
            items[i].audio = (const uint16_t*)fe->in_f16 + poff[i];
            items[i].n_samples = (int)pc[i].len;
            items[i].state = j->stream ? j->stream->st[e] : NULL;
            items[i].is_final = pc[i].fin;
            items[i].out_ld = fe->vae[e];
            items[i].skip_frames = pc[i].skip;
            if (j->rows) {
                items[i].out = (uint16_t*)fe->lat[e]
                             + (size_t)pc[i].row0 * fe->vae[e];
            } else if (ljob) {
                if (j->wait_event)
                    s = vv_dev_stream_wait_event(fe->s_enc[e], j->wait_event);
                items[i].out = (uint16_t*)ljob
                             + (size_t)pc[i].written0 * fe->vae[e];
            } else {
                /* Nothing wants this encoder's latents: write them to the
                   scratch rows past the packed ones and drop them. */
                items[i].out = fe->lat[e];
                items[i].skip_frames = pc[i].frames;
            }
        }
        if (s == VV_OK)
            s = vv_vae_encode(fe->w[e], fe->arena[e], items, np, fe->s_enc[e]);
        for (int i = 0; i < np && s == VV_OK; i++) {
            if (items[i].n_frames != pc[i].frames) {
                VV_LOG_E("frontend: encoder produced %d frames, expected %d",
                         items[i].n_frames, pc[i].frames);
                s = VV_ERR_SHAPE_MISMATCH;
            }
        }
    }

    /* ── Connectors, straight into each job's hidden rows ── */
    if (s == VV_OK && n_rows > 0) {
        vv_vae_rows_t rows[2];
        void* waits[VV_VAE_MAX_ITEMS];
        int n_waits = 0;
        for (int e = 0; e < 2; e++) {
            rows[e].mode = row_mode(fe, e);
            rows[e].n = 0;
        }
        for (int i = 0; i < np; i++) {
            vv_frontend_job_t* j = pc[i].job;
            if (!j->rows || pc[i].frames - pc[i].skip <= 0) continue;
            for (int e = 0; e < 2; e++) {
                vv_vae_row_seg_t* g = &rows[e].seg[rows[e].n++];
                g->row0 = pc[i].row0;
                g->ld = j->rows_ld;
                g->dst = (uint16_t*)j->rows + (size_t)pc[i].written0 * j->rows_ld;
            }
            if (j->wait_event) waits[n_waits++] = j->wait_event;
        }
        const void* lat[2] = { fe->w[0] ? fe->lat[0] : NULL,
                               fe->w[1] ? fe->lat[1] : NULL };
        const int ld[2] = { fe->vae[0], fe->vae[1] };
        s = connectors(fe, lat, ld, n_rows, &rows[0], &rows[1], waits, n_waits);
    }
    /* Whatever happened above, s_main -- where every done_event is
       recorded -- waits for all of it, so no waiter is released while a
       kernel of this launch can still write. */
    const vv_status_t sj = join(fe);
    if (s == VV_OK) s = sj;
    fe->launches++;
    return s;
}

/* ─── Scheduling jobs into launches ─────────────────────────────────────── */

/*
 * The unit of work is one launch, not one job. Whoever drives the front end
 * -- the service thread, or a caller of vv_frontend_run -- keeps a set of
 * active jobs with their progress, and before every launch picks what goes
 * into it; fe->lock is held for one launch at a time. So a job that arrives
 * while a long file is being encoded rides along in the long file's next
 * launch instead of waiting for all of it. Cutting a stream at a different
 * place changes no bit of its output (tests/test_vae_stream.c), so the
 * schedule is free to cut wherever it likes.
 */

typedef struct {
    int64_t done;       /**< samples consumed                              */
    int     produced;   /**< frames produced, skipped included             */
    int     written;
    bool    started;
    bool    finished;
} fe_prog_t;

typedef struct {
    vv_frontend_job_t* job;
    fe_prog_t          pg;
    fe_req_t*          req;   /**< service: the submitter to wake          */
} fe_act_t;

static void finish_job(vv_frontend_t* fe, fe_act_t* a, vv_status_t s) {
    vv_frontend_job_t* j = a->job;
    a->pg.finished = true;
    j->status = s;
    j->n_frames = a->pg.written;
    if (s != VV_OK && j->stream) vv_frontend_stream_reset(j->stream);
    if (j->done_event) vv_dev_event_record(j->done_event, fe->s_main);
}

/** Take a job in. Caller holds fe->lock. */
static void admit_job(vv_frontend_t* fe, fe_act_t* a) {
    memset(&a->pg, 0, sizeof(a->pg));
    vv_frontend_job_t* j = a->job;
    j->status = VV_OK;
    j->n_frames = 0;
    fe->n_jobs++;
    if (!j->audio && j->n_samples > 0) {
        finish_job(fe, a, VV_ERR_NULL_PTR);
    } else if (j->n_samples < 0 || (j->stream && j->stream->fe != fe)) {
        finish_job(fe, a, VV_ERR_INVALID_ARG);
    } else if (!j->stream && j->n_samples > fe->p.max_samples) {
        VV_LOG_E("frontend: a stateless window of %lld samples exceeds the "
                 "%lld of one launch", (long long)j->n_samples,
                 (long long)fe->p.max_samples);
        finish_job(fe, a, VV_ERR_OVERFLOW);
    }
}

static void add_piece(vv_frontend_t* fe, fe_act_t* a, int64_t take,
                      fe_piece_t* p) {
    const vv_vae_weights_t* w = fe->w[0] ? fe->w[0] : fe->w[1];
    vv_frontend_job_t* jb = a->job;
    const int64_t rem = jb->n_samples - a->pg.done;
    p->job = jb;
    p->off = a->pg.done;
    p->len = take;
    p->fin = jb->stream ? (jb->is_final && take == rem) : true;
    {
        const vv_vae_state_t* st = jb->stream
            ? jb->stream->st[fe->w[0] ? 0 : 1] : NULL;
        p->frames = vv_vae_frames(w, st, take, p->fin);
    }
    {
        const int want_skip = jb->skip_frames - a->pg.produced;
        p->skip = want_skip <= 0 ? 0
                : (want_skip > p->frames ? p->frames : want_skip);
    }
    p->written0 = a->pg.written;
    a->pg.started = true;
}

/** A job's share of a launch that still has `room` samples free. */
static bool fits(const fe_act_t* a, int64_t room, int64_t* take) {
    const int64_t rem = a->job->n_samples - a->pg.done;
    if (!a->job->stream) {                      /* whole, or not this time */
        *take = rem;
        return rem <= room;
    }
    *take = rem < room ? rem : room;
    return rem <= 0 || *take > 0;
}

/**
 * @brief Fill one launch from the active set. Caller holds fe->lock.
 *
 * Order of service: jobs with less left than the oldest job, shortest
 * first; then the oldest; then the others in arrival order. So a short
 * request that arrives while a long file is being encoded goes out in the
 * very next launch, and equal jobs go first come, first served. The oldest
 * job always keeps a quarter of the launch (all of it when it is alone), so
 * a stream of short arrivals cannot starve a long file. Jobs with nothing
 * left to send are finished here. `ix[i]` is the active index of piece i.
 */
static int plan_launch(vv_frontend_t* fe, fe_act_t* act, int n,
                       fe_piece_t* pc, int* ix) {
    int cand[VV_VAE_MAX_ITEMS];
    int nc = 0;
    int oldest = -1;
    for (int j = 0; j < n; j++) {
        fe_act_t* a = &act[j];
        if (a->pg.finished) continue;
        vv_frontend_job_t* jb = a->job;
        if (jb->n_samples - a->pg.done <= 0) {
            /* Nothing new. Only the end of a stream still has work: the
               right-edge padding of every layer. */
            if (!jb->stream || !jb->is_final || a->pg.started) {
                finish_job(fe, a, VV_OK);
                continue;
            }
        }
        /* One chunk per stream per launch, and in order. */
        bool blocked = false;
        for (int k = 0; k < j && !blocked; k++)
            blocked = !act[k].pg.finished && jb->stream &&
                      act[k].job->stream == jb->stream;
        if (blocked) continue;
        if (oldest < 0) { oldest = j; continue; }
        if (nc < VV_VAE_MAX_ITEMS) cand[nc++] = j;
    }
    if (oldest < 0) return 0;

    const int64_t cap = fe->p.max_samples;
    const int64_t orem = act[oldest].job->n_samples - act[oldest].pg.done;
    const int64_t reserve = act[oldest].job->stream
                          ? (orem < cap / 4 ? orem : cap / 4) : orem;
    int64_t room = cap - reserve;

    /* Jobs with less left than the oldest, shortest first (insertion sort,
       stable; n is at most 32). The rest keep arrival order. */
    int sh[VV_VAE_MAX_ITEMS], lo[VV_VAE_MAX_ITEMS];
    int nsh = 0, nlo = 0;
    for (int c = 0; c < nc; c++) {
        const int v = cand[c];
        const int64_t rv = act[v].job->n_samples - act[v].pg.done;
        if (rv >= orem) { lo[nlo++] = v; continue; }
        int k = nsh - 1;
        while (k >= 0 && act[sh[k]].job->n_samples - act[sh[k]].pg.done > rv) {
            sh[k + 1] = sh[k];
            k--;
        }
        sh[k + 1] = v;
        nsh++;
    }

    int np = 0;
    int64_t take;
    for (int c = 0; c < nsh && np < fe->p.max_items - 1; c++) {
        fe_act_t* a = &act[sh[c]];
        if (!fits(a, room, &take)) continue;
        add_piece(fe, a, take, &pc[np]);
        ix[np++] = sh[c];
        room -= take;
    }
    room += reserve;
    fits(&act[oldest], room, &take);
    add_piece(fe, &act[oldest], take, &pc[np]);
    ix[np++] = oldest;
    room -= take;
    for (int c = 0; c < nlo && np < fe->p.max_items; c++) {
        fe_act_t* a = &act[lo[c]];
        if (!fits(a, room, &take)) continue;
        add_piece(fe, a, take, &pc[np]);
        ix[np++] = lo[c];
        room -= take;
    }
    return np;
}

/** Book one launch's result into the jobs it carried. */
static void commit_launch(vv_frontend_t* fe, fe_act_t* act,
                          const fe_piece_t* pc, const int* ix, int np,
                          vv_status_t s) {
    for (int i = 0; i < np; i++) {
        fe_act_t* a = &act[ix[i]];
        if (a->pg.finished) continue;
        if (s != VV_OK) {
            finish_job(fe, a, s);
            continue;
        }
        a->pg.done += pc[i].len;
        a->pg.produced += pc[i].frames;
        a->pg.written += pc[i].frames - pc[i].skip;
        if (a->pg.done >= a->job->n_samples) finish_job(fe, a, VV_OK);
    }
}

/**
 * @brief One launch over the active set, under fe->lock.
 * @return false when nothing was left to launch.
 */
static bool step(vv_frontend_t* fe, fe_act_t* act, int n, vv_status_t* err) {
    fe_piece_t pc[VV_VAE_MAX_ITEMS];
    int ix[VV_VAE_MAX_ITEMS];
    vv_mutex_lock(&fe->lock);
    const int np = plan_launch(fe, act, n, pc, ix);
    if (np > 0) {
        const vv_status_t s = launch(fe, pc, np);
        commit_launch(fe, act, pc, ix, np, s);
        if (s != VV_OK && err && *err == VV_OK) *err = s;
    }
    vv_mutex_unlock(&fe->lock);
    return np > 0;
}

static vv_status_t run_ptrs(vv_frontend_t* fe, vv_frontend_job_t** jobs, int n) {
    if (n <= 0) return VV_OK;
    fe_act_t stack_act[VV_VAE_MAX_ITEMS];
    fe_act_t* act = n <= VV_VAE_MAX_ITEMS ? stack_act
                  : (fe_act_t*)vv_alloc((size_t)n * sizeof(fe_act_t));
    if (!act) return VV_ERR_OUT_OF_MEMORY;
    vv_status_t first_err = VV_OK;
    vv_mutex_lock(&fe->lock);
    for (int j = 0; j < n; j++) {
        act[j].job = jobs[j];
        act[j].req = NULL;
        admit_job(fe, &act[j]);
        if (act[j].pg.finished && first_err == VV_OK)
            first_err = jobs[j]->status;
    }
    vv_mutex_unlock(&fe->lock);
    /* The planner looks at up to 32 jobs at a time; more go in windows. */
    for (int b = 0; b < n; b += VV_VAE_MAX_ITEMS) {
        const int m = n - b < VV_VAE_MAX_ITEMS ? n - b : VV_VAE_MAX_ITEMS;
        while (step(fe, act + b, m, &first_err)) {}
    }
    if (act != stack_act) vv_free(act);
    return first_err;
}

vv_status_t vv_frontend_run(vv_frontend_t* fe, vv_frontend_job_t* jobs, int n) {
    if (!fe || (!jobs && n > 0)) return VV_ERR_NULL_PTR;
    vv_frontend_job_t* stack_ptrs[VV_VAE_MAX_ITEMS];
    vv_frontend_job_t** ptrs = n <= VV_VAE_MAX_ITEMS ? stack_ptrs
        : (vv_frontend_job_t**)vv_alloc((size_t)n * sizeof(void*));
    if (!ptrs) return VV_ERR_OUT_OF_MEMORY;
    for (int i = 0; i < n; i++) ptrs[i] = &jobs[i];
    const vv_status_t s = run_ptrs(fe, ptrs, n);
    if (ptrs != stack_ptrs) vv_free(ptrs);
    return s;
}

vv_status_t vv_frontend_connect(vv_frontend_t* fe, const void* ac_latents,
                                const void* sem_latents, int frames,
                                void* rows, int rows_ld,
                                void* wait_event, void* done_event) {
    if (!fe || !rows) return VV_ERR_NULL_PTR;
    if (frames <= 0) return VV_OK;
    vv_mutex_lock(&fe->lock);
    vv_status_t s = VV_OK;
    /* The scratch holds f_max rows; the rows are independent, so a long
       clip goes through in pieces with nothing to carry between them. */
    for (int r0 = 0; r0 < frames && s == VV_OK; r0 += fe->f_max) {
        const int nr = frames - r0 < fe->f_max ? frames - r0 : fe->f_max;
        vv_vae_rows_t r[2];
        for (int e = 0; e < 2; e++) {
            r[e].mode = row_mode(fe, e);
            r[e].n = 1;
            r[e].seg[0].row0 = 0;
            r[e].seg[0].ld = rows_ld;
            r[e].seg[0].dst = (uint16_t*)rows + (size_t)r0 * rows_ld;
        }
        const void* lat[2] = {
            ac_latents ? (const uint16_t*)ac_latents + (size_t)r0 * fe->vae[0] : NULL,
            sem_latents ? (const uint16_t*)sem_latents + (size_t)r0 * fe->vae[1] : NULL };
        const int ld[2] = { fe->vae[0], fe->vae[1] };
        void* waits[1] = { wait_event };
        s = connectors(fe, lat, ld, nr, &r[0], &r[1], waits, 1);
    }
    if (s == VV_OK) s = join(fe);
    if (s == VV_OK && done_event) s = vv_dev_event_record(done_event, fe->s_main);
    vv_mutex_unlock(&fe->lock);
    return s;
}

/* ─── Batching service ──────────────────────────────────────────────────── */

/*
 * One worker per device. Between launches it takes in whatever has been
 * submitted since the last one, so concurrent requests share launches, and
 * wakes each submitter as soon as its own job is enqueued. There is no
 * gather window: while one launch runs on the GPU the next is being planned
 * (the pinned double buffer keeps the host at most two launches ahead), and
 * requests that arrive meanwhile join it. A lone request never waits.
 */
static VV_THREAD_RET service_main(void* arg) {
    vv_frontend_t* fe = (vv_frontend_t*)arg;
    fe_act_t act[VV_VAE_MAX_ITEMS];
    int n = 0;
    vv_dev_set_device(fe->svc_device);
    vv_mutex_lock(&fe->q_lock);
    for (;;) {
        fe_req_t* fresh[VV_VAE_MAX_ITEMS];
        int nf = 0;
        while (fe->q_head && n < VV_VAE_MAX_ITEMS) {
            fe_req_t* r = fe->q_head;
            fe->q_head = r->next;
            if (!fe->q_head) fe->q_tail = NULL;
            act[n].job = r->job;
            act[n].req = r;
            fresh[nf++] = r;
            n++;
        }
        if (n == 0) {
            if (fe->svc_stop) break;
            vv_cond_wait(&fe->q_cond, &fe->q_lock);
            continue;
        }
        vv_mutex_unlock(&fe->q_lock);

        if (nf > 0) {
            vv_mutex_lock(&fe->lock);
            for (int i = n - nf; i < n; i++) admit_job(fe, &act[i]);
            vv_mutex_unlock(&fe->lock);
        }
        step(fe, act, n, NULL);

        /* Wake whoever is done and close the gaps, keeping arrival order. */
        vv_mutex_lock(&fe->q_lock);
        int m = 0;
        bool woke = false;
        for (int i = 0; i < n; i++) {
            if (act[i].pg.finished) {
                act[i].req->done = true;
                woke = true;
            } else {
                act[m++] = act[i];
            }
        }
        n = m;
        if (woke) vv_cond_broadcast(&fe->q_done);
    }
    vv_mutex_unlock(&fe->q_lock);
    VV_THREAD_RETURN;
}

vv_status_t vv_frontend_service_start(vv_frontend_t* fe, int device) {
    if (!fe) return VV_ERR_NULL_PTR;
    if (fe->svc_on) return VV_OK;
    fe->svc_device = device;
    fe->svc_stop = false;
    if (!vv_thread_start(&fe->svc_thread, service_main, fe)) return VV_ERR_IO;
    fe->svc_on = true;
    VV_LOG_I("frontend: batching service on gpu %d (up to %d jobs / %.0f s "
             "per launch)", device, fe->p.max_items,
             (double)fe->p.max_samples / 24000.0);
    return VV_OK;
}

void vv_frontend_service_stop(vv_frontend_t* fe) {
    if (!fe || !fe->svc_on) return;
    vv_mutex_lock(&fe->q_lock);
    fe->svc_stop = true;
    vv_cond_broadcast(&fe->q_cond);
    vv_mutex_unlock(&fe->q_lock);
    vv_thread_join(fe->svc_thread);
    fe->svc_on = false;
}

vv_status_t vv_frontend_submit(vv_frontend_t* fe, vv_frontend_job_t* job) {
    if (!fe || !job) return VV_ERR_NULL_PTR;
    if (!fe->svc_on) {
        vv_frontend_job_t* one = job;
        return run_ptrs(fe, &one, 1);
    }
    fe_req_t r;
    r.job = job;
    r.done = false;
    r.next = NULL;
    vv_mutex_lock(&fe->q_lock);
    if (fe->q_tail) fe->q_tail->next = &r;
    else fe->q_head = &r;
    fe->q_tail = &r;
    vv_cond_signal(&fe->q_cond);
    while (!r.done) vv_cond_wait(&fe->q_done, &fe->q_lock);
    vv_mutex_unlock(&fe->q_lock);
    return job->status;
}
