/**
 * @file stream_ctx.c
 * @brief The streaming session's backend over a real inference context.
 *
 * A session owns its context from open to close. Per session:
 *
 *  - the prompt is prefilled into a reset cache;
 *  - each window goes through the device's speech front end as a stateless
 *    job (no encoder state, exactly what upstream does per window), and the
 *    connectors write the features straight into the chunk's rows, after
 *    the token rows of the same buffer have been embedded;
 *  - the chunk -- `[<|text_chunk_end|>] <|object_ref_start|> 26 features
 *    <|object_ref_end|>` -- is prefilled at the cache's current length, the
 *    last row goes through the final norm and the head, and the argmax is
 *    the chunk's first token;
 *  - greedy steps run on the captured decode step. The captures read the
 *    position from the device, so they are made once per launch shape and
 *    survive every prefill in between: a session re-captures a couple of
 *    dozen times over half an hour of audio, not once per chunk;
 *  - with a drafter (ctx->spec) the steps go a block at a time instead
 *    (decode_block): the drafter's context follows every prefill through
 *    its taps, so the session has to keep it in step -- one plain step
 *    feeds a token the drafter never saw, and blocks are off from there.
 *
 * All device buffers are allocated at open, sized for the prompt as well as
 * a chunk; nothing on the per-chunk path allocates on the device. A closed
 * session parks them on its context for the next one (ctx->stream_bufs), so
 * in `serve` opening and closing sessions does not cudaMalloc or cudaFree --
 * either of which would stall every other live session on the card. The
 * CPU backend is the same sequence over the host kernels.
 *
 * A shared KV pool: the session maps pages for `kv_reserve_sec` of audio
 * when it opens (admission -- a pool that cannot cover them refuses the
 * session), then a page at a time; with `kv_no_wait` a pool that has run
 * dry ends the session instead of blocking it behind other live ones.
 */

#include "vibevoice/stream.h"
#include "vibevoice/inference.h"
#include "vibevoice/frontend.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/text_tokenizer.h"
#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/spec.h"

#include "pipeline_internal.h"
#include "spec_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Text a chunk decodes to, in cache positions, for sizing a reservation:
 * test120 averages 10 tokens per 2.93 s chunk, dense speech 12-14; the
 * rest of a chunk is its 29 prefill rows.
 */
#define KV_TEXT_PER_CHUNK 16

typedef struct {
    vv_inference_ctx_t* ctx;
    bool     gpu;
    int      hs;
    int      vocab;
    int      rows_cap;       /* rows the hidden buffer holds */
    int32_t  pad_id;         /* id embedded under a feature row */
    bool     profile;        /* VV_STREAM_PROFILE: split encode from prefill */
    int      kv_reserve;     /* positions mapped at open (admission) */
    bool     kv_no_wait;     /* a dry pool ends the session, never blocks it */

    /* GPU */
    void*    hidden;         /* [rows_cap][hs] FP16 */
    int32_t* ids_dev;        /* [rows_cap] */
    int32_t* ids_pin;        /* pinned [rows_cap] */
    int32_t* tok_pin;        /* pinned, one token */
    void*    normed;         /* [hs] FP16 */
    void*    hidden_one;     /* [hs] FP16 */
    void*    logits;         /* [vocab] FP32 */
    void*    am_v;
    void*    am_i;
    void*    tok_dev;        /* the current token, written by the argmax */
    int32_t  tok_dev_id;     /* what tok_dev holds, -1 when unknown */
    uint16_t* host_h;        /* head on the host: one row each */
    float*   host_f;
    bool     embed_on_cpu;
    bool     head_on_cpu;
    vv_graph_slot_t graphs[VV_MAX_GPUS];
    bool     graph_ok;
    /* Windows encoded ahead of their prefill (encode_ahead). */
    void*    ahead;          /* [AHEAD_MAX][frames][hs] FP16 */
    void*    ahead_ev[VV_STREAM_AHEAD_MAX];
    int64_t  ahead_first;
    int      ahead_n;
    int      ahead_frames;

    /* CPU */
    float*   hid32;          /* [rows_cap][hs] */
    float*   one32;          /* [hs] */
    float*   norm32;         /* [hs] */

    /* Speculative decoding: the context's drafter, live from the prompt
     * until a plain step feeds a token it has not seen. */
    vv_spec_t*       spec;
    bool             spec_live;

    /* A replay for vv_spec_trace(): every prefilled row's token and role,
     * and what stopped each chunk of the run being replayed. */
    vv_spec_trace_t* trace;
    const int32_t*   trace_stops;
    int              trace_chunks;
    vv_status_t      trace_err;
} sbe_t;

/* Record `n` prefilled rows in the trace, positions kv_len - n .. */
static void trace_rows(sbe_t* b, const int32_t* ids, int n, int kind_first,
                       int kind_rest) {
    vv_spec_trace_t* t = b->trace;
    if (!t || b->trace_err != VV_OK) return;
    if (t->n + n > t->cap) { b->trace_err = VV_ERR_OVERFLOW; return; }
    for (int i = 0; i < n; i++) {
        t->ids[t->n + i] = ids[i];
        t->kind[t->n + i] = (uint8_t)(i == 0 ? kind_first : kind_rest);
    }
    t->n += n;
    /* The taps put position p at row p; the two counts must agree. */
    if (b->ctx->kv_cache && t->n != b->ctx->kv_cache->current_len)
        b->trace_err = VV_ERR_SHAPE_MISMATCH;
}

/* ─── helpers ───────────────────────────────────────────────────────────── */

static void dump_f16_rows(const char* name, sbe_t* b, const void* dev,
                          int rows) {
    if (!vv_debug_dump_dir() || rows <= 0) return;
    const size_t n = (size_t)rows * (size_t)b->hs;
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    float* f = (float*)vv_alloc(n * sizeof(float));
    if (h && f && vv_dev_stream_sync(b->ctx->compute_stream) == VV_OK &&
        vv_dev_memcpy_d2h(h, dev, n * 2, b->ctx->compute_stream) == VV_OK) {
        for (size_t i = 0; i < n; i++) f[i] = vv_half_to_float(h[i]);
        vv_debug_dump(name, f, n * sizeof(float));
    }
    vv_free(h);
    vv_free(f);
}

static void dump_logits_dev(sbe_t* b, int64_t idx) {
    if (!vv_debug_dump_dir() || !b->logits) return;
    float* lg = (float*)vv_alloc((size_t)b->vocab * sizeof(float));
    if (lg && vv_dev_memcpy_d2h(lg, b->logits, (size_t)b->vocab * sizeof(float),
                                b->ctx->compute_stream) == VV_OK) {
        char name[64];
        snprintf(name, sizeof(name), "stream_c%04lld_logits", (long long)idx);
        vv_debug_dump(name, lg, (size_t)b->vocab * sizeof(float));
    }
    vv_free(lg);
}

static vv_status_t bind(sbe_t* b) {
    return b->gpu ? vv_dev_set_device(b->ctx->gpu_id) : VV_OK;
}

/* ─── backend ops: text ─────────────────────────────────────────────────── */

static vv_status_t be_encode_text(void* self, const char* text,
                                  int32_t** ids, int* n) {
    sbe_t* b = (sbe_t*)self;
    return vv_tokenizer_encode(b->ctx->tokenizer, text, ids, n);
}

static vv_status_t be_token_bytes(void* self, int32_t id, char* buf,
                                  size_t cap, size_t* n) {
    sbe_t* b = (sbe_t*)self;
    return vv_stream_token_bytes(b->ctx->tokenizer, id, buf, cap, n);
}

static int64_t be_kv_len(void* self) {
    sbe_t* b = (sbe_t*)self;
    return b->ctx->kv_cache ? b->ctx->kv_cache->current_len : 0;
}

static int64_t be_kv_capacity(void* self) {
    sbe_t* b = (sbe_t*)self;
    return b->ctx->kv_cache ? b->ctx->kv_cache->max_seq_len : 0;
}

/* ─── GPU ───────────────────────────────────────────────────────────────── */

static void gpu_free_rows(sbe_t* b) {
    if (b->hidden) vv_dev_free(b->hidden);
    if (b->ids_dev) vv_dev_free(b->ids_dev);
    if (b->ids_pin) vv_dev_free_pinned(b->ids_pin);
    b->hidden = NULL;
    b->ids_dev = NULL;
    b->ids_pin = NULL;
    b->rows_cap = 0;
}

/* Room for `rows` rows. Only the prompt can outgrow the chunk size, and
 * only once per session, before any chunk. */
static vv_status_t gpu_rows(sbe_t* b, int rows) {
    if (rows <= b->rows_cap) return VV_OK;
    vv_dev_stream_sync(b->ctx->compute_stream);
    gpu_free_rows(b);
    vv_status_t s = vv_dev_alloc(&b->hidden, (size_t)rows * (size_t)b->hs * 2);
    if (s == VV_OK)
        s = vv_dev_alloc((void**)&b->ids_dev, (size_t)rows * sizeof(int32_t));
    if (s == VV_OK)
        s = vv_dev_alloc_pinned((void**)&b->ids_pin,
                                (size_t)rows * sizeof(int32_t));
    if (s != VV_OK) { gpu_free_rows(b); return s; }
    b->rows_cap = rows;
    return VV_OK;
}

/* Embed `ids` into rows 0..n-1 of the hidden buffer, on the compute stream. */
static vv_status_t gpu_embed_rows(sbe_t* b, const int32_t* ids, int n) {
    vv_inference_ctx_t* ctx = b->ctx;
    const size_t row = (size_t)b->hs * 2;
    if (b->embed_on_cpu) {
        /* One FP16 row of the host table is exactly the embedding. */
        const uint8_t* tab = (const uint8_t*)ctx->model->embed_tokens.data;
        for (int r = 0; r < n; r++) {
            if (ids[r] < 0 || ids[r] >= b->vocab) return VV_ERR_INVALID_ARG;
            vv_status_t s = vv_dev_memcpy_h2d((uint8_t*)b->hidden + r * row,
                                              tab + (size_t)ids[r] * row, row,
                                              ctx->compute_stream);
            if (s != VV_OK) return s;
        }
        return VV_OK;
    }
    memcpy(b->ids_pin, ids, (size_t)n * sizeof(int32_t));
    vv_status_t s = vv_dev_memcpy_h2d(b->ids_dev, b->ids_pin,
                                      (size_t)n * sizeof(int32_t),
                                      ctx->compute_stream);
    if (s != VV_OK) return s;
    return vv_embedding_dev(ctx->embed_table_gpu, b->ids_dev, b->hidden, n,
                            b->hs, ctx->compute_stream);
}

/* Final norm + head + argmax of one hidden row; the token lands in tok_dev
 * (device head) and in *tok. */
static vv_status_t gpu_head(sbe_t* b, const void* row, int32_t* tok) {
    const vv_status_t s = vv_pipeline_head_argmax(
        b->ctx, row, b->normed, b->logits, b->am_v, b->am_i, b->tok_dev,
        b->tok_pin, b->host_h, b->host_f, tok);
    /* A device head leaves the token on the device for the next embed. */
    b->tok_dev_id = (s == VV_OK && !b->head_on_cpu) ? *tok : -1;
    return s;
}

/* A prefill at the cache's length, the drafter following it when live. */
static vv_status_t prefill_rows(sbe_t* b, int n) {
    vv_inference_ctx_t* ctx = b->ctx;
    if (!b->spec_live) return vv_pipeline_prefill(ctx, b->hidden, n);
    ctx->taps = vv_spec_prefill_taps(b->spec);
    const vv_status_t s = vv_pipeline_prefill(ctx, b->hidden, n);
    ctx->taps = NULL;
    return s;
}

/* Every cache of the context (primary and shards) waits for pages or not. */
static void set_kv_no_wait(vv_inference_ctx_t* ctx, bool on) {
    if (ctx->kv_cache) ctx->kv_cache->no_wait = on;
    for (int i = 0; i < ctx->n_shards; i++)
        if (ctx->shards[i].kv_cache) ctx->shards[i].kv_cache->no_wait = on;
}

static vv_status_t gpu_prefill_prompt(void* self, const int32_t* ids, int n) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    vv_pipeline_kv_reset(ctx);
    set_kv_no_wait(ctx, b->kv_no_wait);
    if (b->spec) {
        vv_spec_reset(b->spec);
        b->spec_live = true;
    }
    s = gpu_rows(b, n);
    if (s == VV_OK) {
        /* Admission: the reservation now, all of it, or no session. */
        const int want = n > b->kv_reserve ? n : b->kv_reserve;
        s = vv_pipeline_kv_reserve(ctx, want);
        if (s == VV_ERR_KV_POOL_EXHAUSTED && want > n)
            VV_LOG_W("stream: the shared KV pool cannot reserve %d positions "
                     "for another session; refusing it", want);
    }
    if (s == VV_OK) s = gpu_embed_rows(b, ids, n);
    if (s == VV_OK) s = prefill_rows(b, n);
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    if (s == VV_OK) trace_rows(b, ids, n, VV_TRACE_CONTEXT, VV_TRACE_CONTEXT);
    return s;
}

/* Known tokens at the current length, no head: a replay's chunk text. */
static vv_status_t gpu_prefill_tokens(void* self, const int32_t* ids, int n) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK || n <= 0) return s;
    if (b->trace_err != VV_OK) return b->trace_err;
    s = gpu_rows(b, n);
    if (s == VV_OK)
        s = vv_pipeline_kv_reserve(ctx, ctx->kv_cache->current_len + n + 1);
    if (s == VV_OK) s = gpu_embed_rows(b, ids, n);
    if (s == VV_OK) s = vv_pipeline_prefill(ctx, b->hidden, n);
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    if (s == VV_OK) trace_rows(b, ids, n, VV_TRACE_GEN, VV_TRACE_GEN);
    /* The next chunk's token rows must not be mistaken for these. */
    b->tok_dev_id = -1;
    return s == VV_OK ? b->trace_err : s;
}

static vv_status_t gpu_prefill_chunk(void* self, const vv_stream_chunk_t* ch,
                                     const float* window, int32_t* first) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    const int n = ch->n_rows;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    if (n > b->rows_cap) return VV_ERR_OVERFLOW;   /* sized at open */

    /* Pages for the rows plus the chunk end that will follow them; decode
     * steps take theirs one at a time (step_slice). */
    s = vv_pipeline_kv_reserve(ctx, ctx->kv_cache->current_len + n + 1);
    if (s == VV_ERR_KV_POOL_EXHAUSTED && b->kv_no_wait)
        VV_LOG_W("stream: the shared KV pool has no page left for this "
                 "session at %d positions; ending it rather than stalling "
                 "it behind the other live sessions",
                 ctx->kv_cache->current_len);
    if (s != VV_OK) return s;

    int32_t ids[64];
    for (int r = 0; r < n; r++)
        ids[r] = ch->rows[r] >= 0 ? ch->rows[r] : b->pad_id;
    s = gpu_embed_rows(b, ids, n);
    if (s != VV_OK) return s;
    /* The front end's row writes wait for the embedding, not for the host. */
    s = vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);
    if (s != VV_OK) return s;

    void* rows = (uint8_t*)b->hidden + (size_t)ch->feat_offset * b->hs * 2;
    const size_t feat_bytes = (size_t)ch->n_frames * (size_t)b->hs * 2;
    if (b->ahead_n > 0 && ch->index >= b->ahead_first &&
        ch->index < b->ahead_first + b->ahead_n &&
        ch->n_frames == b->ahead_frames) {
        /* Encoded already, with the windows around it. */
        const size_t k = (size_t)(ch->index - b->ahead_first);
        s = vv_dev_memcpy_d2d(rows, (const uint8_t*)b->ahead + k * feat_bytes,
                              feat_bytes, ctx->compute_stream);
        if (s != VV_OK) return s;
    } else {
        const double t0 = vv_time_ms();
        vv_frontend_job_t job;
        memset(&job, 0, sizeof(job));
        job.audio = window;
        job.n_samples = (int64_t)(ch->n_frames) * ctx->family.frame_samples;
        job.stream = NULL;                 /* a stateless window */
        job.is_final = true;
        job.rows = rows;
        job.rows_ld = b->hs;
        job.wait_event = ctx->fe_ready;
        job.done_event = ctx->fe_done;
        s = vv_frontend_submit(ctx->frontend, &job);
        if (s == VV_OK && job.n_frames != ch->n_frames) {
            VV_LOG_E("stream: the window encoded to %d frames, the chunk has "
                     "%d", job.n_frames, ch->n_frames);
            s = VV_ERR_SHAPE_MISMATCH;
        }
        if (s == VV_OK && b->profile) {
            s = vv_dev_event_sync(ctx->fe_done);
            *ch->encode_ms = vv_time_ms() - t0;
        }
        if (s == VV_OK) s = vv_dev_stream_wait_event(ctx->compute_stream,
                                                     ctx->fe_done);
        if (s != VV_OK) return s;
    }

    if (vv_debug_dump_dir()) {
        char name[64];
        snprintf(name, sizeof(name), "stream_c%04lld_feats",
                 (long long)ch->index);
        dump_f16_rows(name, b, rows, ch->n_frames);
    }

    s = prefill_rows(b, n);
    if (s != VV_OK) return s;
    s = gpu_head(b, (const uint8_t*)b->hidden + (size_t)(n - 1) * b->hs * 2,
                 first);
    if (s != VV_OK) return s;
    /*
     * A folded chunk end is what the previous chunk predicted last -- a
     * label -- when that chunk did stop on it (not on EOS or the cap).
     */
    {
        const int32_t tce = ctx->family.tok.text_chunk_end;
        const bool label = ch->rows[0] == tce && b->trace_stops &&
                           ch->index > 0 && ch->index <= b->trace_chunks &&
                           b->trace_stops[ch->index - 1] == tce;
        trace_rows(b, ids, n, label ? VV_TRACE_LABEL : VV_TRACE_CONTEXT,
                   VV_TRACE_CONTEXT);
    }
    if (b->trace_err != VV_OK) return b->trace_err;
    dump_logits_dev(b, ch->index);
    /* Decode reads the length on the device. */
    vv_pipeline_kv_publish(ctx);
    return VV_OK;
}

/*
 * Several ready windows in one go: the front end packs stateless windows
 * into shared launches (8 of them fit its 30 s launch), which costs little
 * more than one window alone. Batched windows come out bit for bit what
 * each would alone (tests/test_vae_stream.c), so this changes timing only.
 */
static vv_status_t gpu_encode_ahead(void* self, int64_t first, int n,
                                    const float* windows, double* encode_ms) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    const vv_family_t* f = &ctx->family;
    const int frames = f->chunk_frames + f->lookahead_frames;
    const int64_t W = (int64_t)frames * f->frame_samples;
    *encode_ms = 0.0;
    b->ahead_n = 0;
    if (n <= 0 || n > VV_STREAM_AHEAD_MAX) return VV_ERR_INVALID_ARG;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    const size_t feat_bytes = (size_t)frames * (size_t)b->hs * 2;
    if (!b->ahead) return VV_ERR_NULL_PTR;   /* allocated at open */
    /* The previous batch's rows were copied out by chunks that have
     * finished (each ends in a sync), so the buffer is free. */
    vv_frontend_job_t jobs[VV_STREAM_AHEAD_MAX];
    memset(jobs, 0, sizeof(jobs));
    for (int i = 0; i < n; i++) {
        jobs[i].audio = windows + (size_t)i * (size_t)W;
        jobs[i].n_samples = W;
        jobs[i].stream = NULL;
        jobs[i].is_final = true;
        jobs[i].rows = (uint8_t*)b->ahead + (size_t)i * feat_bytes;
        jobs[i].rows_ld = b->hs;
        jobs[i].done_event = b->ahead_ev[i];
    }
    const double t0 = vv_time_ms();
    s = vv_frontend_run(ctx->frontend, jobs, n);
    for (int i = 0; s == VV_OK && i < n; i++) {
        if (jobs[i].status != VV_OK) s = jobs[i].status;
        else if (jobs[i].n_frames != frames) s = VV_ERR_SHAPE_MISMATCH;
        if (s == VV_OK)
            s = vv_dev_stream_wait_event(ctx->compute_stream, b->ahead_ev[i]);
    }
    if (s != VV_OK) return s;
    if (b->profile) {
        s = vv_dev_event_sync(b->ahead_ev[n - 1]);
        *encode_ms = vv_time_ms() - t0;
    }
    b->ahead_first = first;
    b->ahead_n = n;
    b->ahead_frames = frames;
    return s;
}

static vv_status_t gpu_decode_step(void* self, int32_t token, int32_t* next) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    if (token < 0 || token >= b->vocab) return VV_ERR_INVALID_ARG;
    if (b->spec_live) {
        /* The drafter will not see this token: blocks end here. */
        b->spec_live = false;
        VV_LOG_I("stream: a plain step at %d positions; no more drafted "
                 "blocks in this session", ctx->kv_cache->current_len);
    }

    if (b->embed_on_cpu) {
        const size_t row = (size_t)b->hs * 2;
        s = vv_dev_memcpy_h2d(b->hidden_one,
                              (const uint8_t*)ctx->model->embed_tokens.data
                              + (size_t)token * row, row, ctx->compute_stream);
    } else {
        /* The argmax left the token on the device; only a token the session
         * chose itself has to travel. */
        if (b->tok_dev_id != token) {
            *b->tok_pin = token;
            s = vv_dev_memcpy_h2d(b->tok_dev, b->tok_pin, sizeof(int32_t),
                                  ctx->compute_stream);
            if (s != VV_OK) return s;
            b->tok_dev_id = token;
        }
        s = vv_embedding_dev(ctx->embed_table_gpu, (const int32_t*)b->tok_dev,
                             b->hidden_one, 1, b->hs, ctx->compute_stream);
    }
    if (s != VV_OK) return s;

    s = vv_pipeline_step(ctx, b->hidden_one, b->graph_ok, b->graphs);
    if (s != VV_OK) return s;
    return gpu_head(b, b->hidden_one, next);
}

static vv_status_t gpu_decode_block(void* self, int32_t token, int32_t* out,
                                    int* n) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    if (!b->spec_live) return VV_ERR_UNSUPPORTED;
    if (token < 0 || token >= b->vocab) return VV_ERR_INVALID_ARG;
    s = vv_spec_cycle(ctx, b->spec, token, out, n);
    /* The step graphs read the length from the device, and the device's
     * token is no longer the one they would embed. */
    b->tok_dev_id = -1;
    if (s == VV_OK) vv_pipeline_kv_publish(ctx);
    return s;
}

static int gpu_block_size(void* self) {
    const sbe_t* b = (const sbe_t*)self;
    return b->spec_live ? vv_spec_block(b->spec) : 0;
}

static vv_status_t gpu_truncate(void* self, int64_t len) {
    sbe_t* b = (sbe_t*)self;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    if (!b->spec_live) return VV_ERR_UNSUPPORTED;
    s = vv_spec_rewind(b->ctx, b->spec, (int)len);
    if (s == VV_OK) vv_pipeline_kv_publish(b->ctx);
    return s;
}

/* The device buffers a session leaves on its context for the next one. */
typedef struct {
    void*    hidden;
    int32_t* ids_dev;
    int32_t* ids_pin;
    int      rows_cap;
    int32_t* tok_pin;
    void*    normed;
    void*    hidden_one;
    void*    logits;
    void*    am_v;
    void*    am_i;
    void*    tok_dev;
    void*    ahead;
    void*    ahead_ev[VV_STREAM_AHEAD_MAX];
} parked_t;

static void park_move(parked_t* p, sbe_t* b, bool to_park) {
#define MV(f) do { if (to_park) { p->f = b->f; b->f = NULL; } \
                   else { b->f = p->f; p->f = NULL; } } while (0)
    MV(hidden); MV(ids_dev); MV(ids_pin); MV(tok_pin); MV(normed);
    MV(hidden_one); MV(logits); MV(am_v); MV(am_i); MV(tok_dev); MV(ahead);
    for (int i = 0; i < VV_STREAM_AHEAD_MAX; i++) MV(ahead_ev[i]);
#undef MV
    if (to_park) { p->rows_cap = b->rows_cap; b->rows_cap = 0; }
    else         { b->rows_cap = p->rows_cap; p->rows_cap = 0; }
}

static void gpu_free_bufs(sbe_t* b) {
    gpu_free_rows(b);
    if (b->ahead) vv_dev_free(b->ahead);
    for (int i = 0; i < VV_STREAM_AHEAD_MAX; i++)
        if (b->ahead_ev[i]) vv_dev_event_destroy(b->ahead_ev[i]);
    if (b->normed) vv_dev_free(b->normed);
    if (b->hidden_one) vv_dev_free(b->hidden_one);
    if (b->logits) vv_dev_free(b->logits);
    if (b->am_v) vv_dev_free(b->am_v);
    if (b->am_i) vv_dev_free(b->am_i);
    if (b->tok_dev) vv_dev_free(b->tok_dev);
    if (b->tok_pin) vv_dev_free_pinned(b->tok_pin);
    memset(b->ahead_ev, 0, sizeof(b->ahead_ev));
    b->ahead = b->normed = b->hidden_one = b->logits = NULL;
    b->am_v = b->am_i = b->tok_dev = NULL;
    b->tok_pin = NULL;
}

void vv_stream_ctx_drop_cache(vv_inference_ctx_t* ctx) {
    if (!ctx || !ctx->stream_bufs) return;
    parked_t* p = (parked_t*)ctx->stream_bufs;
    ctx->stream_bufs = NULL;
    sbe_t tmp;
    memset(&tmp, 0, sizeof(tmp));
    park_move(p, &tmp, false);
    vv_free(p);
    if (ctx->use_gpu) vv_dev_set_device(ctx->gpu_id);
    gpu_free_bufs(&tmp);
}

static void gpu_destroy(void* self) {
    sbe_t* b = (sbe_t*)self;
    if (!b) return;
    vv_inference_ctx_t* ctx = b->ctx;
    bind(b);
    vv_dev_stream_sync(ctx->compute_stream);
    vv_pipeline_graphs_free(ctx, b->graphs);
    /* An idle slot must not sit on pages another slot could use. */
    vv_pipeline_kv_release(ctx);
    set_kv_no_wait(ctx, false);
    /* Keep the buffers for the next session on this slot: a cudaFree here
     * would synchronize the device under every other live session. */
    parked_t* p = ctx->stream_bufs
                  ? NULL : (parked_t*)vv_alloc(sizeof(parked_t));
    if (p) {
        memset(p, 0, sizeof(*p));
        park_move(p, b, true);
        ctx->stream_bufs = p;
    }
    gpu_free_bufs(b);
    vv_free(b->host_h);
    vv_free(b->host_f);
    vv_free(b);
}

static vv_status_t gpu_open(sbe_t* b, int rows, bool ahead) {
    vv_inference_ctx_t* ctx = b->ctx;
    if (!ctx->frontend || !ctx->final_norm_gpu) {
        VV_LOG_E("stream: no speech front end or final norm on the device");
        return VV_ERR_WEIGHT_MISSING;
    }
    b->embed_on_cpu = ctx->embed_table_gpu == NULL;
    b->head_on_cpu = ctx->lm_head_gpu == NULL;
    for (int i = 0; i < VV_MAX_GPUS; i++) {
        b->graphs[i].exec = NULL;
        b->graphs[i].shape = -1;
    }
    b->graph_ok = vv_pipeline_graph_ok(ctx);
    b->tok_dev_id = -1;
    /* Blocks need the head and the embedding on the device, one shard,
     * and no replay (a trace owns the taps). */
    b->spec = ctx->spec && !b->trace && !b->embed_on_cpu && !b->head_on_cpu &&
              ctx->n_shards == 0 ? ctx->spec : NULL;
    b->spec_live = false;
    const size_t row = (size_t)b->hs * 2;

    /* What the previous session on this slot left: the same model, so the
     * same set of buffers, only the row count can differ. */
    if (ctx->stream_bufs) {
        parked_t* p = (parked_t*)ctx->stream_bufs;
        ctx->stream_bufs = NULL;
        park_move(p, b, false);
        vv_free(p);
    }
    vv_status_t s = gpu_rows(b, rows);
    if (s == VV_OK && !b->normed) s = vv_dev_alloc(&b->normed, row);
    if (s == VV_OK && !b->hidden_one) s = vv_dev_alloc(&b->hidden_one, row);
    if (s == VV_OK && !b->tok_dev)
        s = vv_dev_alloc(&b->tok_dev, sizeof(int32_t));
    if (s == VV_OK && !b->tok_pin)
        s = vv_dev_alloc_pinned((void**)&b->tok_pin, sizeof(int32_t));
    if (s != VV_OK) return s;
    if (b->head_on_cpu) {
        b->host_h = (uint16_t*)vv_alloc(row);
        b->host_f = (float*)vv_alloc((size_t)b->hs * sizeof(float));
        if (!b->host_h || !b->host_f) return VV_ERR_OUT_OF_MEMORY;
        if (!ctx->model->lm_head.data) return VV_ERR_WEIGHT_MISSING;
    } else {
        if (!b->logits)
            s = vv_dev_alloc(&b->logits, (size_t)b->vocab * sizeof(float));
        if (s == VV_OK && !b->am_v)
            s = vv_dev_alloc(&b->am_v, VV_ARGMAX_PARTIALS * sizeof(float));
        if (s == VV_OK && !b->am_i)
            s = vv_dev_alloc(&b->am_i, VV_ARGMAX_PARTIALS * sizeof(int32_t));
    }
    if (s == VV_OK && ahead && !b->ahead) {
        /* Up front, not when a live session first falls behind: that is
         * the moment it can least afford a cudaMalloc. */
        const vv_family_t* f = &ctx->family;
        const size_t feat_bytes = (size_t)(f->chunk_frames +
                                           f->lookahead_frames) *
                                  (size_t)b->hs * 2;
        s = vv_dev_alloc(&b->ahead, feat_bytes * VV_STREAM_AHEAD_MAX);
        for (int i = 0; s == VV_OK && i < VV_STREAM_AHEAD_MAX; i++)
            if (!b->ahead_ev[i]) s = vv_dev_event_create(&b->ahead_ev[i]);
    }
    return s;
}

/* ─── CPU ───────────────────────────────────────────────────────────────── */

static vv_status_t cpu_rows(sbe_t* b, int rows) {
    if (rows <= b->rows_cap) return VV_OK;
    float* h = (float*)vv_realloc(b->hid32,
                                  (size_t)rows * (size_t)b->hs * sizeof(float));
    if (!h) return VV_ERR_OUT_OF_MEMORY;
    b->hid32 = h;
    b->rows_cap = rows;
    return VV_OK;
}

static vv_status_t cpu_head(sbe_t* b, const float* row, int32_t* tok) {
    vv_inference_ctx_t* ctx = b->ctx;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    vv_status_t s = vv_rmsnorm_cpu(row, ctx->model->final_norm.data, b->norm32,
                                   1, b->hs, llm->rms_norm_eps);
    if (s != VV_OK) return s;
    return vv_lm_head_argmax_cpu(b->norm32, ctx->model->lm_head.data,
                                 b->vocab, b->hs, tok, NULL);
}

static vv_status_t cpu_prefill_prompt(void* self, const int32_t* ids, int n) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_kv_cache_reset(ctx->kv_cache, NULL);
    vv_status_t s = cpu_rows(b, n);
    if (s != VV_OK) return s;
    s = vv_embedding_cpu(ctx->model->embed_tokens.data, ids, b->hid32, n,
                         b->hs);
    if (s != VV_OK) return s;
    return vv_decoder_prefill_cpu(ctx->model, b->hid32, n, ctx->kv_cache,
                                  (float*)ctx->workspace, ctx->workspace_size);
}

/* One encoder over the window, then its connector: [frames][hs] FP32. */
static vv_status_t cpu_encode_one(vv_conv_vae_encoder_t* enc,
                                  vv_connector_t* conn, const float* window,
                                  int n_samples, int want_frames,
                                  float** feats) {
    float* lat = NULL;
    int frames = 0;
    *feats = NULL;
    vv_status_t s = vv_conv_vae_encode_cpu(enc, window, n_samples, &lat,
                                           &frames);
    if (s == VV_OK && frames < want_frames) s = VV_ERR_SHAPE_MISMATCH;
    if (s == VV_OK) s = vv_connector_forward_cpu(conn, lat, want_frames, feats);
    vv_free(lat);
    return s;
}

static vv_status_t cpu_prefill_chunk(void* self, const vv_stream_chunk_t* ch,
                                     const float* window, int32_t* first) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    const int n = ch->n_rows;
    const int nf = ch->n_frames;
    const size_t hs = (size_t)b->hs;
    if (!ctx->acoustic_encoder || !ctx->semantic_encoder ||
        !ctx->acoustic_connector || !ctx->semantic_connector)
        return VV_ERR_WEIGHT_MISSING;
    vv_status_t s = cpu_rows(b, n);
    if (s != VV_OK) return s;

    const double t0 = vv_time_ms();
    const int n_samples = nf * ctx->family.frame_samples;
    float *fa = NULL, *fs = NULL;
    s = cpu_encode_one(ctx->acoustic_encoder, ctx->acoustic_connector,
                       window, n_samples, nf, &fa);
    if (s == VV_OK)
        s = cpu_encode_one(ctx->semantic_encoder, ctx->semantic_connector,
                           window, n_samples, nf, &fs);
    *ch->encode_ms = vv_time_ms() - t0;

    int32_t ids[64];
    for (int r = 0; r < n; r++)
        ids[r] = ch->rows[r] >= 0 ? ch->rows[r] : b->pad_id;
    if (s == VV_OK)
        s = vv_embedding_cpu(ctx->model->embed_tokens.data, ids, b->hid32, n,
                             b->hs);
    if (s == VV_OK) {
        float* dst = b->hid32 + (size_t)ch->feat_offset * hs;
        for (size_t i = 0; i < (size_t)nf * hs; i++) dst[i] = fa[i] + fs[i];
        if (vv_debug_dump_dir()) {
            char name[64];
            snprintf(name, sizeof(name), "stream_c%04lld_feats",
                     (long long)ch->index);
            vv_debug_dump(name, dst, (size_t)nf * hs * sizeof(float));
        }
    }
    vv_free(fa);
    vv_free(fs);
    if (s != VV_OK) return s;

    s = vv_decoder_prefill_cpu(ctx->model, b->hid32, n, ctx->kv_cache,
                               (float*)ctx->workspace, ctx->workspace_size);
    if (s != VV_OK) return s;
    return cpu_head(b, b->hid32 + (size_t)(n - 1) * hs, first);
}

static vv_status_t cpu_decode_step(void* self, int32_t token, int32_t* next) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    if (token < 0 || token >= b->vocab) return VV_ERR_INVALID_ARG;
    vv_status_t s = vv_embedding_cpu(ctx->model->embed_tokens.data, &token,
                                     b->one32, 1, b->hs);
    if (s == VV_OK)
        s = vv_decoder_step_cpu(ctx->model, b->one32, ctx->kv_cache,
                                (float*)ctx->workspace, ctx->workspace_size);
    if (s != VV_OK) return s;
    return cpu_head(b, b->one32, next);
}

static void cpu_destroy(void* self) {
    sbe_t* b = (sbe_t*)self;
    if (!b) return;
    vv_free(b->hid32);
    vv_free(b->one32);
    vv_free(b->norm32);
    vv_free(b);
}

/* ─── open ──────────────────────────────────────────────────────────────── */

vv_status_t vv_stream_params_for(const vv_inference_ctx_t* ctx,
                                 vv_stream_params_t* p) {
    if (!ctx || !p) return VV_ERR_NULL_PTR;
    vv_stream_params_default(p);
    if (!ctx->family_ok || ctx->family.mode != VV_GEN_CHUNKED)
        return VV_ERR_UNSUPPORTED;
    const vv_family_t* f = &ctx->family;
    p->geom.sample_rate = f->sample_rate;
    p->geom.frame_samples = f->frame_samples;
    p->geom.chunk_frames = f->chunk_frames;
    p->geom.lookahead_frames = f->lookahead_frames;
    p->geom.normalize_audio = f->normalize_audio;
    p->ids.speech_start = f->tok.speech_start;
    p->ids.speech_end = f->tok.speech_end;
    p->ids.text_chunk_end = f->tok.text_chunk_end;
    p->ids.eos = f->tok.endoftext;
    return VV_OK;
}

static vv_status_t stream_open_impl(vv_inference_ctx_t* ctx,
                                    const vv_stream_params_t* params,
                                    vv_spec_trace_t* trace,
                                    const vv_transcription_t* replayed,
                                    vv_stream_t** out);

vv_status_t vv_stream_open(vv_inference_ctx_t* ctx,
                           const vv_stream_params_t* params,
                           vv_stream_t** out) {
    return stream_open_impl(ctx, params, NULL, NULL, out);
}

static vv_status_t stream_open_impl(vv_inference_ctx_t* ctx,
                                    const vv_stream_params_t* params,
                                    vv_spec_trace_t* trace,
                                    const vv_transcription_t* replayed,
                                    vv_stream_t** out) {
    if (!ctx || !params || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!ctx->tokenizer || !ctx->family_ok) {
        VV_LOG_E("stream: no usable tokenizer for this model");
        return VV_ERR_MODEL_FORMAT;
    }
    if (ctx->family.mode != VV_GEN_CHUNKED) {
        VV_LOG_E("stream: %s transcribes whole clips, not chunk by chunk",
                 vv_model_family_name(ctx->family.id));
        return VV_ERR_UNSUPPORTED;
    }

    vv_stream_params_t p = *params;
    vv_stream_params_t m;
    vv_status_t s = vv_stream_params_for(ctx, &m);
    if (s != VV_OK) return s;
    p.geom = m.geom;
    p.ids = m.ids;
    if (p.max_new_tokens <= 0) p.max_new_tokens = 256;

    sbe_t* b = (sbe_t*)vv_alloc(sizeof(sbe_t));
    if (!b) return VV_ERR_OUT_OF_MEMORY;
    memset(b, 0, sizeof(*b));
    b->ctx = ctx;
    b->gpu = ctx->use_gpu && ctx->placement != VV_PLACE_CPU_ONLY;
    b->trace = trace;
    b->trace_stops = replayed ? replayed->chunk_stops : NULL;
    b->trace_chunks = replayed ? replayed->num_chunks : 0;
    b->trace_err = VV_OK;
    b->hs = ctx->model->config.llm.hidden_size;
    b->vocab = ctx->model->config.llm.vocab_size;
    b->pad_id = ctx->family.tok.speech_pad >= 0 ? ctx->family.tok.speech_pad
                                                 : 0;
    {
        const char* e = getenv("VV_STREAM_PROFILE");
        b->profile = e && e[0] && e[0] != '0';
    }
    /* [lead] start feats end, the most one chunk ever prefills. */
    const int chunk_rows = vv_stream_window_frames(&p.geom) + 3;
    if (chunk_rows > 64) { vv_free(b); return VV_ERR_UNSUPPORTED; }

    vv_stream_backend_t be;
    memset(&be, 0, sizeof(be));
    be.self = b;
    be.encode_text = be_encode_text;
    be.token_bytes = be_token_bytes;
    be.kv_len = be_kv_len;
    be.kv_capacity = be_kv_capacity;
    be.append_tokens = NULL;   /* the chunk end is folded into the next prefill */

    /* The prompt is ~31 tokens plus the hotwords, and a BPE token is at
     * least a byte: rows for both, so the buffer never grows mid-session. */
    int rows = 96 + (p.context_info ? (int)strlen(p.context_info) : 0);
    if (rows < chunk_rows) rows = chunk_rows;

    /* Admission: positions for kv_reserve_sec of audio, capped at the
     * window (the prompt comes on top when the prefill reserves). */
    b->kv_no_wait = p.kv_no_wait;
    if (p.kv_reserve_sec > 0.0 && ctx->kv_cache) {
        const double chunk_sec = (double)vv_stream_chunk_samples(&p.geom) /
                                 (double)p.geom.sample_rate;
        const double chunks = p.kv_reserve_sec / chunk_sec + 1.0;
        double pos = rows + chunks * (double)(chunk_rows + KV_TEXT_PER_CHUNK);
        if (pos > (double)ctx->kv_cache->max_seq_len)
            pos = (double)ctx->kv_cache->max_seq_len;
        b->kv_reserve = (int)pos;
    }

    if (b->gpu) {
        /* VV_STREAM_AHEAD=0: every window on its own, for comparison. */
        const char* ah = getenv("VV_STREAM_AHEAD");
        const bool ahead = !(ah && ah[0] == '0');
        s = vv_dev_set_device(ctx->gpu_id);
        if (s == VV_OK) s = gpu_open(b, rows, ahead);
        be.prefill_prompt = gpu_prefill_prompt;
        be.prefill_chunk = gpu_prefill_chunk;
        be.decode_step = gpu_decode_step;
        be.decode_block = gpu_decode_block;
        be.block_size = gpu_block_size;
        be.truncate = gpu_truncate;
        be.prefill_tokens = gpu_prefill_tokens;
        be.destroy = gpu_destroy;
        if (ahead) be.encode_ahead = gpu_encode_ahead;
    } else {
        if (!ctx->model->lm_head.data || !ctx->model->final_norm.data)
            s = VV_ERR_WEIGHT_MISSING;
        if (s == VV_OK) s = cpu_rows(b, rows);
        if (s == VV_OK) {
            b->one32 = (float*)vv_alloc((size_t)b->hs * sizeof(float));
            b->norm32 = (float*)vv_alloc((size_t)b->hs * sizeof(float));
            if (!b->one32 || !b->norm32) s = VV_ERR_OUT_OF_MEMORY;
        }
        be.prefill_prompt = cpu_prefill_prompt;
        be.prefill_chunk = cpu_prefill_chunk;
        be.decode_step = cpu_decode_step;
        be.destroy = cpu_destroy;
    }
    if (s != VV_OK) {
        be.destroy(b);
        return s;
    }

    s = vv_stream_open_backend(&be, &p, out);
    if (s != VV_OK) be.destroy(b);   /* open leaves the backend to us */
    if (s == VV_OK)
        VV_LOG_I("stream: session open on %s, %d+%d frames per window, "
                 "KV %lld/%lld", b->gpu ? "GPU" : "CPU",
                 p.geom.chunk_frames, p.geom.lookahead_frames,
                 (long long)be_kv_len(b), (long long)be_kv_capacity(b));
    return s;
}

/* ─── replay (vv_spec_trace) ────────────────────────────────────────────── */

vv_status_t vv_stream_trace(vv_inference_ctx_t* ctx, const float* pcm24k,
                            int n_samples, const char* context_info,
                            const vv_transcription_t* tr,
                            vv_spec_trace_t* t) {
    if (tr->num_chunks <= 0 || !tr->chunk_tokens) return VV_ERR_INVALID_ARG;
    vv_stream_params_t p;
    vv_status_t s = vv_stream_params_for(ctx, &p);
    if (s != VV_OK) return s;
    p.context_info = context_info;
    p.force_ids = tr->tokens;
    p.force_n = tr->chunk_tokens;
    p.force_chunks = tr->num_chunks;

    t->n = 0;
    ctx->taps = &t->taps;
    vv_stream_t* st = NULL;
    s = stream_open_impl(ctx, &p, t, tr, &st);
    if (s == VV_OK) s = vv_stream_push(st, pcm24k, (size_t)n_samples);
    if (s == VV_OK) s = vv_stream_finish(st);
    if (s == VV_OK) {
        vv_stream_stats_t ss;
        vv_stream_get_stats(st, &ss);
        /* The same audio in the same windows: as many chunks as the run. */
        if (ss.chunks != tr->num_chunks) {
            VV_LOG_E("trace: the replay made %lld chunks, the run %d",
                     (long long)ss.chunks, tr->num_chunks);
            s = VV_ERR_SHAPE_MISMATCH;
        }
    }
    vv_stream_close(st);
    ctx->taps = NULL;

    /* The last chunk's stop follows it as a label with no features. */
    const int32_t stop = tr->chunk_stops ? tr->chunk_stops[tr->num_chunks - 1]
                                         : -1;
    if (s == VV_OK && stop >= 0) {
        if (t->n + 1 > t->cap) return VV_ERR_OVERFLOW;
        t->ids[t->n] = stop;
        t->kind[t->n] = VV_TRACE_LABEL;
        t->n++;
    }
    return s;
}

/* ─── whole clip ────────────────────────────────────────────────────────── */

typedef struct {
    char**   texts;
    int      n, cap;
    /* Every chunk's ids back to back, and per chunk its count and stop. */
    int32_t* ids;
    int      n_ids, cap_ids;
    int*     chunk_n;
    int32_t* chunk_stop;
    vv_stream_event_fn user_fn;
    void*    user;
    vv_status_t oom;
} collect_t;

static void collect_event(void* user, const vv_stream_event_t* ev) {
    collect_t* c = (collect_t*)user;
    if (ev->type == VV_STREAM_EVENT_CHUNK) {
        if (c->n == c->cap) {
            const int nc = c->cap ? c->cap * 2 : 64;
            char** nt = (char**)vv_realloc(c->texts, sizeof(char*) * (size_t)nc);
            int* nn = (int*)vv_realloc(c->chunk_n, sizeof(int) * (size_t)nc);
            if (nn) c->chunk_n = nn;
            int32_t* ns = (int32_t*)vv_realloc(c->chunk_stop,
                                               sizeof(int32_t) * (size_t)nc);
            if (ns) c->chunk_stop = ns;
            if (nt) c->texts = nt;
            if (!nt || !nn || !ns) { c->oom = VV_ERR_OUT_OF_MEMORY; goto fwd; }
            c->cap = nc;
        }
        if (c->n_ids + ev->n_tokens > c->cap_ids) {
            int nc = c->cap_ids ? c->cap_ids : 1024;
            while (nc < c->n_ids + ev->n_tokens) nc *= 2;
            int32_t* ni = (int32_t*)vv_realloc(c->ids,
                                               sizeof(int32_t) * (size_t)nc);
            if (!ni) { c->oom = VV_ERR_OUT_OF_MEMORY; goto fwd; }
            c->ids = ni;
            c->cap_ids = nc;
        }
        char* t = (char*)vv_alloc(ev->text_len + 1);
        if (!t) { c->oom = VV_ERR_OUT_OF_MEMORY; goto fwd; }
        memcpy(t, ev->text, ev->text_len);
        t[ev->text_len] = '\0';
        if (ev->n_tokens > 0 && ev->ids)
            memcpy(c->ids + c->n_ids, ev->ids,
                   sizeof(int32_t) * (size_t)ev->n_tokens);
        c->n_ids += ev->ids ? ev->n_tokens : 0;
        c->chunk_n[c->n] = ev->ids ? ev->n_tokens : 0;
        c->chunk_stop[c->n] = ev->stop_id;
        c->texts[c->n++] = t;
    }
fwd:
    if (c->user_fn) c->user_fn(c->user, ev);
}

vv_status_t vv_stream_transcribe(vv_inference_ctx_t* ctx,
                                 const float* pcm24k, int n_samples,
                                 const char* context_info,
                                 vv_stream_event_fn on_event, void* user,
                                 vv_transcription_t** result) {
    if (!ctx || !pcm24k || !result) return VV_ERR_NULL_PTR;
    *result = NULL;
    const double t_start = vv_time_ms();
    vv_perf_metrics_t* perf = &ctx->last_perf;
    memset(perf, 0, sizeof(*perf));
    perf->audio_duration_sec = (float)n_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = ctx->model->config.llm.hidden_size;
    perf->kv_format = ctx->kv_cache ? ctx->kv_cache->format : 0;
    perf->attn_backend = ctx->kv_cache ? ctx->kv_cache->attn_backend : 0;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    collect_t col;
    memset(&col, 0, sizeof(col));
    col.user_fn = on_event;
    col.user = user;

    vv_stream_params_t p;
    vv_status_t s = vv_stream_params_for(ctx, &p);
    if (s != VV_OK) return s;
    p.context_info = context_info;
    p.on_event = collect_event;
    p.user = &col;

    vv_stream_t* st = NULL;
    s = vv_stream_open(ctx, &p, &st);
    if (s == VV_OK) s = vv_stream_push(st, pcm24k, (size_t)n_samples);
    if (s == VV_OK) s = vv_stream_finish(st);
    if (s == VV_OK) s = col.oom;

    vv_stream_stats_t ss;
    vv_stream_get_stats(st, &ss);
    if (s == VV_OK) {
        const double chunk_sec =
            (double)vv_stream_chunk_samples(&p.geom) / p.geom.sample_rate;
        s = vv_stream_build_transcription((const char* const*)col.texts,
                                          col.n, chunk_sec,
                                          perf->audio_duration_sec, result);
        if (s == VV_OK && *result)
            s = vv_transcription_set_tokens(*result, col.ids, col.n_ids,
                                            col.chunk_n, col.chunk_stop,
                                            col.n);
    }

    if (st) {
        perf->audio_frames = (int)(ss.chunks * vv_stream_window_frames(&p.geom));
        perf->audio_encode_ms = ss.encode_ms;
        perf->prefill_tokens = (int)(ss.prompt_tokens + ss.prefill_rows);
        perf->prefill_ms = ss.prompt_ms + ss.prefill_ms;
        perf->prefill_tok_per_sec = perf->prefill_ms > 0.001
            ? perf->prefill_tokens / perf->prefill_ms * 1000.0 : 0.0;
        perf->decode_ms = ss.decode_ms;
        perf->decode_tokens = (int)ss.tokens;
        perf->decode_tok_per_sec = ss.decode_ms > 0.001
            ? (double)ss.tokens / ss.decode_ms * 1000.0 : 0.0;
        perf->kv_cache_used = (int)ss.kv_len;
        perf->kv_cache_max = (int)ss.kv_capacity;
        perf->kv_cache_pct = ss.kv_capacity > 0
            ? 100.0f * (float)ss.kv_len / (float)ss.kv_capacity : 0.0f;
        vv_stream_close(st);
    }
    for (int i = 0; i < col.n; i++) vv_free(col.texts[i]);
    vv_free(col.texts);
    vv_free(col.ids);
    vv_free(col.chunk_n);
    vv_free(col.chunk_stop);

    perf->total_ms = vv_time_ms() - t_start;
    perf->ttft_ms = ss.prompt_ms;
    perf->rtf = perf->audio_duration_sec > 0.001f
        ? perf->total_ms / 1000.0 / perf->audio_duration_sec : 0.0;
    if (ctx->use_gpu) {
        size_t vfree = 0, vtotal = 0;
        if (vv_dev_get_device_info(ctx->gpu_id, &vtotal, &vfree, NULL) == VV_OK) {
            perf->vram_total_bytes = vtotal;
            perf->vram_free_bytes = vfree;
            perf->vram_used_bytes = vtotal - vfree;
        }
    }
    if (s != VV_OK && *result) {
        vv_transcription_free(*result);
        *result = NULL;
    }
    return s;
}
