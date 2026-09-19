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
 *    dozen times over half an hour of audio, not once per chunk.
 *
 * All device buffers are allocated at open (the prompt may grow the row
 * buffer once); nothing on the per-chunk path allocates on the device. The
 * CPU backend is the same sequence over the host kernels.
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

#include "pipeline_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARGMAX_PARTIALS 256   /* matches VV_ARGMAX_PARTIALS in pipeline.c */

typedef struct {
    vv_inference_ctx_t* ctx;
    bool     gpu;
    int      hs;
    int      vocab;
    int      rows_cap;       /* rows the hidden buffer holds */
    int32_t  pad_id;         /* id embedded under a feature row */
    bool     profile;        /* VV_STREAM_PROFILE: split encode from prefill */

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

    /* CPU */
    float*   hid32;          /* [rows_cap][hs] */
    float*   one32;          /* [hs] */
    float*   norm32;         /* [hs] */
} sbe_t;

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
    vv_inference_ctx_t* ctx = b->ctx;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    vv_status_t s = vv_rmsnorm_dev(row, ctx->final_norm_gpu, b->normed, 1,
                                   b->hs, llm->rms_norm_eps,
                                   ctx->compute_stream);
    if (s != VV_OK) return s;
    if (b->head_on_cpu) {
        s = vv_pipeline_cpu_head_argmax(ctx, b->normed, b->host_h, b->host_f,
                                        tok);
        b->tok_dev_id = -1;
        return s;
    }
    s = vv_lm_head_gemv_dev(b->normed, ctx->lm_head_gpu, b->logits, b->vocab,
                            b->hs, ctx->compute_stream);
    if (s == VV_OK)
        s = vv_argmax_dev(b->logits, b->vocab, b->am_v, b->am_i, b->tok_dev,
                          NULL, ctx->compute_stream);
    /* Into pinned memory the copy is asynchronous: wait for it, not just
     * for the argmax before it. */
    if (s == VV_OK)
        s = vv_dev_memcpy_d2h(b->tok_pin, b->tok_dev, sizeof(int32_t),
                              ctx->compute_stream);
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    if (s != VV_OK) return s;
    *tok = *b->tok_pin;
    b->tok_dev_id = *tok;
    return VV_OK;
}

static vv_status_t gpu_prefill_prompt(void* self, const int32_t* ids, int n) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    vv_pipeline_kv_reset(ctx);
    s = gpu_rows(b, n);
    if (s == VV_OK) s = vv_pipeline_kv_reserve(ctx, n);
    if (s == VV_OK) s = gpu_embed_rows(b, ids, n);
    if (s == VV_OK) s = vv_pipeline_prefill(ctx, b->hidden, n);
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    return s;
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
    if (s != VV_OK) return s;

    int32_t ids[64];
    for (int r = 0; r < n; r++)
        ids[r] = ch->rows[r] >= 0 ? ch->rows[r] : b->pad_id;
    s = gpu_embed_rows(b, ids, n);
    if (s != VV_OK) return s;
    /* The front end's row writes wait for the embedding, not for the host. */
    s = vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);
    if (s != VV_OK) return s;

    const double t0 = vv_time_ms();
    vv_frontend_job_t job;
    memset(&job, 0, sizeof(job));
    job.audio = window;
    job.n_samples = (int64_t)(ch->n_frames) * ctx->family.frame_samples;
    job.stream = NULL;                 /* a stateless window */
    job.is_final = true;
    job.rows = (uint8_t*)b->hidden + (size_t)ch->feat_offset * b->hs * 2;
    job.rows_ld = b->hs;
    job.wait_event = ctx->fe_ready;
    job.done_event = ctx->fe_done;
    s = vv_frontend_submit(ctx->frontend, &job);
    if (s == VV_OK && job.n_frames != ch->n_frames) {
        VV_LOG_E("stream: the window encoded to %d frames, the chunk has %d",
                 job.n_frames, ch->n_frames);
        s = VV_ERR_SHAPE_MISMATCH;
    }
    if (s == VV_OK && b->profile) {
        s = vv_dev_event_sync(ctx->fe_done);
        *ch->encode_ms = vv_time_ms() - t0;
    }
    if (s == VV_OK) s = vv_dev_stream_wait_event(ctx->compute_stream,
                                                 ctx->fe_done);
    if (s != VV_OK) return s;

    if (vv_debug_dump_dir()) {
        char name[64];
        snprintf(name, sizeof(name), "stream_c%04lld_feats",
                 (long long)ch->index);
        dump_f16_rows(name, b, job.rows, ch->n_frames);
    }

    s = vv_pipeline_prefill(ctx, b->hidden, n);
    if (s != VV_OK) return s;
    s = gpu_head(b, (const uint8_t*)b->hidden + (size_t)(n - 1) * b->hs * 2,
                 first);
    if (s != VV_OK) return s;
    dump_logits_dev(b, ch->index);
    /* Decode reads the length on the device. */
    vv_pipeline_kv_publish(ctx);
    return VV_OK;
}

static vv_status_t gpu_decode_step(void* self, int32_t token, int32_t* next) {
    sbe_t* b = (sbe_t*)self;
    vv_inference_ctx_t* ctx = b->ctx;
    vv_status_t s = bind(b);
    if (s != VV_OK) return s;
    if (token < 0 || token >= b->vocab) return VV_ERR_INVALID_ARG;

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

static void gpu_destroy(void* self) {
    sbe_t* b = (sbe_t*)self;
    if (!b) return;
    vv_inference_ctx_t* ctx = b->ctx;
    bind(b);
    vv_dev_stream_sync(ctx->compute_stream);
    vv_pipeline_graphs_free(ctx, b->graphs);
    /* An idle slot must not sit on pages another slot could use. */
    vv_pipeline_kv_release(ctx);
    gpu_free_rows(b);
    if (b->normed) vv_dev_free(b->normed);
    if (b->hidden_one) vv_dev_free(b->hidden_one);
    if (b->logits) vv_dev_free(b->logits);
    if (b->am_v) vv_dev_free(b->am_v);
    if (b->am_i) vv_dev_free(b->am_i);
    if (b->tok_dev) vv_dev_free(b->tok_dev);
    if (b->tok_pin) vv_dev_free_pinned(b->tok_pin);
    vv_free(b->host_h);
    vv_free(b->host_f);
    vv_free(b);
}

static vv_status_t gpu_open(sbe_t* b, int rows) {
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
    const size_t row = (size_t)b->hs * 2;
    vv_status_t s = gpu_rows(b, rows);
    if (s == VV_OK) s = vv_dev_alloc(&b->normed, row);
    if (s == VV_OK) s = vv_dev_alloc(&b->hidden_one, row);
    if (s == VV_OK) s = vv_dev_alloc(&b->tok_dev, sizeof(int32_t));
    if (s == VV_OK)
        s = vv_dev_alloc_pinned((void**)&b->tok_pin, sizeof(int32_t));
    if (s != VV_OK) return s;
    if (b->head_on_cpu) {
        b->host_h = (uint16_t*)vv_alloc(row);
        b->host_f = (float*)vv_alloc((size_t)b->hs * sizeof(float));
        if (!b->host_h || !b->host_f) return VV_ERR_OUT_OF_MEMORY;
        if (!ctx->model->lm_head.data) return VV_ERR_WEIGHT_MISSING;
    } else {
        s = vv_dev_alloc(&b->logits, (size_t)b->vocab * sizeof(float));
        if (s == VV_OK)
            s = vv_dev_alloc(&b->am_v, ARGMAX_PARTIALS * sizeof(float));
        if (s == VV_OK)
            s = vv_dev_alloc(&b->am_i, ARGMAX_PARTIALS * sizeof(int32_t));
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

vv_status_t vv_stream_open(vv_inference_ctx_t* ctx,
                           const vv_stream_params_t* params,
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

    if (b->gpu) {
        s = vv_dev_set_device(ctx->gpu_id);
        if (s == VV_OK) s = gpu_open(b, chunk_rows);
        be.prefill_prompt = gpu_prefill_prompt;
        be.prefill_chunk = gpu_prefill_chunk;
        be.decode_step = gpu_decode_step;
        be.destroy = gpu_destroy;
    } else {
        if (!ctx->model->lm_head.data || !ctx->model->final_norm.data)
            s = VV_ERR_WEIGHT_MISSING;
        if (s == VV_OK) s = cpu_rows(b, chunk_rows);
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

/* ─── whole clip ────────────────────────────────────────────────────────── */

typedef struct {
    char**   texts;
    int      n, cap;
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
            if (!nt) { c->oom = VV_ERR_OUT_OF_MEMORY; goto fwd; }
            c->texts = nt;
            c->cap = nc;
        }
        char* t = (char*)vv_alloc(ev->text_len + 1);
        if (!t) { c->oom = VV_ERR_OUT_OF_MEMORY; goto fwd; }
        memcpy(t, ev->text, ev->text_len);
        t[ev->text_len] = '\0';
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
