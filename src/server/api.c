/**
 * @file api.c
 * @brief Routes and response formats for the HTTP front end. See server.h.
 */

#include "vibevoice/server.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"
#include "vibevoice/tokenizer_encoder.h"

#include "vv_http.h"
#include "vv_queue.h"
#include "vv_ws.h"
#include "cJSON.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct vv_server {
    vv_queue_t   queue;
    vv_mutex_t   stats_lock;
    vv_http_t*   http;
    vv_engine_t* engine;
    char         model_name[128];
    char         api_key[256];
    int          acoustic_sampling;
    int          stream_idle_ms;
    int          stream_slot_wait_ms;
    double       stream_reserve_sec;
    uint64_t     started_at;
    uint64_t     n_requests;
    uint64_t     n_errors;
    double       audio_seconds;
    double       compute_seconds;
};

/* ─── JSON string building ───────────────────────────────────────────────── */

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
} strbuf_t;

static void sb_init(strbuf_t* b) {
    b->cap = 4096;
    b->len = 0;
    b->buf = (char*)vv_alloc(b->cap);
    if (b->buf) b->buf[0] = '\0';
}

static void sb_free(strbuf_t* b) { vv_free(b->buf); b->buf = NULL; }

static void sb_reserve(strbuf_t* b, size_t extra) {
    if (!b->buf) return;
    if (b->len + extra + 1 <= b->cap) return;
    size_t ncap = b->cap;
    while (ncap < b->len + extra + 1) ncap *= 2;
    char* nb = (char*)vv_alloc(ncap);
    if (!nb) return;
    memcpy(nb, b->buf, b->len + 1);
    vv_free(b->buf);
    b->buf = nb;
    b->cap = ncap;
}

static void sb_puts(strbuf_t* b, const char* s) {
    const size_t n = strlen(s);
    sb_reserve(b, n);
    if (!b->buf || b->len + n + 1 > b->cap) return;  /* growth failed */
    memcpy(b->buf + b->len, s, n + 1);
    b->len += n;
}

/*
 * Formatted append, measured first so that nothing is cut: segment text in
 * an SRT or VTT line is as long as the speaker made it, and a fixed 1 KB
 * buffer used to truncate it silently.
 */
static void sb_printf(strbuf_t* b, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    sb_reserve(b, (size_t)n);
    if (!b->buf || b->len + (size_t)n + 1 > b->cap) return;
    va_start(ap, fmt);
    vsnprintf(b->buf + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

/** @brief Append `s` escaped as a JSON string body (no surrounding quotes). */
static void sb_json_escaped(strbuf_t* b, const char* s) {
    if (!s) return;
    for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
        switch (*p) {
            case '"':  sb_puts(b, "\\\""); break;
            case '\\': sb_puts(b, "\\\\"); break;
            case '\n': sb_puts(b, "\\n");  break;
            case '\r': sb_puts(b, "\\r");  break;
            case '\t': sb_puts(b, "\\t");  break;
            default:
                if (*p < 0x20) sb_printf(b, "\\u%04x", *p);
                else {
                    sb_reserve(b, 1);
                    if (b->buf && b->len + 2 <= b->cap) {
                        b->buf[b->len++] = (char)*p;
                        b->buf[b->len] = '\0';
                    }
                }
        }
    }
}

/* ─── Response formats ───────────────────────────────────────────────────── */

static void fmt_timestamp(char* out, size_t n, float t, bool comma) {
    const int total_ms = (int)(t * 1000.0f + 0.5f);
    const int h = total_ms / 3600000;
    const int m = (total_ms / 60000) % 60;
    const int s = (total_ms / 1000) % 60;
    const int ms = total_ms % 1000;
    snprintf(out, n, "%02d:%02d:%02d%c%03d", h, m, s, comma ? ',' : '.', ms);
}

static void build_srt(strbuf_t* b, const vv_transcription_t* tr) {
    for (int i = 0; i < tr->num_segments; i++) {
        char a[32], z[32];
        fmt_timestamp(a, sizeof(a), tr->segments[i].start_time, true);
        fmt_timestamp(z, sizeof(z), tr->segments[i].end_time, true);
        sb_printf(b, "%d\n%s --> %s\n", i + 1, a, z);
        if (tr->segments[i].speaker && tr->segments[i].speaker[0])
            sb_printf(b, "[%s] ", tr->segments[i].speaker);
        sb_printf(b, "%s\n\n", tr->segments[i].text ? tr->segments[i].text : "");
    }
}

static void build_vtt(strbuf_t* b, const vv_transcription_t* tr) {
    sb_puts(b, "WEBVTT\n\n");
    for (int i = 0; i < tr->num_segments; i++) {
        char a[32], z[32];
        fmt_timestamp(a, sizeof(a), tr->segments[i].start_time, false);
        fmt_timestamp(z, sizeof(z), tr->segments[i].end_time, false);
        sb_printf(b, "%s --> %s\n", a, z);
        if (tr->segments[i].speaker && tr->segments[i].speaker[0])
            sb_printf(b, "<v %s>", tr->segments[i].speaker);
        sb_printf(b, "%s\n\n", tr->segments[i].text ? tr->segments[i].text : "");
    }
}

static void build_verbose_json(strbuf_t* b, const vv_transcription_t* tr,
                               const vv_perf_metrics_t* perf) {
    sb_puts(b, "{\"task\":\"transcribe\",\"language\":\"");
    sb_json_escaped(b, tr->language ? tr->language : "en");
    sb_printf(b, "\",\"duration\":%.3f,\"text\":\"", tr->duration);
    sb_json_escaped(b, tr->full_text ? tr->full_text : "");
    sb_puts(b, "\",\"segments\":[");
    for (int i = 0; i < tr->num_segments; i++) {
        const vv_segment_t* s = &tr->segments[i];
        if (i) sb_puts(b, ",");
        sb_printf(b, "{\"id\":%d,\"seek\":0,\"start\":%.3f,\"end\":%.3f,"
                     "\"text\":\"", i, s->start_time, s->end_time);
        sb_json_escaped(b, s->text ? s->text : "");
        sb_puts(b, "\",\"speaker\":\"");
        sb_json_escaped(b, s->speaker ? s->speaker : "");
        sb_puts(b, "\",\"tokens\":[],\"temperature\":0.0,"
                   "\"avg_logprob\":0.0,\"compression_ratio\":1.0,"
                   "\"no_speech_prob\":0.0}");
    }
    sb_puts(b, "]");
    if (perf) {
        /* Not part of the OpenAI schema; harmless to clients, useful to us. */
        sb_printf(b, ",\"x_vibevoice\":{\"rtf\":%.4f,\"decode_tok_per_sec\":%.1f,"
                     "\"prefill_tok_per_sec\":%.1f,\"total_ms\":%.1f}",
                  perf->rtf, perf->decode_tok_per_sec,
                  perf->prefill_tok_per_sec, perf->total_ms);
    }
    sb_puts(b, "}");
}

/* ─── Handlers ───────────────────────────────────────────────────────────── */

static void handle_models(vv_server_t* sv, vv_http_res_t* res, bool single) {
    strbuf_t b; sb_init(&b);
    if (!single) sb_puts(&b, "{\"object\":\"list\",\"data\":[");
    sb_puts(&b, "{\"id\":\"");
    sb_json_escaped(&b, sv->model_name);
    sb_printf(&b, "\",\"object\":\"model\",\"created\":%llu,"
                  "\"owned_by\":\"vibevoice.c\"}",
              (unsigned long long)sv->started_at);
    if (!single) sb_puts(&b, "]}");
    vv_http_respond_json(res, 200, b.buf ? b.buf : "{}");
    sb_free(&b);
}

static void handle_health(vv_server_t* sv, vv_http_res_t* res) {
    uint64_t done = 0;
    int busy = 0;
    vv_engine_stats(sv->engine, &done, &busy);
    /* The model name is the operator's free text: escaped like any other. */
    strbuf_t b; sb_init(&b);
    sb_puts(&b, "{\"status\":\"ok\",\"model\":\"");
    sb_json_escaped(&b, sv->model_name);
    sb_printf(&b, "\",\"version\":\"%s\",\"build\":\"%s\",\"slots\":%d,"
                  "\"busy\":%d,\"completed\":%llu}",
              vv_version(), vv_build_ref(), vv_engine_slots(sv->engine), busy,
              (unsigned long long)done);
    vv_http_respond_json(res, 200, b.buf ? b.buf : "{}");
    sb_free(&b);
}

static void handle_metrics(vv_server_t* sv, vv_http_res_t* res) {
    uint64_t done = 0;
    int busy = 0;
    vv_engine_stats(sv->engine, &done, &busy);
    char buf[1024];
    vv_mutex_lock(&sv->stats_lock);
    snprintf(buf, sizeof(buf),
        "# HELP vibevoice_requests_total Transcription requests served\n"
        "# TYPE vibevoice_requests_total counter\n"
        "vibevoice_requests_total %llu\n"
        "# HELP vibevoice_errors_total Requests that failed\n"
        "# TYPE vibevoice_errors_total counter\n"
        "vibevoice_errors_total %llu\n"
        "# HELP vibevoice_slots_busy Slots currently running\n"
        "# TYPE vibevoice_slots_busy gauge\n"
        "vibevoice_slots_busy %d\n"
        "# HELP vibevoice_slots_total Concurrent slots configured\n"
        "# TYPE vibevoice_slots_total gauge\n"
        "vibevoice_slots_total %d\n"
        "# HELP vibevoice_audio_seconds_total Audio transcribed\n"
        "# TYPE vibevoice_audio_seconds_total counter\n"
        "vibevoice_audio_seconds_total %.3f\n"
        "# HELP vibevoice_compute_seconds_total Time spent transcribing\n"
        "# TYPE vibevoice_compute_seconds_total counter\n"
        "vibevoice_compute_seconds_total %.3f\n",
        (unsigned long long)sv->n_requests, (unsigned long long)sv->n_errors,
        busy, vv_engine_slots(sv->engine),
        sv->audio_seconds, sv->compute_seconds);
    vv_mutex_unlock(&sv->stats_lock);
    char queue_buf[256];
    vv_mutex_lock(&sv->queue.lock);
    snprintf(queue_buf, sizeof(queue_buf),
             "vibevoice_queue_waiting %d\nvibevoice_queue_capacity %d\n"
             "vibevoice_queue_rejected_total %llu\n",
             sv->queue.waiting, sv->queue.capacity,
             (unsigned long long)sv->queue.rejected);
    vv_mutex_unlock(&sv->queue.lock);
    /*
     * A gauge of 1 carrying the version in its labels, which is how a
     * Prometheus target says which build is answering.
     */
    char info_buf[320];
    snprintf(info_buf, sizeof(info_buf),
             "# HELP vibevoice_build_info Version of the binary answering\n"
             "# TYPE vibevoice_build_info gauge\n"
             "vibevoice_build_info{version=\"%s\",build=\"%s\","
             "features=\"%s\"} 1\n",
             vv_version(), vv_build_ref(), vv_build_features());
    strbuf_t b; sb_init(&b); sb_puts(&b, buf); sb_puts(&b, queue_buf);
    sb_puts(&b, info_buf);
    vv_http_respond(res, 200, "text/plain; version=0.0.4", b.buf, b.len);
    sb_free(&b);
}

/** @brief Copy a form field into a NUL-terminated buffer. */
static void part_to_str(const vv_http_part_t* p, char* out, size_t n) {
    out[0] = '\0';
    if (!p) return;
    const size_t len = p->size < n - 1 ? p->size : n - 1;
    memcpy(out, p->data, len);
    out[len] = '\0';
}

static const char* hotbuf_joined(const vv_inference_params_t* ip,
                                 char* prompt);
static void sse_transcribe(vv_server_t* sv, vv_http_res_t* res,
                           const float* pcm, int n_samples, int sample_rate,
                           const vv_inference_params_t* ip,
                           const char* context_info, bool token_deltas);

static void handle_transcriptions(vv_server_t* sv, const vv_http_req_t* req,
                                  vv_http_res_t* res) {
    if (strcmp(req->method, "POST") != 0) {
        vv_http_error(res, 405, "invalid_request_error", "use POST");
        return;
    }

    const char* ctype = vv_http_header(req, "Content-Type");
    if (!ctype || !strstr(ctype, "multipart/form-data")) {
        vv_http_error(res, 415, "invalid_request_error",
                      "expected multipart/form-data with a 'file' field");
        return;
    }

    vv_http_part_t parts[VV_HTTP_MAX_PARTS];
    const int np = vv_http_parse_multipart(req->body, req->body_len, ctype,
                                           parts, VV_HTTP_MAX_PARTS);
    const vv_http_part_t* file = vv_http_part(parts, np, "file");
    if (!file || file->size == 0) {
        vv_http_error(res, 400, "invalid_request_error", "missing 'file'");
        return;
    }

    char fmt[32], prompt[1024], temperature[32];
    part_to_str(vv_http_part(parts, np, "response_format"), fmt, sizeof(fmt));
    part_to_str(vv_http_part(parts, np, "prompt"), prompt, sizeof(prompt));
    part_to_str(vv_http_part(parts, np, "temperature"), temperature,
                sizeof(temperature));
    if (!fmt[0]) snprintf(fmt, sizeof(fmt), "json");

    float* pcm = NULL;
    int n_samples = 0, sample_rate = 0;
    vv_status_t s = vv_audio_load_memory(file->data, file->size, &pcm,
                                         &n_samples, &sample_rate);
    if (s != VV_OK) {
        vv_mutex_lock(&sv->stats_lock); sv->n_errors++; vv_mutex_unlock(&sv->stats_lock);
        vv_http_error(res, 415, "invalid_request_error",
                      "could not decode the uploaded audio "
                      "(WAV is built in; other formats need ffmpeg on PATH)");
        return;
    }

    /*
     * OpenAI's `prompt` is a transcription hint; this model takes the same
     * idea as hotwords, so comma-separated entries map straight across.
     */
    vv_inference_params_t ip;
    memset(&ip, 0, sizeof(ip));
    ip.max_new_tokens = 64000;
    ip.temperature = temperature[0] ? (float)atof(temperature) : 0.0f;
    ip.top_k = 1;
    ip.enable_timestamps = true;
    ip.enable_diarize = true;
    /* Seed 0 asks the pipeline for a fresh draw, once per request. */
    ip.acoustic_sampling = sv->acoustic_sampling;

    const char* hot[32];
    char hotbuf[1024];
    int n_hot = 0;
    if (prompt[0]) {
        snprintf(hotbuf, sizeof(hotbuf), "%s", prompt);
        char* tok = hotbuf;
        while (tok && n_hot < 32) {
            char* next = strchr(tok, ',');
            if (next) *next++ = '\0';
            while (*tok == ' ') tok++;
            if (*tok) hot[n_hot++] = tok;
            tok = next;
        }
        ip.hotwords = hot;
        ip.num_hotwords = n_hot;
    }

    char stream_f[16], gran[16];
    part_to_str(vv_http_part(parts, np, "stream"), stream_f, sizeof(stream_f));
    part_to_str(vv_http_part(parts, np, "stream_granularity"), gran,
                sizeof(gran));
    if (strcmp(stream_f, "true") == 0 || strcmp(stream_f, "1") == 0) {
        sse_transcribe(sv, res, pcm, n_samples, sample_rate, &ip,
                       prompt[0] ? hotbuf_joined(&ip, prompt) : NULL,
                       strcmp(gran, "token") == 0);
        vv_free(pcm);
        return;
    }

    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    s = vv_engine_transcribe(sv->engine, pcm, n_samples, sample_rate,
                             &ip, &tr, &perf);
    vv_free(pcm);

    if (s != VV_OK || !tr) {
        vv_mutex_lock(&sv->stats_lock); sv->n_errors++; vv_mutex_unlock(&sv->stats_lock);
        if (s == VV_ERR_KV_POOL_EXHAUSTED) {
            /* The slots of a device ran their shared KV pool dry at once.
             * Nothing is wrong with the request; it fits once they finish. */
            vv_http_error(res, 503, "server_overloaded",
                          "KV cache pool exhausted by concurrent requests; "
                          "retry later");
            return;
        }
        char msg[256];
        snprintf(msg, sizeof(msg), "transcription failed: %s",
                 vv_status_str(s));
        vv_http_error(res, 500, "server_error", msg);
        return;
    }

    vv_mutex_lock(&sv->stats_lock);
    sv->n_requests++;
    sv->audio_seconds += tr->duration;
    sv->compute_seconds += perf.total_ms / 1000.0;
    vv_mutex_unlock(&sv->stats_lock);

    strbuf_t b; sb_init(&b);
    if (strcmp(fmt, "text") == 0) {
        vv_http_respond(res, 200, "text/plain; charset=utf-8",
                        tr->full_text ? tr->full_text : "",
                        tr->full_text ? strlen(tr->full_text) : 0);
    } else if (strcmp(fmt, "srt") == 0) {
        build_srt(&b, tr);
        vv_http_respond(res, 200, "text/plain; charset=utf-8",
                        b.buf ? b.buf : "", b.len);
    } else if (strcmp(fmt, "vtt") == 0) {
        build_vtt(&b, tr);
        vv_http_respond(res, 200, "text/vtt; charset=utf-8",
                        b.buf ? b.buf : "", b.len);
    } else if (strcmp(fmt, "verbose_json") == 0) {
        build_verbose_json(&b, tr, &perf);
        vv_http_respond_json(res, 200, b.buf ? b.buf : "{}");
    } else {
        sb_puts(&b, "{\"text\":\"");
        sb_json_escaped(&b, tr->full_text ? tr->full_text : "");
        sb_puts(&b, "\"}");
        vv_http_respond_json(res, 200, b.buf ? b.buf : "{}");
    }
    sb_free(&b);

    VV_LOG_I("server: %s %.1fs audio -> %d segments, RTF %.3f",
             file->filename[0] ? file->filename : "(upload)",
             tr->duration, tr->num_segments, perf.rtf);
    vv_transcription_free(tr);
}

/* ─── Streaming: SSE (stream=true) and the WebSocket endpoint ───────────── */

/*
 * The hotwords joined the way the pipeline joins them for the prompt
 * ("A, B"), written back over the raw `prompt` field (1024 bytes), whose
 * split copy the hotword list points into. A streaming session takes the
 * joined string.
 */
static const char* hotbuf_joined(const vv_inference_params_t* ip,
                                 char* prompt) {
    char joined[VV_HOTWORDS_MAX];
    vv_hotwords_join(ip->hotwords, ip->num_hotwords, joined, sizeof(joined));
    snprintf(prompt, 1024, "%s", joined);
    return prompt[0] ? prompt : NULL;
}

static void count_request(vv_server_t* sv, bool ok, double audio_s,
                          double compute_s) {
    vv_mutex_lock(&sv->stats_lock);
    if (ok) {
        sv->n_requests++;
        sv->audio_seconds += audio_s;
        sv->compute_seconds += compute_s;
    } else {
        sv->n_errors++;
    }
    vv_mutex_unlock(&sv->stats_lock);
}

/* What a streaming client is told when its session ends early. */
static const char* stream_status_msg(vv_status_t s) {
    switch (s) {
        case VV_ERR_OVERFLOW:
            return "the session outgrew its KV window (the server's "
                   "--max-seq-len)";
        case VV_ERR_KV_POOL_EXHAUSTED:
            return "KV cache pool exhausted by concurrent sessions; retry "
                   "later";
        case VV_ERR_BUSY:
            return "every slot is held by a live session; retry later";
        default:
            return vv_status_str(s);
    }
}

/* An OpenAI error envelope for status `s`. */
static void error_json(strbuf_t* b, vv_status_t s, const char* prefix) {
    const bool busy = s == VV_ERR_KV_POOL_EXHAUSTED || s == VV_ERR_BUSY;
    sb_puts(b, "{\"type\":\"error\",\"error\":{\"type\":\"");
    sb_puts(b, busy ? "server_overloaded" : "server_error");
    sb_puts(b, "\",\"message\":\"");
    if (prefix) sb_puts(b, prefix);
    sb_json_escaped(b, stream_status_msg(s));
    sb_puts(b, "\"}}");
}

typedef struct {
    vv_http_res_t*      res;
    vv_engine_stream_t* es;
    bool                token_deltas;
    bool                gone;        /* a write failed: the client left */
    int64_t             chunks;
} sse_state_t;

static void sse_event(void* user, const vv_stream_event_t* ev) {
    sse_state_t* st = (sse_state_t*)user;
    if (st->gone) return;
    const bool delta = (ev->type == VV_STREAM_EVENT_DELTA && st->token_deltas) ||
                       (ev->type == VV_STREAM_EVENT_CHUNK && !st->token_deltas);
    strbuf_t b; sb_init(&b);
    if (delta) {
        if (ev->type == VV_STREAM_EVENT_CHUNK) st->chunks++;
        if (ev->text_len == 0) { sb_free(&b); return; }
        sb_puts(&b, "{\"type\":\"transcript.text.delta\",\"delta\":\"");
        sb_json_escaped(&b, ev->text);
        sb_printf(&b, "\",\"x_vibevoice\":{\"chunk\":%lld",
                  (long long)ev->chunk_index);
        if (ev->type == VV_STREAM_EVENT_CHUNK)
            sb_printf(&b, ",\"start\":%.3f,\"end\":%.3f", ev->audio_start,
                      ev->audio_end);
        sb_puts(&b, "}}");
        if (!vv_http_sse_send(st->res, "transcript.text.delta", b.buf))
            st->gone = true;
    } else if (ev->type == VV_STREAM_EVENT_CHUNK) {
        st->chunks++;
    }
    sb_free(&b);
    /* Between chunks nothing is written, so a write cannot fail: look at
     * the socket instead, once per token. */
    if (!st->gone && ev->type == VV_STREAM_EVENT_DELTA &&
        vv_http_peer_gone(st->res))
        st->gone = true;
    /* A client that left cancels the session: the slot is released at the
     * next token, not after the rest of the file. */
    if (st->gone && st->es) vv_engine_stream_cancel(st->es);
}

static void sse_send_done(sse_state_t* st, const char* text, int64_t chunks,
                          double duration, double rtf) {
    strbuf_t b; sb_init(&b);
    sb_puts(&b, "{\"type\":\"transcript.text.done\",\"text\":\"");
    sb_json_escaped(&b, text ? text : "");
    sb_printf(&b, "\",\"x_vibevoice\":{\"chunks\":%lld,\"duration\":%.3f,"
                  "\"rtf\":%.4f}}", (long long)chunks, duration, rtf);
    if (!vv_http_sse_send(st->res, "transcript.text.done", b.buf))
        st->gone = true;
    sb_free(&b);
}

/*
 * POST /v1/audio/transcriptions with stream=true. On a streaming model the
 * upload goes through a session and each chunk's text goes out as it is
 * produced, paced by compute; on a batch model the transcript is one delta.
 */
static void sse_transcribe(vv_server_t* sv, vv_http_res_t* res,
                           const float* pcm, int n_samples, int sample_rate,
                           const vv_inference_params_t* ip,
                           const char* context_info, bool token_deltas) {
    const double t0 = vv_time_ms();
    const double duration = sample_rate > 0 ? (double)n_samples / sample_rate
                                            : 0.0;
    sse_state_t st;
    memset(&st, 0, sizeof(st));
    st.res = res;
    st.token_deltas = token_deltas;

    if (!vv_engine_is_streaming(sv->engine)) {
        vv_transcription_t* tr = NULL;
        vv_perf_metrics_t perf;
        memset(&perf, 0, sizeof(perf));
        const vv_status_t s = vv_engine_transcribe(sv->engine, pcm, n_samples,
                                                   sample_rate, ip, &tr, &perf);
        if (s != VV_OK || !tr) {
            count_request(sv, false, 0, 0);
            vv_http_error(res, s == VV_ERR_KV_POOL_EXHAUSTED ? 503 : 500,
                          s == VV_ERR_KV_POOL_EXHAUSTED ? "server_overloaded"
                                                        : "server_error",
                          vv_status_str(s));
            return;
        }
        vv_http_respond_begin(res, 200, "text/event-stream");
        vv_stream_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = VV_STREAM_EVENT_CHUNK;
        ev.text = tr->full_text ? tr->full_text : "";
        ev.text_len = strlen(ev.text);
        ev.audio_end = tr->duration;
        st.token_deltas = false;
        sse_event(&st, &ev);
        sse_send_done(&st, ev.text, 1, tr->duration, perf.rtf);
        count_request(sv, true, tr->duration, perf.total_ms / 1000.0);
        vv_transcription_free(tr);
        return;
    }

    vv_stream_params_t sp;
    vv_stream_params_default(&sp);
    sp.context_info = context_info;
    sp.on_event = sse_event;
    sp.user = &st;
    /* The whole upload's KV, or a 503 now rather than a stall halfway. */
    sp.kv_reserve_sec = duration;
    sp.kv_no_wait = true;
    (void)ip;

    vv_engine_stream_t* es = NULL;
    vv_status_t s = vv_engine_stream_open_ex(sv->engine, &sp, sample_rate,
                                             sv->stream_slot_wait_ms, &es);
    if (s != VV_OK) {
        const bool busy = s == VV_ERR_KV_POOL_EXHAUSTED || s == VV_ERR_BUSY;
        count_request(sv, false, 0, 0);
        vv_http_error(res, busy ? 503 : 500,
                      busy ? "server_overloaded" : "server_error",
                      stream_status_msg(s));
        return;
    }
    st.es = es;
    if (!vv_http_respond_begin(res, 200, "text/event-stream")) st.gone = true;
    if (st.gone) vv_engine_stream_cancel(es);

    s = vv_engine_stream_push(es, pcm, (size_t)n_samples);
    if (s == VV_OK) s = vv_engine_stream_finish(es);
    const double compute = (vv_time_ms() - t0) / 1000.0;
    if (s == VV_OK) {
        sse_send_done(&st, vv_stream_transcript(vv_engine_stream_session(es)),
                      st.chunks, duration,
                      duration > 0 ? compute / duration : 0.0);
        count_request(sv, true, duration, compute);
        VV_LOG_I("server: streamed %.1fs audio in %lld chunks, RTF %.3f",
                 duration, (long long)st.chunks,
                 duration > 0 ? compute / duration : 0.0);
    } else if (s == VV_ERR_CANCELLED) {
        VV_LOG_I("server: client left after %lld chunks; session cancelled",
                 (long long)st.chunks);
        count_request(sv, false, 0, 0);
    } else {
        strbuf_t b; sb_init(&b);
        error_json(&b, s, "transcription failed: ");
        vv_http_sse_send(res, "error", b.buf);
        sb_free(&b);
        count_request(sv, false, 0, 0);
    }
    vv_engine_stream_close(es);
}

/* ─── WebSocket /v1/audio/stream ────────────────────────────────────────── */

#define WS_MAX_MESSAGE (4u << 20)   /* one message of PCM: 4 MB */
#define WS_CLOSE_TRY_AGAIN 1013     /* RFC 6455 "try again later" */

typedef struct {
    vv_server_t*        sv;
    vv_http_res_t*      res;
    vv_engine_stream_t* es;
    bool                f32;         /* pcm_f32le, else pcm_s16le */
    int                 sample_rate;
    uint8_t             carry[4];    /* a sample split across two frames */
    int                 n_carry;
    float*              pcm;
    size_t              pcm_cap;
    bool                done;        /* stop reading */
    bool                gone;        /* a send failed */
    int                 close_code;
    char                close_reason[96];
    vv_status_t         failed;
    double              t_start;
    double              t_data;      /* last audio or JSON message */
    int64_t             samples;
} ws_state_t;

static bool ws_text(ws_state_t* w, const char* json) {
    if (w->gone) return false;
    if (!vv_http_ws_send(w->res, VV_WS_OP_TEXT, json, strlen(json))) {
        w->gone = true;
        if (w->es) vv_engine_stream_cancel(w->es);
        return false;
    }
    return true;
}

static void ws_close(ws_state_t* w, int code, const char* reason) {
    if (w->gone) return;
    uint8_t buf[2 + 96];
    size_t n = 2;
    buf[0] = (uint8_t)(code >> 8);
    buf[1] = (uint8_t)(code & 0xff);
    if (reason) {
        size_t rl = strlen(reason);
        if (rl > 90) rl = 90;
        memcpy(buf + 2, reason, rl);
        n += rl;
    }
    vv_http_ws_send(w->res, VV_WS_OP_CLOSE, buf, n);
}

static void ws_fail(ws_state_t* w, vv_status_t s, int code,
                    const char* type, const char* msg) {
    strbuf_t b; sb_init(&b);
    sb_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"");
    sb_puts(&b, type);
    sb_puts(&b, "\",\"message\":\"");
    sb_json_escaped(&b, msg ? msg : stream_status_msg(s));
    sb_puts(&b, "\"}}");
    ws_text(w, b.buf);
    sb_free(&b);
    w->failed = s;
    w->done = true;
    w->close_code = code;
}

static void ws_event(void* user, const vv_stream_event_t* ev) {
    ws_state_t* w = (ws_state_t*)user;
    strbuf_t b; sb_init(&b);
    if (ev->type == VV_STREAM_EVENT_CHUNK) {
        sb_puts(&b, "{\"type\":\"transcript.text.delta\",\"delta\":\"");
        sb_json_escaped(&b, ev->text);
        sb_printf(&b, "\",\"chunk\":%lld,\"start\":%.3f,\"end\":%.3f}",
                  (long long)ev->chunk_index, ev->audio_start, ev->audio_end);
        ws_text(w, b.buf);
    } else if (ev->type == VV_STREAM_EVENT_DONE) {
        sb_puts(&b, "{\"type\":\"transcript.text.done\",\"text\":\"");
        sb_json_escaped(&b, ev->text);
        sb_printf(&b, "\",\"duration\":%.3f}", ev->audio_end);
        ws_text(w, b.buf);
    } else if (ev->type == VV_STREAM_EVENT_DELTA && !w->gone &&
               vv_http_peer_gone(w->res)) {
        /* Deltas are not sent here, so no write would fail until the chunk
         * is done: a client that left is noticed at the next token. */
        w->gone = true;
        if (w->es) vv_engine_stream_cancel(w->es);
    }
    sb_free(&b);
}

static const char* json_str(const cJSON* o, const char* key) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, key);
    return cJSON_IsString(v) ? v->valuestring : NULL;
}

static void ws_start(ws_state_t* w, const cJSON* msg) {
    if (w->es) {
        ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                "session.start sent twice");
        return;
    }
    const cJSON* sr = cJSON_GetObjectItemCaseSensitive(msg, "sample_rate");
    /* The range on the double, before any cast: converting a value an int
     * cannot hold (1e300, NaN) is undefined behaviour. */
    const double rate = cJSON_IsNumber(sr) ? sr->valuedouble : 24000.0;
    if (!(rate >= 8000.0 && rate <= 192000.0)) {
        ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                "sample_rate must be 8000..192000");
        return;
    }
    w->sample_rate = (int)rate;
    const char* fmt = json_str(msg, "format");
    if (!fmt || strcmp(fmt, "pcm_s16le") == 0) w->f32 = false;
    else if (strcmp(fmt, "pcm_f32le") == 0) w->f32 = true;
    else {
        ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                "format is pcm_s16le or pcm_f32le");
        return;
    }
    /* "A,B" -> "A, B", the way every prompt joins hotwords. */
    char ctx_info[VV_HOTWORDS_MAX];
    vv_hotwords_join_csv(json_str(msg, "hotwords"), ctx_info,
                         sizeof(ctx_info));

    vv_stream_params_t sp;
    vv_stream_params_default(&sp);
    sp.context_info = ctx_info[0] ? ctx_info : NULL;
    sp.on_event = ws_event;
    sp.user = w;
    /* Admission against the shared KV pool, and a pool that runs dry later
     * ends this session rather than parking it (its socket unread). */
    sp.kv_reserve_sec = w->sv->stream_reserve_sec;
    sp.kv_no_wait = true;
    const vv_status_t s = vv_engine_stream_open_ex(w->sv->engine, &sp,
                                                   w->sample_rate,
                                                   w->sv->stream_slot_wait_ms,
                                                   &w->es);
    if (s != VV_OK) {
        const bool busy = s == VV_ERR_KV_POOL_EXHAUSTED || s == VV_ERR_BUSY;
        ws_fail(w, s, busy ? WS_CLOSE_TRY_AGAIN : 1011,
                busy ? "server_overloaded" : "server_error", NULL);
        return;
    }
    w->t_start = vv_time_ms();
    char ok[128];
    snprintf(ok, sizeof(ok), "{\"type\":\"session.started\",\"sample_rate\":%d,"
             "\"format\":\"%s\"}", w->sample_rate,
             w->f32 ? "pcm_f32le" : "pcm_s16le");
    ws_text(w, ok);
}

static void ws_pcm(ws_state_t* w, const uint8_t* data, size_t len) {
    if (!w->es) {
        ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                "send session.start before audio");
        return;
    }
    const int bps = w->f32 ? 4 : 2;
    const size_t total = (size_t)w->n_carry + len;
    const size_t n = total / (size_t)bps;
    if (n > w->pcm_cap) {
        float* np = (float*)vv_realloc(w->pcm, n * sizeof(float));
        if (!np) {
            ws_fail(w, VV_ERR_OUT_OF_MEMORY, 1011, "server_error", NULL);
            return;
        }
        w->pcm = np;
        w->pcm_cap = n;
    }
    size_t in = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t b[4];
        for (int k = 0; k < bps; k++) {
            if (w->n_carry > 0) {
                b[k] = w->carry[0];
                memmove(w->carry, w->carry + 1, (size_t)--w->n_carry);
            } else {
                b[k] = data[in++];
            }
        }
        if (w->f32) {
            float f;
            memcpy(&f, b, 4);
            w->pcm[i] = f;
        } else {
            const int16_t v = (int16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            w->pcm[i] = (float)v / 32768.0f;
        }
    }
    while (in < len) w->carry[w->n_carry++] = data[in++];
    w->samples += (int64_t)n;
    const vv_status_t s = vv_engine_stream_push(w->es, w->pcm, n);
    if (s == VV_ERR_CANCELLED) { w->done = true; return; }
    if (s != VV_OK)
        ws_fail(w, s, s == VV_ERR_KV_POOL_EXHAUSTED ? WS_CLOSE_TRY_AGAIN : 1011,
                s == VV_ERR_KV_POOL_EXHAUSTED ? "server_overloaded"
                                              : "server_error",
                NULL);
}

static void ws_message(void* user, int opcode, const uint8_t* data,
                       size_t len) {
    ws_state_t* w = (ws_state_t*)user;
    if (w->done) return;
    if (opcode == VV_WS_OP_PING) {
        if (!vv_http_ws_send(w->res, VV_WS_OP_PONG, data, len)) w->gone = true;
        return;
    }
    if (opcode == VV_WS_OP_PONG) return;
    if (opcode == VV_WS_OP_CLOSE) {
        /* The client closed: whatever was not finished is abandoned. */
        w->done = true;
        w->close_code = VV_WS_CLOSE_NORMAL;
        return;
    }
    /* Only audio and requests keep a session alive; pings and pongs are
     * answered but do not, or a client could hold a slot on pings alone. */
    w->t_data = vv_time_ms();
    if (opcode == VV_WS_OP_BINARY) {
        ws_pcm(w, data, len);
        /* The chunks it ran are server time, not client silence. */
        w->t_data = vv_time_ms();
        return;
    }

    cJSON* msg = cJSON_ParseWithLength((const char*)data, len);
    const char* type = msg ? json_str(msg, "type") : NULL;
    if (!type) {
        ws_fail(w, VV_ERR_PARSE, 1008, "invalid_request_error",
                "expected a JSON message with a \"type\"");
    } else if (strcmp(type, "session.start") == 0) {
        ws_start(w, msg);
    } else if (strcmp(type, "session.finish") == 0) {
        if (!w->es) {
            ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                    "no session to finish");
        } else {
            const vv_status_t s = vv_engine_stream_finish(w->es);
            if (s == VV_OK) {
                w->done = true;
                w->close_code = VV_WS_CLOSE_NORMAL;
            } else if (s == VV_ERR_CANCELLED) {
                w->done = true;
            } else {
                ws_fail(w, s, 1011, "server_error", NULL);
            }
        }
    } else {
        ws_fail(w, VV_ERR_INVALID_ARG, 1008, "invalid_request_error",
                "unknown message type");
    }
    cJSON_Delete(msg);
}

static void handle_ws_stream(vv_server_t* sv, const vv_http_req_t* req,
                             vv_http_res_t* res) {
    if (!vv_engine_is_streaming(sv->engine)) {
        vv_http_error(res, 400, "invalid_request_error",
                      "the loaded model transcribes whole clips; live "
                      "streaming needs VibeVoice-ASR-Streaming-7B");
        return;
    }
    if (!vv_http_ws_accept(req, res)) return;

    ws_state_t w;
    memset(&w, 0, sizeof(w));
    w.sv = sv;
    w.res = res;
    w.close_code = VV_WS_CLOSE_NORMAL;
    vv_ws_parser_t parser;
    vv_ws_parser_init(&parser, WS_MAX_MESSAGE);

    /* Frames that rode in with the upgrade request come first. */
    vv_status_t ps = VV_OK;
    if (req->body && req->body_len)
        ps = vv_ws_parser_feed(&parser, (const uint8_t*)req->body,
                               req->body_len, ws_message, &w);
    uint8_t* buf = (uint8_t*)vv_alloc(64 * 1024);
    /* Idle is counted from the last audio or JSON message (connecting
     * counts as one), not the last byte: pings do not hold a slot. */
    w.t_data = vv_time_ms();
    const int idle_ms = sv->stream_idle_ms > 0 ? sv->stream_idle_ms : 60000;
    while (buf && ps == VV_OK && !w.done && !w.gone) {
        const int n = vv_http_read(res, buf, 64 * 1024,
                                   idle_ms < 1000 ? idle_ms : 1000);
        if (n > 0) {
            ps = vv_ws_parser_feed(&parser, buf, (size_t)n, ws_message, &w);
        } else if (n != -2) {
            w.gone = true;                      /* the client went away */
            break;
        }
        if (!w.done && !w.gone && vv_time_ms() - w.t_data >= idle_ms) {
            char why[96];
            snprintf(why, sizeof(why), "no audio or message for %d s",
                     idle_ms / 1000);
            ws_fail(&w, VV_ERR_IO, 1008, "invalid_request_error", why);
        }
        /*
         * A trickle of audio (a few samples a minute) would keep the idle
         * timer happy and the slot held for ever. Past one idle window a
         * session must have sent at least a tenth of the wall time it has
         * held the slot for; a live microphone sends all of it.
         */
        if (!w.done && !w.gone && w.es && w.sample_rate > 0) {
            const double held = vv_time_ms() - w.t_start;
            const double audio_ms = (double)w.samples * 1000.0 /
                                    (double)w.sample_rate;
            if (held > idle_ms && audio_ms * 10.0 < held - idle_ms)
                ws_fail(&w, VV_ERR_IO, 1008, "invalid_request_error",
                        "audio arrives at under a tenth of real time");
        }
    }
    if (ps != VV_OK && !w.gone) {
        ws_close(&w, parser.close_code ? parser.close_code : 1002, NULL);
        w.close_code = 0;
    }
    vv_free(buf);

    const double compute = w.t_start > 0 ? (vv_time_ms() - w.t_start) / 1000.0
                                         : 0.0;
    if (w.es) {
        const bool ok = w.failed == VV_OK && !w.gone &&
                        w.close_code == VV_WS_CLOSE_NORMAL;
        vv_stream_stats_t st;
        vv_stream_get_stats(vv_engine_stream_session(w.es), &st);
        count_request(sv, ok, (double)st.samples / 24000.0, compute);
        VV_LOG_I("server: websocket session %s: %.1fs audio, %lld chunks, "
                 "%.0f ms per chunk", ok ? "done" : "ended",
                 (double)st.samples / 24000.0, (long long)st.chunks,
                 st.chunks ? (st.prefill_ms + st.decode_ms) / st.chunks : 0.0);
        if (w.gone) vv_engine_stream_cancel(w.es);
        vv_engine_stream_close(w.es);
    }
    if (!w.gone && w.close_code) ws_close(&w, w.close_code, NULL);
    vv_ws_parser_free(&parser);
    vv_free(w.pcm);
}

/*
 * Compare in time that depends only on the lengths, not on how many leading
 * bytes of a guess are right: strcmp() returns at the first difference,
 * which lets a key be recovered one byte at a time from response latency.
 */
static bool key_equal(const char* given, const char* key) {
    const size_t n = strlen(key);
    const size_t m = strlen(given);
    unsigned char diff = (unsigned char)(m != n);
    for (size_t i = 0; i < n; i++) {
        const unsigned char g = i < m ? (unsigned char)given[i] : 0;
        diff |= (unsigned char)(g ^ (unsigned char)key[i]);
    }
    return diff == 0;
}

static bool authorized(const vv_server_t* sv, const vv_http_req_t* req) {
    if (!sv->api_key[0]) return true;
    const char* h = vv_http_header(req, "Authorization");
    if (!h && strcmp(req->path, "/v1/audio/stream") == 0) {
        /* Browsers cannot set headers on a WebSocket: ?api_key=..., which
         * a browser percent-encodes ('+', '/', '=' of a base64 key). A
         * value longer than any key (sv->api_key is 256 bytes) cannot
         * match, so it is refused rather than cut to size. */
        char key[sizeof(sv->api_key) + 1];
        const char* q = req->query;
        while (q && *q) {
            if (strncmp(q, "api_key=", 8) == 0) {
                size_t n = 0;
                q += 8;
                while (*q && *q != '&') {
                    if (n + 1 >= sizeof(key)) return false;
                    int c = (unsigned char)*q++;
                    if (c == '+') {
                        c = ' ';
                    } else if (c == '%' && isxdigit((unsigned char)q[0]) &&
                               isxdigit((unsigned char)q[1])) {
                        const char hx[3] = { q[0], q[1], 0 };
                        c = (int)strtol(hx, NULL, 16);
                        q += 2;
                    }
                    if (c == 0) return false;
                    key[n++] = (char)c;
                }
                key[n] = '\0';
                return key_equal(key, sv->api_key);
            }
            q = strchr(q, '&');
            if (q) q++;
        }
        return false;
    }
    if (!h) return false;
    if (strncmp(h, "Bearer ", 7) == 0) h += 7;
    return key_equal(h, sv->api_key);
}

static void route(const vv_http_req_t* req, vv_http_res_t* res, void* user) {
    vv_server_t* sv = (vv_server_t*)user;
    const char* p = req->path;

    if (strcmp(p, "/health") == 0 || strcmp(p, "/healthz") == 0 ||
        strcmp(p, "/v1/health") == 0) {
        handle_health(sv, res);
        return;
    }
    if (strcmp(p, "/metrics") == 0) { handle_metrics(sv, res); return; }

    if (!authorized(sv, req)) {
        vv_http_error(res, 401, "invalid_request_error", "invalid api key");
        return;
    }

    if (strcmp(p, "/v1/models") == 0) { handle_models(sv, res, false); return; }
    if (strncmp(p, "/v1/models/", 11) == 0) {
        handle_models(sv, res, true);
        return;
    }
    if (strcmp(p, "/v1/audio/transcriptions") == 0 ||
        strcmp(p, "/v1/audio/translations") == 0) {
        if (!vv_queue_enter(&sv->queue)) {
            vv_http_error(res, 503, "server_overloaded", "transcription queue is full; retry later");
            return;
        }
        handle_transcriptions(sv, req, res);
        vv_queue_leave(&sv->queue);
        return;
    }
    if (strcmp(p, "/v1/audio/stream") == 0) {
        /*
         * A live session holds its slot for as long as its audio lasts, so
         * it is not queued behind other holders: a slot free within
         * --stream-slot-wait, or 503 before the upgrade.
         */
        if (!vv_engine_is_streaming(sv->engine)) {
            handle_ws_stream(sv, req, res);     /* answers 400 */
            return;
        }
        if (!vv_queue_try_enter(&sv->queue, sv->stream_slot_wait_ms)) {
            vv_http_error(res, 503, "server_overloaded",
                          "every slot is held by a live session; retry "
                          "later");
            return;
        }
        handle_ws_stream(sv, req, res);
        vv_queue_leave(&sv->queue);
        return;
    }
    if (strcmp(p, "/") == 0) {
        char buf[768];
        const int n = snprintf(buf, sizeof(buf),
            "vibevoice.c server\n\n"
            "model: %s\nslots: %d\n\n"
            "POST /v1/audio/transcriptions   multipart: file, "
            "[response_format=json|verbose_json|text|srt|vtt], [prompt], "
            "[stream=true]\n"
            "GET  /v1/audio/stream           WebSocket, live PCM "
            "(streaming models)\n"
            "GET  /v1/models\nGET  /health\nGET  /metrics\n",
            sv->model_name, vv_engine_slots(sv->engine));
        vv_http_respond(res, 200, "text/plain; charset=utf-8", buf, (size_t)n);
        return;
    }

    vv_http_error(res, 404, "invalid_request_error", "no such endpoint");
}

/* ─── Entry points ───────────────────────────────────────────────────────── */

vv_status_t vv_server_run(vv_engine_t* engine, const vv_server_params_t* params,
                          vv_server_t** handle) {
    if (!engine) return VV_ERR_NULL_PTR;

    vv_server_params_t p = params ? *params : vv_server_params_default();
    if (p.queue_size < 0 || p.queue_size > 4096) return VV_ERR_INVALID_ARG;

    vv_server_t* sv = (vv_server_t*)vv_alloc(sizeof(vv_server_t));
    if (!sv) return VV_ERR_OUT_OF_MEMORY;
    memset(sv, 0, sizeof(*sv));
    sv->engine = engine;
    vv_queue_init(&sv->queue, vv_engine_slots(engine), p.queue_size);
    vv_mutex_init(&sv->stats_lock);
    sv->started_at = (uint64_t)time(NULL);

    if (p.model_name && p.model_name[0]) {
        snprintf(sv->model_name, sizeof(sv->model_name), "%s", p.model_name);
    } else {
        /* Default to the model directory's basename, as GPUStack expects. */
        const char* dir = vv_engine_model_dir(engine);
        const char* base = dir;
        for (const char* q = dir; *q; q++)
            if (*q == '/' || *q == '\\') base = q + 1;
        snprintf(sv->model_name, sizeof(sv->model_name), "%s",
                 base[0] ? base : "vibevoice-asr");
    }
    if (p.api_key) snprintf(sv->api_key, sizeof(sv->api_key), "%s", p.api_key);
    sv->acoustic_sampling = p.acoustic_sampling;
    sv->stream_idle_ms = p.stream_idle_ms > 0 ? p.stream_idle_ms : 60000;
    sv->stream_slot_wait_ms = p.stream_slot_wait_ms >= 0
                              ? p.stream_slot_wait_ms : 5000;
    sv->stream_reserve_sec = p.stream_reserve_sec > 0.0
                             ? p.stream_reserve_sec : 0.0;

    const int conns = p.max_conns > 0 ? p.max_conns
                                      : vv_engine_slots(engine) + p.queue_size + 8;
    if (handle) *handle = sv;

    VV_LOG_I("server: model '%s', %d slot(s)%s",
             sv->model_name, vv_engine_slots(engine),
             sv->api_key[0] ? ", api key required" : "");
    if (vv_engine_is_streaming(engine))
        VV_LOG_I("server: live sessions hold a slot each; %.0f s of KV "
                 "reserved per session, dropped after %d s without audio, "
                 "%d ms to wait for a slot", sv->stream_reserve_sec,
                 sv->stream_idle_ms / 1000, sv->stream_slot_wait_ms);
    if (sv->acoustic_sampling)
        VV_LOG_W("server: acoustic sampling is '%s', so identical audio may "
                 "come back worded differently",
                 vv_acoustic_sampling_name(
                     (vv_acoustic_sampling_t)sv->acoustic_sampling));

    vv_status_t s = vv_http_serve(p.host, p.port, conns, p.max_body,
                                  route, sv, &sv->http);
    vv_queue_destroy(&sv->queue);
    vv_mutex_destroy(&sv->stats_lock);
    vv_free(sv);
    if (handle) *handle = NULL;
    return s;
}

void vv_server_stop(vv_server_t* s) {
    if (s && s->http) vv_http_shutdown(s->http);
}
