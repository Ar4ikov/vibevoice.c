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
    if (!b->buf) return;
    memcpy(b->buf + b->len, s, n + 1);
    b->len += n;
}

static void sb_printf(strbuf_t* b, const char* fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sb_puts(b, tmp);
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
                    if (b->buf) { b->buf[b->len++] = (char)*p; b->buf[b->len] = '\0'; }
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
    char buf[640];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"ok\",\"model\":\"%s\",\"version\":\"%s\","
             "\"build\":\"%s\",\"slots\":%d,"
             "\"busy\":%d,\"completed\":%llu}",
             sv->model_name, vv_version(), vv_build_ref(),
             vv_engine_slots(sv->engine), busy,
             (unsigned long long)done);
    vv_http_respond_json(res, 200, buf);
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

    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    s = vv_engine_transcribe(sv->engine, pcm, n_samples, sample_rate,
                             &ip, &tr, &perf);
    vv_free(pcm);

    if (s != VV_OK || !tr) {
        vv_mutex_lock(&sv->stats_lock); sv->n_errors++; vv_mutex_unlock(&sv->stats_lock);
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

static bool authorized(const vv_server_t* sv, const vv_http_req_t* req) {
    if (!sv->api_key[0]) return true;
    const char* h = vv_http_header(req, "Authorization");
    if (!h) return false;
    if (strncmp(h, "Bearer ", 7) == 0) h += 7;
    return strcmp(h, sv->api_key) == 0;
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
    if (strcmp(p, "/") == 0) {
        char buf[768];
        const int n = snprintf(buf, sizeof(buf),
            "vibevoice.c server\n\n"
            "model: %s\nslots: %d\n\n"
            "POST /v1/audio/transcriptions   multipart: file, "
            "[response_format=json|verbose_json|text|srt|vtt], [prompt]\n"
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

    const int conns = p.max_conns > 0 ? p.max_conns
                                      : vv_engine_slots(engine) + p.queue_size + 8;
    if (handle) *handle = sv;

    VV_LOG_I("server: model '%s', %d slot(s)%s",
             sv->model_name, vv_engine_slots(engine),
             sv->api_key[0] ? ", api key required" : "");
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
