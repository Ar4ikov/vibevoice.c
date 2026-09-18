/**
 * @file stream.c
 * @brief Streaming-7B session: chunk scheduler, chunk layout, text deltas.
 *
 * Everything here is model-free. The model work goes through
 * vv_stream_backend_t; the backend over a real inference context is phase 2
 * of issue #20 (see the TODO(phase 2) markers and docs/STREAMING.md).
 */

#include "vibevoice/stream.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

/* ─── Geometry ──────────────────────────────────────────────────────────── */

void vv_stream_geom_default(vv_stream_geom_t* g) {
    if (!g) return;
    g->sample_rate = 24000;
    g->frame_samples = 3200;
    g->chunk_frames = 22;
    g->lookahead_frames = 4;
    g->normalize_audio = false;
}

vv_status_t vv_stream_geom_validate(const vv_stream_geom_t* g) {
    if (!g) return VV_ERR_NULL_PTR;
    if (g->sample_rate <= 0 || g->frame_samples <= 0 ||
        g->chunk_frames <= 0 || g->lookahead_frames < 0)
        return VV_ERR_INVALID_ARG;
    /* A window is encoded in one go; keep it well inside int range. */
    if ((int64_t)(g->chunk_frames + g->lookahead_frames) * g->frame_samples >
        (int64_t)1 << 26)
        return VV_ERR_INVALID_ARG;
    return VV_OK;
}

int64_t vv_stream_chunk_samples(const vv_stream_geom_t* g) {
    return (int64_t)g->chunk_frames * g->frame_samples;
}

int64_t vv_stream_window_samples(const vv_stream_geom_t* g) {
    return (int64_t)(g->chunk_frames + g->lookahead_frames) * g->frame_samples;
}

int vv_stream_window_frames(const vv_stream_geom_t* g) {
    return g->chunk_frames + g->lookahead_frames;
}

int64_t vv_stream_n_windows(const vv_stream_geom_t* g, int64_t total) {
    const int64_t c = vv_stream_chunk_samples(g);
    return total <= 0 ? 0 : (total + c - 1) / c;
}

/* ─── Chunk scheduler ───────────────────────────────────────────────────── */

struct vv_stream_chunker {
    vv_stream_geom_t geom;
    int64_t chunk, window;
    float*  buf;       /* samples [base, base + len) of the stream */
    size_t  cap, len;
    int64_t base;      /* absolute index of buf[0] == next window start */
    int64_t total;     /* samples pushed */
    int64_t next_idx;  /* index of the next window */
    bool    finished;
};

vv_status_t vv_stream_chunker_create(const vv_stream_geom_t* g,
                                     vv_stream_chunker_t** out) {
    if (!g || !out) return VV_ERR_NULL_PTR;
    vv_status_t st = vv_stream_geom_validate(g);
    if (st != VV_OK) return st;
    vv_stream_chunker_t* c =
        (vv_stream_chunker_t*)vv_alloc(sizeof(vv_stream_chunker_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));
    c->geom = *g;
    c->chunk = vv_stream_chunk_samples(g);
    c->window = vv_stream_window_samples(g);
    c->cap = (size_t)(c->window + c->chunk);
    c->buf = (float*)vv_alloc(c->cap * sizeof(float));
    if (!c->buf) { vv_free(c); return VV_ERR_OUT_OF_MEMORY; }
    *out = c;
    return VV_OK;
}

void vv_stream_chunker_free(vv_stream_chunker_t* c) {
    if (!c) return;
    vv_free(c->buf);
    vv_free(c);
}

vv_status_t vv_stream_chunker_push(vv_stream_chunker_t* c,
                                   const float* pcm, size_t n) {
    if (!c) return VV_ERR_NULL_PTR;
    if (c->finished) return VV_ERR_INVALID_ARG;
    if (n == 0) return VV_OK;
    if (!pcm) return VV_ERR_NULL_PTR;
    if (c->len + n > c->cap) {
        size_t cap = c->cap;
        while (cap < c->len + n) cap *= 2;
        float* nb = (float*)vv_realloc(c->buf, cap * sizeof(float));
        if (!nb) return VV_ERR_OUT_OF_MEMORY;
        c->buf = nb;
        c->cap = cap;
    }
    memcpy(c->buf + c->len, pcm, n * sizeof(float));
    c->len += n;
    c->total += (int64_t)n;
    return VV_OK;
}

void vv_stream_chunker_finish(vv_stream_chunker_t* c) {
    if (c) c->finished = true;
}

int64_t vv_stream_chunker_ready(const vv_stream_chunker_t* c) {
    if (!c) return 0;
    const int64_t start = c->base;
    if (c->finished) {
        /* Every window starting inside the audio: upstream's
         * `while text_start < total`. */
        if (start >= c->total) return 0;
        return (c->total - start + c->chunk - 1) / c->chunk;
    }
    if (c->total < start + c->window) return 0;
    return (c->total - start - c->window) / c->chunk + 1;
}

int64_t vv_stream_chunker_total(const vv_stream_chunker_t* c) {
    return c ? c->total : 0;
}

int vv_stream_chunker_next(vv_stream_chunker_t* c, float* out,
                           vv_stream_window_t* info) {
    if (!c || !out || vv_stream_chunker_ready(c) == 0) return 0;

    const int64_t avail = (int64_t)c->len;       /* samples from base on */
    const int64_t n_valid = avail < c->window ? avail : c->window;
    memcpy(out, c->buf, (size_t)n_valid * sizeof(float));
    if (n_valid < c->window)
        memset(out + n_valid, 0, (size_t)(c->window - n_valid) * sizeof(float));

    if (info) {
        info->index = c->next_idx;
        info->start = c->base;
        info->n_valid = n_valid;
        info->is_last = c->finished && c->base + c->chunk >= c->total;
    }

    /* Advance one stride; keep the lookahead (and anything past it). */
    const int64_t drop = avail < c->chunk ? avail : c->chunk;
    memmove(c->buf, c->buf + drop, (size_t)(avail - drop) * sizeof(float));
    c->len = (size_t)(avail - drop);
    c->base += c->chunk;
    c->next_idx++;
    return 1;
}

/* ─── Prompt and layout ─────────────────────────────────────────────────── */

void vv_stream_ids_default(vv_stream_ids_t* ids) {
    if (!ids) return;
    ids->speech_start = 151646;
    ids->speech_end = 151647;
    ids->text_chunk_end = 151665;
    ids->eos = 151643;
}

vv_status_t vv_stream_ids_from_tokenizer(const vv_tokenizer_t* tok,
                                         vv_stream_ids_t* ids) {
    if (!tok || !ids) return VV_ERR_NULL_PTR;
    ids->speech_start = vv_tokenizer_special_id(tok, "<|object_ref_start|>");
    ids->speech_end = vv_tokenizer_special_id(tok, "<|object_ref_end|>");
    ids->text_chunk_end = vv_tokenizer_special_id(tok, "<|text_chunk_end|>");
    ids->eos = vv_tokenizer_special_id(tok, "<|endoftext|>");
    if (ids->speech_start < 0 || ids->speech_end < 0 ||
        ids->text_chunk_end < 0 || ids->eos < 0)
        return VV_ERR_NOT_FOUND;
    return VV_OK;
}

size_t vv_stream_prompt_text(const char* context_info, char* buf, size_t cap) {
    static const char head[] =
        "You are a helpful assistant that transcribes audio input into text "
        "output. Please transcribe the following audios streamingly with "
        "these keys: speaker, content";
    int n;
    /* Upstream tests `if context_info:`, so "" means no hotwords. */
    if (context_info && context_info[0])
        n = snprintf(buf, cap, "%s and extra info: %s\n", head, context_info);
    else
        n = snprintf(buf, cap, "%s\n", head);
    return n < 0 ? 0 : (size_t)n;
}

int vv_stream_chunk_layout(const vv_stream_ids_t* ids, int32_t lead,
                           int n_frames, int32_t* rows, int cap,
                           int* feat_offset) {
    if (!ids || !rows || n_frames <= 0) return -1;
    if (ids->speech_start < 0 || ids->speech_end < 0) return -1;
    const int need = (lead >= 0 ? 1 : 0) + n_frames + 2;
    if (cap < need) return -1;
    int r = 0;
    if (lead >= 0) rows[r++] = lead;
    rows[r++] = ids->speech_start;
    if (feat_offset) *feat_offset = r;
    for (int i = 0; i < n_frames; i++) rows[r++] = VV_STREAM_ROW_FEATURE;
    rows[r++] = ids->speech_end;
    return r;
}

/* ─── Text deltas ───────────────────────────────────────────────────────── */

/* What upstream removes from every chunk's text after decoding. */
static const char* const k_strip[] = {
    "<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>",
};
#define N_STRIP (sizeof(k_strip) / sizeof(k_strip[0]))

typedef struct {
    char*  p;
    size_t len, cap;
} sbuf_t;

static bool sb_reserve(sbuf_t* b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + extra + 1) cap *= 2;
    char* np = (char*)vv_realloc(b->p, cap);
    if (!np) return false;
    b->p = np;
    b->cap = cap;
    return true;
}

static bool sb_put(sbuf_t* b, const char* s, size_t n) {
    if (!sb_reserve(b, n)) return false;
    if (n) memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
    return true;
}

static void sb_consume(sbuf_t* b, size_t n) {
    memmove(b->p, b->p + n, b->len - n);
    b->len -= n;
    if (b->p) b->p[b->len] = '\0';
}

struct vv_stream_text {
    sbuf_t raw;   /* token bytes not yet decoded (an incomplete sequence) */
    sbuf_t text;  /* decoded text not yet released (a control-string prefix) */
    sbuf_t out;   /* the delta handed to the caller */
};

vv_status_t vv_stream_text_create(vv_stream_text_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    vv_stream_text_t* t = (vv_stream_text_t*)vv_alloc(sizeof(*t));
    if (!t) return VV_ERR_OUT_OF_MEMORY;
    memset(t, 0, sizeof(*t));
    if (!sb_reserve(&t->out, 0)) { vv_free(t); return VV_ERR_OUT_OF_MEMORY; }
    t->out.p[0] = '\0';
    *out = t;
    return VV_OK;
}

void vv_stream_text_free(vv_stream_text_t* t) {
    if (!t) return;
    vv_free(t->raw.p);
    vv_free(t->text.p);
    vv_free(t->out.p);
    vv_free(t);
}

/**
 * Length of the valid UTF-8 prefix of s[0..n) that starts a sequence at s[0]:
 *  > 0  a complete, valid sequence of that many bytes;
 *  = 0  a valid but incomplete prefix that the next bytes could complete;
 *  < 0  -k: s[0..k) is a maximal invalid subpart, one U+FFFD.
 * The ranges are Unicode's well-formed table (and CPython's decoder).
 */
static int utf8_seq(const unsigned char* s, size_t n) {
    const unsigned char c = s[0];
    if (c < 0x80) return 1;
    int need;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF)      { need = 1; }
    else if (c == 0xE0)              { need = 2; lo = 0xA0; }
    else if (c == 0xED)              { need = 2; hi = 0x9F; }
    else if (c >= 0xE1 && c <= 0xEF) { need = 2; }
    else if (c == 0xF0)              { need = 3; lo = 0x90; }
    else if (c == 0xF4)              { need = 3; hi = 0x8F; }
    else if (c >= 0xF1 && c <= 0xF3) { need = 3; }
    else return -1;
    for (int i = 1; i <= need; i++) {
        if ((size_t)i >= n) return 0;
        const unsigned char b = s[i];
        const unsigned char l = i == 1 ? lo : 0x80;
        const unsigned char h = i == 1 ? hi : 0xBF;
        if (b < l || b > h) return -i;
    }
    return need + 1;
}

static const char k_replacement[] = "\xEF\xBF\xBD";

/* Decode raw → text as far as possible; at a final flush an incomplete
 * tail becomes one U+FFFD. */
static bool decode_raw(vv_stream_text_t* t, bool final) {
    const unsigned char* s = (const unsigned char*)t->raw.p;
    size_t i = 0;
    while (i < t->raw.len) {
        const int k = utf8_seq(s + i, t->raw.len - i);
        if (k > 0) {
            if (!sb_put(&t->text, (const char*)s + i, (size_t)k)) return false;
            i += (size_t)k;
        } else if (k < 0) {
            if (!sb_put(&t->text, k_replacement, 3)) return false;
            i += (size_t)(-k);
        } else {
            if (!final) break;
            if (!sb_put(&t->text, k_replacement, 3)) return false;
            i = t->raw.len;
        }
    }
    if (t->raw.len) sb_consume(&t->raw, i);
    return true;
}

/* Does text[i..len) start one of the strip strings (proper prefix)? */
static bool is_strip_prefix(const char* s, size_t n) {
    for (size_t k = 0; k < N_STRIP; k++) {
        const size_t pl = strlen(k_strip[k]);
        if (n < pl && memcmp(s, k_strip[k], n) == 0) return true;
    }
    return false;
}

/* Move text → out, removing strip strings; unless final, keep a tail that
 * might still grow into one. */
static bool release_text(vv_stream_text_t* t, bool final) {
    const char* s = t->text.p;
    const size_t n = t->text.len;
    size_t i = 0;
    while (i < n) {
        if (s[i] == '<') {
            size_t hit = 0;
            for (size_t k = 0; k < N_STRIP && !hit; k++) {
                const size_t pl = strlen(k_strip[k]);
                if (n - i >= pl && memcmp(s + i, k_strip[k], pl) == 0) hit = pl;
            }
            if (hit) { i += hit; continue; }
            if (!final && is_strip_prefix(s + i, n - i)) break;
        }
        /* Copy up to the next '<' in one go. */
        size_t j = i + 1;
        while (j < n && s[j] != '<') j++;
        if (!sb_put(&t->out, s + i, j - i)) return false;
        i = j;
    }
    if (n) sb_consume(&t->text, i);
    return true;
}

static vv_status_t text_step(vv_stream_text_t* t, const char* bytes, size_t n,
                             bool final, const char** out, size_t* out_len) {
    t->out.len = 0;
    t->out.p[0] = '\0';
    if (n && !sb_put(&t->raw, bytes, n)) return VV_ERR_OUT_OF_MEMORY;
    if (!decode_raw(t, final) || !release_text(t, final))
        return VV_ERR_OUT_OF_MEMORY;
    if (out) *out = t->out.p;
    if (out_len) *out_len = t->out.len;
    return VV_OK;
}

vv_status_t vv_stream_text_push(vv_stream_text_t* t, const char* bytes,
                                size_t n, const char** out, size_t* out_len) {
    if (!t) return VV_ERR_NULL_PTR;
    if (n && !bytes) return VV_ERR_NULL_PTR;
    return text_step(t, bytes, n, false, out, out_len);
}

vv_status_t vv_stream_text_flush(vv_stream_text_t* t, const char** out,
                                 size_t* out_len) {
    if (!t) return VV_ERR_NULL_PTR;
    return text_step(t, NULL, 0, true, out, out_len);
}

vv_status_t vv_stream_token_bytes(const vv_tokenizer_t* tok, int32_t id,
                                  char* buf, size_t cap, size_t* n) {
    if (!tok || !n || (cap && !buf)) return VV_ERR_NULL_PTR;
    char* s = NULL;
    vv_status_t st = vv_tokenizer_decode_ex(tok, &id, 1, true, &s);
    if (st != VV_OK) return st;
    const size_t len = strlen(s);
    if (len > cap) { vv_free(s); return VV_ERR_OVERFLOW; }
    memcpy(buf, s, len);
    *n = len;
    vv_free(s);
    return VV_OK;
}

/* ─── Session ───────────────────────────────────────────────────────────── */

#define TOKEN_BYTES_MAX 256

struct vv_stream {
    vv_stream_backend_t be;
    vv_stream_params_t  p;
    char*               context_info;  /* owned copy */
    vv_stream_chunker_t* chunker;
    vv_stream_text_t*   text;
    float*              window;        /* window_samples */
    int32_t*            rows;          /* chunk layout */
    int                 rows_cap;
    sbuf_t              chunk_text;    /* current chunk, for the CHUNK event */
    sbuf_t              transcript;
    int32_t             pending_lead;  /* folded <|text_chunk_end|>, or -1 */
    vv_status_t         failed;
    bool                done;
    vv_stream_stats_t   stats;
};

void vv_stream_params_default(vv_stream_params_t* p) {
    if (!p) return;
    memset(p, 0, sizeof(*p));
    vv_stream_geom_default(&p->geom);
    vv_stream_ids_default(&p->ids);
    p->max_new_tokens = 256;
    p->fold_chunk_end = true;
}

static void emit(vv_stream_t* s, vv_stream_event_t* ev) {
    if (s->p.on_event) s->p.on_event(s->p.user, ev);
}

static vv_status_t fail(vv_stream_t* s, vv_status_t st, int64_t chunk) {
    if (s->failed == VV_OK) {
        s->failed = st;
        vv_stream_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = VV_STREAM_EVENT_ERROR;
        ev.chunk_index = chunk;
        ev.status = st;
        ev.text = vv_status_str(st);
        ev.text_len = strlen(ev.text);
        emit(s, &ev);
    }
    return s->failed;
}

static int64_t kv_len(vv_stream_t* s) {
    return s->be.kv_len ? s->be.kv_len(s->be.self) : 0;
}

static int64_t kv_cap(vv_stream_t* s) {
    return s->be.kv_capacity ? s->be.kv_capacity(s->be.self) : INT64_MAX;
}

/* Would `n` more positions overflow the cache? A session never writes past
 * it: the next chunk is refused and the session ends with VV_ERR_OVERFLOW.
 * TODO(phase 2): rollover (re-prompt with a fresh cache) as an option. */
static bool kv_fits(vv_stream_t* s, int64_t n) {
    return kv_len(s) + n <= kv_cap(s);
}

static vv_status_t emit_delta(vv_stream_t* s, int64_t chunk,
                              const char* d, size_t n) {
    if (n == 0) return VV_OK;
    if (!sb_put(&s->chunk_text, d, n)) return VV_ERR_OUT_OF_MEMORY;
    vv_stream_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = VV_STREAM_EVENT_DELTA;
    ev.chunk_index = chunk;
    ev.text = d;
    ev.text_len = n;
    emit(s, &ev);
    return VV_OK;
}

static void dump_chunk(const vv_stream_t* s, int64_t idx,
                       const int32_t* ids, int n_ids) {
    if (!vv_debug_dump_dir()) return;
    /* ids: [count, ids...] so that an empty chunk still leaves a file. */
    char name[64];
    int32_t* buf = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)(n_ids + 1));
    if (buf) {
        buf[0] = n_ids;
        if (n_ids) memcpy(buf + 1, ids, sizeof(int32_t) * (size_t)n_ids);
        snprintf(name, sizeof(name), "stream_c%04lld_ids", (long long)idx);
        vv_debug_dump(name, buf, sizeof(int32_t) * (size_t)(n_ids + 1));
        vv_free(buf);
    }
    snprintf(name, sizeof(name), "stream_c%04lld_text", (long long)idx);
    vv_debug_dump(name, s->chunk_text.p, s->chunk_text.len);
}

/* One window: prefill, greedy decode to a stop, then the chunk end. */
static vv_status_t run_chunk(vv_stream_t* s, const vv_stream_window_t* w) {
    const int n_frames = vv_stream_window_frames(&s->p.geom);
    const int64_t idx = w->index;
    int feat_off = 0;
    const int n_rows = vv_stream_chunk_layout(&s->p.ids, s->pending_lead,
                                              n_frames, s->rows, s->rows_cap,
                                              &feat_off);
    if (n_rows < 0) return fail(s, VV_ERR_INVALID_ARG, idx);
    /* The rows, plus the trailing <|text_chunk_end|> that has to follow. */
    if (!kv_fits(s, n_rows + 1)) return fail(s, VV_ERR_OVERFLOW, idx);

    vv_stream_chunk_t ch;
    ch.index = idx;
    ch.start_sample = w->start;
    ch.n_valid = w->n_valid;
    ch.n_frames = n_frames;
    ch.rows = s->rows;
    ch.n_rows = n_rows;
    ch.feat_offset = feat_off;

    int32_t tok = -1;
    vv_status_t st = s->be.prefill_chunk(s->be.self, &ch, s->window, &tok);
    if (st != VV_OK) return fail(s, st, idx);
    s->pending_lead = -1;

    s->chunk_text.len = 0;
    if (s->chunk_text.p) s->chunk_text.p[0] = '\0';

    int32_t gen_ids_small[256];
    int32_t* gen = gen_ids_small;
    int gen_cap = 256, n_gen = 0;
    vv_stream_stop_t stop = VV_STREAM_STOP_MAX_TOKENS;
    char bytes[TOKEN_BYTES_MAX];

    for (int step = 0; step < s->p.max_new_tokens; step++) {
        if (tok == s->p.ids.text_chunk_end) { stop = VV_STREAM_STOP_CHUNK_END; break; }
        if (tok == s->p.ids.eos) { stop = VV_STREAM_STOP_EOS; break; }

        if (n_gen == gen_cap) {
            const int nc = gen_cap * 2;
            int32_t* ng = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)nc);
            if (!ng) { st = VV_ERR_OUT_OF_MEMORY; break; }
            memcpy(ng, gen, sizeof(int32_t) * (size_t)n_gen);
            if (gen != gen_ids_small) vv_free(gen);
            gen = ng;
            gen_cap = nc;
        }
        gen[n_gen++] = tok;
        s->stats.tokens++;

        size_t nb = 0;
        st = s->be.token_bytes(s->be.self, tok, bytes, sizeof(bytes), &nb);
        if (st != VV_OK) break;
        const char* d = NULL;
        size_t dn = 0;
        st = vv_stream_text_push(s->text, bytes, nb, &d, &dn);
        if (st == VV_OK) st = emit_delta(s, idx, d, dn);
        if (st != VV_OK) break;

        /* The token is fed even when it is the last one the cap allows,
         * exactly as upstream does; one slot stays for the chunk end. */
        if (!kv_fits(s, 2)) { st = VV_ERR_OVERFLOW; break; }
        st = s->be.decode_step(s->be.self, tok, &tok);
        if (st != VV_OK) break;
    }
    if (st != VV_OK) {
        if (gen != gen_ids_small) vv_free(gen);
        return fail(s, st, idx);
    }

    const char* d = NULL;
    size_t dn = 0;
    st = vv_stream_text_flush(s->text, &d, &dn);
    if (st == VV_OK) st = emit_delta(s, idx, d, dn);
    if (st != VV_OK) {
        if (gen != gen_ids_small) vv_free(gen);
        return fail(s, st, idx);
    }
    dump_chunk(s, idx, gen, n_gen);
    if (gen != gen_ids_small) vv_free(gen);

    /* Upstream feeds <|text_chunk_end|> after every chunk, whatever stopped
     * it. Either now, or folded into the next prefill as its first row. */
    const int32_t tce = s->p.ids.text_chunk_end;
    if (s->p.fold_chunk_end || !s->be.append_tokens) {
        s->pending_lead = tce;
    } else {
        st = s->be.append_tokens(s->be.self, &tce, 1);
        if (st != VV_OK) return fail(s, st, idx);
    }

    if (!sb_put(&s->transcript, s->chunk_text.p ? s->chunk_text.p : "",
                s->chunk_text.len))
        return fail(s, VV_ERR_OUT_OF_MEMORY, idx);
    s->stats.chunks++;

    vv_stream_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = VV_STREAM_EVENT_CHUNK;
    ev.chunk_index = idx;
    ev.text = s->chunk_text.p ? s->chunk_text.p : "";
    ev.text_len = s->chunk_text.len;
    ev.n_tokens = n_gen;
    ev.stop = stop;
    ev.audio_start = (double)w->start / s->p.geom.sample_rate;
    ev.audio_end = (double)(w->start + vv_stream_chunk_samples(&s->p.geom)) /
                   s->p.geom.sample_rate;
    emit(s, &ev);
    return VV_OK;
}

static vv_status_t drain(vv_stream_t* s) {
    vv_stream_window_t w;
    while (vv_stream_chunker_next(s->chunker, s->window, &w)) {
        vv_status_t st = run_chunk(s, &w);
        if (st != VV_OK) return st;
    }
    return VV_OK;
}

vv_status_t vv_stream_open(struct vv_inference_ctx* ctx,
                           const vv_stream_params_t* params,
                           vv_stream_t** out) {
    (void)params;
    if (!ctx || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    /* TODO(phase 2, #20): a vv_stream_backend_t over ctx:
     *  - encode_text / token_bytes: ctx->tokenizer (vv_tokenizer_encode,
     *    vv_stream_token_bytes);
     *  - prefill_prompt: embed + vv_decoder_prefill into a reset cache;
     *  - prefill_chunk: the stateless-window encoder (#16) for both
     *    tokenizers + connectors straight into the chunk's hidden rows at
     *    feat_offset, token rows through the embedding, prefill at
     *    kv->current_len (#15 / #17 split-KV small-q path), then final norm,
     *    LM head and argmax of the last row; vv_kv_cache_publish_len after;
     *  - decode_step: the graphed decode step with graphs kept in the
     *    session (position-invariant captures survive across chunks);
     *  - kv_len / kv_capacity: kv->current_len / kv->max_seq_len (every
     *    shard);
     *  - the per-chunk dumps stream_cNNNN_feats / _logits for cmp-stream. */
    vv_log(VV_LOG_ERROR, "stream: Streaming-7B sessions need the phase-2 "
                         "backend (issue #20)");
    return VV_ERR_UNSUPPORTED;
}

vv_status_t vv_stream_open_backend(const vv_stream_backend_t* backend,
                                   const vv_stream_params_t* params,
                                   vv_stream_t** out) {
    if (!backend || !params || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!backend->encode_text || !backend->token_bytes ||
        !backend->prefill_prompt || !backend->prefill_chunk ||
        !backend->decode_step)
        return VV_ERR_INVALID_ARG;
    if (params->max_new_tokens <= 0) return VV_ERR_INVALID_ARG;
    if (params->ids.text_chunk_end < 0 || params->ids.speech_start < 0 ||
        params->ids.speech_end < 0)
        return VV_ERR_INVALID_ARG;
    vv_status_t st = vv_stream_geom_validate(&params->geom);
    if (st != VV_OK) return st;

    vv_stream_t* s = (vv_stream_t*)vv_alloc(sizeof(vv_stream_t));
    if (!s) return VV_ERR_OUT_OF_MEMORY;
    memset(s, 0, sizeof(*s));
    s->be = *backend;
    s->p = *params;
    s->pending_lead = -1;
    s->failed = VV_OK;

    if (params->context_info) {
        const size_t n = strlen(params->context_info);
        s->context_info = (char*)vv_alloc(n + 1);
        if (!s->context_info) { st = VV_ERR_OUT_OF_MEMORY; goto fail; }
        memcpy(s->context_info, params->context_info, n + 1);
    }
    s->p.context_info = s->context_info;

    st = vv_stream_chunker_create(&s->p.geom, &s->chunker);
    if (st != VV_OK) goto fail;
    st = vv_stream_text_create(&s->text);
    if (st != VV_OK) goto fail;
    s->window = (float*)vv_alloc(
        sizeof(float) * (size_t)vv_stream_window_samples(&s->p.geom));
    s->rows_cap = vv_stream_window_frames(&s->p.geom) + 3;
    s->rows = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)s->rows_cap);
    if (!s->window || !s->rows) { st = VV_ERR_OUT_OF_MEMORY; goto fail; }
    if (!sb_reserve(&s->transcript, 0) || !sb_reserve(&s->chunk_text, 0)) {
        st = VV_ERR_OUT_OF_MEMORY;
        goto fail;
    }
    s->transcript.p[0] = '\0';
    s->chunk_text.p[0] = '\0';

    {
        const size_t need = vv_stream_prompt_text(s->context_info, NULL, 0);
        char* prompt = (char*)vv_alloc(need + 1);
        if (!prompt) { st = VV_ERR_OUT_OF_MEMORY; goto fail; }
        vv_stream_prompt_text(s->context_info, prompt, need + 1);
        int32_t* ids = NULL;
        int n = 0;
        st = s->be.encode_text(s->be.self, prompt, &ids, &n);
        vv_free(prompt);
        if (st != VV_OK) goto fail;
        vv_debug_dump("stream_prompt_ids", ids, sizeof(int32_t) * (size_t)n);
        if (s->be.kv_capacity && n + 1 > kv_cap(s)) {
            vv_free(ids);
            st = VV_ERR_OVERFLOW;
            goto fail;
        }
        st = s->be.prefill_prompt(s->be.self, ids, n);
        vv_free(ids);
        if (st != VV_OK) goto fail;
        s->stats.prompt_tokens = n;
    }
    *out = s;
    return VV_OK;

fail:
    /* The caller keeps ownership of a backend we failed to open. */
    s->be.destroy = NULL;
    vv_stream_close(s);
    return st;
}

vv_status_t vv_stream_push(vv_stream_t* s, const float* pcm, size_t n) {
    if (!s) return VV_ERR_NULL_PTR;
    if (s->failed != VV_OK) return s->failed;
    if (s->done) return VV_ERR_INVALID_ARG;
    vv_status_t st = vv_stream_chunker_push(s->chunker, pcm, n);
    if (st != VV_OK) return fail(s, st, -1);
    return drain(s);
}

vv_status_t vv_stream_finish(vv_stream_t* s) {
    if (!s) return VV_ERR_NULL_PTR;
    if (s->failed != VV_OK) return s->failed;
    if (s->done) return VV_OK;
    vv_stream_chunker_finish(s->chunker);
    vv_status_t st = drain(s);
    if (st != VV_OK) return st;
    s->done = true;
    vv_stream_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = VV_STREAM_EVENT_DONE;
    ev.chunk_index = s->stats.chunks;
    ev.text = s->transcript.p;
    ev.text_len = s->transcript.len;
    ev.audio_end = (double)vv_stream_chunker_total(s->chunker) /
                   s->p.geom.sample_rate;
    emit(s, &ev);
    return VV_OK;
}

const char* vv_stream_transcript(const vv_stream_t* s) {
    return s && s->transcript.p ? s->transcript.p : "";
}

void vv_stream_get_stats(const vv_stream_t* s, vv_stream_stats_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!s) return;
    *out = s->stats;
    vv_stream_t* m = (vv_stream_t*)s;
    out->kv_len = kv_len(m);
    out->kv_capacity = s->be.kv_capacity ? kv_cap(m) : 0;
    out->samples = vv_stream_chunker_total(s->chunker);
}

void vv_stream_close(vv_stream_t* s) {
    if (!s) return;
    if (s->be.destroy) s->be.destroy(s->be.self);
    vv_stream_chunker_free(s->chunker);
    vv_stream_text_free(s->text);
    vv_free(s->window);
    vv_free(s->rows);
    vv_free(s->chunk_text.p);
    vv_free(s->transcript.p);
    vv_free(s->context_info);
    vv_free(s);
}
