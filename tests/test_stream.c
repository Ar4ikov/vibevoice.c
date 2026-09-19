/**
 * @file test_stream.c
 * @brief Streaming-7B session pieces, no weights needed.
 *
 *  - window geometry and the chunk scheduler against upstream's
 *    split_then_encode loop (any push sizes, zero-padded tail);
 *  - per-chunk prefill layout and the prompt text;
 *  - UTF-8-safe deltas against CPython's decode("utf-8", "replace") and the
 *    control strings upstream strips;
 *  - the session over a scripted backend: KV accounting, the folded or
 *    appended <|text_chunk_end|>, EOS, the per-chunk cap, overflow refusal.
 *
 * With VV_TEST_STREAM_MODEL pointing at the Streaming-7B directory it also
 * checks the real tokenizer: ids, prompt tokenization and the first jfk
 * chunk's text, all taken from `tools/compare_ref.py dump-stream`, and with
 * VV_TEST_STREAM_REF=<dump dir>[:<dir>...] it rebuilds every chunk text of
 * those dumps from their token ids. VV_TEST_STREAM_E2E=1 on top loads the
 * model (VV_TEST_STREAM_QUANT, default none; VV_TEST_STREAM_CPU=1 for the
 * CPU path) and runs every reference clip through a real session, chunk
 * texts compared with upstream's.
 *
 * Model-free as well: speaker-turn segments from chunk texts, the streaming
 * resampler against the whole-file one, and cancelling a session.
 */

#include "vibevoice/stream.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
#include "vibevoice/audio.h"
#include "vibevoice/quant.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
    g_fail++; } } while (0)

/* ─── Geometry and scheduler ────────────────────────────────────────────── */

static void test_geometry(void) {
    vv_stream_geom_t g;
    vv_stream_geom_default(&g);
    CHECK(vv_stream_geom_validate(&g) == VV_OK);
    CHECK(vv_stream_chunk_samples(&g) == 70400);
    CHECK(vv_stream_window_samples(&g) == 83200);
    CHECK(vv_stream_window_frames(&g) == 26);
    CHECK(!g.normalize_audio);
    /* Window counts of the parity set (dump-stream meta.json). */
    CHECK(vv_stream_n_windows(&g, 264000) == 4);     /* jfk */
    CHECK(vv_stream_n_windows(&g, 720000) == 11);    /* test30 */
    CHECK(vv_stream_n_windows(&g, 2880000) == 41);   /* test120 */
    CHECK(vv_stream_n_windows(&g, 7200000) == 103);  /* long30m, 5 min */
    CHECK(vv_stream_n_windows(&g, 0) == 0);
    CHECK(vv_stream_n_windows(&g, 1) == 1);
    CHECK(vv_stream_n_windows(&g, 70400) == 1);
    CHECK(vv_stream_n_windows(&g, 70401) == 2);
    vv_stream_geom_t bad = g;
    bad.chunk_frames = 0;
    CHECK(vv_stream_geom_validate(&bad) == VV_ERR_INVALID_ARG);
}

/* Upstream: start = 0; while start < total: window [start, min(start + W,
 * total)) padded to W; start += C. Returns the number of windows. */
static int64_t ref_windows(int64_t total, int64_t C, int64_t W,
                           int64_t* starts, int64_t* valid, int64_t cap) {
    int64_t n = 0;
    for (int64_t s = 0; s < total; s += C) {
        const int64_t e = s + W < total ? s + W : total;
        if (n < cap) { starts[n] = s; valid[n] = e - s; }
        n++;
    }
    return n;
}

static uint32_t rng_state = 12345u;
static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* Push `total` samples x[i] = i in pieces from `sizes` (cycled), check each
 * window as it pops, and that nothing pops early. */
static void run_chunker(int64_t total, const size_t* sizes, int n_sizes) {
    vv_stream_geom_t g;
    vv_stream_geom_default(&g);
    const int64_t C = vv_stream_chunk_samples(&g), W = vv_stream_window_samples(&g);
    vv_stream_chunker_t* ch = NULL;
    CHECK(vv_stream_chunker_create(&g, &ch) == VV_OK);
    if (!ch) return;

    int64_t starts[256], valid[256];
    const int64_t n_ref = ref_windows(total, C, W, starts, valid, 256);
    float* win = (float*)malloc(sizeof(float) * (size_t)W);
    float* piece = (float*)malloc(sizeof(float) * 200000);
    int64_t pushed = 0, got = 0;
    int si = 0;

    #define CHECK_WINDOW(info) do {                                        \
        CHECK((info).index == got);                                        \
        CHECK((info).start == starts[got]);                                \
        CHECK((info).n_valid == valid[got]);                               \
        int ok = 1;                                                        \
        for (int64_t k = 0; k < W; k++) {                                  \
            const float want = k < valid[got] ? (float)(starts[got] + k) : 0.f; \
            if (win[k] != want) { ok = 0; break; }                         \
        }                                                                  \
        CHECK(ok);                                                         \
        got++;                                                             \
    } while (0)

    while (pushed < total) {
        size_t n = sizes[si++ % n_sizes];
        if ((int64_t)n > total - pushed) n = (size_t)(total - pushed);
        for (size_t k = 0; k < n; k++) piece[k] = (float)(pushed + (int64_t)k);
        CHECK(vv_stream_chunker_push(ch, piece, n) == VV_OK);
        pushed += (int64_t)n;
        vv_stream_window_t info;
        while (vv_stream_chunker_next(ch, win, &info)) {
            /* A window pops before finish only when all of it is there. */
            CHECK(starts[got] + W <= pushed);
            CHECK(!info.is_last);
            CHECK_WINDOW(info);
        }
        CHECK(vv_stream_chunker_ready(ch) == 0);
    }
    vv_stream_chunker_finish(ch);
    CHECK(vv_stream_chunker_push(ch, piece, 1) == VV_ERR_INVALID_ARG);
    CHECK(vv_stream_chunker_ready(ch) == n_ref - got);
    vv_stream_window_t info;
    while (vv_stream_chunker_next(ch, win, &info)) {
        const int64_t this_idx = got;
        CHECK_WINDOW(info);
        CHECK(info.is_last == (this_idx == n_ref - 1));
    }
    CHECK(got == n_ref);
    CHECK(vv_stream_chunker_total(ch) == total);
    #undef CHECK_WINDOW
    free(win);
    free(piece);
    vv_stream_chunker_free(ch);
}

static void test_chunker(void) {
    const size_t whole[] = { 200000 };
    const size_t tiny[] = { 1, 7, 3199, 3200, 3201 };
    const size_t odd[] = { 70399, 12801, 1, 83200, 480 };
    size_t rnd[64];
    for (int i = 0; i < 64; i++) rnd[i] = 1 + rng() % 40000;

    const int64_t totals[] = { 0, 1, 1000, 70400, 70401, 83199, 83200, 83201,
                               153600, 264000, 720000 };
    for (size_t t = 0; t < sizeof(totals) / sizeof(totals[0]); t++) {
        run_chunker(totals[t], whole, 1);
        run_chunker(totals[t], tiny, 5);
        run_chunker(totals[t], odd, 5);
        run_chunker(totals[t], rnd, 64);
    }
}

/* ─── Layout and prompt ─────────────────────────────────────────────────── */

static void test_layout(void) {
    vv_stream_ids_t ids;
    vv_stream_ids_default(&ids);
    CHECK(ids.speech_start == 151646 && ids.speech_end == 151647);
    CHECK(ids.text_chunk_end == 151665 && ids.eos == 151643);

    int32_t rows[40];
    int off = -1;
    int n = vv_stream_chunk_layout(&ids, -1, 26, rows, 40, &off);
    CHECK(n == 28 && off == 1);
    CHECK(rows[0] == 151646 && rows[27] == 151647);
    for (int i = 1; i < 27; i++) CHECK(rows[i] == VV_STREAM_ROW_FEATURE);

    n = vv_stream_chunk_layout(&ids, ids.text_chunk_end, 26, rows, 40, &off);
    CHECK(n == 29 && off == 2);
    CHECK(rows[0] == 151665 && rows[1] == 151646 && rows[28] == 151647);

    CHECK(vv_stream_chunk_layout(&ids, 151665, 26, rows, 28, &off) == -1);
    CHECK(vv_stream_chunk_layout(&ids, -1, 26, rows, 28, NULL) == 28);
    CHECK(vv_stream_chunk_layout(&ids, -1, 0, rows, 28, NULL) == -1);
}

static void test_prompt(void) {
    static const char want[] =
        "You are a helpful assistant that transcribes audio input into text "
        "output. Please transcribe the following audios streamingly with these "
        "keys: speaker, content\n";
    char buf[512];
    size_t n = vv_stream_prompt_text(NULL, buf, sizeof(buf));
    CHECK(n == strlen(want) && strcmp(buf, want) == 0);
    n = vv_stream_prompt_text("", buf, sizeof(buf));
    CHECK(strcmp(buf, want) == 0);
    n = vv_stream_prompt_text("VibeVoice,Azure", buf, sizeof(buf));
    CHECK(strcmp(buf, "You are a helpful assistant that transcribes audio input "
                      "into text output. Please transcribe the following audios "
                      "streamingly with these keys: speaker, content and extra "
                      "info: VibeVoice,Azure\n") == 0);
    CHECK(n == strlen(buf));
    /* Size query and truncation behave like snprintf. */
    CHECK(vv_stream_prompt_text(NULL, NULL, 0) == strlen(want));
    char small[8];
    CHECK(vv_stream_prompt_text(NULL, small, sizeof(small)) == strlen(want));
    CHECK(strlen(small) == 7);
}

/* ─── Text deltas ───────────────────────────────────────────────────────── */

/* Push `in` as pieces of the given sizes, then flush; return the joined
 * output (static buffer). */
static const char* run_text(const char* in, size_t len, const size_t* sizes,
                            int n_sizes, int* n_nonempty_before_last) {
    static char acc[4096];
    size_t an = 0;
    vv_stream_text_t* t = NULL;
    if (vv_stream_text_create(&t) != VV_OK) return "";
    size_t off = 0;
    int si = 0, nonempty = 0;
    while (off < len) {
        size_t n = sizes ? sizes[si++ % n_sizes] : len;
        if (n > len - off) n = len - off;
        const char* d = NULL;
        size_t dn = 0;
        CHECK(vv_stream_text_push(t, in + off, n, &d, &dn) == VV_OK);
        CHECK(strlen(d) == dn);
        if (dn) nonempty++;
        memcpy(acc + an, d, dn);
        an += dn;
        off += n;
    }
    const char* d = NULL;
    size_t dn = 0;
    CHECK(vv_stream_text_flush(t, &d, &dn) == VV_OK);
    memcpy(acc + an, d, dn);
    an += dn;
    acc[an] = '\0';
    if (n_nonempty_before_last) *n_nonempty_before_last = nonempty;
    vv_stream_text_free(t);
    return acc;
}

static void expect_text(const char* in, size_t len, const char* want) {
    const size_t ones[] = { 1 };
    const size_t twos[] = { 2, 1, 3 };
    const char* got = run_text(in, len, NULL, 0, NULL);
    CHECK(strcmp(got, want) == 0);
    if (strcmp(got, want) != 0) fprintf(stderr, "  whole: got '%s' want '%s'\n", got, want);
    got = run_text(in, len, ones, 1, NULL);
    CHECK(strcmp(got, want) == 0);
    if (strcmp(got, want) != 0) fprintf(stderr, "  bytewise: got '%s' want '%s'\n", got, want);
    got = run_text(in, len, twos, 3, NULL);
    CHECK(strcmp(got, want) == 0);
}

#define FFFD "\xEF\xBF\xBD"
#define EXPECT(lit, want) expect_text(lit, sizeof(lit) - 1, want)

static void test_text(void) {
    /* Valid text passes through, any split. */
    EXPECT("Hello, world", "Hello, world");
    EXPECT(" \n Speaker 0:And so,", " \n Speaker 0:And so,");
    EXPECT("caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x98\x80",
           "caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x98\x80");

    /* Invalid bytes: CPython's bytes.decode("utf-8", "replace"). */
    EXPECT("\xE2\x28\xA1", FFFD "(" FFFD);
    EXPECT("\xF0\x90\x80" "A", FFFD "A");
    EXPECT("\xED\xA0\x80", FFFD FFFD FFFD);
    EXPECT("\xC0\xAF", FFFD FFFD);
    EXPECT("\xF4\x90\x80\x80", FFFD FFFD FFFD FFFD);
    EXPECT("\xE0\x80\xAF", FFFD FFFD FFFD);
    EXPECT("a\xFF\xFE" "b", "a" FFFD FFFD "b");
    /* An incomplete tail at the end of a chunk is one U+FFFD. */
    EXPECT("\xF0\x9F\x98", FFFD);
    EXPECT("ok\xE4\xBD", "ok" FFFD);

    /* Control strings upstream strips, whole or across pieces. */
    EXPECT("a<|text_chunk_end|>b", "ab");
    EXPECT("<|object_ref_start|><|object_ref_end|><|box_start|>", "");
    EXPECT("x<|speech_start|>y<|speech_end|>z<|speech_pad|>", "xyz");
    /* Near misses stay text; a partial string at the end is released. */
    EXPECT("a < b <| c <|box", "a < b <| c <|box");
    EXPECT("<|box_end|>", "<|box_end|>");
    EXPECT("<<|box_start|>", "<");

    /* Holding back: an incomplete character or control-string prefix
     * produces no delta until it resolves. */
    {
        vv_stream_text_t* t = NULL;
        CHECK(vv_stream_text_create(&t) == VV_OK);
        const char* d;
        size_t n;
        CHECK(vv_stream_text_push(t, "\xC3", 1, &d, &n) == VV_OK && n == 0);
        CHECK(vv_stream_text_push(t, "\xA9", 1, &d, &n) == VV_OK &&
              strcmp(d, "\xC3\xA9") == 0);
        CHECK(vv_stream_text_push(t, "a<|text", 7, &d, &n) == VV_OK &&
              strcmp(d, "a") == 0);
        CHECK(vv_stream_text_push(t, "_chunk_end|>", 12, &d, &n) == VV_OK && n == 0);
        CHECK(vv_stream_text_push(t, "b<", 2, &d, &n) == VV_OK && strcmp(d, "b") == 0);
        CHECK(vv_stream_text_push(t, "x", 1, &d, &n) == VV_OK && strcmp(d, "<x") == 0);
        CHECK(vv_stream_text_flush(t, &d, &n) == VV_OK && n == 0);
        /* Flush leaves the emitter clean for the next chunk. */
        CHECK(vv_stream_text_push(t, "\xE4", 1, &d, &n) == VV_OK && n == 0);
        CHECK(vv_stream_text_flush(t, &d, &n) == VV_OK && strcmp(d, FFFD) == 0);
        CHECK(vv_stream_text_push(t, "z", 1, &d, &n) == VV_OK && strcmp(d, "z") == 0);
        vv_stream_text_free(t);
    }

    /* Split invariance on random bytes: however the stream is cut, the
     * joined output is the same as one push + flush. */
    for (int it = 0; it < 300; it++) {
        char in[200];
        const size_t len = 1 + rng() % sizeof(in);
        for (size_t k = 0; k < len; k++) {
            const uint32_t r = rng() % 10;
            /* Mix ASCII, '<|' fragments and high bytes. */
            if (r < 4) in[k] = (char)('a' + rng() % 26);
            else if (r < 6) in[k] = "<|_>"[rng() % 4];
            else in[k] = (char)(0x80 + rng() % 0x80);
        }
        char whole[4096];
        strcpy(whole, run_text(in, len, NULL, 0, NULL));
        size_t sizes[16];
        for (int k = 0; k < 16; k++) sizes[k] = 1 + rng() % 5;
        const char* cut = run_text(in, len, sizes, 16, NULL);
        CHECK(strcmp(whole, cut) == 0);
    }
}

/* ─── Session over a scripted backend ───────────────────────────────────── */

#define TCE 151665
#define EOS 151643
#define SPECIAL_TOKEN 151644   /* <|im_start|>: special, no text bytes */

typedef struct {
    /* script[c] = tokens the model "generates" for chunk c: the prefill
     * returns script[c][0], each decode step the next one. */
    const int32_t* script[16];
    int            script_len[16];
    int            n_script;

    int64_t kv, cap;
    int     n_prompt;
    int     chunk;       /* chunk being decoded */
    int     pos;         /* index into script[chunk] */
    int     n_prefill_chunks, n_steps, n_appends;
    int32_t fed[512];    /* every token fed through decode_step/append */
    int     n_fed;
    int32_t leads[16];   /* rows[0] of each chunk prefill */
    int64_t starts[16], valid[16];
    int     feat_ok;
    bool    destroyed;
    bool    overflowed;  /* the backend saw a write past the capacity */
} mock_t;

static vv_status_t m_encode(void* self, const char* text, int32_t** ids, int* n) {
    (void)self;
    const int len = (int)strlen(text);
    *ids = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)len);
    for (int i = 0; i < len; i++) (*ids)[i] = (unsigned char)text[i];
    *n = len;
    return VV_OK;
}

/* Ids below 256 are that byte; SPECIAL_TOKEN has none. */
static vv_status_t m_bytes(void* self, int32_t id, char* buf, size_t cap,
                           size_t* n) {
    (void)self;
    (void)cap;
    if (id == SPECIAL_TOKEN) { *n = 0; return VV_OK; }
    if (id < 0 || id > 255) return VV_ERR_INVALID_ARG;
    buf[0] = (char)id;
    *n = 1;
    return VV_OK;
}

static void m_grow(mock_t* m, int64_t n) {
    m->kv += n;
    if (m->kv > m->cap) m->overflowed = true;
}

static vv_status_t m_prompt(void* self, const int32_t* ids, int n) {
    mock_t* m = (mock_t*)self;
    (void)ids;
    m->n_prompt = n;
    m_grow(m, n);
    return VV_OK;
}

static vv_status_t m_chunk(void* self, const vv_stream_chunk_t* c,
                           const float* window, int32_t* first) {
    mock_t* m = (mock_t*)self;
    const int k = m->n_prefill_chunks++;
    if (k < 16) {
        m->leads[k] = c->rows[0];
        m->starts[k] = c->start_sample;
        m->valid[k] = c->n_valid;
    }
    /* The layout the session hands over is the documented one. */
    int ok = c->n_frames == 26 && c->rows[c->feat_offset - 1] == 151646 &&
             c->rows[c->n_rows - 1] == 151647 &&
             c->feat_offset + c->n_frames == c->n_rows - 1;
    for (int i = 0; i < c->n_frames; i++)
        ok &= c->rows[c->feat_offset + i] == VV_STREAM_ROW_FEATURE;
    /* The window is x[i] = start + i, zero-padded. */
    ok &= window[0] == (c->n_valid ? (float)c->start_sample : 0.f);
    ok &= window[83199] == (c->n_valid == 83200 ? (float)(c->start_sample + 83199) : 0.f);
    if (!ok) m->feat_ok = 0;
    m_grow(m, c->n_rows);
    m->chunk = (int)c->index;
    m->pos = 0;
    *first = m->chunk < m->n_script ? m->script[m->chunk][0] : TCE;
    return VV_OK;
}

static vv_status_t m_step(void* self, int32_t token, int32_t* next) {
    mock_t* m = (mock_t*)self;
    m->n_steps++;
    if (m->n_fed < 512) m->fed[m->n_fed++] = token;
    m_grow(m, 1);
    m->pos++;
    const int c = m->chunk;
    *next = (c < m->n_script && m->pos < m->script_len[c]) ? m->script[c][m->pos]
                                                           : 'z';
    return VV_OK;
}

static vv_status_t m_append(void* self, const int32_t* ids, int n) {
    mock_t* m = (mock_t*)self;
    m->n_appends++;
    for (int i = 0; i < n && m->n_fed < 512; i++) m->fed[m->n_fed++] = ids[i];
    m_grow(m, n);
    return VV_OK;
}

static int64_t m_len(void* self) { return ((mock_t*)self)->kv; }
static int64_t m_cap(void* self) { return ((mock_t*)self)->cap; }
static void m_destroy(void* self) { ((mock_t*)self)->destroyed = true; }

static vv_stream_backend_t mock_backend(mock_t* m, bool with_append) {
    vv_stream_backend_t b;
    memset(&b, 0, sizeof(b));
    b.self = m;
    b.encode_text = m_encode;
    b.token_bytes = m_bytes;
    b.prefill_prompt = m_prompt;
    b.prefill_chunk = m_chunk;
    b.decode_step = m_step;
    b.append_tokens = with_append ? m_append : NULL;
    b.kv_len = m_len;
    b.kv_capacity = m_cap;
    b.destroy = m_destroy;
    return b;
}

typedef struct {
    char    deltas[16][256];   /* joined deltas per chunk */
    char    chunks[16][256];   /* CHUNK event text */
    int     n_tokens[16];
    int     stop[16];
    double  a0[16], a1[16];
    int     n_chunks, n_done, n_error;
    char    done[1024];
    vv_status_t error;
} events_t;

static void on_event(void* user, const vv_stream_event_t* ev) {
    events_t* e = (events_t*)user;
    const int64_t c = ev->chunk_index;
    CHECK(strlen(ev->text) == ev->text_len);
    switch (ev->type) {
    case VV_STREAM_EVENT_DELTA:
        CHECK(ev->text_len > 0);
        if (c >= 0 && c < 16) strcat(e->deltas[c], ev->text);
        break;
    case VV_STREAM_EVENT_CHUNK:
        CHECK(c == e->n_chunks);
        if (c >= 0 && c < 16) {
            strcpy(e->chunks[c], ev->text);
            e->n_tokens[c] = ev->n_tokens;
            e->stop[c] = (int)ev->stop;
            e->a0[c] = ev->audio_start;
            e->a1[c] = ev->audio_end;
        }
        e->n_chunks++;
        break;
    case VV_STREAM_EVENT_DONE:
        e->n_done++;
        strcpy(e->done, ev->text);
        break;
    case VV_STREAM_EVENT_ERROR:
        e->n_error++;
        e->error = ev->status;
        break;
    }
}

static float* ramp(int64_t n) {
    float* x = (float*)malloc(sizeof(float) * (size_t)(n ? n : 1));
    for (int64_t i = 0; i < n; i++) x[i] = (float)i;
    return x;
}

/* Push `total` samples of a ramp in pieces of `piece`. */
static vv_status_t push_ramp(vv_stream_t* s, const float* x, int64_t total,
                             int64_t piece) {
    for (int64_t off = 0; off < total; off += piece) {
        const int64_t n = total - off < piece ? total - off : piece;
        vv_status_t st = vv_stream_push(s, x + off, (size_t)n);
        if (st != VV_OK) return st;
    }
    return VV_OK;
}

static void test_session(bool fold) {
    /* 4 windows (jfk length). Chunk 1 splits "é" across tokens, chunk 2
     * carries a special token and stops on EOS, chunk 3 is empty. */
    static const int32_t c0[] = { ' ', 'H', 'i', ',', TCE };
    static const int32_t c1[] = { ' ', 'c', 'a', 'f', 0xC3, 0xA9, TCE };
    static const int32_t c2[] = { SPECIAL_TOKEN, ' ', 'o', 'k', EOS };
    static const int32_t c3[] = { TCE };
    mock_t m;
    memset(&m, 0, sizeof(m));
    m.script[0] = c0; m.script_len[0] = 5;
    m.script[1] = c1; m.script_len[1] = 7;
    m.script[2] = c2; m.script_len[2] = 5;
    m.script[3] = c3; m.script_len[3] = 1;
    m.n_script = 4;
    m.cap = 100000;
    m.feat_ok = 1;

    events_t ev;
    memset(&ev, 0, sizeof(ev));
    vv_stream_params_t p;
    vv_stream_params_default(&p);
    p.fold_chunk_end = fold;
    p.on_event = on_event;
    p.user = &ev;

    vv_stream_backend_t b = mock_backend(&m, true);
    vv_stream_t* s = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_OK);
    if (!s) return;
    char prompt[256];
    CHECK(m.n_prompt == (int)vv_stream_prompt_text(NULL, prompt, sizeof(prompt)));

    const int64_t total = 264000;
    float* x = ramp(total);
    CHECK(push_ramp(s, x, total, 4096) == VV_OK);
    /* Before finish only windows fully inside the audio ran: 0, 1, 2
     * (window 2 ends at 223999 < 264000); window 3 needs the tail. */
    CHECK(ev.n_chunks == 3);
    CHECK(vv_stream_finish(s) == VV_OK);
    CHECK(vv_stream_finish(s) == VV_OK);   /* idempotent */
    CHECK(ev.n_chunks == 4 && ev.n_done == 1 && ev.n_error == 0);
    CHECK(m.feat_ok);

    CHECK(strcmp(ev.chunks[0], " Hi,") == 0);
    CHECK(strcmp(ev.chunks[1], " caf\xC3\xA9") == 0);
    CHECK(strcmp(ev.chunks[2], " ok") == 0);
    CHECK(strcmp(ev.chunks[3], "") == 0);
    for (int c = 0; c < 4; c++) CHECK(strcmp(ev.deltas[c], ev.chunks[c]) == 0);
    CHECK(strcmp(ev.done, " Hi, caf\xC3\xA9 ok") == 0);
    CHECK(strcmp(vv_stream_transcript(s), ev.done) == 0);
    CHECK(ev.n_tokens[0] == 4 && ev.n_tokens[1] == 6 && ev.n_tokens[2] == 4 &&
          ev.n_tokens[3] == 0);
    CHECK(ev.stop[0] == VV_STREAM_STOP_CHUNK_END);
    CHECK(ev.stop[2] == VV_STREAM_STOP_EOS);
    CHECK(ev.a0[1] > 2.93 && ev.a0[1] < 2.94 && ev.a1[1] > 5.86 && ev.a1[1] < 5.87);

    /* Window placement. */
    CHECK(m.n_prefill_chunks == 4);
    for (int c = 0; c < 4; c++) CHECK(m.starts[c] == 70400 * c);
    CHECK(m.valid[0] == 83200 && m.valid[3] == 264000 - 211200);

    /* Every generated token is fed; stop tokens never are. */
    const int n_gen = 4 + 6 + 4 + 0;
    CHECK(m.n_steps == n_gen);

    /* <|text_chunk_end|> follows every chunk: folded into the next prefill
     * as its first row, or appended right away. */
    if (fold) {
        CHECK(m.n_appends == 0);
        CHECK(m.leads[0] == 151646);
        for (int c = 1; c < 4; c++) CHECK(m.leads[c] == TCE);
    } else {
        CHECK(m.n_appends == 4);
        for (int c = 0; c < 4; c++) CHECK(m.leads[c] == 151646);
    }

    /* KV: prompt + 4 × 28 rows + generated + chunk ends (the last one is
     * still pending when folding). */
    const int64_t want_kv = m.n_prompt + 4 * 28 + n_gen + (fold ? 3 : 4);
    CHECK(m.kv == want_kv);
    vv_stream_stats_t st;
    vv_stream_get_stats(s, &st);
    CHECK(st.chunks == 4 && st.tokens == n_gen && st.kv_len == want_kv);
    CHECK(st.samples == total && st.prompt_tokens == m.n_prompt);

    CHECK(vv_stream_push(s, x, 10) == VV_ERR_INVALID_ARG);   /* after finish */
    vv_stream_close(s);
    CHECK(m.destroyed);
    free(x);
}

static void test_session_cap_and_overflow(void) {
    /* Per-chunk cap: a chunk that never stops yields exactly max_new_tokens
     * tokens, all of them fed, then the chunk end. */
    static const int32_t run[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' };
    mock_t m;
    memset(&m, 0, sizeof(m));
    for (int c = 0; c < 4; c++) { m.script[c] = run; m.script_len[c] = 8; }
    m.n_script = 4;
    m.cap = 100000;
    m.feat_ok = 1;
    events_t ev;
    memset(&ev, 0, sizeof(ev));
    vv_stream_params_t p;
    vv_stream_params_default(&p);
    p.max_new_tokens = 4;
    p.on_event = on_event;
    p.user = &ev;
    vv_stream_backend_t b = mock_backend(&m, false);   /* forces folding */
    vv_stream_t* s = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_OK);
    float* x = ramp(83200);
    CHECK(vv_stream_push(s, x, 83200) == VV_OK);
    CHECK(ev.n_chunks == 1 && strcmp(ev.chunks[0], "abcd") == 0);
    CHECK(ev.stop[0] == VV_STREAM_STOP_MAX_TOKENS && m.n_steps == 4);
    vv_stream_close(s);

    /* Overflow: the cache holds the prompt, one chunk and a few tokens. The
     * session refuses cleanly instead of writing past it. */
    memset(&m, 0, sizeof(m));
    for (int c = 0; c < 4; c++) { m.script[c] = run; m.script_len[c] = 8; }
    m.n_script = 4;
    m.feat_ok = 1;
    char prompt[256];
    const int np = (int)vv_stream_prompt_text(NULL, prompt, sizeof(prompt));
    m.cap = np + 28 + 3;
    memset(&ev, 0, sizeof(ev));
    p.max_new_tokens = 256;
    b = mock_backend(&m, true);
    s = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_OK);
    free(x);
    x = ramp(264000);
    CHECK(push_ramp(s, x, 264000, 264000) == VV_ERR_OVERFLOW);
    CHECK(ev.n_error == 1 && ev.error == VV_ERR_OVERFLOW);
    CHECK(!m.overflowed && m.kv <= m.cap);
    CHECK(vv_stream_push(s, x, 1) == VV_ERR_OVERFLOW);
    CHECK(vv_stream_finish(s) == VV_ERR_OVERFLOW);
    CHECK(ev.n_error == 1 && ev.n_done == 0);
    vv_stream_close(s);

    /* A prompt that does not fit is refused at open, and the caller keeps
     * the backend. */
    memset(&m, 0, sizeof(m));
    m.cap = 10;
    b = mock_backend(&m, true);
    s = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_ERR_OVERFLOW && !s);
    CHECK(!m.destroyed && m.kv == 0);

    /* Missing ops are rejected, and so is a context with no model. */
    b.prefill_chunk = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_ERR_INVALID_ARG);
    vv_inference_ctx_t empty;
    memset(&empty, 0, sizeof(empty));
    CHECK(vv_stream_open(&empty, &p, &s) == VV_ERR_MODEL_FORMAT && !s);
    free(x);
}

/* ─── Real tokenizer (optional) ─────────────────────────────────────────── */

static void test_tokenizer(void) {
    const char* dir = getenv("VV_TEST_STREAM_MODEL");
    if (!dir || !dir[0]) {
        puts("stream: tokenizer checks SKIP (set VV_TEST_STREAM_MODEL)");
        return;
    }
    vv_tokenizer_t* tok = NULL;
    CHECK(vv_tokenizer_load(dir, &tok) == VV_OK);
    if (!tok) return;
    vv_stream_ids_t ids, def;
    vv_stream_ids_default(&def);
    CHECK(vv_stream_ids_from_tokenizer(tok, &ids) == VV_OK);
    CHECK(memcmp(&ids, &def, sizeof(ids)) == 0);

    /* prompt_ids from dump-stream (jfk, no hotwords). */
    static const int32_t want[] = {
        2610, 525, 264, 10950, 17847, 429, 1356, 55136, 7699, 1946, 1119,
        1467, 2550, 13, 5209, 1356, 3114, 279, 2701, 6136, 3530, 4269, 11307,
        448, 1493, 6894, 25, 18601, 11, 2213, 198 };
    char prompt[256];
    vv_stream_prompt_text(NULL, prompt, sizeof(prompt));
    int32_t* got = NULL;
    int n = 0;
    CHECK(vv_tokenizer_encode(tok, prompt, &got, &n) == VV_OK);
    CHECK(n == (int)(sizeof(want) / sizeof(want[0])));
    CHECK(got && memcmp(got, want, sizeof(want)) == 0);
    vv_free(got);

    /* jfk chunk 0 through token_bytes + the emitter equals upstream's
     * decode(skip_special_tokens=True) + strip. */
    static const int32_t c0[] = { 715, 29073, 220, 15, 25, 3036, 773, 11, 847,
                                  12357, 8877, 11, 220 };
    vv_stream_text_t* t = NULL;
    CHECK(vv_stream_text_create(&t) == VV_OK);
    char acc[512] = "";
    for (size_t i = 0; i < sizeof(c0) / sizeof(c0[0]); i++) {
        char b[64];
        size_t nb = 0;
        CHECK(vv_stream_token_bytes(tok, c0[i], b, sizeof(b), &nb) == VV_OK);
        const char* d;
        size_t dn;
        CHECK(vv_stream_text_push(t, b, nb, &d, &dn) == VV_OK);
        strcat(acc, d);
    }
    const char* d;
    size_t dn;
    CHECK(vv_stream_text_flush(t, &d, &dn) == VV_OK);
    strcat(acc, d);
    CHECK(strcmp(acc, " \n Speaker 0:And so, my fellow Americans, ") == 0);
    vv_stream_text_free(t);

    /* Special tokens carry no text. */
    char b[64];
    size_t nb = 99;
    CHECK(vv_stream_token_bytes(tok, 151665, b, sizeof(b), &nb) == VV_OK && nb == 0);
    CHECK(vv_stream_token_bytes(tok, 151643, b, sizeof(b), &nb) == VV_OK && nb == 0);
    puts("stream: tokenizer ids, prompt ids and jfk chunk 0 text match the reference");

    /* VV_TEST_STREAM_REF=<dump-stream dir>[:<dir>...]: every chunk's ids
     * through the emitter must give upstream's chunk text. */
    const char* refs = getenv("VV_TEST_STREAM_REF");
    char list[2048];
    snprintf(list, sizeof(list), "%s", refs ? refs : "");
    for (char* ref = strtok(list, ":"); ref; ref = strtok(NULL, ":")) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/meta.json", ref);
        FILE* f = fopen(path, "rb");
        CHECK(f != NULL);
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        const long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* js = (char*)malloc((size_t)len + 1);
        CHECK(fread(js, 1, (size_t)len, f) == (size_t)len);
        js[len] = '\0';
        fclose(f);
        cJSON* meta = cJSON_Parse(js);
        free(js);
        CHECK(meta != NULL);
        const cJSON* chunks = cJSON_GetObjectItem(meta, "chunks");
        int n_chunks = 0, n_same = 0;
        const cJSON* ch;
        cJSON_ArrayForEach(ch, chunks) {
            vv_stream_text_t* tt = NULL;
            CHECK(vv_stream_text_create(&tt) == VV_OK);
            size_t cap = 4096, an = 0;
            char* out = (char*)malloc(cap);
            out[0] = '\0';
            const cJSON* id;
            cJSON_ArrayForEach(id, cJSON_GetObjectItem(ch, "ids")) {
                char tb[256];
                size_t tn = 0;
                CHECK(vv_stream_token_bytes(tok, (int32_t)id->valueint, tb,
                                            sizeof(tb), &tn) == VV_OK);
                const char* dd;
                size_t ddn;
                CHECK(vv_stream_text_push(tt, tb, tn, &dd, &ddn) == VV_OK);
                if (an + ddn + 1 > cap) { cap = 2 * (an + ddn + 1); out = (char*)realloc(out, cap); }
                memcpy(out + an, dd, ddn + 1);
                an += ddn;
            }
            const char* dd;
            size_t ddn;
            CHECK(vv_stream_text_flush(tt, &dd, &ddn) == VV_OK);
            if (an + ddn + 1 > cap) { cap = 2 * (an + ddn + 1); out = (char*)realloc(out, cap); }
            memcpy(out + an, dd, ddn + 1);
            const char* want_text = cJSON_GetObjectItem(ch, "text")->valuestring;
            const int same = strcmp(out, want_text) == 0;
            if (!same)
                fprintf(stderr, "  %s chunk %d: got '%s' want '%s'\n", ref,
                        n_chunks, out, want_text);
            n_same += same;
            n_chunks++;
            free(out);
            vv_stream_text_free(tt);
        }
        CHECK(n_chunks > 0 && n_same == n_chunks);
        printf("stream: %s: %d/%d chunk texts identical to the reference\n",
               ref, n_same, n_chunks);
        cJSON_Delete(meta);
    }
    vv_tokenizer_free(tok);
}

/* ─── Segments from chunk texts ─────────────────────────────────────────── */

static void test_segments(void) {
    const double C = 22.0 * 3200.0 / 24000.0;   /* 2.9333 s */
    /* jfk's four chunks: one turn over all of them. */
    const char* jfk[] = { " \n Speaker 0:And so, my fellow Americans, ",
                          "ask not what your ",
                          "country can do for you, ask what ",
                          "you can do for your country." };
    vv_transcription_t* tr = NULL;
    CHECK(vv_stream_build_transcription(jfk, 4, C, 11.0, &tr) == VV_OK);
    CHECK(tr && tr->num_segments == 1);
    if (tr && tr->num_segments == 1) {
        CHECK(strcmp(tr->segments[0].speaker, "Speaker 0") == 0);
        CHECK(strcmp(tr->segments[0].text, "And so, my fellow Americans, ask "
                     "not what your country can do for you, ask what you can "
                     "do for your country.") == 0);
        CHECK(tr->segments[0].start_time == 0.0f);
        CHECK(tr->segments[0].end_time == 11.0f);   /* clamped */
        CHECK(strcmp(tr->full_text, tr->segments[0].text) == 0);
    }
    vv_transcription_free(tr);

    /* A turn marker split across chunks, text before any marker, empty
     * chunks, and a turn that ends before the last chunk. */
    const char* two[] = { "hello there \n Spea", "ker 1:how are", "",
                          " you\n Speaker 0: fine", "" };
    tr = NULL;
    CHECK(vv_stream_build_transcription(two, 5, C, 0.0, &tr) == VV_OK);
    CHECK(tr && tr->num_segments == 3);
    if (tr && tr->num_segments == 3) {
        CHECK(strcmp(tr->segments[0].speaker, "Speaker 0") == 0);
        CHECK(strcmp(tr->segments[0].text, "hello there") == 0);
        CHECK(tr->segments[0].start_time == 0.0f &&
              tr->segments[0].end_time == (float)C);
        CHECK(strcmp(tr->segments[1].speaker, "Speaker 1") == 0);
        CHECK(strcmp(tr->segments[1].text, "how are you") == 0);
        CHECK(tr->segments[1].start_time == 0.0f);          /* marker in 0 */
        CHECK(tr->segments[1].end_time == (float)(4 * C));  /* "you" in 3 */
        CHECK(strcmp(tr->segments[2].text, "fine") == 0);
        CHECK(tr->segments[2].start_time == (float)(3 * C));
        CHECK(strcmp(tr->full_text, "hello there how are you fine") == 0);
    }
    vv_transcription_free(tr);

    /* Nothing said: no segments, empty text. */
    const char* none[] = { "", "  " };
    tr = NULL;
    CHECK(vv_stream_build_transcription(none, 2, C, 5.0, &tr) == VV_OK);
    CHECK(tr && tr->num_segments == 0 && tr->full_text &&
          tr->full_text[0] == '\0');
    vv_transcription_free(tr);
    CHECK(vv_stream_build_transcription(NULL, 0, C, 0.0, &tr) == VV_OK);
    vv_transcription_free(tr);
}

/* ─── Streaming resampler == whole-file resampler ───────────────────────── */

static void check_resampler(int in_sr, int n, const size_t* sizes,
                            int n_sizes) {
    float* x = (float*)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * sinf(0.013f * (float)i) + 0.25f * sinf(0.31f * (float)i);
    float* want = NULL;
    int n_want = 0;
    CHECK(vv_audio_resample(x, in_sr, n, &want, 24000, &n_want) == VV_OK);

    vv_resampler_t* r = NULL;
    CHECK(vv_resampler_create(in_sr, 24000, &r) == VV_OK);
    float* got = (float*)malloc(sizeof(float) * (size_t)(n_want + 64));
    int n_got = 0, off = 0, k = 0;
    while (off < n) {
        size_t m = sizes[k++ % n_sizes];
        if (m > (size_t)(n - off)) m = (size_t)(n - off);
        const float* o = NULL;
        size_t no = 0;
        CHECK(vv_resampler_push(r, x + off, m, &o, &no) == VV_OK);
        if (n_got + (int)no <= n_want + 64) memcpy(got + n_got, o, no * sizeof(float));
        n_got += (int)no;
        off += (int)m;
    }
    const float* o = NULL;
    size_t no = 0;
    CHECK(vv_resampler_finish(r, &o, &no) == VV_OK);
    if (n_got + (int)no <= n_want + 64) memcpy(got + n_got, o, no * sizeof(float));
    n_got += (int)no;
    CHECK(n_got == n_want);
    CHECK(n_got == n_want && memcmp(got, want, sizeof(float) * (size_t)n_want) == 0);
    vv_resampler_free(r);
    vv_free(want);
    free(got);
    free(x);
}

static void test_resampler(void) {
    const size_t one[] = { 1 };
    const size_t mixed[] = { 160, 7, 3200, 1, 511, 0, 44100 };
    const size_t big[] = { 1u << 20 };
    check_resampler(16000, 48000, one, 1);
    check_resampler(16000, 160001, mixed, 7);
    check_resampler(44100, 100000, mixed, 7);
    check_resampler(48000, 96001, big, 1);
    check_resampler(8000, 17, one, 1);
    check_resampler(24000, 5000, mixed, 7);   /* pass-through */
}

/* ─── Cancel ────────────────────────────────────────────────────────────── */

typedef struct { vv_stream_t* s; int n_chunks, n_errors; } cancel_ev_t;

static void cancel_on_chunk(void* user, const vv_stream_event_t* ev) {
    cancel_ev_t* c = (cancel_ev_t*)user;
    if (ev->type == VV_STREAM_EVENT_CHUNK && ++c->n_chunks == 2)
        vv_stream_cancel(c->s);     /* as a server does when a write fails */
    if (ev->type == VV_STREAM_EVENT_ERROR) c->n_errors++;
}

static void test_cancel(void) {
    mock_t m;
    memset(&m, 0, sizeof(m));
    m.cap = 100000;
    m.feat_ok = 1;
    static const int32_t say[] = { 'a', 'b', TCE };
    for (int i = 0; i < 16; i++) { m.script[i] = say; m.script_len[i] = 3; }
    m.n_script = 16;
    vv_stream_backend_t b = mock_backend(&m, false);
    vv_stream_params_t p;
    vv_stream_params_default(&p);
    cancel_ev_t ce;
    memset(&ce, 0, sizeof(ce));
    p.on_event = cancel_on_chunk;
    p.user = &ce;
    vv_stream_t* s = NULL;
    CHECK(vv_stream_open_backend(&b, &p, &s) == VV_OK);
    ce.s = s;
    float* x = (float*)calloc(70400 * 6, sizeof(float));
    CHECK(vv_stream_push(s, x, 70400 * 6) == VV_ERR_CANCELLED);
    CHECK(ce.n_chunks == 2 && ce.n_errors == 0);   /* no ERROR for a cancel */
    CHECK(m.n_prefill_chunks == 2);
    CHECK(vv_stream_push(s, x, 1) == VV_ERR_CANCELLED);
    CHECK(vv_stream_finish(s) == VV_ERR_CANCELLED);
    vv_stream_close(s);
    CHECK(m.destroyed);
    free(x);
}

/* ─── The real model (optional, slow) ───────────────────────────────────── */

typedef struct { char** texts; int n; } e2e_ev_t;

static void e2e_event(void* user, const vv_stream_event_t* ev) {
    e2e_ev_t* e = (e2e_ev_t*)user;
    if (ev->type != VV_STREAM_EVENT_CHUNK || e->n >= 4096) return;
    e->texts[e->n] = (char*)malloc(ev->text_len + 1);
    memcpy(e->texts[e->n], ev->text, ev->text_len + 1);
    e->n++;
}

static void test_model_e2e(void) {
    const char* dir = getenv("VV_TEST_STREAM_MODEL");
    const char* refs = getenv("VV_TEST_STREAM_REF");
    const char* on = getenv("VV_TEST_STREAM_E2E");
    if (!dir || !dir[0] || !refs || !refs[0] || !on || on[0] != '1') {
        puts("stream: model run SKIP (set VV_TEST_STREAM_E2E=1, "
             "VV_TEST_STREAM_MODEL and VV_TEST_STREAM_REF)");
        return;
    }
    vv_init_params_t ip = vv_init_params_default();
    const char* q = getenv("VV_TEST_STREAM_QUANT");
    ip.weight_quant = (int)vv_load_quant_parse(q && q[0] ? q : "none");
    const char* cpu = getenv("VV_TEST_STREAM_CPU");
    ip.cpu_only = cpu && cpu[0] == '1';
    vv_inference_ctx_t* ctx = NULL;
    CHECK(vv_inference_init(dir, 0, &ip, &ctx) == VV_OK);
    if (!ctx) return;

    char list[2048];
    snprintf(list, sizeof(list), "%s", refs);
    for (char* ref = strtok(list, ":"); ref; ref = strtok(NULL, ":")) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/audio24k.f32", ref);
        FILE* f = fopen(path, "rb");
        CHECK(f != NULL);
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        const long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        float* pcm = (float*)malloc((size_t)bytes);
        CHECK(fread(pcm, 1, (size_t)bytes, f) == (size_t)bytes);
        fclose(f);

        snprintf(path, sizeof(path), "%s/meta.json", ref);
        f = fopen(path, "rb");
        CHECK(f != NULL);
        if (!f) { free(pcm); continue; }
        fseek(f, 0, SEEK_END);
        const long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* js = (char*)malloc((size_t)len + 1);
        CHECK(fread(js, 1, (size_t)len, f) == (size_t)len);
        js[len] = '\0';
        fclose(f);
        cJSON* meta = cJSON_Parse(js);
        free(js);

        e2e_ev_t e;
        e.texts = (char**)calloc(4096, sizeof(char*));
        e.n = 0;
        vv_transcription_t* tr = NULL;
        const double t0 = vv_time_ms();
        CHECK(vv_stream_transcribe(ctx, pcm, (int)(bytes / 4), NULL,
                                   e2e_event, &e, &tr) == VV_OK);
        const double ms = vv_time_ms() - t0;
        int n_ref = 0, n_same = 0;
        const cJSON* ch;
        cJSON_ArrayForEach(ch, cJSON_GetObjectItem(meta, "chunks")) {
            const char* want = cJSON_GetObjectItem(ch, "text")->valuestring;
            const int same = n_ref < e.n && strcmp(e.texts[n_ref], want) == 0;
            if (!same && n_ref < e.n)
                fprintf(stderr, "  %s chunk %d: got '%s' want '%s'\n", ref,
                        n_ref, e.texts[n_ref], want);
            n_same += same;
            n_ref++;
        }
        CHECK(e.n == n_ref && n_same == n_ref);
        printf("stream: %s: %d/%d chunk texts identical through a live "
               "session (%.0f ms, RTF %.3f)\n", ref, n_same, n_ref, ms,
               ms / 1000.0 / ((double)bytes / 4 / 24000.0));
        for (int i = 0; i < e.n; i++) free(e.texts[i]);
        free(e.texts);
        vv_transcription_free(tr);
        cJSON_Delete(meta);
        free(pcm);
    }
    vv_inference_free(ctx);
}

int main(void) {
    test_geometry();
    test_chunker();
    test_layout();
    test_prompt();
    test_text();
    test_session(true);
    test_session(false);
    test_session_cap_and_overflow();
    test_tokenizer();
    test_segments();
    test_resampler();
    test_cancel();
    test_model_e2e();
    if (g_fail) {
        fprintf(stderr, "stream: %d check(s) failed\n", g_fail);
        return 1;
    }
    puts("stream: geometry, scheduler, layout, prompt, deltas and session passed");
    return 0;
}
