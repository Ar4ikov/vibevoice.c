/**
 * @file dflash_data.c
 * @brief Training data for DFlash 2 drafters: the target's own transcripts,
 *        then its hidden states along them.
 *
 *   vv_dflash_data gen   --model DIR --list clips.tsv --out gen.jsonl
 *   vv_dflash_data trace --model DIR --gen gen.jsonl --dir /dev/shm/vvd
 *                        --layers 1,7,13,19,25 [--ring 48] [--epochs 3]
 *
 * `gen` transcribes every clip of a list (one per line: id, path and
 * optional hotwords, tab-separated) greedily and writes one JSON line per
 * clip with the generated ids of every chunk. It skips ids the output
 * already has, so an interrupted run picks up where it stopped.
 *
 * `trace` replays those transcriptions (vv_spec_trace: prefills of the known
 * tokens, so at prefill speed) and writes one file per clip holding every
 * position's token, its role and the tapped layers' hidden states. With
 * `--ring N` it keeps at most N files in the directory and waits for the
 * reader (tools/dflash/train.py) to delete some, which makes it a feature
 * server for training: no disk, and the features are the runtime's own.
 *
 * File layout (little endian):
 *   "VVDT", u32 version = 1, u32 n_pos, u32 n_taps, u32 hidden,
 *   u32 id_len, i32 layers[n_taps], char id[id_len] (padded to 4),
 *   i32 ids[n_pos], u8 kind[n_pos] (padded to 4),
 *   f16 feats[n_pos][n_taps][hidden]
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
#include "vibevoice/audio.h"
#include "vibevoice/quant.h"
#include "vibevoice/spec.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"
#include "cJSON.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
static void sleep_ms(int ms) { Sleep((DWORD)ms); }
static int make_dir(const char* p) { return _mkdir(p); }
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
static void sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }
static int make_dir(const char* p) { return mkdir(p, 0755); }
#endif

/* mkdir -p: every missing component of `dir`, existing ones are fine. */
static int make_dirs(const char* dir) {
    char buf[1024];
    const size_t n = strlen(dir);
    if (n == 0 || n >= sizeof(buf)) return -1;
    memcpy(buf, dir, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] != '/' && buf[i] != '\\' && buf[i] != '\0') continue;
        const char c = buf[i];
        buf[i] = '\0';
        if (make_dir(buf) != 0 && errno != EEXIST) return -1;
        buf[i] = c;
    }
    return 0;
}

typedef struct {
    const char* cmd;
    const char* model;
    const char* list;
    const char* out;
    const char* gen;
    const char* dir;
    const char* layers;
    const char* quant;
    const char* kv;
    const char* draft;   /* gen: decode with this DFlash 2 drafter */
    int gpu;
    int max_seq_len;
    int max_tokens;
    int ring;
    int epochs;
    unsigned seed;
    int max_pos;
    int shard;       /* process clips i with i % n_shards == shard */
    int n_shards;
} args_t;

static char* dup_str(const char* x) {
    const size_t n = strlen(x);
    char* d = (char*)vv_alloc(n + 1);
    if (d) memcpy(d, x, n + 1);
    return d;
}

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  vv_dflash_data gen   --model DIR --list clips.tsv --out gen.jsonl\n"
        "                       [--gpu N] [--quant Q] [--max-seq-len N]\n"
        "                       [--max-tokens N] [--shard i/n] [--draft DIR]\n"
        "  vv_dflash_data trace --model DIR --gen gen.jsonl --dir OUT\n"
        "                       --layers 1,7,13,19,25 [--ring N] [--epochs E]\n"
        "                       [--seed S] [--max-pos P] [--gpu N] [--quant Q]\n");
}

static int parse_args(int argc, char** argv, args_t* a) {
    memset(a, 0, sizeof(*a));
    a->max_seq_len = 32768;
    a->max_tokens = 16384;
    a->epochs = 1;
    a->seed = 1;
    a->n_shards = 1;
    if (argc < 2) return -1;
    a->cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        const char* k = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
#define S(name, field) if (!strcmp(k, name) && v) { a->field = v; i++; continue; }
#define I(name, field) if (!strcmp(k, name) && v) { a->field = atoi(v); i++; continue; }
        S("--model", model) S("--list", list) S("--out", out) S("--gen", gen)
        S("--dir", dir) S("--layers", layers) S("--quant", quant)
        S("--kv-cache", kv) S("--draft", draft)
        I("--gpu", gpu) I("--max-seq-len", max_seq_len)
        I("--max-tokens", max_tokens) I("--ring", ring) I("--epochs", epochs)
        I("--max-pos", max_pos)
        if (!strcmp(k, "--seed") && v) { a->seed = (unsigned)strtoul(v, NULL, 10); i++; continue; }
        if (!strcmp(k, "--shard") && v) {
            if (sscanf(v, "%d/%d", &a->shard, &a->n_shards) != 2 ||
                a->n_shards < 1 || a->shard < 0 || a->shard >= a->n_shards)
                return -1;
            i++;
            continue;
        }
#undef S
#undef I
        VV_LOG_E("unknown or incomplete option: %s", k);
        return -1;
    }
    return a->model ? 0 : -1;
}

static vv_status_t open_model(const args_t* a, vv_inference_ctx_t** ctx) {
    vv_init_params_t p = vv_init_params_default();
    p.max_seq_len = a->max_seq_len;
    /* A drafted transcript is the plain one (exact check) with the
     * attention `--draft` picks -- flashinfer for `auto` -- only sooner. */
    p.draft_dir = a->draft;
    if (a->quant) {
        const vv_load_quant_t q = vv_load_quant_parse(a->quant);
        if (q == VV_LOAD_QUANT_COUNT) return VV_ERR_INVALID_ARG;
        p.weight_quant = (int)q;
    }
    if (a->kv) {
        const vv_kv_format_t f = vv_kv_format_parse(a->kv);
        if (f == VV_KV_FORMAT_COUNT) return VV_ERR_INVALID_ARG;
        p.kv_format = (int)f;
    }
    return vv_inference_init(a->model, a->gpu, &p, ctx);
}

/* Decode and prepare a clip the way vv_cli does for this model. */
static vv_status_t load_clip(vv_inference_ctx_t* ctx, const char* path,
                             float** pcm, int* n) {
    float* raw = NULL;
    int raw_n = 0, raw_sr = 0;
    vv_status_t s = vv_audio_load_any(path, &raw, &raw_n, &raw_sr);
    if (s != VV_OK) return s;
    if (ctx->family_ok && ctx->family.vibeasr_audio)
        s = vv_audio_prepare_vibeasr(raw, raw_n, raw_sr,
                                     ctx->family.normalize_audio, pcm, n);
    else
        s = vv_audio_prepare_ex(raw, raw_n, raw_sr,
                                ctx->family_ok ? ctx->family.normalize_audio
                                               : true, pcm, n);
    vv_free(raw);
    return s;
}

/* ─── gen ───────────────────────────────────────────────────────────────── */

/* The ids already in `path`, so a restarted run skips them. */
static char** done_ids(const char* path, int* n_out) {
    *n_out = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int cap = 1024, n = 0;
    char** ids = (char**)vv_alloc(sizeof(char*) * (size_t)cap);
    char* line = NULL;
    size_t lcap = 0;
    for (;;) {
        /* getline is POSIX; a fixed buffer would cut long token lists. */
        size_t len = 0;
        int c;
        while ((c = fgetc(f)) != EOF && c != '\n') {
            if (len + 2 > lcap) {
                lcap = lcap ? lcap * 2 : 1 << 16;
                line = (char*)vv_realloc(line, lcap);
            }
            line[len++] = (char)c;
        }
        if (len == 0 && c == EOF) break;
        if (!line) continue;
        line[len] = '\0';
        cJSON* j = cJSON_Parse(line);
        const cJSON* id = j ? cJSON_GetObjectItemCaseSensitive(j, "id") : NULL;
        if (cJSON_IsString(id)) {
            if (n == cap) { cap *= 2; ids = (char**)vv_realloc(ids, sizeof(char*) * (size_t)cap); }
            ids[n++] = dup_str(id->valuestring);
        }
        cJSON_Delete(j);
        if (c == EOF) break;
    }
    vv_free(line);
    fclose(f);
    *n_out = n;
    return ids;
}

static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

static int run_gen(const args_t* a) {
    if (!a->list || !a->out) { usage(); return 2; }
    int n_done = 0;
    char** done = done_ids(a->out, &n_done);
    if (n_done) qsort(done, (size_t)n_done, sizeof(char*), cmp_str);

    vv_inference_ctx_t* ctx = NULL;
    vv_status_t s = open_model(a, &ctx);
    if (s != VV_OK) {
        VV_LOG_E("cannot load %s: %s", a->model, vv_status_str(s));
        return 1;
    }
    FILE* in = fopen(a->list, "rb");
    FILE* out = fopen(a->out, "ab");
    if (!in || !out) { VV_LOG_E("cannot open %s or %s", a->list, a->out); return 1; }

    vv_inference_params_t ip;
    memset(&ip, 0, sizeof(ip));
    ip.max_new_tokens = a->max_tokens;

    char line[8192];
    long idx = -1, n_new = 0;
    double audio_sec = 0.0, wall = 0.0;
    int64_t tokens = 0;
    while (fgets(line, sizeof(line), in)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        idx++;
        if ((int)(idx % a->n_shards) != a->shard) continue;
        char* id = line;
        char* path = strchr(id, '\t');
        if (!path) continue;
        *path++ = '\0';
        char* hot = strchr(path, '\t');
        if (hot) *hot++ = '\0';
        if (n_done && bsearch(&id, done, (size_t)n_done, sizeof(char*), cmp_str))
            continue;

        float* pcm = NULL;
        int n = 0;
        s = load_clip(ctx, path, &pcm, &n);
        if (s != VV_OK) {
            VV_LOG_W("skip %s: %s", id, vv_status_str(s));
            continue;
        }
        const char* words[1] = { hot };
        ip.hotwords = (hot && hot[0]) ? words : NULL;
        ip.num_hotwords = (hot && hot[0]) ? 1 : 0;

        vv_transcription_t* tr = NULL;
        const double t0 = vv_time_ms();
        s = vv_inference_transcribe(ctx, pcm, n, &ip, &tr);
        const double dt = vv_time_ms() - t0;
        vv_free(pcm);
        if (s != VV_OK || !tr) {
            VV_LOG_W("skip %s: %s", id, vv_status_str(s));
            continue;
        }

        cJSON* j = cJSON_CreateObject();
        cJSON_AddStringToObject(j, "id", id);
        cJSON_AddStringToObject(j, "audio", path);
        cJSON_AddStringToObject(j, "hotwords", hot ? hot : "");
        cJSON_AddNumberToObject(j, "samples", n);
        cJSON_AddItemToObject(j, "tokens",
                              cJSON_CreateIntArray(tr->tokens, tr->num_tokens));
        cJSON_AddItemToObject(j, "chunks",
                              cJSON_CreateIntArray(tr->chunk_tokens, tr->num_chunks));
        cJSON_AddItemToObject(j, "stops",
                              cJSON_CreateIntArray(tr->chunk_stops, tr->num_chunks));
        cJSON_AddNumberToObject(j, "ms", dt);
        char* txt = cJSON_PrintUnformatted(j);
        fprintf(out, "%s\n", txt);
        fflush(out);
        cJSON_free(txt);
        cJSON_Delete(j);

        audio_sec += n / 24000.0;
        wall += dt / 1000.0;
        tokens += tr->num_tokens;
        n_new++;
        if (n_new % 20 == 0)
            fprintf(stderr, "gen: %ld clips, %.1f h audio, %lld tokens, "
                    "RTF %.3f, %.0f tok/s\n", n_new, audio_sec / 3600.0,
                    (long long)tokens, wall / (audio_sec > 0 ? audio_sec : 1),
                    tokens / (wall > 0 ? wall : 1));
        vv_transcription_free(tr);
    }
    fprintf(stderr, "gen: done, %ld new clips, %.1f h audio, %lld tokens\n",
            n_new, audio_sec / 3600.0, (long long)tokens);
    fclose(in);
    fclose(out);
    for (int i = 0; i < n_done; i++) vv_free(done[i]);
    vv_free(done);
    vv_inference_free(ctx);
    return 0;
}

/* ─── trace ─────────────────────────────────────────────────────────────── */

typedef struct {
    char*    id;
    char*    audio;
    char*    hot;
    int32_t* tokens;
    int      n_tokens;
    int*     chunks;
    int32_t* stops;
    int      n_chunks;
    int      samples;
} clip_t;

static int32_t* json_ints(const cJSON* arr, int* n) {
    *n = cJSON_GetArraySize(arr);
    int32_t* v = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)(*n > 0 ? *n : 1));
    int i = 0;
    const cJSON* e;
    cJSON_ArrayForEach(e, arr) v[i++] = (int32_t)e->valuedouble;
    return v;
}

static clip_t* load_gen(const char* path, int* n_out) {
    *n_out = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int cap = 1024, n = 0;
    clip_t* c = (clip_t*)vv_alloc(sizeof(clip_t) * (size_t)cap);
    char* line = NULL;
    size_t lcap = 0;
    for (;;) {
        size_t len = 0;
        int ch;
        while ((ch = fgetc(f)) != EOF && ch != '\n') {
            if (len + 2 > lcap) {
                lcap = lcap ? lcap * 2 : 1 << 16;
                line = (char*)vv_realloc(line, lcap);
            }
            line[len++] = (char)ch;
        }
        if (len == 0 && ch == EOF) break;
        if (line) {
            line[len] = '\0';
            cJSON* j = cJSON_Parse(line);
            if (j) {
                if (n == cap) {
                    cap *= 2;
                    c = (clip_t*)vv_realloc(c, sizeof(clip_t) * (size_t)cap);
                }
                clip_t* k = &c[n];
                memset(k, 0, sizeof(*k));
                const cJSON* x;
                x = cJSON_GetObjectItemCaseSensitive(j, "id");
                k->id = dup_str(cJSON_IsString(x) ? x->valuestring : "?");
                x = cJSON_GetObjectItemCaseSensitive(j, "audio");
                k->audio = dup_str(cJSON_IsString(x) ? x->valuestring : "");
                x = cJSON_GetObjectItemCaseSensitive(j, "hotwords");
                k->hot = dup_str(cJSON_IsString(x) ? x->valuestring : "");
                x = cJSON_GetObjectItemCaseSensitive(j, "samples");
                k->samples = cJSON_IsNumber(x) ? (int)x->valuedouble : 0;
                k->tokens = json_ints(cJSON_GetObjectItemCaseSensitive(j, "tokens"),
                                      &k->n_tokens);
                int nc = 0, ns = 0;
                int32_t* ch32 = json_ints(cJSON_GetObjectItemCaseSensitive(j, "chunks"),
                                          &nc);
                k->stops = json_ints(cJSON_GetObjectItemCaseSensitive(j, "stops"),
                                     &ns);
                k->chunks = (int*)vv_alloc(sizeof(int) * (size_t)(nc > 0 ? nc : 1));
                for (int i = 0; i < nc; i++) k->chunks[i] = ch32[i];
                vv_free(ch32);
                k->n_chunks = nc < ns ? nc : ns;
                n++;
                cJSON_Delete(j);
            }
        }
        if (ch == EOF) break;
    }
    vv_free(line);
    fclose(f);
    *n_out = n;
    return c;
}

static int count_files(const char* dir) {
#ifdef _WIN32
    (void)dir;
    return 0;
#else
    DIR* d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d)))
        if (strstr(e->d_name, ".vvdt") && !strstr(e->d_name, ".tmp")) n++;
    closedir(d);
    return n;
#endif
}

static uint32_t rng_next(uint32_t* s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return *s;
}

static int write_trace(const char* path, const clip_t* k, const int* layers,
                       int n_taps, int hs, const vv_spec_trace_t* t,
                       const uint16_t* feats) {
    char tmp[1100];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp))
        return -1;
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    const uint32_t id_len = (uint32_t)strlen(k->id);
    const uint32_t hdr[6] = { 0x54445656u /* "VVDT" */, 1u, (uint32_t)t->n,
                              (uint32_t)n_taps, (uint32_t)hs, id_len };
    static const char pad[4] = { 0, 0, 0, 0 };
    int ok = fwrite(hdr, sizeof(hdr), 1, f) == 1;
    ok = ok && fwrite(layers, sizeof(int32_t), (size_t)n_taps, f) == (size_t)n_taps;
    ok = ok && fwrite(k->id, 1, id_len, f) == id_len;
    ok = ok && fwrite(pad, 1, (4 - id_len % 4) % 4, f) == (4 - id_len % 4) % 4;
    ok = ok && fwrite(t->ids, sizeof(int32_t), (size_t)t->n, f) == (size_t)t->n;
    ok = ok && fwrite(t->kind, 1, (size_t)t->n, f) == (size_t)t->n;
    ok = ok && fwrite(pad, 1, (4 - t->n % 4) % 4, f) == (size_t)((4 - t->n % 4) % 4);
    const size_t nf = (size_t)t->n * (size_t)n_taps * (size_t)hs;
    ok = ok && fwrite(feats, 2, nf, f) == nf;
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove(tmp); return -1; }
    return rename(tmp, path) == 0 ? 0 : -1;
}

static int run_trace(const args_t* a) {
    if (!a->gen || !a->dir || !a->layers) { usage(); return 2; }
    if (make_dirs(a->dir) != 0) {
        VV_LOG_E("cannot create %s: %s", a->dir, strerror(errno));
        return 1;
    }
    int layers[VV_TAPS_MAX], n_taps = 0;
    {
        const char* p = a->layers;
        while (*p && n_taps < VV_TAPS_MAX) {
            layers[n_taps++] = atoi(p);
            const char* c = strchr(p, ',');
            if (!c) break;
            p = c + 1;
        }
    }
    int n_clips = 0;
    clip_t* clips = load_gen(a->gen, &n_clips);
    if (!clips || n_clips == 0) { VV_LOG_E("no clips in %s", a->gen); return 1; }

    vv_inference_ctx_t* ctx = NULL;
    vv_status_t s = open_model(a, &ctx);
    if (s != VV_OK) {
        VV_LOG_E("cannot load %s: %s", a->model, vv_status_str(s));
        return 1;
    }
    const int hs = ctx->model->config.llm.hidden_size;
    const int cap = a->max_pos > 0 ? a->max_pos : ctx->kv_cache->max_seq_len;

    vv_spec_trace_t t;
    memset(&t, 0, sizeof(t));
    t.taps.n = n_taps;
    for (int i = 0; i < n_taps; i++) t.taps.layers[i] = layers[i];
    t.taps.rows = cap;
    t.cap = cap + 1;
    const size_t feat_bytes = (size_t)cap * n_taps * hs * 2;
    s = vv_dev_alloc(&t.taps.buf, feat_bytes);
    t.ids = (int32_t*)vv_alloc(sizeof(int32_t) * (size_t)t.cap);
    t.kind = (uint8_t*)vv_alloc((size_t)t.cap);
    uint16_t* host = NULL;
    if (s == VV_OK) s = vv_dev_alloc_pinned((void**)&host, feat_bytes);
    if (s != VV_OK || !t.ids || !t.kind) {
        VV_LOG_E("cannot allocate %zu MB of taps", feat_bytes >> 20);
        return 1;
    }

    int* order = (int*)vv_alloc(sizeof(int) * (size_t)n_clips);
    uint32_t rng = a->seed ? a->seed : 1;
    long written = 0;
    int64_t positions = 0;
    const double t_start = vv_time_ms();
    for (int ep = 0; ep < a->epochs; ep++) {
        for (int i = 0; i < n_clips; i++) order[i] = i;
        for (int i = n_clips - 1; i > 0; i--) {     /* Fisher-Yates */
            const int j = (int)(rng_next(&rng) % (uint32_t)(i + 1));
            const int x = order[i]; order[i] = order[j]; order[j] = x;
        }
        for (int oi = 0; oi < n_clips; oi++) {
            const clip_t* k = &clips[order[oi]];
            if ((int)(order[oi] % a->n_shards) != a->shard) continue;
            while (a->ring > 0 && count_files(a->dir) >= a->ring) sleep_ms(20);

            float* pcm = NULL;
            int n = 0;
            s = load_clip(ctx, k->audio, &pcm, &n);
            if (s != VV_OK) { VV_LOG_W("skip %s: audio", k->id); continue; }
            vv_transcription_t tr;
            memset(&tr, 0, sizeof(tr));
            tr.tokens = k->tokens;
            tr.num_tokens = k->n_tokens;
            tr.chunk_tokens = k->chunks;
            tr.chunk_stops = k->stops;
            tr.num_chunks = k->n_chunks;
            s = vv_spec_trace(ctx, pcm, n, k->hot[0] ? k->hot : NULL, &tr, &t);
            vv_free(pcm);
            if (s != VV_OK) {
                VV_LOG_W("skip %s: %s (%d positions)", k->id,
                         vv_status_str(s), t.n);
                continue;
            }
            /* The last position may be a label with no row of its own. */
            const int rows = t.n <= cap ? t.n : cap;
            s = vv_dev_memcpy_d2h(host, t.taps.buf,
                                  (size_t)rows * n_taps * hs * 2, NULL);
            if (s != VV_OK) { VV_LOG_E("copy of the taps failed"); return 1; }
            if (rows < t.n)
                memset(host + (size_t)rows * n_taps * hs, 0,
                       (size_t)(t.n - rows) * n_taps * hs * 2);
            char path[1024];
            snprintf(path, sizeof(path), "%s/e%02d_%07d.vvdt", a->dir, ep, oi);
            if (write_trace(path, k, layers, n_taps, hs, &t, host) != 0) {
                VV_LOG_E("cannot write %s: %s", path, strerror(errno));
                return 1;
            }
            written++;
            positions += t.n;
            if (written % 50 == 0) {
                const double sec = (vv_time_ms() - t_start) / 1000.0;
                fprintf(stderr, "trace: epoch %d, %ld files, %.0f pos/s\n", ep,
                        written, positions / (sec > 0 ? sec : 1));
            }
        }
    }
    /* A reader waiting on the ring learns the stream has ended -- once every
     * shard writing into it has: each leaves DONE.<shard> first, and the one
     * that then finds them all leaves DONE. */
    {
        char path[1024];
        int finished = a->n_shards;
        if (a->n_shards > 1) {
            snprintf(path, sizeof(path), "%s/DONE.%d", a->dir, a->shard);
            FILE* f = fopen(path, "wb");
            if (f) fclose(f);
            finished = 0;
            for (int i = 0; i < a->n_shards; i++) {
                snprintf(path, sizeof(path), "%s/DONE.%d", a->dir, i);
                FILE* g = fopen(path, "rb");
                if (g) { finished++; fclose(g); }
            }
        }
        if (finished == a->n_shards) {
            snprintf(path, sizeof(path), "%s/DONE", a->dir);
            FILE* f = fopen(path, "wb");
            if (f) fclose(f);
        }
    }
    fprintf(stderr, "trace: done, %ld files, %lld positions\n", written,
            (long long)positions);
    vv_dev_free(t.taps.buf);
    vv_dev_free_pinned(host);
    vv_inference_free(ctx);
    return 0;
}

int main(int argc, char** argv) {
    args_t a;
    if (parse_args(argc, argv, &a) != 0) { usage(); return 2; }
    vv_log_set_level(getenv("VV_DFLASH_VERBOSE") ? VV_LOG_INFO : VV_LOG_WARN);
    if (!strcmp(a.cmd, "gen")) return run_gen(&a);
    if (!strcmp(a.cmd, "trace")) return run_trace(&a);
    usage();
    return 2;
}
