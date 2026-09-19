/**
 * @file test_model_load.c
 * @brief Config parsing, model families, and loading dense checkpoints.
 *
 * No real weights: the published configs of the BitNet and Streaming models
 * live in tests/data, and the loader is exercised on a tiny synthetic
 * checkpoint this test writes itself — BF16 projections, an F32 embedding,
 * F16 norms — so every conversion and routing decision is checked against
 * values computed here.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/model.h"
#include "vibevoice/quant.h"
#include "vibevoice/family.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/device.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#define RMDIR(p) _rmdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0755)
#define RMDIR(p) rmdir(p)
#endif

#ifndef VV_TEST_DATA_DIR
#define VV_TEST_DATA_DIR "tests/data"
#endif

static int failures = 0;
#define CHECK(cond, msg) do {                                             \
    if (cond) printf("  PASS: %s\n", msg);                                \
    else { printf("  FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
} while (0)

/* ─── Configs of the published models ───────────────────────────────────── */

static void test_published_configs(void) {
    printf("published configs:\n");
    vv_model_config_t c;
    vv_status_t s;

    s = vv_config_parse(VV_TEST_DATA_DIR "/bitnet_config.json", &c);
    CHECK(s == VV_OK, "BitNet config parses");
    CHECK(c.family == VV_FAMILY_ASR_BITNET, "BitNet family detected");
    CHECK(c.llm.hidden_size == 1536 && c.llm.num_hidden_layers == 28 &&
          c.llm.num_attention_heads == 12 && c.llm.num_key_value_heads == 2 &&
          c.llm.intermediate_size == 8960 && c.llm.vocab_size == 151936,
          "BitNet dims 1536/28/12/2/8960/151936");
    CHECK(c.llm.head_dim == 128, "BitNet head_dim 128");
    CHECK(c.llm.tie_word_embeddings, "BitNet ties its head (decoder_config)");
    CHECK(c.llm.max_position_embeddings == 65536, "BitNet 64K positions");
    CHECK(c.audio.normalize_audio && c.audio.chunk_frames == 0,
          "BitNet: normalized, one-shot");

    s = vv_config_parse(VV_TEST_DATA_DIR "/streaming_config.json", &c);
    CHECK(s == VV_OK, "Streaming config parses");
    CHECK(c.family == VV_FAMILY_ASR_STREAMING_7B,
          "Streaming family from its architecture");
    CHECK(c.llm.hidden_size == 3584 && c.llm.vocab_size == 152064 &&
          !c.llm.tie_word_embeddings, "Streaming is 7B-shaped, untied");
    s = vv_config_parse_preprocessor(
        VV_TEST_DATA_DIR "/streaming_preprocessor_config.json", &c);
    CHECK(s == VV_OK, "Streaming preprocessor config parses");
    CHECK(c.audio.chunk_frames == 22 && c.audio.lookahead_frames == 4,
          "Streaming chunk 22 + lookahead 4 frames");
    CHECK(!c.audio.normalize_audio, "Streaming does not normalize");

    s = vv_config_parse(VV_TEST_DATA_DIR "/asr7b_bf16_config.json", &c);
    CHECK(s == VV_OK && c.family == VV_FAMILY_ASR_7B,
          "microsoft/VibeVoice-ASR is asr-7b");
    CHECK(c.llm.attention_bias && !c.llm.tie_word_embeddings,
          "7B: qkv bias on, head untied");
    /* No preprocessor file: the processor's defaults. */
    s = vv_config_parse_preprocessor("does/not/exist.json", &c);
    CHECK(s == VV_OK && c.audio.normalize_audio &&
          c.audio.compress_ratio == 3200, "missing preprocessor = defaults");
}

/*
 * Append to a fixed buffer, refusing rather than overflowing: snprintf's
 * return is what it *would* have written, so adding it to the length
 * unchecked walks past the end once the buffer fills.
 */
static int appendf(char* buf, size_t cap, size_t* len, const char* fmt, ...) {
    if (*len >= cap) return -1;
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap - *len) return -1;
    *len += (size_t)n;
    return 0;
}

static int write_text(const char* path, const char* text) {
    FILE* f = fopen(path, "wb");
    if (!f) return 1;
    fputs(text, f);
    fclose(f);
    return 0;
}

static void test_config_refusals(void) {
    printf("config refusals:\n");
    vv_model_config_t c;
    const char* path = "test_model_load_cfg.json";

    write_text(path, "{\"decoder_config\":{\"hidden_size\":1536,"
                     "\"num_hidden_layers\":28,\"num_attention_heads\":12,"
                     "\"vocab_size\":100}}");
    CHECK(vv_config_parse(path, &c) == VV_ERR_MODEL_FORMAT,
          "missing intermediate_size is an error, not 18944");

    write_text(path, "{\"hidden_size\":64,\"num_hidden_layers\":1,"
                     "\"num_attention_heads\":1,\"intermediate_size\":64,"
                     "\"vocab_size\":10,\"hidden_act\":\"gelu\"}");
    CHECK(vv_config_parse(path, &c) == VV_ERR_UNSUPPORTED,
          "hidden_act gelu is refused");

    write_text(path, "{\"hidden_size\":64,\"num_hidden_layers\":1,"
                     "\"num_attention_heads\":1,\"intermediate_size\":64,"
                     "\"vocab_size\":10,\"rope_scaling\":{\"type\":\"yarn\"}}");
    CHECK(vv_config_parse(path, &c) == VV_ERR_UNSUPPORTED,
          "rope_scaling is refused");

    write_text(path, "{\"hidden_size\":64,\"num_hidden_layers\":1,"
                     "\"num_attention_heads\":1,\"intermediate_size\":64,"
                     "\"vocab_size\":10,\"tie_word_embeddings\":true}");
    CHECK(vv_config_parse(path, &c) == VV_OK && c.llm.tie_word_embeddings &&
          c.llm.num_key_value_heads == 1 && c.llm.head_dim == 64,
          "flat config: root tie, kv heads default to heads");
    remove(path);
}

/* ─── Family behaviour without a tokenizer ──────────────────────────────── */

static void test_family_logic(void) {
    printf("family stop / chunk prompt:\n");
    vv_family_t f;
    memset(&f, 0, sizeof(f));
    f.tok.im_start = 151644; f.tok.im_end = 151645; f.tok.endoftext = 151643;
    f.tok.speech_start = 151646; f.tok.speech_end = 151647;
    f.tok.speech_pad = 151648; f.tok.text_chunk_end = 151665;
    f.frame_samples = 3200;

    f.id = VV_FAMILY_ASR_7B; f.mode = VV_GEN_ONE_SHOT;
    CHECK(vv_family_stop(&f, 151645) == VV_STOP_END, "7B: im_end ends");
    CHECK(vv_family_stop(&f, 151643) == VV_STOP_END, "7B: endoftext ends");
    CHECK(vv_family_stop(&f, 1000) == VV_STOP_CONTINUE, "7B: text continues");

    f.id = VV_FAMILY_ASR_STREAMING_7B; f.mode = VV_GEN_CHUNKED;
    f.chunk_frames = 22; f.lookahead_frames = 4;
    CHECK(vv_family_stop(&f, 151665) == VV_STOP_YIELD,
          "streaming: text_chunk_end yields");
    CHECK(vv_family_stop(&f, 151643) == VV_STOP_END,
          "streaming: endoftext ends");
    CHECK(vv_family_stop(&f, 151645) == VV_STOP_CONTINUE,
          "streaming: im_end is text, as upstream");
    CHECK(vv_family_chunk_samples(&f) == 70400 &&
          vv_family_window_samples(&f) == 83200,
          "streaming: 22 frames of text in a 26-frame window");

    vv_prompt_t p;
    CHECK(vv_family_chunk_prompt(&f, 26, true, &p) == VV_OK, "chunk prompt");
    CHECK(p.n == 29 && p.ids[0] == 151665 && p.ids[1] == 151646 &&
          p.audio_offset == 2 && p.n_audio == 26 && p.ids[2] == 151648 &&
          p.ids[27] == 151648 && p.ids[28] == 151647,
          "chunk: [chunk_end, start, pad x26, end]");
    vv_prompt_free(&p);
    CHECK(vv_family_chunk_prompt(&f, 26, false, &p) == VV_OK &&
          p.n == 28 && p.audio_offset == 1, "first chunk has no chunk_end");
    vv_prompt_free(&p);
}

/* ─── A synthetic checkpoint ────────────────────────────────────────────── */

enum { HS = 128, NH = 1, NKV = 1, HD = 128, INTER = 256, VOCAB = 64,
       LAYERS = 2, VAE_A = 4, VAE_S = 8, TIE_AT = 40 };

/* ─── FP16 rounding ─────────────────────────────────────────────────────── */

/**
 * @brief Nearest FP16 to `f`, ties to even, found by search rather than by
 *        bit manipulation, so it shares nothing with the code it checks.
 *        Finite inputs below 65504 only.
 */
static uint16_t half_rne_ref(float f) {
    const uint16_t sign = (f < 0.0f || (f == 0.0f && 1.0f / f < 0.0f))
                        ? 0x8000u : 0u;
    const double a = fabs((double)f);
    /* Magnitudes are monotonic in the bit pattern: bisect for the last
       half not above |f|, then compare it with the next one up. */
    uint16_t lo = 0, hi = 0x7BFF;
    while (lo < hi) {
        const uint16_t mid = (uint16_t)((lo + hi + 1) / 2);
        if ((double)vv_half_to_float(mid) <= a) lo = mid; else hi = mid - 1;
    }
    uint16_t best = lo;
    if (lo < 0x7BFF) {
        const double d0 = a - (double)vv_half_to_float(lo);
        const double d1 = (double)vv_half_to_float((uint16_t)(lo + 1)) - a;
        if (d1 < d0 || (d1 == d0 && (lo & 1u))) best = (uint16_t)(lo + 1);
    }
    return (uint16_t)(sign | best);
}

static void test_half_rne(void) {
    printf("FP16 round-to-nearest-even:\n");
    CHECK(vv_float_to_half_rne(1.0f + 1.0f / 2048.0f) == 0x3C00,
          "1 + 2^-11 -> 1.0 (0x3C00), a tie to even");
    CHECK(vv_float_to_half_rne(1.0f + 3.0f / 2048.0f) == 0x3C02,
          "1 + 3*2^-11 -> 0x3C02, a tie to even");
    CHECK(vv_float_to_half_rne(3.0f * ldexpf(1.0f, -25)) == 0x0002 &&
          vv_float_to_half_rne(ldexpf(1.0f, -25)) == 0x0000 &&
          vv_float_to_half_rne(ldexpf(1.0f, -25) * 1.0001f) == 0x0001,
          "subnormal ties and the underflow edge");
    CHECK(vv_float_to_half_rne(65520.0f) == 0x7C00 &&
          vv_float_to_half_rne(65519.0f) == 0x7BFF &&
          vv_float_to_half_rne(-1.0e9f) == 0xFC00,
          "65520 rounds to infinity, 65519 does not");
    uint32_t r = 987654321u;
    int bad = 0;
    for (int i = 0; i < 200000; i++) {
        r = r * 1664525u + 1013904223u;
        /* Exponents from far below the subnormals to just under 65504. */
        const uint32_t e = 90u + (r >> 8) % 52u;
        uint32_t u = (r & 0x80000000u) | (e << 23) | ((r * 2654435761u) >> 9);
        /* Every eighth value is made an exact tie. */
        if ((i & 7) == 0) u = (u & ~0x1FFFu) | 0x1000u;
        float f;
        memcpy(&f, &u, sizeof(f));
        if (fabsf(f) >= 65504.0f) continue;
        if (vv_float_to_half_rne(f) != half_rne_ref(f)) bad++;
    }
    CHECK(bad == 0, "200k values, an eighth of them ties, match a search");
}

typedef struct {
    char        name[128];
    const char* dtype;     /* "BF16" | "F16" | "F32" */
    int64_t     shape[2];
    int         ndim;
    void*       data;
    size_t      bytes;
} fake_t;

static fake_t g_t[64];
static int g_n;

static uint32_t g_rng = 12345u;
static float frand(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float)(g_rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static uint16_t f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);      /* truncation is fine for test data */
}

static void add(const char* name, const char* dtype, int64_t d0, int64_t d1,
                float scale) {
    fake_t* t = &g_t[g_n++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dtype = dtype;
    t->shape[0] = d0; t->shape[1] = d1;
    t->ndim = d1 > 0 ? 2 : 1;
    const size_t n = (size_t)d0 * (size_t)(d1 > 0 ? d1 : 1);
    const size_t es = strcmp(dtype, "F32") == 0 ? 4 : 2;
    t->bytes = n * es;
    t->data = malloc(t->bytes);
    for (size_t i = 0; i < n; i++) {
        float v = frand() * scale;
        /* A few tiny values: the old BF16->FP16 truncation flushed these. */
        if (i % 97 == 5) v = 3.0e-6f;
        if (es == 4) ((float*)t->data)[i] = v;
        else if (strcmp(dtype, "BF16") == 0) ((uint16_t*)t->data)[i] = f2bf(v);
        else ((uint16_t*)t->data)[i] = vv_float_to_half(v);
    }
}

static const fake_t* get(const char* name) {
    for (int i = 0; i < g_n; i++)
        if (strcmp(g_t[i].name, name) == 0) return &g_t[i];
    return NULL;
}

static void build_tensors(bool with_head) {
    for (int i = 0; i < g_n; i++) free(g_t[i].data);
    g_n = 0;
    g_rng = 12345u;
    char nm[128];
    const char* P = "model.language_model.";
    snprintf(nm, sizeof(nm), "%sembed_tokens.weight", P);
    add(nm, "F32", VOCAB, HS, 0.05f);
    {
        /* Exact FP16 ties: 1 + 2^-11 and 1 + 3 * 2^-11 sit halfway between
           two halves, as does 3 * 2^-25 between subnormals 1 and 2. */
        float* e = (float*)g_t[g_n - 1].data;
        e[TIE_AT]     = 1.0f + 1.0f / 2048.0f;
        e[TIE_AT + 1] = 1.0f + 3.0f / 2048.0f;
        e[TIE_AT + 2] = -(1.0f + 1.0f / 2048.0f);
        e[TIE_AT + 3] = 3.0f * ldexpf(1.0f, -25);
    }
    snprintf(nm, sizeof(nm), "%snorm.weight", P);
    add(nm, "F16", HS, 0, 1.0f);
    if (with_head) {
        /* Same values as the embedding: tied and untied must agree. */
        fake_t* h = &g_t[g_n++];
        *h = g_t[0];
        snprintf(h->name, sizeof(h->name), "lm_head.weight");
        h->data = malloc(h->bytes);
        memcpy(h->data, g_t[0].data, h->bytes);
    }
    for (int l = 0; l < LAYERS; l++) {
        struct { const char* w; int n, k; bool bias; } pr[] = {
            {"self_attn.q_proj", NH * HD, HS, true},
            {"self_attn.k_proj", NKV * HD, HS, true},
            {"self_attn.v_proj", NKV * HD, HS, true},
            {"self_attn.o_proj", HS, NH * HD, false},
            {"mlp.gate_proj", INTER, HS, false},
            {"mlp.up_proj", INTER, HS, false},
            {"mlp.down_proj", HS, INTER, false},
        };
        snprintf(nm, sizeof(nm), "%slayers.%d.input_layernorm.weight", P, l);
        add(nm, "F16", HS, 0, 1.0f);
        snprintf(nm, sizeof(nm), "%slayers.%d.post_attention_layernorm.weight",
                 P, l);
        add(nm, "F16", HS, 0, 1.0f);
        for (int i = 0; i < 7; i++) {
            snprintf(nm, sizeof(nm), "%slayers.%d.%s.weight", P, l, pr[i].w);
            add(nm, "BF16", pr[i].n, pr[i].k, 0.1f);
            if (pr[i].bias) {
                snprintf(nm, sizeof(nm), "%slayers.%d.%s.bias", P, l, pr[i].w);
                add(nm, "BF16", pr[i].n, 0, 0.1f);
            }
        }
    }
    add("model.acoustic_connector.fc1.weight", "BF16", HS, VAE_A, 0.1f);
    add("model.acoustic_connector.fc1.bias", "BF16", HS, 0, 0.1f);
    add("model.acoustic_connector.norm.weight", "BF16", HS, 0, 1.0f);
    add("model.acoustic_connector.fc2.weight", "BF16", HS, HS, 0.1f);
    add("model.semantic_connector.fc1.weight", "BF16", HS, VAE_S, 0.1f);
    add("model.semantic_connector.norm.weight", "BF16", HS, 0, 1.0f);
    add("model.semantic_connector.fc2.weight", "BF16", HS, HS, 0.1f);
    add("model.acoustic_tokenizer.encoder.probe.weight", "BF16", 16, 0, 1.0f);
    add("model.semantic_tokenizer.encoder.probe.weight", "BF16", 16, 0, 1.0f);
}

/** @brief Write `dir` with config.json and model.safetensors. `skip`/`bad`
 *  name one tensor to leave out or to write with the wrong shape. */
static int write_model(const char* dir, bool tie, const char* skip,
                       const char* bad) {
    MKDIR(dir);
    char path[512];
    char cfg[1024];
    snprintf(cfg, sizeof(cfg),
        "{\"architectures\":[\"VibeVoiceForASRTraining\"],"
        "\"acoustic_tokenizer_config\":{\"vae_dim\":%d},"
        "\"semantic_tokenizer_config\":{\"vae_dim\":%d},"
        "\"decoder_config\":{\"hidden_size\":%d,\"num_hidden_layers\":%d,"
        "\"num_attention_heads\":%d,\"num_key_value_heads\":%d,"
        "\"intermediate_size\":%d,\"vocab_size\":%d,\"hidden_act\":\"silu\","
        "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-6,"
        "\"tie_word_embeddings\":%s}}",
        VAE_A, VAE_S, HS, LAYERS, NH, NKV, INTER, VOCAB, tie ? "true" : "false");
    snprintf(path, sizeof(path), "%s/config.json", dir);
    if (write_text(path, cfg)) return 1;

    /* Header: JSON, padded with spaces to a multiple of 8. */
    size_t cap = 65536, len = 0, off = 0;
    char* hdr = (char*)malloc(cap);
    if (!hdr) return 1;
    int trunc = appendf(hdr, cap, &len, "{");
    bool first = true;
    for (int i = 0; i < g_n; i++) {
        const fake_t* t = &g_t[i];
        if (skip && strcmp(t->name, skip) == 0) continue;
        int64_t d0 = t->shape[0];
        if (bad && strcmp(t->name, bad) == 0) d0 /= 2;   /* bytes stay */
        if (t->ndim == 2)
            trunc |= appendf(hdr, cap, &len,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld,%lld],"
                "\"data_offsets\":[%zu,%zu]}", first ? "" : ",", t->name,
                t->dtype, (long long)d0, (long long)t->shape[1],
                off, off + t->bytes);
        else
            trunc |= appendf(hdr, cap, &len,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],"
                "\"data_offsets\":[%zu,%zu]}", first ? "" : ",", t->name,
                t->dtype, (long long)d0, off, off + t->bytes);
        off += t->bytes;
        first = false;
    }
    trunc |= appendf(hdr, cap, &len, "}");
    if (trunc || len + 8 > cap) { free(hdr); return 1; }   /* header too big */
    while (len % 8) hdr[len++] = ' ';

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { free(hdr); return 1; }
    uint64_t hl = len;
    fwrite(&hl, 8, 1, f);
    fwrite(hdr, 1, len, f);
    for (int i = 0; i < g_n; i++) {
        if (skip && strcmp(g_t[i].name, skip) == 0) continue;
        fwrite(g_t[i].data, 1, g_t[i].bytes, f);
    }
    fclose(f);
    free(hdr);
    return 0;
}

static vv_status_t load(const char* dir, int quant, vv_model_t** m) {
    vv_model_load_opts_t o = vv_model_load_opts_default();
    o.quant = quant;
    return vv_model_load_ex(dir, &o, m);
}

static void test_dense_load(void) {
    printf("dense load:\n");
    const char* dir = "test_model_load_dense";
    build_tensors(false);
    if (write_model(dir, true, NULL, NULL)) {
        printf("  SKIP: cannot write %s\n", dir);
        return;
    }

    vv_model_t* m = NULL;
    vv_status_t s = load(dir, VV_LOAD_QUANT_AUTO, &m);
    CHECK(s == VV_OK && m, "tiny BF16/F32 checkpoint loads");
    if (!m) return;

    const vv_weight_t* q = &m->layers[1].attn.q_proj;
    const fake_t* src = get("model.language_model.layers.1.self_attn.q_proj.weight");
    bool exact = q->quant_kind == VV_QUANT_NONE && q->tensor.dtype == VV_DTYPE_F16 &&
                 q->tensor.size_bytes == (size_t)NH * HD * HS * 2;
    for (int i = 0; exact && i < NH * HD * HS; i++)
        exact = ((const uint16_t*)q->tensor.data)[i] ==
                vv_float_to_half(vv_bf16_to_float(((uint16_t*)src->data)[i]));
    CHECK(exact, "dense projection is FP16 of the BF16, subnormals kept");
    CHECK(((const uint16_t*)q->tensor.data)[5] != 0,
          "3e-6 survives (was flushed to zero)");

    const fake_t* es = get("model.language_model.embed_tokens.weight");
    exact = m->embed_tokens.dtype == VV_DTYPE_F16;
    for (int i = 0; exact && i < VOCAB * HS; i++)
        exact = ((const uint16_t*)m->embed_tokens.data)[i] ==
                half_rne_ref(((float*)es->data)[i]);
    CHECK(exact, "F32 embedding becomes FP16, rounded to nearest-even");
    {
        const uint16_t* e16 = (const uint16_t*)m->embed_tokens.data;
        CHECK(e16[TIE_AT] == 0x3C00 && e16[TIE_AT + 1] == 0x3C02 &&
              e16[TIE_AT + 2] == 0xBC00 && e16[TIE_AT + 3] == 0x0002,
              "exact ties in the F32 embedding went to even");
    }

    const fake_t* fc = get("model.acoustic_connector.fc1.weight");
    exact = m->acoustic_connector_fc1.tensor.dtype == VV_DTYPE_F32;
    for (int i = 0; exact && i < HS * VAE_A; i++)
        exact = ((const float*)m->acoustic_connector_fc1.tensor.data)[i] ==
                vv_bf16_to_float(((uint16_t*)fc->data)[i]);
    CHECK(exact, "front-end BF16 widens to FP32 exactly");
    CHECK(m->semantic_connector_fc1.quant.packed.data == NULL,
          "absent connector bias stays absent");
    CHECK(m->layers[0].attn.q_proj.bias.data && !m->layers[0].attn.o_proj.bias.data,
          "q bias loaded, o has none");

    CHECK(m->lm_head_tied && m->lm_head.data == m->embed_tokens.data,
          "tied head is the embedding buffer");
    CHECK(m->config.family == VV_FAMILY_ASR_7B, "family asr-7b");

    /* Tied vs an explicit head with the same values: same greedy token. */
    float x[HS];
    for (int i = 0; i < HS; i++) x[i] = frand();
    int32_t t_tied = -1, t_expl = -2;
    float v_tied = 0, v_expl = 1;
    vv_lm_head_argmax_cpu(x, m->lm_head.data, VOCAB, HS, &t_tied, &v_tied);
    vv_model_free(m);

    build_tensors(true);
    write_model(dir, false, NULL, NULL);
    m = NULL;
    s = load(dir, VV_LOAD_QUANT_AUTO, &m);
    CHECK(s == VV_OK && m && !m->lm_head_tied &&
          m->lm_head.data != m->embed_tokens.data, "untied head is its own");
    if (m) {
        vv_lm_head_argmax_cpu(x, m->lm_head.data, VOCAB, HS, &t_expl, &v_expl);
        CHECK(t_tied == t_expl && v_tied == v_expl,
              "tied head == explicit head (token and logit)");
        vv_model_free(m);
    }

    /* Refusals name the tensor instead of crashing later. */
    build_tensors(false);
    write_model(dir, true,
                "model.language_model.layers.1.mlp.down_proj.weight", NULL);
    m = NULL;
    CHECK(load(dir, VV_LOAD_QUANT_AUTO, &m) == VV_ERR_WEIGHT_MISSING && !m,
          "missing projection refused");
    write_model(dir, true, "model.language_model.layers.0.self_attn.k_proj.bias",
                NULL);
    CHECK(load(dir, VV_LOAD_QUANT_AUTO, &m) == VV_ERR_WEIGHT_MISSING,
          "missing qkv bias refused (attention_bias)");
    write_model(dir, true, NULL,
                "model.language_model.layers.0.mlp.gate_proj.weight");
    CHECK(load(dir, VV_LOAD_QUANT_AUTO, &m) == VV_ERR_SHAPE_MISMATCH,
          "misshaped projection refused");
    write_model(dir, true, NULL, NULL);
}

/** @brief Weight [n,k] of the source checkpoint, as the BF16 it was stored as. */
static double src_w(const fake_t* src, int K, int n, int k) {
    return vv_bf16_to_float(((const uint16_t*)src->data)[(size_t)n * K + k]);
}

/**
 * @brief Largest |dequant - source| relative to the row's largest |w|.
 *
 * Decodes the codes by the documented layout (quant.h), independently of
 * the kernels; test_kernels() then checks the kernels agree with it.
 */
static double quant_error(const vv_weight_t* w, const fake_t* src, int N, int K) {
    double worst = 0.0;
    extern const float VV_NF4_TABLE[16];
    for (int n = 0; n < N; n++) {
        double amax = 0.0;
        for (int k = 0; k < K; k++) {
            const double v = fabs(src_w(src, K, n, k));
            if (v > amax) amax = v;
        }
        for (int k = 0; k < K; k++) {
            double d;
            if (w->quant_kind == VV_QUANT_INT8) {
                d = ((const int8_t*)w->tensor.data)[(size_t)n * K + k] *
                    (double)((const float*)w->quant.scales.data)[n];
            } else {
                const uint8_t b = ((const uint8_t*)w->tensor.data)[(n * K + k) / 2];
                const int code = (k & 1) ? (b & 15) : (b >> 4);
                if (w->quant_kind == VV_QUANT_NF4) {
                    const uint16_t sc = ((const uint16_t*)w->quant.scales.data)[(n * K + k) / 64];
                    d = VV_NF4_TABLE[code] * vv_half_to_float(sc);
                } else {
                    const int g = k / w->group_size, G = K / w->group_size;
                    d = code * vv_half_to_float(((const uint16_t*)w->quant.scales.data)[n * G + g])
                        + vv_half_to_float(((const uint16_t*)w->mins.data)[n * G + g]);
                }
            }
            const double e = fabs(d - src_w(src, K, n, k));
            if (amax > 0 && e / amax > worst) worst = e / amax;
        }
    }
    return worst;
}

static bool same_bytes(const vv_tensor_t* a, const vv_tensor_t* b) {
    return a->size_bytes == b->size_bytes &&
           (a->size_bytes == 0 || memcmp(a->data, b->data, a->size_bytes) == 0);
}

static double cosine(const float* a, const double* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) {
        ab += a[i] * b[i]; aa += (double)a[i] * a[i]; bb += b[i] * b[i];
    }
    return (aa > 0 && bb > 0) ? ab / sqrt(aa * bb) : 0.0;
}

/** @brief The CPU kernel the decoder would run for this weight. */
static vv_status_t cpu_linear(const vv_weight_t* w, const float* x, float* y,
                              int M, int N, int K) {
    switch (w->quant_kind) {
    case VV_QUANT_NF4:
        return vv_nf4_gemm_cpu(x, (const uint8_t*)w->tensor.data,
                               w->quant.scales.data, NULL, y, M, N, K);
    case VV_QUANT_INT4G:
        return vv_int4g_gemm_cpu(x, (const uint8_t*)w->tensor.data,
                                 w->quant.scales.data, w->mins.data, NULL, y,
                                 M, N, K, w->group_size);
    case VV_QUANT_INT8:
        return vv_int8_gemm_cpu(x, (const int8_t*)w->tensor.data,
                                (const float*)w->quant.scales.data, NULL, y,
                                M, N, K);
    default:
        return vv_gemm_f16w_cpu(x, w->tensor.data, NULL, y, M, N, K);
    }
}

#ifdef VV_HAS_ACCEL
/** @brief The GPU kernels quant_linear() in decoder.c dispatches to. */
static vv_status_t gpu_linear(const vv_weight_t* w, void* const dt[3],
                              const void* x, void* y, void* scratch,
                              int M, int N, int K, void* stream) {
    switch (w->quant_kind) {
    case VV_QUANT_NF4:
        return M == 1
            ? vv_nf4_gemv_dev(x, (const uint8_t*)dt[0], dt[1], NULL, y, N, K,
                              stream)
            : vv_nf4_gemm_dev(x, (const uint8_t*)dt[0], dt[1], y, scratch,
                              M, N, K, 64, stream);
    case VV_QUANT_INT4G:
        return M == 1
            ? vv_awq_gemv_dev(x, (const uint32_t*)dt[0],
                              (const uint32_t*)dt[2], dt[1], NULL, y, N, K,
                              w->group_size, stream)
            : vv_awq_gemm_dev(x, (const uint32_t*)dt[0],
                              (const uint32_t*)dt[2], dt[1], y, scratch,
                              M, N, K, w->group_size, stream);
    case VV_QUANT_INT8:
        return M == 1
            ? vv_int8_gemv_dev(x, (const int8_t*)dt[0], (const float*)dt[1],
                               NULL, y, N, K, stream)
            : vv_int8_gemm_dev(x, (const int8_t*)dt[0], (const float*)dt[1],
                               y, scratch, M, N, K, stream);
    default:
        return vv_gemm_fp16_dev(x, dt[0], y, M, N, K, 1.0f, 0.0f, stream);
    }
}
#endif

/**
 * @brief The quantized weight through the kernels that will run it, against
 *        the dense source: pins the quantizer's layout to the kernels'.
 *
 * M = 1 is the decode GEMV, 5 the small-batch path (and the CPU's packed
 * GEMM), 17 the GPU's dequantize-then-GEMM path.
 */
static void test_kernels(const char* nm, const vv_weight_t* w,
                         const fake_t* src, int N, int K) {
    enum { MMAX = 17 };
    static const int Ms[3] = { 1, 5, MMAX };
    float* x = (float*)malloc((size_t)MMAX * K * sizeof(float));
    uint16_t* xh = (uint16_t*)malloc((size_t)MMAX * K * sizeof(uint16_t));
    float* y = (float*)malloc((size_t)MMAX * N * sizeof(float));
    double* ref = (double*)malloc((size_t)MMAX * N * sizeof(double));
    for (int i = 0; i < MMAX * K; i++) {
        xh[i] = vv_float_to_half_rne(frand());
        x[i] = vv_half_to_float(xh[i]);     /* same inputs on both sides */
    }
    for (int m = 0; m < MMAX; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double)x[(size_t)m * K + k] * src_w(src, K, n, k);
            ref[(size_t)m * N + n] = acc;
        }
    /* Quantization error alone bounds how far off an honest kernel can be;
       a layout mismatch (wrong nibble order, wrong scale index) scrambles
       the weights and lands near 0. */
    const double min_cos = w->quant_kind == VV_QUANT_INT8 ? 0.9999
                         : w->quant_kind == VV_QUANT_NONE ? 0.99999 : 0.99;
    char msg[160];

    for (int i = 0; i < 3; i++) {
        const int M = Ms[i];
        const vv_status_t s = cpu_linear(w, x, y, M, N, K);
        const double c = cosine(y, ref, (size_t)M * N);
        snprintf(msg, sizeof(msg), "%s: CPU kernel M=%d vs dense, cos %.6f",
                 nm, M, c);
        CHECK(s == VV_OK && c > min_cos, msg);
    }

#ifdef VV_HAS_ACCEL
    size_t total = 0;
    if (vv_dev_get_device_info(0, &total, NULL, NULL) != VV_OK || total == 0) {
        printf("  SKIP: %s GPU kernels, no device\n", nm);
        goto done;
    }
    {
        const vv_tensor_t* ht[3] = { &w->tensor, &w->quant.scales, &w->mins };
        void* dt[3] = { NULL, NULL, NULL };
        void *dx = NULL, *dy = NULL, *dscr = NULL, *stream = NULL;
        bool ok = vv_dev_stream_create(&stream) == VV_OK;
        for (int j = 0; j < 3 && ok; j++) {
            if (!ht[j]->data) continue;
            ok = vv_dev_alloc(&dt[j], ht[j]->size_bytes) == VV_OK &&
                 vv_dev_memcpy_h2d(dt[j], ht[j]->data, ht[j]->size_bytes,
                                   stream) == VV_OK;
        }
        ok = ok && vv_dev_alloc(&dx, (size_t)MMAX * K * 2) == VV_OK &&
             vv_dev_alloc(&dy, (size_t)MMAX * N * 2) == VV_OK &&
             vv_dev_alloc(&dscr, (size_t)N * K * 2) == VV_OK &&
             vv_dev_memcpy_h2d(dx, xh, (size_t)MMAX * K * 2, stream) == VV_OK;
        uint16_t* yh = (uint16_t*)malloc((size_t)MMAX * N * sizeof(uint16_t));
        for (int i = 0; i < 3 && ok; i++) {
            const int M = Ms[i];
            vv_status_t s = gpu_linear(w, dt, dx, dy, dscr, M, N, K, stream);
            if (s == VV_OK) s = vv_dev_stream_sync(stream);
            if (s == VV_OK)
                s = vv_dev_memcpy_d2h(yh, dy, (size_t)M * N * 2, stream);
            if (s == VV_OK) s = vv_dev_stream_sync(stream);
            for (int j = 0; j < M * N; j++) y[j] = vv_half_to_float(yh[j]);
            const double c = cosine(y, ref, (size_t)M * N);
            snprintf(msg, sizeof(msg), "%s: GPU kernel M=%d vs dense, cos %.6f",
                     nm, M, c);
            CHECK(s == VV_OK && c > min_cos, msg);
        }
        if (!ok) printf("  SKIP: %s GPU kernels, allocation\n", nm);
        free(yh);
        for (int j = 0; j < 3; j++) if (dt[j]) vv_dev_free(dt[j]);
        if (dx) vv_dev_free(dx);
        if (dy) vv_dev_free(dy);
        if (dscr) vv_dev_free(dscr);
        if (stream) vv_dev_stream_destroy(stream);
    }
done:
#endif
    free(x); free(xh); free(y); free(ref);
}

static void test_load_quant(void) {
    printf("load-time quantization:\n");
    const char* dir = "test_model_load_dense";
    const fake_t* src = get("model.language_model.layers.1.mlp.down_proj.weight");
    static const int QS[3] = { VV_LOAD_QUANT_NF4, VV_LOAD_QUANT_INT4,
                               VV_LOAD_QUANT_INT8 };
    for (int qi = 0; qi < 3; qi++) {
        const int q = QS[qi];
        const char* nm = vv_load_quant_name((vv_load_quant_t)q);
        char msg[128];
        vv_model_t *a = NULL, *b = NULL;
#ifdef _OPENMP
        const int max_threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        vv_status_t sa = load(dir, q, &a);
#ifdef _OPENMP
        omp_set_num_threads(max_threads > 1 ? max_threads : 4);
#endif
        vv_status_t sb = load(dir, q, &b);
#ifdef _OPENMP
        CHECK(omp_get_max_threads() == (max_threads > 1 ? max_threads : 4),
              "the load puts the caller's thread count back");
        omp_set_num_threads(max_threads);
#endif
        snprintf(msg, sizeof(msg), "--quant %s loads", nm);
        CHECK(sa == VV_OK && sb == VV_OK && a && b, msg);
        if (!a || !b) { vv_model_free(a); vv_model_free(b); continue; }

        const vv_weight_t* w = &a->layers[1].mlp.down_proj;
        snprintf(msg, sizeof(msg), "%s: kind, code and scale sizes", nm);
        if (q == VV_LOAD_QUANT_INT8)
            CHECK(w->quant_kind == VV_QUANT_INT8 &&
                  w->tensor.dtype == VV_DTYPE_I8 &&
                  w->tensor.size_bytes == (size_t)HS * INTER &&
                  w->quant.scales.dtype == VV_DTYPE_F32 &&
                  w->quant.scales.size_bytes == (size_t)HS * 4 &&
                  w->mins.data == NULL, msg);
        else
            CHECK(w->quant_kind == (q == VV_LOAD_QUANT_INT4 ? VV_QUANT_INT4G
                                                            : VV_QUANT_NF4) &&
                  w->tensor.size_bytes == (size_t)HS * INTER / 2 &&
                  w->quant.scales.size_bytes ==
                      (size_t)HS * (INTER / (q == VV_LOAD_QUANT_INT4 ? 128 : 64)) * 2 &&
                  (q == VV_LOAD_QUANT_INT4 ? w->group_size == 128 && w->mins.data != NULL
                                           : w->mins.data == NULL), msg);

        bool same = true;
        for (int l = 0; l < LAYERS && same; l++) {
            vv_tensor_t *ta[VV_LAYER_TENSOR_SLOTS], *tb[VV_LAYER_TENSOR_SLOTS];
            const int n = vv_layer_tensors(&a->layers[l], ta);
            vv_layer_tensors(&b->layers[l], tb);
            for (int k = 0; k < n && same; k++) same = same_bytes(ta[k], tb[k]);
        }
        snprintf(msg, sizeof(msg), "%s: same bytes on 1 thread and on many", nm);
        CHECK(same, msg);

        /* Half a step of each format, as a fraction of the row's max:
           NF4's widest gap is -1 to -0.6962, INT4's step is at most 2/15,
           INT8's 1/127. */
        const double bound = q == VV_LOAD_QUANT_NF4 ? 0.153
                           : q == VV_LOAD_QUANT_INT4 ? 0.07 : 0.0041;
        const double err = quant_error(w, src, HS, INTER);
        snprintf(msg, sizeof(msg), "%s: worst error %.4f of the row's max "
                 "(bound %.3f)", nm, err, bound);
        CHECK(err < bound, msg);

        test_kernels(nm, w, src, HS, INTER);
        vv_model_free(a);
        vv_model_free(b);
    }

    /* The dense path through the same check, as the control. */
    vv_model_t* d = NULL;
    if (load(dir, VV_LOAD_QUANT_NONE, &d) == VV_OK && d)
        test_kernels("none", &d->layers[1].mlp.down_proj, src, HS, INTER);
    vv_model_free(d);
}

int main(void) {
    printf("=== test_model_load ===\n");
    test_published_configs();
    test_config_refusals();
    test_family_logic();
    test_half_rne();
    test_dense_load();
    test_load_quant();
    /* Leave nothing behind in the directory ctest runs from. */
    remove("test_model_load_dense/config.json");
    remove("test_model_load_dense/model.safetensors");
    RMDIR("test_model_load_dense");
    for (int i = 0; i < g_n; i++) free(g_t[i].data);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "OK", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
