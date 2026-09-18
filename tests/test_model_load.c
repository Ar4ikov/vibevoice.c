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

#include <math.h>
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
       LAYERS = 2, VAE_A = 4, VAE_S = 8 };

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
    len += (size_t)snprintf(hdr + len, cap - len, "{");
    bool first = true;
    for (int i = 0; i < g_n; i++) {
        const fake_t* t = &g_t[i];
        if (skip && strcmp(t->name, skip) == 0) continue;
        int64_t d0 = t->shape[0];
        if (bad && strcmp(t->name, bad) == 0) d0 /= 2;   /* bytes stay */
        if (t->ndim == 2)
            len += (size_t)snprintf(hdr + len, cap - len,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld,%lld],"
                "\"data_offsets\":[%zu,%zu]}", first ? "" : ",", t->name,
                t->dtype, (long long)d0, (long long)t->shape[1],
                off, off + t->bytes);
        else
            len += (size_t)snprintf(hdr + len, cap - len,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],"
                "\"data_offsets\":[%zu,%zu]}", first ? "" : ",", t->name,
                t->dtype, (long long)d0, off, off + t->bytes);
        off += t->bytes;
        first = false;
    }
    len += (size_t)snprintf(hdr + len, cap - len, "}");
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
                vv_float_to_half(((float*)es->data)[i]);
    CHECK(exact, "F32 embedding becomes FP16, rounded to nearest-even");

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

/** @brief Largest |dequant - source| relative to the row's largest |w|. */
static double quant_error(const vv_weight_t* w, const fake_t* src, int N, int K) {
    double worst = 0.0;
    extern const float VV_NF4_TABLE[16];
    for (int n = 0; n < N; n++) {
        double amax = 0.0;
        for (int k = 0; k < K; k++) {
            double v = fabs(vv_bf16_to_float(((uint16_t*)src->data)[n * K + k]));
            if (v > amax) amax = v;
        }
        for (int k = 0; k < K; k++) {
            const uint8_t b = ((const uint8_t*)w->tensor.data)[(n * K + k) / 2];
            const int code = (k & 1) ? (b & 15) : (b >> 4);
            double d;
            if (w->quant_kind == VV_QUANT_NF4) {
                const uint16_t sc = ((const uint16_t*)w->quant.scales.data)[(n * K + k) / 64];
                d = VV_NF4_TABLE[code] * vv_half_to_float(sc);
            } else {
                const int g = k / w->group_size, G = K / w->group_size;
                d = code * vv_half_to_float(((const uint16_t*)w->quant.scales.data)[n * G + g])
                    + vv_half_to_float(((const uint16_t*)w->mins.data)[n * G + g]);
            }
            const double e = fabs(d - vv_bf16_to_float(((uint16_t*)src->data)[n * K + k]));
            if (amax > 0 && e / amax > worst) worst = e / amax;
        }
    }
    return worst;
}

static bool same_bytes(const vv_tensor_t* a, const vv_tensor_t* b) {
    return a->size_bytes == b->size_bytes &&
           (a->size_bytes == 0 || memcmp(a->data, b->data, a->size_bytes) == 0);
}

static void test_load_quant(void) {
    printf("load-time quantization:\n");
    const char* dir = "test_model_load_dense";
    const fake_t* src = get("model.language_model.layers.1.mlp.down_proj.weight");
    for (int qi = 0; qi < 2; qi++) {
        const int q = qi ? VV_LOAD_QUANT_INT4 : VV_LOAD_QUANT_NF4;
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
        omp_set_num_threads(max_threads);
#endif
        snprintf(msg, sizeof(msg), "--quant %s loads", nm);
        CHECK(sa == VV_OK && sb == VV_OK && a && b, msg);
        if (!a || !b) { vv_model_free(a); vv_model_free(b); continue; }

        const vv_weight_t* w = &a->layers[1].mlp.down_proj;
        snprintf(msg, sizeof(msg), "%s: kind, packed [N][K/2], scales per group", nm);
        CHECK(w->quant_kind == (qi ? VV_QUANT_INT4G : VV_QUANT_NF4) &&
              w->tensor.size_bytes == (size_t)HS * INTER / 2 &&
              w->quant.scales.size_bytes ==
                  (size_t)HS * (INTER / (qi ? 128 : 64)) * 2 &&
              (qi ? w->group_size == 128 && w->mins.data != NULL
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

        const double err = quant_error(w, src, HS, INTER);
        snprintf(msg, sizeof(msg), "%s: worst error %.3f of the row's max",
                 nm, err);
        CHECK(err < (qi ? 0.08 : 0.2), msg);
        vv_model_free(a);
        vv_model_free(b);
    }
}

int main(void) {
    printf("=== test_model_load ===\n");
    test_published_configs();
    test_config_refusals();
    test_family_logic();
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
