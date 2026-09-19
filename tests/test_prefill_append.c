/**
 * @file test_prefill_append.c
 * @brief Prefill on top of a cache that already holds tokens.
 *
 * A streaming session prefills a prompt, decodes, then prefills the next
 * chunk of audio on the same KV cache. That is only right if the second
 * prefill continues the positions — RoPE angles, cache slots and the causal
 * window — from where the cache is, which prefill used not to do (it always
 * started at position 0). The bar: prefill P, one decode step, prefill the
 * rest, gives the same hidden states as one prefill of everything.
 *
 * A tiny random dense model (2 layers, 2 heads of 128, one KV head) runs on
 * the CPU always and on the GPU when there is one. Q and K are scaled so the
 * attention is sharp: a position off by one would move it a lot.
 *
 * The CPU half also checks chunked prefill (a workspace too small for the
 * whole prompt) against one pass.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HS = 256, NH = 2, NKV = 1, HD = 128, INTER = 512, LAYERS = 2,
       SEQ = 48, SPLIT = 24 };

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

static uint32_t rng = 0x1234567u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;
}

static void fill_f16(vv_tensor_t* t, int64_t n0, int64_t n1, float scale,
                     float bias) {
    const size_t n = (size_t)n0 * (size_t)(n1 > 0 ? n1 : 1);
    uint16_t* d = (uint16_t*)malloc(n * 2);
    for (size_t i = 0; i < n; i++) d[i] = vv_float_to_half(bias + scale * frand());
    memset(t, 0, sizeof(*t));
    t->data = d;
    t->dtype = VV_DTYPE_F16;
    t->size_bytes = n * 2;
    t->ndim = n1 > 0 ? 2 : 1;
    t->shape[0] = n0;
    t->shape[1] = n1;
}

static void build_model(vv_model_t* m, vv_layer_weights_t* layers) {
    memset(m, 0, sizeof(*m));
    vv_llm_config_t* c = &m->config.llm;
    c->hidden_size = HS;
    c->num_hidden_layers = LAYERS;
    c->num_attention_heads = NH;
    c->num_key_value_heads = NKV;
    c->head_dim = HD;
    c->intermediate_size = INTER;
    c->vocab_size = 16;
    c->max_position_embeddings = 4096;
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-6f;
    m->num_layers = LAYERS;
    m->layers = layers;
    memset(layers, 0, LAYERS * sizeof(*layers));

    for (int l = 0; l < LAYERS; l++) {
        vv_layer_weights_t* L = &layers[l];
        const float s_in = 1.0f / sqrtf((float)HS);
        fill_f16(&L->input_layernorm, HS, 0, 0.1f, 1.0f);
        fill_f16(&L->post_attn_layernorm, HS, 0, 0.1f, 1.0f);
        fill_f16(&L->attn.q_proj.tensor, NH * HD, HS, 6.0f * s_in, 0.0f);
        fill_f16(&L->attn.k_proj.tensor, NKV * HD, HS, 6.0f * s_in, 0.0f);
        fill_f16(&L->attn.v_proj.tensor, NKV * HD, HS, 1.5f * s_in, 0.0f);
        fill_f16(&L->attn.q_proj.bias, NH * HD, 0, 0.1f, 0.0f);
        fill_f16(&L->attn.k_proj.bias, NKV * HD, 0, 0.1f, 0.0f);
        fill_f16(&L->attn.v_proj.bias, NKV * HD, 0, 0.1f, 0.0f);
        fill_f16(&L->attn.o_proj.tensor, HS, NH * HD, 1.5f * s_in, 0.0f);
        fill_f16(&L->mlp.gate_proj.tensor, INTER, HS, 1.5f * s_in, 0.0f);
        fill_f16(&L->mlp.up_proj.tensor, INTER, HS, 1.5f * s_in, 0.0f);
        fill_f16(&L->mlp.down_proj.tensor, HS, INTER,
                 1.5f / sqrtf((float)INTER), 0.0f);
    }
}

static void free_model(vv_model_t* m) {
    for (int l = 0; l < m->num_layers; l++) {
        vv_tensor_t* t[VV_LAYER_TENSOR_SLOTS];
        const int n = vv_layer_tensors(&m->layers[l], t);
        for (int k = 0; k < n; k++)
            if (t[k]->data) {
                if (t[k]->on_gpu) vv_dev_free(t[k]->data);
                else free(t[k]->data);
                t[k]->data = NULL;
            }
    }
}

/** @brief Cosine and worst absolute difference of two float vectors. */
static void compare(const float* a, const float* b, size_t n,
                    double* cos_out, double* maxabs_out, double* scale_out) {
    double dot = 0, na = 0, nb = 0, worst = 0, peak = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        const double d = fabs((double)a[i] - b[i]);
        if (d > worst) worst = d;
        if (fabs(a[i]) > peak) peak = fabs(a[i]);
    }
    *cos_out = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    *maxabs_out = worst;
    *scale_out = peak;
}

/* ─── CPU ───────────────────────────────────────────────────────────────── */

static void test_cpu(vv_model_t* m, const float* x) {
    printf("CPU:\n");
    const size_t row = HS;
    const size_t ws_big = (size_t)64 << 20;
    float* ws = (float*)malloc(ws_big);
    float* a = (float*)malloc(SEQ * row * sizeof(float));
    float* b = (float*)malloc(SEQ * row * sizeof(float));
    float* c = (float*)malloc(SEQ * row * sizeof(float));
    memcpy(a, x, SEQ * row * sizeof(float));
    memcpy(b, x, SEQ * row * sizeof(float));
    memcpy(c, x, SEQ * row * sizeof(float));

    vv_kv_cache_t *ka = NULL, *kb = NULL, *kc = NULL;
    vv_kv_cache_create(&ka, LAYERS, NKV, HD, 256, VV_KV_FP32, true);
    vv_kv_cache_create(&kb, LAYERS, NKV, HD, 256, VV_KV_FP32, true);
    vv_kv_cache_create(&kc, LAYERS, NKV, HD, 256, VV_KV_FP32, true);

    vv_status_t s = vv_decoder_prefill_cpu(m, a, SEQ, ka, ws, ws_big);
    check(s == VV_OK && ka->current_len == SEQ, "one-shot prefill");

    s = vv_decoder_prefill_cpu(m, b, SPLIT, kb, ws, ws_big);
    check(s == VV_OK && kb->current_len == SPLIT, "prefill the first part");
    s = vv_decoder_step_cpu(m, b + SPLIT * row, kb, ws, ws_big);
    check(s == VV_OK && kb->current_len == SPLIT + 1, "one decode step");
    s = vv_decoder_prefill_cpu(m, b + (SPLIT + 1) * row, SEQ - SPLIT - 1, kb,
                               ws, ws_big);
    check(s == VV_OK && kb->current_len == SEQ,
          "prefill the rest on top of the cache");

    double cs, mx, sc;
    compare(a + SPLIT * row, b + SPLIT * row, (SEQ - SPLIT) * row, &cs, &mx, &sc);
    printf("  prefill-after-decode vs one shot: cos %.8f, max |d| %.2e "
           "(peak %.2f)\n", cs, mx, sc);
    check(cs > 0.999999 && mx < 1e-3 * sc, "CPU: same hidden states");

    /* Room for 10 tokens: five chunks instead of one. */
    const size_t per_tok = sizeof(float) *
        (3 * HS + NH * HD + 2 * NKV * HD + 2 * INTER);
    s = vv_decoder_prefill_cpu(m, c, SEQ, kc, ws, per_tok * 10);
    compare(a, c, SEQ * row, &cs, &mx, &sc);
    printf("  chunked (10 tokens a chunk) vs one shot: cos %.8f, max |d| %.2e\n",
           cs, mx);
    check(s == VV_OK && kc->current_len == SEQ && cs > 0.999999 &&
          mx < 1e-3 * sc, "CPU: chunked prefill == one pass");

    /* The window is a hard limit, checked before anything is written. */
    s = vv_decoder_prefill_cpu(m, c, 250, kc, ws, ws_big);
    check(s == VV_ERR_OVERFLOW && kc->current_len == SEQ,
          "prefill past the window refused, cache untouched");

    vv_kv_cache_free(ka); vv_kv_cache_free(kb); vv_kv_cache_free(kc);
    free(ws); free(a); free(b); free(c);
}

/* ─── GPU ───────────────────────────────────────────────────────────────── */

#ifdef VV_HAS_ACCEL
static int upload(vv_model_t* g) {
    for (int l = 0; l < g->num_layers; l++) {
        vv_tensor_t* t[VV_LAYER_TENSOR_SLOTS];
        const int n = vv_layer_tensors(&g->layers[l], t);
        for (int k = 0; k < n; k++) {
            if (!t[k]->data) continue;
            void* d = NULL;
            if (vv_dev_alloc(&d, t[k]->size_bytes) != VV_OK) return 1;
            vv_dev_memcpy_h2d(d, t[k]->data, t[k]->size_bytes, NULL);
            t[k]->data = d;           /* host copy stays with the CPU model */
            t[k]->on_gpu = true;
        }
    }
    vv_dev_stream_sync(NULL);
    return 0;
}

static void to_float(const uint16_t* h, float* f, size_t n) {
    for (size_t i = 0; i < n; i++) f[i] = vv_half_to_float(h[i]);
}

static void test_gpu(const vv_model_t* cpu_model, const float* x) {
    printf("GPU:\n");
    size_t total = 0;
    if (vv_dev_get_device_info(0, &total, NULL, NULL) != VV_OK || total == 0) {
        printf("  SKIP: no device available\n");
        return;
    }
    vv_layer_weights_t layers[LAYERS];
    memcpy(layers, cpu_model->layers, sizeof(layers));
    vv_model_t g = *cpu_model;
    g.layers = layers;
    if (upload(&g)) { printf("  SKIP: allocation\n"); return; }

    void* stream = NULL;
    vv_dev_stream_create(&stream);
    const size_t row_b = HS * 2, ws_size = (size_t)64 << 20;
    void *ws = NULL, *ha = NULL, *hb = NULL;
    vv_dev_alloc(&ws, ws_size);
    vv_dev_alloc(&ha, SEQ * row_b);
    vv_dev_alloc(&hb, SEQ * row_b);

    uint16_t* xh = (uint16_t*)malloc(SEQ * row_b);
    for (size_t i = 0; i < (size_t)SEQ * HS; i++) xh[i] = vv_float_to_half(x[i]);
    vv_dev_memcpy_h2d(ha, xh, SEQ * row_b, stream);
    vv_dev_memcpy_h2d(hb, xh, SEQ * row_b, stream);

    vv_kv_cache_t *ka = NULL, *kb = NULL;
    vv_kv_cache_create(&ka, LAYERS, NKV, HD, 256, VV_KV_FP16, false);
    vv_kv_cache_create(&kb, LAYERS, NKV, HD, 256, VV_KV_FP16, false);

    vv_status_t s = vv_decoder_prefill(&g, ha, SEQ, ka, NULL, ws, ws_size,
                                       stream, stream, 0, LAYERS);
    check(s == VV_OK, "one-shot prefill");

    s = vv_decoder_prefill(&g, hb, SPLIT, kb, NULL, ws, ws_size, stream,
                           stream, 0, LAYERS);
    if (s == VV_OK) s = vv_kv_cache_publish_len(kb, stream);
    if (s == VV_OK)
        s = vv_decoder_step(&g, (uint8_t*)hb + SPLIT * row_b, kb, NULL, ws,
                            ws_size, stream, stream, 0, LAYERS);
    check(s == VV_OK && kb->current_len == SPLIT + 1,
          "prefill, publish, one decode step");
    s = vv_decoder_prefill(&g, (uint8_t*)hb + (SPLIT + 1) * row_b,
                           SEQ - SPLIT - 1, kb, NULL, ws, ws_size, stream,
                           stream, 0, LAYERS);
    check(s == VV_OK && kb->current_len == SEQ,
          "prefill the rest on top of the cache");

    uint16_t* oa = (uint16_t*)malloc(SEQ * row_b);
    uint16_t* ob = (uint16_t*)malloc(SEQ * row_b);
    vv_dev_stream_sync(stream);
    vv_dev_memcpy_d2h(oa, ha, SEQ * row_b, stream);
    vv_dev_memcpy_d2h(ob, hb, SEQ * row_b, stream);
    float* fa = (float*)malloc(SEQ * HS * sizeof(float));
    float* fb = (float*)malloc(SEQ * HS * sizeof(float));
    to_float(oa, fa, (size_t)SEQ * HS);
    to_float(ob, fb, (size_t)SEQ * HS);

    double cs, mx, sc;
    compare(fa + SPLIT * HS, fb + SPLIT * HS, (SEQ - SPLIT) * (size_t)HS,
            &cs, &mx, &sc);
    printf("  prefill-after-decode vs one shot: cos %.8f, max |d| %.2e "
           "(peak %.2f)\n", cs, mx, sc);
    /* FP16 activations, and the decode row takes the GEMV path. */
    check(cs > 0.9999 && mx < 2e-2 * sc, "GPU: same hidden states");

    /* And the CPU agrees with the GPU on the one-shot result. */
    {
        vv_kv_cache_t* kc = NULL;
        vv_kv_cache_create(&kc, LAYERS, NKV, HD, 256, VV_KV_FP32, true);
        float* c = (float*)malloc(SEQ * HS * sizeof(float));
        float* w = (float*)malloc((size_t)64 << 20);
        memcpy(c, x, SEQ * HS * sizeof(float));
        vv_decoder_prefill_cpu((vv_model_t*)cpu_model, c, SEQ, kc, w,
                               (size_t)64 << 20);
        compare(c, fa, (size_t)SEQ * HS, &cs, &mx, &sc);
        printf("  GPU vs CPU one shot: cos %.8f\n", cs);
        check(cs > 0.999, "GPU and CPU agree");
        free(c); free(w);
        vv_kv_cache_free(kc);
    }

    vv_kv_cache_free(ka); vv_kv_cache_free(kb);
    vv_dev_free(ws); vv_dev_free(ha); vv_dev_free(hb);
    vv_dev_stream_destroy(stream);
    free(xh); free(oa); free(ob); free(fa); free(fb);
    for (int l = 0; l < LAYERS; l++) {
        vv_tensor_t* t[VV_LAYER_TENSOR_SLOTS];
        const int n = vv_layer_tensors(&layers[l], t);
        for (int k = 0; k < n; k++) if (t[k]->data) vv_dev_free(t[k]->data);
    }
}
#endif

int main(void) {
    printf("=== test_prefill_append ===\n");
    vv_model_t m;
    vv_layer_weights_t layers[LAYERS];
    build_model(&m, layers);

    float* x = (float*)malloc(SEQ * HS * sizeof(float));
    for (int i = 0; i < SEQ * HS; i++) x[i] = frand();

    test_cpu(&m, x);
#ifdef VV_HAS_ACCEL
    test_gpu(&m, x);
#else
    printf("GPU: SKIP (built without an accelerator backend)\n");
#endif

    free(x);
    free_model(&m);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "OK", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
