/**
 * @file loader_gguf.c
 * @brief asr-bitnet from the GGUF pair VibeASR.cpp ships, and the pieces of
 *        the BitNet loader both sources share.
 *
 * The LM file holds the 196 ternary projections as I2_S (the runtime's own
 * ternary layout, so the codes are copied as they are), the embedding as
 * Q6_K, a separate F16 head and FP32 norms and q/k/v biases. The VAE file
 * holds every encoder and connector weight as I8_S with FP32 biases, norms
 * and layer scales, conv kernels padded at the front to SIMD lengths.
 *
 * Nothing is left pointing into the mappings: every tensor is copied into
 * memory the model owns and both files are closed again, so the rest of the
 * runtime (uploads free host copies, vv_model_free frees everything) needs
 * no notion of borrowed tensors.
 */

#include "vibevoice/model.h"
#include "vibevoice/gguf.h"
#include "vibevoice/bitnet.h"
#include "vibevoice/vae_i8.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
#include "bitnet_load.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define BITNET_LM_GGUF  "vibeasr-lm-i2_s-embed-q6_k.gguf"
#define BITNET_VAE_GGUF "vibeasr-vae-encoder-i8_s.gguf"

static void join(char* buf, size_t n, const char* dir, const char* name) {
    const size_t l = strlen(dir);
    if (l > 0 && (dir[l - 1] == '/' || dir[l - 1] == '\\'))
        snprintf(buf, n, "%s%s", dir, name);
    else
        snprintf(buf, n, "%s/%s", dir, name);
}

static bool file_exists(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

bool vv_model_has_bitnet_gguf(const char* model_dir) {
    if (!model_dir) return false;
    char a[1024], b[1024];
    join(a, sizeof(a), model_dir, BITNET_LM_GGUF);
    join(b, sizeof(b), model_dir, BITNET_VAE_GGUF);
    return file_exists(a) && file_exists(b);
}

static void set_t(vv_tensor_t* t, void* data, vv_dtype_t dt, size_t bytes,
                  int nd, int64_t d0, int64_t d1, int64_t d2) {
    memset(t, 0, sizeof(*t));
    t->data = data;
    t->dtype = dt;
    t->size_bytes = bytes;
    t->ndim = nd;
    t->shape[0] = d0;
    if (nd > 1) t->shape[1] = d1;
    if (nd > 2) t->shape[2] = d2;
}

/** A tensor of the given type whose ne matches (0 = any, -1 = absent dim). */
static const vv_gguf_tensor_t* need(const vv_gguf_t* g, uint32_t type,
                                    int64_t ne0, int64_t ne1, int64_t ne2,
                                    const char* fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    const vv_gguf_tensor_t* t = vv_gguf_find_tensor(g, name);
    if (!t) {
        VV_LOG_E("loader: '%s' is missing from the GGUF", name);
        return NULL;
    }
    if (t->type != type) {
        VV_LOG_E("loader: '%s' is %s, expected %s", name,
                 vv_ggml_type_name(t->type), vv_ggml_type_name(type));
        return NULL;
    }
    const int64_t want[3] = { ne0, ne1, ne2 };
    for (int d = 0; d < 3; d++) {
        if (want[d] == 0) continue;
        const int64_t got = d < t->n_dims ? t->ne[d] : 1;
        if (want[d] > 0 && got != want[d]) {
            VV_LOG_E("loader: '%s' has ne[%d] = %lld, the config says %lld",
                     name, d, (long long)got, (long long)want[d]);
            return NULL;
        }
    }
    return t;
}

/* ─── Shared helpers ────────────────────────────────────────────────────── */

float* vv_i8vae_copy_f32(vv_i8vae_t* v, const float* src, size_t n) {
    float* d = (float*)vv_i8vae_own(v, n * sizeof(float));
    if (d) memcpy(d, src, n * sizeof(float));
    return d;
}

vv_status_t vv_i8vae_tower_alloc(vv_i8vae_t* v, vv_i8_tower_t* tw,
                                 int n_stages, const int* depth) {
    if (n_stages < 1 || n_stages > VV_I8VAE_MAX_STAGES) return VV_ERR_MODEL_FORMAT;
    tw->n_stages = n_stages;
    for (int i = 0; i < n_stages; i++) {
        tw->depth[i] = depth[i];
        tw->blocks[i] = (vv_i8_block_t*)vv_i8vae_own(
            v, sizeof(vv_i8_block_t) * (size_t)(depth[i] > 0 ? depth[i] : 1));
        if (!tw->blocks[i]) return VV_ERR_OUT_OF_MEMORY;
        memset(tw->blocks[i], 0, sizeof(vv_i8_block_t) * (size_t)(depth[i] > 0 ? depth[i] : 1));
    }
    return VV_OK;
}

vv_status_t vv_i8vae_layer_from_q(vv_i8vae_t* v, const int8_t* q, float scale,
                                  int out, int in, int kp, int k, int stride,
                                  int depthwise, const float* bias,
                                  vv_i8_layer_t* L) {
    if (kp < k || k < 1) return VV_ERR_SHAPE_MISMATCH;
    const int z = kp - k;
    const int rows = depthwise ? out : out * in;
    for (int r = 0; r < rows; r++)
        for (int j = 0; j < z; j++)
            if (q[(size_t)r * kp + j] != 0) {
                VV_LOG_E("loader: an encoder kernel has non-zero taps in its "
                         "front padding");
                return VV_ERR_MODEL_FORMAT;
            }
    memset(L, 0, sizeof(*L));
    L->in_ch = depthwise ? out : in;
    L->out_ch = out;
    L->k = k;
    L->stride = stride;
    L->depthwise = depthwise;
    L->w_scale = scale;
    const size_t n = depthwise ? (size_t)k * out : (size_t)out * k * in;
    L->w = (int8_t*)vv_i8vae_own(v, n);
    L->bias = vv_i8vae_copy_f32(v, bias, (size_t)out);
    if (!L->w || !L->bias) return VV_ERR_OUT_OF_MEMORY;
    if (depthwise) {
        /* [C][kp] -> [k][C] */
        for (int c = 0; c < out; c++)
            for (int j = 0; j < k; j++)
                L->w[(size_t)j * out + c] = q[(size_t)c * kp + z + j];
    } else {
        /* [out][in][kp] -> [out][k][in] */
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                for (int j = 0; j < k; j++)
                    L->w[((size_t)o * k + j) * in + i] =
                        q[((size_t)o * in + i) * kp + z + j];
    }
    return VV_OK;
}

int vv_bitnet_vae_stride(const vv_model_config_t* c, bool acoustic, int ds) {
    if (ds <= 0) return 1;
    const int n = acoustic ? c->acoustic.n_ratios : c->semantic.n_ratios;
    const int* r = acoustic ? c->acoustic.encoder_ratios : c->semantic.encoder_ratios;
    return (ds <= n) ? r[n - ds] : 0;
}

int vv_bitnet_vae_kernel(const vv_model_config_t* c, bool acoustic, int ds) {
    if (ds <= 0) return 7;
    return 2 * vv_bitnet_vae_stride(c, acoustic, ds);
}

vv_status_t vv_bitnet_head_i8(vv_model_t* m, const void* src, int f16,
                              int V, int K) {
    int8_t* q = (int8_t*)vv_alloc((size_t)V * K);
    float* sc = (float*)vv_alloc((size_t)V * sizeof(float));
    if (!q || !sc) { vv_free(q); vv_free(sc); return VV_ERR_OUT_OF_MEMORY; }
    enum { ROWS = 256 };
    const int nb = (V + ROWS - 1) / ROWS;
    vv_status_t err = VV_OK;
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (b = 0; b < nb; b++) {
        const int r0 = b * ROWS;
        const int nr = (V - r0) < ROWS ? (V - r0) : ROWS;
        float* tmp = (float*)vv_alloc((size_t)nr * K * sizeof(float));
        if (!tmp) { err = VV_ERR_OUT_OF_MEMORY; continue; }
        for (int64_t i = 0; i < (int64_t)nr * K; i++)
            tmp[i] = f16 ? vv_half_to_float(((const uint16_t*)src)[(int64_t)r0 * K + i])
                         : ((const float*)src)[(int64_t)r0 * K + i];
        vv_i8_rowquant_f32(tmp, nr, K, q + (size_t)r0 * K, sc + r0);
        vv_free(tmp);
    }
    if (err != VV_OK) { vv_free(q); vv_free(sc); return err; }
    set_t(&m->head_i8, q, VV_DTYPE_I8, (size_t)V * K, 2, V, K, 0);
    set_t(&m->head_i8_scale, sc, VV_DTYPE_F32, (size_t)V * sizeof(float), 1, V, 0, 0);
    return VV_OK;
}

vv_status_t vv_bitnet_head_filter_prepare(vv_model_t* m) {
    if (!m) return VV_ERR_NULL_PTR;
    if (!m->lm_head.data || m->lm_head.dtype != VV_DTYPE_F16) return VV_OK;
    const int V = m->config.llm.vocab_size, K = m->config.llm.hidden_size;
    int8_t* q = (int8_t*)vv_alloc((size_t)V * K);
    float* sc = (float*)vv_alloc((size_t)V * sizeof(float));
    float* b = (float*)vv_alloc((size_t)V * 3 * sizeof(float));
    vv_status_t s = (q && sc && b) ? VV_OK : VV_ERR_OUT_OF_MEMORY;
    if (s == VV_OK)
        s = vv_bitnet_head_filter_build((const uint16_t*)m->lm_head.data, V, K,
                                        q, sc, b);
    if (s != VV_OK) { vv_free(q); vv_free(sc); vv_free(b); return s; }
    set_t(&m->head_i8, q, VV_DTYPE_I8, (size_t)V * K, 2, V, K, 0);
    set_t(&m->head_i8_scale, sc, VV_DTYPE_F32, (size_t)V * sizeof(float), 1, V, 0, 0);
    set_t(&m->head_bound, b, VV_DTYPE_F32, (size_t)V * 3 * sizeof(float), 2, V, 3, 0);
    return VV_OK;
}

/* ─── LM ────────────────────────────────────────────────────────────────── */

static vv_status_t copy_f32(const vv_gguf_tensor_t* t, vv_tensor_t* out) {
    const int64_t n = vv_gguf_nelements(t);
    float* d = (float*)vv_alloc((size_t)n * sizeof(float));
    if (!d) return VV_ERR_OUT_OF_MEMORY;
    memcpy(d, t->data, (size_t)n * sizeof(float));
    set_t(out, d, VV_DTYPE_F32, (size_t)n * sizeof(float), 1, n, 0, 0);
    return VV_OK;
}

static vv_status_t load_ternary(const vv_gguf_t* g, int layer, const char* gname,
                                const char* hfname, int N, int K, bool bias,
                                vv_weight_t* w) {
    const vv_gguf_tensor_t* t = need(g, VV_GGML_I2_S, K, N, 0, "blk.%d.%s.weight",
                                     layer, gname);
    if (!t) return VV_ERR_WEIGHT_MISSING;
    if (K % VV_TERNARY_BLOCK) return VV_ERR_UNSUPPORTED;
    const uint8_t* codes;
    float sc;
    vv_status_t s = vv_gguf_i2s_view(t, &codes, &sc);
    if (s != VV_OK) return s;
    const size_t nb = (size_t)N * K / 4;
    uint8_t* d = (uint8_t*)vv_alloc(nb);
    if (!d) return VV_ERR_OUT_OF_MEMORY;
    memcpy(d, codes, nb);
    snprintf(w->name, sizeof(w->name), "layers.%d.%s", layer, hfname);
    set_t(&w->tensor, d, VV_DTYPE_U8, nb, 2, N, K / 4, 0);
    w->tscale = sc;
    w->quant_kind = VV_QUANT_TERNARY;
    w->is_quantized = true;
    if (bias) {
        const vv_gguf_tensor_t* b = need(g, VV_GGML_F32, N, -1, -1,
                                         "blk.%d.%s.bias", layer, gname);
        if (!b) return VV_ERR_WEIGHT_MISSING;
        s = copy_f32(b, &w->bias);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

static vv_status_t load_lm(vv_model_t* m, const vv_gguf_t* g,
                           const vv_model_load_opts_t* o) {
    const vv_llm_config_t* c = &m->config.llm;
    const int hs = c->hidden_size, V = c->vocab_size;
    const int qd = c->num_attention_heads * c->head_dim;
    const int kvd = c->num_key_value_heads * c->head_dim;
    const int inter = c->intermediate_size;
    const char* arch = vv_gguf_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, "qwen2") != 0) {
        VV_LOG_E("loader: the LM GGUF is '%s', not qwen2", arch ? arch : "?");
        return VV_ERR_MODEL_FORMAT;
    }
    const int64_t nl = vv_gguf_get_int(g, "qwen2.block_count", -1);
    if (nl != c->num_hidden_layers ||
        vv_gguf_get_int(g, "qwen2.embedding_length", -1) != hs ||
        vv_gguf_get_int(g, "qwen2.attention.head_count", -1) != c->num_attention_heads ||
        vv_gguf_get_int(g, "qwen2.attention.head_count_kv", -1) != c->num_key_value_heads ||
        vv_gguf_get_int(g, "qwen2.feed_forward_length", -1) != inter) {
        VV_LOG_E("loader: the LM GGUF's dimensions disagree with config.json");
        return VV_ERR_SHAPE_MISMATCH;
    }

    /* Embedding: kept as Q6_K, which is what the reference looks rows up in. */
    const vv_gguf_tensor_t* te = need(g, VV_GGML_Q6_K, hs, V, 0, "token_embd.weight");
    if (!te || hs % 256) return VV_ERR_WEIGHT_MISSING;
    const size_t row = (size_t)hs / 256 * 210;
    uint8_t* e = (uint8_t*)vv_alloc(row * V);
    if (!e) return VV_ERR_OUT_OF_MEMORY;
    memcpy(e, te->data, row * V);
    set_t(&m->embed_q6k, e, VV_DTYPE_U8, row * V, 2, V, (int64_t)row, 0);

    /* Head: a separate F16 table, or int8 rows of it. */
    const vv_gguf_tensor_t* th = need(g, VV_GGML_F16, hs, V, 0, "output.weight");
    if (!th) return VV_ERR_WEIGHT_MISSING;
    vv_status_t s;
    if (o->head == VV_HEAD_INT8) {
        s = vv_bitnet_head_i8(m, th->data, 1, V, hs);
        if (s != VV_OK) return s;
    } else {
        const size_t nb = (size_t)V * hs * 2;
        uint16_t* h = (uint16_t*)vv_alloc(nb);
        if (!h) return VV_ERR_OUT_OF_MEMORY;
        memcpy(h, th->data, nb);
        set_t(&m->lm_head, h, VV_DTYPE_F16, nb, 2, V, hs, 0);
    }
    m->lm_head_tied = false;

    const vv_gguf_tensor_t* tn = need(g, VV_GGML_F32, hs, -1, -1, "output_norm.weight");
    if (!tn) return VV_ERR_WEIGHT_MISSING;
    if ((s = copy_f32(tn, &m->final_norm)) != VV_OK) return s;

    for (int i = 0; i < m->num_layers; i++) {
        vv_layer_weights_t* L = &m->layers[i];
        const vv_gguf_tensor_t* a = need(g, VV_GGML_F32, hs, -1, -1, "blk.%d.attn_norm.weight", i);
        const vv_gguf_tensor_t* f = need(g, VV_GGML_F32, hs, -1, -1, "blk.%d.ffn_norm.weight", i);
        if (!a || !f) return VV_ERR_WEIGHT_MISSING;
        if ((s = copy_f32(a, &L->input_layernorm)) != VV_OK) return s;
        if ((s = copy_f32(f, &L->post_attn_layernorm)) != VV_OK) return s;
        const bool qb = c->attention_bias;
        if ((s = load_ternary(g, i, "attn_q", "self_attn.q_proj", qd, hs, qb, &L->attn.q_proj)) != VV_OK ||
            (s = load_ternary(g, i, "attn_k", "self_attn.k_proj", kvd, hs, qb, &L->attn.k_proj)) != VV_OK ||
            (s = load_ternary(g, i, "attn_v", "self_attn.v_proj", kvd, hs, qb, &L->attn.v_proj)) != VV_OK ||
            (s = load_ternary(g, i, "attn_output", "self_attn.o_proj", hs, qd, false, &L->attn.o_proj)) != VV_OK ||
            (s = load_ternary(g, i, "ffn_gate", "mlp.gate_proj", inter, hs, false, &L->mlp.gate_proj)) != VV_OK ||
            (s = load_ternary(g, i, "ffn_up", "mlp.up_proj", inter, hs, false, &L->mlp.up_proj)) != VV_OK ||
            (s = load_ternary(g, i, "ffn_down", "mlp.down_proj", hs, inter, false, &L->mlp.down_proj)) != VV_OK)
            return s;
    }
    return VV_OK;
}

/* ─── VAE ───────────────────────────────────────────────────────────────── */

static const float* f32_of(const vv_gguf_t* g, int64_t n, const char* fmt,
                           const char* a, int i, int j) {
    const vv_gguf_tensor_t* t = need(g, VV_GGML_F32, 0, 0, 0, fmt, a, i, j);
    if (!t) return NULL;
    if (vv_gguf_nelements(t) != n) {
        VV_LOG_E("loader: a VAE vector has %lld elements, expected %lld",
                 (long long)vv_gguf_nelements(t), (long long)n);
        return NULL;
    }
    return (const float*)t->data;
}

/* One I8_S weight + its bias as an int8 layer. */
static vv_status_t i8_from_gguf(vv_i8vae_t* v, const vv_gguf_t* g,
                                const char* wname, const char* bname,
                                int k, int stride, int depthwise,
                                vv_i8_layer_t* L) {
    const vv_gguf_tensor_t* t = vv_gguf_find_tensor(g, wname);
    const vv_gguf_tensor_t* b = vv_gguf_find_tensor(g, bname);
    if (!t || !b || t->type != VV_GGML_I8_S || b->type != VV_GGML_F32) {
        VV_LOG_E("loader: '%s' / '%s' missing or not I8_S + F32", wname, bname);
        return VV_ERR_WEIGHT_MISSING;
    }
    int kp, in, out;
    if (t->n_dims == 3) { kp = (int)t->ne[0]; in = (int)t->ne[1]; out = (int)t->ne[2]; }
    else if (t->n_dims == 2) { kp = 1; in = (int)t->ne[0]; out = (int)t->ne[1]; }
    else return VV_ERR_MODEL_FORMAT;
    if (depthwise && in != 1) return VV_ERR_SHAPE_MISMATCH;
    if (vv_gguf_nelements(b) != out) return VV_ERR_SHAPE_MISMATCH;
    const int8_t* q;
    float sc;
    vv_status_t s = vv_gguf_i8s_view(t, &q, &sc);
    if (s != VV_OK) return s;
    return vv_i8vae_layer_from_q(v, q, sc, out, in, kp, k > 0 ? k : kp, stride,
                                 depthwise, (const float*)b->data, L);
}

static vv_status_t vae_i8_tower(vv_i8vae_t* v, const vv_gguf_t* g,
                                const vv_model_config_t* c, bool ac,
                                vv_i8_tower_t* tw) {
    const char* p = ac ? "acoustic" : "semantic";
    const int* depth = ac ? c->acoustic.encoder_depths : c->semantic.encoder_depths;
    const int ns = ac ? c->acoustic.n_depths : c->semantic.n_depths;
    vv_status_t s = vv_i8vae_tower_alloc(v, tw, ns, depth);
    if (s != VV_OK) return s;
    char wn[256], bn[256];
    for (int i = 0; i < ns; i++) {
        snprintf(wn, sizeof(wn), "%s.downsample_layers.%d.0.conv.conv.weight", p, i);
        snprintf(bn, sizeof(bn), "%s.downsample_layers.%d.0.conv.conv.bias", p, i);
        s = i8_from_gguf(v, g, wn, bn, vv_bitnet_vae_kernel(c, ac, i),
                         vv_bitnet_vae_stride(c, ac, i), 0, &tw->ds[i]);
        if (s != VV_OK) return s;
        const int C = tw->ds[i].out_ch;
        for (int b = 0; b < depth[i]; b++) {
            vv_i8_block_t* B = &tw->blocks[i][b];
            snprintf(wn, sizeof(wn), "%s.stages.%d.%d.mixer.conv.conv.conv.weight", p, i, b);
            snprintf(bn, sizeof(bn), "%s.stages.%d.%d.mixer.conv.conv.conv.bias", p, i, b);
            if ((s = i8_from_gguf(v, g, wn, bn, 7, 1, 1, &B->mixer)) != VV_OK) return s;
            snprintf(wn, sizeof(wn), "%s.stages.%d.%d.ffn.linear1.weight", p, i, b);
            snprintf(bn, sizeof(bn), "%s.stages.%d.%d.ffn.linear1.bias", p, i, b);
            if ((s = i8_from_gguf(v, g, wn, bn, 1, 1, 0, &B->fc1)) != VV_OK) return s;
            snprintf(wn, sizeof(wn), "%s.stages.%d.%d.ffn.linear2.weight", p, i, b);
            snprintf(bn, sizeof(bn), "%s.stages.%d.%d.ffn.linear2.bias", p, i, b);
            if ((s = i8_from_gguf(v, g, wn, bn, 1, 1, 0, &B->fc2)) != VV_OK) return s;
            const float *n1 = f32_of(g, C, "%s.stages.%d.%d.norm.weight", p, i, b),
                        *g1 = f32_of(g, C, "%s.stages.%d.%d.gamma", p, i, b),
                        *n2 = f32_of(g, C, "%s.stages.%d.%d.ffn_norm.weight", p, i, b),
                        *g2 = f32_of(g, C, "%s.stages.%d.%d.ffn_gamma", p, i, b);
            if (!n1 || !g1 || !n2 || !g2) return VV_ERR_WEIGHT_MISSING;
            B->norm = vv_i8vae_copy_f32(v, n1, C);
            B->gamma = vv_i8vae_copy_f32(v, g1, C);
            B->ffn_norm = vv_i8vae_copy_f32(v, n2, C);
            B->ffn_gamma = vv_i8vae_copy_f32(v, g2, C);
            if (!B->norm || !B->gamma || !B->ffn_norm || !B->ffn_gamma)
                return VV_ERR_OUT_OF_MEMORY;
        }
    }
    snprintf(wn, sizeof(wn), "%s.head.conv.conv.weight", p);
    snprintf(bn, sizeof(bn), "%s.head.conv.conv.bias", p);
    if ((s = i8_from_gguf(v, g, wn, bn, 7, 1, 0, &tw->head)) != VV_OK) return s;
    snprintf(wn, sizeof(wn), "%s_connector.fc1.weight", p);
    snprintf(bn, sizeof(bn), "%s_connector.fc1.bias", p);
    if ((s = i8_from_gguf(v, g, wn, bn, 1, 1, 0, &tw->cfc1)) != VV_OK) return s;
    snprintf(wn, sizeof(wn), "%s_connector.fc2.weight", p);
    snprintf(bn, sizeof(bn), "%s_connector.fc2.bias", p);
    if ((s = i8_from_gguf(v, g, wn, bn, 1, 1, 0, &tw->cfc2)) != VV_OK) return s;
    const vv_gguf_tensor_t* cn = need(g, VV_GGML_F32, tw->cfc1.out_ch, -1, -1,
                                      "%s_connector.norm.weight", p);
    if (!cn) return VV_ERR_WEIGHT_MISSING;
    tw->cnorm = vv_i8vae_copy_f32(v, (const float*)cn->data,
                                  (size_t)tw->cfc1.out_ch);
    return tw->cnorm ? VV_OK : VV_ERR_OUT_OF_MEMORY;
}

/*
 * The float encoder on the same weights: every I8_S tensor dequantized
 * (w = q * scale) under its safetensors name, conv kernels without the
 * converter's front padding, so vv_conv_vae_init binds them as usual.
 */
static vv_status_t vae_float(vv_model_t* m, const vv_gguf_t* g, bool ac) {
    const char* p = ac ? "acoustic" : "semantic";
    const size_t pl = strlen(p);
    int n = 0;
    for (int i = 0; i < vv_gguf_n_tensors(g); i++) {
        const char* nm = vv_gguf_tensor(g, i)->name;
        if (strncmp(nm, p, pl) == 0 && nm[pl] == '.') n++;
    }
    vv_weight_t* w = (vv_weight_t*)vv_alloc(sizeof(vv_weight_t) * (size_t)(n ? n : 1));
    if (!w) return VV_ERR_OUT_OF_MEMORY;
    memset(w, 0, sizeof(vv_weight_t) * (size_t)(n ? n : 1));
    if (ac) { m->acoustic_weights = w; m->n_acoustic_weights = 0; }
    else    { m->semantic_weights = w; m->n_semantic_weights = 0; }
    int k = 0;
    for (int i = 0; i < vv_gguf_n_tensors(g) && k < n; i++) {
        const vv_gguf_tensor_t* t = vv_gguf_tensor(g, i);
        if (strncmp(t->name, p, pl) != 0 || t->name[pl] != '.') continue;
        const char* sfx = t->name + pl + 1;
        vv_weight_t* o = &w[k];
        snprintf(o->name, sizeof(o->name), "model.%s_tokenizer.encoder.%s", p, sfx);
        const int64_t ne = vv_gguf_nelements(t);
        if (t->type == VV_GGML_F32) {
            float* d = (float*)vv_alloc((size_t)ne * sizeof(float));
            if (!d) return VV_ERR_OUT_OF_MEMORY;
            memcpy(d, t->data, (size_t)ne * sizeof(float));
            set_t(&o->tensor, d, VV_DTYPE_F32, (size_t)ne * sizeof(float), 1, ne, 0, 0);
        } else if (t->type == VV_GGML_I8_S) {
            const int8_t* q;
            float sc;
            vv_gguf_i8s_view(t, &q, &sc);
            int kp = 1, in, out, kk = 1;
            if (t->n_dims == 3) {
                kp = (int)t->ne[0]; in = (int)t->ne[1]; out = (int)t->ne[2];
                int ds = -1, st, bl;
                if (sscanf(sfx, "downsample_layers.%d.", &ds) == 1)
                    kk = vv_bitnet_vae_kernel(&m->config, ac, ds);
                else if (sscanf(sfx, "stages.%d.%d.", &st, &bl) == 2 ||
                         strncmp(sfx, "head.", 5) == 0)
                    kk = 7;
                else kk = kp;
            } else if (t->n_dims == 2) {
                in = (int)t->ne[0]; out = (int)t->ne[1];
            } else return VV_ERR_MODEL_FORMAT;
            if (kk > kp) return VV_ERR_SHAPE_MISMATCH;
            const int z = kp - kk;
            const size_t cnt = (size_t)out * in * kk;
            float* d = (float*)vv_alloc(cnt * sizeof(float));
            if (!d) return VV_ERR_OUT_OF_MEMORY;
            for (size_t r = 0; r < (size_t)out * in; r++) {
                for (int j = 0; j < z; j++)
                    if (q[r * kp + j] != 0) { vv_free(d); return VV_ERR_MODEL_FORMAT; }
                for (int j = 0; j < kk; j++)
                    d[r * kk + j] = (float)q[r * kp + z + j] * sc;
            }
            if (t->n_dims == 3)
                set_t(&o->tensor, d, VV_DTYPE_F32, cnt * sizeof(float), 3, out, in, kk);
            else
                set_t(&o->tensor, d, VV_DTYPE_F32, cnt * sizeof(float), 2, out, in, 0);
        } else {
            return VV_ERR_MODEL_FORMAT;
        }
        k++;
        if (ac) m->n_acoustic_weights = k; else m->n_semantic_weights = k;
    }

    /* The connector, into the model's connector slots (biases in .quant.packed). */
    struct { const char* part; vv_weight_t* wt; bool bias; } cp[3] = {
        { "fc1", ac ? &m->acoustic_connector_fc1 : &m->semantic_connector_fc1, true },
        { "norm", ac ? &m->acoustic_connector_norm : &m->semantic_connector_norm, false },
        { "fc2", ac ? &m->acoustic_connector_fc2 : &m->semantic_connector_fc2, true },
    };
    for (int i = 0; i < 3; i++) {
        char nm[128];
        snprintf(nm, sizeof(nm), "%s_connector.%s.weight", p, cp[i].part);
        const vv_gguf_tensor_t* t = vv_gguf_find_tensor(g, nm);
        if (!t) { VV_LOG_E("loader: '%s' missing", nm); return VV_ERR_WEIGHT_MISSING; }
        const int64_t ne = vv_gguf_nelements(t);
        float* d = (float*)vv_alloc((size_t)ne * sizeof(float));
        if (!d) return VV_ERR_OUT_OF_MEMORY;
        if (vv_gguf_dequant_rows_f32(t, 0, ne / t->ne[0], d) != VV_OK) {
            vv_free(d);
            return VV_ERR_MODEL_FORMAT;
        }
        if (t->n_dims == 2)
            set_t(&cp[i].wt->tensor, d, VV_DTYPE_F32, (size_t)ne * 4, 2, t->ne[1], t->ne[0], 0);
        else
            set_t(&cp[i].wt->tensor, d, VV_DTYPE_F32, (size_t)ne * 4, 1, ne, 0, 0);
        snprintf(cp[i].wt->name, sizeof(cp[i].wt->name), "model.%s", nm);
        if (cp[i].bias) {
            snprintf(nm, sizeof(nm), "%s_connector.%s.bias", p, cp[i].part);
            const vv_gguf_tensor_t* b = vv_gguf_find_tensor(g, nm);
            if (!b || b->type != VV_GGML_F32) return VV_ERR_WEIGHT_MISSING;
            vv_status_t s = copy_f32(b, &cp[i].wt->quant.packed);
            if (s != VV_OK) return s;
        }
    }
    return VV_OK;
}

/* ─── Entry points ──────────────────────────────────────────────────────── */

vv_status_t vv_model_load_bitnet_gguf(vv_model_t* m, const char* model_dir,
                                      const vv_model_load_opts_t* o) {
    if (!m || !model_dir || !o) return VV_ERR_NULL_PTR;
    char lp[1024], vp[1024];
    join(lp, sizeof(lp), model_dir, BITNET_LM_GGUF);
    join(vp, sizeof(vp), model_dir, BITNET_VAE_GGUF);
    vv_gguf_t *lm = NULL, *vae = NULL;
    vv_status_t s = vv_gguf_open(lp, &lm);
    if (s != VV_OK) { VV_LOG_E("loader: cannot open '%s'", lp); return s; }
    s = vv_gguf_open(vp, &vae);
    if (s != VV_OK) {
        VV_LOG_E("loader: cannot open '%s'", vp);
        vv_gguf_close(lm);
        return s;
    }
    const char* va = vv_gguf_get_str(vae, "general.architecture");
    if (!va || strcmp(va, "vibeasr-vae") != 0) {
        VV_LOG_E("loader: '%s' is not a vibeasr-vae GGUF", vp);
        s = VV_ERR_MODEL_FORMAT;
    }

    if (s == VV_OK) s = load_lm(m, lm, o);
    if (s == VV_OK) {
        if (o->vae == VV_VAE_INT8) {
            vv_i8vae_t* v = NULL;
            s = vv_i8vae_create(&v);
            if (s == VV_OK) s = vae_i8_tower(v, vae, &m->config, true, &v->tower[0]);
            if (s == VV_OK) s = vae_i8_tower(v, vae, &m->config, false, &v->tower[1]);
            if (s == VV_OK) s = vv_i8vae_prepare(v);
            if (s == VV_OK) m->i8vae = v;
            else vv_i8vae_free(v);
        } else {
            s = vae_float(m, vae, true);
            if (s == VV_OK) s = vae_float(m, vae, false);
        }
    }
    vv_gguf_close(vae);
    vv_gguf_close(lm);
    if (s == VV_OK) {
        m->weights_source = VV_SOURCE_GGUF;
        m->vae_numerics = o->vae;
    }
    return s;
}

vv_status_t vv_model_bitnet_prepare_gpu(vv_model_t* m) {
    if (!m) return VV_ERR_NULL_PTR;
    if (m->config.family != VV_FAMILY_ASR_BITNET) return VV_OK;
    const int hs = m->config.llm.hidden_size;
    vv_tensor_t* norms[2];
    for (int i = -1; i < m->num_layers; i++) {
        int nn = 0;
        if (i < 0) norms[nn++] = &m->final_norm;
        else {
            norms[nn++] = &m->layers[i].input_layernorm;
            norms[nn++] = &m->layers[i].post_attn_layernorm;
        }
        for (int k = 0; k < nn; k++) {
            vv_tensor_t* t = norms[k];
            if (!t->data || t->on_gpu || t->dtype != VV_DTYPE_F32) continue;
            const size_t n = t->size_bytes / 4;
            uint16_t* h = (uint16_t*)vv_alloc(n * 2);
            if (!h) return VV_ERR_OUT_OF_MEMORY;
            for (size_t j = 0; j < n; j++)
                h[j] = vv_float_to_half_rne(((const float*)t->data)[j]);
            vv_free(t->data);
            t->data = h;
            t->dtype = VV_DTYPE_F16;
            t->size_bytes = n * 2;
        }
    }
    if (!m->embed_tokens.data && m->lm_head.data &&
        m->lm_head.dtype == VV_DTYPE_F16) {
        /*
         * The GGUF's F16 head is the tied embedding table rounded to F16
         * (row 151648 checked against the F32 checkpoint), so the GPU looks
         * tokens up in it rather than in an FP16 copy of the Q6_K table: one
         * 467 MB table instead of two, and closer to the trained weights.
         * The CPU keeps the reference's Q6_K lookup.
         */
        m->embed_tokens = m->lm_head;
        m->lm_head_tied = true;
        vv_tensor_free(&m->embed_q6k);
    }
    if (!m->embed_tokens.data && m->embed_q6k.data) {
        const int V = m->config.llm.vocab_size;
        const size_t row = (size_t)m->embed_q6k.shape[1];
        uint16_t* e = (uint16_t*)vv_alloc((size_t)V * hs * 2);
        if (!e) return VV_ERR_OUT_OF_MEMORY;
        int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < V; r++) {
            float buf[8192];
            vv_q6k_dequant_row((const uint8_t*)m->embed_q6k.data + (size_t)r * row,
                               buf, hs);
            for (int j = 0; j < hs; j++)
                e[(size_t)r * hs + j] = vv_float_to_half_rne(buf[j]);
        }
        set_t(&m->embed_tokens, e, VV_DTYPE_F16, (size_t)V * hs * 2, 2, V, hs, 0);
    }
    return VV_OK;
}
