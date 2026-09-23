/**
 * @file spec.c
 * @brief Speculative decoding with a DFlash 2 drafter: loading it, the
 *        context it reads, the draft pass, and one draft-and-verify cycle.
 *
 * The drafter (docs/DFLASH.md) is a few Qwen3-style layers whose keys and
 * values are extended with the target's own hidden states at a handful of
 * its layers. Those arrive through the target's taps (inference.h): every
 * prefill of the target -- the prompt, and each verified block -- leaves the
 * tapped layers' outputs for its rows, and here they go through the
 * drafter's `fc` and `hidden_norm` and every drafter layer's k/v projection
 * into the drafter's own KV cache, position for position. A draft is then
 * one pass of B rows, [last token, MASK x (B-1)], over that cache; the
 * target checks all B rows in one pass and keeps the longest prefix it
 * agrees with, plus its own next token.
 *
 * Weights are FP16 on the device, activations FP16 except the residual
 * stream, which is FP32: a drafter trained in BF16 has activations FP16
 * cannot hold (a SwiGLU output of 2e4 feeding a 9472-wide down projection).
 * Three rescalings keep every FP16 intermediate in range without changing
 * the function: `fc` is stored divided by VV_SPEC_FC_SCALE and the RMSNorm
 * after it takes eps / VV_SPEC_FC_SCALE^2 (the target's features carry
 * activations of ~1.3e4); `up_proj` is divided by `mlp_div` and `down_proj`
 * multiplied by mlp_div / out_div, `o_proj` divided by out_div, and the two
 * sublayer outputs are added into the residual times out_div.
 *
 * With VV_DRAFT_QUANT=int4 the projections are quantized at load into INT4
 * groups of 128 (round to nearest, exact zero point) and run on the W4A16
 * kernels, a quarter of the bytes per draft pass. The drafter's numbers
 * only decide what gets proposed; the target's check decides what is kept.
 */

#include "vibevoice/spec.h"
#include "vibevoice/inference.h"
#include "vibevoice/safetensors.h"
#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/quant.h"
#include "cJSON.h"

#include "spec_internal.h"
#include "vv_thread.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VV_SPEC_FC_SCALE 64.0f

/* ─── Drafter ───────────────────────────────────────────────────────────── */

#define DRAFT_Q_GROUP 128

/* A projection [N][K]: FP16, or INT4 groups in the W4A16 GPU layout. */
typedef struct {
    void* w;              /* FP16 [N][K], NULL when packed */
    void* packed;         /* [N][K/2] */
    void* sz;             /* half2 {scale, zero} [N][K/G] */
} dlin_t;

typedef struct {
    void*  ln1;           /* [H] */
    void*  ln2;           /* [H] */
    dlin_t q, k, v, o;    /* [nh*hd][H], [nkv*hd][H] x2, [H][nh*hd] */
    void*  qn, *kn;       /* [hd] */
    dlin_t gate, up;      /* [I][H] */
    dlin_t down;          /* [H][I] */
    dlin_t aproj, mproj;  /* [2*K*G][H] */
    float* abase, *mbase; /* FP32 [2][K][H] */
} dlayer_t;

struct vv_drafter {
    vv_drafter_config_t cfg;
    dlayer_t* L;
    dlin_t fc;            /* [H][taps*H], divided by VV_SPEC_FC_SCALE */
    void*  hnorm;         /* [H] */
    void*  norm;          /* [H] */
    void*  pred, *succ;   /* [V][rank] */
    void*  hproj;         /* [rank][H] */
    void*  head;          /* [Vd][H] FP16: the target head's draft rows, or NULL */
    int32_t* vmap;        /* [Vd] their ids */
    int    Vd;            /* rows the draft head scores (V without a subset) */
    float  hnorm_eps;
    float  mlp_div;       /* up_proj stored / mlp_div                    */
    float  out_div;       /* o_proj, down_proj outputs stored / out_div   */
    int    gpu_id;
    size_t bytes;
    void** allocs;        /* every device buffer, for free */
    int    n_allocs, cap_allocs;
};

static int json_int(const cJSON* o, const char* k, int dflt) {
    const cJSON* x = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(x) ? (int)x->valuedouble : dflt;
}

static double json_num(const cJSON* o, const char* k, double dflt) {
    const cJSON* x = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(x) ? x->valuedouble : dflt;
}

static char* read_file(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = sz >= 0 ? (char*)vv_alloc((size_t)sz + 1) : NULL;
    if (b && fread(b, 1, (size_t)sz, f) != (size_t)sz) { vv_free(b); b = NULL; }
    fclose(f);
    if (b) { b[sz] = '\0'; if (n) *n = (size_t)sz; }
    return b;
}

vv_status_t vv_drafter_config_load(const char* dir, vv_drafter_config_t* c) {
    if (!dir || !c) return VV_ERR_NULL_PTR;
    memset(c, 0, sizeof(*c));
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char* txt = read_file(path, NULL);
    if (!txt) {
        VV_LOG_E("spec: cannot read %s", path);
        return VV_ERR_NOT_FOUND;
    }
    cJSON* j = cJSON_Parse(txt);
    vv_free(txt);
    if (!j) return VV_ERR_MODEL_FORMAT;
    const cJSON* dc = cJSON_GetObjectItemCaseSensitive(j, "dflash_config");
    c->hidden_size = json_int(j, "hidden_size", 0);
    c->num_layers = json_int(j, "num_hidden_layers", 0);
    c->num_heads = json_int(j, "num_attention_heads", 0);
    c->num_kv_heads = json_int(j, "num_key_value_heads", c->num_heads);
    c->head_dim = json_int(j, "head_dim",
                           c->num_heads ? c->hidden_size / c->num_heads : 0);
    c->intermediate_size = json_int(j, "intermediate_size", 0);
    c->vocab_size = json_int(j, "vocab_size", 0);
    c->num_target_layers = json_int(j, "num_target_layers", 0);
    c->rms_norm_eps = (float)json_num(j, "rms_norm_eps", 1e-6);
    {
        const cJSON* rp = cJSON_GetObjectItemCaseSensitive(j, "rope_parameters");
        c->rope_theta = (float)json_num(j, "rope_theta",
                                        rp ? json_num(rp, "rope_theta", 1e6) : 1e6);
    }
    c->sliding_window = 0;
    {
        const cJSON* lt = cJSON_GetObjectItemCaseSensitive(j, "layer_types");
        const cJSON* e;
        cJSON_ArrayForEach(e, lt)
            if (cJSON_IsString(e) && !strcmp(e->valuestring, "sliding_attention"))
                c->sliding_window = json_int(j, "sliding_window", 0);
    }
    c->block_size = dc ? json_int(dc, "block_size", json_int(j, "block_size", 0))
                       : json_int(j, "block_size", 0);
    c->mask_token_id = dc ? json_int(dc, "mask_token_id", -1) : -1;
    c->conv_kernel = dc ? json_int(dc, "conv_kernel_size", 0) : 0;
    c->conv_group = dc ? json_int(dc, "conv_group_size", 16) : 16;
    c->selector_rank = dc ? json_int(dc, "selector_rank", 0) : 0;
    c->selector_top_k = dc ? json_int(dc, "selector_top_k", 0) : 0;
    c->draft_vocab_size = dc ? json_int(dc, "draft_vocab_size", 0) : 0;
    c->weight_quant = VV_DRAFTER_INT4;
    {
        const cJSON* f16 = cJSON_GetObjectItemCaseSensitive(j, "vv_fp16");
        c->fp16_mlp_div = f16 ? (float)json_num(f16, "mlp_div", 0.0) : 0.0f;
        c->fp16_out_div = f16 ? (float)json_num(f16, "out_div", 0.0) : 0.0f;
    }
    const cJSON* tl = dc ? cJSON_GetObjectItemCaseSensitive(dc, "target_layer_ids")
                         : NULL;
    const cJSON* e;
    cJSON_ArrayForEach(e, tl) {
        if (c->n_taps == VV_TAPS_MAX) { c->n_taps++; break; }
        c->target_layer_ids[c->n_taps++] = (int)e->valuedouble;
    }
    cJSON_Delete(j);

    /* What the kernels here take; a checkpoint outside it is refused whole. */
    const char* bad = NULL;
    if (c->hidden_size <= 0 || c->num_layers <= 0 || c->num_heads <= 0 ||
        c->intermediate_size <= 0 || c->vocab_size <= 0)
        bad = "missing dimensions";
    else if (c->head_dim != 128) bad = "head_dim must be 128";
    else if (c->num_kv_heads <= 0 || c->num_heads % c->num_kv_heads)
        bad = "query heads must be a multiple of the KV heads";
    else if (c->block_size < 2 || c->block_size > 16) bad = "block_size must be 2..16";
    else if (c->mask_token_id < 0 || c->mask_token_id >= c->vocab_size)
        bad = "mask_token_id outside the vocabulary";
    else if (c->n_taps <= 0 || c->n_taps > VV_TAPS_MAX)
        bad = "target_layer_ids: 1..8 layers";
    else if (c->conv_kernel < 1 || c->conv_kernel > 4 || c->conv_group <= 0 ||
             c->hidden_size % c->conv_group)
        bad = "not a DFlash 2 drafter (conv_kernel_size / conv_group_size)";
    else if (c->selector_top_k < 1 || c->selector_top_k > 32)
        bad = "selector_top_k must be 1..32";
    else if (c->selector_rank < 32 || c->selector_rank > 1024 ||
             c->selector_rank % 32)
        bad = "selector_rank must be a multiple of 32 up to 1024";
    else if (c->sliding_window)
        bad = "sliding-window drafters are not supported yet";
    else if (c->draft_vocab_size < 0 || c->draft_vocab_size > c->vocab_size ||
             (c->draft_vocab_size > 0 && c->draft_vocab_size < c->selector_top_k))
        bad = "draft_vocab_size outside top_k..vocab_size";
    if (bad) {
        VV_LOG_E("spec: %s/config.json: %s", dir, bad);
        return VV_ERR_MODEL_FORMAT;
    }
    return VV_OK;
}

/* Device bytes of a projection with n weights in the drafter's format. */
static size_t lin_bytes(const vv_drafter_config_t* c, size_t n) {
    return c->weight_quant == VV_DRAFTER_INT4 ? n / 2 + n / DRAFT_Q_GROUP * 4
                                              : n * 2;
}

size_t vv_drafter_weight_bytes(const vv_drafter_config_t* c) {
    const size_t H = (size_t)c->hidden_size, I = (size_t)c->intermediate_size;
    const size_t qd = (size_t)c->num_heads * c->head_dim;
    const size_t kd = (size_t)c->num_kv_heads * c->head_dim;
    const size_t G = H / (size_t)c->conv_group, K = (size_t)c->conv_kernel;
    const size_t lin = lin_bytes(c, qd * H * 2 + kd * H * 2 + 3 * I * H +
                                    2 * (2 * K * G * H));
    size_t b = ((2 * H + 2 * (size_t)c->head_dim) * 2 + lin) *
               (size_t)c->num_layers +
               2 * (2 * K * H) * 4 * (size_t)c->num_layers;
    b += lin_bytes(c, (size_t)c->n_taps * H * H);
    b += (2 * H + 2 * (size_t)c->vocab_size * c->selector_rank +
          (size_t)c->selector_rank * H) * 2;
    b += (size_t)c->draft_vocab_size * (H * 2 + 4);
    return b;
}

size_t vv_spec_context_bytes(const vv_drafter_config_t* c, int max_pos) {
    const size_t H = (size_t)c->hidden_size;
    const size_t kd = (size_t)c->num_kv_heads * c->head_dim;
    const size_t kv = (size_t)c->num_layers * 2 *
                      ((size_t)max_pos + (size_t)c->block_size) * kd * 2;
    const size_t taps = (size_t)VV_SPEC_TAP_ROWS * c->n_taps * H * 2;
    const size_t ctx = (size_t)VV_SPEC_TAP_ROWS * (H + 2 * kd) * 2 * 2;
    const size_t ws = (size_t)16 * (8 * H + 2 * (size_t)c->intermediate_size +
                                    (size_t)c->num_heads * c->head_dim * 3) * 2
                    + (size_t)16 * c->vocab_size * 4;
    return kv + taps + ctx + ws + vv_attn_scratch_bytes(c->num_heads,
                                                       c->num_kv_heads,
                                                       c->head_dim)
         + ((size_t)64 << 20);
}

/* One device buffer holding `n` bytes, remembered for vv_drafter_free. */
static vv_status_t dalloc(vv_drafter_t* d, void** p, size_t n) {
    if (d->n_allocs == d->cap_allocs) {
        const int nc = d->cap_allocs ? d->cap_allocs * 2 : 64;
        void** na = (void**)vv_realloc(d->allocs, sizeof(void*) * (size_t)nc);
        if (!na) return VV_ERR_OUT_OF_MEMORY;
        d->allocs = na;
        d->cap_allocs = nc;
    }
    vv_status_t s = vv_dev_alloc(p, n);
    if (s != VV_OK) return s;
    d->allocs[d->n_allocs++] = *p;
    d->bytes += n;
    return VV_OK;
}

typedef struct {
    vv_safetensors_t* st;
    const char* prefix;   /* "" or "model." */
} wsrc_t;

/*
 * Tensor `name` of shape [d0][d1][d2] (trailing 0s: fewer dimensions) read
 * into a new host buffer, times `mul`, as FP32 or FP16.
 */
static vv_status_t read_host(const wsrc_t* ws, const char* name, int64_t d0,
                             int64_t d1, int64_t d2, bool f32, float mul,
                             void** out) {
    *out = NULL;
    char full[256];
    snprintf(full, sizeof(full), "%s%s", ws->prefix, name);
    vv_st_tensor_info_t info;
    vv_status_t s = vv_safetensors_find(ws->st, full, &info);
    if (s != VV_OK) {
        VV_LOG_E("spec: drafter tensor '%s' missing", full);
        return VV_ERR_WEIGHT_MISSING;
    }
    const int64_t want[3] = { d0, d1, d2 };
    const int nd = d2 > 0 ? 3 : (d1 > 0 ? 2 : 1);
    bool ok = info.ndim == nd;
    for (int i = 0; ok && i < nd; i++) ok = info.shape[i] == want[i];
    if (!ok) {
        VV_LOG_E("spec: drafter tensor '%s' has the wrong shape", full);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const size_t n = (size_t)d0 * (size_t)(d1 > 0 ? d1 : 1) *
                     (size_t)(d2 > 0 ? d2 : 1);
    const void* src = NULL;
    s = vv_safetensors_get_data(ws->st, &info, &src);
    if (s != VV_OK) return s;
    if (info.dtype != VV_DTYPE_BF16 && info.dtype != VV_DTYPE_F16 &&
        info.dtype != VV_DTYPE_F32) {
        VV_LOG_E("spec: drafter tensor '%s': BF16, F16 or F32 expected", full);
        return VV_ERR_MODEL_FORMAT;
    }
    void* host = vv_alloc(n * (f32 ? 4 : 2));
    if (!host) return VV_ERR_OUT_OF_MEMORY;
    for (size_t i = 0; i < n; i++) {
        float v;
        if (info.dtype == VV_DTYPE_BF16)
            v = vv_bf16_to_float(((const uint16_t*)src)[i]);
        else if (info.dtype == VV_DTYPE_F16)
            v = vv_half_to_float(((const uint16_t*)src)[i]);
        else
            v = ((const float*)src)[i];
        v *= mul;
        if (f32) ((float*)host)[i] = v;
        else ((uint16_t*)host)[i] = vv_float_to_half_rne(v);
    }
    *out = host;
    return VV_OK;
}

/* Tensor `name` as FP16 (scaled by `mul`) or FP32 onto the device. */
static vv_status_t upload(vv_drafter_t* d, const wsrc_t* ws, const char* name,
                          int64_t d0, int64_t d1, int64_t d2, bool f32,
                          float mul, void** out) {
    void* host = NULL;
    vv_status_t s = read_host(ws, name, d0, d1, d2, f32, mul, &host);
    if (s != VV_OK) return s;
    const size_t n = (size_t)d0 * (size_t)(d1 > 0 ? d1 : 1) *
                     (size_t)(d2 > 0 ? d2 : 1) * (f32 ? 4 : 2);
    s = dalloc(d, out, n);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(*out, host, n, NULL);
    vv_free(host);
    return s;
}

/*
 * A projection [N][K] (scaled by `mul`) in the drafter's format: FP16 as
 * upload() does it, or quantized here to INT4 groups and put in the GPU
 * layout the W4A16 kernels read. The FP16 value is what gets quantized, so
 * the INT4 drafter approximates the FP16 one.
 */
static vv_status_t upload_lin(vv_drafter_t* d, const wsrc_t* ws,
                              const char* name, int64_t N, int64_t K,
                              float mul, dlin_t* out) {
    memset(out, 0, sizeof(*out));
    if (d->cfg.weight_quant != VV_DRAFTER_INT4 || N % 8 || K % 64 ||
        K % DRAFT_Q_GROUP)
        return upload(d, ws, name, N, K, 0, false, mul, &out->w);
    float* f32 = NULL;
    vv_status_t s = read_host(ws, name, N, K, 0, true, mul, (void**)&f32);
    if (s != VV_OK) return s;
    const size_t n = (size_t)N * (size_t)K;
    const size_t ng = n / DRAFT_Q_GROUP;
    for (size_t i = 0; i < n; i++)
        f32[i] = vv_half_to_float(vv_float_to_half_rne(f32[i]));
    uint8_t* packed = (uint8_t*)vv_alloc(n / 2);
    uint16_t* scales = (uint16_t*)vv_alloc(ng * 2);
    uint16_t* mins = (uint16_t*)vv_alloc(ng * 2);
    uint8_t* zeros = (uint8_t*)vv_alloc(ng);
    uint16_t* sz = (uint16_t*)vv_alloc(ng * 4);
    if (!packed || !scales || !mins || !zeros || !sz) s = VV_ERR_OUT_OF_MEMORY;
    if (s == VV_OK)
        s = vv_int4g_quantize(f32, (int)N, (int)K, DRAFT_Q_GROUP, packed,
                              scales, mins, zeros);
    if (s == VV_OK)
        s = vv_int4g_to_gpu_layout(packed, scales, zeros, (int)N, (int)K,
                                   DRAFT_Q_GROUP, sz);
    if (s == VV_OK) s = dalloc(d, &out->packed, n / 2);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(out->packed, packed, n / 2, NULL);
    if (s == VV_OK) s = dalloc(d, &out->sz, ng * 4);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(out->sz, sz, ng * 4, NULL);
    vv_free(f32); vv_free(packed); vv_free(scales); vv_free(mins);
    vv_free(zeros); vv_free(sz);
    return s;
}

/*
 * The draft head: the rows of the target's LM head for the ids in the
 * `draft_vocab` tensor, gathered once. The target keeps its whole head for
 * checking; this one only ranks the drafter's own candidates, so a fifth of
 * the rows is a fifth of the bytes per draft pass.
 */
static vv_status_t load_draft_vocab(vv_drafter_t* d, const wsrc_t* ws,
                                    const vv_model_t* target) {
    const vv_drafter_config_t* c = &d->cfg;
    const int Vd = c->draft_vocab_size, V = c->vocab_size;
    const size_t H = (size_t)c->hidden_size;
    char full[256];
    snprintf(full, sizeof(full), "%sdraft_vocab", ws->prefix);
    vv_st_tensor_info_t info;
    vv_status_t s = vv_safetensors_find(ws->st, full, &info);
    if (s != VV_OK) {
        VV_LOG_E("spec: draft_vocab_size is %d but '%s' is missing", Vd, full);
        return VV_ERR_WEIGHT_MISSING;
    }
    if (info.dtype != VV_DTYPE_I32 || info.ndim != 1 || info.shape[0] != Vd) {
        VV_LOG_E("spec: '%s' must be I32 [%d]", full, Vd);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const int32_t* ids = NULL;
    s = vv_safetensors_get_data(ws->st, &info, (const void**)&ids);
    if (s != VV_OK) return s;
    for (int i = 0; i < Vd; i++)
        if (ids[i] < 0 || ids[i] >= V || (i > 0 && ids[i] <= ids[i - 1])) {
            VV_LOG_E("spec: '%s' must list ascending ids below %d", full, V);
            return VV_ERR_MODEL_FORMAT;
        }
    const vv_tensor_t* h = target->lm_head_tied ? &target->embed_tokens
                                                : &target->lm_head;
    if (!h->data || h->on_gpu || h->dtype != VV_DTYPE_F16 ||
        h->size_bytes != (size_t)V * H * 2) {
        VV_LOG_E("spec: a draft vocabulary needs the target's F16 LM head "
                 "in host memory");
        return VV_ERR_UNSUPPORTED;
    }
    uint16_t* rows = (uint16_t*)vv_alloc((size_t)Vd * H * 2);
    if (!rows) return VV_ERR_OUT_OF_MEMORY;
    for (int i = 0; i < Vd; i++)
        memcpy(rows + (size_t)i * H, (const uint16_t*)h->data + (size_t)ids[i] * H,
               H * 2);
    s = dalloc(d, &d->head, (size_t)Vd * H * 2);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(d->head, rows, (size_t)Vd * H * 2, NULL);
    vv_free(rows);
    if (s == VV_OK) s = dalloc(d, (void**)&d->vmap, (size_t)Vd * 4);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(d->vmap, ids, (size_t)Vd * 4, NULL);
    if (s == VV_OK) d->Vd = Vd;
    return s;
}

void vv_drafter_free(vv_drafter_t* d) {
    if (!d) return;
    if (d->gpu_id >= 0) vv_dev_set_device(d->gpu_id);
    for (int i = 0; i < d->n_allocs; i++) vv_dev_free(d->allocs[i]);
    vv_free(d->allocs);
    vv_free(d->L);
    vv_free(d);
}

vv_drafter_quant_t vv_drafter_quant_parse(const char* name) {
    if (!name) return VV_DRAFTER_QUANT_COUNT;
    if (!strcmp(name, "int4")) return VV_DRAFTER_INT4;
    if (!strcmp(name, "f16") || !strcmp(name, "fp16")) return VV_DRAFTER_F16;
    return VV_DRAFTER_QUANT_COUNT;
}

vv_status_t vv_drafter_load(const char* dir, const vv_model_t* target,
                            int gpu_id, int quant, vv_drafter_t** out) {
    if (!dir || !target || !out) return VV_ERR_NULL_PTR;
    if (quant < 0 || quant >= VV_DRAFTER_QUANT_COUNT) return VV_ERR_INVALID_ARG;
    *out = NULL;
    vv_drafter_t* d = (vv_drafter_t*)vv_alloc(sizeof(*d));
    if (!d) return VV_ERR_OUT_OF_MEMORY;
    memset(d, 0, sizeof(*d));
    d->gpu_id = gpu_id;
    vv_status_t s = vv_drafter_config_load(dir, &d->cfg);
    d->cfg.weight_quant = quant;
    const vv_drafter_config_t* c = &d->cfg;
    const vv_llm_config_t* t = &target->config.llm;
    if (s == VV_OK) {
        /* A drafter is trained against one target: its width, vocabulary
         * and tapped layers have to be that model's. */
        const char* bad = NULL;
        if (c->hidden_size != t->hidden_size) bad = "hidden size";
        else if (c->vocab_size != t->vocab_size) bad = "vocabulary";
        else if (c->num_target_layers && c->num_target_layers != t->num_hidden_layers)
            bad = "target layer count";
        for (int i = 0; !bad && i < c->n_taps; i++)
            if (c->target_layer_ids[i] < 0 ||
                c->target_layer_ids[i] >= t->num_hidden_layers)
                bad = "tapped layer";
        if (bad) {
            VV_LOG_E("spec: the drafter in %s was not made for this model "
                     "(%s differs)", dir, bad);
            s = VV_ERR_MODEL_FORMAT;
        }
    }
    if (s != VV_OK) { vv_free(d); return s; }
    s = vv_dev_set_device(gpu_id);
    if (s != VV_OK) { vv_free(d); return s; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    wsrc_t ws = { NULL, "" };
    s = vv_safetensors_open(path, &ws.st);
    if (s != VV_OK) {
        VV_LOG_E("spec: cannot open %s", path);
        vv_free(d);
        return s;
    }
    {
        vv_st_tensor_info_t info;
        if (vv_safetensors_find(ws.st, "fc.weight", &info) != VV_OK &&
            vv_safetensors_find(ws.st, "model.fc.weight", &info) == VV_OK)
            ws.prefix = "model.";
    }

    const int64_t H = c->hidden_size, I = c->intermediate_size;
    const int64_t qd = (int64_t)c->num_heads * c->head_dim;
    const int64_t kd = (int64_t)c->num_kv_heads * c->head_dim;
    const int64_t K = c->conv_kernel, G = H / c->conv_group;
    d->mlp_div = c->fp16_mlp_div > 0.0f ? c->fp16_mlp_div : 32.0f;
    d->out_div = c->fp16_out_div > 0.0f ? c->fp16_out_div : 64.0f;
    d->L = (dlayer_t*)vv_alloc(sizeof(dlayer_t) * (size_t)c->num_layers);
    if (!d->L) s = VV_ERR_OUT_OF_MEMORY;
    else memset(d->L, 0, sizeof(dlayer_t) * (size_t)c->num_layers);
    for (int l = 0; s == VV_OK && l < c->num_layers; l++) {
        dlayer_t* L = &d->L[l];
        char n[160];
#define W(field, suffix, a, b, cc, f32) do {                                 \
            snprintf(n, sizeof(n), "layers.%d.%s", l, suffix);                \
            if (s == VV_OK)                                                   \
                s = upload(d, &ws, n, a, b, cc, f32, 1.0f, (void**)&L->field);\
        } while (0)
#define WL(field, suffix, a, b, mul) do {                                     \
            snprintf(n, sizeof(n), "layers.%d.%s", l, suffix);                \
            if (s == VV_OK) s = upload_lin(d, &ws, n, a, b, mul, &L->field);  \
        } while (0)
        W(ln1, "input_layernorm.weight", H, 0, 0, false);
        W(ln2, "post_attention_layernorm.weight", H, 0, 0, false);
        WL(q, "self_attn.q_proj.weight", qd, H, 1.0f);
        WL(k, "self_attn.k_proj.weight", kd, H, 1.0f);
        WL(v, "self_attn.v_proj.weight", kd, H, 1.0f);
        WL(o, "self_attn.o_proj.weight", H, qd, 1.0f / d->out_div);
        W(qn, "self_attn.q_norm.weight", c->head_dim, 0, 0, false);
        W(kn, "self_attn.k_norm.weight", c->head_dim, 0, 0, false);
        WL(gate, "mlp.gate_proj.weight", I, H, 1.0f);
        WL(up, "mlp.up_proj.weight", I, H, 1.0f / d->mlp_div);
        WL(down, "mlp.down_proj.weight", H, I, d->mlp_div / d->out_div);
        W(abase, "attention_conv.base_kernel", 2, K, H, true);
        WL(aproj, "attention_conv.kernel_projection.weight", 2 * K * G, H, 1.0f);
        W(mbase, "mlp_conv.base_kernel", 2, K, H, true);
        WL(mproj, "mlp_conv.kernel_projection.weight", 2 * K * G, H, 1.0f);
#undef W
#undef WL
    }
    if (s == VV_OK)
        s = upload_lin(d, &ws, "fc.weight", H, (int64_t)c->n_taps * H,
                       1.0f / VV_SPEC_FC_SCALE, &d->fc);
    if (s == VV_OK) s = upload(d, &ws, "hidden_norm.weight", H, 0, 0, false, 1.0f, &d->hnorm);
    if (s == VV_OK) s = upload(d, &ws, "norm.weight", H, 0, 0, false, 1.0f, &d->norm);
    if (s == VV_OK)
        s = upload(d, &ws, "candidate_selector.predecessor_codebook",
                   c->vocab_size, c->selector_rank, 0, false, 1.0f, &d->pred);
    if (s == VV_OK)
        s = upload(d, &ws, "candidate_selector.successor_codebook",
                   c->vocab_size, c->selector_rank, 0, false, 1.0f, &d->succ);
    if (s == VV_OK)
        s = upload(d, &ws, "candidate_selector.hidden_projection.weight",
                   c->selector_rank, H, 0, false, 1.0f, &d->hproj);
    d->Vd = c->vocab_size;
    if (s == VV_OK && c->draft_vocab_size > 0)
        s = load_draft_vocab(d, &ws, target);
    vv_safetensors_close(ws.st);
    d->hnorm_eps = c->rms_norm_eps / (VV_SPEC_FC_SCALE * VV_SPEC_FC_SCALE);
    if (s != VV_OK) { vv_drafter_free(d); return s; }
    VV_LOG_I("spec: DFlash 2 drafter from %s: %d layers, block %d, taps %d, "
             "%d draft ids, %s, %.1f MB on gpu %d", dir, c->num_layers,
             c->block_size, c->n_taps, d->Vd,
             c->weight_quant == VV_DRAFTER_INT4 ? "int4" : "f16",
             (double)d->bytes / (1024.0 * 1024.0), gpu_id);
    *out = d;
    return VV_OK;
}

const vv_drafter_config_t* vv_drafter_get_config(const vv_drafter_t* d) {
    return d ? &d->cfg : NULL;
}

size_t vv_drafter_device_bytes(const vv_drafter_t* d) {
    return d ? d->bytes : 0;
}

/* ─── Per-context state ─────────────────────────────────────────────────── */

struct vv_spec {
    const vv_drafter_t* d;
    int B, H, nh, nkv, hd, I, V, rank, topk, G, n_taps, max_pos;
    int Bv;                    /* rows a cycle checks: the anchor, Bv-1 drafts */
    int ctx_len;               /* drafter context: positions [0, ctx_len) */
    int backend;               /* attention backend for the draft pass */
    void** kc;                 /* [layers] K [max_pos][nkv][hd] FP16 */
    void** vc;
    vv_taps_t taps;            /* the target's taps, VV_SPEC_TAP_ROWS rows */
    void* stream;              /* the context's compute stream */
    /* draft pass */
    int32_t* ids;              /* [B]: anchor, then MASK / the drafts */
    int32_t* anchor;           /* [1] */
    int32_t* draft;            /* [B-1] */
    int32_t* post;             /* [B] the target's tokens */
    int32_t* out;              /* [B] accepted drafts + the target's next */
    int32_t* n_out;            /* [1] */
    float* h;                  /* [B][H] the residual stream, FP32 */
    void *xe, *xn, *xc, *dyn, *q, *qn, *kk, *kn, *vv, *att, *o, *oc;
    void *gate, *up, *hp;
    float* logits;             /* [B][V] */
    float* tk_v;               /* [B][topk] */
    int32_t* tk_i;
    void* am;                  /* argmax scratch, B rows */
    void* attn_ws;
    void* gemm_ws;             /* W4A16 split-K partials */
    size_t gemm_ws_bytes;
    /* context update, VV_SPEC_TAP_ROWS rows */
    void *cx, *cxn, *ck, *ckn, *cv;
    /* verify */
    void* vh;                  /* [B][Ht] the target's rows */
    void* vn;                  /* [B][Ht] normed */
    int32_t* h_pin;            /* pinned: out[B], n_out */
    void** allocs;
    int n_allocs, cap_allocs;
    vv_spec_stats_t stats;
};

static vv_status_t salloc(vv_spec_t* s, void** p, size_t n) {
    if (s->n_allocs == s->cap_allocs) {
        const int nc = s->cap_allocs ? s->cap_allocs * 2 : 64;
        void** na = (void**)vv_realloc(s->allocs, sizeof(void*) * (size_t)nc);
        if (!na) return VV_ERR_OUT_OF_MEMORY;
        s->allocs = na;
        s->cap_allocs = nc;
    }
    const vv_status_t st = vv_dev_alloc(p, n > 0 ? n : 16);
    if (st == VV_OK) s->allocs[s->n_allocs++] = *p;
    return st;
}

void vv_spec_free(vv_spec_t* s) {
    if (!s) return;
    for (int i = 0; i < s->n_allocs; i++) vv_dev_free(s->allocs[i]);
    vv_free(s->allocs);
    if (s->h_pin) vv_dev_free_pinned(s->h_pin);
    vv_free(s->kc);
    vv_free(s->vc);
    vv_free(s);
}

vv_status_t vv_spec_create(const vv_drafter_t* d, int target_hidden,
                           int max_pos, int attn_backend, int verify_rows,
                           void* stream, vv_spec_t** out) {
    if (!d || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    const vv_drafter_config_t* c = &d->cfg;
    vv_spec_t* s = (vv_spec_t*)vv_alloc(sizeof(*s));
    if (!s) return VV_ERR_OUT_OF_MEMORY;
    memset(s, 0, sizeof(*s));
    s->d = d;
    s->B = c->block_size;
    s->H = c->hidden_size;
    s->nh = c->num_heads;
    s->nkv = c->num_kv_heads;
    s->hd = c->head_dim;
    s->I = c->intermediate_size;
    s->V = c->vocab_size;
    s->rank = c->selector_rank;
    s->topk = c->selector_top_k;
    s->G = c->hidden_size / c->conv_group;
    s->n_taps = c->n_taps;
    s->max_pos = max_pos;
    s->stream = stream;
    s->backend = vv_attn_resolve(attn_backend, VV_KV_FP16, false, s->nh,
                                 s->nkv, s->hd);
    const size_t B = (size_t)s->B, H = (size_t)s->H;
    const size_t qd = (size_t)s->nh * s->hd, kd = (size_t)s->nkv * s->hd;
    const size_t T = VV_SPEC_TAP_ROWS;
    vv_status_t st = VV_OK;
    s->kc = (void**)vv_alloc(sizeof(void*) * (size_t)c->num_layers);
    s->vc = (void**)vv_alloc(sizeof(void*) * (size_t)c->num_layers);
    if (!s->kc || !s->vc) st = VV_ERR_OUT_OF_MEMORY;
    /* A block's worth past the window: the draft pass writes the whole
     * block even when the target checks fewer rows. */
    const size_t kv_rows = (size_t)max_pos + (size_t)s->B;
    for (int l = 0; st == VV_OK && l < c->num_layers; l++) {
        st = salloc(s, &s->kc[l], kv_rows * kd * 2);
        if (st == VV_OK) st = salloc(s, &s->vc[l], kv_rows * kd * 2);
    }
#define A(p, n) if (st == VV_OK) st = salloc(s, (void**)&(p), (n))
    A(s->taps.buf, T * (size_t)s->n_taps * H * 2);
    A(s->ids, B * 4); A(s->anchor, 4); A(s->draft, B * 4); A(s->post, B * 4);
    A(s->out, B * 4); A(s->n_out, 4);
    A(s->h, B * H * 4); A(s->xe, B * H * 2);
    A(s->xn, B * H * 2); A(s->xc, B * H * 2);
    A(s->dyn, B * 2 * (size_t)c->conv_kernel * s->G * 2);
    A(s->q, B * qd * 2); A(s->qn, B * qd * 2); A(s->kk, B * kd * 2);
    A(s->kn, B * kd * 2); A(s->vv, B * kd * 2); A(s->att, B * qd * 2);
    A(s->o, B * H * 2); A(s->oc, B * H * 2);
    A(s->gate, B * (size_t)s->I * 2); A(s->up, B * (size_t)s->I * 2);
    A(s->hp, B * (size_t)s->rank * 2);
    A(s->logits, B * (size_t)s->V * 4);
    A(s->tk_v, B * (size_t)s->topk * 4); A(s->tk_i, B * (size_t)s->topk * 4);
    A(s->am, vv_argmax_rows_scratch_bytes(s->B));
    A(s->attn_ws, vv_attn_scratch_bytes(s->nh, s->nkv, s->hd));
    s->gemm_ws_bytes = (size_t)8 << 20;
    A(s->gemm_ws, s->gemm_ws_bytes);
    A(s->cx, T * H * 2); A(s->cxn, T * H * 2);
    A(s->ck, T * kd * 2); A(s->ckn, T * kd * 2); A(s->cv, T * kd * 2);
    A(s->vh, B * (size_t)target_hidden * 2); A(s->vn, B * (size_t)target_hidden * 2);
#undef A
    if (st == VV_OK) st = vv_dev_alloc_pinned((void**)&s->h_pin, (B + 1) * 4);
    if (st != VV_OK) { vv_spec_free(s); return st; }
    s->taps.n = s->n_taps;
    for (int i = 0; i < s->n_taps; i++) s->taps.layers[i] = c->target_layer_ids[i];
    s->taps.rows = VV_SPEC_TAP_ROWS;
    /*
     * The drafter always drafts its whole block; the target may check fewer
     * of the drafts (VV_SPEC_VERIFY=n rows, the anchor included). Checking
     * costs about linearly in rows while the later drafts are the least
     * likely to be kept, so the best n depends on the drafter and the card.
     */
    s->Bv = verify_rows >= 2 ? verify_rows : 4;
    {
        const char* e = getenv("VV_SPEC_VERIFY");      /* measurements */
        const int v = e ? atoi(e) : 0;
        if (v >= 2) s->Bv = v;
    }
    if (s->Bv > s->B) s->Bv = s->B;
    *out = s;
    return VV_OK;
}

void vv_spec_reset(vv_spec_t* s) {
    if (s) s->ctx_len = 0;
}

int vv_spec_block(const vv_spec_t* s) { return s ? s->Bv : 0; }

int vv_spec_context_len(const vv_spec_t* s) { return s ? s->ctx_len : 0; }

const vv_spec_stats_t* vv_spec_get_stats(const vv_spec_t* s) {
    return s ? &s->stats : NULL;
}

void vv_spec_stats_reset(vv_spec_t* s) {
    if (s) memset(&s->stats, 0, sizeof(s->stats));
}

/*
 * Switches read once per process:
 *   VV_SPEC_DEBUG=1    absmax and non-finite counts of the draft pass's
 *                      buffers, synchronously (dbg below);
 *   VV_SPEC_PROFILE=1  drain the stream after the draft so the cycle's time
 *                      splits into drafting and checking (costs a sync);
 *   VV_SPEC_EXACT=0    check blocks with the prefill kernels instead of the
 *                      decode-exact ones: faster on some shapes, but a row's
 *                      logits then differ from its decode step's in the last
 *                      bits, so near-ties can flip. For measurements only.
 *   VV_SPEC_LOG=path   append "p anchor drafts..." per cycle to `path`, for
 *                      checking the drafter against its trainer.
 */
static bool s_debug, s_profile, s_exact = true;
static const char* s_log;
static vv_once_t s_env_once = VV_ONCE_INIT;
static void env_probe(void) {
    const char* e = getenv("VV_SPEC_DEBUG");
    s_debug = e && e[0] == '1';
    e = getenv("VV_SPEC_PROFILE");
    s_profile = e && e[0] == '1';
    e = getenv("VV_SPEC_EXACT");
    s_exact = !(e && e[0] == '0');
    e = getenv("VV_SPEC_LOG");
    s_log = e && e[0] ? e : NULL;
}

/* Absmax and non-finite count of a buffer, synchronously (VV_SPEC_DEBUG). */
static void dbg(const char* what, const void* dev, size_t n, bool f32,
                void* stream) {
    vv_once(&s_env_once, env_probe);
    if (!s_debug || !dev || !n) return;
    vv_dev_stream_sync(stream);
    void* h = vv_alloc(n * (f32 ? 4 : 2));
    if (!h) return;
    vv_dev_memcpy_d2h(h, dev, n * (f32 ? 4 : 2), NULL);
    double mx = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        const float v = f32 ? ((float*)h)[i] : vv_half_to_float(((uint16_t*)h)[i]);
        if (!isfinite(v)) { bad++; continue; }
        if (fabs(v) > mx) mx = fabs(v);
    }
    VV_LOG_I("spec-debug: %-12s n=%zu absmax=%.4g nonfinite=%zu", what, n, mx, bad);
    vv_free(h);
}

/* y[M][N] = x[M][K] . W^T in W's format. */
static vv_status_t dlin(vv_spec_t* s, const void* x, const dlin_t* W, void* y,
                        int M, int N, int K, void* stream) {
    if (W->packed)
        return vv_w4a16_gemm_dev(x, W->packed, W->sz, NULL, y, s->gemm_ws,
                                 s->gemm_ws_bytes, M, N, K, DRAFT_Q_GROUP,
                                 stream);
    return vv_gemm_fp16_dev(x, W->w, y, M, N, K, 1.0f, 0.0f, stream);
}

/* A layer's K/V rows [pos, pos + n) of the drafter cache. */
static void* kv_row(void* base, int pos, int kd) {
    return (uint8_t*)base + (size_t)pos * (size_t)kd * 2;
}

/*
 * Context positions [pos0, pos0 + n) from their tap rows: fc, hidden_norm,
 * then every layer's k/v projection, k_norm and RoPE, into the cache.
 */
static vv_status_t ctx_update(vv_spec_t* s, const void* rows, int pos0, int n,
                              void* stream) {
    const vv_drafter_t* d = s->d;
    const int H = s->H, kd = s->nkv * s->hd;
    if (n <= 0) return VV_OK;
    if (pos0 + n > s->max_pos) return VV_ERR_OVERFLOW;
    dbg("ctx.taps", rows, (size_t)n * s->n_taps * H, false, stream);
    vv_status_t st = dlin(s, rows, &d->fc, s->cx, n, H, s->n_taps * H, stream);
    dbg("ctx.fc", s->cx, (size_t)n * H, false, stream);
    if (st == VV_OK)
        st = vv_rmsnorm_dev(s->cx, d->hnorm, s->cxn, n, H, d->hnorm_eps, stream);
    dbg("ctx.norm", s->cxn, (size_t)n * H, false, stream);
    for (int l = 0; st == VV_OK && l < d->cfg.num_layers; l++) {
        const dlayer_t* L = &d->L[l];
        st = dlin(s, s->cxn, &L->k, s->ck, n, kd, H, stream);
        if (st == VV_OK) st = dlin(s, s->cxn, &L->v, s->cv, n, kd, H, stream);
        if (st == VV_OK)
            st = vv_rmsnorm_dev(s->ck, L->kn, s->ckn, n * s->nkv, s->hd,
                                d->cfg.rms_norm_eps, stream);
        if (st == VV_OK)
            st = vv_rope_dev(s->ckn, n, s->nkv, s->hd, pos0, NULL,
                             d->cfg.rope_theta, stream);
        if (st == VV_OK)
            st = vv_dev_memcpy_d2d(kv_row(s->kc[l], pos0, kd), s->ckn,
                                   (size_t)n * kd * 2, stream);
        if (st == VV_OK)
            st = vv_dev_memcpy_d2d(kv_row(s->vc[l], pos0, kd), s->cv,
                                   (size_t)n * kd * 2, stream);
    }
    return st;
}

/* taps->on_chunk of a prompt prefill: its rows are the context there. */
static vv_status_t on_chunk(void* user, int pos0, int n, void* stream) {
    vv_spec_t* s = (vv_spec_t*)user;
    if (pos0 != s->ctx_len) return VV_ERR_INVALID_ARG;
    const vv_status_t st = ctx_update(s, s->taps.buf, pos0, n, stream);
    if (st == VV_OK) s->ctx_len = pos0 + n;
    return st;
}

const vv_taps_t* vv_spec_prefill_taps(vv_spec_t* s) {
    if (!s) return NULL;
    s->taps.on_chunk = on_chunk;
    s->taps.user = s;
    s->taps.row_base = 0;
    return &s->taps;
}

vv_status_t vv_spec_truncate(vv_spec_t* s, int len) {
    if (!s) return VV_ERR_NULL_PTR;
    if (len < 0 || len > s->ctx_len) return VV_ERR_INVALID_ARG;
    s->ctx_len = len;
    return VV_OK;
}

/*
 * The draft pass: B rows [anchor, MASK...] at positions p.. over the
 * context [0, p). The block's own keys and values go into the cache at
 * [p, p + B) for the attention and are overwritten by the next context
 * update, which starts at p.
 */
static vv_status_t draft(vv_spec_t* s, const vv_inference_ctx_t* ctx,
                         void* stream) {
    const vv_drafter_t* d = s->d;
    const vv_drafter_config_t* c = &d->cfg;
    const int B = s->B, H = s->H, p = s->ctx_len;
    const int qd = s->nh * s->hd, kd = s->nkv * s->hd;
    const int dyn_ld = 2 * c->conv_kernel * s->G;
    const int half = c->conv_kernel * s->G;
    if (p + B > s->max_pos + s->B) return VV_ERR_OVERFLOW;

    vv_status_t st = vv_dflash_block_ids_dev(s->anchor, c->mask_token_id, B,
                                             s->ids, stream);
    if (st == VV_OK)
        st = vv_embedding_dev(ctx->embed_table_gpu, s->ids, s->xe, B, H, stream);
    if (st == VV_OK)
        st = vv_add_scaled_f16_dev(s->h, s->xe, 1.0f, B * H, true, stream);
    const float od = d->out_div;
    for (int l = 0; st == VV_OK && l < c->num_layers; l++) {
        const dlayer_t* L = &d->L[l];
        vv_kv_view_t view;
        memset(&view, 0, sizeof(view));
        view.k = s->kc[l];
        view.v = s->vc[l];
        view.format = VV_KV_FP16;
        view.n_kv_heads = s->nkv;
        view.head_dim = s->hd;
#define OK(x) if (st == VV_OK) st = (x)
        OK(vv_rmsnorm_f32in_dev(s->h, L->ln1, s->xn, B, H, c->rms_norm_eps, stream));
        OK(dlin(s, s->xn, &L->aproj, s->dyn, B, dyn_ld, H, stream));
        OK(vv_dflash_conv_dev(s->xn, s->dyn, dyn_ld, L->abase, s->xc, B, H,
                              c->conv_kernel, c->conv_group, B, stream));
        OK(dlin(s, s->xc, &L->q, s->q, B, qd, H, stream));
        OK(dlin(s, s->xc, &L->k, s->kk, B, kd, H, stream));
        OK(dlin(s, s->xc, &L->v, s->vv, B, kd, H, stream));
        OK(vv_rmsnorm_dev(s->q, L->qn, s->qn, B * s->nh, s->hd, c->rms_norm_eps, stream));
        OK(vv_rmsnorm_dev(s->kk, L->kn, s->kn, B * s->nkv, s->hd, c->rms_norm_eps, stream));
        OK(vv_rope_dev(s->qn, B, s->nh, s->hd, p, NULL, c->rope_theta, stream));
        OK(vv_rope_dev(s->kn, B, s->nkv, s->hd, p, NULL, c->rope_theta, stream));
        OK(vv_dev_memcpy_d2d(kv_row(s->kc[l], p, kd), s->kn, (size_t)B * kd * 2, stream));
        OK(vv_dev_memcpy_d2d(kv_row(s->vc[l], p, kd), s->vv, (size_t)B * kd * 2, stream));
        OK(vv_attn_prefill(s->backend, s->qn, &view, s->att, s->nh, B, p,
                           p + B, false, s->attn_ws, stream));
        if (l == 0) {
            dbg("draft.emb", s->xn, (size_t)B * H, false, stream);
            dbg("draft.q", s->qn, (size_t)B * qd, false, stream);
            dbg("draft.kcache", s->kc[l], (size_t)(p + B) * kd, false, stream);
            dbg("draft.att", s->att, (size_t)B * qd, false, stream);
        }
        OK(dlin(s, s->att, &L->o, s->o, B, H, qd, stream));
        if (l == 0) dbg("draft.oproj", s->o, (size_t)B * H, false, stream);
        if (l == 0) dbg("draft.dyn", s->dyn, (size_t)B * dyn_ld, false, stream);
        OK(vv_dflash_conv_dev(s->o, (const uint8_t*)s->dyn + (size_t)half * 2,
                              dyn_ld, L->abase + (size_t)c->conv_kernel * H,
                              s->oc, B, H, c->conv_kernel, c->conv_group, B,
                              stream));
        if (l == 0) dbg("draft.oconv", s->oc, (size_t)B * H, false, stream);
        OK(vv_add_scaled_f16_dev(s->h, s->oc, od, B * H, false, stream));
        if (l == 0) dbg("draft.h1", s->h, (size_t)B * H, true, stream);
        OK(vv_rmsnorm_f32in_dev(s->h, L->ln2, s->xn, B, H, c->rms_norm_eps, stream));
        OK(dlin(s, s->xn, &L->mproj, s->dyn, B, dyn_ld, H, stream));
        OK(vv_dflash_conv_dev(s->xn, s->dyn, dyn_ld, L->mbase, s->xc, B, H,
                              c->conv_kernel, c->conv_group, B, stream));
        OK(dlin(s, s->xc, &L->gate, s->gate, B, s->I, H, stream));
        OK(dlin(s, s->xc, &L->up, s->up, B, s->I, H, stream));
        OK(vv_swiglu_dev(s->gate, s->up, s->gate, B * s->I, stream));
        if (l == 0) dbg("draft.act", s->gate, (size_t)B * s->I, false, stream);
        OK(dlin(s, s->gate, &L->down, s->o, B, H, s->I, stream));
        OK(vv_dflash_conv_dev(s->o, (const uint8_t*)s->dyn + (size_t)half * 2,
                              dyn_ld, L->mbase + (size_t)c->conv_kernel * H,
                              s->oc, B, H, c->conv_kernel, c->conv_group, B,
                              stream));
        OK(vv_add_scaled_f16_dev(s->h, s->oc, od, B * H, false, stream));
    }
    dbg("draft.h", s->h, (size_t)B * H, true, stream);
    /* Rows 1..B-1 through the norm, the draft head and the selector; the
     * candidates leave as vocabulary ids whichever head ranked them. */
    const size_t row = (size_t)H * 2;
    OK(vv_rmsnorm_f32in_dev(s->h, d->norm, s->xn, B, H, c->rms_norm_eps, stream));
    OK(vv_lm_head_rows_dev((const uint8_t*)s->xn + row,
                           d->head ? d->head : ctx->lm_head_gpu,
                           s->logits, B - 1, d->Vd, H, stream));
    dbg("draft.logits", s->logits, (size_t)(B - 1) * d->Vd, true, stream);
    OK(vv_topk_rows_dev(s->logits, B - 1, d->Vd, s->topk, s->tk_v, s->tk_i,
                        stream));
    if (d->vmap)
        OK(vv_gather_i32_dev(d->vmap, s->tk_i, (B - 1) * s->topk, stream));
    OK(vv_gemm_fp16_dev((const uint8_t*)s->xn + row, d->hproj, s->hp, B - 1,
                        s->rank, H, 1.0f, 0.0f, stream));
    OK(vv_dflash_walk_dev(s->hp, s->tk_v, s->tk_i, d->pred, d->succ,
                          s->anchor, B - 1, s->topk, s->rank, s->draft,
                          stream));
#undef OK
    return st;
}

static bool verify_exact(void) {
    vv_once(&s_env_once, env_probe);
    return s_exact;
}

/*
 * One cycle on the GPU. The target's cache holds [0, p) and `anchor` is the
 * token at p, not yet fed. The drafter proposes B - 1 tokens; the target
 * runs the B rows [anchor, drafts] as one prefill and gives its own token
 * after each; the drafts it agrees with are kept, then its next token. Both
 * caches end at p + kept + 1, the last produced token the new anchor.
 */
vv_status_t vv_spec_cycle(vv_inference_ctx_t* ctx, vv_spec_t* s,
                          int32_t anchor, int32_t* out, int* n_out) {
    if (!ctx || !s || !out || !n_out) return VV_ERR_NULL_PTR;
    *n_out = 0;
    vv_kv_cache_t* kv = ctx->kv_cache;
    const int B = s->Bv, p = kv->current_len;     /* rows checked */
    const int Ht = ctx->model->config.llm.hidden_size;
    void* stream = ctx->compute_stream;
    if (p != s->ctx_len) {
        VV_LOG_E("spec: drafter context at %d, target cache at %d",
                 s->ctx_len, p);
        return VV_ERR_INVALID_ARG;
    }
    if (p + B > kv->max_seq_len || p + B > s->max_pos) return VV_ERR_OVERFLOW;
    const double t0 = vv_time_ms();

    s->h_pin[0] = anchor;
    vv_status_t st = vv_dev_memcpy_h2d(s->anchor, s->h_pin, 4, stream);
    if (st == VV_OK) st = draft(s, ctx, stream);
    vv_once(&s_env_once, env_probe);
    if (st == VV_OK && s_profile) st = vv_dev_stream_sync(stream);
    /* The block to verify: [anchor, drafts]. */
    if (st == VV_OK) st = vv_dev_memcpy_d2d(s->ids, s->anchor, 4, stream);
    if (st == VV_OK)
        st = vv_dev_memcpy_d2d(s->ids + 1, s->draft, (size_t)(B - 1) * 4, stream);
    if (st == VV_OK && kv->pool) st = vv_kv_cache_reserve_wait(kv, p + B, stream);
    if (st == VV_OK)
        st = vv_embedding_dev(ctx->embed_table_gpu, s->ids, s->vh, B, Ht, stream);
    const double t1 = vv_time_ms();
    if (st == VV_OK) {
        s->taps.on_chunk = NULL;
        s->taps.row_base = p;           /* position p lands at row 0 */
        if (verify_exact()) {
            /* Every row exactly as its own decode step: the same tokens. */
            st = vv_decoder_verify(ctx->model, s->vh, B, kv, ctx->layer_pool,
                                   ctx->workspace, ctx->workspace_size,
                                   stream, ctx->transfer_stream, &s->taps);
        } else {
            ctx->taps = &s->taps;
            st = vv_pipeline_prefill(ctx, s->vh, B);
            ctx->taps = NULL;
        }
    }
    if (st == VV_OK)
        st = vv_rmsnorm_dev(s->vh, ctx->final_norm_gpu, s->vn, B, Ht,
                            ctx->model->config.llm.rms_norm_eps, stream);
    if (st == VV_OK)
        st = vv_lm_head_rows_dev(s->vn, ctx->lm_head_gpu, s->logits, B, s->V,
                                 Ht, stream);
    if (st == VV_OK) st = vv_argmax_rows_dev(s->logits, B, s->V, s->am, s->post,
                                             stream);
    if (st == VV_OK)
        st = vv_dflash_accept_dev(s->draft, s->post, B - 1, s->out, s->n_out,
                                  stream);
    if (st == VV_OK) st = vv_dev_memcpy_d2h(s->h_pin, s->out, (size_t)B * 4, stream);
    if (st == VV_OK) st = vv_dev_memcpy_d2h(s->h_pin + B, s->n_out, 4, stream);
    if (st == VV_OK) st = vv_dev_stream_sync(stream);
    if (st != VV_OK) return st;

    const int n = s->h_pin[B];
    if (n < 1 || n > B) return VV_ERR_INVALID_ARG;
    /* Fed: anchor and the n - 1 kept drafts, positions p .. p + n - 1. */
    kv->current_len = p + n;
    st = ctx_update(s, s->taps.buf, p, n, stream);
    if (st != VV_OK) return st;
    s->ctx_len = p + n;
    for (int i = 0; i < n; i++) out[i] = s->h_pin[i];
    *n_out = n;
    if (s_log) {
        int32_t dr[VV_SPEC_MAX_BLOCK];
        if (vv_dev_memcpy_d2h(dr, s->draft, (size_t)(s->B - 1) * 4, stream) ==
                VV_OK && vv_dev_stream_sync(stream) == VV_OK) {
            FILE* f = fopen(s_log, "a");
            if (f) {
                fprintf(f, "%d %d", p, anchor);
                for (int i = 0; i < s->B - 1; i++) fprintf(f, " %d", dr[i]);
                fprintf(f, "\n");
                fclose(f);
            }
        }
    }
    s->stats.cycles++;
    s->stats.drafted += B - 1;
    s->stats.accepted += n - 1;
    s->stats.tokens += n;
    s->stats.draft_ms += t1 - t0;
    s->stats.verify_ms += vv_time_ms() - t1;
    return VV_OK;
}

vv_status_t vv_spec_rewind(vv_inference_ctx_t* ctx, vv_spec_t* s, int len) {
    if (!ctx || !s) return VV_ERR_NULL_PTR;
    if (len < 0 || len > ctx->kv_cache->current_len || len > s->ctx_len)
        return VV_ERR_INVALID_ARG;
    ctx->kv_cache->current_len = len;
    s->ctx_len = len;
    return VV_OK;
}
