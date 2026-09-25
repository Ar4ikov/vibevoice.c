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
 * cannot hold (a SwiGLU output of 2e4 feeding a 9472-wide down projection;
 * the residual of the 7B's drafter reaches 3e6 in a few channels, the
 * massive activations its target has too). Three rescalings keep every
 * FP16 intermediate in range without changing the function: `fc` is stored
 * divided by VV_SPEC_FC_SCALE and the RMSNorm after it takes
 * eps / VV_SPEC_FC_SCALE^2 (the target's features carry activations of
 * ~1.3e4); `up_proj` is divided by `mlp_div` (32) and `down_proj`
 * multiplied by mlp_div / out_div, `o_proj` divided by out_div (1024), and
 * the two sublayer outputs are added into the residual times out_div.
 *
 * By default (VV_DRAFTER_INT4) the projections are quantized at load into
 * INT4 groups of 128 (round to nearest, exact zero point) and run on the
 * W4A16 kernels, a quarter of the bytes per draft pass. The drafter's
 * numbers only decide what gets proposed; the target's check decides what
 * is kept.
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
    int    n_prequant;    /* projections the checkpoint stores quantized */
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

/* An INT4G row-major weight [N][K] (codes, FP16 scales, integer zeros per
 * group of DRAFT_Q_GROUP) onto the device in the W4A16 GPU layout. */
static vv_status_t put_int4g(vv_drafter_t* d, uint8_t* packed,
                             const uint16_t* scales, const uint8_t* zeros,
                             int64_t N, int64_t K, dlin_t* out) {
    const size_t n = (size_t)N * (size_t)K, ng = n / DRAFT_Q_GROUP;
    uint16_t* sz = (uint16_t*)vv_alloc(ng * 4);
    vv_status_t s = sz ? VV_OK : VV_ERR_OUT_OF_MEMORY;
    if (s == VV_OK)
        s = vv_int4g_to_gpu_layout(packed, scales, zeros, (int)N, (int)K,
                                   DRAFT_Q_GROUP, sz);
    if (s == VV_OK) s = dalloc(d, &out->packed, n / 2);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(out->packed, packed, n / 2, NULL);
    if (s == VV_OK) s = dalloc(d, &out->sz, ng * 4);
    if (s == VV_OK) s = vv_dev_memcpy_h2d(out->sz, sz, ng * 4, NULL);
    vv_free(sz);
    return s;
}

/* A float of any stored width. */
static float stored_float(const void* p, vv_dtype_t t, size_t i) {
    if (t == VV_DTYPE_BF16) return vv_bf16_to_float(((const uint16_t*)p)[i]);
    if (t == VV_DTYPE_F16) return vv_half_to_float(((const uint16_t*)p)[i]);
    return ((const float*)p)[i];
}

/*
 * A projection stored quantized already, the way compressed-tensors writes a
 * pack-quantized INT4 Linear (tools/dflash/awq_drafter.py does):
 *
 *   <base>.weight_packed      I32 [N][K/8], element k in bits 4(k%8) of word
 *                             k/8, stored as value + 8
 *   <base>.weight_scale       float [N][K/G]
 *   <base>.weight_zero_point  I32 [N/8][K/G] packed along N (z + 8), or
 *                             I8 [N][K/G]; absent: symmetric
 *
 * with w = (q - z) * s and G = DRAFT_Q_GROUP. Read into INT4G (codes and
 * zeros 0..15, FP16 scales times `mul`, a power of two, so exact down to
 * FP16's subnormals). VV_ERR_WEIGHT_MISSING, quietly, when there is no
 * `<base>.weight_packed`.
 */
static vv_status_t read_packed(const wsrc_t* ws, const char* base, int64_t N,
                               int64_t K, float mul, uint8_t** packed_out,
                               uint16_t** scales_out, uint8_t** zeros_out) {
    char full[256];
    vv_st_tensor_info_t pi, si, zi;
    snprintf(full, sizeof(full), "%s%s.weight_packed", ws->prefix, base);
    if (vv_safetensors_find(ws->st, full, &pi) != VV_OK)
        return VV_ERR_WEIGHT_MISSING;
    const int64_t ng = K / DRAFT_Q_GROUP;
    if (pi.dtype != VV_DTYPE_I32 || pi.ndim != 2 || pi.shape[0] != N ||
        pi.shape[1] != K / 8 || K % DRAFT_Q_GROUP || N % 8) {
        VV_LOG_E("spec: '%s' is not the int32 [%lld, %lld] of a packed "
                 "4-bit [%lld x %lld] weight", full, (long long)N,
                 (long long)(K / 8), (long long)N, (long long)K);
        return VV_ERR_SHAPE_MISMATCH;
    }
    snprintf(full, sizeof(full), "%s%s.weight_scale", ws->prefix, base);
    if (vv_safetensors_find(ws->st, full, &si) != VV_OK ||
        (si.dtype != VV_DTYPE_BF16 && si.dtype != VV_DTYPE_F16 &&
         si.dtype != VV_DTYPE_F32) ||
        si.ndim != 2 || si.shape[0] != N || si.shape[1] != ng) {
        VV_LOG_E("spec: '%s' must be float [%lld, %lld]: groups of %d",
                 full, (long long)N, (long long)ng, DRAFT_Q_GROUP);
        return VV_ERR_SHAPE_MISMATCH;
    }
    snprintf(full, sizeof(full), "%s%s.weight_zero_point", ws->prefix, base);
    const bool has_z = vv_safetensors_find(ws->st, full, &zi) == VV_OK;
    const bool z_packed = has_z && zi.dtype == VV_DTYPE_I32 && zi.ndim == 2 &&
                          zi.shape[0] == N / 8 && zi.shape[1] == ng;
    const bool z_plain = has_z && zi.dtype == VV_DTYPE_I8 && zi.ndim == 2 &&
                         zi.shape[0] == N && zi.shape[1] == ng;
    if (has_z && !z_packed && !z_plain) {
        VV_LOG_E("spec: '%s' is not a zero point the runtime reads", full);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const void *pd = NULL, *sd = NULL, *zd = NULL;
    vv_status_t s = vv_safetensors_get_data(ws->st, &pi, &pd);
    if (s == VV_OK) s = vv_safetensors_get_data(ws->st, &si, &sd);
    if (s == VV_OK && has_z) s = vv_safetensors_get_data(ws->st, &zi, &zd);
    if (s != VV_OK) return s;

    const size_t n = (size_t)N * (size_t)K, nG = (size_t)N * (size_t)ng;
    uint8_t* packed = (uint8_t*)vv_alloc(n / 2);
    uint16_t* scales = (uint16_t*)vv_alloc(nG * 2);
    uint8_t* zeros = (uint8_t*)vv_alloc(nG);
    if (!packed || !scales || !zeros) {
        vv_free(packed); vv_free(scales); vv_free(zeros);
        return VV_ERR_OUT_OF_MEMORY;
    }
    const uint32_t* w = (const uint32_t*)pd;
    for (int64_t r = 0; r < N; r++) {
        const uint32_t* wr = w + (size_t)r * (size_t)(K / 8);
        uint8_t* prow = packed + (size_t)r * (size_t)(K / 2);
        for (int64_t k = 0; k < K; k += 2) {
            const unsigned c0 = (wr[k >> 3] >> (4 * (k & 7))) & 15u;
            const unsigned c1 = (wr[(k + 1) >> 3] >> (4 * ((k + 1) & 7))) & 15u;
            prow[k >> 1] = (uint8_t)((c0 << 4) | c1);
        }
    }
    int bad = 0;
    for (int64_t r = 0; r < N; r++)
        for (int64_t g = 0; g < ng; g++) {
            const size_t i = (size_t)r * (size_t)ng + (size_t)g;
            scales[i] = vv_float_to_half_rne(stored_float(sd, si.dtype, i) * mul);
            int z = 8;
            if (z_packed)
                z = (int)((((const uint32_t*)zd)[(size_t)(r >> 3) * (size_t)ng + g]
                           >> (4 * (r & 7))) & 15u);
            else if (z_plain)
                z = ((const int8_t*)zd)[i] + 8;
            if (z < 0 || z > 15) { bad++; z = z < 0 ? 0 : 15; }
            zeros[i] = (uint8_t)z;
        }
    if (bad) {
        VV_LOG_E("spec: '%s%s' holds %d zero points outside 4 bits",
                 ws->prefix, base, bad);
        vv_free(packed); vv_free(scales); vv_free(zeros);
        return VV_ERR_MODEL_FORMAT;
    }
    *packed_out = packed;
    *scales_out = scales;
    *zeros_out = zeros;
    return VV_OK;
}

/*
 * A projection [N][K] (scaled by `mul`) in the drafter's format: FP16 as
 * upload() does it, or quantized here to INT4 groups and put in the GPU
 * layout the W4A16 kernels read. The FP16 value is what gets quantized, so
 * the INT4 drafter approximates the FP16 one. A checkpoint that stores the
 * projection quantized (read_packed) keeps its codes: in the GPU layout, or
 * dequantized to FP16 for an f16 drafter.
 */
static vv_status_t upload_lin(vv_drafter_t* d, const wsrc_t* ws,
                              const char* name, int64_t N, int64_t K,
                              float mul, dlin_t* out) {
    memset(out, 0, sizeof(*out));
    vv_status_t s;
    {
        char base[224];
        const size_t nl = strlen(name);
        if (nl > 7 && nl - 7 < sizeof(base) && !strcmp(name + nl - 7, ".weight")) {
            memcpy(base, name, nl - 7);
            base[nl - 7] = '\0';
            uint8_t* packed = NULL;
            uint16_t* scales = NULL;
            uint8_t* zeros = NULL;
            s = read_packed(ws, base, N, K, mul, &packed, &scales, &zeros);
            if (s != VV_ERR_WEIGHT_MISSING) {
                if (s == VV_OK) d->n_prequant++;
                if (s == VV_OK && d->cfg.weight_quant == VV_DRAFTER_INT4 &&
                    K % 64 == 0) {
                    s = put_int4g(d, packed, scales, zeros, N, K, out);
                } else if (s == VV_OK) {
                    const size_t n = (size_t)N * (size_t)K;
                    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
                    if (!h) s = VV_ERR_OUT_OF_MEMORY;
                    for (size_t i = 0; h && i < n; i++) {
                        const size_t r = i / (size_t)K, k = i % (size_t)K;
                        const size_t g = r * (size_t)(K / DRAFT_Q_GROUP) +
                                         k / DRAFT_Q_GROUP;
                        const uint8_t b = packed[i >> 1];
                        const int q = (i & 1) ? (b & 15) : (b >> 4);
                        h[i] = vv_float_to_half_rne(
                            (float)(q - zeros[g]) * vv_half_to_float(scales[g]));
                    }
                    if (s == VV_OK) s = dalloc(d, &out->w, n * 2);
                    if (s == VV_OK) s = vv_dev_memcpy_h2d(out->w, h, n * 2, NULL);
                    vv_free(h);
                }
                vv_free(packed); vv_free(scales); vv_free(zeros);
                return s;
            }
        }
    }
    if (d->cfg.weight_quant != VV_DRAFTER_INT4 || N % 8 || K % 64 ||
        K % DRAFT_Q_GROUP)
        return upload(d, ws, name, N, K, 0, false, mul, &out->w);
    float* f32 = NULL;
    s = read_host(ws, name, N, K, 0, true, mul, (void**)&f32);
    if (s != VV_OK) return s;
    const size_t n = (size_t)N * (size_t)K;
    const size_t ng = n / DRAFT_Q_GROUP;
    for (size_t i = 0; i < n; i++)
        f32[i] = vv_half_to_float(vv_float_to_half_rne(f32[i]));
    uint8_t* packed = (uint8_t*)vv_alloc(n / 2);
    uint16_t* scales = (uint16_t*)vv_alloc(ng * 2);
    uint16_t* mins = (uint16_t*)vv_alloc(ng * 2);
    uint8_t* zeros = (uint8_t*)vv_alloc(ng);
    if (!packed || !scales || !mins || !zeros) s = VV_ERR_OUT_OF_MEMORY;
    if (s == VV_OK)
        s = vv_int4g_quantize(f32, (int)N, (int)K, DRAFT_Q_GROUP, packed,
                              scales, mins, zeros);
    if (s == VV_OK) s = put_int4g(d, packed, scales, zeros, N, K, out);
    vv_free(f32); vv_free(packed); vv_free(scales); vv_free(mins);
    vv_free(zeros);
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

vv_draft_check_t vv_draft_check_parse(const char* name) {
    if (!name) return VV_DRAFT_CHECK_COUNT;
    if (!strcmp(name, "auto")) return VV_DRAFT_CHECK_AUTO;
    if (!strcmp(name, "exact")) return VV_DRAFT_CHECK_EXACT;
    if (!strcmp(name, "fast")) return VV_DRAFT_CHECK_FAST;
    return VV_DRAFT_CHECK_COUNT;
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
    d->out_div = c->fp16_out_div > 0.0f ? c->fp16_out_div : 1024.0f;
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
             "%d draft ids, %s%s, %.1f MB on gpu %d", dir, c->num_layers,
             c->block_size, c->n_taps, d->Vd,
             c->weight_quant == VV_DRAFTER_INT4 ? "int4" : "f16",
             d->n_prequant ? " (stored quantized)" : "",
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
    bool exact;                /* rows with their decode steps' arithmetic */
    int ctx_len;               /* drafter context: positions [0, ctx_len) */
    int backend;               /* attention backend for the draft pass */
    void** kc;                 /* [layers] K [max_pos][nkv][hd] FP16 */
    void** vc;
    vv_taps_t taps;            /* the target's taps, VV_SPEC_TAP_ROWS rows */
    void* stream;              /* the context's compute stream */
    /* draft pass */
    int32_t* ids;              /* [B]: anchor, then MASK / the drafts */
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
    int32_t* h_pin;            /* pinned: out[B], n_out, then the inputs */
    /* A cycle reads its positions on the device (cycle_body), so one
     * capture replays for every cycle of a shape (vv_spec_cycle). */
    int32_t* d_in;             /* [IN_N] device copy of h_pin + IN_PIN */
    void* graph[2];            /* the captured cycle per row count, or NULL */
    int graph_key[2][4];       /* rows, check, rows' attention, block's */
    bool graph_bad[2];         /* the capture for graph_key failed: launch
                                  kernels until the key or sequence changes */
    bool graph_off;            /* no capture here: launch every kernel */
    /* Rows checked when --draft-block did not say: half the block or all
     * of it, whichever keeps more tokens per ms (vv_spec_note_block). */
    bool bv_auto;
    int bv_cand[2];            /* B / 2, B */
    double bv_tok[2];          /* tokens a block of that many rows keeps */
    double bv_ms[2];           /* ms such a block takes, 0: not seen yet */
    int bv_full_ago;           /* blocks since the last full-width one */
    bool block_attn;           /* the draft pass attends on flashinfer with
                                  its length on the device (else the host's,
                                  and no capture) */
    /* plain steps */
    vv_taps_t step_taps;       /* one row */
    /* blocks or steps (vv_spec_want_block) */
    double ctl_blk_tok, ctl_blk_ms, ctl_step_ms;
    int ctl_n_blk, ctl_plain_left, ctl_backoff;
    int ctl_since_step;        /* blocks since the last plain step */
    int ctl_step_high;         /* step times in a row set aside as outliers */
    bool ctl_skip_step;        /* the next step's time is not a step's */
    bool ctl_remeasure;        /* the plain run is one step, to measure it */
    void** allocs;
    int n_allocs, cap_allocs;
    vv_spec_stats_t stats;
};

static vv_status_t dlin_gemm(vv_spec_t* s, const void* x, const dlin_t* W,
                             void* y, int M, int N, int K, void* stream);
static size_t largest_packed(const vv_drafter_t* d);

/*
 * A cycle's inputs, in d_in (device) from the pinned h_pin + IN_PIN: the
 * anchor token, the target's length (row 0's position) and one more (row
 * 0's cache length), the drafter's context length and it plus the block.
 */
enum { IN_ANCHOR, IN_POS, IN_NEXT, IN_CTX, IN_CTX_END, IN_N };
#define IN_PIN (VV_SPEC_MAX_BLOCK + 1)

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
    for (int i = 0; i < 2; i++)
        if (s->graph[i]) vv_dev_graph_destroy(s->graph[i]);
    if (s->h_pin) vv_dev_free_pinned(s->h_pin);
    vv_free(s->kc);
    vv_free(s->vc);
    vv_free(s);
}

vv_status_t vv_spec_create(const vv_drafter_t* d, int target_hidden,
                           int max_pos, int attn_backend, int verify_rows,
                           int check, void* stream, vv_spec_t** out) {
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
    /* The drafter's numbers need not match anything, so its block attends
     * on tensor cores whatever the target uses: flashinfer reads each K/V
     * once for all rows and heads (7B, test120: 0.21 ms a cycle less than
     * the fa2 prefill path, the same drafts kept). */
    (void)attn_backend;
    s->backend = vv_attn_resolve(VV_ATTN_FLASHINFER, VV_KV_FP16, false, s->nh,
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
    A(s->ids, B * 4); A(s->draft, B * 4); A(s->post, B * 4);
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
    A(s->step_taps.buf, (size_t)s->n_taps * H * 2);
    A(s->d_in, IN_N * 4);
#undef A
    s->block_attn = s->backend == VV_ATTN_FLASHINFER &&
                    vv_attn_block_shape(s->nh, s->nkv, s->B, s->B) > 0;
    s->graph_off = !s->block_attn;
    if (st == VV_OK)
        st = vv_dev_alloc_pinned((void**)&s->h_pin, (IN_PIN + IN_N) * 4);
    if (st != VV_OK) { vv_spec_free(s); return st; }
    s->taps.n = s->n_taps;
    for (int i = 0; i < s->n_taps; i++) s->taps.layers[i] = c->target_layer_ids[i];
    s->taps.rows = VV_SPEC_TAP_ROWS;
    s->step_taps.n = s->n_taps;
    for (int i = 0; i < s->n_taps; i++)
        s->step_taps.layers[i] = c->target_layer_ids[i];
    s->step_taps.rows = 1;
    /*
     * The drafter always drafts its whole block; the target may check fewer
     * of the drafts (--draft-block: rows, the anchor included). From three
     * rows on a checked row costs about as much as a step while the later
     * drafts are the least likely to be kept, so the best count depends on
     * the drafter and the card.
     */
    s->Bv = verify_rows >= 2 ? verify_rows : s->B;
    if (s->Bv > s->B) s->Bv = s->B;
    s->bv_auto = verify_rows < 2 && s->B >= 4;
    s->bv_cand[0] = s->B / 2;
    s->bv_cand[1] = s->B;
    s->exact = check != VV_DRAFT_CHECK_FAST;
    /*
     * The INT4 drafter's GEMMs run on the W4A16 tensor-core kernels. A card
     * without them (below sm_80, or VV_W4A16_MMA=0) expands a weight into
     * the scratch first: try the widest projection, and if that is what
     * happens, give the scratch room for the largest.
     */
    if (d->fc.packed) {
        const int K = s->n_taps * s->H;
        st = dlin_gemm(s, s->cx, &d->fc, s->cxn, 2, s->H, K, stream);
        if (st == VV_ERR_OVERFLOW) {
            const size_t need = largest_packed(d) * 2;
            VV_LOG_W("spec: no tensor-core W4A16 GEMM here; the INT4 drafter "
                     "expands its weights into %zu MB of scratch "
                     "(--draft-quant f16 reads them as they are)",
                     need >> 20);
            s->gemm_ws_bytes = need;
            st = salloc(s, &s->gemm_ws, need);
            if (st == VV_OK)
                st = dlin_gemm(s, s->cx, &d->fc, s->cxn, 2, s->H, K, stream);
        }
        if (st == VV_OK) st = vv_dev_stream_sync(stream);
        if (st != VV_OK) { vv_spec_free(s); return st; }
    }
    vv_spec_reset(s);
    *out = s;
    return VV_OK;
}

/* Plain steps a sequence starts with: the step's cost, measured. The first
 * one's time is not used: it captures the tapped step's graph. */
#define CTL_WARM_STEPS 3
/* Blocks seen before they are judged, and the first plain run after a loss
 * (doubling up to CTL_RUN_MAX while blocks keep losing). */
#define CTL_MIN_BLOCKS 4
#define CTL_RUN_MIN 16
#define CTL_RUN_MAX 512
/* Blocks between two plain steps taken while blocks pay, to measure a step
 * again: both slow down as the context grows, and are compared at the same
 * length. */
#define CTL_REMEASURE 64

void vv_spec_reset(vv_spec_t* s) {
    if (!s) return;
    s->ctx_len = 0;
    s->ctl_blk_tok = s->ctl_blk_ms = s->ctl_step_ms = 0.0;
    s->ctl_n_blk = 0;
    s->ctl_plain_left = CTL_WARM_STEPS;
    s->ctl_backoff = CTL_RUN_MIN;
    s->ctl_since_step = 0;
    s->ctl_step_high = 0;
    s->ctl_skip_step = true;
    s->ctl_remeasure = false;
    s->graph_bad[0] = s->graph_bad[1] = false;
    if (s->bv_auto) {
        /* A sequence starts wide: a full block tells what every narrower
         * one would have kept too. */
        s->Bv = s->bv_cand[1];
        s->bv_tok[0] = s->bv_tok[1] = 0.0;
        s->bv_ms[0] = s->bv_ms[1] = 0.0;
        s->bv_full_ago = 0;
    }
}

bool vv_spec_want_block(vv_spec_t* s) {
    return s && s->ctl_plain_left <= 0;
}

void vv_spec_note_step(vv_spec_t* s, double ms) {
    if (!s) return;
    /* A step that captured its graph (the first of a sequence, or one at a
     * new attention split) is not what a step costs: such outliers are set
     * aside, unless three in a row say the cost has really moved. */
    if (s->ctl_skip_step) {
        s->ctl_skip_step = false;
    } else if (s->ctl_step_ms <= 0.0) {
        s->ctl_step_ms = ms;
    } else if (ms < 1.5 * s->ctl_step_ms || ++s->ctl_step_high >= 3) {
        s->ctl_step_ms = 0.8 * s->ctl_step_ms + 0.2 * ms;
        s->ctl_step_high = 0;
    }
    s->ctl_since_step = 0;
    if (s->ctl_plain_left > 0 && --s->ctl_plain_left == 0) {
        /* Blocks again, judged on what they do from here -- unless the run
         * was one step to measure, and they were paying. */
        if (!s->ctl_remeasure) {
            s->ctl_n_blk = 0;
            s->ctl_blk_tok = s->ctl_blk_ms = 0.0;
        }
        s->ctl_remeasure = false;
    }
}

/* Blocks between two full-width ones while narrower ones pay better: what
 * the rows past the narrow width would keep is only seen in a full one. */
#define BV_REFRESH 32

/*
 * Half the block or all of it. A checked row costs little next to a step
 * where the projections read the weights once for every row (W4A16), and
 * more where each row is its own GEMV (NF4, INT8) or its own walk over a
 * long fa2 cache -- so the better width depends on the model, the card and
 * the length, and is measured: a full block of n tokens says a half one
 * would have kept min(n, B/2), and each width's time is its own EMA.
 */
static void bv_choose(vv_spec_t* s, int tokens, double ms) {
    const int i = s->Bv == s->bv_cand[1] ? 1 : 0;
    s->bv_ms[i] = s->bv_ms[i] > 0.0 ? 0.8 * s->bv_ms[i] + 0.2 * ms : ms;
    if (i == 1) {
        for (int k = 0; k < 2; k++) {
            const int kept = tokens < s->bv_cand[k] ? tokens : s->bv_cand[k];
            s->bv_tok[k] = s->bv_tok[k] > 0.0 ? 0.8 * s->bv_tok[k] + 0.2 * kept
                                              : kept;
        }
        s->bv_full_ago = 0;
    } else {
        s->bv_tok[0] = 0.8 * s->bv_tok[0] + 0.2 * tokens;
        s->bv_full_ago++;
    }
    if (s->ctl_n_blk < CTL_MIN_BLOCKS) return;
    int next;
    if (s->bv_ms[0] <= 0.0) {
        next = 0;                          /* the narrow one's time, once */
    } else if (i == 0 && s->bv_full_ago >= BV_REFRESH) {
        next = 1;
    } else {
        next = s->bv_tok[0] * s->bv_ms[1] > s->bv_tok[1] * s->bv_ms[0] ? 0 : 1;
    }
    s->Bv = s->bv_cand[next];
}

void vv_spec_note_block(vv_spec_t* s, int tokens, double ms) {
    if (!s) return;
    if (s->bv_auto) bv_choose(s, tokens, ms);
    if (s->ctl_n_blk == 0) {
        s->ctl_blk_tok = tokens;
        s->ctl_blk_ms = ms;
    } else {
        s->ctl_blk_tok = 0.8 * s->ctl_blk_tok + 0.2 * tokens;
        s->ctl_blk_ms = 0.8 * s->ctl_blk_ms + 0.2 * ms;
    }
    s->ctl_since_step++;
    if (++s->ctl_n_blk >= CTL_MIN_BLOCKS && s->ctl_step_ms > 0.0) {
        /* tokens per ms of blocks against a step's one token */
        if (s->ctl_blk_tok * s->ctl_step_ms < s->ctl_blk_ms) {
            s->ctl_plain_left = s->ctl_backoff;
            if (s->ctl_backoff < CTL_RUN_MAX) s->ctl_backoff *= 2;
            s->stats.fallbacks++;
            return;
        }
        if (s->ctl_n_blk >= 4 * CTL_MIN_BLOCKS)
            s->ctl_backoff = CTL_RUN_MIN;   /* blocks pay: forget the losses */
    }
    if (s->ctl_since_step >= CTL_REMEASURE) {
        s->ctl_plain_left = 1;
        s->ctl_remeasure = true;
    }
}

void vv_spec_pause(vv_spec_t* s) {
    if (!s || s->ctl_plain_left > 0) return;
    s->ctl_plain_left = CTL_RUN_MIN;
    s->ctl_remeasure = false;
}

const vv_taps_t* vv_spec_step_taps(vv_spec_t* s) {
    return s ? &s->step_taps : NULL;
}

int vv_spec_block(const vv_spec_t* s) { return s ? s->Bv : 0; }

void vv_spec_limit_rows(vv_spec_t* s, int rows) {
    if (!s || rows < 2) return;
    for (int i = 0; i < 2; i++)
        if (s->bv_cand[i] > rows) s->bv_cand[i] = rows;
    if (s->Bv > rows) s->Bv = rows;
    if (s->bv_cand[0] == s->bv_cand[1]) s->bv_auto = false;
}

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
 *   VV_SPEC_LOG=path   append "p anchor drafts..." per cycle to `path`, for
 *                      checking the drafter against its trainer.
 */
static bool s_debug, s_profile;
static const char* s_log;
static vv_once_t s_env_once = VV_ONCE_INIT;
static void env_probe(void) {
    const char* e = getenv("VV_SPEC_DEBUG");
    s_debug = e && e[0] == '1';
    e = getenv("VV_SPEC_PROFILE");
    s_profile = e && e[0] == '1';
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

/* An INT4 projection of M rows on the tensor-core GEMM. */
static vv_status_t dlin_gemm(vv_spec_t* s, const void* x, const dlin_t* W,
                             void* y, int M, int N, int K, void* stream) {
    return vv_w4a16_gemm_dev(x, W->packed, W->sz, NULL, y, s->gemm_ws,
                             s->gemm_ws_bytes, M, N, K, DRAFT_Q_GROUP, stream);
}

/* y[M][N] = x[M][K] . W^T in W's format. A block's rows (and a few context
 * rows) of INT4 go through the tensor-core GEMV, which reads the weight once
 * for up to 16 rows; more rows, or where it declines, the GEMM. */
static vv_status_t dlin(vv_spec_t* s, const void* x, const dlin_t* W, void* y,
                        int M, int N, int K, void* stream) {
    if (W->packed) {
        if (M <= VV_W4A16_MV_MAX_ROWS) {
            vv_w4a16_proj_t p;
            p.packed = W->packed;
            p.sz = W->sz;
            p.bias = NULL;
            p.y = y;
            p.N = N;
            const vv_status_t st = vv_w4a16_mv_dev(x, M, &p, 1, K,
                                                   DRAFT_Q_GROUP, stream);
            if (st != VV_ERR_UNSUPPORTED) return st;
        }
        return dlin_gemm(s, x, W, y, M, N, K, stream);
    }
    return vv_gemm_fp16_dev(x, W->w, y, M, N, K, 1.0f, 0.0f, stream);
}

/* Projections of the same rows (q/k/v, gate/up, k/v): one launch of the
 * tensor-core GEMV when they are all INT4, else one dlin each. */
static vv_status_t dlin_group(vv_spec_t* s, const void* x,
                              const dlin_t* const* W, void* const* y,
                              const int* N, int n, int M, int K,
                              void* stream) {
    bool fuse = M <= VV_W4A16_MV_MAX_ROWS && n <= 3;
    for (int i = 0; i < n && fuse; i++) fuse = W[i]->packed != NULL;
    if (fuse) {
        vv_w4a16_proj_t p[3];
        for (int i = 0; i < n; i++) {
            p[i].packed = W[i]->packed;
            p[i].sz = W[i]->sz;
            p[i].bias = NULL;
            p[i].y = y[i];
            p[i].N = N[i];
        }
        const vv_status_t st = vv_w4a16_mv_dev(x, M, p, n, K, DRAFT_Q_GROUP,
                                               stream);
        if (st != VV_ERR_UNSUPPORTED) return st;
    }
    for (int i = 0; i < n; i++) {
        const vv_status_t st = dlin(s, x, W[i], y[i], M, N[i], K, stream);
        if (st != VV_OK) return st;
    }
    return VV_OK;
}

/* Elements of the largest INT4 projection, for the dequantizing GEMM. */
static size_t largest_packed(const vv_drafter_t* d) {
    const vv_drafter_config_t* c = &d->cfg;
    const size_t H = (size_t)c->hidden_size;
    const size_t qd = (size_t)c->num_heads * c->head_dim;
    size_t m = (size_t)c->n_taps * H * H;                        /* fc */
    if (qd * H > m) m = qd * H;                                  /* q, o */
    if ((size_t)c->intermediate_size * H > m)                    /* MLP */
        m = (size_t)c->intermediate_size * H;
    const size_t conv = 2 * (size_t)c->conv_kernel *
                        (H / (size_t)c->conv_group) * H;         /* kernels */
    return conv > m ? conv : m;
}

/* A layer's K/V rows [pos, pos + n) of the drafter cache. */
static void* kv_row(void* base, int pos, int kd) {
    return (uint8_t*)base + (size_t)pos * (size_t)kd * 2;
}

/*
 * Context positions [pos0, pos0 + n) from their tap rows: fc, hidden_norm,
 * then every layer's k/v projection, k_norm and RoPE, into the cache.
 * `d_pos` set: the positions start at *d_pos instead (a cycle's rows, in a
 * replayed graph); pos0 only bounds them.
 */
static vv_status_t ctx_update_at(vv_spec_t* s, const void* rows, int pos0,
                                 int n, const int* d_pos, void* stream) {
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
        const dlin_t* W[2] = { &L->k, &L->v };
        void* Y[2] = { s->ck, s->cv };
        const int N[2] = { kd, kd };
        st = dlin_group(s, s->cxn, W, Y, N, 2, n, H, stream);
        if (st == VV_OK)
            st = vv_rmsnorm_dev(s->ck, L->kn, s->ckn, n * s->nkv, s->hd,
                                d->cfg.rms_norm_eps, stream);
        if (st == VV_OK)
            st = vv_rope_dev(s->ckn, n, s->nkv, s->hd, pos0, d_pos,
                             d->cfg.rope_theta, stream);
        if (st == VV_OK)
            st = d_pos ? vv_dev_memcpy_d2d_rows_at(s->kc[l], s->ckn,
                                                   (size_t)kd * 2, n, d_pos,
                                                   stream)
                       : vv_dev_memcpy_d2d(kv_row(s->kc[l], pos0, kd), s->ckn,
                                           (size_t)n * kd * 2, stream);
        if (st == VV_OK)
            st = d_pos ? vv_dev_memcpy_d2d_rows_at(s->vc[l], s->cv,
                                                   (size_t)kd * 2, n, d_pos,
                                                   stream)
                       : vv_dev_memcpy_d2d(kv_row(s->vc[l], pos0, kd), s->cv,
                                           (size_t)n * kd * 2, stream);
    }
    return st;
}

static vv_status_t ctx_update(vv_spec_t* s, const void* rows, int pos0, int n,
                              void* stream) {
    return ctx_update_at(s, rows, pos0, n, NULL, stream);
}

vv_status_t vv_spec_push_step(vv_spec_t* s) {
    if (!s) return VV_ERR_NULL_PTR;
    const vv_status_t st = ctx_update(s, s->step_taps.buf, s->ctx_len, 1,
                                      s->stream);
    if (st == VV_OK) {
        s->ctx_len++;
        s->stats.steps++;
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
 * update, which starts at p. Positions, the length and the anchor are read
 * on the device (d_in), so a capture of it replays for any p of its shape.
 */
static vv_status_t draft(vv_spec_t* s, const vv_inference_ctx_t* ctx,
                         void* stream) {
    const vv_drafter_t* d = s->d;
    const vv_drafter_config_t* c = &d->cfg;
    const int B = s->B, H = s->H, p = s->ctx_len;
    const int qd = s->nh * s->hd, kd = s->nkv * s->hd;
    const int dyn_ld = 2 * c->conv_kernel * s->G;
    const int half = c->conv_kernel * s->G;
    const int32_t* anchor = s->d_in + IN_ANCHOR;
    const int* d_ctx = s->d_in + IN_CTX;
    if (p + B > s->max_pos + s->B) return VV_ERR_OVERFLOW;

    vv_status_t st = vv_dflash_block_ids_dev(anchor, c->mask_token_id, B,
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
        {
            const dlin_t* W[3] = { &L->q, &L->k, &L->v };
            void* Y[3] = { s->q, s->kk, s->vv };
            const int N[3] = { qd, kd, kd };
            OK(dlin_group(s, s->xc, W, Y, N, 3, B, H, stream));
        }
        OK(vv_rmsnorm_dev(s->q, L->qn, s->qn, B * s->nh, s->hd, c->rms_norm_eps, stream));
        OK(vv_rmsnorm_dev(s->kk, L->kn, s->kn, B * s->nkv, s->hd, c->rms_norm_eps, stream));
        OK(vv_rope_dev(s->qn, B, s->nh, s->hd, p, d_ctx, c->rope_theta, stream));
        OK(vv_rope_dev(s->kn, B, s->nkv, s->hd, p, d_ctx, c->rope_theta, stream));
        OK(vv_dev_memcpy_d2d_rows_at(s->kc[l], s->kn, (size_t)kd * 2, B, d_ctx,
                                     stream));
        OK(vv_dev_memcpy_d2d_rows_at(s->vc[l], s->vv, (size_t)kd * 2, B, d_ctx,
                                     stream));
        if (s->block_attn)
            OK(vv_attn_block_dev(s->qn, &view, s->att, s->nh, B, p + B,
                                 s->d_in + IN_CTX_END, s->attn_ws, stream));
        else
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
        {
            const dlin_t* W[2] = { &L->gate, &L->up };
            void* Y[2] = { s->gate, s->up };
            const int N[2] = { s->I, s->I };
            OK(dlin_group(s, s->xc, W, Y, N, 2, B, H, stream));
        }
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
        OK(vv_gather_i32_dev(d->vmap, d->Vd, s->tk_i, (B - 1) * s->topk,
                             stream));
    OK(vv_gemm_fp16_dev((const uint8_t*)s->xn + row, d->hproj, s->hp, B - 1,
                        s->rank, H, 1.0f, 0.0f, stream));
    OK(vv_dflash_walk_dev(s->hp, s->tk_v, s->tk_i, d->pred, d->succ,
                          anchor, B - 1, s->topk, s->rank, s->draft,
                          stream));
#undef OK
    return st;
}

/*
 * One cycle on the GPU. The target's cache holds [0, p) and `anchor` is the
 * token at p, not yet fed. The drafter proposes B - 1 tokens; the target
 * runs the B rows [anchor, drafts] -- each exactly as its decode step would
 * (exact), or with tensor-core attention (fast) -- and gives its own token
 * after each; the drafts it agrees with are kept, then its next token. The
 * B rows' taps go into the drafter's context at p..p+B-1 at once: the kept
 * ones are its new context, the rest lie past it and the next draft pass
 * writes over them. Both caches end at p + kept + 1, the last produced token
 * the new anchor.
 *
 * Everything that moves from one cycle to the next -- the anchor, both
 * lengths -- the body reads from d_in, which it fills from pinned memory
 * first; the tokens and their count go back to pinned memory last. So the
 * body can be captured once and replayed until the attention launches
 * change shape (every 1024 positions), and a cycle is one graph launch
 * instead of ~600 kernel launches: eager launches cost a Windows (WDDM)
 * 3070 about 1.7x the GPU time of a 1.5B step, a Linux 3090 about 6%.
 */
static vv_status_t cycle_body(vv_inference_ctx_t* ctx, vv_spec_t* s,
                              int backend, double* draft_ms, void* stream) {
    vv_kv_cache_t* kv = ctx->kv_cache;
    const int B = s->Bv, p = kv->current_len;
    const int Ht = ctx->model->config.llm.hidden_size;
    int32_t* in_pin = s->h_pin + IN_PIN;
    const double t0 = vv_time_ms();
#define OK(x) if (st == VV_OK) st = (x)
    vv_status_t st = vv_dev_memcpy_h2d(s->d_in, in_pin, IN_N * 4, stream);
    OK(draft(s, ctx, stream));
    /* VV_SPEC_PROFILE (launched, never captured): the draft pass alone. */
    if (draft_ms) {
        OK(vv_dev_stream_sync(stream));
        *draft_ms = vv_time_ms() - t0;
    }
    /* The block to verify: [anchor, drafts]. */
    OK(vv_dev_memcpy_d2d(s->ids, s->d_in + IN_ANCHOR, 4, stream));
    OK(vv_dev_memcpy_d2d(s->ids + 1, s->draft, (size_t)(B - 1) * 4, stream));
    OK(vv_embedding_dev(ctx->embed_table_gpu, s->ids, s->vh, B, Ht, stream));
    if (st == VV_OK) {
        s->taps.on_chunk = NULL;
        s->taps.row_base = p;           /* position p lands at row 0 */
        vv_verify_opts_t vo;
        vo.d_pos = s->d_in + IN_POS;
        /* fa1's rows run its one-row kernel row after row, on the host's
         * lengths: its cycles are launched, never replayed (below). */
        vo.d_next = backend == VV_ATTN_FA1 ? NULL : s->d_in + IN_NEXT;
        vo.attn_backend = backend;
        vo.fast = !s->exact;
        st = vv_decoder_verify(ctx->model, s->vh, B, kv, ctx->layer_pool,
                               ctx->workspace, ctx->workspace_size, stream,
                               ctx->transfer_stream, &s->taps, &vo);
    }
    OK(vv_rmsnorm_dev(s->vh, ctx->final_norm_gpu, s->vn, B, Ht,
                      ctx->model->config.llm.rms_norm_eps, stream));
    OK(vv_lm_head_rows_dev(s->vn, ctx->lm_head_gpu, s->logits, B, s->V, Ht,
                           stream));
    OK(vv_argmax_rows_dev(s->logits, B, s->V, s->am, s->post, stream));
    OK(vv_dflash_accept_dev(s->draft, s->post, B - 1, s->out, s->n_out,
                            stream));
    OK(ctx_update_at(s, s->taps.buf, s->ctx_len, B, s->d_in + IN_CTX, stream));
    OK(vv_dev_memcpy_d2h(s->h_pin, s->out, (size_t)B * 4, stream));
    OK(vv_dev_memcpy_d2h(s->h_pin + B, s->n_out, 4, stream));
#undef OK
    return st;
}

/* VV_SPEC_GRAPH=0: every cycle launches its kernels, for comparison. */
static bool s_graph_off;
static vv_once_t s_graph_once = VV_ONCE_INIT;
static void graph_probe(void) {
    const char* e = getenv("VV_SPEC_GRAPH");
    s_graph_off = e && e[0] == '0';
}

vv_status_t vv_spec_cycle(vv_inference_ctx_t* ctx, vv_spec_t* s,
                          int32_t anchor, int32_t* out, int* n_out) {
    if (!ctx || !s || !out || !n_out) return VV_ERR_NULL_PTR;
    *n_out = 0;
    vv_kv_cache_t* kv = ctx->kv_cache;
    const int B = s->Bv, p = kv->current_len;     /* rows checked */
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    void* stream = ctx->compute_stream;
    if (p != s->ctx_len) {
        VV_LOG_E("spec: drafter context at %d, target cache at %d",
                 s->ctx_len, p);
        return VV_ERR_INVALID_ARG;
    }
    if (p + B > kv->max_seq_len || p + B > s->max_pos) return VV_ERR_OVERFLOW;
    const double t0 = vv_time_ms();
    vv_status_t st = VV_OK;
    if (kv->pool) st = vv_kv_cache_reserve_wait(kv, p + B, stream);
    if (st != VV_OK) return st;

    /* The rows' attention: the cache's own (exact), or flashinfer's, which
     * reads each K/V tile once for two or three rows (fast). */
    const int backend = s->exact
        ? kv->attn_backend
        : vv_attn_resolve(VV_ATTN_FLASHINFER, kv->format, kv->pool != NULL,
                          llm->num_attention_heads, llm->num_key_value_heads,
                          llm->head_dim);
    int32_t* in_pin = s->h_pin + IN_PIN;
    in_pin[IN_ANCHOR] = anchor;
    in_pin[IN_POS] = p;
    in_pin[IN_NEXT] = p + 1;
    in_pin[IN_CTX] = s->ctx_len;
    in_pin[IN_CTX_END] = s->ctx_len + s->B;

    vv_once(&s_env_once, env_probe);
    vv_once(&s_graph_once, graph_probe);
    const bool graph = !s_graph_off && !s->graph_off && !s_debug &&
                       !s_profile && backend != VV_ATTN_FA1 &&
                       vv_pipeline_graph_ok(ctx);
    const int slot = B == s->bv_cand[1] ? 1 : 0;
    if (graph) {
        const int key[4] = {
            B, backend,
            vv_attn_decode_shape(backend, llm->num_attention_heads,
                                 llm->num_key_value_heads, p + B),
            vv_attn_block_shape(s->nh, s->nkv, s->B, s->ctx_len + s->B)
        };
        const bool same_key = memcmp(key, s->graph_key[slot], sizeof(key)) == 0;
        /* A capture that failed is tried again at the next key or the next
         * sequence, not every cycle: another slot freeing memory mid-capture
         * (vv_dev_graph_end) should not cost this one its graphs for good. */
        if (!(same_key && (s->graph[slot] || s->graph_bad[slot]))) {
            if (s->graph[slot]) {
                vv_dev_graph_destroy(s->graph[slot]);
                s->graph[slot] = NULL;
            }
            void* g = NULL;
            vv_status_t cs = vv_dev_graph_begin(stream);
            if (cs == VV_OK) {
                cs = cycle_body(ctx, s, backend, NULL, stream);
                const vv_status_t es = vv_dev_graph_end(stream, &g);
                if (cs == VV_OK) cs = es;
            }
            kv->current_len = p;        /* a capture runs nothing */
            memcpy(s->graph_key[slot], key, sizeof(key));
            s->graph_bad[slot] = !(cs == VV_OK && g);
            if (!s->graph_bad[slot]) {
                s->graph[slot] = g;
            } else {
                if (g) vv_dev_graph_destroy(g);
                VV_LOG_W("spec: cannot capture a cycle (%s), launching each "
                         "kernel", vv_status_str(cs));
            }
        }
    }
    double draft_ms = 0.0;
    if (graph && s->graph[slot])
        st = vv_dev_graph_launch(s->graph[slot], stream);
    else
        st = cycle_body(ctx, s, backend, s_profile ? &draft_ms : NULL, stream);
    if (st == VV_OK) st = vv_dev_stream_sync(stream);
    if (st != VV_OK) {
        kv->current_len = p;
        return st;
    }

    const int n = s->h_pin[B];
    if (n < 1 || n > B) return VV_ERR_INVALID_ARG;
    /* Fed: anchor and the n - 1 kept drafts, positions p .. p + n - 1. */
    kv->current_len = p + n;
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
    s->stats.draft_ms += draft_ms;
    s->stats.verify_ms += vv_time_ms() - t0 - draft_ms;
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
