/**
 * @file vae_gpu.c
 * @brief Batched, streaming Conv-VAE encoder on the accelerator.
 *
 * What used to happen per 60 s segment and encoder: ~110 cudaMalloc/cudaFree
 * pairs (each cudaFree synchronises the whole device, stalling every other
 * slot's decode), a stream_sync after every stage, a private stream created
 * and destroyed per call, and a host round trip for the latents. What happens
 * now: nothing but kernel launches on the caller's stream, reading weights
 * uploaded once per device and scratch sized once per arena.
 *
 * Layout: channel-first [C][ld] with every item of the batch packed along
 * time. Per block of a stage that is: one RMS-statistics launch, one tail
 * save, one fused norm + depthwise conv + gamma-residual launch, one more
 * RMS-statistics launch and two GEMMs per FFN tile (norm folded into the
 * first's operand load, bias + GELU into its epilogue, bias + gamma-residual
 * into the second's).
 */

#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vae_plan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Weights ───────────────────────────────────────────────────────────── */

typedef struct {
    const void *norm_w, *conv_w, *conv_b, *gamma;
    const void *ffn_norm_w, *ffn_gamma, *l1_w, *l1_b, *l2_w, *l2_b;
    int hidden;
} vae_block_dev_t;

struct vv_vae_weights {
    const vv_conv_vae_encoder_t* enc;
    vae_plan_t      plan;
    void*           blob;
    size_t          bytes;
    const void     *stem_w, *stem_b, *head_w, *head_b;
    const void     *ds_w[VAE_MAX_STAGES], *ds_b[VAE_MAX_STAGES];
    vae_block_dev_t* blocks;           /**< flat, in layer order           */
    int             stage_first[VAE_MAX_STAGES];
};

#define VAE_ALIGN 256

static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

typedef struct {
    const vv_tensor_t* t;
    const void**       dst;
} vae_upload_t;

static size_t tensor_elems(const vv_tensor_t* t) {
    return (t && t->data) ? t->size_bytes / sizeof(float) : 0;
}

/**
 * @brief Every tensor of the encoder, paired with where its device copy goes.
 * Returns the count; `list` may be NULL to count only.
 */
static int collect_tensors(const vv_conv_vae_encoder_t* e,
                           struct vv_vae_weights* w, vae_upload_t* list) {
    int n = 0;
#define ADD(tensor, slot)                                                     \
    do {                                                                      \
        if ((tensor).data) {                                                  \
            if (list) { list[n].t = &(tensor); list[n].dst = (slot); }        \
            n++;                                                              \
        }                                                                     \
    } while (0)
    ADD(e->input_conv.weight, w ? &w->stem_w : NULL);
    ADD(e->input_conv.bias,   w ? &w->stem_b : NULL);
    int bi = 0;
    for (int s = 0; s < e->n_stages; s++) {
        for (int b = 0; b < e->stages[s].n_blocks; b++, bi++) {
            const vv_encoder_block_t* k = &e->stages[s].blocks[b];
            vae_block_dev_t* d = w ? &w->blocks[bi] : NULL;
            ADD(k->mixer_norm_weight,  d ? &d->norm_w : NULL);
            ADD(k->mixer_conv.weight,  d ? &d->conv_w : NULL);
            ADD(k->mixer_conv.bias,    d ? &d->conv_b : NULL);
            ADD(k->mixer_layer_scale,  d ? &d->gamma : NULL);
            ADD(k->ffn_norm_weight,    d ? &d->ffn_norm_w : NULL);
            ADD(k->ffn_layer_scale,    d ? &d->ffn_gamma : NULL);
            ADD(k->ffn_linear1_weight, d ? &d->l1_w : NULL);
            ADD(k->ffn_linear1_bias,   d ? &d->l1_b : NULL);
            ADD(k->ffn_linear2_weight, d ? &d->l2_w : NULL);
            ADD(k->ffn_linear2_bias,   d ? &d->l2_b : NULL);
        }
        ADD(e->stages[s].downsample.weight, w ? &w->ds_w[s] : NULL);
        ADD(e->stages[s].downsample.bias,   w ? &w->ds_b[s] : NULL);
    }
    ADD(e->proj_mean.weight, w ? &w->head_w : NULL);
    ADD(e->proj_mean.bias,   w ? &w->head_b : NULL);
#undef ADD
    return n;
}

static size_t weights_layout(const vv_conv_vae_encoder_t* e, vae_upload_t* list,
                             int n, size_t* max_elems) {
    size_t off = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        const size_t ne = tensor_elems(list[i].t);
        if (ne > mx) mx = ne;
        off = align_up(off + ne * 2, VAE_ALIGN);
    }
    (void)e;
    if (max_elems) *max_elems = mx;
    return off;
}

size_t vv_vae_weights_bytes(const vv_conv_vae_encoder_t* e) {
    if (!e) return 0;
    const int n = collect_tensors(e, NULL, NULL);
    vae_upload_t* list = (vae_upload_t*)vv_alloc((size_t)(n > 0 ? n : 1)
                                                 * sizeof(vae_upload_t));
    if (!list) return 0;
    struct vv_vae_weights tmp;
    memset(&tmp, 0, sizeof(tmp));
    /* Only the tensor list is wanted; point the slots at a dummy. */
    int total_blocks = 0;
    for (int s = 0; s < e->n_stages; s++) total_blocks += e->stages[s].n_blocks;
    tmp.blocks = (vae_block_dev_t*)vv_alloc((size_t)(total_blocks > 0 ? total_blocks : 1)
                                            * sizeof(vae_block_dev_t));
    if (!tmp.blocks) { vv_free(list); return 0; }
    collect_tensors(e, &tmp, list);
    const size_t bytes = weights_layout(e, list, n, NULL);
    vv_free(tmp.blocks);
    vv_free(list);
    return bytes;
}

vv_status_t vv_vae_weights_upload(const vv_conv_vae_encoder_t* e, void* stream,
                                  vv_vae_weights_t** out) {
    if (!e || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!vv_conv_vae_complete(e)) return VV_ERR_WEIGHT_MISSING;

    struct vv_vae_weights* w =
        (struct vv_vae_weights*)vv_alloc(sizeof(struct vv_vae_weights));
    if (!w) return VV_ERR_OUT_OF_MEMORY;
    memset(w, 0, sizeof(*w));
    w->enc = e;

    vv_status_t s = vae_plan_build(e, &w->plan);
    if (s != VV_OK) { vv_free(w); return s; }

    int total_blocks = 0;
    for (int st = 0; st < e->n_stages; st++) {
        w->stage_first[st] = total_blocks;
        total_blocks += e->stages[st].n_blocks;
    }
    w->blocks = (vae_block_dev_t*)vv_alloc((size_t)(total_blocks > 0 ? total_blocks : 1)
                                           * sizeof(vae_block_dev_t));
    if (!w->blocks) { vv_free(w); return VV_ERR_OUT_OF_MEMORY; }
    memset(w->blocks, 0, (size_t)(total_blocks > 0 ? total_blocks : 1)
                         * sizeof(vae_block_dev_t));
    {
        int bi = 0;
        for (int st = 0; st < e->n_stages; st++)
            for (int b = 0; b < e->stages[st].n_blocks; b++, bi++)
                w->blocks[bi].hidden =
                    (int)e->stages[st].blocks[b].ffn_linear1_weight.shape[0];
    }

    const int n = collect_tensors(e, w, NULL);
    vae_upload_t* list = (vae_upload_t*)vv_alloc((size_t)n * sizeof(vae_upload_t));
    if (!list) { vv_free(w->blocks); vv_free(w); return VV_ERR_OUT_OF_MEMORY; }
    collect_tensors(e, w, list);

    for (int i = 0; i < n; i++) {
        if (list[i].t->dtype != VV_DTYPE_F32) {
            VV_LOG_E("conv_vae: encoder weights must be FP32 on the host");
            vv_free(list); vv_free(w->blocks); vv_free(w);
            return VV_ERR_MODEL_FORMAT;
        }
    }

    size_t max_elems = 0;
    w->bytes = weights_layout(e, list, n, &max_elems);

    void* staging = NULL;
    s = vv_dev_alloc(&w->blob, w->bytes);
    if (s == VV_OK) s = vv_dev_alloc(&staging, max_elems * sizeof(float));
    if (s == VV_OK) {
        size_t off = 0;
        for (int i = 0; i < n && s == VV_OK; i++) {
            const size_t ne = tensor_elems(list[i].t);
            void* dst = (char*)w->blob + off;
            s = vv_dev_memcpy_h2d(staging, list[i].t->data, ne * sizeof(float),
                                  stream);
            if (s == VV_OK)
                s = vv_vae_f32_to_f16_dev((const float*)staging, dst,
                                          (int64_t)ne, stream);
            /* One staging buffer: the next copy may not start early. */
            if (s == VV_OK) s = vv_dev_stream_sync(stream);
            *list[i].dst = dst;
            off = align_up(off + ne * 2, VAE_ALIGN);
        }
    }
    if (staging) vv_dev_free(staging);
    vv_free(list);
    if (s != VV_OK) {
        if (w->blob) vv_dev_free(w->blob);
        vv_free(w->blocks);
        vv_free(w);
        return s;
    }
    *out = w;
    return VV_OK;
}

void vv_vae_weights_free(vv_vae_weights_t* w) {
    if (!w) return;
    if (w->blob) vv_dev_free(w->blob);
    if (w->blocks) vv_free(w->blocks);
    vv_free(w);
}

const vv_conv_vae_encoder_t* vv_vae_weights_encoder(const vv_vae_weights_t* w) {
    return w ? w->enc : NULL;
}

/* ─── State ─────────────────────────────────────────────────────────────── */

struct vv_vae_state {
    const vv_vae_weights_t* w;
    void*   buf;                  /**< two tail buffers, ping-pong          */
    int     have[VAE_MAX_LAYERS]; /**< context columns per layer            */
    int     cur;                  /**< which buffer holds the live tails     */
    bool    fresh;
};

size_t vv_vae_state_bytes(const vv_conv_vae_encoder_t* e) {
    vae_plan_t p;
    if (vae_plan_build(e, &p) != VV_OK) return 0;
    return 2 * p.tail_elems * 2;
}

vv_status_t vv_vae_state_create(const vv_vae_weights_t* w, vv_vae_state_t** out) {
    if (!w || !out) return VV_ERR_NULL_PTR;
    struct vv_vae_state* st =
        (struct vv_vae_state*)vv_alloc(sizeof(struct vv_vae_state));
    if (!st) return VV_ERR_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));
    st->w = w;
    st->fresh = true;
    const vv_status_t s = vv_dev_alloc(&st->buf, 2 * w->plan.tail_elems * 2);
    if (s != VV_OK) { vv_free(st); return s; }
    *out = st;
    return VV_OK;
}

void vv_vae_state_reset(vv_vae_state_t* st) {
    if (!st) return;
    st->fresh = true;
    st->cur = 0;
}

void vv_vae_state_free(vv_vae_state_t* st) {
    if (!st) return;
    if (st->buf) vv_dev_free(st->buf);
    vv_free(st);
}

static void* state_tail(const vv_vae_state_t* st, int layer, int which) {
    return (uint16_t*)st->buf + (size_t)which * st->w->plan.tail_elems
           + st->w->plan.L[layer].tail_off;
}

int vv_vae_frames(const vv_vae_weights_t* w, const vv_vae_state_t* st,
                  int64_t n_samples, bool is_final) {
    if (!w || n_samples < 0) return 0;
    const bool fresh = !st || st->fresh;
    const bool fin = !st || is_final;
    int64_t len = n_samples;
    for (int i = 0; i < w->plan.n_layers; i++) {
        const vae_layer_t* L = &w->plan.L[i];
        const int64_t have = fresh ? vae_layer_fresh_have(L) : st->have[i];
        int64_t ol, keep;
        vae_layer_step(L, have, len, fin, &ol, &keep);
        len = ol;
    }
    return (int)len;
}

/* ─── Arena ─────────────────────────────────────────────────────────────── */

/** Room for the FFN hidden tile; wider segments are tiled over time. */
#define VAE_FFN_TILE_BYTES ((size_t)32 * 1024 * 1024)

typedef struct {
    int64_t ld[VAE_MAX_STAGES];     /**< packed-length bound per stage     */
    size_t  act_elems;              /**< each of the two activation buffers */
    size_t  rinv_elems;
    size_t  hid_elems;
    size_t  off_act1, off_rinv, off_hid, bytes;
} vae_arena_layout_t;

static int64_t round8(int64_t v) { return (v + 7) / 8 * 8; }

static void arena_layout(const vae_plan_t* p, int max_items,
                         int64_t max_samples, vae_arena_layout_t* l) {
    memset(l, 0, sizeof(*l));
    size_t act = 0, hid_full = 0;
    for (int s = 0; s < p->n_stages; s++) {
        /* Per item each downsample adds at most 3 columns over the exact
           ratio (context + ceil), 6 over the whole chain. */
        l->ld[s] = round8(max_samples / p->div[s] + 8 * (int64_t)max_items);
        const size_t a = (size_t)p->C[s] * (size_t)l->ld[s];
        if (a > act) act = a;
        const size_t h = (size_t)p->hidden[s] * (size_t)l->ld[s];
        if (h > hid_full) hid_full = h;
    }
    /* The head's input is the last stage's, already counted. */
    l->act_elems = act;
    l->rinv_elems = (size_t)l->ld[0] + (size_t)max_items * (size_t)(p->max_cap + 1);
    size_t hid = VAE_FFN_TILE_BYTES / 2;
    /* At least one 128-column tile of the widest FFN must fit. */
    int widest = 0;
    for (int s = 0; s < p->n_stages; s++)
        if (p->hidden[s] > widest) widest = p->hidden[s];
    if (hid < (size_t)widest * 128) hid = (size_t)widest * 128;
    /* The downsample and head convs lay their im2col tiles out in the
       same buffer. */
    for (int i = 0; i < p->n_layers; i++) {
        const vae_layer_t* L = &p->L[i];
        if (L->kind != VAE_L_DS && L->kind != VAE_L_HEAD) continue;
        const size_t kc = (size_t)L->in_ch * (size_t)L->k;
        if (hid < kc * 128) hid = kc * 128;
        const int so = (L->kind == VAE_L_DS && L->stage + 1 < p->n_stages)
                     ? L->stage + 1 : p->n_stages - 1;
        const size_t full = kc * (size_t)l->ld[so];
        if (full > hid_full) hid_full = full;
    }
    if (hid > hid_full) hid = hid_full;
    l->hid_elems = hid;

    l->off_act1 = align_up(act * 2, VAE_ALIGN);
    l->off_rinv = l->off_act1 + align_up(act * 2, VAE_ALIGN);
    l->off_hid  = l->off_rinv + align_up(l->rinv_elems * 4, VAE_ALIGN);
    l->bytes    = l->off_hid + align_up(l->hid_elems * 2, VAE_ALIGN);
}

struct vv_vae_arena {
    vae_plan_t         plan;
    vae_arena_layout_t lay;
    int                max_items;
    int64_t            max_samples;
    void*              blob;
    void*              act[2];
    float*             rinv;
    void*              hid;
    size_t             high_water;
    bool               stats;
    /* Host scratch: per item per layer, the context left after this call. */
    int*               keep;
};

size_t vv_vae_arena_bytes(const vv_conv_vae_encoder_t* e, int max_items,
                          int64_t max_samples) {
    vae_plan_t p;
    if (!e || vae_plan_build(e, &p) != VV_OK) return 0;
    if (max_items < 1) max_items = 1;
    if (max_items > VV_VAE_MAX_ITEMS) max_items = VV_VAE_MAX_ITEMS;
    vae_arena_layout_t l;
    arena_layout(&p, max_items, max_samples, &l);
    return l.bytes;
}

vv_status_t vv_vae_arena_create(const vv_conv_vae_encoder_t* e, int max_items,
                                int64_t max_samples, vv_vae_arena_t** out) {
    if (!e || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (max_items < 1 || max_samples < 1) return VV_ERR_INVALID_ARG;
    if (max_items > VV_VAE_MAX_ITEMS) max_items = VV_VAE_MAX_ITEMS;

    struct vv_vae_arena* a =
        (struct vv_vae_arena*)vv_alloc(sizeof(struct vv_vae_arena));
    if (!a) return VV_ERR_OUT_OF_MEMORY;
    memset(a, 0, sizeof(*a));
    vv_status_t s = vae_plan_build(e, &a->plan);
    if (s != VV_OK) { vv_free(a); return s; }
    a->max_items = max_items;
    a->max_samples = max_samples;
    arena_layout(&a->plan, max_items, max_samples, &a->lay);

    a->keep = (int*)vv_alloc((size_t)max_items * VAE_MAX_LAYERS * sizeof(int));
    if (!a->keep) { vv_free(a); return VV_ERR_OUT_OF_MEMORY; }

    s = vv_dev_alloc(&a->blob, a->lay.bytes);
    if (s != VV_OK) { vv_free(a->keep); vv_free(a); return s; }
    a->act[0] = a->blob;
    a->act[1] = (char*)a->blob + a->lay.off_act1;
    a->rinv   = (float*)((char*)a->blob + a->lay.off_rinv);
    a->hid    = (char*)a->blob + a->lay.off_hid;

    const char* ev = getenv("VV_ENC_STATS");
    a->stats = ev && ev[0] && ev[0] != '0';
    *out = a;
    return VV_OK;
}

void vv_vae_arena_free(vv_vae_arena_t* a) {
    if (!a) return;
    if (a->blob) vv_dev_free(a->blob);
    if (a->keep) vv_free(a->keep);
    vv_free(a);
}

int vv_vae_arena_max_items(const vv_vae_arena_t* a) {
    return a ? a->max_items : 0;
}

int64_t vv_vae_arena_max_samples(const vv_vae_arena_t* a) {
    return a ? a->max_samples : 0;
}

size_t vv_vae_arena_high_water(const vv_vae_arena_t* a) {
    return a ? a->high_water : 0;
}

/* ─── Debug statistics ──────────────────────────────────────────────────── */

/**
 * @brief Checksum a packed FP16 activation, when VV_ENC_STATS is set.
 *
 * The one place an encode synchronises, and only on request: it is how the
 * first diverging stage between two builds gets found.
 */
static void stats_dev(const vv_vae_arena_t* a, const char* tag, const void* x,
                      int C, int64_t T, int64_t ld, void* stream) {
    if (!a->stats || T <= 0) return;
    const size_t n = (size_t)C * (size_t)ld;
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    if (!h) return;
    vv_dev_stream_sync(stream);
    vv_dev_memcpy_d2h(h, x, n * 2, stream);
    vv_dev_stream_sync(stream);
    double sum = 0.0, sumsq = 0.0;
    float mx = 0.0f;
    for (int c = 0; c < C; c++)
        for (int64_t t = 0; t < T; t++) {
            const float v = vv_half_to_float(h[(size_t)c * ld + t]);
            sum += v;
            sumsq += (double)v * v;
            const float av = v < 0.0f ? -v : v;
            if (av > mx) mx = av;
        }
    const double cnt = (double)C * (double)T;
    VV_LOG_I("conv_vae/stats %-12s n=%.0f sum=%.6f rms=%.6f max=%.6f",
             tag, cnt, sum, sqrt(sumsq / cnt), (double)mx);
    vv_free(h);
}

/* ─── Encode ────────────────────────────────────────────────────────────── */

/**
 * @brief Fill one layer's descriptor from each item's input length.
 *
 * Returns the packed output length; `out_len[i]` and the arena's keep table
 * get each item's share. Nothing is committed to the states here: a failed
 * launch later leaves them as they were.
 */
static int64_t make_desc(const vv_vae_arena_t* a, int li, const vae_layer_t* L,
                         const vv_vae_item_t* items, int n,
                         const int64_t* in_len, bool own_input,
                         vv_vae_conv_desc_t* d, int64_t* out_len) {
    memset(d, 0, sizeof(*d));
    d->n = n;
    d->cap = L->cap;
    int64_t in_off = 0, out_off = 0;
    for (int i = 0; i < n; i++) {
        const vv_vae_state_t* st = items[i].state;
        const bool fresh = !st || st->fresh;
        const bool fin = !st || items[i].is_final;
        const int64_t have = fresh ? vae_layer_fresh_have(L) : st->have[li];
        int64_t ol, keep;
        vae_layer_step(L, have, in_len[i], fin, &ol, &keep);

        vv_vae_conv_item_t* it = &d->it[i];
        it->in_off = in_off;
        it->out_off = out_off;
        it->in_ptr = own_input ? items[i].audio : NULL;
        it->tail_in = (!fresh && have > 0) ? state_tail(st, li, st->cur) : NULL;
        it->tail_out = (st && !fin && keep > 0)
                     ? state_tail(st, li, st->cur ^ 1) : NULL;
        it->in_len = (int)in_len[i];
        it->out_len = (int)ol;
        it->have = (int)have;
        it->keep = (int)keep;

        a->keep[(size_t)i * VAE_MAX_LAYERS + li] = (int)keep;
        in_off += in_len[i];
        out_off += ol;
        out_len[i] = ol;
    }
    return out_off;
}

vv_status_t vv_vae_encode(const vv_vae_weights_t* w, vv_vae_arena_t* a,
                          vv_vae_item_t* items, int n, void* stream) {
    if (!w || !a || !items) return VV_ERR_NULL_PTR;
    if (n < 1 || n > a->max_items) return VV_ERR_INVALID_ARG;
    const vae_plan_t* P = &w->plan;
    if (P->n_layers != a->plan.n_layers || P->tail_elems != a->plan.tail_elems)
        return VV_ERR_SHAPE_MISMATCH;

    int64_t total = 0;
    for (int i = 0; i < n; i++) {
        if (!items[i].audio && items[i].n_samples > 0) return VV_ERR_NULL_PTR;
        if (!items[i].out) return VV_ERR_NULL_PTR;
        if (items[i].n_samples < 0 || items[i].out_ld < P->vae_dim ||
            items[i].skip_frames < 0)
            return VV_ERR_INVALID_ARG;
        if (items[i].state && items[i].state->w != w) return VV_ERR_INVALID_ARG;
        for (int j = 0; j < i; j++)
            if (items[i].state && items[i].state == items[j].state)
                return VV_ERR_INVALID_ARG;   /* one chunk per stream per call */
        total += items[i].n_samples;
    }
    if (total > a->max_samples) return VV_ERR_OVERFLOW;

    const vv_conv_vae_encoder_t* e = w->enc;
    const float eps = e->eps;
    int64_t len[VV_VAE_MAX_ITEMS], nxt[VV_VAE_MAX_ITEMS];
    for (int i = 0; i < n; i++) len[i] = items[i].n_samples;

    vv_vae_conv_desc_t d;
    vv_status_t s = VV_OK;
    size_t used_act = 0, used_hid = 0;
    int li = 0;
    int cur = 0;

    /* ── Stem: 1 → C0, reading each item's audio where it lies ── */
    const vae_layer_t* L = &P->L[li];
    int64_t T = make_desc(a, li, L, items, n, len, true, &d, nxt);
    int64_t ld = round8(T);
    if ((size_t)L->out_ch * (size_t)ld > a->lay.act_elems) return VV_ERR_OVERFLOW;
    if (T > 0)
        s = vv_vae_conv_dev(&d, NULL, 0, w->stem_w, w->stem_b, a->act[cur], ld,
                            1, L->out_ch, L->k, L->stride, false, stream);
    /* Contexts are saved even for an empty chunk: the state flips to the
       other buffer either way, so the old context has to be carried over. */
    if (s == VV_OK) s = vv_vae_tail_dev(&d, NULL, 0, 1, stream);
    if (s != VV_OK) return s;
    used_act = (size_t)L->out_ch * (size_t)ld;
    memcpy(len, nxt, sizeof(int64_t) * (size_t)n);
    stats_dev(a, "gpu/stem", a->act[cur], L->out_ch, T, ld, stream);
    li++;

    for (int st = 0; st < P->n_stages; st++) {
        const int C = P->C[st];
        const int bfirst = w->stage_first[st];
        for (int b = 0; b < e->stages[st].n_blocks; b++) {
            L = &P->L[li];
            const vae_block_dev_t* bw = &w->blocks[bfirst + b];

            /* Mixer: statistics of the input and of each item's context,
               the next context, then the fused norm + conv + residual. */
            make_desc(a, li, L, items, n, len, false, &d, nxt);
            s = vv_vae_tail_dev(&d, a->act[cur], ld, C, stream);
            if (s != VV_OK) return s;
            if (T > 0) {
                s = vv_vae_rms_dev(a->act[cur], ld, T, C, eps, a->rinv, &d,
                                   stream);
                if (s == VV_OK)
                    s = vv_vae_mixer_dev(&d, a->act[cur], a->act[cur ^ 1], ld,
                                         a->rinv, T, bw->norm_w, bw->conv_w,
                                         bw->conv_b, bw->gamma, C, L->k,
                                         stream);
                if (s != VV_OK) return s;
            }
            cur ^= 1;
            li++;

            /* FFN, tiled over time when the hidden buffer cannot take all
               of it: the columns are independent, so a tile boundary
               changes nothing. */
            if (T > 0) {
                s = vv_vae_rms_dev(a->act[cur], ld, T, C, eps, a->rinv, NULL,
                                   stream);
                if (s != VV_OK) return s;
                int64_t tile = (int64_t)(a->lay.hid_elems / (size_t)bw->hidden);
                tile = tile / 128 * 128;
                if (tile < 128) return VV_ERR_OVERFLOW;
                for (int64_t p0 = 0; p0 < T; p0 += tile) {
                    const int64_t pc = (T - p0 < tile) ? T - p0 : tile;
                    const int64_t ldh = round8(pc);
                    if ((size_t)bw->hidden * (size_t)ldh > a->lay.hid_elems)
                        return VV_ERR_OVERFLOW;
                    if ((size_t)bw->hidden * (size_t)ldh > used_hid)
                        used_hid = (size_t)bw->hidden * (size_t)ldh;
                    void* x = (uint16_t*)a->act[cur] + p0;
                    s = vv_vae_gemm_nn_dev(VV_VAE_EPI_BIAS_GELU, bw->l1_w, x, ld,
                                           a->hid, ldh, bw->hidden, C, (int)pc,
                                           bw->l1_b, NULL, a->rinv + p0,
                                           bw->ffn_norm_w, 0, stream);
                    if (s == VV_OK)
                        s = vv_vae_gemm_nn_dev(VV_VAE_EPI_BIAS_RESID, bw->l2_w,
                                               a->hid, ldh, x, ld, C,
                                               bw->hidden, (int)pc, bw->l2_b,
                                               bw->ffn_gamma, NULL, NULL,
                                               0, stream);
                    if (s != VV_OK) return s;
                }
            }
        }

        if (vv_vae_stage_downsamples(&e->stages[st])) {
            L = &P->L[li];
            const int64_t T2 = make_desc(a, li, L, items, n, len, false, &d, nxt);
            const int64_t ld2 = round8(T2);
            if ((size_t)L->out_ch * (size_t)ld2 > a->lay.act_elems)
                return VV_ERR_OVERFLOW;
            if (T2 > 0) {
                /* A tiled GEMM over im2col columns in the FFN's hidden
                   buffer, summing in the direct kernel's order; the direct
                   kernel spent two thirds of the encoder here, one scalar
                   load pair per multiply-add. */
                const int Kc = C * L->k;
                int64_t tile = (int64_t)(a->lay.hid_elems / (size_t)Kc);
                tile = tile >= 128 ? tile / 128 * 128 : tile / 8 * 8;
                if (tile < 8) return VV_ERR_OVERFLOW;
                for (int64_t p0 = 0; p0 < T2 && s == VV_OK; p0 += tile) {
                    const int64_t pc = (T2 - p0 < tile) ? T2 - p0 : tile;
                    const int64_t ldc = round8(pc);
                    if ((size_t)Kc * (size_t)ldc > used_hid)
                        used_hid = (size_t)Kc * (size_t)ldc;
                    s = vv_vae_im2col_dev(&d, a->act[cur], ld, C, L->k,
                                          L->stride, p0, (int)pc, a->hid, ldc,
                                          stream);
                    if (s == VV_OK)
                        s = vv_vae_conv_gemm_dev(NULL, w->ds_w[st], a->hid, ldc,
                                                 w->ds_b[st],
                                                 (uint16_t*)a->act[cur ^ 1] + p0,
                                                 ld2, L->out_ch, Kc, p0,
                                                 (int)pc, 0, stream);
                }
                if (s != VV_OK) return s;
            }
            s = vv_vae_tail_dev(&d, a->act[cur], ld, C, stream);
            if (s != VV_OK) return s;
            if ((size_t)L->out_ch * (size_t)ld2 > used_act)
                used_act = (size_t)L->out_ch * (size_t)ld2;
            cur ^= 1;
            T = T2;
            ld = ld2;
            memcpy(len, nxt, sizeof(int64_t) * (size_t)n);
            li++;
        }
        if (a->stats) {
            char tag[32];
            const int ch = (st + 1 < P->n_stages) ? P->C[st + 1] : C;
            snprintf(tag, sizeof(tag), "gpu/stage%d", st + 1);
            stats_dev(a, tag, a->act[cur], ch, T, ld, stream);
        }
    }

    /* ── Head: C → vae_dim, written as [frame][vae_dim] rows per item ── */
    L = &P->L[li];
    make_desc(a, li, L, items, n, len, false, &d, nxt);
    for (int i = 0; i < n; i++) {
        d.it[i].out_ptr = items[i].out;
        d.it[i].out_ld = items[i].out_ld;
        d.it[i].skip = items[i].skip_frames;
    }
    {
        /* The same im2col GEMM as the downsamples, writing each item's
           frames as rows. The direct kernel ran one thread per output over
           in_ch * k = 14336 products on a 64-128 block grid. */
        int64_t To = 0;
        for (int i = 0; i < n; i++) To += nxt[i];
        const int Kc = L->in_ch * L->k;
        int64_t tile = (int64_t)(a->lay.hid_elems / (size_t)Kc);
        tile = tile >= 128 ? tile / 128 * 128 : tile / 8 * 8;
        if (To > 0 && tile < 8) return VV_ERR_OVERFLOW;
        for (int64_t p0 = 0; p0 < To && s == VV_OK; p0 += tile) {
            const int64_t pc = (To - p0 < tile) ? To - p0 : tile;
            const int64_t ldc = round8(pc);
            if ((size_t)Kc * (size_t)ldc > used_hid)
                used_hid = (size_t)Kc * (size_t)ldc;
            s = vv_vae_im2col_dev(&d, a->act[cur], ld, L->in_ch, L->k,
                                  L->stride, p0, (int)pc, a->hid, ldc, stream);
            if (s == VV_OK)
                s = vv_vae_conv_gemm_dev(&d, w->head_w, a->hid, ldc, w->head_b,
                                         NULL, 0, L->out_ch, Kc, p0, (int)pc,
                                         0, stream);
        }
    }
    if (s == VV_OK) s = vv_vae_tail_dev(&d, a->act[cur], ld, L->in_ch, stream);
    if (s != VV_OK) return s;

    /* Everything is enqueued: commit the new contexts. */
    for (int i = 0; i < n; i++) {
        items[i].n_frames = (int)nxt[i];
        vv_vae_state_t* sst = items[i].state;
        if (!sst) continue;
        if (items[i].is_final) {
            vv_vae_state_reset(sst);
            continue;
        }
        for (int l = 0; l < P->n_layers; l++)
            sst->have[l] = a->keep[(size_t)i * VAE_MAX_LAYERS + l];
        sst->cur ^= 1;
        sst->fresh = false;
    }

    {
        /* What a tight layout of this call's buffers would need. */
        const size_t used = 2 * align_up(used_act * 2, VAE_ALIGN)
                          + align_up(((size_t)round8(total) + (size_t)n *
                                      (size_t)(P->max_cap + 1)) * 4, VAE_ALIGN)
                          + align_up(used_hid * 2, VAE_ALIGN);
        if (used > a->high_water) a->high_water = used;
    }
    return VV_OK;
}
