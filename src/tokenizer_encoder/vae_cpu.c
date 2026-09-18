/**
 * @file vae_cpu.c
 * @brief Conv-VAE encoder on the CPU: time-tiled, streaming, OpenMP.
 *
 * The old CPU path processed the whole clip at once — one scalar thread and
 * O(clip) memory, 22 GB for the stage-0 FFN buffer of a 30-minute file — and
 * on a CUDA build it quietly ran on the GPU even under --cpu. This one never
 * touches an accelerator and holds a bounded amount of memory.
 *
 * Layout is time-major, [rows][channels], which is what the packed GEMM
 * wants: the FFNs are plain X @ W^T, and a strided convolution whose kernel
 * is a whole number of strides (k = 2s here) is a GEMM over overlapping
 * windows of the input rows, copied out (im2col) and multiplied against the
 * kernel reordered to [out][tap][in].
 *
 * The audio goes in as fixed tiles, but the stages do not run per tile: each
 * stage buffers its input until it has enough rows for its GEMMs to be worth
 * running (a stage-6 FFN reads 67 MB of weights, so it should not do that
 * for 15 rows), then streams that chunk through with the same carried-context
 * rule the GPU uses. Any chunking gives the same result, so this is free.
 */

#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"
#include "vae_plan.h"

#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Audio samples pushed through the stem at a time. */
#define CPU_TILE_SAMPLES 48000
/** Rows a stage waits for before it runs, unless the stream is ending. */
#define CPU_MIN_ROWS 256

/* ─── Weights ───────────────────────────────────────────────────────────── */

typedef struct {
    uint16_t* l1;      /**< [hidden][C]        */
    uint16_t* l1b;     /**< [hidden] or NULL   */
    uint16_t* l2;      /**< [C][hidden]        */
    uint16_t* l2b;     /**< [C] or NULL        */
} cpu_block_t;

struct vv_vae_cpu_weights {
    cpu_block_t* blocks;
    int          n_blocks;
    uint16_t*    ds_w[VAE_MAX_STAGES];   /**< [out][k][in] */
    uint16_t*    ds_b[VAE_MAX_STAGES];
    uint16_t*    head_w;                 /**< [out][k][in] */
    uint16_t*    head_b;
};

static uint16_t* to_f16(const vv_tensor_t* t) {
    if (!t->data) return NULL;
    const size_t n = t->size_bytes / sizeof(float);
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    if (!h) return NULL;
    const float* f = (const float*)t->data;
    for (size_t i = 0; i < n; i++) h[i] = vv_float_to_half(f[i]);
    return h;
}

/** @brief [out][in][k] -> [out][k][in], FP16: im2col rows are tap-major. */
static uint16_t* conv_to_f16_tapmajor(const vv_tensor_t* t) {
    if (!t->data || t->ndim < 3) return NULL;
    const int oc = (int)t->shape[0], ic = (int)t->shape[1], k = (int)t->shape[2];
    uint16_t* h = (uint16_t*)vv_alloc((size_t)oc * ic * k * 2);
    if (!h) return NULL;
    const float* f = (const float*)t->data;
    for (int o = 0; o < oc; o++)
        for (int c = 0; c < ic; c++)
            for (int kk = 0; kk < k; kk++)
                h[((size_t)o * k + kk) * ic + c] =
                    vv_float_to_half(f[((size_t)o * ic + c) * k + kk]);
    return h;
}

void vv_vae_cpu_weights_free(struct vv_vae_cpu_weights* cw) {
    if (!cw) return;
    if (cw->blocks) {
        for (int i = 0; i < cw->n_blocks; i++) {
            vv_free(cw->blocks[i].l1);
            vv_free(cw->blocks[i].l1b);
            vv_free(cw->blocks[i].l2);
            vv_free(cw->blocks[i].l2b);
        }
        vv_free(cw->blocks);
    }
    for (int s = 0; s < VAE_MAX_STAGES; s++) {
        vv_free(cw->ds_w[s]);
        vv_free(cw->ds_b[s]);
    }
    vv_free(cw->head_w);
    vv_free(cw->head_b);
    vv_free(cw);
}

static vv_status_t cpu_weights_build(vv_conv_vae_encoder_t* e) {
    if (e->cpu) return VV_OK;
    struct vv_vae_cpu_weights* cw =
        (struct vv_vae_cpu_weights*)vv_alloc(sizeof(*cw));
    if (!cw) return VV_ERR_OUT_OF_MEMORY;
    memset(cw, 0, sizeof(*cw));
    for (int s = 0; s < e->n_stages; s++) cw->n_blocks += e->stages[s].n_blocks;
    cw->blocks = (cpu_block_t*)vv_alloc((size_t)(cw->n_blocks > 0 ? cw->n_blocks : 1)
                                        * sizeof(cpu_block_t));
    if (!cw->blocks) { vv_free(cw); return VV_ERR_OUT_OF_MEMORY; }
    memset(cw->blocks, 0, (size_t)(cw->n_blocks > 0 ? cw->n_blocks : 1)
                          * sizeof(cpu_block_t));
    bool ok = true;
    int bi = 0;
    for (int s = 0; s < e->n_stages; s++) {
        for (int b = 0; b < e->stages[s].n_blocks; b++, bi++) {
            const vv_encoder_block_t* k = &e->stages[s].blocks[b];
            cpu_block_t* c = &cw->blocks[bi];
            c->l1  = to_f16(&k->ffn_linear1_weight);
            c->l1b = to_f16(&k->ffn_linear1_bias);
            c->l2  = to_f16(&k->ffn_linear2_weight);
            c->l2b = to_f16(&k->ffn_linear2_bias);
            ok = ok && c->l1 && c->l2 &&
                 (c->l1b || !k->ffn_linear1_bias.data) &&
                 (c->l2b || !k->ffn_linear2_bias.data);
        }
        if (e->stages[s].downsample.weight.data) {
            cw->ds_w[s] = conv_to_f16_tapmajor(&e->stages[s].downsample.weight);
            cw->ds_b[s] = to_f16(&e->stages[s].downsample.bias);
            ok = ok && cw->ds_w[s];
        }
    }
    cw->head_w = conv_to_f16_tapmajor(&e->proj_mean.weight);
    cw->head_b = to_f16(&e->proj_mean.bias);
    ok = ok && cw->head_w;
    if (!ok) { vv_vae_cpu_weights_free(cw); return VV_ERR_OUT_OF_MEMORY; }
    e->cpu = cw;
    return VV_OK;
}

/* ─── Streaming state ───────────────────────────────────────────────────── */

typedef struct {
    float* tail;   /**< [cap][in_ch], time-major */
    int    have;
} cpu_ctx_t;

/** @brief A growable [rows][ch] FP32 buffer. */
typedef struct {
    float*  d;
    int64_t rows;
    int64_t cap_rows;
    int     ch;
} rowbuf_t;

static vv_status_t rb_reserve(rowbuf_t* b, int64_t rows) {
    if (rows <= b->cap_rows) return VV_OK;
    int64_t want = b->cap_rows ? b->cap_rows : 64;
    while (want < rows) want *= 2;
    float* nd = (float*)vv_alloc((size_t)want * (size_t)b->ch * sizeof(float));
    if (!nd) return VV_ERR_OUT_OF_MEMORY;
    if (b->d && b->rows > 0)
        memcpy(nd, b->d, (size_t)b->rows * (size_t)b->ch * sizeof(float));
    vv_free(b->d);
    b->d = nd;
    b->cap_rows = want;
    return VV_OK;
}

static vv_status_t rb_append(rowbuf_t* b, const float* rows, int64_t n) {
    if (n <= 0) return VV_OK;
    const vv_status_t s = rb_reserve(b, b->rows + n);
    if (s != VV_OK) return s;
    memcpy(b->d + (size_t)b->rows * b->ch, rows, (size_t)n * b->ch * sizeof(float));
    b->rows += n;
    return VV_OK;
}

/* ─── Kernels ───────────────────────────────────────────────────────────── */

static void rms_rows(const float* x, const float* w, float* y, int rows, int C,
                     float eps) {
    int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * C;
        float* yr = y + (size_t)r * C;
        float ss = 0.0f;
        for (int c = 0; c < C; c++) ss += xr[c] * xr[c];
        const float inv = 1.0f / sqrtf(ss / (float)C + eps);
        for (int c = 0; c < C; c++) yr[c] = xr[c] * inv * w[c];
    }
}

/**
 * @brief [context ++ x] into `cat`; returns the row count.
 * Fresh context is the causal left padding: zeros.
 */
static int64_t concat_ctx(const cpu_ctx_t* ctx, const float* x, int64_t len,
                          int ch, float* cat) {
    const size_t rowb = (size_t)ch * sizeof(float);
    if (ctx->have > 0) memcpy(cat, ctx->tail, (size_t)ctx->have * rowb);
    if (len > 0) memcpy(cat + (size_t)ctx->have * ch, x, (size_t)len * rowb);
    return ctx->have + len;
}

static void save_ctx(cpu_ctx_t* ctx, const float* cat, int64_t total,
                     int64_t keep, int ch) {
    if (keep > 0)
        memmove(ctx->tail, cat + (size_t)(total - keep) * ch,
                (size_t)keep * ch * sizeof(float));
    ctx->have = (int)keep;
}

/** Scratch shared by the layers of one encode. */
typedef struct {
    float* cat;  size_t cat_n;
    float* nrm;  size_t nrm_n;
    float* hid;  size_t hid_n;
    float* col;  size_t col_n;
} cpu_scratch_t;

static vv_status_t grow(float** p, size_t* have, size_t want) {
    if (want <= *have) return VV_OK;
    vv_free(*p);
    *p = (float*)vv_alloc(want * sizeof(float));
    *have = *p ? want : 0;
    return *p ? VV_OK : VV_ERR_OUT_OF_MEMORY;
}

/**
 * @brief Strided causal conv as im2col + packed GEMM (downsample, head).
 * `out` receives [ol][out_ch].
 */
static vv_status_t conv_gemm(const vae_layer_t* L, cpu_ctx_t* ctx,
                             const float* x, int64_t len, bool fin,
                             const uint16_t* w, const uint16_t* b,
                             rowbuf_t* out, cpu_scratch_t* sc) {
    int64_t ol, keep;
    vae_layer_step(L, ctx->have, len, fin, &ol, &keep);
    const int ic = L->in_ch, k = L->k, s = L->stride;
    const int64_t pad_rows = ol > 0 ? (ol - 1) * s + k : 0;
    const int64_t rows = (ctx->have + len > pad_rows) ? ctx->have + len : pad_rows;
    vv_status_t st = grow(&sc->cat, &sc->cat_n, (size_t)rows * ic);
    if (st != VV_OK) return st;
    const int64_t total = concat_ctx(ctx, x, len, ic, sc->cat);
    if (rows > total)
        memset(sc->cat + (size_t)total * ic, 0,
               (size_t)(rows - total) * ic * sizeof(float));

    if (ol > 0) {
        const size_t K = (size_t)k * ic;
        st = grow(&sc->col, &sc->col_n, (size_t)ol * K);
        if (st != VV_OK) return st;
        int j;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (j = 0; j < (int)ol; j++)
            memcpy(sc->col + (size_t)j * K, sc->cat + (size_t)j * s * ic,
                   K * sizeof(float));
        st = rb_reserve(out, out->rows + ol);
        if (st != VV_OK) return st;
        st = vv_gemm_f16w_cpu(sc->col, w, b, out->d + (size_t)out->rows * out->ch,
                              (int)ol, L->out_ch, (int)K);
        if (st != VV_OK) return st;
        out->rows += ol;
    }
    save_ctx(ctx, sc->cat, total, fin ? 0 : keep, ic);
    return VV_OK;
}

/** @brief One encoder block in place over rows of x. */
static vv_status_t block_cpu(const vv_conv_vae_encoder_t* e,
                             const vv_encoder_block_t* blk, const cpu_block_t* cb,
                             const vae_layer_t* L, cpu_ctx_t* ctx,
                             float* x, int64_t len, bool fin,
                             cpu_scratch_t* sc) {
    const int C = L->in_ch, k = L->k;
    const float eps = e->eps;
    int64_t ol, keep;
    vae_layer_step(L, ctx->have, len, fin, &ol, &keep);

    /* Mixer: the context holds raw rows, normalised here with the rest. */
    vv_status_t st = grow(&sc->cat, &sc->cat_n, (size_t)(ctx->have + len) * C);
    if (st == VV_OK) st = grow(&sc->nrm, &sc->nrm_n, (size_t)(ctx->have + len) * C);
    if (st != VV_OK) return st;
    const int64_t total = concat_ctx(ctx, x, len, C, sc->cat);
    rms_rows(sc->cat, (const float*)blk->mixer_norm_weight.data, sc->nrm,
             (int)total, C, eps);

    const float* cw = (const float*)blk->mixer_conv.weight.data;
    const float* cbias = (const float*)blk->mixer_conv.bias.data;
    const float* g = (const float*)blk->mixer_layer_scale.data;
    const int64_t base = total - len - (k - 1);   /* row of tap 0 for t = 0 */
    int t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (t = 0; t < (int)len; t++) {
        float* xr = x + (size_t)t * C;
        for (int c = 0; c < C; c++) {
            float acc = cbias ? cbias[c] : 0.0f;
            for (int kk = 0; kk < k; kk++) {
                const int64_t r = base + t + kk;
                if (r >= 0) acc += cw[c * k + kk] * sc->nrm[(size_t)r * C + c];
            }
            xr[c] += (g ? g[c] : 1.0f) * acc;
        }
    }
    save_ctx(ctx, sc->cat, total, fin ? 0 : keep, C);
    if (len <= 0) return VV_OK;

    /* FFN through the packed GEMM. */
    const int hidden = (int)blk->ffn_linear1_weight.shape[0];
    st = grow(&sc->nrm, &sc->nrm_n, (size_t)len * C);
    if (st == VV_OK) st = grow(&sc->hid, &sc->hid_n, (size_t)len * hidden);
    if (st == VV_OK) st = grow(&sc->cat, &sc->cat_n, (size_t)len * C);
    if (st != VV_OK) return st;
    rms_rows(x, (const float*)blk->ffn_norm_weight.data, sc->nrm, (int)len, C, eps);
    st = vv_gemm_f16w_cpu(sc->nrm, cb->l1, cb->l1b, sc->hid, (int)len, hidden, C);
    if (st != VV_OK) return st;
    {
        const int64_t n = len * hidden;
        int64_t i;
        int blk_i;
        const int nblk = (int)((n + 4095) / 4096);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
        for (blk_i = 0; blk_i < nblk; blk_i++) {
            const int64_t lo = (int64_t)blk_i * 4096;
            const int64_t hi = lo + 4096 < n ? lo + 4096 : n;
            for (i = lo; i < hi; i++) {
                const float v = sc->hid[i];
                sc->hid[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
            }
        }
    }
    st = vv_gemm_f16w_cpu(sc->hid, cb->l2, cb->l2b, sc->cat, (int)len, C, hidden);
    if (st != VV_OK) return st;
    const float* fg = (const float*)blk->ffn_layer_scale.data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (t = 0; t < (int)len; t++) {
        float* xr = x + (size_t)t * C;
        const float* yr = sc->cat + (size_t)t * C;
        for (int c = 0; c < C; c++) xr[c] += (fg ? fg[c] : 1.0f) * yr[c];
    }
    return VV_OK;
}

/* ─── Driver ────────────────────────────────────────────────────────────── */

typedef struct {
    vae_plan_t  plan;
    cpu_ctx_t   ctx[VAE_MAX_LAYERS];
    float*      ctx_mem;
    rowbuf_t    fifo[VAE_MAX_STAGES + 1];   /**< stage inputs; last = head */
    rowbuf_t    next;
    int         stage_layer[VAE_MAX_STAGES];
    int         ds_layer[VAE_MAX_STAGES];
    int         head_layer;
    int         stage_block0[VAE_MAX_STAGES];
    cpu_scratch_t sc;
} cpu_run_t;

static void run_free(cpu_run_t* r) {
    vv_free(r->ctx_mem);
    for (int i = 0; i <= VAE_MAX_STAGES; i++) vv_free(r->fifo[i].d);
    vv_free(r->next.d);
    vv_free(r->sc.cat);
    vv_free(r->sc.nrm);
    vv_free(r->sc.hid);
    vv_free(r->sc.col);
}

/** @brief Run stage `st` over what it has buffered, feeding the next one. */
static vv_status_t run_stage(vv_conv_vae_encoder_t* e, cpu_run_t* r, int st,
                             bool fin) {
    rowbuf_t* in = &r->fifo[st];
    if (!fin && in->rows < CPU_MIN_ROWS) return VV_OK;
    const vv_encoder_stage_t* stage = &e->stages[st];
    vv_status_t s = VV_OK;
    for (int b = 0; b < stage->n_blocks && s == VV_OK; b++) {
        const int li = r->stage_layer[st] + b;
        s = block_cpu(e, &stage->blocks[b], &e->cpu->blocks[r->stage_block0[st] + b],
                      &r->plan.L[li], &r->ctx[li], in->d, in->rows, fin, &r->sc);
    }
    if (s != VV_OK) return s;
    rowbuf_t* dst = &r->fifo[st + 1];
    if (r->ds_layer[st] >= 0) {
        const int li = r->ds_layer[st];
        s = conv_gemm(&r->plan.L[li], &r->ctx[li], in->d, in->rows, fin,
                      e->cpu->ds_w[st], e->cpu->ds_b[st], dst, &r->sc);
    } else {
        s = rb_append(dst, in->d, in->rows);
    }
    in->rows = 0;
    return s;
}

vv_status_t vv_conv_vae_encode_cpu(vv_conv_vae_encoder_t* e,
                                    const float* audio, int n_samples,
                                    float** output, int* n_frames) {
    if (!e || !audio || !output || !n_frames) return VV_ERR_NULL_PTR;
    if (n_samples <= 0) return VV_ERR_INVALID_ARG;
    *output = NULL;
    *n_frames = 0;
    if (!vv_conv_vae_complete(e)) return VV_ERR_WEIGHT_MISSING;

    vv_status_t s = cpu_weights_build(e);
    if (s != VV_OK) return s;

    cpu_run_t* r = (cpu_run_t*)vv_alloc(sizeof(cpu_run_t));
    if (!r) return VV_ERR_OUT_OF_MEMORY;
    memset(r, 0, sizeof(*r));
    s = vae_plan_build(e, &r->plan);
    if (s != VV_OK) { vv_free(r); return s; }
    const vae_plan_t* P = &r->plan;
    const double t0 = vv_time_ms();

    /* Contexts start as the causal left padding: zeros. */
    size_t ctx_total = 0;
    for (int i = 0; i < P->n_layers; i++)
        ctx_total += (size_t)P->L[i].cap * P->L[i].in_ch;
    r->ctx_mem = (float*)vv_alloc((ctx_total > 0 ? ctx_total : 1) * sizeof(float));
    if (!r->ctx_mem) { vv_free(r); return VV_ERR_OUT_OF_MEMORY; }
    memset(r->ctx_mem, 0, (ctx_total > 0 ? ctx_total : 1) * sizeof(float));
    {
        size_t off = 0;
        for (int i = 0; i < P->n_layers; i++) {
            r->ctx[i].tail = r->ctx_mem + off;
            r->ctx[i].have = (int)vae_layer_fresh_have(&P->L[i]);
            off += (size_t)P->L[i].cap * P->L[i].in_ch;
        }
    }
    {
        int li = 1, bi = 0;
        for (int st = 0; st < P->n_stages; st++) {
            r->stage_layer[st] = li;
            r->stage_block0[st] = bi;
            li += e->stages[st].n_blocks;
            bi += e->stages[st].n_blocks;
            r->ds_layer[st] = -1;
            if (e->stages[st].downsample.weight.data) r->ds_layer[st] = li++;
            r->fifo[st].ch = P->C[st];
        }
        r->head_layer = li;
        r->fifo[P->n_stages].ch = P->L[li].in_ch;
        r->next.ch = P->vae_dim;
    }

    const vae_layer_t* stem = &P->L[0];
    const float* sw = (const float*)e->input_conv.weight.data;
    const float* sb = (const float*)e->input_conv.bias.data;
    const int C0 = stem->out_ch, ks = stem->k;
    float* cat = (float*)vv_alloc((size_t)(CPU_TILE_SAMPLES + ks) * sizeof(float));
    if (!cat) { run_free(r); vv_free(r); return VV_ERR_OUT_OF_MEMORY; }

    for (int64_t off = 0; off < n_samples && s == VV_OK; off += CPU_TILE_SAMPLES) {
        const int64_t len = (n_samples - off < CPU_TILE_SAMPLES)
                          ? n_samples - off : CPU_TILE_SAMPLES;
        const bool fin = off + len >= n_samples;

        /* Stem, 1 → C0, directly: 7 taps per output are not a GEMM. */
        int64_t ol, keep;
        vae_layer_step(stem, r->ctx[0].have, len, fin, &ol, &keep);
        const int64_t total = concat_ctx(&r->ctx[0], audio + off, len, 1, cat);
        rowbuf_t* f0 = &r->fifo[0];
        s = rb_reserve(f0, f0->rows + ol);
        if (s != VV_OK) break;
        float* o = f0->d + (size_t)f0->rows * C0;
        int t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (t = 0; t < (int)ol; t++) {
            for (int c = 0; c < C0; c++) {
                float acc = sb ? sb[c] : 0.0f;
                for (int kk = 0; kk < ks; kk++) {
                    const int64_t p = (int64_t)t + kk;
                    if (p < total) acc += sw[c * ks + kk] * cat[p];
                }
                o[(size_t)t * C0 + c] = acc;
            }
        }
        f0->rows += ol;
        save_ctx(&r->ctx[0], cat, total, fin ? 0 : keep, 1);

        for (int st = 0; st < P->n_stages && s == VV_OK; st++)
            s = run_stage(e, r, st, fin);

        rowbuf_t* hin = &r->fifo[P->n_stages];
        if (s == VV_OK && (fin || hin->rows >= CPU_MIN_ROWS)) {
            const int li = r->head_layer;
            s = conv_gemm(&P->L[li], &r->ctx[li], hin->d, hin->rows, fin,
                          e->cpu->head_w, e->cpu->head_b, &r->next, &r->sc);
            hin->rows = 0;
        }
    }
    vv_free(cat);

    if (s == VV_OK) {
        *output = r->next.d;
        *n_frames = (int)r->next.rows;
        r->next.d = NULL;
        VV_LOG_I("conv_vae_cpu: encoded %d samples -> %d frames, vae_dim=%d, "
                 "%.0f ms", n_samples, *n_frames, P->vae_dim,
                 vv_time_ms() - t0);
    }
    run_free(r);
    vv_free(r);
    return s;
}
