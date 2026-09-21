/**
 * @file mtl_vae.c
 * @brief The batched, streaming Conv-VAE ops on Metal (kernels/vae.metal).
 *
 * The descriptor tables are passed by value, as on CUDA, inside the launch
 * parameters. Their pointers become GPU addresses on the way (the kernels
 * dereference them), and every buffer they name is declared to the launch
 * so it is resident while the kernel runs.
 */

#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"
#include "metal_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static vv_mtl_launch_t launch(const char* kernel, const void* params,
                              size_t params_size) {
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    l.kernel = kernel;
    l.params = params;
    l.params_size = params_size;
    l.grid[1] = l.grid[2] = 1;
    l.block[1] = l.block[2] = 1;
    return l;
}

static void* gpu_addr(const void* p) {
    return (void*)(uintptr_t)vv_mtl_addr(p);
}

static void use(vv_mtl_launch_t* l, const void* p) {
    if (p && l->nuses < VV_MTL_MAX_USES) l->uses[l->nuses++] = p;
}

/** @brief The table with GPU addresses, its buffers declared to `l`. */
static vv_status_t translate_desc(const vv_vae_conv_desc_t* d,
                                  vv_vae_conv_desc_t* out,
                                  vv_mtl_launch_t* l) {
    memset(out, 0, sizeof(*out));
    if (!d) return VV_OK;
    if (d->n < 0 || d->n > VV_VAE_MAX_ITEMS) return VV_ERR_INVALID_ARG;
    out->n = d->n;
    out->cap = d->cap;
    for (int i = 0; i < d->n; i++) {
        const vv_vae_conv_item_t* s = &d->it[i];
        vv_vae_conv_item_t* t = &out->it[i];
        *t = *s;
        t->in_ptr = gpu_addr(s->in_ptr);
        t->tail_in = gpu_addr(s->tail_in);
        t->tail_out = gpu_addr(s->tail_out);
        t->out_ptr = gpu_addr(s->out_ptr);
        if ((s->in_ptr && !t->in_ptr) || (s->tail_in && !t->tail_in) ||
            (s->tail_out && !t->tail_out) || (s->out_ptr && !t->out_ptr)) {
            VV_LOG_E("metal: VAE item %d points outside device memory", i);
            return VV_ERR_INVALID_ARG;
        }
        use(l, s->in_ptr); use(l, s->tail_in);
        use(l, s->tail_out); use(l, s->out_ptr);
    }
    return VV_OK;
}

static int desc_max(const vv_vae_conv_desc_t* d, bool out) {
    int m = 0;
    for (int i = 0; i < d->n; i++) {
        const int v = out ? d->it[i].out_len : d->it[i].in_len;
        if (v > m) m = v;
    }
    return m;
}

/* ─── Convolutions ──────────────────────────────────────────────────────── */

typedef struct {
    vv_vae_conv_desc_t d;
    int64_t ld_in, ld_out;
    int in_ch, k, stride, has_bias;
} conv_p;

vv_status_t vv_vae_conv_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, const void* w, const void* b,
                            void* y, int64_t ld_out, int in_ch, int out_ch,
                            int k, int stride, bool transposed, void* stream) {
    if (!d || !w) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->n > VV_VAE_MAX_ITEMS) return VV_ERR_INVALID_ARG;
    const int m = desc_max(d, true);
    if (m <= 0) return VV_OK;
    conv_p p;
    vv_mtl_launch_t l = launch(transposed ? "vv_vae_conv_tr" : "vv_vae_conv",
                               &p, sizeof(p));
    vv_status_t s = translate_desc(d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ld_in = ld_in; p.ld_out = ld_out; p.in_ch = in_ch; p.k = k;
    p.stride = stride; p.has_bias = b != NULL;
    l.bufs[0] = x; l.bufs[1] = w; l.bufs[2] = b;
    l.bufs[3] = transposed ? NULL : y; l.nbufs = 4;
    l.grid[0] = (uint32_t)((m + 255) / 256);
    l.grid[1] = (uint32_t)out_ch;
    l.grid[2] = (uint32_t)d->n;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

typedef struct { vv_vae_conv_desc_t d; int64_t ld_in; int in_ch, pad; } tail_p;

vv_status_t vv_vae_tail_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, int in_ch, void* stream) {
    if (!d) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->cap <= 0) return VV_OK;
    tail_p p;
    vv_mtl_launch_t l = launch("vv_vae_tail", &p, sizeof(p));
    vv_status_t s = translate_desc(d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ld_in = ld_in; p.in_ch = in_ch; p.pad = 0;
    l.bufs[0] = x; l.nbufs = 1;
    const int n = in_ch * d->cap;
    l.grid[0] = (uint32_t)((n + 255) / 256);
    l.grid[1] = (uint32_t)d->n;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

typedef struct {
    vv_vae_conv_desc_t d;
    int64_t ld_in, p0, ldcol;
    int k, stride, pc, pad;
} im2col_p;

vv_status_t vv_vae_im2col_dev(const vv_vae_conv_desc_t* d, const void* x,
                              int64_t ld_in, int in_ch, int k, int stride,
                              int64_t p0, int pc, void* col, int64_t ldcol,
                              void* stream) {
    if (!d || !col) return VV_ERR_NULL_PTR;
    if (d->n <= 0 || d->n > VV_VAE_MAX_ITEMS || in_ch * k > 65535)
        return VV_ERR_INVALID_ARG;
    if (pc <= 0) return VV_OK;
    im2col_p p;
    vv_mtl_launch_t l = launch("vv_vae_im2col", &p, sizeof(p));
    vv_status_t s = translate_desc(d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ld_in = ld_in; p.p0 = p0; p.ldcol = ldcol; p.k = k; p.stride = stride;
    p.pc = pc; p.pad = 0;
    l.bufs[0] = x; l.bufs[1] = col; l.nbufs = 2;
    l.grid[0] = (uint32_t)((pc + 255) / 256);
    l.grid[1] = (uint32_t)(in_ch * k);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

typedef struct {
    vv_vae_conv_desc_t d;
    int64_t ldb, ldc, p0;
    int M, K, P, has_bias;
} cgemm_p;

/** Fewer groups than this leave an Apple GPU's cores idle. */
#define VAE_CONV_GRID 64

vv_status_t vv_vae_conv_gemm_dev(const vv_vae_conv_desc_t* d, const void* w,
                                 const void* col, int64_t ldcol, const void* b,
                                 void* y, int64_t ld_out, int out_ch, int K,
                                 int64_t p0, int pc, int tile, void* stream) {
    if (!w || !col) return VV_ERR_NULL_PTR;
    if (out_ch <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    if (tile != 0 && tile != 16 && tile != 32 && tile != 64)
        return VV_ERR_INVALID_ARG;
    if (!y && (!d || d->n <= 0 || d->n > VV_VAE_MAX_ITEMS))
        return VV_ERR_INVALID_ARG;
    if (pc <= 0) return VV_OK;
    if (tile == 0) {
        const int64_t b64 = (int64_t)((pc + 63) / 64) * ((out_ch + 63) / 64);
        const int64_t b32 = (int64_t)((pc + 31) / 32) * ((out_ch + 31) / 32);
        tile = b64 >= VAE_CONV_GRID ? 64 : b32 >= VAE_CONV_GRID ? 32 : 16;
    }
    char name[40];
    snprintf(name, sizeof(name), "vv_vae_conv_gemm_%s%d", y ? "" : "tr_", tile);
    cgemm_p p;
    vv_mtl_launch_t l = launch(name, &p, sizeof(p));
    vv_status_t s = translate_desc(y ? NULL : d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ldb = ldcol; p.ldc = ld_out; p.p0 = p0;
    p.M = out_ch; p.K = K; p.P = pc; p.has_bias = b != NULL;
    l.bufs[0] = w; l.bufs[1] = col; l.bufs[2] = y; l.bufs[3] = b; l.nbufs = 4;
    l.grid[0] = (uint32_t)((pc + tile - 1) / tile);
    l.grid[1] = (uint32_t)((out_ch + tile - 1) / tile);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

/* ─── RMSNorm statistics and the mixer ──────────────────────────────────── */

typedef struct { vv_vae_conv_desc_t d; int64_t ld, T; int C; float eps; } rms_p;

vv_status_t vv_vae_rms_dev(const void* x, int64_t ld, int64_t T, int C,
                           float eps, float* rinv,
                           const vv_vae_conv_desc_t* d, void* stream) {
    if (!x || !rinv) return VV_ERR_NULL_PTR;
    rms_p p;
    vv_mtl_launch_t l = launch("vv_vae_rms", &p, sizeof(p));
    vv_status_t s = translate_desc(d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ld = ld; p.T = T; p.C = C; p.eps = eps;
    const int64_t n = T + (int64_t)p.d.n * (p.d.cap > 0 ? p.d.cap : 0);
    if (n <= 0) return VV_OK;
    l.bufs[0] = x; l.bufs[1] = rinv; l.nbufs = 2;
    vv_mtl_grid1(&l, (uint64_t)n, 128);
    return vv_mtl_run(stream, &l);
}

typedef struct {
    vv_vae_conv_desc_t d;
    int64_t ld, T_total;
    int C, k, has_cb, has_gamma;
} mixer_p;

vv_status_t vv_vae_mixer_dev(const vv_vae_conv_desc_t* d, const void* xin,
                             void* xout, int64_t ld, const float* rinv,
                             int64_t T_total, const void* norm_w,
                             const void* conv_w, const void* conv_b,
                             const void* gamma, int C, int k, void* stream) {
    if (!d || !xin || !xout || !rinv || !norm_w || !conv_w)
        return VV_ERR_NULL_PTR;
    if (k < 1 || k > 16 || d->n <= 0 || d->n > VV_VAE_MAX_ITEMS)
        return VV_ERR_INVALID_ARG;
    const int m = desc_max(d, false);
    if (m <= 0) return VV_OK;
    mixer_p p;
    vv_mtl_launch_t l = launch("vv_vae_mixer", &p, sizeof(p));
    vv_status_t s = translate_desc(d, &p.d, &l);
    if (s != VV_OK) return s;
    p.ld = ld; p.T_total = T_total; p.C = C; p.k = k;
    p.has_cb = conv_b != NULL; p.has_gamma = gamma != NULL;
    l.bufs[0] = xin; l.bufs[1] = xout; l.bufs[2] = rinv; l.bufs[3] = norm_w;
    l.bufs[4] = conv_w; l.bufs[5] = conv_b; l.bufs[6] = gamma; l.nbufs = 7;
    l.grid[0] = (uint32_t)((m + 63) / 64);
    l.grid[1] = (uint32_t)((C + 31) / 32);
    l.grid[2] = (uint32_t)d->n;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

/* ─── GEMMs ─────────────────────────────────────────────────────────────── */

typedef struct {
    int64_t ldb, ldc;
    int M, K, P, has_bias, has_gamma, has_norm;
} gemm_p;

/* The epilogue staging of four simdgroups (32 x 36 floats each), larger
 * than the A and B tiles it reuses. */
#define VAE_GEMM_TG_MEM (4 * 32 * 36 * 4)

vv_status_t vv_vae_gemm_nn_dev(int epilogue, const void* A, const void* B,
                               int64_t ldb, void* C, int64_t ldc,
                               int M, int K, int P, const void* bias,
                               const void* gamma, const float* rinv,
                               const void* norm_w, int tile, void* stream) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || P <= 0) return VV_ERR_INVALID_ARG;
    if ((norm_w != NULL) != (rinv != NULL)) return VV_ERR_INVALID_ARG;
    /* One 64 x 64 tile serves both sizes the CUDA kernels offer: an
     * element's sum does not depend on the tile either way. */
    if (tile != 0 && tile != 64 && tile != 128) return VV_ERR_INVALID_ARG;
    const char* name;
    if (epilogue == VV_VAE_EPI_BIAS_GELU)       name = "vv_vae_gemm_nn_gelu";
    else if (epilogue == VV_VAE_EPI_BIAS_RESID) name = "vv_vae_gemm_nn_resid";
    else if (epilogue == VV_VAE_EPI_BIAS)       name = "vv_vae_gemm_nn_bias";
    else return VV_ERR_INVALID_ARG;
    gemm_p p = { ldb, ldc, M, K, P, bias != NULL, gamma != NULL,
                 norm_w != NULL };
    vv_mtl_launch_t l = launch(name, &p, sizeof(p));
    l.bufs[0] = A; l.bufs[1] = B; l.bufs[2] = C; l.bufs[3] = bias;
    l.bufs[4] = gamma; l.bufs[5] = rinv; l.bufs[6] = norm_w; l.nbufs = 7;
    l.grid[0] = (uint32_t)((P + 63) / 64);
    l.grid[1] = (uint32_t)((M + 63) / 64);
    l.block[0] = 128;
    l.tg_mem = VAE_GEMM_TG_MEM;
    return vv_mtl_run(stream, &l);
}

typedef struct {
    vv_vae_rows_t rows;
    int64_t lda, ldc;
    int M, N, K, has_bias;
} gemm_tn_p;

vv_status_t vv_vae_gemm_tn_dev(const void* A, int64_t lda, const void* B,
                               void* C, int64_t ldc, int M, int N, int K,
                               const void* bias, const vv_vae_rows_t* rows,
                               void* stream) {
    if (!A || !B) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    const int mode = rows ? rows->mode : VV_VAE_ROWS_PLAIN;
    if (mode == VV_VAE_ROWS_PLAIN && !C) return VV_ERR_NULL_PTR;
    if (mode != VV_VAE_ROWS_PLAIN &&
        (rows->n <= 0 || rows->n > VV_VAE_MAX_ITEMS || rows->seg[0].row0 != 0))
        return VV_ERR_INVALID_ARG;
    gemm_tn_p p;
    memset(&p, 0, sizeof(p));
    vv_mtl_launch_t l = launch("vv_vae_gemm_tn", &p, sizeof(p));
    p.rows.mode = mode;
    if (mode != VV_VAE_ROWS_PLAIN) {
        p.rows.n = rows->n;
        for (int i = 0; i < rows->n; i++) {
            p.rows.seg[i].row0 = rows->seg[i].row0;
            p.rows.seg[i].ld = rows->seg[i].ld;
            p.rows.seg[i].dst = gpu_addr(rows->seg[i].dst);
            if (!p.rows.seg[i].dst) {
                VV_LOG_E("metal: connector row segment %d points outside "
                         "device memory", i);
                return VV_ERR_INVALID_ARG;
            }
            use(&l, rows->seg[i].dst);
        }
    }
    p.lda = lda; p.ldc = ldc; p.M = M; p.N = N; p.K = K;
    p.has_bias = bias != NULL;
    l.bufs[0] = A; l.bufs[1] = B;
    l.bufs[2] = mode == VV_VAE_ROWS_PLAIN ? C : NULL;
    l.bufs[3] = bias; l.nbufs = 4;
    l.grid[0] = (uint32_t)((N + 63) / 64);
    l.grid[1] = (uint32_t)((M + 63) / 64);
    l.block[0] = 128;
    l.tg_mem = VAE_GEMM_TG_MEM;
    return vv_mtl_run(stream, &l);
}
