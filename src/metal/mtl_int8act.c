/**
 * @file mtl_int8act.c
 * @brief Int8 activations (W8A8, W4A8) on Metal (kernels/int8act.metal):
 *        the fused quantizers, calibration absmax, and the linear layers.
 *
 * The path argument keeps its meaning where Metal has one: GEMV is the
 * M <= 8 kernel, SIMT the tiled one for any M. There are no integer matrix
 * units to target, so MMA declines (VV_ERR_UNSUPPORTED) and AUTO takes SIMT.
 */

#include "vibevoice/device.h"
#include "metal_internal.h"

#include <limits.h>
#include <string.h>

typedef struct { int K, layout, op; float eps; int has_xsum; } q8_p;
typedef struct { int M, K; } colmax_p;
typedef struct {
    int M, K, G, y_f32;
    int n[3], first_block[3], has_bias[3], has_res[3];
} i8_p;
typedef struct { int M, N, K, G, y_f32, has_bias, has_res, pad; } i8g_p;

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

/* ─── Quantizers ────────────────────────────────────────────────────────── */

static vv_status_t quant(int op, const void* a, const void* b, int M, int K,
                         float eps, int layout, int8_t* xq, float* sx,
                         int32_t* xsum, void* stream) {
    if (!a || !xq || !sx) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    if ((K & 31) != 0 || K > VV_ACT_QUANT_MAX_K) return VV_ERR_UNSUPPORTED;
    if (layout < 0 || layout >= VV_Q8_LAYOUT_COUNT) return VV_ERR_INVALID_ARG;
    q8_p p = { K, layout, op, eps, xsum != NULL };
    vv_mtl_launch_t l = launch("vv_q8_quant", &p, sizeof(p));
    l.bufs[0] = a; l.bufs[1] = b; l.bufs[2] = xq; l.bufs[3] = sx;
    l.bufs[4] = xsum; l.nbufs = 5;
    l.grid[0] = (uint32_t)M;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_act_quant_dev(const void* x, int M, int K, int layout,
                             int8_t* xq, float* sx, int32_t* xsum,
                             void* stream) {
    return quant(0, x, NULL, M, K, 0.0f, layout, xq, sx, xsum, stream);
}

vv_status_t vv_rmsnorm_q8_dev(const void* x, const void* weight, int M, int K,
                              float eps, int layout, int8_t* xq, float* sx,
                              int32_t* xsum, void* stream) {
    if (!weight) return VV_ERR_NULL_PTR;
    /* vv_rmsnorm_dev narrows its group below 256 columns; match it or bail. */
    if (K < VV_ACT_QUANT_MIN_K) return VV_ERR_UNSUPPORTED;
    return quant(1, x, weight, M, K, eps, layout, xq, sx, xsum, stream);
}

vv_status_t vv_swiglu_q8_dev(const void* gate, const void* up, int M, int K,
                             int layout, int8_t* xq, float* sx,
                             int32_t* xsum, void* stream) {
    if (!up) return VV_ERR_NULL_PTR;
    return quant(2, gate, up, M, K, 0.0f, layout, xq, sx, xsum, stream);
}

vv_status_t vv_col_absmax_dev(const void* x, int M, int K, float* acc,
                              void* stream) {
    if (!x || !acc) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    colmax_p p = { M, K };
    vv_mtl_launch_t l = launch("vv_col_absmax", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = acc; l.nbufs = 2;
    l.grid[0] = (uint32_t)((K + 255) / 256);
    l.grid[1] = (uint32_t)((M + 63) / 64);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

/* ─── Linear layers ─────────────────────────────────────────────────────── */

/** @brief One GEMV launch over up to three projections of the same xq. */
static vv_status_t gemv(const int8_t* xq, const float* sx,
                        const int32_t* xsum, int w4,
                        const vv_i8_proj_t* projs, int n, const void* res,
                        int y_f32, int M, int K, int G, void* stream) {
    i8_p p;
    memset(&p, 0, sizeof(p));
    p.M = M; p.K = K; p.G = G; p.y_f32 = y_f32;
    vv_mtl_launch_t l = launch(w4 ? "vv_i8_gemv_w4" : "vv_i8_gemv_w8", &p,
                               sizeof(p));
    l.bufs[0] = xq; l.bufs[1] = sx; l.bufs[2] = xsum;
    int blocks = 0;
    for (int i = 0; i < 3; i++) {
        if (i < n) {
            p.n[i] = projs[i].N;
            p.first_block[i] = blocks;
            p.has_bias[i] = projs[i].bias != NULL;
            blocks += (projs[i].N + 3) / 4;
        } else {
            p.first_block[i] = INT_MAX;
        }
    }
    p.has_res[0] = res != NULL;
    /* seg 0: w, sw, sz, bias, res, y; segs 1, 2: w, sw, sz, bias, y */
    if (n > 0) {
        l.bufs[3] = projs[0].w; l.bufs[4] = w4 ? NULL : projs[0].sw;
        l.bufs[5] = w4 ? projs[0].sz : NULL; l.bufs[6] = projs[0].bias;
        l.bufs[7] = res; l.bufs[8] = projs[0].y;
    }
    for (int i = 1; i < n; i++) {
        const int b = 9 + 5 * (i - 1);
        l.bufs[b] = projs[i].w; l.bufs[b + 1] = w4 ? NULL : projs[i].sw;
        l.bufs[b + 2] = w4 ? projs[i].sz : NULL; l.bufs[b + 3] = projs[i].bias;
        l.bufs[b + 4] = projs[i].y;
    }
    l.nbufs = 19;
    l.grid[0] = (uint32_t)blocks;
    l.block[0] = 128;
    return vv_mtl_run(stream, &l);
}

static vv_status_t gemm(const int8_t* xq, const float* sx, int w4,
                        const void* w, const float* sw, const void* sz,
                        const void* bias, const void* res, void* y, int y_f32,
                        int M, int N, int K, int G, void* stream) {
    i8g_p p = { M, N, K, G, y_f32, bias != NULL, res != NULL, 0 };
    vv_mtl_launch_t l = launch(w4 ? "vv_i8_gemm_w4" : "vv_i8_gemm_w8", &p,
                               sizeof(p));
    l.bufs[0] = xq; l.bufs[1] = sx; l.bufs[2] = w; l.bufs[3] = sw;
    l.bufs[4] = sz; l.bufs[5] = bias; l.bufs[6] = res; l.bufs[7] = y;
    l.nbufs = 8;
    l.grid[0] = (uint32_t)((M + 63) / 64);
    l.grid[1] = (uint32_t)((N + 63) / 64);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

static vv_status_t dispatch(const int8_t* xq, int x_layout, const float* sx,
                            const int32_t* xsum, int w4, const void* w,
                            const float* sw, const void* sz, int G,
                            const void* bias, const void* res, void* y,
                            int y_f32, int M, int N, int K, int path,
                            void* stream) {
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    if ((K & 31) != 0) return VV_ERR_UNSUPPORTED;
    if (w4 && (G <= 0 || (G & 31) != 0 || (K % G) != 0))
        return VV_ERR_UNSUPPORTED;
    const int gemm_layout = w4 ? VV_Q8_W4_MMA : VV_Q8_NATURAL;
    if (path == VV_I8_PATH_AUTO) {
        const bool gv = w4 ? x_layout == VV_Q8_W4_GEMV : M <= 8;
        path = gv ? VV_I8_PATH_GEMV : VV_I8_PATH_SIMT;
    }
    switch (path) {
    case VV_I8_PATH_GEMV: {
        if (M > 8) return VV_ERR_UNSUPPORTED;
        if (w4 && (x_layout != VV_Q8_W4_GEMV || !xsum)) return VV_ERR_UNSUPPORTED;
        const vv_i8_proj_t pr = { w, sw, sz, bias, y, N };
        return gemv(xq, sx, xsum, w4, &pr, 1, res, y_f32, M, K, G, stream);
    }
    case VV_I8_PATH_MMA:
        return VV_ERR_UNSUPPORTED;            /* no integer matrix units */
    case VV_I8_PATH_SIMT:
        if (x_layout != gemm_layout) return VV_ERR_UNSUPPORTED;
        return gemm(xq, sx, w4, w, sw, sz, bias, res, y, y_f32, M, N, K, G,
                    stream);
    default:
        return VV_ERR_INVALID_ARG;
    }
}

vv_status_t vv_w8a8_linear_dev(const int8_t* xq, const float* sx,
                               const int8_t* w, const float* sw,
                               const void* bias, const void* residual,
                               void* y, int y_f32, int M, int N, int K,
                               int path, void* stream) {
    if (!xq || !sx || !w || !sw || !y) return VV_ERR_NULL_PTR;
    return dispatch(xq, VV_Q8_NATURAL, sx, NULL, 0, w, sw, NULL, K, bias,
                    residual, y, y_f32, M, N, K, path, stream);
}

vv_status_t vv_w4a8_linear_dev(const int8_t* xq, int x_layout,
                               const float* sx, const int32_t* xsum,
                               const void* packed, const void* sz,
                               int group_size, const void* bias,
                               const void* residual, void* y, int y_f32,
                               int M, int N, int K, int path, void* stream) {
    if (!xq || !sx || !packed || !sz || !y) return VV_ERR_NULL_PTR;
    if (x_layout != VV_Q8_W4_GEMV && x_layout != VV_Q8_W4_MMA)
        return VV_ERR_INVALID_ARG;
    return dispatch(xq, x_layout, sx, xsum, 1, packed, NULL, sz, group_size,
                    bias, residual, y, y_f32, M, N, K, path, stream);
}

vv_status_t vv_i8_linear_multi_dev(const int8_t* xq, int x_layout,
                                   const float* sx, const int32_t* xsum,
                                   int w4, const vv_i8_proj_t* projs, int n,
                                   int group_size, int M, int K,
                                   void* stream) {
    if (!xq || !sx || !projs) return VV_ERR_NULL_PTR;
    if (n < 1) return VV_ERR_INVALID_ARG;
    const bool gv = w4 ? x_layout == VV_Q8_W4_GEMV : M <= 8;
    if (gv && n <= 3 && M <= 8 && (K & 31) == 0 && (!w4 || xsum)) {
        for (int i = 0; i < n; i++)
            if (!projs[i].w || !projs[i].y) return VV_ERR_NULL_PTR;
        return gemv(xq, sx, xsum, w4, projs, n, NULL, 0, M, K,
                    w4 ? group_size : K, stream);
    }
    for (int i = 0; i < n; i++) {
        const vv_i8_proj_t* q = &projs[i];
        vv_status_t s = w4
            ? vv_w4a8_linear_dev(xq, x_layout, sx, xsum, q->w, q->sz,
                                 group_size, q->bias, NULL, q->y, 0, M, q->N,
                                 K, VV_I8_PATH_AUTO, stream)
            : vv_w8a8_linear_dev(xq, sx, (const int8_t*)q->w,
                                 (const float*)q->sw, q->bias, NULL, q->y, 0,
                                 M, q->N, K, VV_I8_PATH_AUTO, stream);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}
