/**
 * @file mtl_bitnet.c
 * @brief BitNet's integer ops on Metal (kernels/bitnet.metal).
 *
 * Up to 8 rows run the GEMVs; more run the tiled kernel, or the GEMV over
 * row chunks of 8 under VV_BITNET_MMA=0 (the CUDA fallback's shape, kept
 * so the two can be compared on one machine). Both are exact in int32
 * with the CPU's epilogue, so every choice gives the same bits.
 */

#include "vibevoice/device.h"
#include "metal_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int M, N, K, tern;
    float w_scale;
    int y_f16, has_acc, has_y, has_bias, has_absmax;
} epi_p;

typedef struct { int K, x_f16, has_sum; } aq_p;
typedef struct { uint64_t n; int relu, has_out_scale; } rq_p;
typedef struct { int V, K, nb, has_value; } head_p;

#define HEAD_ROWS 64

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

static bool tiled_allowed(void) {
    const char* e = getenv("VV_BITNET_MMA");
    return !(e && e[0] == '0');
}

vv_status_t vv_act_quant_i8_dev(const void* x, int x_f16, int M, int K,
                                int8_t* q, float* scale, int32_t* sum,
                                void* stream) {
    if (!x || !q || !scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    aq_p p = { K, x_f16 ? 1 : 0, sum != NULL };
    vv_mtl_launch_t l = launch("vv_bn_act_quant", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = q; l.bufs[2] = scale; l.bufs[3] = sum;
    l.nbufs = 4;
    l.grid[0] = (uint32_t)M;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

/** The GEMV over chunks of 8 rows, each with its own offsets. */
static vv_status_t gemv_chunks(bool tern, const int8_t* a, const void* w,
                               epi_p e, const int32_t* xsum,
                               const float* x_scale, const float* a_scale,
                               const float* bias, int32_t* acc, void* y,
                               float* absmax, void* stream) {
    const int M = e.M;
    for (int m0 = 0; m0 < M; m0 += 8) {
        epi_p c = e;
        c.M = M - m0 < 8 ? M - m0 : 8;
        vv_mtl_launch_t l = launch(tern ? "vv_bn_tern_gemv" : "vv_bn_i8_gemv",
                                   &c, sizeof(c));
        const size_t yb = (size_t)m0 * e.N * (e.y_f16 ? 2 : 4);
        if (tern) {
            l.bufs[0] = a + (size_t)m0 * e.K; l.bufs[1] = w;
            l.bufs[2] = xsum + m0; l.bufs[3] = x_scale ? x_scale + m0 : NULL;
            l.bufs[4] = bias; l.bufs[5] = acc ? acc + (size_t)m0 * e.N : NULL;
            l.bufs[6] = y ? (uint8_t*)y + yb : NULL;
            l.nbufs = 7;
        } else {
            l.bufs[0] = a + (size_t)m0 * e.K; l.bufs[1] = w;
            l.bufs[2] = a_scale; l.bufs[3] = bias;
            l.bufs[4] = acc ? acc + (size_t)m0 * e.N : NULL;
            l.bufs[5] = y ? (uint8_t*)y + yb : NULL; l.bufs[6] = absmax;
            l.nbufs = 7;
        }
        l.grid[0] = (uint32_t)((e.N + 3) / 4);
        l.block[0] = 128;
        const vv_status_t s = vv_mtl_run(stream, &l);
        if (s != VV_OK) return s;
    }
    return VV_OK;
}

static vv_status_t tiled(bool tern, const int8_t* a, const void* w, epi_p e,
                         const int32_t* xsum, const float* x_scale,
                         const float* a_scale, const float* bias,
                         int32_t* acc, void* y, float* absmax, void* stream) {
    vv_mtl_launch_t l = launch(tern ? "vv_bn_gemm_tern" : "vv_bn_gemm_i8", &e,
                               sizeof(e));
    l.bufs[0] = a; l.bufs[1] = w; l.bufs[2] = xsum; l.bufs[3] = x_scale;
    l.bufs[4] = a_scale; l.bufs[5] = bias; l.bufs[6] = acc; l.bufs[7] = y;
    l.bufs[8] = absmax; l.nbufs = 9;
    l.grid[0] = (uint32_t)((e.M + 63) / 64);
    l.grid[1] = (uint32_t)((e.N + 63) / 64);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_ternary_gemm_dev(const int8_t* q, const int32_t* xsum,
                                const float* x_scale, const uint8_t* codes,
                                float w_scale, const float* bias,
                                int32_t* acc, void* y, int y_f16,
                                int M, int N, int K, void* stream) {
    if (!q || !xsum || !codes) return VV_ERR_NULL_PTR;
    if (y && !x_scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0 || K % 128) return VV_ERR_INVALID_ARG;
    epi_p e = { M, N, K, 1, w_scale, y_f16 ? 1 : 0, acc != NULL, y != NULL,
                bias != NULL, 0 };
    if (M > 8 && tiled_allowed())
        return tiled(true, q, codes, e, xsum, x_scale, NULL, bias, acc, y,
                     NULL, stream);
    return gemv_chunks(true, q, codes, e, xsum, x_scale, NULL, bias, acc, y,
                       NULL, stream);
}

vv_status_t vv_i8_gemm_dev(const int8_t* a, const float* a_scale,
                           const int8_t* w, float w_scale, const float* bias,
                           int32_t* acc, float* y, float* absmax,
                           int M, int N, int K, void* stream) {
    if (!a || !w) return VV_ERR_NULL_PTR;
    if (y && !a_scale) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    float* am = y ? absmax : NULL;
    epi_p e = { M, N, K, 0, w_scale, 0, acc != NULL, y != NULL, bias != NULL,
                am != NULL };
    if (M > 8 && tiled_allowed())
        return tiled(false, a, w, e, NULL, NULL, a_scale, bias, acc, y, am,
                     stream);
    return gemv_chunks(false, a, w, e, NULL, NULL, a_scale, bias, acc, y, am,
                       stream);
}

vv_status_t vv_i8s_requant_dev(const float* y, int64_t n, const float* absmax,
                               int relu, int8_t* q, float* out_scale,
                               void* stream) {
    if (!y || !absmax || !q) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_ERR_INVALID_ARG;
    rq_p p = { (uint64_t)n, relu ? 1 : 0, out_scale != NULL };
    vv_mtl_launch_t l = launch("vv_bn_requant", &p, sizeof(p));
    l.bufs[0] = y; l.bufs[1] = absmax; l.bufs[2] = q; l.bufs[3] = out_scale;
    l.nbufs = 4;
    int64_t blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    l.grid[0] = (uint32_t)blocks;
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

size_t vv_i8_head_argmax_scratch_bytes(int V) {
    const size_t nb = (size_t)(V + HEAD_ROWS - 1) / HEAD_ROWS;
    return nb * (sizeof(float) + sizeof(int));
}

vv_status_t vv_i8_head_argmax_dev(const int8_t* q, const float* scale,
                                  const int8_t* w, const float* w_scale,
                                  int V, int K, int32_t* token, float* value,
                                  void* scratch, void* stream) {
    if (!q || !scale || !w || !w_scale || !token || !scratch)
        return VV_ERR_NULL_PTR;
    if (V <= 0 || K <= 0 || K % 4) return VV_ERR_INVALID_ARG;
    const int nb = (V + HEAD_ROWS - 1) / HEAD_ROWS;
    float* pv = (float*)scratch;
    int* pi = (int*)(pv + nb);
    head_p p = { V, K, nb, value != NULL };
    vv_mtl_launch_t l = launch("vv_bn_head", &p, sizeof(p));
    l.bufs[0] = q; l.bufs[1] = scale; l.bufs[2] = w; l.bufs[3] = w_scale;
    l.bufs[4] = pv; l.bufs[5] = pi; l.nbufs = 6;
    l.grid[0] = (uint32_t)nb;
    l.block[0] = 256;
    vv_status_t s = vv_mtl_run(stream, &l);
    if (s != VV_OK) return s;
    vv_mtl_launch_t f = launch("vv_bn_head_final", &p, sizeof(p));
    f.bufs[0] = pv; f.bufs[1] = pi; f.bufs[2] = token; f.bufs[3] = value;
    f.nbufs = 4;
    f.grid[0] = 1;
    f.block[0] = 256;
    return vv_mtl_run(stream, &f);
}
