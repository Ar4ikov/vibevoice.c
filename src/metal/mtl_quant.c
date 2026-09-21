/**
 * @file mtl_quant.c
 * @brief INT8 (W8A16) and legacy row-major INT4 (AWQ) linear layers on
 *        Metal (kernels/quant.metal).
 */

#include "vibevoice/device.h"
#include "metal_internal.h"

#include <string.h>

typedef struct { int M, N, K, has_bias; } i8w_p;
typedef struct { int M, N, K, gshift, has_bias; } awq_p;

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

/* ─── INT8 ──────────────────────────────────────────────────────────────── */

vv_status_t vv_int8_gemv_dev(const void* x, const int8_t* q,
                             const float* scales, const void* bias, void* y,
                             int N, int K, void* stream) {
    if (!x || !q || !scales || !y) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;
    i8w_p p = { 1, N, K, bias != NULL };
    vv_mtl_launch_t l = launch("vv_int8_gemv", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = q; l.bufs[2] = scales; l.bufs[3] = bias;
    l.bufs[4] = y; l.nbufs = 5;
    l.grid[0] = (uint32_t)((N + 7) / 8);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_dequant_int8_dev(const int8_t* q, const float* scales,
                                void* output_fp16, int N, int K,
                                void* stream) {
    if (!q || !scales || !output_fp16) return VV_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;
    i8w_p p = { 0, N, K, 0 };
    vv_mtl_launch_t l = launch("vv_int8_dequant", &p, sizeof(p));
    l.bufs[0] = q; l.bufs[1] = scales; l.bufs[2] = output_fp16; l.nbufs = 3;
    vv_mtl_grid1(&l, (uint64_t)N * (uint64_t)(K >> 4), 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_int8_gemm_dev(const void* input_fp16, const int8_t* q,
                             const float* scales, void* output_fp16,
                             void* temp_weight_fp16, int M, int N, int K,
                             void* stream) {
    if (!input_fp16 || !q || !scales || !output_fp16) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0 || (K & 15) != 0) return VV_ERR_UNSUPPORTED;
    if (M <= 8) {
        i8w_p p = { M, N, K, 0 };
        vv_mtl_launch_t l = launch("vv_int8_gemm_small", &p, sizeof(p));
        l.bufs[0] = input_fp16; l.bufs[1] = q; l.bufs[2] = scales;
        l.bufs[3] = output_fp16; l.nbufs = 4;
        l.grid[0] = (uint32_t)((N + 7) / 8);
        l.block[0] = 256;
        return vv_mtl_run(stream, &l);
    }
    if (M <= VV_SKINNY_M_MAX) {
        const vv_skinny_proj_t pr = { q, scales, NULL, output_fp16, N };
        const vv_status_t s = vv_skinny_linear_dev(input_fp16, VV_SKINNY_INT8,
                                                   &pr, 1, M, K, 1.0f, stream);
        if (s != VV_ERR_UNSUPPORTED) return s;
    }
    if (!temp_weight_fp16) return VV_ERR_NULL_PTR;
    vv_status_t s = vv_dequant_int8_dev(q, scales, temp_weight_fp16, N, K,
                                        stream);
    if (s != VV_OK) return s;
    return vv_gemm_fp16_tile_dev(input_fp16, temp_weight_fp16, output_fp16,
                                 M, N, K, 1.0f, 0.0f, stream);
}

/* ─── INT4, row-major legacy layout ─────────────────────────────────────── */

static int gshift_of(int group_size) {
    switch (group_size) {
        case 32:  return 5;
        case 64:  return 6;
        case 128: return 7;
        case 256: return 8;
        default:  return -1;
    }
}

vv_status_t vv_awq_gemv_dev(const void* x, const uint32_t* packed,
                            const uint32_t* mins, const void* scales,
                            const void* bias, void* y, int N, int K,
                            int group_size, void* stream) {
    if (!x || !packed || !scales || !mins || !y) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 255) != 0) return VV_ERR_UNSUPPORTED;
    awq_p p = { 1, N, K, gs, bias != NULL };
    vv_mtl_launch_t l = launch("vv_awq_gemv", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = packed; l.bufs[2] = scales; l.bufs[3] = mins;
    l.bufs[4] = bias; l.bufs[5] = y; l.nbufs = 6;
    l.grid[0] = (uint32_t)((N + 7) / 8);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_dequant_awq_dev(const uint32_t* packed, const uint32_t* mins,
                               const void* scales, void* output_fp16,
                               int N, int K, int group_size, void* stream) {
    if (!packed || !scales || !mins || !output_fp16) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 7) != 0) return VV_ERR_UNSUPPORTED;
    awq_p p = { 0, N, K, gs, 0 };
    vv_mtl_launch_t l = launch("vv_awq_dequant", &p, sizeof(p));
    l.bufs[0] = packed; l.bufs[1] = scales; l.bufs[2] = mins;
    l.bufs[3] = output_fp16; l.nbufs = 4;
    vv_mtl_grid1(&l, (uint64_t)N * (uint64_t)(K >> 3), 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_awq_gemm_dev(const void* input_fp16, const uint32_t* packed,
                            const uint32_t* mins, const void* scales,
                            void* output_fp16, void* temp_weight_fp16,
                            int M, int N, int K, int group_size,
                            void* stream) {
    if (!input_fp16 || !packed || !scales || !mins || !output_fp16)
        return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0) return VV_ERR_UNSUPPORTED;
    if (M <= 8 && (K & 255) == 0) {
        awq_p p = { M, N, K, gs, 0 };
        vv_mtl_launch_t l = launch("vv_awq_gemm_small", &p, sizeof(p));
        l.bufs[0] = input_fp16; l.bufs[1] = packed; l.bufs[2] = scales;
        l.bufs[3] = mins; l.bufs[4] = output_fp16; l.nbufs = 5;
        l.grid[0] = (uint32_t)((N + 7) / 8);
        l.block[0] = 256;
        return vv_mtl_run(stream, &l);
    }
    if (!temp_weight_fp16) return VV_ERR_NULL_PTR;
    vv_status_t s = vv_dequant_awq_dev(packed, mins, scales, temp_weight_fp16,
                                       N, K, group_size, stream);
    if (s != VV_OK) return s;
    return vv_gemm_fp16_dev(input_fp16, temp_weight_fp16, output_fp16,
                            M, N, K, 1.0f, 0.0f, stream);
}
