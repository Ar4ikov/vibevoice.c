/**
 * @file mtl_w4a16.c
 * @brief INT4 group-affine linear layers in the GPU layout on Metal
 *        (kernels/w4a16.metal).
 *
 * Decode runs the GEMV (several projections of one x in one launch);
 * prefill dequantizes into the caller's scratch and runs the tile GEMM,
 * which is what the CUDA path does on cards without its fused kernel.
 */

#include "vibevoice/device.h"
#include "metal_internal.h"

#include <limits.h>
#include <string.h>

typedef struct {
    int K, gshift;
    int n[3], first_block[3], has_bias[3];
} w4v_p;

typedef struct { int M, K; } gather_p;
typedef struct { int N, K, gshift; } deq_p;

static int gshift_of(int group_size) {
    switch (group_size) {
        case 32:  return 5;
        case 64:  return 6;
        case 128: return 7;
        case 256: return 8;
        default:  return -1;
    }
}

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

static vv_status_t gemv(const void* x, const vv_w4a16_proj_t* projs, int n,
                        int K, int gs, void* stream) {
    w4v_p p;
    memset(&p, 0, sizeof(p));
    p.K = K;
    p.gshift = gs;
    vv_mtl_launch_t l = launch("vv_w4a16_gemv", &p, sizeof(p));
    l.bufs[0] = x;
    int blocks = 0;
    for (int i = 0; i < 3; i++) {
        if (i < n) {
            p.n[i] = projs[i].N;
            p.first_block[i] = blocks;
            p.has_bias[i] = projs[i].bias != NULL;
            blocks += (projs[i].N + 3) / 4;
            l.bufs[1 + 4 * i] = projs[i].packed;
            l.bufs[2 + 4 * i] = projs[i].sz;
            l.bufs[3 + 4 * i] = projs[i].bias;
            l.bufs[4 + 4 * i] = projs[i].y;
        } else {
            p.first_block[i] = INT_MAX;
        }
    }
    l.nbufs = 13;
    l.grid[0] = (uint32_t)blocks;
    l.block[0] = 128;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_w4a16_gemv_dev(const void* x, const void* packed,
                              const void* sz, const void* bias, void* y,
                              int N, int K, int group_size, void* stream) {
    if (!x || !packed || !sz || !y) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 31) != 0 || N <= 0) return VV_ERR_UNSUPPORTED;
    const vv_w4a16_proj_t pr = { packed, sz, bias, y, N };
    return gemv(x, &pr, 1, K, gs, stream);
}

vv_status_t vv_w4a16_gemv_multi_dev(const void* x,
                                    const vv_w4a16_proj_t* projs, int n_proj,
                                    int K, int group_size, void* stream) {
    if (!x || !projs) return VV_ERR_NULL_PTR;
    if (n_proj < 1 || n_proj > 3) return VV_ERR_INVALID_ARG;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 31) != 0) return VV_ERR_UNSUPPORTED;
    for (int i = 0; i < n_proj; i++) {
        if (!projs[i].packed || !projs[i].sz || !projs[i].y)
            return VV_ERR_NULL_PTR;
        if (projs[i].N <= 0) return VV_ERR_UNSUPPORTED;
    }
    return gemv(x, projs, n_proj, K, gs, stream);
}

vv_status_t vv_w4a16_gather_dev(const void* x, const int32_t* perm, void* out,
                                int M, int K, void* stream) {
    if (!x || !perm || !out) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || (K & 7) != 0) return VV_ERR_INVALID_ARG;
    gather_p p = { M, K };
    vv_mtl_launch_t l = launch("vv_w4a16_gather", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = perm; l.bufs[2] = out; l.nbufs = 3;
    vv_mtl_grid1(&l, (uint64_t)M * (uint64_t)K / 8, 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_w4a16_dequant_dev(const void* packed, const void* sz,
                                 void* out_fp16, int N, int K, int group_size,
                                 void* stream) {
    if (!packed || !sz || !out_fp16) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 31) != 0) return VV_ERR_UNSUPPORTED;
    deq_p p = { N, K, gs };
    vv_mtl_launch_t l = launch("vv_w4a16_dequant", &p, sizeof(p));
    l.bufs[0] = packed; l.bufs[1] = sz; l.bufs[2] = out_fp16; l.nbufs = 3;
    vv_mtl_grid1(&l, (uint64_t)N * (uint64_t)(K >> 5), 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_w4a16_gemm_dev(const void* A, const void* packed,
                              const void* sz, const void* bias, void* C,
                              void* scratch, size_t scratch_bytes,
                              int M, int N, int K, int group_size,
                              void* stream) {
    if (!A || !packed || !sz || !C) return VV_ERR_NULL_PTR;
    const int gs = gshift_of(group_size);
    if (gs < 0 || (K & 31) != 0 || M <= 0 || N <= 0 || (N & 7) != 0)
        return VV_ERR_UNSUPPORTED;
    if (M == 1)
        return vv_w4a16_gemv_dev(A, packed, sz, bias, C, N, K, group_size,
                                 stream);
    if (!scratch) return VV_ERR_NULL_PTR;
    if (scratch_bytes < (size_t)N * K * 2) return VV_ERR_OVERFLOW;
    vv_status_t s = vv_w4a16_dequant_dev(packed, sz, scratch, N, K,
                                         group_size, stream);
    if (s == VV_OK)
        s = vv_gemm_fp16_dev(A, scratch, C, M, N, K, 1.0f, 0.0f, stream);
    if (s != VV_OK || !bias) return s;
    return vv_bias_add_dev(C, bias, M, N, stream);
}
