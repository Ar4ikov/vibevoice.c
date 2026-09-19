/**
 * @file mtl_linear.c
 * @brief NF4 GEMV/GEMM, dense FP16 GEMM, the LM head and argmax on Metal
 *        (kernels/linear.metal).
 */

#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"
#include "metal_internal.h"

#include <string.h>

typedef struct { int N, K, has_bias; } nf4_gemv_p;
typedef struct { int n_elements; } nf4_deq_p;
typedef struct { int M, N, K; float alpha, beta; } gemm_p;
typedef struct { int V, K; } lm_head_p;
typedef struct { int V, n, has_value; } argmax_p;

/* Threadgroup memory of the tiled GEMMs: the epilogue's staging, which is
 * larger than the A and B tiles it reuses. */
#define GEMM_TG_MEM (4 * 32 * (32 + 4) * 4)

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

vv_status_t vv_nf4_gemv_dev(const void* x, const uint8_t* packed,
                            const void* scales, const void* bias, void* y,
                            int N, int K, void* stream) {
    if (!x || !packed || !scales || !y) return VV_ERR_NULL_PTR;
    if ((K & 255) != 0) return VV_ERR_UNSUPPORTED;     /* caller falls back */
    nf4_gemv_p p = { N, K, bias != NULL };
    vv_mtl_launch_t l = launch("vv_nf4_gemv", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = packed; l.bufs[2] = scales; l.bufs[3] = bias;
    l.bufs[4] = y; l.nbufs = 5;
    l.grid[0] = (uint32_t)((N + 7) / 8);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_dequant_nf4_dev(const uint8_t* packed, const void* scales_fp16,
                               void* output_fp16, int n_elements,
                               int block_size, void* stream) {
    if (!packed || !scales_fp16 || !output_fp16) return VV_ERR_NULL_PTR;
    if (n_elements <= 0 || block_size <= 0) return VV_ERR_INVALID_ARG;
    if (block_size != 64 || (n_elements & 7) != 0) return VV_ERR_UNSUPPORTED;
    nf4_deq_p p = { n_elements };
    vv_mtl_launch_t l = launch("vv_dequant_nf4", &p, sizeof(p));
    l.bufs[0] = packed; l.bufs[1] = scales_fp16; l.bufs[2] = output_fp16;
    l.nbufs = 3;
    vv_mtl_grid1(&l, (uint64_t)(n_elements / 8), 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_gemm_fp16_dev(const void* A, const void* B, void* C,
                             int M, int N, int K, float alpha, float beta,
                             void* stream) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    gemm_p p = { M, N, K, alpha, beta };
    if (M <= 8 && (K & 7) == 0) {
        vv_mtl_launch_t l = launch("vv_gemm_tn_skinny", &p, sizeof(p));
        l.bufs[0] = A; l.bufs[1] = B; l.bufs[2] = C; l.nbufs = 3;
        l.grid[0] = (uint32_t)((N + 7) / 8);
        l.block[0] = 256;
        return vv_mtl_run(stream, &l);
    }
    return vv_gemm_fp16_tile_dev(A, B, C, M, N, K, alpha, beta, stream);
}

vv_status_t vv_gemm_fp16_tile_dev(const void* A, const void* B, void* C,
                                  int M, int N, int K, float alpha, float beta,
                                  void* stream) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return VV_ERR_INVALID_ARG;
    gemm_p p = { M, N, K, alpha, beta };
    vv_mtl_launch_t l = launch("vv_gemm_tn", &p, sizeof(p));
    l.bufs[0] = A; l.bufs[1] = B; l.bufs[2] = C; l.nbufs = 3;
    l.grid[0] = (uint32_t)((N + 63) / 64);
    l.grid[1] = (uint32_t)((M + 63) / 64);
    l.block[0] = 128;
    l.tg_mem = GEMM_TG_MEM;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_gemm_fp16_nn_dev(const void* A, const void* B, void* C,
                                int M, int K, int P, float alpha, float beta,
                                void* stream) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;
    if (M <= 0 || K <= 0 || P <= 0) return VV_ERR_INVALID_ARG;
    gemm_p p = { M, P, K, alpha, beta };
    vv_mtl_launch_t l = launch("vv_gemm_nn", &p, sizeof(p));
    l.bufs[0] = A; l.bufs[1] = B; l.bufs[2] = C; l.nbufs = 3;
    l.grid[0] = (uint32_t)((P + 63) / 64);
    l.grid[1] = (uint32_t)((M + 63) / 64);
    l.block[0] = 128;
    l.tg_mem = GEMM_TG_MEM;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_nf4_gemm_dev(const void* input_fp16,
                            const uint8_t* weight_packed,
                            const void* weight_scales_fp16, void* output_fp16,
                            void* temp_weight_fp16, int M, int N, int K,
                            int block_size, void* stream) {
    if (!input_fp16 || !weight_packed || !weight_scales_fp16 ||
        !output_fp16 || !temp_weight_fp16)
        return VV_ERR_NULL_PTR;
    vv_status_t s = vv_dequant_nf4_dev(weight_packed, weight_scales_fp16,
                                       temp_weight_fp16, N * K, block_size,
                                       stream);
    if (s != VV_OK) return s;
    return vv_gemm_fp16_dev(input_fp16, temp_weight_fp16, output_fp16,
                            M, N, K, 1.0f, 0.0f, stream);
}

vv_status_t vv_lm_head_gemv_dev(const void* x, const void* W, void* logits,
                                int V, int K, void* stream) {
    if (!x || !W || !logits) return VV_ERR_NULL_PTR;
    lm_head_p p = { V, K };
    vv_mtl_launch_t l = launch("vv_lm_head_gemv", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = W; l.bufs[2] = logits; l.nbufs = 3;
    l.grid[0] = (uint32_t)((V + 7) / 8);
    l.block[0] = 256;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_argmax_dev(const void* logits, int V, void* scratch_v,
                          void* scratch_i, void* out_token, void* out_value,
                          void* stream) {
    if (!logits || !scratch_v || !scratch_i || !out_token)
        return VV_ERR_NULL_PTR;
    const int nblocks = 256;              /* must match VV_ARGMAX_PARTIALS */
    argmax_p p = { V, nblocks, out_value != NULL };
    vv_mtl_launch_t l = launch("vv_argmax_partial", &p, sizeof(p));
    l.bufs[0] = logits; l.bufs[1] = scratch_v; l.bufs[2] = scratch_i;
    l.nbufs = 3;
    l.grid[0] = (uint32_t)nblocks;
    l.block[0] = 256;
    vv_status_t s = vv_mtl_run(stream, &l);
    if (s != VV_OK) return s;
    vv_mtl_launch_t f = launch("vv_argmax_final", &p, sizeof(p));
    f.bufs[0] = scratch_v; f.bufs[1] = scratch_i; f.bufs[2] = out_token;
    f.bufs[3] = out_value; f.nbufs = 4;
    f.grid[0] = 1;
    f.block[0] = 256;
    return vv_mtl_run(stream, &f);
}
