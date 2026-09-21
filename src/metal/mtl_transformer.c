/**
 * @file mtl_transformer.c
 * @brief Embedding, RMSNorm, RoPE, SwiGLU, residual and bias adds, and the
 *        FP16/FP32 conversions on Metal (kernels/transformer.metal).
 */

#include "vibevoice/device.h"
#include "metal_internal.h"

#include <string.h>

typedef struct { int seq_len; int hidden; } embed_p;
typedef struct { int hidden; float eps; } rmsnorm_p;
typedef struct { int seq_len, n_heads, head_dim, position_offset, use_dev_pos;
                 float theta; } rope_p;
typedef struct { uint64_t n; } n_p;
typedef struct { int rows, inter; } swiglu_fused_p;
typedef struct { int M, N; } bias_p;

static vv_mtl_launch_t launch(const char* kernel, const void* params,
                              size_t params_size) {
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    l.kernel = kernel;
    l.params = params;
    l.params_size = params_size;
    return l;
}

vv_status_t vv_embedding_dev(const void* table, const int32_t* ids,
                             void* output, int seq_len, int hidden_size,
                             void* stream) {
    if (!table || !ids || !output) return VV_ERR_NULL_PTR;
    if (seq_len <= 0) return VV_OK;
    embed_p p = { seq_len, hidden_size };
    vv_mtl_launch_t l = launch("vv_embedding", &p, sizeof(p));
    l.bufs[0] = table; l.bufs[1] = ids; l.bufs[2] = output; l.nbufs = 3;
    l.grid[0] = (uint32_t)seq_len; l.grid[1] = l.grid[2] = 1;
    l.block[0] = 256; l.block[1] = l.block[2] = 1;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_rmsnorm_dev(const void* input, const void* weight,
                           void* output, int seq_len, int hidden_size,
                           float eps, void* stream) {
    if (!input || !weight || !output) return VV_ERR_NULL_PTR;
    if (seq_len <= 0) return VV_OK;
    int threads = 256;
    if (hidden_size < 256) {
        /* The tree needs a power of two; rmsnorm.cu uses hidden_size here,
         * which only ever is one. */
        threads = 1;
        while (threads * 2 <= hidden_size) threads *= 2;
    }
    rmsnorm_p p = { hidden_size, eps };
    vv_mtl_launch_t l = launch("vv_rmsnorm", &p, sizeof(p));
    l.bufs[0] = input; l.bufs[1] = weight; l.bufs[2] = output; l.nbufs = 3;
    l.grid[0] = (uint32_t)seq_len; l.grid[1] = l.grid[2] = 1;
    l.block[0] = (uint32_t)threads; l.block[1] = l.block[2] = 1;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_rope_dev(void* x, int seq_len, int n_heads, int head_dim,
                        int position_offset, const int* d_position,
                        float theta, void* stream) {
    if (!x) return VV_ERR_NULL_PTR;
    if (seq_len <= 0 || n_heads <= 0) return VV_OK;
    rope_p p = { seq_len, n_heads, head_dim, position_offset,
                 d_position != NULL, theta };
    vv_mtl_launch_t l = launch("vv_rope", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = d_position; l.nbufs = 2;
    l.grid[0] = (uint32_t)seq_len; l.grid[1] = (uint32_t)n_heads; l.grid[2] = 1;
    l.block[0] = (uint32_t)(head_dim / 2); l.block[1] = l.block[2] = 1;
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_swiglu_dev(const void* gate, const void* up, void* output,
                          int n_elements, void* stream) {
    if (!gate || !up || !output) return VV_ERR_NULL_PTR;
    if (n_elements <= 0) return VV_OK;
    n_p p = { (uint64_t)n_elements };
    vv_mtl_launch_t l = launch("vv_swiglu", &p, sizeof(p));
    l.bufs[0] = gate; l.bufs[1] = up; l.bufs[2] = output; l.nbufs = 3;
    vv_mtl_grid1(&l, (uint64_t)n_elements, 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_swiglu_fused_dev(const void* gate_up, void* output, int rows,
                                int inter, void* stream) {
    if (!gate_up || !output) return VV_ERR_NULL_PTR;
    if (rows <= 0 || inter <= 0) return VV_OK;
    swiglu_fused_p p = { rows, inter };
    vv_mtl_launch_t l = launch("vv_swiglu_fused", &p, sizeof(p));
    l.bufs[0] = gate_up; l.bufs[1] = output; l.nbufs = 2;
    vv_mtl_grid1(&l, (uint64_t)rows * (uint64_t)inter, 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_residual_add_dev(void* x, const void* y, int total,
                                void* stream) {
    if (!x || !y) return VV_ERR_NULL_PTR;
    if (total <= 0) return VV_OK;
    n_p p = { (uint64_t)total };
    vv_mtl_launch_t l = launch("vv_residual_add", &p, sizeof(p));
    l.bufs[0] = x; l.bufs[1] = y; l.nbufs = 2;
    vv_mtl_grid1(&l, (uint64_t)total, 256);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_bias_add_dev(void* output, const void* bias, int M, int N,
                            void* stream) {
    if (!output || !bias) return VV_ERR_NULL_PTR;
    if (M <= 0 || N <= 0) return VV_OK;
    bias_p p = { M, N };
    vv_mtl_launch_t l = launch("vv_bias_add", &p, sizeof(p));
    l.bufs[0] = output; l.bufs[1] = bias; l.nbufs = 2;
    vv_mtl_grid1(&l, (uint64_t)M * (uint64_t)N, 256);
    return vv_mtl_run(stream, &l);
}

/** @brief A 2-D grid of 256-thread groups covering n (past 2^32 threads). */
static void grid_2d(vv_mtl_launch_t* l, uint64_t n) {
    const uint64_t groups = (n + 255) / 256;
    const uint64_t gx = groups < 65535 ? groups : 65535;
    l->grid[0] = (uint32_t)(gx ? gx : 1);
    l->grid[1] = (uint32_t)((groups + gx - 1) / (gx ? gx : 1));
    l->grid[2] = 1;
    l->block[0] = 256; l->block[1] = l->block[2] = 1;
}

vv_status_t vv_fp32_to_fp16_dev(const void* in_fp32, void* out_fp16, int n,
                                void* stream) {
    if (!in_fp32 || !out_fp16) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    n_p p = { (uint64_t)n };
    vv_mtl_launch_t l = launch("vv_f32_to_f16", &p, sizeof(p));
    l.bufs[0] = in_fp32; l.bufs[1] = out_fp16; l.nbufs = 2;
    grid_2d(&l, (uint64_t)n);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_vae_f32_to_f16_dev(const float* in, void* out, int64_t n,
                                  void* stream) {
    if (!in || !out) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    n_p p = { (uint64_t)n };
    vv_mtl_launch_t l = launch("vv_f32_to_f16", &p, sizeof(p));
    l.bufs[0] = in; l.bufs[1] = out; l.nbufs = 2;
    grid_2d(&l, (uint64_t)n);
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_fp16_to_fp32_dev(const void* in_fp16, void* out_fp32, int n,
                                void* stream) {
    if (!in_fp16 || !out_fp32) return VV_ERR_NULL_PTR;
    if (n <= 0) return VV_OK;
    n_p p = { (uint64_t)n };
    vv_mtl_launch_t l = launch("vv_f16_to_f32", &p, sizeof(p));
    l.bufs[0] = in_fp16; l.bufs[1] = out_fp32; l.nbufs = 2;
    vv_mtl_grid1(&l, (uint64_t)n, 256);
    return vv_mtl_run(stream, &l);
}
