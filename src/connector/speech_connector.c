/**
 * @file speech_connector.c
 * @brief SpeechConnector MLP: fc1 → RMSNorm → fc2 (no activation)
 *
 * Two instances:
 * - Acoustic connector: 64 → 3584
 * - Semantic connector: 128 → 3584
 */

#include "vibevoice/connector.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

/* ─── CUDA extern declarations ─────────────────────────────────────────── */

extern vv_status_t vv_cuda_alloc(void** ptr, size_t size);
extern vv_status_t vv_cuda_free(void* ptr);
extern vv_status_t vv_cuda_memcpy_h2d(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_memcpy_d2h(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_stream_sync(void* stream);
extern vv_status_t vv_cuda_stream_create(void** stream);
extern vv_status_t vv_cuda_stream_destroy(void* stream);
extern vv_status_t vv_gemm_fp16_cuda(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta, void* stream);
extern vv_status_t vv_gelu_cuda(void* data, int total, void* stream);
extern vv_status_t vv_bias_add_cuda(void* output, const void* bias,
                                      int M, int N, void* stream);
extern vv_status_t vv_rmsnorm_cuda(
    const void* input, const void* weight, void* output,
    int seq_len, int hidden_size, float eps, void* stream);
extern vv_status_t vv_fp32_to_fp16_cuda(const void* in, void* out,
                                          int n, void* stream);
extern vv_status_t vv_fp16_to_fp32_cuda(const void* in, void* out,
                                          int n, void* stream);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── GELU activation ───────────────────────────────────────────────────── */

static float gelu(float x) {
    /* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    float c = 0.7978845608028654f; /* sqrt(2/pi) */
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/* ─── RMSNorm ───────────────────────────────────────────────────────────── */

static void rmsnorm(const float* input, const float* weight,
                     float* output, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = sqrtf(sum_sq / (float)dim + eps);
    float inv_rms = 1.0f / rms;

    for (int i = 0; i < dim; i++) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/* ─── Init / Free ───────────────────────────────────────────────────────── */

vv_status_t vv_connector_init(vv_connector_t* conn,
                               int vae_dim, int hidden_size) {
    if (!conn) return VV_ERR_NULL_PTR;
    memset(conn, 0, sizeof(*conn));
    conn->vae_dim = vae_dim;
    conn->hidden_size = hidden_size;
    conn->rms_eps = 1e-6f;
    return VV_OK;
}

vv_status_t vv_connector_free(vv_connector_t* conn) {
    if (!conn) return VV_ERR_NULL_PTR;
    vv_tensor_free(&conn->fc1_weight);
    vv_tensor_free(&conn->fc1_bias);
    vv_tensor_free(&conn->norm_weight);
    vv_tensor_free(&conn->fc2_weight);
    vv_tensor_free(&conn->fc2_bias);
    return VV_OK;
}

/* ─── CPU forward pass ──────────────────────────────────────────────────── */

vv_status_t vv_connector_forward_cpu(const vv_connector_t* conn,
                                      const float* input, int n_frames,
                                      float** output) {
    if (!conn || !input || !output) return VV_ERR_NULL_PTR;
    if (n_frames <= 0) return VV_ERR_INVALID_ARG;

    int vd = conn->vae_dim;
    int hs = conn->hidden_size;

    /* Allocate intermediate and output buffers */
    float* fc1_out = (float*)vv_alloc((size_t)n_frames * (size_t)hs * sizeof(float));
    float* norm_out = (float*)vv_alloc((size_t)n_frames * (size_t)hs * sizeof(float));
    float* result = (float*)vv_alloc((size_t)n_frames * (size_t)hs * sizeof(float));
    if (!fc1_out || !norm_out || !result) {
        if (fc1_out) vv_free(fc1_out);
        if (norm_out) vv_free(norm_out);
        if (result) vv_free(result);
        return VV_ERR_OUT_OF_MEMORY;
    }

    const float* w1 = (const float*)conn->fc1_weight.data;
    const float* b1 = conn->fc1_bias.data ?
                       (const float*)conn->fc1_bias.data : NULL;
    const float* nw = (const float*)conn->norm_weight.data;
    const float* w2 = (const float*)conn->fc2_weight.data;
    const float* b2 = conn->fc2_bias.data ?
                       (const float*)conn->fc2_bias.data : NULL;

    for (int f = 0; f < n_frames; f++) {
        const float* x = input + f * vd;
        float* y1 = fc1_out + f * hs;
        float* yn = norm_out + f * hs;
        float* y2 = result + f * hs;

        /* fc1: [vae_dim] → [hidden_size] */
        for (int o = 0; o < hs; o++) {
            float sum = b1 ? b1[o] : 0.0f;
            for (int i = 0; i < vd; i++) {
                sum += w1[o * vd + i] * x[i];
            }
            y1[o] = sum;
        }

        /* RMSNorm (no activation — matches Python: fc1 → norm → fc2) */
        rmsnorm(y1, nw, yn, hs, conn->rms_eps);

        /* fc2: [hidden_size] → [hidden_size] */
        for (int o = 0; o < hs; o++) {
            float sum = b2 ? b2[o] : 0.0f;
            for (int i = 0; i < hs; i++) {
                sum += w2[o * hs + i] * yn[i];
            }
            y2[o] = sum;
        }
    }

    vv_free(fc1_out);
    vv_free(norm_out);

    *output = result;
    VV_LOG_D("connector: processed %d frames (%d → %d) [CPU]", n_frames, vd, hs);
    return VV_OK;
}

/* ─── GPU forward pass ─────────────────────────────────────────────────── */

/** Helper: upload FP32 CPU tensor to GPU as FP16 */
static vv_status_t upload_fp32_as_fp16_conn(const float* cpu, size_t n,
                                              void** out_gpu, void* stream) {
    if (!cpu || n == 0) { *out_gpu = NULL; return VV_OK; }
    void *fp32_gpu = NULL, *fp16_gpu = NULL;
    vv_status_t s;
    s = vv_cuda_alloc(&fp32_gpu, n * 4);
    if (s != VV_OK) return s;
    s = vv_cuda_alloc(&fp16_gpu, n * 2);
    if (s != VV_OK) { vv_cuda_free(fp32_gpu); return s; }
    vv_cuda_memcpy_h2d(fp32_gpu, cpu, n * 4, stream);
    vv_fp32_to_fp16_cuda(fp32_gpu, fp16_gpu, (int)n, stream);
    vv_cuda_stream_sync(stream);
    vv_cuda_free(fp32_gpu);
    *out_gpu = fp16_gpu;
    return VV_OK;
}

/**
 * @brief GPU connector forward: input[n_frames, vd] FP32 → output[n_frames, hs] FP32
 *
 * Runs fc1 → RMSNorm → fc2 entirely on GPU (no activation).
 * Weights are uploaded as FP16 each call (total ~52 MB for fc2).
 */
static vv_status_t vv_connector_forward_gpu(const vv_connector_t* conn,
                                              const float* input_cpu, int n_frames,
                                              float** output_cpu) {
    vv_status_t s;
    void* stream = NULL;
    s = vv_cuda_stream_create(&stream);
    if (s != VV_OK) return s;

    int vd = conn->vae_dim;
    int hs = conn->hidden_size;
    size_t in_elems  = (size_t)n_frames * vd;
    size_t out_elems = (size_t)n_frames * hs;

    /* Upload input as FP16 */
    void* input_gpu = NULL;
    s = upload_fp32_as_fp16_conn(input_cpu, in_elems, &input_gpu, stream);
    if (s != VV_OK) { vv_cuda_stream_destroy(stream); return s; }

    /* Upload weights as FP16 */
    void *w1_gpu = NULL, *b1_gpu = NULL, *nw_gpu = NULL;
    void *w2_gpu = NULL, *b2_gpu = NULL;
    upload_fp32_as_fp16_conn((const float*)conn->fc1_weight.data,
                              (size_t)hs * vd, &w1_gpu, stream);
    if (conn->fc1_bias.data)
        upload_fp32_as_fp16_conn((const float*)conn->fc1_bias.data,
                                  (size_t)hs, &b1_gpu, stream);
    upload_fp32_as_fp16_conn((const float*)conn->norm_weight.data,
                              (size_t)hs, &nw_gpu, stream);
    upload_fp32_as_fp16_conn((const float*)conn->fc2_weight.data,
                              (size_t)hs * hs, &w2_gpu, stream);
    if (conn->fc2_bias.data)
        upload_fp32_as_fp16_conn((const float*)conn->fc2_bias.data,
                                  (size_t)hs, &b2_gpu, stream);

    /* Allocate intermediate buffers */
    void *fc1_gpu = NULL, *norm_gpu = NULL, *fc2_gpu = NULL;
    s = vv_cuda_alloc(&fc1_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;
    s = vv_cuda_alloc(&norm_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;
    s = vv_cuda_alloc(&fc2_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;

    /* fc1: [n_frames, vd] @ [hs, vd]^T → [n_frames, hs] */
    s = vv_gemm_fp16_cuda(input_gpu, w1_gpu, fc1_gpu,
                            n_frames, hs, vd, 1.0f, 0.0f, stream);
    if (s != VV_OK) goto cleanup;

    /* bias add */
    if (b1_gpu)
        vv_bias_add_cuda(fc1_gpu, b1_gpu, n_frames, hs, stream);

    /* RMSNorm (no activation — matches Python: fc1 → norm → fc2) */
    vv_rmsnorm_cuda(fc1_gpu, nw_gpu, norm_gpu,
                     n_frames, hs, conn->rms_eps, stream);

    /* fc2: [n_frames, hs] @ [hs, hs]^T → [n_frames, hs] */
    s = vv_gemm_fp16_cuda(norm_gpu, w2_gpu, fc2_gpu,
                            n_frames, hs, hs, 1.0f, 0.0f, stream);
    if (s != VV_OK) goto cleanup;

    /* bias add */
    if (b2_gpu)
        vv_bias_add_cuda(fc2_gpu, b2_gpu, n_frames, hs, stream);

    /* Download result as FP32 */
    {
        void* fp32_gpu = NULL;
        s = vv_cuda_alloc(&fp32_gpu, out_elems * 4);
        if (s != VV_OK) goto cleanup;
        vv_fp16_to_fp32_cuda(fc2_gpu, fp32_gpu, (int)out_elems, stream);
        vv_cuda_stream_sync(stream);

        float* result = (float*)vv_alloc(out_elems * 4);
        if (!result) { vv_cuda_free(fp32_gpu); s = VV_ERR_OUT_OF_MEMORY; goto cleanup; }
        vv_cuda_memcpy_d2h(result, fp32_gpu, out_elems * 4, NULL);
        vv_cuda_free(fp32_gpu);
        *output_cpu = result;
    }

    VV_LOG_D("connector: processed %d frames (%d → %d) [GPU]", n_frames, vd, hs);
    s = VV_OK;

cleanup:
    if (input_gpu) vv_cuda_free(input_gpu);
    if (w1_gpu) vv_cuda_free(w1_gpu);
    if (b1_gpu) vv_cuda_free(b1_gpu);
    if (nw_gpu) vv_cuda_free(nw_gpu);
    if (w2_gpu) vv_cuda_free(w2_gpu);
    if (b2_gpu) vv_cuda_free(b2_gpu);
    if (fc1_gpu) vv_cuda_free(fc1_gpu);
    if (norm_gpu) vv_cuda_free(norm_gpu);
    if (fc2_gpu) vv_cuda_free(fc2_gpu);
    vv_cuda_stream_destroy(stream);
    return s;
}

/* ─── Public API: forward with GPU auto-fallback ──────────────────────── */

vv_status_t vv_connector_forward_auto(const vv_connector_t* conn,
                                        const float* input, int n_frames,
                                        float** output) {
    if (!conn || !input || !output) return VV_ERR_NULL_PTR;
    if (n_frames <= 0) return VV_ERR_INVALID_ARG;

    /* Try GPU first */
    void* test_ptr = NULL;
    if (vv_cuda_alloc(&test_ptr, 256) == VV_OK) {
        vv_cuda_free(test_ptr);
        vv_status_t gs = vv_connector_forward_gpu(conn, input, n_frames, output);
        if (gs == VV_OK) return VV_OK;
        VV_LOG_W("connector: GPU forward failed (status=%d), falling back to CPU", gs);
    }
    return vv_connector_forward_cpu(conn, input, n_frames, output);
}
