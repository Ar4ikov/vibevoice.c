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
#include "vibevoice/device.h"

/* ─── CUDA extern declarations ─────────────────────────────────────────── */


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
    s = vv_dev_alloc(&fp32_gpu, n * 4);
    if (s != VV_OK) return s;
    s = vv_dev_alloc(&fp16_gpu, n * 2);
    if (s != VV_OK) { vv_dev_free(fp32_gpu); return s; }
    vv_dev_memcpy_h2d(fp32_gpu, cpu, n * 4, stream);
    vv_fp32_to_fp16_dev(fp32_gpu, fp16_gpu, (int)n, stream);
    vv_dev_stream_sync(stream);
    vv_dev_free(fp32_gpu);
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
    s = vv_dev_stream_create(&stream);
    if (s != VV_OK) return s;

    int vd = conn->vae_dim;
    int hs = conn->hidden_size;
    size_t in_elems  = (size_t)n_frames * vd;
    size_t out_elems = (size_t)n_frames * hs;

    /* Upload input as FP16 */
    void* input_gpu = NULL;
    s = upload_fp32_as_fp16_conn(input_cpu, in_elems, &input_gpu, stream);
    if (s != VV_OK) { vv_dev_stream_destroy(stream); return s; }

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
    s = vv_dev_alloc(&fc1_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;
    s = vv_dev_alloc(&norm_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;
    s = vv_dev_alloc(&fc2_gpu, out_elems * 2);
    if (s != VV_OK) goto cleanup;

    /* ── Diagnostic: check input and weight stats ── */
    {
        /* Sample connector input (first few values of FP32 on CPU) */
        float imin = input_cpu[0], imax = input_cpu[0], isum = 0.0f;
        int in_n = vd < 64 ? vd : 64;
        for (int i = 0; i < in_n; i++) {
            float v = input_cpu[i];
            if (v < imin) imin = v;
            if (v > imax) imax = v;
            isum += v;
        }
        VV_LOG_D("connector diag: input[0] (%d dims): min=%.4f max=%.4f mean=%.6f",
                 in_n, imin, imax, isum / in_n);

        /* Sample fc1 weight stats */
        const float* w1_cpu = (const float*)conn->fc1_weight.data;
        float wmin = w1_cpu[0], wmax = w1_cpu[0], wsum = 0.0f;
        int wn = (hs * vd) < 1024 ? (hs * vd) : 1024;
        for (int i = 0; i < wn; i++) {
            float v = w1_cpu[i];
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
            wsum += v;
        }
        VV_LOG_D("connector diag: fc1_weight (%d vals): min=%.4f max=%.4f mean=%.6f",
                 wn, wmin, wmax, wsum / wn);

        /* Sample fc2 weight stats */
        const float* w2_cpu = (const float*)conn->fc2_weight.data;
        wmin = w2_cpu[0]; wmax = w2_cpu[0]; wsum = 0.0f;
        wn = (hs * hs) < 1024 ? (hs * hs) : 1024;
        for (int i = 0; i < wn; i++) {
            float v = w2_cpu[i];
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
            wsum += v;
        }
        VV_LOG_D("connector diag: fc2_weight (%d vals): min=%.4f max=%.4f mean=%.6f",
                 wn, wmin, wmax, wsum / wn);
    }

    /* fc1: [n_frames, vd] @ [hs, vd]^T → [n_frames, hs] */
    s = vv_gemm_fp16_dev(input_gpu, w1_gpu, fc1_gpu,
                            n_frames, hs, vd, 1.0f, 0.0f, stream);
    if (s != VV_OK) goto cleanup;

    /* bias add */
    if (b1_gpu)
        vv_bias_add_dev(fc1_gpu, b1_gpu, n_frames, hs, stream);

    /* ── Diagnostic: dump intermediate stats ── */
    #define CONN_DIAG_N 256
    {
        vv_dev_stream_sync(stream);
        int diag_n = (int)out_elems < CONN_DIAG_N ? (int)out_elems : CONN_DIAG_N;
        uint16_t diag_h[CONN_DIAG_N];
        vv_dev_memcpy_d2h(diag_h, fc1_gpu, (size_t)diag_n * 2, NULL);
        float dmin = 1e30f, dmax = -1e30f, dsum = 0.0f;
        for (int i = 0; i < diag_n; i++) {
            /* inline fp16→fp32 */
            uint16_t h = diag_h[i];
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp_ = (h >> 10) & 0x1F;
            uint32_t frac_ = h & 0x03FF;
            union { float f; uint32_t u; } u_;
            if (exp_ == 0) u_.u = sign;
            else if (exp_ == 0x1F) u_.u = sign | 0x7F800000;
            else u_.u = sign | ((exp_ + 112) << 23) | (frac_ << 13);
            float v = u_.f;
            if (v < dmin) dmin = v;
            if (v > dmax) dmax = v;
            dsum += v;
        }
        VV_LOG_D("connector diag: after fc1+bias: min=%.4f max=%.4f mean=%.6f (%d vals)",
                 dmin, dmax, dsum / diag_n, diag_n);
    }

    /* RMSNorm (no activation — matches Python: fc1 → norm → fc2) */
    vv_rmsnorm_dev(fc1_gpu, nw_gpu, norm_gpu,
                     n_frames, hs, conn->rms_eps, stream);

    {
        vv_dev_stream_sync(stream);
        int diag_n = (int)out_elems < CONN_DIAG_N ? (int)out_elems : CONN_DIAG_N;
        uint16_t diag_h[CONN_DIAG_N];
        vv_dev_memcpy_d2h(diag_h, norm_gpu, (size_t)diag_n * 2, NULL);
        float dmin = 1e30f, dmax = -1e30f, dsum = 0.0f;
        for (int i = 0; i < diag_n; i++) {
            uint16_t h = diag_h[i];
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp_ = (h >> 10) & 0x1F;
            uint32_t frac_ = h & 0x03FF;
            union { float f; uint32_t u; } u_;
            if (exp_ == 0) u_.u = sign;
            else if (exp_ == 0x1F) u_.u = sign | 0x7F800000;
            else u_.u = sign | ((exp_ + 112) << 23) | (frac_ << 13);
            float v = u_.f;
            if (v < dmin) dmin = v;
            if (v > dmax) dmax = v;
            dsum += v;
        }
        VV_LOG_D("connector diag: after RMSNorm: min=%.4f max=%.4f mean=%.6f (%d vals)",
                 dmin, dmax, dsum / diag_n, diag_n);
    }

    /* fc2: [n_frames, hs] @ [hs, hs]^T → [n_frames, hs] */
    s = vv_gemm_fp16_dev(norm_gpu, w2_gpu, fc2_gpu,
                            n_frames, hs, hs, 1.0f, 0.0f, stream);
    if (s != VV_OK) goto cleanup;

    /* bias add */
    if (b2_gpu)
        vv_bias_add_dev(fc2_gpu, b2_gpu, n_frames, hs, stream);

    {
        vv_dev_stream_sync(stream);
        int diag_n = (int)out_elems < CONN_DIAG_N ? (int)out_elems : CONN_DIAG_N;
        uint16_t diag_h[CONN_DIAG_N];
        vv_dev_memcpy_d2h(diag_h, fc2_gpu, (size_t)diag_n * 2, NULL);
        float dmin = 1e30f, dmax = -1e30f, dsum = 0.0f;
        for (int i = 0; i < diag_n; i++) {
            uint16_t h = diag_h[i];
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp_ = (h >> 10) & 0x1F;
            uint32_t frac_ = h & 0x03FF;
            union { float f; uint32_t u; } u_;
            if (exp_ == 0) u_.u = sign;
            else if (exp_ == 0x1F) u_.u = sign | 0x7F800000;
            else u_.u = sign | ((exp_ + 112) << 23) | (frac_ << 13);
            float v = u_.f;
            if (v < dmin) dmin = v;
            if (v > dmax) dmax = v;
            dsum += v;
        }
        VV_LOG_D("connector diag: after fc2+bias: min=%.4f max=%.4f mean=%.6f (%d vals)",
                 dmin, dmax, dsum / diag_n, diag_n);
    }
    #undef CONN_DIAG_N

    /* Download result as FP32 */
    {
        void* fp32_gpu = NULL;
        s = vv_dev_alloc(&fp32_gpu, out_elems * 4);
        if (s != VV_OK) goto cleanup;
        vv_fp16_to_fp32_dev(fc2_gpu, fp32_gpu, (int)out_elems, stream);
        vv_dev_stream_sync(stream);

        float* result = (float*)vv_alloc(out_elems * 4);
        if (!result) { vv_dev_free(fp32_gpu); s = VV_ERR_OUT_OF_MEMORY; goto cleanup; }
        vv_dev_memcpy_d2h(result, fp32_gpu, out_elems * 4, NULL);
        vv_dev_free(fp32_gpu);
        *output_cpu = result;
    }

    VV_LOG_D("connector: processed %d frames (%d → %d) [GPU]", n_frames, vd, hs);
    s = VV_OK;

cleanup:
    if (input_gpu) vv_dev_free(input_gpu);
    if (w1_gpu) vv_dev_free(w1_gpu);
    if (b1_gpu) vv_dev_free(b1_gpu);
    if (nw_gpu) vv_dev_free(nw_gpu);
    if (w2_gpu) vv_dev_free(w2_gpu);
    if (b2_gpu) vv_dev_free(b2_gpu);
    if (fc1_gpu) vv_dev_free(fc1_gpu);
    if (norm_gpu) vv_dev_free(norm_gpu);
    if (fc2_gpu) vv_dev_free(fc2_gpu);
    vv_dev_stream_destroy(stream);
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
    if (vv_dev_alloc(&test_ptr, 256) == VV_OK) {
        vv_dev_free(test_ptr);
        vv_status_t gs = vv_connector_forward_gpu(conn, input, n_frames, output);
        if (gs == VV_OK) return VV_OK;
        VV_LOG_W("connector: GPU forward failed (status=%d), falling back to CPU", gs);
    }
    return vv_connector_forward_cpu(conn, input, n_frames, output);
}
