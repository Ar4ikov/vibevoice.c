/**
 * @file speech_connector.c
 * @brief SpeechConnector MLP: fc1 → GELU → RMSNorm → fc2
 *
 * Two instances:
 * - Acoustic connector: 64 → 3584
 * - Semantic connector: 128 → 3584
 */

#include "vibevoice/connector.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

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

        /* GELU activation */
        for (int o = 0; o < hs; o++) {
            y1[o] = gelu(y1[o]);
        }

        /* RMSNorm */
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
    VV_LOG_D("connector: processed %d frames (%d → %d)", n_frames, vd, hs);
    return VV_OK;
}
