/**
 * @file speech_connector.c
 * @brief SpeechConnector MLP: fc1 → RMSNorm → fc2 (no activation)
 *
 * Two instances:
 * - Acoustic connector: 64 → 3584
 * - Semantic connector: 128 → 3584
 *
 * On the device the weights are uploaded once per device and the forward pass
 * is three launches that write the prompt's hidden-state rows directly. It
 * used to re-upload ~51 MB of FP32 weights per connector per request, sync
 * three times for debug statistics nobody printed, and send the features
 * through the host twice.
 */

#include "vibevoice/connector.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

/* ─── Init / Free ───────────────────────────────────────────────────────── */

struct vv_connector_cpu {
    uint16_t* fc1_w;
    uint16_t* fc1_b;
    uint16_t* norm_w;
    uint16_t* fc2_w;
    uint16_t* fc2_b;
};

vv_status_t vv_connector_init(vv_connector_t* conn,
                               int vae_dim, int hidden_size) {
    if (!conn) return VV_ERR_NULL_PTR;
    memset(conn, 0, sizeof(*conn));
    conn->vae_dim = vae_dim;
    conn->hidden_size = hidden_size;
    conn->rms_eps = 1e-6f;
    return VV_OK;
}

static void cpu_free(struct vv_connector_cpu* c) {
    if (!c) return;
    vv_free(c->fc1_w);
    vv_free(c->fc1_b);
    vv_free(c->norm_w);
    vv_free(c->fc2_w);
    vv_free(c->fc2_b);
    vv_free(c);
}

vv_status_t vv_connector_free(vv_connector_t* conn) {
    if (!conn) return VV_ERR_NULL_PTR;
    cpu_free(conn->cpu);
    conn->cpu = NULL;
    return VV_OK;
}

/* ─── CPU forward pass ──────────────────────────────────────────────────── */

static uint16_t* f16_copy(const vv_tensor_t* t, size_t n) {
    if (!t->data) return NULL;
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    if (!h) return NULL;
    const float* f = (const float*)t->data;
    for (size_t i = 0; i < n; i++) h[i] = vv_float_to_half(f[i]);
    return h;
}

vv_status_t vv_connector_forward_cpu(vv_connector_t* conn,
                                      const float* input, int n_frames,
                                      float** output) {
    if (!conn || !input || !output) return VV_ERR_NULL_PTR;
    if (n_frames <= 0) return VV_ERR_INVALID_ARG;
    const int vd = conn->vae_dim;
    const int hs = conn->hidden_size;

    if (!conn->cpu) {
        struct vv_connector_cpu* c =
            (struct vv_connector_cpu*)vv_alloc(sizeof(*c));
        if (!c) return VV_ERR_OUT_OF_MEMORY;
        c->fc1_w  = f16_copy(&conn->fc1_weight, (size_t)hs * vd);
        c->fc1_b  = f16_copy(&conn->fc1_bias, (size_t)hs);
        c->norm_w = f16_copy(&conn->norm_weight, (size_t)hs);
        c->fc2_w  = f16_copy(&conn->fc2_weight, (size_t)hs * hs);
        c->fc2_b  = f16_copy(&conn->fc2_bias, (size_t)hs);
        if (!c->fc1_w || !c->norm_w || !c->fc2_w ||
            (conn->fc1_bias.data && !c->fc1_b) ||
            (conn->fc2_bias.data && !c->fc2_b)) {
            cpu_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }
        conn->cpu = c;
    }

    float* y1 = (float*)vv_alloc((size_t)n_frames * hs * sizeof(float));
    float* yn = (float*)vv_alloc((size_t)n_frames * hs * sizeof(float));
    float* y2 = (float*)vv_alloc((size_t)n_frames * hs * sizeof(float));
    if (!y1 || !yn || !y2) {
        vv_free(y1); vv_free(yn); vv_free(y2);
        return VV_ERR_OUT_OF_MEMORY;
    }
    vv_status_t s = vv_gemm_f16w_cpu(input, conn->cpu->fc1_w, conn->cpu->fc1_b,
                                     y1, n_frames, hs, vd);
    if (s == VV_OK)
        s = vv_rmsnorm_cpu(y1, conn->cpu->norm_w, yn, n_frames, hs,
                           conn->rms_eps);
    if (s == VV_OK)
        s = vv_gemm_f16w_cpu(yn, conn->cpu->fc2_w, conn->cpu->fc2_b, y2,
                             n_frames, hs, hs);
    vv_free(y1);
    vv_free(yn);
    if (s != VV_OK) { vv_free(y2); return s; }
    *output = y2;
    return VV_OK;
}

/* ─── Device weights ────────────────────────────────────────────────────── */

#define CONN_ALIGN 256

static size_t al(size_t v) { return (v + CONN_ALIGN - 1) / CONN_ALIGN * CONN_ALIGN; }

size_t vv_connector_dev_bytes(const vv_connector_t* c) {
    if (!c) return 0;
    const size_t hs = (size_t)c->hidden_size, vd = (size_t)c->vae_dim;
    return al(hs * vd * 2) + al(hs * 2) + al(hs * 2) + al(hs * hs * 2)
         + al(hs * 2);
}

vv_status_t vv_connector_upload(const vv_connector_t* c, void* stream,
                                vv_connector_dev_t** out) {
    if (!c || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!c->fc1_weight.data || !c->norm_weight.data || !c->fc2_weight.data)
        return VV_ERR_WEIGHT_MISSING;
    const size_t hs = (size_t)c->hidden_size, vd = (size_t)c->vae_dim;

    vv_connector_dev_t* d = (vv_connector_dev_t*)vv_alloc(sizeof(*d));
    if (!d) return VV_ERR_OUT_OF_MEMORY;
    memset(d, 0, sizeof(*d));
    d->vae_dim = c->vae_dim;
    d->hidden_size = c->hidden_size;
    d->rms_eps = c->rms_eps;
    d->bytes = vv_connector_dev_bytes(c);

    const vv_tensor_t* src[5] = { &c->fc1_weight, &c->fc1_bias,
                                  &c->norm_weight, &c->fc2_weight,
                                  &c->fc2_bias };
    const size_t n[5] = { hs * vd, hs, hs, hs * hs, hs };
    const void** dst[5] = { &d->fc1_w, &d->fc1_b, &d->norm_w, &d->fc2_w,
                            &d->fc2_b };

    void* staging = NULL;
    vv_status_t s = vv_dev_alloc(&d->blob, d->bytes);
    if (s == VV_OK) s = vv_dev_alloc(&staging, hs * hs * sizeof(float));
    size_t off = 0;
    for (int i = 0; i < 5 && s == VV_OK; i++) {
        void* p = (char*)d->blob + off;
        off += al(n[i] * 2);
        if (!src[i]->data) continue;           /* optional bias */
        if (src[i]->size_bytes < n[i] * sizeof(float)) {
            s = VV_ERR_SHAPE_MISMATCH;
            break;
        }
        s = vv_dev_memcpy_h2d(staging, src[i]->data, n[i] * sizeof(float),
                              stream);
        if (s == VV_OK)
            s = vv_vae_f32_to_f16_dev((const float*)staging, p, (int64_t)n[i],
                                      stream);
        if (s == VV_OK) s = vv_dev_stream_sync(stream);
        *dst[i] = p;
    }
    if (staging) vv_dev_free(staging);
    if (s != VV_OK) {
        if (d->blob) vv_dev_free(d->blob);
        vv_free(d);
        return s;
    }
    *out = d;
    return VV_OK;
}

void vv_connector_dev_free(vv_connector_dev_t* d) {
    if (!d) return;
    if (d->blob) vv_dev_free(d->blob);
    vv_free(d);
}

size_t vv_connector_scratch_bytes(const vv_connector_dev_t* d, int n_frames) {
    if (!d || n_frames <= 0) return 0;
    return 2 * al((size_t)n_frames * (size_t)d->hidden_size * 2);
}

/* ─── Device forward pass ───────────────────────────────────────────────── */

vv_status_t vv_connector_forward_dev(const vv_connector_dev_t* d,
                                     const void* input, int input_ld,
                                     int n_frames, void* scratch,
                                     const vv_vae_rows_t* rows, void* stream) {
    if (!d || !input || !scratch || !rows) return VV_ERR_NULL_PTR;
    if (n_frames <= 0) return VV_OK;
    const int hs = d->hidden_size;
    void* h1 = scratch;
    void* h2 = (char*)scratch + al((size_t)n_frames * (size_t)hs * 2);

    /* fc1 + bias, rounded as the separate bias kernel stored it. */
    vv_status_t s = vv_vae_gemm_tn_dev(input, input_ld, d->fc1_w, h1, hs,
                                       n_frames, hs, d->vae_dim, d->fc1_b,
                                       NULL, stream);
    if (s == VV_OK)
        s = vv_rmsnorm_dev(h1, d->norm_w, h2, n_frames, hs, d->rms_eps, stream);
    if (s == VV_OK)
        s = vv_vae_gemm_tn_dev(h2, hs, d->fc2_w, NULL, 0, n_frames, hs, hs,
                               d->fc2_b, rows, stream);
    return s;
}
