/**
 * @file test_kv_quant.c
 * @brief Round-trip the quantized KV store against the FP16 attention path.
 *
 * Attention is the only consumer of the KV cache, so that is what the test
 * checks: build a random cache, run the FP16 decode kernel over it, then run
 * each quantized format and compare. A format is judged by the cosine
 * similarity of the attention output, which is what actually reaches the
 * next layer — not by the error on individual stored values.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_Q_HEADS  28
#define N_KV_HEADS 4
#define HEAD_DIM   128
#define N_POS      257          /* deliberately not a tile multiple */

static uint32_t rng_state = 12345u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

/** @brief Cosine similarity between two FP16 buffers read back to the host. */
static double cosine(const uint16_t* a, const uint16_t* b, size_t n) {
    double num = 0.0, da = 0.0, db = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double x = vv_half_to_float(a[i]);
        const double y = vv_half_to_float(b[i]);
        num += x * y; da += x * x; db += y * y;
    }
    return num / (sqrt(da) * sqrt(db) + 1e-30);
}

int main(void) {
#ifndef VV_HAS_ACCEL
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(0, &total, &freem, NULL) != VV_OK || total == 0) {
        printf("SKIP: no device available\n");
        return 0;
    }

    const size_t kv_elems = (size_t)N_POS * N_KV_HEADS * HEAD_DIM;
    const size_t q_elems  = (size_t)N_Q_HEADS * HEAD_DIM;

    uint16_t* h_k = malloc(kv_elems * 2);
    uint16_t* h_v = malloc(kv_elems * 2);
    uint16_t* h_q = malloc(q_elems * 2);
    uint16_t* h_ref = malloc(q_elems * 2);
    uint16_t* h_got = malloc(q_elems * 2);
    if (!h_k || !h_v || !h_q || !h_ref || !h_got) return 1;

    /*
     * Shaped like real Qwen2 attention inputs, because that is what the
     * formats have to survive: a few channels carry a large component that
     * is nearly constant across positions (|k| ends up ~20x |k - mean|),
     * plus per-channel scale spread on top.
     */
    for (size_t i = 0; i < kv_elems; i++) {
        const int ch = (int)(i % HEAD_DIM);
        float k = frand() * 2.0f;
        float v = frand() * 1.5f;
        if (ch == 7)  k = 60.0f + k * 8.0f;
        if (ch == 33) k = -40.0f + k * 4.0f;
        if (ch == 64) v *= 6.0f;
        h_k[i] = vv_float_to_half(k);
        h_v[i] = vv_float_to_half(v);
    }
    for (size_t i = 0; i < q_elems; i++) h_q[i] = vv_float_to_half(frand());

    void *d_k = NULL, *d_v = NULL, *d_q = NULL, *d_qr = NULL, *d_o = NULL;
    void* d_scratch = NULL;
    if (vv_dev_alloc(&d_k, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&d_v, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&d_q, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&d_qr, q_elems * 2) != VV_OK ||
        vv_dev_alloc(&d_o, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&d_scratch,
                     vv_gqa_decode_scratch_bytes(N_Q_HEADS, HEAD_DIM))
            != VV_OK) {
        printf("SKIP: device allocation failed\n");
        return 0;
    }
    vv_dev_memcpy_h2d(d_k, h_k, kv_elems * 2, NULL);
    vv_dev_memcpy_h2d(d_v, h_v, kv_elems * 2, NULL);
    vv_dev_memcpy_h2d(d_q, h_q, q_elems * 2, NULL);

    vv_status_t s = vv_gqa_attention_decode_dev(
        d_q, d_k, d_v, d_o, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, N_POS,
        d_scratch, NULL);
    if (s != VV_OK) { printf("FAIL: fp16 decode returned %d\n", s); return 1; }
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(h_ref, d_o, q_elems * 2, NULL);

    /*
     * Minimum cosine similarity of the attention output. These are the
     * numbers the formats actually reach on this input; they exist to catch
     * regressions, not to express a quality target.
     */
    const struct { vv_kv_format_t fmt; double min_cos; } cases[] = {
        { VV_KV_FP8_E4M3, 0.9990 },
        { VV_KV_FP8_E5M2, 0.9950 },
        { VV_KV_TQ4,      0.9900 },
        { VV_KV_TQ3,      0.9600 },
        { VV_KV_TQ2,      0.8900 },
        { VV_KV_TQ1_5,    0.8000 },
    };

    int failures = 0;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const vv_kv_format_t fmt = cases[c].fmt;
        const int bpv = vv_kv_bytes_per_vec(fmt, HEAD_DIM);
        const size_t store_bytes = (size_t)N_POS * N_KV_HEADS * bpv;
        const size_t meta_bytes  = (size_t)N_POS * N_KV_HEADS * 2;

        void *ks = NULL, *vs = NULL, *km = NULL, *vm = NULL, *kr = NULL;
        vv_dev_alloc(&ks, store_bytes);
        vv_dev_alloc(&vs, store_bytes);
        vv_dev_alloc(&kr, (size_t)N_KV_HEADS * HEAD_DIM * 2);
        vv_dev_memset(ks, 0, store_bytes);
        vv_dev_memset(vs, 0, store_bytes);
        if (vv_kv_has_meta(fmt)) {
            vv_dev_alloc(&km, meta_bytes);
            vv_dev_alloc(&vm, meta_bytes);
            vv_dev_memset(km, 0, meta_bytes);
            vv_dev_memset(vm, 0, meta_bytes);
        }

        s = vv_kv_quant_store_dev(d_k, d_v, ks, vs, km, vm, kr, true,
                                  N_KV_HEADS, HEAD_DIM, 0, N_POS,
                                  (int)fmt, NULL);
        if (s != VV_OK) { printf("FAIL: store %s -> %d\n",
                                 vv_kv_format_name(fmt), s); failures++; continue; }

        /* Element-wise error of the store itself, before attention. */
        double kerr = 0.0, kmag = 0.0;
        {
            void* d_deq = NULL;
            vv_dev_alloc(&d_deq, kv_elems * 2);
            vv_kv_dequant_dev(ks, km, kr, d_deq, N_KV_HEADS, HEAD_DIM,
                              N_POS, (int)fmt, NULL);
            vv_dev_stream_sync(NULL);
            uint16_t* h_deq = malloc(kv_elems * 2);
            vv_dev_memcpy_d2h(h_deq, d_deq, kv_elems * 2, NULL);
            for (size_t i = 0; i < kv_elems; i++) {
                const double a = vv_half_to_float(h_k[i]);
                const double b = vv_half_to_float(h_deq[i]);
                kerr += (a - b) * (a - b);
                kmag += a * a;
            }
            kerr = sqrt(kerr / kmag);
            free(h_deq);
            vv_dev_free(d_deq);
        }

        vv_dev_memcpy_d2d(d_qr, d_q, q_elems * 2, NULL);
        if (vv_kv_rotates(fmt))
            vv_kv_rotate_dev(d_qr, N_Q_HEADS, HEAD_DIM, 1, NULL);

        s = vv_gqa_attention_decode_q_dev(d_qr, ks, vs, km, vm, d_o,
                                          N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                                          N_POS, (int)fmt, d_scratch, NULL);
        if (s != VV_OK) { printf("FAIL: decode %s -> %d\n",
                                 vv_kv_format_name(fmt), s); failures++; continue; }
        if (vv_kv_rotates(fmt))
            vv_kv_unrotate_dev(d_o, N_Q_HEADS, HEAD_DIM, 1, NULL);

        vv_dev_stream_sync(NULL);
        vv_dev_memcpy_d2h(h_got, d_o, q_elems * 2, NULL);

        const double cs = cosine(h_ref, h_got, q_elems);
        const int ok = cs >= cases[c].min_cos;
        printf("%-9s bytes/vec=%3d  K rel-rms=%.4f  attn cos=%.6f "
               "(min %.4f) %s\n",
               vv_kv_format_name(fmt), bpv, kerr, cs, cases[c].min_cos,
               ok ? "ok" : "FAIL");
        if (!ok) failures++;

        vv_dev_free(ks); vv_dev_free(vs); vv_dev_free(kr);
        if (km) vv_dev_free(km);
        if (vm) vv_dev_free(vm);
    }

    /* The prefill kernel must agree with the decode kernel on a single row. */
    {
        const vv_kv_format_t fmt = VV_KV_TQ4;
        const int bpv = vv_kv_bytes_per_vec(fmt, HEAD_DIM);
        void *ks = NULL, *vs = NULL, *km = NULL, *vm = NULL, *kr = NULL;
        vv_dev_alloc(&ks, (size_t)N_POS * N_KV_HEADS * bpv);
        vv_dev_alloc(&vs, (size_t)N_POS * N_KV_HEADS * bpv);
        vv_dev_alloc(&km, (size_t)N_POS * N_KV_HEADS * 2);
        vv_dev_alloc(&vm, (size_t)N_POS * N_KV_HEADS * 2);
        vv_dev_alloc(&kr, (size_t)N_KV_HEADS * HEAD_DIM * 2);
        vv_kv_quant_store_dev(d_k, d_v, ks, vs, km, vm, kr, true,
                              N_KV_HEADS, HEAD_DIM, 0, N_POS,
                              (int)fmt, NULL);

        vv_dev_memcpy_d2d(d_qr, d_q, q_elems * 2, NULL);
        vv_kv_rotate_dev(d_qr, N_Q_HEADS, HEAD_DIM, 1, NULL);
        vv_gqa_attention_decode_q_dev(d_qr, ks, vs, km, vm, d_o,
                                      N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                                      N_POS, (int)fmt, d_scratch, NULL);
        vv_kv_unrotate_dev(d_o, N_Q_HEADS, HEAD_DIM, 1, NULL);
        vv_dev_stream_sync(NULL);
        vv_dev_memcpy_d2h(h_ref, d_o, q_elems * 2, NULL);

        vv_dev_memcpy_d2d(d_qr, d_q, q_elems * 2, NULL);
        vv_kv_rotate_dev(d_qr, N_Q_HEADS, HEAD_DIM, 1, NULL);
        s = vv_gqa_attention_prefill_q_dev(d_qr, ks, vs, km, vm, d_o,
                                           N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                                           1, N_POS - 1, N_POS, true,
                                           (int)fmt, NULL);
        if (s != VV_OK) { printf("FAIL: prefill_q -> %d\n", s); failures++; }
        vv_kv_unrotate_dev(d_o, N_Q_HEADS, HEAD_DIM, 1, NULL);
        vv_dev_stream_sync(NULL);
        vv_dev_memcpy_d2h(h_got, d_o, q_elems * 2, NULL);

        const double cs = cosine(h_ref, h_got, q_elems);
        printf("%-9s prefill vs decode cos=%.6f %s\n", "tq4", cs,
               cs >= 0.9999 ? "ok" : "FAIL");
        if (cs < 0.9999) failures++;

        vv_dev_free(ks); vv_dev_free(vs); vv_dev_free(km);
        vv_dev_free(vm); vv_dev_free(kr);
    }

    vv_dev_free(d_k); vv_dev_free(d_v); vv_dev_free(d_q);
    vv_dev_free(d_qr); vv_dev_free(d_o);
    free(h_k); free(h_v); free(h_q); free(h_ref); free(h_got);

    printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
#endif
}
