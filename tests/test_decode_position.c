/**
 * @file test_decode_position.c
 * @brief Reading the position from device memory must change nothing.
 *
 * A decode step is captured once and replayed for every token, which only
 * works because nothing in it carries the position as a kernel argument: RoPE,
 * the cache write and the attention length all read a device int instead. That
 * is a rewrite of the hot path for a 4% gain, so the bar is exact equality —
 * every one of these kernels, given a device int holding the same number the
 * host would have passed, has to produce the identical bytes.
 *
 * A position bug here would not crash. It would shift one token's rotation by
 * one, or attend over one key too few, and the transcript would come back
 * plausible and wrong.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_Q_HEADS  28
#define N_KV_HEADS 4
#define HEAD_DIM   128
#define N_POS      1500        /* two split buckets away from a boundary */

static uint32_t rng_state = 0xB5297A4Du;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;
}

static int failures = 0;

static void check(bool ok, const char* what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

/** @brief A device int holding `v`. */
static int* dev_int(int v) {
    void* p = NULL;
    if (vv_dev_alloc(&p, sizeof(int)) != VV_OK) return NULL;
    vv_dev_memcpy_h2d(p, &v, sizeof(int), NULL);
    vv_dev_stream_sync(NULL);
    return (int*)p;
}

static int read_dev_int(const int* p) {
    int v = -1;
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(&v, p, sizeof(int), NULL);
    return v;
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

    /* ── The position itself ── */
    {
        int* a = dev_int(41);
        int* b = dev_int(0);
        if (!a || !b) { printf("SKIP: allocation\n"); return 0; }
        check(vv_pos_add_dev(b, a, 1, NULL) == VV_OK, "pos_add returns ok");
        check(read_dev_int(b) == 42, "pos_add writes src + delta");
        check(vv_pos_add_dev(a, a, 1, NULL) == VV_OK &&
              read_dev_int(a) == 42, "pos_add in place");
        vv_dev_free(a); vv_dev_free(b);
    }

    /* ── Writing one row at a device-side index ── */
    {
        const size_t row = 512;
        const int at = 7;
        void *base = NULL, *src = NULL;
        vv_dev_alloc(&base, row * 16);
        vv_dev_alloc(&src, row);
        vv_dev_memset(base, 0, row * 16);

        uint8_t* pattern = (uint8_t*)malloc(row);
        for (size_t i = 0; i < row; i++) pattern[i] = (uint8_t)(i * 7 + 3);
        vv_dev_memcpy_h2d(src, pattern, row, NULL);

        int* idx = dev_int(at);
        check(vv_dev_memcpy_d2d_at(base, src, row, idx, NULL) == VV_OK,
              "memcpy_d2d_at returns ok");
        vv_dev_stream_sync(NULL);

        uint8_t* got = (uint8_t*)malloc(row * 16);
        vv_dev_memcpy_d2h(got, base, row * 16, NULL);
        bool placed = memcmp(got + (size_t)at * row, pattern, row) == 0;
        bool clean = true;
        for (size_t i = 0; i < row * 16 && clean; i++)
            if (i < (size_t)at * row || i >= (size_t)(at + 1) * row)
                clean = (got[i] == 0);
        check(placed, "memcpy_d2d_at lands at (*d_index) * size");
        check(clean, "memcpy_d2d_at touches nothing else");

        free(pattern); free(got);
        vv_dev_free(base); vv_dev_free(src); vv_dev_free(idx);
    }

    /* ── The kernels that read the length ── */
    const size_t kv_elems = (size_t)N_POS * N_KV_HEADS * HEAD_DIM;
    const size_t q_elems  = (size_t)N_Q_HEADS * HEAD_DIM;
    const int    bpv      = vv_kv_bytes_per_vec(VV_KV_TQ4, HEAD_DIM);

    uint16_t* h_buf = (uint16_t*)malloc(kv_elems * 2);
    uint16_t* h_q   = (uint16_t*)malloc(q_elems * 2);
    uint16_t* h_a   = (uint16_t*)malloc(q_elems * 2);
    uint16_t* h_b   = (uint16_t*)malloc(q_elems * 2);
    if (!h_buf || !h_q || !h_a || !h_b) return 1;

    void *dK = NULL, *dV = NULL, *dQ = NULL, *dO = NULL, *scratch = NULL;
    void *ks = NULL, *vs = NULL, *km = NULL, *vm = NULL, *kr = NULL;
    void *ks2 = NULL, *vs2 = NULL, *km2 = NULL, *vm2 = NULL, *kr2 = NULL;
    const size_t store_bytes = (size_t)N_POS * N_KV_HEADS * bpv;
    const size_t meta_size   = (size_t)N_POS * N_KV_HEADS * 2;
    if (vv_dev_alloc(&dK, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dV, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&dQ, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&dO, q_elems * 2)  != VV_OK ||
        vv_dev_alloc(&scratch,
                     vv_gqa_decode_scratch_bytes(N_Q_HEADS, HEAD_DIM)) != VV_OK ||
        vv_dev_alloc(&ks, store_bytes) != VV_OK ||
        vv_dev_alloc(&vs, store_bytes) != VV_OK ||
        vv_dev_alloc(&km, meta_size)   != VV_OK ||
        vv_dev_alloc(&vm, meta_size)   != VV_OK ||
        vv_dev_alloc(&kr, (size_t)N_KV_HEADS * HEAD_DIM * 2) != VV_OK ||
        vv_dev_alloc(&ks2, store_bytes) != VV_OK ||
        vv_dev_alloc(&vs2, store_bytes) != VV_OK ||
        vv_dev_alloc(&km2, meta_size)   != VV_OK ||
        vv_dev_alloc(&vm2, meta_size)   != VV_OK ||
        vv_dev_alloc(&kr2, (size_t)N_KV_HEADS * HEAD_DIM * 2) != VV_OK) {
        printf("SKIP: device allocation failed\n");
        return 0;
    }

    for (size_t i = 0; i < kv_elems; i++) h_buf[i] = vv_float_to_half(frand() * 2.f);
    vv_dev_memcpy_h2d(dK, h_buf, kv_elems * 2, NULL);
    for (size_t i = 0; i < kv_elems; i++) h_buf[i] = vv_float_to_half(frand() * 1.5f);
    vv_dev_memcpy_h2d(dV, h_buf, kv_elems * 2, NULL);
    for (size_t i = 0; i < q_elems; i++) h_q[i] = vv_float_to_half(frand());
    vv_dev_memcpy_h2d(dQ, h_q, q_elems * 2, NULL);
    vv_dev_stream_sync(NULL);

    int* d_len = dev_int(N_POS);

    /* FP16 decode: host length against device length. */
    vv_gqa_attention_decode_dev(dQ, dK, dV, dO, N_Q_HEADS, N_KV_HEADS,
                                HEAD_DIM, N_POS, NULL, scratch, NULL);
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(h_a, dO, q_elems * 2, NULL);
    vv_dev_memset(dO, 0, q_elems * 2);
    vv_gqa_attention_decode_dev(dQ, dK, dV, dO, N_Q_HEADS, N_KV_HEADS,
                                HEAD_DIM, N_POS, d_len, scratch, NULL);
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(h_b, dO, q_elems * 2, NULL);
    check(memcmp(h_a, h_b, q_elems * 2) == 0,
          "fp16 decode: device length matches host length exactly");

    /* Quantized store: the same, for both the store and the decode over it. */
    vv_status_t s1 = vv_kv_quant_store_dev(dK, dV, ks, vs, km, vm, kr, true,
                                           N_KV_HEADS, HEAD_DIM, 0, NULL,
                                           N_POS, (int)VV_KV_TQ4, NULL);
    int* d_zero = dev_int(0);
    vv_status_t s2 = vv_kv_quant_store_dev(dK, dV, ks2, vs2, km2, vm2, kr2, true,
                                           N_KV_HEADS, HEAD_DIM, 0, d_zero,
                                           N_POS, (int)VV_KV_TQ4, NULL);
    vv_dev_stream_sync(NULL);
    check(s1 == VV_OK && s2 == VV_OK, "quantized store returns ok both ways");
    {
        uint8_t* a = (uint8_t*)malloc(store_bytes);
        uint8_t* b = (uint8_t*)malloc(store_bytes);
        vv_dev_memcpy_d2h(a, ks, store_bytes, NULL);
        vv_dev_memcpy_d2h(b, ks2, store_bytes, NULL);
        check(memcmp(a, b, store_bytes) == 0,
              "quantized store: device position matches host position exactly");
        free(a); free(b);
    }

    vv_dev_memset(dO, 0, q_elems * 2);
    vv_gqa_attention_decode_q_dev(dQ, ks, vs, km, vm, dO, N_Q_HEADS,
                                  N_KV_HEADS, HEAD_DIM, N_POS, NULL,
                                  (int)VV_KV_TQ4, scratch, NULL);
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(h_a, dO, q_elems * 2, NULL);
    vv_dev_memset(dO, 0, q_elems * 2);
    vv_gqa_attention_decode_q_dev(dQ, ks, vs, km, vm, dO, N_Q_HEADS,
                                  N_KV_HEADS, HEAD_DIM, N_POS, d_len,
                                  (int)VV_KV_TQ4, scratch, NULL);
    vv_dev_stream_sync(NULL);
    vv_dev_memcpy_d2h(h_b, dO, q_elems * 2, NULL);
    check(memcmp(h_a, h_b, q_elems * 2) == 0,
          "tq4 decode: device length matches host length exactly");

    /*
     * The launch shape has to hold still inside a bucket and move between
     * them, or a captured graph would either be re-made every token or keep a
     * grid too small for the cache it is walking.
     */
    check(vv_gqa_decode_shape(1) == vv_gqa_decode_shape(1024),
          "launch shape holds across a bucket");
    check(vv_gqa_decode_shape(1024) != vv_gqa_decode_shape(1025),
          "launch shape moves at the bucket edge");
    check(vv_gqa_decode_shape(1025) == vv_gqa_decode_shape(2048),
          "launch shape holds across the next bucket");

    /* RoPE, at a position only the device knows. */
    {
        const size_t n = q_elems;
        void *xa = NULL, *xb = NULL;
        vv_dev_alloc(&xa, n * 2);
        vv_dev_alloc(&xb, n * 2);
        vv_dev_memcpy_h2d(xa, h_q, n * 2, NULL);
        vv_dev_memcpy_h2d(xb, h_q, n * 2, NULL);
        int* d_p = dev_int(913);
        vv_rope_dev(xa, 1, N_Q_HEADS, HEAD_DIM, 913, NULL, 1000000.0f, NULL);
        vv_rope_dev(xb, 1, N_Q_HEADS, HEAD_DIM, 0, d_p, 1000000.0f, NULL);
        vv_dev_stream_sync(NULL);
        vv_dev_memcpy_d2h(h_a, xa, n * 2, NULL);
        vv_dev_memcpy_d2h(h_b, xb, n * 2, NULL);
        check(memcmp(h_a, h_b, n * 2) == 0,
              "rope: device position matches host position exactly");
        check(memcmp(h_a, h_q, n * 2) != 0, "rope actually rotated something");
        vv_dev_free(xa); vv_dev_free(xb); vv_dev_free(d_p);
    }

    vv_dev_free(dK); vv_dev_free(dV); vv_dev_free(dQ); vv_dev_free(dO);
    vv_dev_free(scratch); vv_dev_free(d_len); vv_dev_free(d_zero);
    vv_dev_free(ks); vv_dev_free(vs); vv_dev_free(km); vv_dev_free(vm);
    vv_dev_free(kr);
    vv_dev_free(ks2); vv_dev_free(vs2); vv_dev_free(km2); vv_dev_free(vm2);
    vv_dev_free(kr2);
    free(h_buf); free(h_q); free(h_a); free(h_b);

    if (failures) {
        printf("=== %d check(s) failed ===\n", failures);
        return 1;
    }
    printf("=== the device-side position is exactly the host-side one ===\n");
    return 0;
#endif
}
