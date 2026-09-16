/**
 * @file test_concurrent_decode.c
 * @brief Decode attention must give one answer whether or not other threads
 *        are decoding at the same time.
 *
 * The server runs several inference contexts against one copy of the weights,
 * each on its own stream and its own thread. For a while the split-K decode
 * kernels kept their per-slice softmax partials in a file-scope buffer, so two
 * requests in flight merged each other's slices and the second transcript
 * quietly drifted — words doubled, endings changed. Nothing crashed, which is
 * why it survived.
 *
 * So: compute each thread's answer serially first, then have every thread
 * hammer the same kernels at once and demand the identical bytes back. Same
 * kernel, same inputs, same split count — bit-exact is the right bar.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include "vv_thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_Q_HEADS   28
#define N_KV_HEADS  4
#define HEAD_DIM    128
#define N_POS       1023        /* deliberately not a split multiple */
#define N_THREADS   4
#define N_ITERS     150

static uint32_t rng_state = 9876u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

/** Everything one worker owns; only the K/V cache is shared, and read-only. */
typedef struct {
    int      id;
    void*    q;            /* device, this worker's query rows       */
    void*    out;          /* device, scratch output                 */
    void*    scratch;      /* device, split-K partials               */
    void*    stream;
    uint16_t* expect;      /* host, the serial answer                */
    uint16_t* got;         /* host, read back under load             */
    int      mismatches;
    vv_status_t status;

    /* Shared, read-only. */
    const void* k;
    const void* v;
    const void* ks;
    const void* vs;
    const void* km;
    const void* vm;
} worker_t;

static size_t q_elems(void) { return (size_t)N_Q_HEADS * HEAD_DIM; }

/** @brief One decode through both the raw and the quantized kernel. */
static vv_status_t decode_once(worker_t* w, uint16_t* dst) {
    vv_status_t s = vv_gqa_attention_decode_dev(
        w->q, w->k, w->v, w->out, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, N_POS,
        NULL, w->scratch, w->stream);
    if (s != VV_OK) return s;
    vv_dev_stream_sync(w->stream);
    vv_dev_memcpy_d2h(dst, w->out, q_elems() * 2, w->stream);
    vv_dev_stream_sync(w->stream);

    s = vv_gqa_attention_decode_q_dev(
        w->q, w->ks, w->vs, w->km, w->vm, w->out,
        N_Q_HEADS, N_KV_HEADS, HEAD_DIM, N_POS, NULL, (int)VV_KV_TQ4,
        w->scratch, w->stream);
    if (s != VV_OK) return s;
    vv_dev_stream_sync(w->stream);
    vv_dev_memcpy_d2h(dst + q_elems(), w->out, q_elems() * 2, w->stream);
    vv_dev_stream_sync(w->stream);
    return VV_OK;
}

static VV_THREAD_RET worker_main(void* arg) {
    worker_t* w = (worker_t*)arg;
    for (int i = 0; i < N_ITERS && w->status == VV_OK; i++) {
        w->status = decode_once(w, w->got);
        if (w->status != VV_OK) break;
        if (memcmp(w->got, w->expect, q_elems() * 2 * 2) != 0)
            w->mismatches++;
    }
    VV_THREAD_RETURN;
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
    const int    bpv      = vv_kv_bytes_per_vec(VV_KV_TQ4, HEAD_DIM);
    const size_t store_bytes = (size_t)N_POS * N_KV_HEADS * bpv;
    const size_t meta_bytes  = (size_t)N_POS * N_KV_HEADS * 2;

    uint16_t* h_kv = (uint16_t*)malloc(kv_elems * 2);
    uint16_t* h_q  = (uint16_t*)malloc(q_elems() * 2);
    if (!h_kv || !h_q) return 1;

    void *d_k = NULL, *d_v = NULL;
    void *d_ks = NULL, *d_vs = NULL, *d_km = NULL, *d_vm = NULL, *d_kr = NULL;
    if (vv_dev_alloc(&d_k, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&d_v, kv_elems * 2) != VV_OK ||
        vv_dev_alloc(&d_ks, store_bytes) != VV_OK ||
        vv_dev_alloc(&d_vs, store_bytes) != VV_OK ||
        vv_dev_alloc(&d_km, meta_bytes)  != VV_OK ||
        vv_dev_alloc(&d_vm, meta_bytes)  != VV_OK ||
        vv_dev_alloc(&d_kr, (size_t)N_KV_HEADS * HEAD_DIM * 2) != VV_OK) {
        printf("SKIP: device allocation failed\n");
        return 0;
    }

    /* One cache, shared by every worker the way the weights are shared. */
    for (size_t i = 0; i < kv_elems; i++) h_kv[i] = vv_float_to_half(frand() * 2.0f);
    vv_dev_memcpy_h2d(d_k, h_kv, kv_elems * 2, NULL);
    for (size_t i = 0; i < kv_elems; i++) h_kv[i] = vv_float_to_half(frand() * 1.5f);
    vv_dev_memcpy_h2d(d_v, h_kv, kv_elems * 2, NULL);
    if (vv_kv_quant_store_dev(d_k, d_v, d_ks, d_vs, d_km, d_vm, d_kr, true,
                              N_KV_HEADS, HEAD_DIM, 0, NULL, N_POS,
                              (int)VV_KV_TQ4, NULL) != VV_OK) {
        printf("FAIL: quantized store failed\n");
        return 1;
    }
    vv_dev_stream_sync(NULL);

    worker_t w[N_THREADS];
    memset(w, 0, sizeof(w));
    const size_t scratch_bytes =
        vv_gqa_decode_scratch_bytes(N_Q_HEADS, HEAD_DIM);

    for (int i = 0; i < N_THREADS; i++) {
        w[i].id = i;
        w[i].k = d_k; w[i].v = d_v;
        w[i].ks = d_ks; w[i].vs = d_vs; w[i].km = d_km; w[i].vm = d_vm;
        w[i].expect = (uint16_t*)malloc(q_elems() * 2 * 2);
        w[i].got    = (uint16_t*)malloc(q_elems() * 2 * 2);
        if (!w[i].expect || !w[i].got) return 1;
        if (vv_dev_alloc(&w[i].q, q_elems() * 2) != VV_OK ||
            vv_dev_alloc(&w[i].out, q_elems() * 2) != VV_OK ||
            vv_dev_alloc(&w[i].scratch, scratch_bytes) != VV_OK ||
            vv_dev_stream_create(&w[i].stream) != VV_OK) {
            printf("SKIP: per-worker allocation failed\n");
            return 0;
        }
        /* Distinct queries, so a swapped answer is visible rather than lucky. */
        for (size_t e = 0; e < q_elems(); e++)
            h_q[e] = vv_float_to_half(frand());
        vv_dev_memcpy_h2d(w[i].q, h_q, q_elems() * 2, NULL);
        vv_dev_stream_sync(NULL);
    }

    /* Reference: one worker at a time, nothing else on the device. */
    for (int i = 0; i < N_THREADS; i++) {
        vv_status_t s = decode_once(&w[i], w[i].expect);
        if (s != VV_OK) { printf("FAIL: serial decode %d -> %d\n", i, s); return 1; }
    }

    vv_thread_t th[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        if (!vv_thread_start(&th[i], worker_main, &w[i])) {
            printf("FAIL: cannot start thread %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < N_THREADS; i++) vv_thread_join(th[i]);

    int failures = 0;
    for (int i = 0; i < N_THREADS; i++) {
        if (w[i].status != VV_OK) {
            printf("FAIL: worker %d returned %d\n", i, w[i].status);
            failures++;
        } else if (w[i].mismatches) {
            printf("FAIL: worker %d differed from its serial answer in "
                   "%d of %d decodes\n", i, w[i].mismatches, N_ITERS);
            failures++;
        } else {
            printf("worker %d: %d concurrent decodes, all identical\n",
                   i, N_ITERS);
        }
    }

    for (int i = 0; i < N_THREADS; i++) {
        vv_dev_free(w[i].q); vv_dev_free(w[i].out); vv_dev_free(w[i].scratch);
        vv_dev_stream_destroy(w[i].stream);
        free(w[i].expect); free(w[i].got);
    }
    vv_dev_free(d_k); vv_dev_free(d_v);
    vv_dev_free(d_ks); vv_dev_free(d_vs);
    vv_dev_free(d_km); vv_dev_free(d_vm); vv_dev_free(d_kr);
    free(h_kv); free(h_q);

    if (failures) return 1;
    printf("PASS: %d threads x %d decodes agree with the serial answer\n",
           N_THREADS, N_ITERS);
    return 0;
#endif
}
