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
 *
 * Then the same again on a paged cache, the way the server's slots run: each
 * worker owns a cache whose pages come from one pool shared by all of them,
 * too small for everyone at once. Every round a worker gives its pages back,
 * waits for a fresh set (whichever pages the others left), writes K and V
 * into them and decodes; the answer must not depend on which pages it got.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
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

    /* Paged phase: this worker's caches on the shared pools (NULL = the
     * resolved backend does not read pages for that format). */
    vv_kv_cache_t* pc[2];
    uint16_t* expect_paged;
    int      paged_mismatches;

    /* Shared, read-only. */
    const void* k;
    const void* v;
    const void* ks;
    const void* vs;
    const void* km;
    const void* vm;
} worker_t;

static size_t q_elems(void) { return (size_t)N_Q_HEADS * HEAD_DIM; }

/* The backend under test (VV_ATTN, as ctest sets it per run) for each of the
 * two caches. Resolved once, before any thread starts. */
static int be_raw = 0, be_q = 0;
/* ... and for the paged caches, which may resolve differently. */
static int be_paged[2] = {0, 0};
static const int paged_format[2] = {VV_KV_FP16, VV_KV_TQ4};
#define N_PAGED_ITERS 40

/** @brief One decode through both the raw and the quantized kernel. */
static vv_status_t decode_once(worker_t* w, uint16_t* dst) {
    vv_kv_view_t kv;
    memset(&kv, 0, sizeof(kv));
    kv.k = w->k; kv.v = w->v; kv.format = VV_KV_FP16;
    kv.n_kv_heads = N_KV_HEADS; kv.head_dim = HEAD_DIM;
    vv_status_t s = vv_attn_decode(be_raw, w->q, &kv, w->out, N_Q_HEADS,
                                   N_POS, NULL, w->scratch, w->stream);
    if (s != VV_OK) return s;
    vv_dev_stream_sync(w->stream);
    vv_dev_memcpy_d2h(dst, w->out, q_elems() * 2, w->stream);
    vv_dev_stream_sync(w->stream);

    kv.k = w->ks; kv.v = w->vs; kv.k_meta = w->km; kv.v_meta = w->vm;
    kv.format = VV_KV_TQ4;
    s = vv_attn_decode(be_q, w->q, &kv, w->out, N_Q_HEADS, N_POS, NULL,
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

/**
 * @brief One paged round per format: fresh pages from the pool (waiting for
 *        them if the others hold them), K/V written into them, a decode.
 */
static vv_status_t paged_once(worker_t* w, uint16_t* dst) {
    for (int f = 0; f < 2; f++) {
        vv_kv_cache_t* c = w->pc[f];
        if (!c) continue;
        vv_status_t s = vv_kv_cache_reset(c, w->stream);
        if (s == VV_OK) s = vv_kv_cache_reserve_wait(c, N_POS, w->stream);
        if (s == VV_OK)
            s = vv_kv_cache_append(c, 0, w->k, w->v, N_POS, false, w->stream);
        if (s != VV_OK) return s;
        const vv_kv_view_t kv = vv_kv_cache_view(c, 0);
        s = vv_attn_decode(be_paged[f], w->q, &kv, w->out, N_Q_HEADS, N_POS,
                           NULL, w->scratch, w->stream);
        if (s != VV_OK) return s;
        vv_dev_memcpy_d2h(dst + (size_t)f * q_elems(), w->out, q_elems() * 2,
                          w->stream);
        vv_dev_stream_sync(w->stream);
        /* Done with them: another worker may be waiting. */
        vv_kv_cache_release(c);
    }
    return VV_OK;
}

static VV_THREAD_RET paged_main(void* arg) {
    worker_t* w = (worker_t*)arg;
    for (int i = 0; i < N_PAGED_ITERS && w->status == VV_OK; i++) {
        w->status = paged_once(w, w->got);
        if (w->status != VV_OK) break;
        if (memcmp(w->got, w->expect_paged, q_elems() * 2 * 2) != 0)
            w->paged_mismatches++;
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
        vv_attn_scratch_bytes(N_Q_HEADS, N_KV_HEADS, HEAD_DIM);
    const int want = (int)vv_attn_backend_from_env(VV_ATTN_AUTO);
    be_raw = vv_attn_resolve(want, VV_KV_FP16, false, N_Q_HEADS, N_KV_HEADS,
                             HEAD_DIM);
    be_q = vv_attn_resolve(want, VV_KV_TQ4, false, N_Q_HEADS, N_KV_HEADS,
                           HEAD_DIM);
    printf("backends: fp16 %s, tq4 %s\n",
           vv_attn_backend_name((vv_attn_backend_t)be_raw),
           vv_attn_backend_name((vv_attn_backend_t)be_q));

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

    /*
     * Paged. One pool per format with room for two workers' windows and a
     * half: two decode while the other two wait for pages.
     */
    const int per_cache = (N_POS + VV_KV_PAGE_SIZE - 1) / VV_KV_PAGE_SIZE;
    int n_paged = 0;
    for (int f = 0; f < 2; f++) {
        be_paged[f] = vv_attn_resolve(want, paged_format[f], true, N_Q_HEADS,
                                      N_KV_HEADS, HEAD_DIM);
        if (be_paged[f] == VV_ATTN_FA1) {
            printf("paged %s: no backend reads pages here, skipped\n",
                   f ? "tq4" : "fp16");
            continue;
        }
        vv_kv_pool_t* pool = NULL;
        if (vv_kv_pool_create(&pool, 1, 0, 1, N_KV_HEADS, HEAD_DIM,
                              per_cache * 5 / 2, paged_format[f]) != VV_OK) {
            printf("FAIL: paged pool\n");
            return 1;
        }
        for (int i = 0; i < N_THREADS; i++) {
            if (vv_kv_cache_create_paged(&w[i].pc[f], pool, N_POS) != VV_OK) {
                printf("FAIL: paged cache\n");
                return 1;
            }
            w[i].pc[f]->attn_backend = be_paged[f];
        }
        vv_kv_pool_release(pool);       /* the caches hold it */
        n_paged++;
    }
    if (n_paged) {
        printf("paged backends: fp16 %s, tq4 %s\n",
               vv_attn_backend_name((vv_attn_backend_t)be_paged[0]),
               vv_attn_backend_name((vv_attn_backend_t)be_paged[1]));
        for (int i = 0; i < N_THREADS; i++) {
            w[i].expect_paged = (uint16_t*)calloc(q_elems() * 2, 2);
            if (!w[i].expect_paged) return 1;
            w[i].status = VV_OK;
            vv_status_t s = paged_once(&w[i], w[i].expect_paged);
            if (s != VV_OK) {
                printf("FAIL: serial paged decode %d -> %d\n", i, s);
                return 1;
            }
        }
        /*
         * Paged is contiguous with the pages moved: same backend, same bits.
         * fa2's decode is fa1's bit for bit, so either slab answer will do.
         */
        const bool same_fp16 = be_paged[0] == be_raw ||
            (be_paged[0] == VV_ATTN_FA2 && be_raw == VV_ATTN_FA1);
        if (w[0].pc[0] && same_fp16) {
            for (int i = 0; i < N_THREADS; i++)
                if (memcmp(w[i].expect_paged, w[i].expect,
                           q_elems() * 2) != 0) {
                    printf("FAIL: worker %d: paged fp16 decode differs from "
                           "the slab\n", i);
                    failures++;
                }
        }
        for (int i = 0; i < N_THREADS; i++) {
            memset(w[i].got, 0, q_elems() * 2 * 2);
            if (!vv_thread_start(&th[i], paged_main, &w[i])) {
                printf("FAIL: cannot start thread %d\n", i);
                return 1;
            }
        }
        for (int i = 0; i < N_THREADS; i++) vv_thread_join(th[i]);
        for (int i = 0; i < N_THREADS; i++) {
            if (w[i].status != VV_OK) {
                printf("FAIL: paged worker %d returned %d\n", i, w[i].status);
                failures++;
            } else if (w[i].paged_mismatches) {
                printf("FAIL: paged worker %d differed in %d of %d rounds\n",
                       i, w[i].paged_mismatches, N_PAGED_ITERS);
                failures++;
            } else {
                printf("worker %d: %d paged rounds on a shared pool, all "
                       "identical\n", i, N_PAGED_ITERS);
            }
        }
        for (int f = 0; f < 2; f++) {
            if (!w[0].pc[f]) continue;
            int n_free = 0;
            const int n_pages = vv_kv_pool_pages(w[0].pc[f]->pool, &n_free);
            if (n_free != n_pages) {
                printf("FAIL: %d of %d pages did not come back\n",
                       n_pages - n_free, n_pages);
                failures++;
            }
        }
    }

    for (int i = 0; i < N_THREADS; i++) {
        for (int f = 0; f < 2; f++)
            if (w[i].pc[f]) vv_kv_cache_free(w[i].pc[f]);
        free(w[i].expect_paged);
        vv_dev_free(w[i].q); vv_dev_free(w[i].out); vv_dev_free(w[i].scratch);
        vv_dev_stream_destroy(w[i].stream);
        free(w[i].expect); free(w[i].got);
    }
    vv_dev_free(d_k); vv_dev_free(d_v);
    vv_dev_free(d_ks); vv_dev_free(d_vs);
    vv_dev_free(d_km); vv_dev_free(d_vm); vv_dev_free(d_kr);
    free(h_kv); free(h_q);

    if (failures) return 1;
    printf("PASS: %d threads x %d decodes agree with the serial answer%s\n",
           N_THREADS, N_ITERS, n_paged ? ", slab and paged" : "");
    return 0;
#endif
}
