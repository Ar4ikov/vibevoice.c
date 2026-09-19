/**
 * @file test_kv_pool.c
 * @brief The host side of the paged KV cache: a pool shared by several
 *        caches, reserve / release, and what happens when it runs dry.
 *
 * The kernels that read pages are tested in test_attn_backends against a
 * page table built by hand. This is the other half, the bookkeeping the
 * server's slots run concurrently: which pages a cache holds, what its device
 * table says, and that a short pool never passes for a full window. It used
 * to: pool exhaustion came back as VV_ERR_OVERFLOW, the decode loop read that
 * as "the window is full", and two concurrent requests that each fit their
 * window came back truncated with HTTP 200.
 *
 * Checked here:
 *  - pages for partial last pages, no-op re-reserves, the window bound;
 *  - the device page table matches the host's list after every change;
 *  - exhaustion is VV_ERR_KV_POOL_EXHAUSTED, never VV_ERR_OVERFLOW;
 *  - reserve_wait waits while another holder runs, and fails exactly one
 *    waiter when every holder is waiting (then the survivor proceeds);
 *  - a cache holding nothing waits for pages instead of failing;
 *  - pages come back on release and are handed out again;
 *  - four threads growing and releasing caches on a pool smaller than their
 *    sum never share a page, never hang, and leave the pool whole.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/inference.h"
#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"

#include "vv_thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYERS   2
#define N_KV_HEADS 4
#define HEAD_DIM   128
#define PG         VV_KV_PAGE_SIZE

static int failures = 0;

#define EXPECT(cond, ...) do {                                              \
        if (!(cond)) {                                                      \
            printf("FAIL line %d: ", __LINE__);                             \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
            failures++;                                                     \
        }                                                                   \
    } while (0)

/** @brief The device table must say what the host list says. */
static void check_table(const vv_kv_cache_t* c, void* stream, const char* what) {
    if (c->n_pages == 0) return;
    int* got = (int*)malloc((size_t)c->n_pages * sizeof(int));
    if (!got) { failures++; return; }
    vv_dev_stream_sync(stream);
    vv_dev_memcpy_d2h(got, c->page_table, (size_t)c->n_pages * sizeof(int),
                      stream);
    vv_dev_stream_sync(stream);
    for (int i = 0; i < c->n_pages; i++)
        EXPECT(got[i] == c->pages[i], "%s: table[%d] = %d on the device, "
               "%d on the host", what, i, got[i], c->pages[i]);
    free(got);
}

/** @brief No page held by two caches at once. */
static void check_disjoint(const vv_kv_cache_t* a, const vv_kv_cache_t* b) {
    for (int i = 0; i < a->n_pages; i++)
        for (int j = 0; j < b->n_pages; j++)
            EXPECT(a->pages[i] != b->pages[j], "page %d held twice",
                   a->pages[i]);
}

static void check_stats(const vv_kv_pool_t* p, int free_want, int holders_want,
                        int stalled_want, const char* what) {
    int f = -1, h = -1, st = -1;
    vv_kv_pool_stats(p, &f, &h, &st);
    EXPECT(f == free_want && h == holders_want && st == stalled_want,
           "%s: free %d holders %d stalled %d, want %d %d %d", what,
           f, h, st, free_want, holders_want, stalled_want);
}

/* ── A reserve_wait on its own thread ─────────────────────────────────── */

typedef struct {
    vv_kv_cache_t* c;
    int            n_positions;
    void*          stream;
    volatile int   done;
    vv_status_t    status;
} waiter_t;

static VV_THREAD_RET waiter_main(void* arg) {
    waiter_t* w = (waiter_t*)arg;
    w->status = vv_kv_cache_reserve_wait(w->c, w->n_positions, w->stream);
    w->done = 1;
    VV_THREAD_RETURN;
}

/** @brief Wait (bounded) until the pool shows `stalled` stalled holders. */
static bool wait_stalled(const vv_kv_pool_t* p, int stalled) {
    for (int i = 0; i < 500; i++) {
        int st = 0;
        vv_kv_pool_stats(p, NULL, NULL, &st);
        if (st == stalled) return true;
        vv_sleep_ms(10);
    }
    return false;
}

static void test_basics(void* stream) {
    vv_kv_pool_t* pool = NULL;
    vv_status_t s = vv_kv_pool_create(&pool, N_LAYERS, 0, N_LAYERS,
                                      N_KV_HEADS, HEAD_DIM, 8, VV_KV_FP16);
    EXPECT(s == VV_OK, "pool_create -> %d", s);
    if (s != VV_OK) return;
    vv_kv_cache_t *a = NULL, *b = NULL, *big = NULL;
    EXPECT(vv_kv_cache_create_paged(&a, pool, 8 * PG) == VV_OK, "create a");
    EXPECT(vv_kv_cache_create_paged(&b, pool, 8 * PG) == VV_OK, "create b");
    EXPECT(vv_kv_cache_create_paged(&big, pool, 16 * PG) == VV_OK, "create big");
    vv_kv_pool_release(pool);            /* the caches hold it now */
    if (!a || !b || !big) return;
    check_stats(pool, 8, 0, 0, "fresh");

    /* Partial pages round up; a re-reserve inside what is held is free. */
    EXPECT(vv_kv_cache_reserve(a, 100, stream) == VV_OK, "a 100");
    EXPECT(a->n_pages == 2, "a holds %d pages for 100 positions", a->n_pages);
    EXPECT(vv_kv_cache_reserve(a, 100, stream) == VV_OK, "a 100 again");
    EXPECT(vv_kv_cache_reserve(a, 128, stream) == VV_OK, "a 128");
    EXPECT(a->n_pages == 2, "a holds %d pages for 128 positions", a->n_pages);
    EXPECT(vv_kv_cache_reserve(a, 129, stream) == VV_OK, "a 129");
    EXPECT(a->n_pages == 3, "a holds %d pages for 129 positions", a->n_pages);
    check_table(a, stream, "a after 129");
    check_stats(pool, 5, 1, 0, "a holds 3");

    /* The window is the cache's own bound: that one is VV_ERR_OVERFLOW. */
    EXPECT(vv_kv_cache_reserve(a, 8 * PG + 1, stream) == VV_ERR_OVERFLOW,
           "past the window must be VV_ERR_OVERFLOW");

    /* The rest to b; then the pool is dry and says so, distinctly. */
    EXPECT(vv_kv_cache_reserve(b, 5 * PG, stream) == VV_OK, "b 5 pages");
    check_table(b, stream, "b");
    check_disjoint(a, b);
    check_stats(pool, 0, 2, 0, "pool dry");
    s = vv_kv_cache_reserve(b, 5 * PG + 1, stream);
    EXPECT(s == VV_ERR_KV_POOL_EXHAUSTED,
           "a dry pool returned %d (%s), not VV_ERR_KV_POOL_EXHAUSTED",
           s, vv_status_str(s));
    EXPECT(b->n_pages == 5, "a failed reserve changed b (%d pages)",
           b->n_pages);

    /* a waits for a page (b is still running, so that is worth doing);
     * then b asks too -- now both holders wait, nobody can free anything,
     * and b, the one that notices, fails at once instead of hanging. */
    waiter_t w;
    memset(&w, 0, sizeof(w));
    w.c = a; w.n_positions = 4 * PG; w.stream = stream;
    vv_thread_t th;
    EXPECT(vv_thread_start(&th, waiter_main, &w), "start waiter");
    EXPECT(wait_stalled(pool, 1), "a never showed up as stalled");
    EXPECT(!w.done, "a returned without a page to give it (status %d)",
           w.status);
    s = vv_kv_cache_reserve_wait(b, 6 * PG, stream);
    EXPECT(s == VV_ERR_KV_POOL_EXHAUSTED,
           "two holders both waiting: b got %d, want exhausted", s);
    check_stats(pool, 0, 2, 1, "a still waiting");
    /* b gives up its pages, as a failed request does; a proceeds. */
    vv_kv_cache_release(b);
    vv_thread_join(th);
    EXPECT(w.status == VV_OK, "a after b released: %d", w.status);
    EXPECT(a->n_pages == 4, "a holds %d pages, want 4", a->n_pages);
    check_table(a, stream, "a after the wait");
    check_stats(pool, 4, 1, 0, "a alone");

    /* Released pages are handed out again, and every id stays unique. */
    EXPECT(vv_kv_cache_reserve(b, 4 * PG, stream) == VV_OK, "b again");
    check_table(b, stream, "b reused");
    check_disjoint(a, b);
    {
        int seen[8] = {0};
        for (int i = 0; i < a->n_pages; i++) seen[a->pages[i]]++;
        for (int i = 0; i < b->n_pages; i++) seen[b->pages[i]]++;
        for (int i = 0; i < 8; i++)
            EXPECT(seen[i] == 1, "page %d is held %d times", i, seen[i]);
    }
    vv_kv_cache_release(a);
    vv_kv_cache_release(b);
    check_stats(pool, 8, 0, 0, "all back");

    /* A cache that holds nothing blocks nobody: it waits, and gets its
     * pages once they come back. */
    EXPECT(vv_kv_cache_reserve(a, 8 * PG, stream) == VV_OK, "a takes all");
    memset(&w, 0, sizeof(w));
    w.c = b; w.n_positions = PG; w.stream = stream;
    EXPECT(vv_thread_start(&th, waiter_main, &w), "start admission waiter");
    vv_sleep_ms(200);
    EXPECT(!w.done, "b returned %d while every page was taken", w.status);
    check_stats(pool, 0, 1, 0, "b waiting, not stalled");
    vv_kv_cache_reset(a, stream);        /* reset gives the pages back too */
    vv_thread_join(th);
    EXPECT(w.status == VV_OK && b->n_pages == 1,
           "admission wait ended with %d, %d pages", w.status, b->n_pages);
    check_table(b, stream, "b admitted");
    vv_kv_cache_release(b);

    /* More than the whole pool can never be granted: fail, do not wait. */
    s = vv_kv_cache_reserve_wait(big, 9 * PG, stream);
    EXPECT(s == VV_ERR_KV_POOL_EXHAUSTED,
           "a request larger than the pool got %d", s);
    check_stats(pool, 8, 0, 0, "end");

    vv_kv_cache_free(a);
    vv_kv_cache_free(b);
    vv_kv_cache_free(big);
    printf("basics: done\n");
}

/* ── Contention ───────────────────────────────────────────────────────── */

#define ST_THREADS 4
#define ST_PAGES   16
#define ST_MAX     8          /* pages one cache may grow to */
#define ST_ITERS   300

typedef struct {
    int            id;
    vv_kv_cache_t* c;
    void*          stream;
    vv_mutex_t*    lock;
    int*           owner;     /* [ST_PAGES], -1 = free */
    uint32_t       rng;
    int            exhausted, granted, double_owned;
    vv_status_t    status;
} stress_t;

static int stress_rand(stress_t* t, int n) {
    t->rng = t->rng * 1664525u + 1013904223u;
    return (int)((t->rng >> 8) % (uint32_t)n);
}

static void disown(stress_t* t) {
    vv_mutex_lock(t->lock);
    for (int i = 0; i < t->c->n_pages; i++)
        if (t->owner[t->c->pages[i]] == t->id) t->owner[t->c->pages[i]] = -1;
    vv_mutex_unlock(t->lock);
}

static VV_THREAD_RET stress_main(void* arg) {
    stress_t* t = (stress_t*)arg;
    for (int it = 0; it < ST_ITERS && t->status == VV_OK; it++) {
        const int target = 1 + stress_rand(t, ST_MAX);
        int held = 0;
        while (held < target) {
            int step = 1 + stress_rand(t, 3);
            if (held + step > target) step = target - held;
            const vv_status_t s = vv_kv_cache_reserve_wait(
                t->c, (held + step) * PG - stress_rand(t, PG), t->stream);
            if (s == VV_ERR_KV_POOL_EXHAUSTED) { t->exhausted++; break; }
            if (s != VV_OK) { t->status = s; break; }
            t->granted++;
            vv_mutex_lock(t->lock);
            for (int i = held; i < t->c->n_pages; i++) {
                const int pg = t->c->pages[i];
                if (t->owner[pg] != -1) t->double_owned++;
                t->owner[pg] = t->id;
            }
            vv_mutex_unlock(t->lock);
            held = t->c->n_pages;
        }
        disown(t);
        vv_dev_stream_sync(t->stream);
        vv_kv_cache_release(t->c);
    }
    VV_THREAD_RETURN;
}

static void test_contention(void) {
    vv_kv_pool_t* pool = NULL;
    if (vv_kv_pool_create(&pool, N_LAYERS, 0, N_LAYERS, N_KV_HEADS,
                          HEAD_DIM, ST_PAGES, VV_KV_TQ4) != VV_OK) {
        printf("FAIL: contention pool\n");
        failures++;
        return;
    }
    vv_mutex_t lock;
    vv_mutex_init(&lock);
    int owner[ST_PAGES];
    for (int i = 0; i < ST_PAGES; i++) owner[i] = -1;

    stress_t t[ST_THREADS];
    memset(t, 0, sizeof(t));
    for (int i = 0; i < ST_THREADS; i++) {
        t[i].id = i;
        t[i].lock = &lock;
        t[i].owner = owner;
        t[i].rng = 777u + 31u * (uint32_t)i;
        if (vv_kv_cache_create_paged(&t[i].c, pool, ST_MAX * PG) != VV_OK ||
            vv_dev_stream_create(&t[i].stream) != VV_OK) {
            printf("FAIL: contention cache %d\n", i);
            failures++;
            return;
        }
    }
    vv_kv_pool_release(pool);

    const double t0 = vv_time_ms();
    vv_thread_t th[ST_THREADS];
    for (int i = 0; i < ST_THREADS; i++)
        EXPECT(vv_thread_start(&th[i], stress_main, &t[i]), "start %d", i);
    for (int i = 0; i < ST_THREADS; i++) vv_thread_join(th[i]);

    int exhausted = 0, granted = 0;
    for (int i = 0; i < ST_THREADS; i++) {
        EXPECT(t[i].status == VV_OK, "thread %d: %d", i, t[i].status);
        EXPECT(t[i].double_owned == 0, "thread %d got %d pages another "
               "thread held", i, t[i].double_owned);
        exhausted += t[i].exhausted;
        granted += t[i].granted;
    }
    check_stats(pool, ST_PAGES, 0, 0, "after contention");
    printf("contention: %d threads x %d requests on %d pages (sum of maxima "
           "%d): %d grants, %d explicit exhaustions, %.0f ms\n",
           ST_THREADS, ST_ITERS, ST_PAGES, ST_THREADS * ST_MAX, granted,
           exhausted, vv_time_ms() - t0);

    for (int i = 0; i < ST_THREADS; i++) {
        vv_kv_cache_free(t[i].c);
        vv_dev_stream_destroy(t[i].stream);
    }
    vv_mutex_destroy(&lock);
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
    void* stream = NULL;
    if (vv_dev_stream_create(&stream) != VV_OK) {
        printf("SKIP: no stream\n");
        return 0;
    }
    test_basics(stream);
    test_contention();
    vv_dev_stream_destroy(stream);
    if (failures) {
        printf("FAIL: %d check(s)\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
#endif
}
