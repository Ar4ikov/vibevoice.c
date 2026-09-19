/**
 * @file test_device_memory.c
 * @brief The memory, stream, event and graph contract of device.h, on
 *        whichever backend is built in.
 *
 * Every kernel test assumes these work; this one checks them directly:
 * copies in all directions (through offset pointers, odd sizes, on busy
 * streams), fills, the device-side index copy and position add a captured
 * decode step relies on, ordering across streams through events, pinned
 * sources read in stream order, and a graph that replays what it recorded.
 */

#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); \
                                       printf("\n"); failures++; } } while (0)

#ifdef VV_HAS_ACCEL
static void fill_pattern(uint8_t* p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)((i * 131u + seed * 7u) ^ (i >> 8));
}

static void test_copies(void) {
    printf("copies: h2d, d2h, d2d through offsets and odd sizes\n");
    const size_t N = (1u << 20) + 37;
    uint8_t* h = (uint8_t*)malloc(N);
    uint8_t* back = (uint8_t*)malloc(N);
    fill_pattern(h, N, 1);
    void *a = NULL, *b = NULL;
    CHECK(vv_dev_alloc(&a, N) == VV_OK && a, "alloc a");
    CHECK(vv_dev_alloc(&b, N) == VV_OK && b, "alloc b");
    void* st = NULL;
    CHECK(vv_dev_stream_create(&st) == VV_OK, "stream");

    /* default stream */
    CHECK(vv_dev_memcpy_h2d(a, h, N, NULL) == VV_OK, "h2d null");
    memset(back, 0, N);
    CHECK(vv_dev_memcpy_d2h(back, a, N, NULL) == VV_OK, "d2h null");
    CHECK(memcmp(back, h, N) == 0, "round trip on the default stream");

    /* on a stream, through offsets that are not 16-byte aligned */
    static const size_t offs[] = { 0, 1, 3, 16, 4093 };
    static const size_t lens[] = { 1, 7, 16, 4096, 65537 };
    for (size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); i++)
        for (size_t j = 0; j < sizeof(lens) / sizeof(lens[0]); j++) {
            const size_t o = offs[i], n = lens[j];
            CHECK(vv_dev_memset_async(b, 0xEE, N, st) == VV_OK, "memset");
            CHECK(vv_dev_memcpy_d2d((uint8_t*)b + o, (uint8_t*)a + 5, n, st)
                  == VV_OK, "d2d");
            CHECK(vv_dev_memcpy_d2h(back, b, o + n + 1, st) == VV_OK, "d2h");
            int bad = memcmp(back + o, h + 5, n) != 0;
            for (size_t k = 0; k < o && !bad; k++) bad = back[k] != 0xEE;
            bad = bad || back[o + n] != 0xEE;
            CHECK(!bad, "d2d offset %zu length %zu", o, n);
        }

    /* host copy onto a stream that still has work queued: stream order */
    fill_pattern(back, N, 9);
    CHECK(vv_dev_memset_async(b, 0x11, N, st) == VV_OK, "memset");
    CHECK(vv_dev_memcpy_h2d((uint8_t*)b + 100, back, 3000, st) == VV_OK, "h2d small");
    CHECK(vv_dev_memcpy_h2d((uint8_t*)b + 5000, back, 200000, st) == VV_OK, "h2d large");
    uint8_t* got = (uint8_t*)malloc(N);
    CHECK(vv_dev_memcpy_d2h(got, b, 205000, st) == VV_OK, "d2h");
    int bad = memcmp(got + 100, back, 3000) != 0 ||
              memcmp(got + 5000, back, 200000) != 0 ||
              got[99] != 0x11 || got[3100] != 0x11 || got[4999] != 0x11;
    CHECK(!bad, "h2d behind a queued memset lands after it");

    /* synchronous fill */
    CHECK(vv_dev_memset(a, 0x5A, 1000) == VV_OK, "memset sync");
    CHECK(vv_dev_memcpy_d2h(got, a, 1001, st) == VV_OK, "d2h");
    bad = 0;
    for (int k = 0; k < 1000; k++) bad |= got[k] != 0x5A;
    CHECK(!bad && got[1000] == h[1000], "vv_dev_memset");

    vv_dev_stream_destroy(st);
    vv_dev_free(a);
    vv_dev_free(b);
    free(h); free(back); free(got);
}

static void test_device_index(void) {
    printf("device-side index: pos_add and memcpy_d2d_at\n");
    void *pos = NULL, *pos2 = NULL, *rows = NULL, *row = NULL;
    const size_t RB = 48, NR = 64;     /* 48 = a multiple of 16; also try 20 */
    vv_dev_alloc(&pos, 4); vv_dev_alloc(&pos2, 4);
    vv_dev_alloc(&rows, RB * NR); vv_dev_alloc(&row, RB);
    void* st = NULL;
    vv_dev_stream_create(&st);
    const int seven = 7;
    vv_dev_memcpy_h2d(pos, &seven, 4, st);
    CHECK(vv_pos_add_dev((int*)pos2, (const int*)pos, 5, st) == VV_OK, "pos_add");
    int got = 0;
    vv_dev_memcpy_d2h(&got, pos2, 4, st);
    CHECK(got == 12, "pos_add gave %d", got);

    uint8_t h[48];
    for (int i = 0; i < 48; i++) h[i] = (uint8_t)(200 + i);
    vv_dev_memset_async(rows, 0, RB * NR, st);
    vv_dev_memcpy_h2d(row, h, RB, st);
    CHECK(vv_dev_memcpy_d2d_at(rows, row, RB, (const int*)pos2, st) == VV_OK, "at16");
    CHECK(vv_dev_memcpy_d2d_at(rows, row, 20, (const int*)pos, st) == VV_OK, "at1");
    uint8_t* back = (uint8_t*)malloc(RB * NR);
    vv_dev_memcpy_d2h(back, rows, RB * NR, st);
    CHECK(memcmp(back + 12 * RB, h, RB) == 0, "row 12 of 48 bytes");
    CHECK(memcmp(back + 7 * 20, h, 20) == 0, "row 7 of 20 bytes");
    CHECK(back[12 * RB - 1] == 0 && back[13 * RB] == 0, "neighbours untouched");
    free(back);
    vv_dev_stream_destroy(st);
    vv_dev_free(pos); vv_dev_free(pos2); vv_dev_free(rows); vv_dev_free(row);
}

static void test_events(void) {
    printf("events: one stream waits for another\n");
    const size_t N = 8u << 20;
    uint8_t* h = (uint8_t*)malloc(N);
    uint8_t* back = (uint8_t*)malloc(N);
    fill_pattern(h, N, 3);
    void *x = NULL, *y = NULL, *z = NULL, *s1 = NULL, *s2 = NULL, *ev = NULL;
    vv_dev_alloc(&x, N); vv_dev_alloc(&y, N); vv_dev_alloc(&z, N);
    vv_dev_stream_create(&s1); vv_dev_stream_create(&s2);
    CHECK(vv_dev_event_create(&ev) == VV_OK, "event");
    vv_dev_memcpy_h2d(x, h, N, NULL);
    for (int round = 0; round < 4; round++) {
        vv_dev_memset_async(z, 0, N, s2);
        vv_dev_stream_sync(s2);
        /* s1: a chain of copies ending in y; s2 must see the last one */
        for (int i = 0; i < 8; i++) vv_dev_memcpy_d2d(y, x, N, s1);
        CHECK(vv_dev_event_record(ev, s1) == VV_OK, "record");
        CHECK(vv_dev_stream_wait_event(s2, ev) == VV_OK, "wait");
        vv_dev_memcpy_d2d(z, y, N, s2);
        CHECK(vv_dev_event_sync(ev) == VV_OK, "event sync");
        memset(back, 0, N);
        vv_dev_memcpy_d2h(back, z, N, s2);
        CHECK(memcmp(back, h, N) == 0, "round %d: z holds x", round);
    }

    printf("pinned source: read when the copy runs, not when it is issued\n");
    void* pin = NULL;
    CHECK(vv_dev_alloc_pinned(&pin, N) == VV_OK, "pinned");
    memcpy(pin, h, N);
    vv_dev_memset_async(z, 0, N, s1);
    vv_dev_memcpy_h2d(z, pin, N, s1);
    vv_dev_event_record(ev, s1);
    vv_dev_event_sync(ev);
    memset(pin, 0, N);                   /* safe now: the copy has run */
    vv_dev_memcpy_d2h(back, z, N, s1);
    CHECK(memcmp(back, h, N) == 0, "pinned h2d after an event sync");
    vv_dev_free_pinned(pin);

    vv_dev_event_destroy(ev);
    vv_dev_stream_destroy(s1); vv_dev_stream_destroy(s2);
    vv_dev_free(x); vv_dev_free(y); vv_dev_free(z);
    free(h); free(back);
}

static void test_graph(void) {
    printf("graph: capture once, replay many\n");
    void *pos = NULL, *rows = NULL, *row = NULL, *st = NULL;
    const size_t RB = 32, NR = 16;
    vv_dev_alloc(&pos, 4); vv_dev_alloc(&rows, RB * NR); vv_dev_alloc(&row, RB);
    vv_dev_stream_create(&st);
    int zero = 0;
    vv_dev_memcpy_h2d(pos, &zero, 4, st);
    vv_dev_memset_async(rows, 0, RB * NR, st);
    uint8_t h[32];
    for (int i = 0; i < 32; i++) h[i] = (uint8_t)(i + 1);
    vv_dev_memcpy_h2d(row, h, RB, st);
    vv_dev_stream_sync(st);

    void* g = NULL;
    vv_status_t s = vv_dev_graph_begin(st);
    if (s != VV_OK) {
        printf("  SKIP: capture not supported (%s)\n", vv_status_str(s));
    } else {
        /* the step: write the row at *pos, then advance *pos */
        vv_dev_memcpy_d2d_at(rows, row, RB, (const int*)pos, st);
        vv_pos_add_dev((int*)pos, (const int*)pos, 1, st);
        CHECK(vv_dev_graph_end(st, &g) == VV_OK && g, "end");
        int p = -1;
        vv_dev_memcpy_d2h(&p, pos, 4, st);
        CHECK(p == 0, "capture ran nothing (pos %d)", p);
        for (int i = 0; i < 5; i++)
            CHECK(vv_dev_graph_launch(g, st) == VV_OK, "launch %d", i);
        vv_dev_memcpy_d2h(&p, pos, 4, st);
        CHECK(p == 5, "five replays advanced pos to %d", p);
        uint8_t back[32 * 16];
        static const uint8_t zeros[32] = {0};
        vv_dev_memcpy_d2h(back, rows, sizeof(back), st);
        int bad = 0;
        for (int r = 0; r < 16; r++)
            bad |= memcmp(back + r * RB, r < 5 ? h : zeros, RB) != 0;
        CHECK(!bad, "rows 0..4 written, the rest untouched");
        vv_dev_graph_destroy(g);

        /* an uncapturable call fails the capture instead of running */
        CHECK(vv_dev_graph_begin(st) == VV_OK, "begin 2");
        vv_dev_stream_sync(st);
        void* g2 = NULL;
        CHECK(vv_dev_graph_end(st, &g2) != VV_OK && !g2,
              "a sync inside a capture breaks it");
    }
    vv_dev_stream_destroy(st);
    vv_dev_free(pos); vv_dev_free(rows); vv_dev_free(row);
}
#endif

int main(void) {
#ifndef VV_HAS_ACCEL
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    if (vv_dev_device_count() <= 0) {
        printf("SKIP: no device available\n");
        return 0;
    }
    char name[128] = "";
    size_t total = 0, freem = 0;
    int sm = 0;
    vv_dev_get_device_name(0, name, sizeof(name));
    vv_dev_get_device_info(0, &total, &freem, &sm);
    printf("device 0: %s, %.1f/%.1f GiB free, %d SM/cores\n", name,
           freem / 1073741824.0, total / 1073741824.0, sm);
    CHECK(total > 0 && freem <= total && sm > 0, "device info");

    test_copies();
    test_device_index();
    test_events();
    test_graph();
    if (failures) {
        printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("all device memory checks passed\n");
    return 0;
#endif
}
