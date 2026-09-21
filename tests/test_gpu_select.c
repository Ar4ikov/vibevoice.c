/**
 * @file test_gpu_select.c
 * @brief `--gpus` and `--gpu-memory` parsing.
 *
 * No device is needed for any of this: what is being checked is that a size
 * an operator types means the number they meant, and that a list they got
 * wrong is refused rather than silently reinterpreted. The parts that do
 * need a device — that the ids exist, what is free on each — are checked at
 * runtime by vv_gpu_set_resolve and logged there.
 */

#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define KiB ((size_t)1024)
#define MiB (KiB * 1024)
#define GiB (MiB * 1024)

static void ok_size(const char* text, size_t want) {
    vv_mem_cap_t cap;
    const vv_status_t s = vv_parse_mem_cap(text, &cap);
    const size_t got = (s == VV_OK) ? vv_mem_cap_bytes(&cap, 24 * GiB) : 0;
    const int pass = (s == VV_OK && got == want);
    printf("  %-14s -> %-14zu %s\n", text, got, pass ? "ok" : "FAIL");
    if (!pass) { printf("      wanted %zu (status %d)\n", want, (int)s); failures++; }
}

/**
 * @brief A percentage, which is only ever as exact as a float.
 *
 * 80% of 24 GiB is 300 bytes off whichever way it is rounded, and pretending
 * otherwise would be a test of IEEE-754 rather than of the parser.
 */
static void ok_pct(const char* text, double want_frac) {
    vv_mem_cap_t cap;
    const vv_status_t s = vv_parse_mem_cap(text, &cap);
    const double want = want_frac * (double)(24 * GiB);
    const size_t got = (s == VV_OK) ? vv_mem_cap_bytes(&cap, 24 * GiB) : 0;
    const int pass = (s == VV_OK) && ((double)got > want * 0.999999) &&
                     ((double)got < want * 1.000001);
    printf("  %-14s -> %-14zu %s\n", text, got, pass ? "ok" : "FAIL");
    if (!pass) { printf("      wanted ~%.0f (status %d)\n", want, (int)s); failures++; }
}

static void bad_size(const char* text) {
    vv_mem_cap_t cap;
    const int pass = vv_parse_mem_cap(text, &cap) != VV_OK;
    printf("  %-14s -> rejected     %s\n", text, pass ? "ok" : "FAIL");
    if (!pass) failures++;
}

static void ok_list(const char* text, int n, const int* ids) {
    vv_gpu_set_t set;
    int pass = vv_gpu_set_parse(text, &set) == VV_OK && set.n == n;
    for (int i = 0; pass && i < n; i++) pass = set.id[i] == ids[i];
    printf("  %-14s -> %d device(s)  %s\n", text, pass ? set.n : -1,
           pass ? "ok" : "FAIL");
    if (!pass) failures++;
}

static void bad_list(const char* text) {
    vv_gpu_set_t set;
    const int pass = vv_gpu_set_parse(text, &set) != VV_OK;
    printf("  %-14s -> rejected     %s\n", text, pass ? "ok" : "FAIL");
    if (!pass) failures++;
}

/**
 * @brief With --cpu the device flags are not evaluated at all.
 *
 * `all` is the case that broke: a CPU-only container started with
 * `--gpus all --cpu` failed because no device was visible. A list that would
 * not even parse shows the flags are skipped rather than checked, and none
 * of it asks a device anything, so this passes on a machine without one.
 */
static void ok_cpu(const char* gpus, const char* mem) {
    vv_gpu_set_t set;
    memset(&set, 0, sizeof(set));
    int id = 0;
    const vv_status_t s = vv_gpu_set_from_flags(gpus, mem, true, &id, &set);
    const int pass = s == VV_OK && set.n == 1 && set.id[0] == 0 && id == 0;
    printf("  --cpu %-8s %-8s -> cpu      %s\n", gpus ? gpus : "-",
           mem ? mem : "-", pass ? "ok" : "FAIL");
    if (!pass) { printf("      status %d, n %d\n", (int)s, set.n); failures++; }
}

/**
 * @brief CUDA_VISIBLE_DEVICES against the cards a container actually holds.
 *
 * `nodes` is what `/dev/nvidiaN` says is mounted. `want` is the list to use,
 * "" for "drop it", NULL for "leave it alone".
 */
static void ok_visible(const char* cvd, const int* nodes, int n_nodes,
                       const char* want) {
    char got[64] = "";
    const vv_status_t s = vv_cuda_visible_map(cvd, nodes, n_nodes,
                                              got, sizeof(got));
    int pass;
    if (want == NULL)      pass = (s == VV_ERR_UNSUPPORTED);
    else if (!want[0])     pass = (s == VV_ERR_NOT_FOUND);
    else                   pass = (s == VV_OK && strcmp(got, want) == 0);
    printf("  %-12s on %d node(s) -> %-8s %s\n", cvd ? cvd : "(unset)",
           n_nodes, s == VV_OK ? got : (s == VV_ERR_NOT_FOUND ? "drop" : "keep"),
           pass ? "ok" : "FAIL");
    if (!pass) { printf("      wanted %s (status %d)\n",
                        want ? (want[0] ? want : "drop") : "keep", (int)s);
                 failures++; }
}

/** @brief Without --cpu, a bad list or cap still stops the run. */
static void bad_flags(const char* gpus, const char* mem) {
    vv_gpu_set_t set;
    memset(&set, 0, sizeof(set));
    int id = 0;
    const int pass = vv_gpu_set_from_flags(gpus, mem, false, &id, &set) != VV_OK;
    printf("  %-8s %-10s -> rejected     %s\n", gpus ? gpus : "-",
           mem ? mem : "-", pass ? "ok" : "FAIL");
    if (!pass) failures++;
}

int main(void) {
    printf("gpu_select: sizes\n");
    ok_size("8589934592", 8 * GiB);      /* a bare byte count            */
    ok_size("1024", 1024);
    ok_size("8G", 8 * GiB);
    ok_size("8GB", 8 * GiB);             /* GB and GiB both mean 1024    */
    ok_size("8GiB", 8 * GiB);
    ok_size("18.5G", (size_t)(18.5 * (double)GiB));
    ok_size("8192M", 8 * GiB);
    ok_size("512K", 512 * KiB);
    ok_size("2T", 2048 * GiB);
    ok_pct("100%", 1.0);                 /* resolved against a 24 GiB card */
    ok_pct("80%", 0.8);
    ok_pct("12.5%", 0.125);
    ok_size("  12G  ", 12 * GiB);
    ok_size("12 G", 12 * GiB);
    bad_size("");
    bad_size("G");
    bad_size("12Q");
    bad_size("-4G");
    bad_size("101%");
    bad_size("0");
    bad_size("8G trailing");

    printf("gpu_select: device lists\n");
    { const int e[] = {0};       ok_list("0", 1, e); }
    { const int e[] = {0, 1};    ok_list("0,1", 2, e); }
    { const int e[] = {2, 0, 1}; ok_list("2, 0 ,1", 3, e); }
    { const int e[] = {3};       ok_list("3", 1, e); }
    bad_list("");
    bad_list("0,x");
    bad_list("0,0");            /* the same card twice is a typo         */
    bad_list("-1");
    bad_list("0 1");            /* spaces do not separate; commas do     */

    printf("gpu_select: caps against a list\n");
    {
        vv_gpu_set_t set;
        if (vv_gpu_set_parse("0,1", &set) != VV_OK) { failures++; }
        else {
            /* One value covers every device. */
            const int one = vv_gpu_set_caps("8G", &set) == VV_OK &&
                            vv_mem_cap_bytes(&set.cap[0], 0) == 8 * GiB &&
                            vv_mem_cap_bytes(&set.cap[1], 0) == 8 * GiB;
            printf("  %-14s -> both capped  %s\n", "8G", one ? "ok" : "FAIL");
            if (!one) failures++;

            /* One value each, in the order the devices were given. */
            const int two = vv_gpu_set_caps("20G,10G", &set) == VV_OK &&
                            vv_mem_cap_bytes(&set.cap[0], 0) == 20 * GiB &&
                            vv_mem_cap_bytes(&set.cap[1], 0) == 10 * GiB;
            printf("  %-14s -> 20G then 10G %s\n", "20G,10G",
                   two ? "ok" : "FAIL");
            if (!two) failures++;

            /* A count that does not match is a mistake, not a default. */
            const int three = vv_gpu_set_caps("1G,2G,3G", &set) != VV_OK;
            printf("  %-14s -> rejected     %s\n", "1G,2G,3G",
                   three ? "ok" : "FAIL");
            if (!three) failures++;
        }
    }

    printf("gpu_select: the flags as a command applies them\n");
    ok_cpu("all", NULL);
    ok_cpu("all", "80%");
    ok_cpu("0,x", "12Q");        /* not parsed, so not refused            */
    ok_cpu(NULL, NULL);
    bad_flags("0,x", NULL);      /* both stop before any device is asked  */
    bad_flags("0", "1G,2G");

    printf("gpu_select: CUDA_VISIBLE_DEVICES inside a container\n");
    {
        const int host2[]  = {0, 1};      /* the whole host, or both cards */
        const int only1[]  = {1};         /* one card, host index 1        */
        const int only0[]  = {0};
        const int odd[]    = {1, 2};      /* cards 1 and 2 of four         */

        /* On the host the two numberings agree, so nothing moves. */
        ok_visible("0,1", host2, 2, "0,1");
        ok_visible("1", host2, 2, "1");
        ok_visible("0", only0, 1, "0");

        /* GPUStack hands a replica host card 1 alone: '1' means the second
         * of the one card mounted, which is why CUDA found none. */
        ok_visible("1", only1, 1, "0");
        ok_visible("1,2", odd, 2, "0,1");
        ok_visible("2", odd, 2, "1");
        ok_visible(" 2 , 1 ", odd, 2, "1,0");

        /* Nothing to map, or nothing to map onto. */
        ok_visible("3", only1, 1, "");
        ok_visible("0,3", host2, 2, "");
        ok_visible("GPU-6e7a1750-b891-de8b-b5dc-e301a2ac5b66", only1, 1, NULL);
        ok_visible("MIG-GPU-6e7a1750/1/0", host2, 2, NULL);
        ok_visible("", host2, 2, NULL);
        ok_visible("0", host2, 0, NULL);   /* no driver in the container    */
    }

    printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
