/**
 * @file test_kv_shard.c
 * @brief A KV cache that holds only part of a model's layers.
 *
 * This is the primitive layer sharding rests on: each device allocates the
 * slice it owns, the pointer arrays stay one entry per model layer so every
 * caller keeps indexing by the layer number it already has, and the position
 * counter advances on the last layer *this* cache owns rather than on the
 * last layer of the model — which is the thing that silently stops working
 * if someone reverts it, because a cache that never advances still returns
 * VV_OK from everything.
 *
 * All of it on a CPU cache, so it runs on a machine with no GPU.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(const char* what, int ok) {
    printf("  %-44s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

enum { LAYERS = 28, HEADS = 4, DIM = 128, SEQ = 64 };

int main(void) {
    /* ── A cache for layers 13..27, as the second of two devices ── */
    vv_kv_cache_t* c = NULL;
    vv_status_t s = vv_kv_cache_create_range(&c, LAYERS, 13, 15,
                                             HEADS, DIM, SEQ,
                                             VV_KV_FP32, true);
    check("create_range(13, 15) succeeds", s == VV_OK && c != NULL);
    if (!c) { printf("FAILED (%d)\n", ++failures); return 1; }

    check("layer numbering stays global", c->num_layers == LAYERS);
    check("knows its own first layer", c->first_layer == 13);
    check("knows its own last layer", c->last_layer == 27);

    int allocated = 0, below = 0;
    for (int i = 0; i < LAYERS; i++) {
        if (c->k_cache[i] && c->v_cache[i]) allocated++;
        else if (i >= 13) below++;
    }
    check("allocates exactly its 15 layers", allocated == 15 && below == 0);
    check("leaves the other device's layers alone",
          c->k_cache[0] == NULL && c->k_cache[12] == NULL);

    /* ── The position advances once per token, on layer 27 ── */
    const size_t vec = (size_t)HEADS * DIM;
    float* k = (float*)calloc(vec, sizeof(float));
    float* v = (float*)calloc(vec, sizeof(float));
    if (!k || !v) return 1;

    for (int i = 13; i < 27; i++) vv_kv_cache_append(c, i, k, v, 1, false, NULL);
    check("stays put while the shard is mid-flight", c->current_len == 0);
    vv_kv_cache_append(c, 27, k, v, 1, false, NULL);
    check("advances on the shard's last layer", c->current_len == 1);

    for (int t = 0; t < 4; t++)
        for (int i = 13; i <= 27; i++)
            vv_kv_cache_append(c, i, k, v, 1, false, NULL);
    check("one position per token, not per layer", c->current_len == 5);

    vv_kv_cache_reset(c, NULL);
    check("reset returns to the start", c->current_len == 0);
    vv_kv_cache_free(c);

    /* ── The first device's slice ends where the second's begins ── */
    vv_kv_cache_t* a = NULL;
    s = vv_kv_cache_create_range(&a, LAYERS, 0, 13, HEADS, DIM, SEQ,
                                 VV_KV_FP32, true);
    check("create_range(0, 13) succeeds", s == VV_OK && a != NULL);
    if (a) {
        check("the two slices tile the model",
              a->first_layer == 0 && a->last_layer == 12);
        for (int i = 0; i < 13; i++) vv_kv_cache_append(a, i, k, v, 1, false, NULL);
        check("the first slice advances on layer 12", a->current_len == 1);
        vv_kv_cache_free(a);
    }

    /* ── An unsharded cache is still the whole model ── */
    vv_kv_cache_t* w = NULL;
    s = vv_kv_cache_create(&w, LAYERS, HEADS, DIM, SEQ, VV_KV_FP32, true);
    check("plain create still covers every layer",
          s == VV_OK && w && w->first_layer == 0 &&
          w->last_layer == LAYERS - 1 && w->k_cache[0] && w->k_cache[27]);
    if (w) vv_kv_cache_free(w);

    /* ── Ranges that make no sense are refused ── */
    vv_kv_cache_t* bad = NULL;
    check("refuses a slice that runs off the end",
          vv_kv_cache_create_range(&bad, LAYERS, 20, 20, HEADS, DIM, SEQ,
                                   VV_KV_FP32, true) != VV_OK);
    check("refuses an empty slice",
          vv_kv_cache_create_range(&bad, LAYERS, 4, 0, HEADS, DIM, SEQ,
                                   VV_KV_FP32, true) != VV_OK);

    free(k); free(v);
    printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
