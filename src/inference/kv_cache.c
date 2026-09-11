/**
 * @file kv_cache.c
 * @brief KV-cache management for Qwen2 transformer.
 *
 * 28 layers x 2 (K+V) x 4 KV heads x 128 head_dim
 * Supports both GPU (VRAM) and CPU (RAM) allocation.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"

#include <string.h>
#include "vibevoice/device.h"

/* Forward declare CUDA functions */

vv_status_t vv_kv_cache_create(vv_kv_cache_t** cache,
                                int num_layers, int n_kv_heads,
                                int head_dim, int max_seq_len,
                                bool fp8, bool on_cpu) {
    if (!cache) return VV_ERR_NULL_PTR;

    vv_kv_cache_t* c = (vv_kv_cache_t*)vv_alloc(sizeof(vv_kv_cache_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));

    c->num_layers = num_layers;
    c->n_kv_heads = n_kv_heads;
    c->head_dim = head_dim;
    c->max_seq_len = max_seq_len;
    c->current_len = 0;
    c->fp8 = fp8;
    c->on_cpu = on_cpu;

    /* Element size: FP16 = 2 bytes, FP8 = 1 byte */
    size_t elem_size = fp8 ? 1 : 2;
    size_t per_layer_size = (size_t)max_seq_len * (size_t)n_kv_heads *
                             (size_t)head_dim * elem_size;

    c->k_cache = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
    c->v_cache = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
    if (!c->k_cache || !c->v_cache) {
        vv_kv_cache_free(c);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(c->k_cache, 0, (size_t)num_layers * sizeof(void*));
    memset(c->v_cache, 0, (size_t)num_layers * sizeof(void*));

    for (int i = 0; i < num_layers; i++) {
        vv_status_t s;
        if (on_cpu) {
            c->k_cache[i] = vv_alloc(per_layer_size);
            if (!c->k_cache[i]) { vv_kv_cache_free(c); return VV_ERR_OUT_OF_MEMORY; }
            memset(c->k_cache[i], 0, per_layer_size);
            c->v_cache[i] = vv_alloc(per_layer_size);
            if (!c->v_cache[i]) { vv_kv_cache_free(c); return VV_ERR_OUT_OF_MEMORY; }
            memset(c->v_cache[i], 0, per_layer_size);
        } else {
            s = vv_dev_alloc(&c->k_cache[i], per_layer_size);
            if (s != VV_OK) { vv_kv_cache_free(c); return s; }
            s = vv_dev_alloc(&c->v_cache[i], per_layer_size);
            if (s != VV_OK) { vv_kv_cache_free(c); return s; }
        }
    }

    VV_LOG_I("kv_cache: allocated %d layers, %d heads, dim=%d, max_seq=%d, "
             "fp%d, %s, total=%.1f MB",
             num_layers, n_kv_heads, head_dim, max_seq_len,
             fp8 ? 8 : 16,
             on_cpu ? "CPU" : "GPU",
             (float)(2 * num_layers * per_layer_size) / (1024.0f * 1024.0f));

    *cache = c;
    return VV_OK;
}

vv_status_t vv_kv_cache_append(vv_kv_cache_t* cache, int layer,
                                const void* k, const void* v,
                                int seq_len, void* stream) {
    if (!cache || !k || !v) return VV_ERR_NULL_PTR;
    if (layer < 0 || layer >= cache->num_layers) return VV_ERR_INVALID_ARG;
    if (cache->current_len + seq_len > cache->max_seq_len) {
        return VV_ERR_OVERFLOW;
    }

    size_t elem_size = cache->fp8 ? 1 : 2;
    size_t row_size = (size_t)cache->n_kv_heads * (size_t)cache->head_dim *
                       elem_size;
    size_t offset = (size_t)cache->current_len * row_size;
    size_t copy_size = (size_t)seq_len * row_size;

    vv_status_t s;
    if (cache->on_cpu) {
        memcpy((uint8_t*)cache->k_cache[layer] + offset, k, copy_size);
        memcpy((uint8_t*)cache->v_cache[layer] + offset, v, copy_size);
        s = VV_OK;
    } else {
        s = vv_dev_memcpy_d2d((uint8_t*)cache->k_cache[layer] + offset,
                                k, copy_size, stream);
        if (s != VV_OK) return s;
        s = vv_dev_memcpy_d2d((uint8_t*)cache->v_cache[layer] + offset,
                                v, copy_size, stream);
        if (s != VV_OK) return s;
    }

    /* Only increment on last layer to keep it consistent */
    if (layer == cache->num_layers - 1) {
        cache->current_len += seq_len;
    }

    return VV_OK;
}

vv_status_t vv_kv_cache_get(const vv_kv_cache_t* cache, int layer,
                              const void** k, const void** v, int* len) {
    if (!cache || !k || !v || !len) return VV_ERR_NULL_PTR;
    if (layer < 0 || layer >= cache->num_layers) return VV_ERR_INVALID_ARG;

    *k = cache->k_cache[layer];
    *v = cache->v_cache[layer];
    *len = cache->current_len;
    return VV_OK;
}

vv_status_t vv_kv_cache_reset(vv_kv_cache_t* cache) {
    if (!cache) return VV_ERR_NULL_PTR;
    cache->current_len = 0;
    return VV_OK;
}

vv_status_t vv_kv_cache_free(vv_kv_cache_t* cache) {
    if (!cache) return VV_ERR_NULL_PTR;

    if (cache->k_cache) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->k_cache[i]) {
                if (cache->on_cpu) vv_free(cache->k_cache[i]);
                else vv_dev_free(cache->k_cache[i]);
            }
        }
        vv_free(cache->k_cache);
    }
    if (cache->v_cache) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->v_cache[i]) {
                if (cache->on_cpu) vv_free(cache->v_cache[i]);
                else vv_dev_free(cache->v_cache[i]);
            }
        }
        vv_free(cache->v_cache);
    }

    vv_free(cache);
    return VV_OK;
}
