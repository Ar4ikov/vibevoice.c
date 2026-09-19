/**
 * @file kv_cache.c
 * @brief KV-cache management for the Qwen2 backbone.
 *
 * 28 layers x 2 (K+V) x 4 KV heads x 128 head_dim, in whichever storage format
 * kv_quant.h defines. FP16 is a straight d2d copy of the projection output;
 * every other format runs the values through a quantizing store kernel.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/kv_quant.h"

#include <string.h>
#include "vibevoice/device.h"

/** @brief One layer's store bytes, for `max_seq_len` positions. */
static size_t store_bytes(int max_seq_len, int n_kv_heads, int bytes_per_vec) {
    return (size_t)max_seq_len * (size_t)n_kv_heads * (size_t)bytes_per_vec;
}

/** @brief One layer's metadata bytes (FP16 scale per position and head). */
static size_t meta_bytes(int max_seq_len, int n_kv_heads) {
    return (size_t)max_seq_len * (size_t)n_kv_heads * sizeof(uint16_t);
}

size_t vv_kv_cache_bytes(int num_layers, int n_kv_heads, int head_dim,
                         int max_seq_len, int format) {
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)format, head_dim);
    size_t per_layer = 2 * store_bytes(max_seq_len, n_kv_heads, bpv);
    if (vv_kv_has_meta((vv_kv_format_t)format))
        per_layer += 2 * meta_bytes(max_seq_len, n_kv_heads);
    if (!vv_kv_is_raw((vv_kv_format_t)format))
        per_layer += (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    return per_layer * (size_t)num_layers;
}

vv_status_t vv_kv_cache_create(vv_kv_cache_t** cache,
                                int num_layers, int n_kv_heads,
                                int head_dim, int max_seq_len,
                                int format, bool on_cpu) {
    return vv_kv_cache_create_range(cache, num_layers, 0, num_layers,
                                    n_kv_heads, head_dim, max_seq_len,
                                    format, on_cpu);
}

vv_status_t vv_kv_cache_create_range(vv_kv_cache_t** cache,
                                     int num_layers, int first, int count,
                                     int n_kv_heads, int head_dim,
                                     int max_seq_len, int format, bool on_cpu) {
    if (!cache) return VV_ERR_NULL_PTR;
    if (first < 0 || count <= 0 || first + count > num_layers)
        return VV_ERR_INVALID_ARG;

    vv_kv_cache_t* c = (vv_kv_cache_t*)vv_alloc(sizeof(vv_kv_cache_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));

    c->num_layers    = num_layers;
    c->first_layer   = first;
    c->last_layer    = first + count - 1;
    c->n_kv_heads    = n_kv_heads;
    c->head_dim      = head_dim;
    c->max_seq_len   = max_seq_len;
    c->current_len   = 0;
    c->format        = format;
    c->bytes_per_vec = vv_kv_bytes_per_vec((vv_kv_format_t)format, head_dim);
    c->on_cpu        = on_cpu;
    c->attn_backend  = VV_ATTN_FA1;

    const bool has_meta = vv_kv_has_meta((vv_kv_format_t)format);
    const size_t sb = store_bytes(max_seq_len, n_kv_heads, c->bytes_per_vec);
    const size_t mb = has_meta ? meta_bytes(max_seq_len, n_kv_heads) : 0;

    c->k_cache = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
    c->v_cache = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
    if (!c->k_cache || !c->v_cache) {
        vv_kv_cache_free(c);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(c->k_cache, 0, (size_t)num_layers * sizeof(void*));
    memset(c->v_cache, 0, (size_t)num_layers * sizeof(void*));

    if (!vv_kv_is_raw((vv_kv_format_t)format)) {
        /* Reference key per layer: softmax is shift-invariant, so storing
         * K relative to a fixed vector is free and cuts the magnitude the
         * quantizer has to cover by more than 20x on this model. */
        c->k_ref = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
        c->ref_ready = (bool*)vv_alloc((size_t)num_layers * sizeof(bool));
        if (!c->k_ref || !c->ref_ready) {
            vv_kv_cache_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memset(c->k_ref, 0, (size_t)num_layers * sizeof(void*));
        memset(c->ref_ready, 0, (size_t)num_layers * sizeof(bool));
    }

    if (has_meta) {
        c->k_meta = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
        c->v_meta = (void**)vv_alloc((size_t)num_layers * sizeof(void*));
        if (!c->k_meta || !c->v_meta) {
            vv_kv_cache_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memset(c->k_meta, 0, (size_t)num_layers * sizeof(void*));
        memset(c->v_meta, 0, (size_t)num_layers * sizeof(void*));
    }

    #define ALLOC_SLOT(dst, bytes) do {                                      \
        if (on_cpu) {                                                        \
            (dst) = vv_alloc(bytes);                                         \
            if (!(dst)) { vv_kv_cache_free(c); return VV_ERR_OUT_OF_MEMORY; }\
            memset((dst), 0, (bytes));                                       \
        } else {                                                             \
            vv_status_t s_ = vv_dev_alloc(&(dst), (bytes));                  \
            if (s_ != VV_OK) { vv_kv_cache_free(c); return s_; }             \
            vv_dev_memset((dst), 0, (bytes));                                \
        }                                                                    \
    } while (0)

    for (int i = first; i <= c->last_layer; i++) {
        ALLOC_SLOT(c->k_cache[i], sb);
        ALLOC_SLOT(c->v_cache[i], sb);
        if (has_meta) {
            ALLOC_SLOT(c->k_meta[i], mb);
            ALLOC_SLOT(c->v_meta[i], mb);
        }
        if (c->k_ref) {
            const size_t rb = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
            ALLOC_SLOT(c->k_ref[i], rb);
        }
    }
    #undef ALLOC_SLOT

    if (!on_cpu) {
        vv_status_t sp = vv_dev_alloc(&c->d_len, sizeof(int));
        if (sp == VV_OK) sp = vv_dev_alloc(&c->d_len_next, sizeof(int));
        if (sp != VV_OK) { vv_kv_cache_free(c); return sp; }
        vv_dev_memset(c->d_len, 0, sizeof(int));
        vv_dev_memset(c->d_len_next, 0, sizeof(int));
    }

    c->bytes_total = (size_t)count * 2 * (sb + mb);

    if (count == num_layers)
        VV_LOG_I("kv_cache: %d layers, %d heads, dim=%d, max_seq=%d, %s, %s, "
                 "%.1f MB",
                 num_layers, n_kv_heads, head_dim, max_seq_len,
                 vv_kv_format_name((vv_kv_format_t)format),
                 on_cpu ? "CPU" : "GPU",
                 (double)c->bytes_total / (1024.0 * 1024.0));
    else
        VV_LOG_I("kv_cache: layers %d..%d of %d, max_seq=%d, %s, %s, %.1f MB",
                 first, c->last_layer, num_layers, max_seq_len,
                 vv_kv_format_name((vv_kv_format_t)format),
                 on_cpu ? "CPU" : "GPU",
                 (double)c->bytes_total / (1024.0 * 1024.0));

    *cache = c;
    return VV_OK;
}

vv_status_t vv_kv_cache_append(vv_kv_cache_t* cache, int layer,
                                const void* k, const void* v,
                                int seq_len, bool use_device_pos,
                                void* stream) {
    if (!cache || !k || !v) return VV_ERR_NULL_PTR;
    if (layer < 0 || layer >= cache->num_layers) return VV_ERR_INVALID_ARG;
    if (cache->current_len + seq_len > cache->max_seq_len) return VV_ERR_OVERFLOW;
    if (use_device_pos && (cache->on_cpu || !cache->d_len || seq_len != 1))
        return VV_ERR_INVALID_ARG;

    const int* d_pos = use_device_pos ? (const int*)cache->d_len : NULL;

    vv_status_t s;
    if (vv_kv_is_raw((vv_kv_format_t)cache->format)) {
        const size_t row = (size_t)cache->n_kv_heads * (size_t)cache->bytes_per_vec;
        const size_t off = (size_t)cache->current_len * row;
        const size_t n   = (size_t)seq_len * row;
        if (cache->on_cpu) {
            memcpy((uint8_t*)cache->k_cache[layer] + off, k, n);
            memcpy((uint8_t*)cache->v_cache[layer] + off, v, n);
            s = VV_OK;
        } else if (d_pos) {
            s = vv_dev_memcpy_d2d_at(cache->k_cache[layer], k, n, d_pos, stream);
            if (s == VV_OK)
                s = vv_dev_memcpy_d2d_at(cache->v_cache[layer], v, n, d_pos,
                                         stream);
        } else {
            s = vv_dev_memcpy_d2d((uint8_t*)cache->k_cache[layer] + off, k, n, stream);
            if (s == VV_OK)
                s = vv_dev_memcpy_d2d((uint8_t*)cache->v_cache[layer] + off, v, n, stream);
        }
    } else {
        if (cache->on_cpu) return VV_ERR_UNSUPPORTED;
        const bool build_ref = cache->ref_ready && !cache->ref_ready[layer];
        s = vv_kv_quant_store_dev(
            k, v, cache->k_cache[layer], cache->v_cache[layer],
            cache->k_meta ? cache->k_meta[layer] : NULL,
            cache->v_meta ? cache->v_meta[layer] : NULL,
            cache->k_ref ? cache->k_ref[layer] : NULL, build_ref,
            cache->n_kv_heads, cache->head_dim,
            cache->current_len, d_pos, seq_len, cache->format, stream);
        if (s == VV_OK && build_ref) cache->ref_ready[layer] = true;
    }
    if (s != VV_OK) return s;

    /* current_len advances once per position, not once per layer. */
    if (layer == cache->last_layer) cache->current_len += seq_len;
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

vv_status_t vv_kv_cache_get_meta(const vv_kv_cache_t* cache, int layer,
                                 const void** k_meta, const void** v_meta) {
    if (!cache || !k_meta || !v_meta) return VV_ERR_NULL_PTR;
    if (layer < 0 || layer >= cache->num_layers) return VV_ERR_INVALID_ARG;

    *k_meta = cache->k_meta ? cache->k_meta[layer] : NULL;
    *v_meta = cache->v_meta ? cache->v_meta[layer] : NULL;
    return VV_OK;
}

vv_status_t vv_kv_cache_publish_len(vv_kv_cache_t* cache, void* stream) {
    if (!cache) return VV_ERR_NULL_PTR;
    if (!cache->d_len) return VV_OK;
    const int len = cache->current_len;
    const int next = len + 1;
    vv_status_t s = vv_dev_memcpy_h2d(cache->d_len, &len, sizeof(int), stream);
    if (s == VV_OK)
        s = vv_dev_memcpy_h2d(cache->d_len_next, &next, sizeof(int), stream);
    return s;
}

vv_status_t vv_kv_cache_reset(vv_kv_cache_t* cache, void* stream) {
    if (!cache) return VV_ERR_NULL_PTR;
    cache->current_len = 0;
    if (cache->d_len) {
        vv_dev_memset_async(cache->d_len, 0, sizeof(int), stream);
        vv_dev_memset_async(cache->d_len_next, 0, sizeof(int), stream);
    }
    /* The reference key is tied to the keys currently stored; a new session
     * must derive a fresh one from its own first chunk. */
    if (cache->ref_ready)
        memset(cache->ref_ready, 0, (size_t)cache->num_layers * sizeof(bool));
    return VV_OK;
}

vv_status_t vv_kv_cache_free(vv_kv_cache_t* cache) {
    if (!cache) return VV_ERR_NULL_PTR;

    #define FREE_ARRAY(arr) do {                                            \
        if (arr) {                                                          \
            for (int i = 0; i < cache->num_layers; i++)                     \
                if ((arr)[i]) {                                             \
                    if (cache->on_cpu) vv_free((arr)[i]);                   \
                    else vv_dev_free((arr)[i]);                             \
                }                                                           \
            vv_free(arr);                                                   \
        }                                                                   \
    } while (0)

    FREE_ARRAY(cache->k_cache);
    FREE_ARRAY(cache->v_cache);
    FREE_ARRAY(cache->k_meta);
    FREE_ARRAY(cache->v_meta);
    FREE_ARRAY(cache->k_ref);
    #undef FREE_ARRAY
    if (cache->ref_ready) vv_free(cache->ref_ready);
    if (cache->d_len) vv_dev_free(cache->d_len);
    if (cache->d_len_next) vv_dev_free(cache->d_len_next);

    vv_free(cache);
    return VV_OK;
}

vv_kv_view_t vv_kv_cache_view(const vv_kv_cache_t* c, int layer) {
    vv_kv_view_t v;
    memset(&v, 0, sizeof(v));
    if (!c || layer < 0 || layer >= c->num_layers) return v;
    v.k = c->k_cache[layer];
    v.v = c->v_cache[layer];
    v.k_meta = c->k_meta ? c->k_meta[layer] : NULL;
    v.v_meta = c->v_meta ? c->v_meta[layer] : NULL;
    v.format = c->format;
    v.n_kv_heads = c->n_kv_heads;
    v.head_dim = c->head_dim;
    return v;
}
