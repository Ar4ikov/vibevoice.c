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
#include "vv_thread.h"

/*
 * The pool: one allocation per layer for K, V and (TurboQuant) their scales,
 * each holding n_pages * VV_KV_PAGE_SIZE rows. Pages are handed out from a
 * stack under a lock, because the slots that share a pool run on their own
 * threads.
 *
 * A pool may be smaller than every slot's window put together, so a slot can
 * find it empty. vv_kv_cache_reserve_wait then waits for another slot to give
 * pages back -- which is only worth doing while some other holder is still
 * running. `n_holders` counts caches that hold pages, `n_stalled` those of
 * them waiting for more; when every other holder is stalled too, nobody will
 * ever free anything, and the slot that notices gives up with
 * VV_ERR_KV_POOL_EXHAUSTED (its pages then unblock the rest). Pages coming
 * back make every stalled holder worth another look, so a return clears
 * `n_stalled` and bumps `stall_gen`; waiters that still come up short count
 * themselves again. Without that, a holder woken by the return but not yet
 * scheduled would still look stuck, and another slot could fail for it.
 */
struct vv_kv_pool {
    void**     k;              /* [num_layers] */
    void**     v;
    void**     k_meta;         /* [num_layers] or NULL */
    void**     v_meta;
    int        num_layers, first, count;
    int        n_kv_heads, head_dim, format, bytes_per_vec;
    int        n_pages;
    int*       free_stack;     /* [n_pages] */
    int        n_free;
    int        refs;
    int        n_holders;      /* caches with n_pages > 0               */
    int        n_stalled;      /* holders blocked in reserve_wait       */
    unsigned   stall_gen;      /* bumped when pages return               */
    size_t     bytes;
    vv_mutex_t lock;
    vv_cond_t  freed;          /* broadcast whenever pages come back    */
};

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
    if (cache->pool) {
        /* Pages were reserved outside any capture; a decode step writes at
         * the device position, which the host length bounds from below. */
        if (cache->current_len + seq_len > cache->n_pages * VV_KV_PAGE_SIZE)
            return VV_ERR_KV_POOL_EXHAUSTED;   /* not reserved */
        const bool build_ref = cache->ref_ready && !cache->ref_ready[layer];
        s = vv_kv_store_dev(
            k, v, cache->k_cache[layer], cache->v_cache[layer],
            cache->k_meta ? cache->k_meta[layer] : NULL,
            cache->v_meta ? cache->v_meta[layer] : NULL,
            cache->k_ref ? cache->k_ref[layer] : NULL, build_ref,
            cache->n_kv_heads, cache->head_dim,
            cache->current_len, d_pos, seq_len, cache->format,
            cache->page_table, stream);
        if (s == VV_OK && build_ref) cache->ref_ready[layer] = true;
    } else if (vv_kv_is_raw((vv_kv_format_t)cache->format)) {
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
    vv_kv_cache_release(cache);
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

    if (cache->pool) {
        /* The store belongs to the pool: hand back the pages, drop the
         * reference, and free only this cache's own arrays. */
        vv_kv_cache_release(cache);
        if (cache->k_cache) vv_free(cache->k_cache);
        if (cache->v_cache) vv_free(cache->v_cache);
        if (cache->k_meta) vv_free(cache->k_meta);
        if (cache->v_meta) vv_free(cache->v_meta);
        cache->k_cache = cache->v_cache = cache->k_meta = cache->v_meta = NULL;
        if (cache->page_table) vv_dev_free(cache->page_table);
        if (cache->pages) vv_free(cache->pages);
        vv_kv_pool_release(cache->pool);
        cache->pool = NULL;
    }
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Paging
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_kv_pool_create(vv_kv_pool_t** out, int num_layers,
                              int first, int count, int n_kv_heads,
                              int head_dim, int n_pages, int format) {
    if (!out) return VV_ERR_NULL_PTR;
    if (first < 0 || count <= 0 || first + count > num_layers || n_pages <= 0)
        return VV_ERR_INVALID_ARG;

    vv_kv_pool_t* p = (vv_kv_pool_t*)vv_alloc(sizeof(vv_kv_pool_t));
    if (!p) return VV_ERR_OUT_OF_MEMORY;
    memset(p, 0, sizeof(*p));
    p->num_layers = num_layers;
    p->first = first;
    p->count = count;
    p->n_kv_heads = n_kv_heads;
    p->head_dim = head_dim;
    p->format = format;
    p->bytes_per_vec = vv_kv_bytes_per_vec((vv_kv_format_t)format, head_dim);
    p->n_pages = n_pages;
    p->refs = 1;
    vv_mutex_init(&p->lock);
    vv_cond_init(&p->freed);

    const bool has_meta = vv_kv_has_meta((vv_kv_format_t)format);
    const int rows = n_pages * VV_KV_PAGE_SIZE;
    const size_t sb = store_bytes(rows, n_kv_heads, p->bytes_per_vec);
    const size_t mb = has_meta ? meta_bytes(rows, n_kv_heads) : 0;
    const size_t arr = (size_t)num_layers * sizeof(void*);

    p->k = (void**)vv_alloc(arr);
    p->v = (void**)vv_alloc(arr);
    p->free_stack = (int*)vv_alloc((size_t)n_pages * sizeof(int));
    if (has_meta) {
        p->k_meta = (void**)vv_alloc(arr);
        p->v_meta = (void**)vv_alloc(arr);
    }
    if (!p->k || !p->v || !p->free_stack ||
        (has_meta && (!p->k_meta || !p->v_meta))) {
        vv_kv_pool_release(p);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(p->k, 0, arr);
    memset(p->v, 0, arr);
    if (has_meta) { memset(p->k_meta, 0, arr); memset(p->v_meta, 0, arr); }

    for (int l = first; l < first + count; l++) {
        vv_status_t s = vv_dev_alloc(&p->k[l], sb);
        if (s == VV_OK) s = vv_dev_alloc(&p->v[l], sb);
        if (s == VV_OK && has_meta) s = vv_dev_alloc(&p->k_meta[l], mb);
        if (s == VV_OK && has_meta) s = vv_dev_alloc(&p->v_meta[l], mb);
        if (s != VV_OK) { vv_kv_pool_release(p); return s; }
        /* Zeroed, so nothing stale can read as a NaN; the kernels mask by
         * length and zero-fill past it in any case. */
        vv_dev_memset(p->k[l], 0, sb);
        vv_dev_memset(p->v[l], 0, sb);
        if (has_meta) {
            vv_dev_memset(p->k_meta[l], 0, mb);
            vv_dev_memset(p->v_meta[l], 0, mb);
        }
    }
    /* Low ids on top, so a lone user walks the pool in order. */
    for (int i = 0; i < n_pages; i++) p->free_stack[i] = n_pages - 1 - i;
    p->n_free = n_pages;
    p->bytes = (size_t)count * 2 * (sb + mb);

    VV_LOG_I("kv_pool: %d pages of %d positions (%d tokens), %s, %.1f MB",
             n_pages, VV_KV_PAGE_SIZE, rows,
             vv_kv_format_name((vv_kv_format_t)format),
             (double)p->bytes / (1024.0 * 1024.0));
    *out = p;
    return VV_OK;
}

void vv_kv_pool_release(vv_kv_pool_t* p) {
    if (!p) return;
    vv_mutex_lock(&p->lock);
    const int left = --p->refs;
    vv_mutex_unlock(&p->lock);
    if (left > 0) return;

    for (int l = 0; l < p->num_layers; l++) {
        if (p->k && p->k[l]) vv_dev_free(p->k[l]);
        if (p->v && p->v[l]) vv_dev_free(p->v[l]);
        if (p->k_meta && p->k_meta[l]) vv_dev_free(p->k_meta[l]);
        if (p->v_meta && p->v_meta[l]) vv_dev_free(p->v_meta[l]);
    }
    if (p->k) vv_free(p->k);
    if (p->v) vv_free(p->v);
    if (p->k_meta) vv_free(p->k_meta);
    if (p->v_meta) vv_free(p->v_meta);
    if (p->free_stack) vv_free(p->free_stack);
    vv_cond_destroy(&p->freed);
    vv_mutex_destroy(&p->lock);
    vv_free(p);
}

int vv_kv_pool_pages(const vv_kv_pool_t* p, int* n_free) {
    if (!p) { if (n_free) *n_free = 0; return 0; }
    vv_mutex_lock((vv_mutex_t*)&p->lock);
    if (n_free) *n_free = p->n_free;
    vv_mutex_unlock((vv_mutex_t*)&p->lock);
    return p->n_pages;
}

size_t vv_kv_pool_bytes(const vv_kv_pool_t* p) { return p ? p->bytes : 0; }

void vv_kv_pool_stats(const vv_kv_pool_t* p, int* n_free, int* n_holders,
                      int* n_stalled) {
    int f = 0, h = 0, st = 0;
    if (p) {
        vv_mutex_lock((vv_mutex_t*)&p->lock);
        f = p->n_free; h = p->n_holders; st = p->n_stalled;
        vv_mutex_unlock((vv_mutex_t*)&p->lock);
    }
    if (n_free) *n_free = f;
    if (n_holders) *n_holders = h;
    if (n_stalled) *n_stalled = st;
}

vv_status_t vv_kv_cache_create_paged(vv_kv_cache_t** cache,
                                     vv_kv_pool_t* pool, int max_seq_len) {
    if (!cache || !pool) return VV_ERR_NULL_PTR;
    if (max_seq_len <= 0) return VV_ERR_INVALID_ARG;

    vv_kv_cache_t* c = (vv_kv_cache_t*)vv_alloc(sizeof(vv_kv_cache_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));

    const int L = pool->num_layers;
    c->num_layers    = L;
    c->first_layer   = pool->first;
    c->last_layer    = pool->first + pool->count - 1;
    c->n_kv_heads    = pool->n_kv_heads;
    c->head_dim      = pool->head_dim;
    c->max_seq_len   = max_seq_len;
    c->format        = pool->format;
    c->bytes_per_vec = pool->bytes_per_vec;
    c->attn_backend  = VV_ATTN_FLASHINFER;
    c->max_pages     = (max_seq_len + VV_KV_PAGE_SIZE - 1) / VV_KV_PAGE_SIZE;

    vv_mutex_lock(&pool->lock);
    pool->refs++;
    vv_mutex_unlock(&pool->lock);
    c->pool = pool;

    const size_t arr = (size_t)L * sizeof(void*);
    const bool has_meta = pool->k_meta != NULL;
    c->k_cache = (void**)vv_alloc(arr);
    c->v_cache = (void**)vv_alloc(arr);
    c->pages = (int*)vv_alloc((size_t)c->max_pages * sizeof(int));
    if (has_meta) {
        c->k_meta = (void**)vv_alloc(arr);
        c->v_meta = (void**)vv_alloc(arr);
    }
    if (!c->k_cache || !c->v_cache || !c->pages ||
        (has_meta && (!c->k_meta || !c->v_meta))) {
        vv_kv_cache_free(c);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memcpy(c->k_cache, pool->k, arr);
    memcpy(c->v_cache, pool->v, arr);
    if (has_meta) {
        memcpy(c->k_meta, pool->k_meta, arr);
        memcpy(c->v_meta, pool->v_meta, arr);
    }

    void* table = NULL;
    vv_status_t s = vv_dev_alloc(&table, (size_t)c->max_pages * sizeof(int));
    if (s != VV_OK) { vv_kv_cache_free(c); return s; }
    c->page_table = (int*)table;
    vv_dev_memset(table, 0, (size_t)c->max_pages * sizeof(int));

    if (!vv_kv_is_raw((vv_kv_format_t)c->format)) {
        c->k_ref = (void**)vv_alloc(arr);
        c->ref_ready = (bool*)vv_alloc((size_t)L * sizeof(bool));
        if (!c->k_ref || !c->ref_ready) {
            vv_kv_cache_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memset(c->k_ref, 0, arr);
        memset(c->ref_ready, 0, (size_t)L * sizeof(bool));
        const size_t rb = (size_t)c->n_kv_heads * c->head_dim * sizeof(uint16_t);
        for (int i = c->first_layer; i <= c->last_layer; i++) {
            s = vv_dev_alloc(&c->k_ref[i], rb);
            if (s != VV_OK) { vv_kv_cache_free(c); return s; }
            vv_dev_memset(c->k_ref[i], 0, rb);
        }
    }

    s = vv_dev_alloc(&c->d_len, sizeof(int));
    if (s == VV_OK) s = vv_dev_alloc(&c->d_len_next, sizeof(int));
    if (s != VV_OK) { vv_kv_cache_free(c); return s; }
    vv_dev_memset(c->d_len, 0, sizeof(int));
    vv_dev_memset(c->d_len_next, 0, sizeof(int));

    c->bytes_total = (size_t)c->max_pages * sizeof(int);
    *cache = c;
    return VV_OK;
}

/*
 * Pages for [0, n_positions). With `wait`, a short pool is waited on for as
 * long as another holder is still running (see the pool comment); without
 * it, or once nobody could free anything, the answer is
 * VV_ERR_KV_POOL_EXHAUSTED -- never VV_ERR_OVERFLOW, which callers read as
 * "this request's window is full" and end the transcript on.
 */
static vv_status_t reserve_pages(vv_kv_cache_t* c, int n_positions,
                                 void* stream, bool wait) {
    if (!c) return VV_ERR_NULL_PTR;
    if (n_positions > c->max_seq_len) return VV_ERR_OVERFLOW;
    if (!c->pool) return VV_OK;
    if (c->no_wait) wait = false;

    const int want = (n_positions + VV_KV_PAGE_SIZE - 1) / VV_KV_PAGE_SIZE;
    if (want <= c->n_pages) return VV_OK;
    const int need = want - c->n_pages;

    vv_kv_pool_t* p = c->pool;
    vv_mutex_lock(&p->lock);
    bool counted = false;       /* in n_stalled (as of stall_gen my_gen) */
    unsigned my_gen = 0;
    bool waited = false;
    while (p->n_free < need) {
        if (counted && p->stall_gen != my_gen) counted = false;
        const bool holds = c->n_pages > 0;
        /* Other holders that are not waiting themselves: only they can
         * give pages back. A cache that holds nothing blocks nobody, so it
         * may wait for as long as it takes; it gives up only when the pool
         * could never hold the request at all. */
        const int running = (p->n_holders - (holds ? 1 : 0)) -
                            (p->n_stalled - (counted ? 1 : 0));
        const bool hopeless = need > p->n_pages ||
                              (holds && running <= 0) ||
                              (!holds && p->n_holders == 0);
        if (!wait || hopeless) {
            const int free_now = p->n_free, holders = p->n_holders;
            if (counted) p->n_stalled--;
            vv_mutex_unlock(&p->lock);
            if (wait && need > p->n_pages)
                VV_LOG_E("kv_pool: %d pages wanted, the pool has %d",
                         need, p->n_pages);
            else if (wait)
                VV_LOG_E("kv_pool: out of pages -- %d more wanted at %d "
                         "positions, %d of %d free, and the %d other "
                         "request(s) holding the rest are waiting too; "
                         "failing this one. The pool is sized by "
                         "--gpu-memory and --slots, not --max-seq-len",
                         need, n_positions, free_now, p->n_pages,
                         holders - (holds ? 1 : 0));
            return VV_ERR_KV_POOL_EXHAUSTED;
        }
        if (holds && !counted) {
            p->n_stalled++;
            counted = true;
            my_gen = p->stall_gen;
        }
        if (!waited) {
            VV_LOG_I("kv_pool: waiting for %d page(s) (%d of %d free)",
                     need, p->n_free, p->n_pages);
            waited = true;
        }
        vv_cond_wait(&p->freed, &p->lock);
    }
    if (counted && p->stall_gen == my_gen) p->n_stalled--;
    if (c->n_pages == 0) p->n_holders++;
    for (int i = 0; i < need; i++)
        c->pages[c->n_pages + i] = p->free_stack[--p->n_free];
    vv_mutex_unlock(&p->lock);

    const vv_status_t s = vv_kv_page_map_dev(c->page_table, c->n_pages, need,
                                             c->pages + c->n_pages, stream);
    if (s != VV_OK) {
        vv_mutex_lock(&p->lock);
        for (int i = need - 1; i >= 0; i--)
            p->free_stack[p->n_free++] = c->pages[c->n_pages + i];
        if (c->n_pages == 0) p->n_holders--;
        p->n_stalled = 0;
        p->stall_gen++;
        vv_cond_broadcast(&p->freed);
        vv_mutex_unlock(&p->lock);
        return s;
    }
    c->n_pages = want;
    return VV_OK;
}

vv_status_t vv_kv_cache_reserve(vv_kv_cache_t* c, int n_positions,
                                void* stream) {
    return reserve_pages(c, n_positions, stream, false);
}

vv_status_t vv_kv_cache_reserve_wait(vv_kv_cache_t* c, int n_positions,
                                     void* stream) {
    return reserve_pages(c, n_positions, stream, true);
}

void vv_kv_cache_release(vv_kv_cache_t* c) {
    if (!c || !c->pool || c->n_pages == 0) return;
    vv_kv_pool_t* p = c->pool;
    vv_mutex_lock(&p->lock);
    for (int i = c->n_pages - 1; i >= 0; i--)
        p->free_stack[p->n_free++] = c->pages[i];
    p->n_holders--;
    p->n_stalled = 0;
    p->stall_gen++;
    vv_cond_broadcast(&p->freed);
    vv_mutex_unlock(&p->lock);
    c->n_pages = 0;
}

vv_kv_view_t vv_kv_cache_view(const vv_kv_cache_t* c, int layer) {
    vv_kv_view_t v;
    memset(&v, 0, sizeof(v));
    if (!c || layer < 0 || layer >= c->num_layers) return v;
    v.k = c->k_cache[layer];
    v.v = c->v_cache[layer];
    v.k_meta = c->k_meta ? c->k_meta[layer] : NULL;
    v.v_meta = c->v_meta ? c->v_meta[layer] : NULL;
    v.page_table = c->pool ? c->page_table : NULL;
    v.format = c->format;
    v.n_kv_heads = c->n_kv_heads;
    v.head_dim = c->head_dim;
    return v;
}
