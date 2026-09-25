/**
 * @file alloc.c
 * @brief Tracked memory allocation wrappers for TurboQwen
 */

#include "vibevoice/vibevoice.h"
#include <stdlib.h>
#include <string.h>

/* Thread-local allocation tracking counter. */
static _Thread_local size_t s_alloc_total = 0;

void* vv_alloc(size_t size) {
    if (size == 0) return NULL;

    /*
     * Allocate extra header to store the block size for tracking.
     * Layout: [size_t block_size][user data ...]
     */
    size_t total = size + sizeof(size_t);
    void* raw = malloc(total);
    if (!raw) {
        VV_LOG_E("vv_alloc: failed to allocate %zu bytes", size);
        return NULL;
    }

    *(size_t*)raw = size;
    s_alloc_total += size;

    return (char*)raw + sizeof(size_t);
}

void* vv_realloc(void* ptr, size_t new_size) {
    if (!ptr) return vv_alloc(new_size);
    if (new_size == 0) {
        vv_free(ptr);
        return NULL;
    }

    void* raw = (char*)ptr - sizeof(size_t);
    size_t old_size = *(size_t*)raw;

    size_t total = new_size + sizeof(size_t);
    void* new_raw = realloc(raw, total);
    if (!new_raw) {
        VV_LOG_E("vv_realloc: failed to reallocate %zu bytes", new_size);
        return NULL;
    }

    s_alloc_total = s_alloc_total - old_size + new_size;
    *(size_t*)new_raw = new_size;

    return (char*)new_raw + sizeof(size_t);
}

void vv_free(void* ptr) {
    if (!ptr) return;

    void* raw = (char*)ptr - sizeof(size_t);
    size_t size = *(size_t*)raw;

    if (s_alloc_total >= size) {
        s_alloc_total -= size;
    }

    free(raw);
}

size_t vv_alloc_total(void) {
    return s_alloc_total;
}
