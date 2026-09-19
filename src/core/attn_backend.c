/**
 * @file attn_backend.c
 * @brief Names of the attention backends and the environment overrides.
 *
 * Which backend `auto` becomes depends on the device and lives with the
 * kernels (vv_attn_resolve); this file only deals with what a user typed.
 */

#include "vibevoice/types.h"

#include <stdlib.h>
#include <string.h>

vv_attn_backend_t vv_attn_backend_parse(const char* name) {
    if (!name || !*name) return VV_ATTN_AUTO;
    if (strcmp(name, "auto") == 0)       return VV_ATTN_AUTO;
    if (strcmp(name, "fa1") == 0)        return VV_ATTN_FA1;
    if (strcmp(name, "scalar") == 0)     return VV_ATTN_FA1;
    if (strcmp(name, "fa2") == 0)        return VV_ATTN_FA2;
    if (strcmp(name, "flashinfer") == 0) return VV_ATTN_FLASHINFER;
    if (strcmp(name, "fi") == 0)         return VV_ATTN_FLASHINFER;
    return (vv_attn_backend_t)VV_ATTN_BACKEND_COUNT;
}

const char* vv_attn_backend_name(vv_attn_backend_t b) {
    switch (b) {
        case VV_ATTN_AUTO:       return "auto";
        case VV_ATTN_FA1:        return "fa1";
        case VV_ATTN_FA2:        return "fa2";
        case VV_ATTN_FLASHINFER: return "flashinfer";
        default:                 return "?";
    }
}

int vv_kv_paging_parse(const char* name) {
    if (!name) return -1;
    if (strcmp(name, "auto") == 0) return VV_KV_PAGED_AUTO;
    if (strcmp(name, "on") == 0 || strcmp(name, "1") == 0)  return VV_KV_PAGED_ON;
    if (strcmp(name, "off") == 0 || strcmp(name, "0") == 0) return VV_KV_PAGED_OFF;
    return -1;
}

vv_attn_backend_t vv_attn_backend_from_env(vv_attn_backend_t requested) {
    if (requested != VV_ATTN_AUTO) return requested;
    const char* e = getenv("VV_ATTN");
    if (e && *e) {
        const vv_attn_backend_t b = vv_attn_backend_parse(e);
        if (b < VV_ATTN_BACKEND_COUNT) return b;
    }
    /* The switch that predates the backends: off meant the scalar prefill. */
    const char* m = getenv("VV_ATTN_MMA");
    if (m && m[0] == '0') return VV_ATTN_FA1;
    return VV_ATTN_AUTO;
}
