/**
 * @file smooth.c
 * @brief SmoothQuant statistics: storage, file format and the factor.
 *
 * Collecting the statistics is the decoder's job (a layer with
 * `calib_absmax` set records its inputs), folding them the loader's; this
 * file holds what both share.
 */

#include "vibevoice/smooth.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const char VVSQ_MAGIC[8] = { 'V', 'V', 'S', 'Q', '0', '0', '0', '1' };

vv_status_t vv_smooth_stats_alloc(int n_layers, int hidden, int q_dim,
                                  int inter, vv_smooth_stats_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (n_layers <= 0 || hidden <= 0 || q_dim <= 0 || inter <= 0)
        return VV_ERR_INVALID_ARG;
    vv_smooth_stats_t* st = (vv_smooth_stats_t*)vv_alloc(sizeof(*st));
    if (!st) return VV_ERR_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));
    st->n_layers = n_layers;
    st->hidden = hidden;
    st->q_dim = q_dim;
    st->inter = inter;
    st->width = 2 * hidden + q_dim + inter;
    const size_t n = (size_t)n_layers * (size_t)st->width;
    st->absmax = (float*)vv_alloc(n * sizeof(float));
    if (!st->absmax) { vv_free(st); return VV_ERR_OUT_OF_MEMORY; }
    memset(st->absmax, 0, n * sizeof(float));
    *out = st;
    return VV_OK;
}

void vv_smooth_stats_free(vv_smooth_stats_t* st) {
    if (!st) return;
    vv_free(st->absmax);
    vv_free(st);
}

int vv_smooth_offset(const vv_smooth_stats_t* st, vv_smooth_slot_t slot) {
    switch (slot) {
    case VV_SMOOTH_ATTN_IN:  return 0;
    case VV_SMOOTH_MLP_IN:   return st->hidden;
    case VV_SMOOTH_ATTN_OUT: return 2 * st->hidden;
    case VV_SMOOTH_MLP_MID:  return 2 * st->hidden + st->q_dim;
    default:                 return 0;
    }
}

int vv_smooth_len(const vv_smooth_stats_t* st, vv_smooth_slot_t slot) {
    switch (slot) {
    case VV_SMOOTH_ATTN_IN:
    case VV_SMOOTH_MLP_IN:   return st->hidden;
    case VV_SMOOTH_ATTN_OUT: return st->q_dim;
    case VV_SMOOTH_MLP_MID:  return st->inter;
    default:                 return 0;
    }
}

vv_status_t vv_smooth_stats_save(const vv_smooth_stats_t* st,
                                 const char* path) {
    if (!st || !path) return VV_ERR_NULL_PTR;
    FILE* f = fopen(path, "wb");
    if (!f) {
        VV_LOG_E("smooth: cannot write '%s'", path);
        return VV_ERR_IO;
    }
    const int32_t dims[5] = { st->n_layers, st->hidden, st->q_dim, st->inter,
                              (int32_t)(st->n_tokens > 0x7fffffffL
                                        ? 0x7fffffffL : st->n_tokens) };
    const size_t n = (size_t)st->n_layers * (size_t)st->width;
    const bool ok = fwrite(VVSQ_MAGIC, 1, 8, f) == 8 &&
                    fwrite(dims, sizeof(int32_t), 5, f) == 5 &&
                    fwrite(st->absmax, sizeof(float), n, f) == n;
    const bool closed = fclose(f) == 0;
    if (!ok || !closed) {
        VV_LOG_E("smooth: short write to '%s'", path);
        return VV_ERR_IO;
    }
    return VV_OK;
}

vv_status_t vv_smooth_stats_load(const char* path, vv_smooth_stats_t** out) {
    if (!path || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) {
        VV_LOG_E("smooth: cannot read '%s'", path);
        return VV_ERR_IO;
    }
    char magic[8];
    int32_t dims[5];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, VVSQ_MAGIC, 8) != 0 ||
        fread(dims, sizeof(int32_t), 5, f) != 5) {
        fclose(f);
        VV_LOG_E("smooth: '%s' is not a calibration file", path);
        return VV_ERR_MODEL_FORMAT;
    }
    /* Bounds a sane model stays in; a corrupt header must not size a
       multi-gigabyte allocation. */
    for (int i = 0; i < 4; i++)
        if (dims[i] <= 0 || dims[i] > (1 << 20)) {
            fclose(f);
            VV_LOG_E("smooth: '%s' has a bad header", path);
            return VV_ERR_MODEL_FORMAT;
        }
    vv_smooth_stats_t* st = NULL;
    vv_status_t s = vv_smooth_stats_alloc(dims[0], dims[1], dims[2], dims[3],
                                          &st);
    if (s != VV_OK) { fclose(f); return s; }
    st->n_tokens = dims[4];
    const size_t n = (size_t)st->n_layers * (size_t)st->width;
    const bool ok = fread(st->absmax, sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) {
        vv_smooth_stats_free(st);
        VV_LOG_E("smooth: '%s' is truncated", path);
        return VV_ERR_MODEL_FORMAT;
    }
    for (size_t i = 0; i < n; i++)
        if (!(st->absmax[i] >= 0.0f) || isinf(st->absmax[i])) {
            vv_smooth_stats_free(st);
            VV_LOG_E("smooth: '%s' holds a value that is not a range", path);
            return VV_ERR_MODEL_FORMAT;
        }
    *out = st;
    return VV_OK;
}

float vv_smooth_factor(float amax, float wmax, float alpha) {
    if (!(amax > 0.0f) || !(wmax > 0.0f)) return 1.0f;
    const double s = pow((double)amax, (double)alpha) /
                     pow((double)wmax, 1.0 - (double)alpha);
    if (!(s > 1e-4)) return 1e-4f;
    if (s > 1e4) return 1e4f;
    return (float)s;
}
