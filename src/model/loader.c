/**
 * @file loader.c
 * @brief Orchestrates loading all model components from safetensors + config.
 *
 * What a tensor becomes is decided by what the file holds, not by its name:
 *
 *   projection `.weight` U8 + `.absmax`  → NF4 (bitsandbytes)
 *   projection `.qweight` I32            → INT4G (AWQ / GPTQ, repacked)
 *   projection `.weight` F16/BF16/F32    → dense FP16, or quantized to NF4 or
 *                                          INT4G while it is read (--quant)
 *   everything else in the LM            → FP16, rounded to nearest-even
 *   speech encoders and connectors       → FP32, exactly
 *
 * Every tensor the model needs is looked up by name, checked against the
 * config's shape, and a missing or misshaped one fails the load with its
 * name. The old loader walked the files and routed by substring, so a dense
 * checkpoint's projections were tagged NF4 and the run died at "prefill
 * layer 0 failed" with nothing pointing at the cause.
 */

#include "vibevoice/model.h"
#include "vibevoice/quant.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Tensor index over every shard ─────────────────────────────────────── */

typedef struct {
    vv_st_tensor_info_t info;
    int                 file;
} st_entry_t;

typedef struct {
    vv_model_t*        m;
    st_entry_t*        ent;      /**< sorted by name */
    int                n_ent;
    vv_load_quant_t    quant;
    char               prefix[128];  /**< "model." / "model.language_model." */
    /* What the projections turned out to be, for the summary line. */
    int                n_nf4, n_int4g, n_dense, n_converted;
    size_t             proj_bytes;
} loader_t;

static int entry_cmp(const void* a, const void* b) {
    return strcmp(((const st_entry_t*)a)->info.name,
                  ((const st_entry_t*)b)->info.name);
}

static const st_entry_t* find(const loader_t* L, const char* name) {
    st_entry_t key;
    size_t n = strlen(name);
    if (n >= sizeof(key.info.name)) return NULL;
    memcpy(key.info.name, name, n + 1);
    return (const st_entry_t*)bsearch(&key, L->ent, (size_t)L->n_ent,
                                      sizeof(st_entry_t), entry_cmp);
}

/** @brief First tensor whose name ends with `suffix`, or NULL. */
static const st_entry_t* find_suffix(const loader_t* L, const char* suffix) {
    const size_t ns = strlen(suffix);
    for (int i = 0; i < L->n_ent; i++) {
        const char* nm = L->ent[i].info.name;
        const size_t n = strlen(nm);
        if (n >= ns && strcmp(nm + n - ns, suffix) == 0) return &L->ent[i];
    }
    return NULL;
}

static const void* data_of(const loader_t* L, const st_entry_t* e) {
    const void* p = NULL;
    if (vv_safetensors_get_data(L->m->st_files[e->file], &e->info, &p) != VV_OK)
        return NULL;
    return p;
}

static int64_t numel_of(const vv_st_tensor_info_t* in) {
    int64_t n = 1;
    for (int d = 0; d < in->ndim; d++) n *= in->shape[d];
    return n;
}

/**
 * @brief The header's byte count agrees with its shape and dtype.
 *
 * Everything below trusts `data_size`; a header whose offsets and shape
 * disagree would otherwise be read past its end.
 */
static vv_status_t check_bytes(const st_entry_t* e) {
    const size_t es = vv_dtype_size(e->info.dtype);
    if (es == 0 || (size_t)numel_of(&e->info) * es != e->info.data_size) {
        VV_LOG_E("loader: '%s' holds %zu bytes, which is not its shape in "
                 "its dtype", e->info.name, e->info.data_size);
        return VV_ERR_MODEL_FORMAT;
    }
    return VV_OK;
}

/** @brief `e` has exactly the dimensions given (-1 = any, 0 = absent). */
static vv_status_t check_shape(const st_entry_t* e, int64_t d0, int64_t d1) {
    const int want_nd = (d1 > 0) ? 2 : 1;
    bool ok = e->info.ndim == want_nd && e->info.shape[0] == d0 &&
              (want_nd == 1 || e->info.shape[1] == d1);
    if (!ok) {
        char got[96] = {0};
        int w = 0;
        for (int d = 0; d < e->info.ndim && w < (int)sizeof(got) - 24; d++)
            w += snprintf(got + w, sizeof(got) - (size_t)w, "%s%lld",
                          d ? "," : "", (long long)e->info.shape[d]);
        if (want_nd == 2)
            VV_LOG_E("loader: '%s' is [%s], the config says [%lld,%lld]",
                     e->info.name, got, (long long)d0, (long long)d1);
        else
            VV_LOG_E("loader: '%s' is [%s], the config says [%lld]",
                     e->info.name, got, (long long)d0);
        return VV_ERR_SHAPE_MISMATCH;
    }
    return check_bytes(e);
}

static const char* dtype_name(vv_dtype_t d) {
    switch (d) {
        case VV_DTYPE_F32:  return "F32";
        case VV_DTYPE_F16:  return "F16";
        case VV_DTYPE_BF16: return "BF16";
        case VV_DTYPE_U8:   return "U8";
        case VV_DTYPE_I8:   return "I8";
        case VV_DTYPE_I32:  return "I32";
        default:            return "other";
    }
}

/* ─── Element conversion, straight out of the mapping ───────────────────── */

/*
 * Conversions run in fixed blocks so the OpenMP loop index stays an `int`
 * (MSVC's OpenMP 2.0) and so a 545M-element embedding converts on every
 * core. Each element is independent, so the result is thread-count free.
 */
#define CONV_BLOCK 65536

static void to_f32(const void* src, vv_dtype_t dt, float* dst, size_t n) {
    const int blocks = (int)((n + CONV_BLOCK - 1) / CONV_BLOCK);
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (b = 0; b < blocks; b++) {
        const size_t i0 = (size_t)b * CONV_BLOCK;
        const size_t i1 = (i0 + CONV_BLOCK < n) ? i0 + CONV_BLOCK : n;
        if (dt == VV_DTYPE_BF16) {
            const uint16_t* s = (const uint16_t*)src;
            for (size_t i = i0; i < i1; i++) dst[i] = vv_bf16_to_float(s[i]);
        } else if (dt == VV_DTYPE_F16) {
            const uint16_t* s = (const uint16_t*)src;
            for (size_t i = i0; i < i1; i++) dst[i] = vv_half_to_float(s[i]);
        } else {
            memcpy(dst + i0, (const float*)src + i0, (i1 - i0) * sizeof(float));
        }
    }
}

/**
 * @brief To FP16, rounding to nearest-even.
 *
 * BF16 has the wider exponent and the narrower mantissa, so a normal BF16
 * value is exact in FP16; the old truncating conversion also flushed every
 * |x| < 6.1e-5 to zero instead of keeping it as an FP16 subnormal. F32 used
 * to stay F32 and reach FP16 kernels as garbage.
 */
static void to_f16(const void* src, vv_dtype_t dt, uint16_t* dst, size_t n) {
    const int blocks = (int)((n + CONV_BLOCK - 1) / CONV_BLOCK);
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (b = 0; b < blocks; b++) {
        const size_t i0 = (size_t)b * CONV_BLOCK;
        const size_t i1 = (i0 + CONV_BLOCK < n) ? i0 + CONV_BLOCK : n;
        if (dt == VV_DTYPE_BF16) {
            const uint16_t* s = (const uint16_t*)src;
            for (size_t i = i0; i < i1; i++)
                dst[i] = vv_float_to_half(vv_bf16_to_float(s[i]));
        } else if (dt == VV_DTYPE_F32) {
            const float* s = (const float*)src;
            for (size_t i = i0; i < i1; i++) dst[i] = vv_float_to_half(s[i]);
        } else {
            memcpy(dst + i0, (const uint16_t*)src + i0,
                   (i1 - i0) * sizeof(uint16_t));
        }
    }
}

static bool is_float_dtype(vv_dtype_t d) {
    return d == VV_DTYPE_F16 || d == VV_DTYPE_BF16 || d == VV_DTYPE_F32;
}

static void set_tensor(vv_tensor_t* t, void* data, vv_dtype_t dt, size_t bytes,
                       int ndim, const int64_t* shape) {
    memset(t, 0, sizeof(*t));
    t->data = data;
    t->dtype = dt;
    t->size_bytes = bytes;
    t->ndim = ndim;
    for (int d = 0; d < ndim && d < VV_MAX_DIMS; d++) t->shape[d] = shape[d];
    t->on_gpu = false;
}

/** @brief An LM tensor (embedding, norm, bias, head) as FP16. */
static vv_status_t load_f16(const loader_t* L, const st_entry_t* e,
                            vv_tensor_t* t) {
    if (!is_float_dtype(e->info.dtype)) {
        VV_LOG_E("loader: '%s' is %s; the LM needs a float tensor here",
                 e->info.name, dtype_name(e->info.dtype));
        return VV_ERR_MODEL_FORMAT;
    }
    const void* src = data_of(L, e);
    if (!src) return VV_ERR_MODEL_FORMAT;
    const size_t n = (size_t)numel_of(&e->info);
    uint16_t* dst = (uint16_t*)vv_alloc(n * sizeof(uint16_t));
    if (!dst) return VV_ERR_OUT_OF_MEMORY;
    to_f16(src, e->info.dtype, dst, n);
    set_tensor(t, dst, VV_DTYPE_F16, n * sizeof(uint16_t),
               e->info.ndim, e->info.shape);
    return VV_OK;
}

/**
 * @brief A speech front-end tensor (encoder, connector) as FP32, exactly.
 *
 * These were BF16 → FP16 (truncated, flushed below 6.1e-5) → FP32: on the
 * AWQ checkpoint 1.2M encoder weights came out as zero, 47% of one norm.
 */
static vv_status_t load_f32(const loader_t* L, const st_entry_t* e,
                            vv_tensor_t* t) {
    if (!is_float_dtype(e->info.dtype)) {
        VV_LOG_E("loader: '%s' is %s; the speech front end needs floats",
                 e->info.name, dtype_name(e->info.dtype));
        return VV_ERR_MODEL_FORMAT;
    }
    vv_status_t s = check_bytes(e);
    if (s != VV_OK) return s;
    const void* src = data_of(L, e);
    if (!src) return VV_ERR_MODEL_FORMAT;
    const size_t n = (size_t)numel_of(&e->info);
    float* dst = (float*)vv_alloc(n * sizeof(float));
    if (!dst) return VV_ERR_OUT_OF_MEMORY;
    to_f32(src, e->info.dtype, dst, n);
    set_tensor(t, dst, VV_DTYPE_F32, n * sizeof(float),
               e->info.ndim, e->info.shape);
    return VV_OK;
}

/** @brief A tensor copied as-is (quantized codes, scales, metadata). */
static vv_status_t load_raw(const loader_t* L, const st_entry_t* e,
                            vv_tensor_t* t) {
    vv_status_t s = check_bytes(e);
    if (s != VV_OK) return s;
    const void* src = data_of(L, e);
    if (!src) return VV_ERR_MODEL_FORMAT;
    void* dst = vv_alloc(e->info.data_size ? e->info.data_size : 1);
    if (!dst) return VV_ERR_OUT_OF_MEMORY;
    memcpy(dst, src, e->info.data_size);
    set_tensor(t, dst, e->info.dtype, e->info.data_size,
               e->info.ndim, e->info.shape);
    return VV_OK;
}

/** @brief Required: find `name`, check its shape, load it as FP16. */
static vv_status_t need_f16(const loader_t* L, const char* name,
                            int64_t d0, int64_t d1, vv_tensor_t* t) {
    const st_entry_t* e = find(L, name);
    if (!e) {
        VV_LOG_E("loader: '%s' is missing from the checkpoint", name);
        return VV_ERR_WEIGHT_MISSING;
    }
    vv_status_t s = check_shape(e, d0, d1);
    return s != VV_OK ? s : load_f16(L, e, t);
}

/* ─── NF4 (bitsandbytes) ────────────────────────────────────────────────── */

/* FP32 → FP16 the way the NF4 path always has (truncating). Its scales are
 * what the existing NF4 transcripts were produced with, so they stay. */
static uint16_t f32_to_half_trunc(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = b & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/*
 * Default bitsandbytes signed dynamic map: create_dynamic_map(signed=True).
 * 255 entries covering [-0.993, +0.993], symmetric around 0; index 255 is
 * padded with 1.0. The code for absmax values when the quant_state carries
 * no map of its own.
 */
static const float BNB_DEFAULT_CODE[256] = {
    -9.9296875e-01f, -9.7890625e-01f, -9.6484375e-01f, -9.5078125e-01f,
    -9.3671875e-01f, -9.2265625e-01f, -9.0859375e-01f, -8.9453125e-01f,
    -8.8046875e-01f, -8.6640625e-01f, -8.5234375e-01f, -8.3828125e-01f,
    -8.2421875e-01f, -8.1015625e-01f, -7.9609375e-01f, -7.8203125e-01f,
    -7.6796875e-01f, -7.5390625e-01f, -7.3984375e-01f, -7.2578125e-01f,
    -7.1171875e-01f, -6.9765625e-01f, -6.8359375e-01f, -6.6953125e-01f,
    -6.5546875e-01f, -6.4140625e-01f, -6.2734375e-01f, -6.1328125e-01f,
    -5.9921875e-01f, -5.8515625e-01f, -5.7109375e-01f, -5.5703125e-01f,
    -5.4296875e-01f, -5.2890625e-01f, -5.1484375e-01f, -5.0078125e-01f,
    -4.8671875e-01f, -4.7265625e-01f, -4.5859375e-01f, -4.4453125e-01f,
    -4.3046875e-01f, -4.1640625e-01f, -4.0234375e-01f, -3.8828125e-01f,
    -3.7421875e-01f, -3.6015625e-01f, -3.4609375e-01f, -3.3203125e-01f,
    -3.1796875e-01f, -3.0390625e-01f, -2.8984375e-01f, -2.7578125e-01f,
    -2.6171875e-01f, -2.4765625e-01f, -2.3359375e-01f, -2.1953125e-01f,
    -2.0546875e-01f, -1.9140625e-01f, -1.7734375e-01f, -1.6328125e-01f,
    -1.4921875e-01f, -1.3515625e-01f, -1.2109375e-01f, -1.0703125e-01f,
    -9.8593750e-02f, -9.5781250e-02f, -9.2968750e-02f, -9.0156250e-02f,
    -8.7343750e-02f, -8.4531250e-02f, -8.1718750e-02f, -7.8906250e-02f,
    -7.6093750e-02f, -7.3281250e-02f, -7.0468750e-02f, -6.7656250e-02f,
    -6.4843750e-02f, -6.2031250e-02f, -5.9218750e-02f, -5.6406250e-02f,
    -5.3593750e-02f, -5.0781250e-02f, -4.7968750e-02f, -4.5156250e-02f,
    -4.2343750e-02f, -3.9531250e-02f, -3.6718750e-02f, -3.3906250e-02f,
    -3.1093750e-02f, -2.8281250e-02f, -2.5468750e-02f, -2.2656250e-02f,
    -1.9843750e-02f, -1.7031250e-02f, -1.4218750e-02f, -1.1406250e-02f,
    -9.7187500e-03f, -9.1562500e-03f, -8.5937500e-03f, -8.0312500e-03f,
    -7.4687500e-03f, -6.9062500e-03f, -6.3437500e-03f, -5.7812500e-03f,
    -5.2187500e-03f, -4.6562500e-03f, -4.0937500e-03f, -3.5312500e-03f,
    -2.9687500e-03f, -2.4062500e-03f, -1.8437500e-03f, -1.2812500e-03f,
    -9.4375000e-04f, -8.3125000e-04f, -7.1875000e-04f, -6.0625000e-04f,
    -4.9375000e-04f, -3.8125000e-04f, -2.6875000e-04f, -1.5625000e-04f,
    -8.8750000e-05f, -6.6250000e-05f, -4.3750000e-05f, -2.1250000e-05f,
    -7.7500000e-06f, -3.2500000e-06f, -5.5000000e-07f,  0.0000000e+00f,
     5.5000000e-07f,  3.2500000e-06f,  7.7500000e-06f,  2.1250000e-05f,
     4.3750000e-05f,  6.6250000e-05f,  8.8750000e-05f,  1.5625000e-04f,
     2.6875000e-04f,  3.8125000e-04f,  4.9375000e-04f,  6.0625000e-04f,
     7.1875000e-04f,  8.3125000e-04f,  9.4375000e-04f,  1.2812500e-03f,
     1.8437500e-03f,  2.4062500e-03f,  2.9687500e-03f,  3.5312500e-03f,
     4.0937500e-03f,  4.6562500e-03f,  5.2187500e-03f,  5.7812500e-03f,
     6.3437500e-03f,  6.9062500e-03f,  7.4687500e-03f,  8.0312500e-03f,
     8.5937500e-03f,  9.1562500e-03f,  9.7187500e-03f,  1.1406250e-02f,
     1.4218750e-02f,  1.7031250e-02f,  1.9843750e-02f,  2.2656250e-02f,
     2.5468750e-02f,  2.8281250e-02f,  3.1093750e-02f,  3.3906250e-02f,
     3.6718750e-02f,  3.9531250e-02f,  4.2343750e-02f,  4.5156250e-02f,
     4.7968750e-02f,  5.0781250e-02f,  5.3593750e-02f,  5.6406250e-02f,
     5.9218750e-02f,  6.2031250e-02f,  6.4843750e-02f,  6.7656250e-02f,
     7.0468750e-02f,  7.3281250e-02f,  7.6093750e-02f,  7.8906250e-02f,
     8.1718750e-02f,  8.4531250e-02f,  8.7343750e-02f,  9.0156250e-02f,
     9.2968750e-02f,  9.5781250e-02f,  9.8593750e-02f,  1.0703125e-01f,
     1.2109375e-01f,  1.3515625e-01f,  1.4921875e-01f,  1.6328125e-01f,
     1.7734375e-01f,  1.9140625e-01f,  2.0546875e-01f,  2.1953125e-01f,
     2.3359375e-01f,  2.4765625e-01f,  2.6171875e-01f,  2.7578125e-01f,
     2.8984375e-01f,  3.0390625e-01f,  3.1796875e-01f,  3.3203125e-01f,
     3.4609375e-01f,  3.6015625e-01f,  3.7421875e-01f,  3.8828125e-01f,
     4.0234375e-01f,  4.1640625e-01f,  4.3046875e-01f,  4.4453125e-01f,
     4.5859375e-01f,  4.7265625e-01f,  4.8671875e-01f,  5.0078125e-01f,
     5.1484375e-01f,  5.2890625e-01f,  5.4296875e-01f,  5.5703125e-01f,
     5.7109375e-01f,  5.8515625e-01f,  5.9921875e-01f,  6.1328125e-01f,
     6.2734375e-01f,  6.4140625e-01f,  6.5546875e-01f,  6.6953125e-01f,
     6.8359375e-01f,  6.9765625e-01f,  7.1171875e-01f,  7.2578125e-01f,
     7.3984375e-01f,  7.5390625e-01f,  7.6796875e-01f,  7.8203125e-01f,
     7.9609375e-01f,  8.1015625e-01f,  8.2421875e-01f,  8.3828125e-01f,
     8.5234375e-01f,  8.6640625e-01f,  8.8046875e-01f,  8.9453125e-01f,
     9.0859375e-01f,  9.2265625e-01f,  9.3671875e-01f,  9.5078125e-01f,
     9.6484375e-01f,  9.7890625e-01f,  9.9296875e-01f,  1.0000000e+00f,
};

/**
 * @brief NF4 per-block scales as FP16, `n_blocks` of them.
 *
 * Handles bitsandbytes double quantization: `absmax` U8 codes into a
 * dynamic map, times a per-superblock `nested_absmax`, plus the offset from
 * the `quant_state` JSON blob. Single-level absmax (F16/F32) is converted.
 */
static vv_status_t load_nf4_scales(const loader_t* L, const char* wname,
                                   size_t n_blocks, vv_weight_t* w) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s.absmax", wname);
    const st_entry_t* ea = find(L, buf);
    if (!ea) {
        VV_LOG_E("loader: '%s' is U8 but has no .absmax", wname);
        return VV_ERR_WEIGHT_MISSING;
    }
    vv_status_t s = check_bytes(ea);
    if (s != VV_OK) return s;
    if ((size_t)numel_of(&ea->info) != n_blocks) {
        VV_LOG_E("loader: '%s' has %lld scales for %zu blocks of 64",
                 buf, (long long)numel_of(&ea->info), n_blocks);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const void* abs_src = data_of(L, ea);
    if (!abs_src) return VV_ERR_MODEL_FORMAT;

    uint16_t* scales = (uint16_t*)vv_alloc(n_blocks * sizeof(uint16_t));
    if (!scales) return VV_ERR_OUT_OF_MEMORY;
    w->quant.block_size = VV_NF4_BLOCK;

    if (ea->info.dtype != VV_DTYPE_U8) {
        if (!is_float_dtype(ea->info.dtype)) {
            vv_free(scales);
            return VV_ERR_MODEL_FORMAT;
        }
        to_f16(abs_src, ea->info.dtype, scales, n_blocks);
        w->quant.double_quant = false;
    } else {
        w->quant.double_quant = true;
        snprintf(buf, sizeof(buf), "%s.nested_absmax", wname);
        const st_entry_t* en = find(L, buf);
        if (!en || en->info.dtype != VV_DTYPE_F32 || check_bytes(en) != VV_OK) {
            VV_LOG_E("loader: '%s' is missing or not F32", buf);
            vv_free(scales);
            return VV_ERR_WEIGHT_MISSING;
        }
        const float* nested = (const float*)data_of(L, en);
        const int n_super = (int)numel_of(&en->info);

        float nested_offset = 0.0f;
        int nested_block = 256;
        const float* qmap = BNB_DEFAULT_CODE;
        float custom[256];

        snprintf(buf, sizeof(buf), "%s.quant_state.bitsandbytes__nf4", wname);
        const st_entry_t* eq = find(L, buf);
        const char* qs = eq ? (const char*)data_of(L, eq) : NULL;
        if (qs && eq->info.data_size > 0) {
            char* js = (char*)vv_alloc(eq->info.data_size + 1);
            if (js) {
                memcpy(js, qs, eq->info.data_size);
                js[eq->info.data_size] = '\0';
                cJSON* root = cJSON_Parse(js);
                if (root) {
                    cJSON* off = cJSON_GetObjectItem(root, "nested_offset");
                    if (off && cJSON_IsNumber(off))
                        nested_offset = (float)off->valuedouble;
                    cJSON* nbs = cJSON_GetObjectItem(root, "nested_blocksize");
                    if (nbs && cJSON_IsNumber(nbs) && nbs->valuedouble >= 1.0)
                        nested_block = (int)nbs->valuedouble;
                    cJSON* nqm = cJSON_GetObjectItem(root, "nested_quant_map");
                    if (nqm && cJSON_IsArray(nqm)) {
                        const int m = cJSON_GetArraySize(nqm);
                        if (m >= 255 && m <= 256) {
                            for (int i = 0; i < m; i++) {
                                cJSON* v = cJSON_GetArrayItem(nqm, i);
                                custom[i] = v ? (float)v->valuedouble : 0.0f;
                            }
                            if (m == 255) custom[255] = 1.0f;
                            qmap = custom;
                        }
                    }
                    cJSON_Delete(root);
                }
                vv_free(js);
            }
        }

        const uint8_t* codes = (const uint8_t*)abs_src;
        for (size_t i = 0; i < n_blocks; i++) {
            int si = (int)(i / (size_t)nested_block);
            if (si >= n_super) si = n_super - 1;
            scales[i] = f32_to_half_trunc(qmap[codes[i]] * nested[si]
                                          + nested_offset);
        }
    }

    const int64_t shp[1] = { (int64_t)n_blocks };
    set_tensor(&w->quant.scales, scales, VV_DTYPE_F16,
               n_blocks * sizeof(uint16_t), 1, shp);
    return VV_OK;
}

/* ─── AWQ / GPTQ ────────────────────────────────────────────────────────── */

/**
 * @brief An AWQ / GPTQ projection, repacked into the INT4G layout.
 *
 * `base` is the projection name without a suffix; qweight, qzeros and
 * scales sit next to it. The repack is what makes the GEMV coalesce — see
 * src/quant/awq_repack.c.
 */
static vv_status_t load_awq_weight(const loader_t* L, const char* base,
                                   int N_want, int K_want, vv_weight_t* w) {
    char buf[320];
    vv_tensor_t qweight = {0}, qzeros = {0}, scales = {0};
    const st_entry_t* e;
    vv_status_t s;

    snprintf(buf, sizeof(buf), "%s.qweight", base);
    e = find(L, buf);
    if (!e || e->info.dtype != VV_DTYPE_I32 || e->info.ndim != 2) {
        VV_LOG_E("loader: '%s' is missing or not a 2-D I32", buf);
        return VV_ERR_MODEL_FORMAT;
    }
    if ((s = load_raw(L, e, &qweight)) != VV_OK) return s;

    snprintf(buf, sizeof(buf), "%s.qzeros", base);
    e = find(L, buf);
    if (!e || e->info.dtype != VV_DTYPE_I32) {
        VV_LOG_E("loader: '%s' is missing or not I32", buf);
        vv_tensor_free(&qweight);
        return VV_ERR_WEIGHT_MISSING;
    }
    if ((s = load_raw(L, e, &qzeros)) != VV_OK) {
        vv_tensor_free(&qweight);
        return s;
    }
    snprintf(buf, sizeof(buf), "%s.scales", base);
    e = find(L, buf);
    if (!e || e->info.dtype != VV_DTYPE_F16 || e->info.ndim != 2) {
        VV_LOG_E("loader: '%s' is missing or not a 2-D F16", buf);
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        return VV_ERR_WEIGHT_MISSING;
    }
    if ((s = load_raw(L, e, &scales)) != VV_OK) {
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        return s;
    }

    /* qweight is [K, N/8] int32; scales is [K/G, N] fp16. */
    const int K = (int)qweight.shape[0];
    const int N = (int)qweight.shape[1] * 8;
    const int n_groups = (int)scales.shape[0];
    if (K != K_want || N != N_want || n_groups <= 0 || (K % n_groups) != 0 ||
        scales.shape[1] != N ||
        qzeros.size_bytes != (size_t)n_groups * (size_t)(N / 8) * 4) {
        VV_LOG_E("loader: '%s' AWQ tensors are [%d x %d] with %d groups; "
                 "the config says [%d x %d]", base, K, N, n_groups,
                 K_want, N_want);
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        vv_tensor_free(&scales);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const int group_size = K / n_groups;

    /*
     * GPTQ writes zero_point - 1; AWQ writes it directly. The two formats
     * are otherwise identical here, and a g_idx tensor is the tell.
     */
    snprintf(buf, sizeof(buf), "%s.g_idx", base);
    const int zero_bias = find(L, buf) ? 1 : 0;

    const size_t packed_bytes = (size_t)N * (K / 2);
    const size_t group_bytes  = (size_t)N * n_groups * sizeof(uint16_t);
    uint8_t*  packed = (uint8_t*)vv_alloc(packed_bytes);
    uint16_t* sc     = (uint16_t*)vv_alloc(group_bytes);
    uint16_t* mn     = (uint16_t*)vv_alloc(group_bytes);
    if (!packed || !sc || !mn) {
        vv_free(packed); vv_free(sc); vv_free(mn);
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        vv_tensor_free(&scales);
        return VV_ERR_OUT_OF_MEMORY;
    }

    s = vv_awq_repack((const uint32_t*)qweight.data,
                      (const uint32_t*)qzeros.data,
                      (const uint16_t*)scales.data,
                      K, N, group_size, zero_bias, packed, sc, mn);
    vv_tensor_free(&qweight);
    vv_tensor_free(&qzeros);
    vv_tensor_free(&scales);
    if (s != VV_OK) {
        vv_free(packed); vv_free(sc); vv_free(mn);
        return s;
    }

    w->quant_kind   = VV_QUANT_INT4G;
    w->is_quantized = true;
    w->group_size   = group_size;

    const int64_t pshape[2] = { N, K / 2 };
    const int64_t gshape[2] = { N, n_groups };
    set_tensor(&w->tensor, packed, VV_DTYPE_U8, packed_bytes, 2, pshape);
    set_tensor(&w->quant.scales, sc, VV_DTYPE_F16, group_bytes, 2, gshape);
    set_tensor(&w->mins, mn, VV_DTYPE_F16, group_bytes, 2, gshape);

    VV_LOG_D("loader: AWQ '%s' N=%d K=%d group=%d%s",
             base, N, K, group_size, zero_bias ? " (gptq zeros)" : "");
    return VV_OK;
}

/* ─── Dense projections, kept or quantized on the way in ────────────────── */

/**
 * @brief A dense [N,K] projection: FP16, or NF4 / INT4G quantized straight
 *        from the mapping.
 *
 * Quantization goes through an FP32 slab of rows at a time (about 32 MB), so
 * peak memory is the quantized model plus one slab, never a dense copy of
 * the whole thing. Rows are independent, so the bytes do not depend on the
 * thread count or the slab size.
 */
static vv_status_t load_dense_weight(loader_t* L, const st_entry_t* e,
                                     int N, int K, vv_weight_t* w) {
    vv_status_t s = check_shape(e, N, K);
    if (s != VV_OK) return s;
    const void* src = data_of(L, e);
    if (!src) return VV_ERR_MODEL_FORMAT;
    const vv_dtype_t dt = e->info.dtype;
    const size_t es = vv_dtype_size(dt);

    const vv_load_quant_t q = L->quant;
    if (q == VV_LOAD_QUANT_AUTO || q == VV_LOAD_QUANT_NONE) {
        s = load_f16(L, e, &w->tensor);
        if (s != VV_OK) return s;
        w->quant_kind = VV_QUANT_NONE;
        w->is_quantized = false;
        L->n_dense++;
        L->proj_bytes += w->tensor.size_bytes;
        return VV_OK;
    }

    const bool nf4 = (q == VV_LOAD_QUANT_NF4);
    const int group = nf4 ? VV_NF4_BLOCK : VV_INT4G_LOAD_GROUP;
    if (K % group != 0) {
        VV_LOG_E("loader: '%s' has K=%d, not a multiple of the %s group %d",
                 e->info.name, K, vv_load_quant_name(q), group);
        return VV_ERR_UNSUPPORTED;
    }
    const int n_groups = K / group;
    const size_t packed_bytes = (size_t)N * (size_t)(K / 2);
    const size_t scale_bytes = (size_t)N * (size_t)n_groups * sizeof(uint16_t);

    uint8_t*  packed = (uint8_t*)vv_alloc(packed_bytes);
    uint16_t* sc = (uint16_t*)vv_alloc(scale_bytes);
    uint16_t* mn = nf4 ? NULL : (uint16_t*)vv_alloc(scale_bytes);
    int slab = (int)(((size_t)32 << 20) / ((size_t)K * sizeof(float)));
    if (slab < 1) slab = 1;
    if (slab > N) slab = N;
    float* tmp = (float*)vv_alloc((size_t)slab * (size_t)K * sizeof(float));
    if (!packed || !sc || (!nf4 && !mn) || !tmp) {
        vv_free(packed); vv_free(sc); vv_free(mn); vv_free(tmp);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (int r0 = 0; r0 < N && s == VV_OK; r0 += slab) {
        const int rows = (r0 + slab <= N) ? slab : N - r0;
        to_f32((const uint8_t*)src + (size_t)r0 * K * es, dt, tmp,
               (size_t)rows * K);
        if (nf4)
            s = vv_nf4_quantize(tmp, rows, K,
                                packed + (size_t)r0 * (K / 2),
                                sc + (size_t)r0 * n_groups);
        else
            s = vv_int4g_quantize(tmp, rows, K, group,
                                  packed + (size_t)r0 * (K / 2),
                                  sc + (size_t)r0 * n_groups,
                                  mn + (size_t)r0 * n_groups);
    }
    vv_free(tmp);
    if (s != VV_OK) {
        vv_free(packed); vv_free(sc); vv_free(mn);
        return s;
    }

    const int64_t pshape[2] = { N, K / 2 };
    const int64_t gshape[2] = { N, n_groups };
    set_tensor(&w->tensor, packed, VV_DTYPE_U8, packed_bytes, 2, pshape);
    set_tensor(&w->quant.scales, sc, VV_DTYPE_F16, scale_bytes, 2, gshape);
    w->is_quantized = true;
    if (nf4) {
        w->quant_kind = VV_QUANT_NF4;
        w->quant.block_size = VV_NF4_BLOCK;
        w->quant.double_quant = false;
        L->n_nf4++;
    } else {
        set_tensor(&w->mins, mn, VV_DTYPE_F16, scale_bytes, 2, gshape);
        w->quant_kind = VV_QUANT_INT4G;
        w->group_size = group;
        L->n_int4g++;
    }
    L->n_converted++;
    L->proj_bytes += packed_bytes + scale_bytes * (nf4 ? 1 : 2);
    return VV_OK;
}

/**
 * @brief One projection of one layer, in whatever form the file holds it.
 */
static vv_status_t load_projection(loader_t* L, int layer, const char* which,
                                   int N, int K, bool want_bias,
                                   vv_weight_t* w) {
    char base[256], buf[320];
    snprintf(base, sizeof(base), "%slayers.%d.%s", L->prefix, layer, which);
    strncpy(w->name, base, sizeof(w->name) - 1);
    vv_status_t s;

    snprintf(buf, sizeof(buf), "%s.weight", base);
    const st_entry_t* e = find(L, buf);
    if (e && e->info.dtype == VV_DTYPE_U8) {
        /* bitsandbytes NF4: the codes of [N,K], two to a byte. */
        if (L->quant != VV_LOAD_QUANT_AUTO && L->quant != VV_LOAD_QUANT_NF4) {
            VV_LOG_E("loader: the checkpoint is NF4; --quant %s would need "
                     "the dense weights it no longer has",
                     vv_load_quant_name(L->quant));
            return VV_ERR_UNSUPPORTED;
        }
        if (K % VV_NF4_BLOCK != 0 ||
            (size_t)numel_of(&e->info) != (size_t)N * K / 2) {
            VV_LOG_E("loader: '%s' holds %lld bytes, not the %lld of a "
                     "[%d x %d] NF4 matrix", buf,
                     (long long)numel_of(&e->info),
                     (long long)N * K / 2, N, K);
            return VV_ERR_SHAPE_MISMATCH;
        }
        if ((s = load_raw(L, e, &w->tensor)) != VV_OK) return s;
        s = load_nf4_scales(L, buf, (size_t)N * K / VV_NF4_BLOCK, w);
        if (s != VV_OK) return s;
        w->quant_kind = VV_QUANT_NF4;
        w->is_quantized = true;
        L->n_nf4++;
        L->proj_bytes += w->tensor.size_bytes + w->quant.scales.size_bytes;
    } else if (e && is_float_dtype(e->info.dtype)) {
        if ((s = load_dense_weight(L, e, N, K, w)) != VV_OK) return s;
    } else if (e) {
        VV_LOG_E("loader: '%s' is %s, which no projection format uses "
                 "(int8 checkpoints are not supported yet)",
                 buf, dtype_name(e->info.dtype));
        return VV_ERR_UNSUPPORTED;
    } else {
        snprintf(buf, sizeof(buf), "%s.qweight", base);
        if (!find(L, buf)) {
            VV_LOG_E("loader: '%s' has neither .weight nor .qweight", base);
            return VV_ERR_WEIGHT_MISSING;
        }
        if (L->quant != VV_LOAD_QUANT_AUTO && L->quant != VV_LOAD_QUANT_INT4) {
            VV_LOG_E("loader: the checkpoint is AWQ/GPTQ int4; --quant %s "
                     "is not available for it", vv_load_quant_name(L->quant));
            return VV_ERR_UNSUPPORTED;
        }
        if ((s = load_awq_weight(L, base, N, K, w)) != VV_OK) return s;
        L->n_int4g++;
        L->proj_bytes += w->tensor.size_bytes + w->quant.scales.size_bytes
                       + w->mins.size_bytes;
    }

    snprintf(buf, sizeof(buf), "%s.bias", base);
    e = find(L, buf);
    if (e) {
        if ((s = check_shape(e, N, 0)) != VV_OK) return s;
        if ((s = load_f16(L, e, &w->bias)) != VV_OK) return s;
    } else if (want_bias) {
        VV_LOG_E("loader: '%s' is missing (attention_bias is on)", buf);
        return VV_ERR_WEIGHT_MISSING;
    }
    return VV_OK;
}

/* ─── Shards ────────────────────────────────────────────────────────────── */

static void build_path(char* buf, size_t buf_size,
                       const char* dir, const char* filename) {
    size_t dlen = strlen(dir);
    if (dlen > 0 && (dir[dlen-1] == '/' || dir[dlen-1] == '\\'))
        snprintf(buf, buf_size, "%s%s", dir, filename);
    else
        snprintf(buf, buf_size, "%s/%s", dir, filename);
}

/**
 * @brief The shard files: every distinct value of the index's weight_map, or
 *        the single `model.safetensors`. No limit on how many.
 */
static vv_status_t find_safetensors_files(const char* model_dir,
                                          char*** out_files, int* out_count) {
    char path[1024];
    build_path(path, sizeof(path), model_dir, "model.safetensors.index.json");
    *out_files = NULL;
    *out_count = 0;

    FILE* f = fopen(path, "rb");
    if (!f) {
        build_path(path, sizeof(path), model_dir, "model.safetensors");
        f = fopen(path, "rb");
        if (!f) {
            VV_LOG_E("loader: no safetensors files found in '%s'", model_dir);
            return VV_ERR_NOT_FOUND;
        }
        fclose(f);
        char** files = (char**)vv_alloc(sizeof(char*));
        if (!files) return VV_ERR_OUT_OF_MEMORY;
        files[0] = (char*)vv_alloc(sizeof("model.safetensors"));
        if (!files[0]) { vv_free(files); return VV_ERR_OUT_OF_MEMORY; }
        memcpy(files[0], "model.safetensors", sizeof("model.safetensors"));
        *out_files = files;
        *out_count = 1;
        return VV_OK;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return VV_ERR_PARSE; }
    char* json = (char*)vv_alloc((size_t)size + 1);
    if (!json) { fclose(f); return VV_ERR_OUT_OF_MEMORY; }
    const size_t got = fread(json, 1, (size_t)size, f);
    fclose(f);
    json[got] = '\0';

    cJSON* root = cJSON_Parse(json);
    vv_free(json);
    if (!root) return VV_ERR_PARSE;
    cJSON* weight_map = cJSON_GetObjectItem(root, "weight_map");
    if (!weight_map) { cJSON_Delete(root); return VV_ERR_PARSE; }

    int cap = cJSON_GetArraySize(weight_map);
    char** files = (char**)vv_alloc((size_t)(cap > 0 ? cap : 1) * sizeof(char*));
    if (!files) { cJSON_Delete(root); return VV_ERR_OUT_OF_MEMORY; }
    int n = 0;
    cJSON* entry;
    cJSON_ArrayForEach(entry, weight_map) {
        if (!cJSON_IsString(entry)) continue;
        const char* fname = entry->valuestring;
        /* A shard is a file name in the model directory, nothing else. */
        if (strstr(fname, "..") || strchr(fname, '/') || strchr(fname, '\\')) {
            VV_LOG_E("loader: index names '%s' outside the model directory",
                     fname);
            for (int i = 0; i < n; i++) vv_free(files[i]);
            vv_free(files);
            cJSON_Delete(root);
            return VV_ERR_MODEL_FORMAT;
        }
        bool seen = false;
        for (int i = 0; i < n && !seen; i++) seen = strcmp(files[i], fname) == 0;
        if (seen) continue;
        const size_t len = strlen(fname) + 1;
        files[n] = (char*)vv_alloc(len);
        if (!files[n]) {
            for (int i = 0; i < n; i++) vv_free(files[i]);
            vv_free(files);
            cJSON_Delete(root);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memcpy(files[n], fname, len);
        n++;
    }
    cJSON_Delete(root);

    *out_files = files;
    *out_count = n;
    VV_LOG_I("loader: found %d safetensors files", n);
    return n > 0 ? VV_OK : VV_ERR_NOT_FOUND;
}

/** @brief Sorted index of every tensor in every shard. */
static vv_status_t build_index(loader_t* L) {
    vv_model_t* m = L->m;
    int total = 0;
    for (int i = 0; i < m->n_st_files; i++)
        total += vv_safetensors_num_tensors(m->st_files[i]);
    L->ent = (st_entry_t*)vv_alloc((size_t)(total > 0 ? total : 1)
                                   * sizeof(st_entry_t));
    if (!L->ent) return VV_ERR_OUT_OF_MEMORY;
    int k = 0;
    for (int i = 0; i < m->n_st_files; i++) {
        const int nt = vv_safetensors_num_tensors(m->st_files[i]);
        for (int t = 0; t < nt; t++) {
            if (vv_safetensors_get_info(m->st_files[i], t, &L->ent[k].info)
                != VV_OK) continue;
            L->ent[k].file = i;
            k++;
        }
    }
    L->n_ent = k;
    qsort(L->ent, (size_t)k, sizeof(st_entry_t), entry_cmp);
    for (int i = 1; i < k; i++)
        if (strcmp(L->ent[i - 1].info.name, L->ent[i].info.name) == 0) {
            VV_LOG_E("loader: '%s' appears in two shards", L->ent[i].info.name);
            return VV_ERR_MODEL_FORMAT;
        }
    return VV_OK;
}

/* ─── Model parts ───────────────────────────────────────────────────────── */

/** @brief Where the LM lives: "model." (4-bit) or "model.language_model.". */
static vv_status_t find_prefix(loader_t* L) {
    const st_entry_t* e = find_suffix(L, "layers.0.input_layernorm.weight");
    if (!e) {
        VV_LOG_E("loader: no 'layers.0.input_layernorm.weight' — this is "
                 "not a Qwen2 checkpoint the runtime knows");
        return VV_ERR_WEIGHT_MISSING;
    }
    const size_t n = strlen(e->info.name) - strlen("layers.0.input_layernorm.weight");
    if (n >= sizeof(L->prefix)) return VV_ERR_MODEL_FORMAT;
    memcpy(L->prefix, e->info.name, n);
    L->prefix[n] = '\0';
    return VV_OK;
}

static vv_status_t load_layers(loader_t* L) {
    vv_model_t* m = L->m;
    const vv_llm_config_t* c = &m->config.llm;
    const int hs = c->hidden_size, qd = c->num_attention_heads * c->head_dim;
    const int kvd = c->num_key_value_heads * c->head_dim;
    const int inter = c->intermediate_size;
    const bool qkv_bias = c->attention_bias;
    char buf[320];
    vv_status_t s;

    /* A layer past the config's count is a config the file disagrees with. */
    snprintf(buf, sizeof(buf), "%slayers.%d.input_layernorm.weight",
             L->prefix, c->num_hidden_layers);
    if (find(L, buf)) {
        VV_LOG_E("loader: the checkpoint has more than the %d layers its "
                 "config names", c->num_hidden_layers);
        return VV_ERR_SHAPE_MISMATCH;
    }

    for (int i = 0; i < m->num_layers; i++) {
        vv_layer_weights_t* Ly = &m->layers[i];
        snprintf(buf, sizeof(buf), "%slayers.%d.input_layernorm.weight",
                 L->prefix, i);
        if ((s = need_f16(L, buf, hs, 0, &Ly->input_layernorm)) != VV_OK)
            return s;
        snprintf(buf, sizeof(buf), "%slayers.%d.post_attention_layernorm.weight",
                 L->prefix, i);
        if ((s = need_f16(L, buf, hs, 0, &Ly->post_attn_layernorm)) != VV_OK)
            return s;

        if ((s = load_projection(L, i, "self_attn.q_proj", qd, hs, qkv_bias,
                                 &Ly->attn.q_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.k_proj", kvd, hs, qkv_bias,
                                 &Ly->attn.k_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.v_proj", kvd, hs, qkv_bias,
                                 &Ly->attn.v_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.o_proj", hs, qd, false,
                                 &Ly->attn.o_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.gate_proj", inter, hs, false,
                                 &Ly->mlp.gate_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.up_proj", inter, hs, false,
                                 &Ly->mlp.up_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.down_proj", hs, inter, false,
                                 &Ly->mlp.down_proj)) != VV_OK)
            return s;
    }
    return VV_OK;
}

static vv_status_t load_lm_globals(loader_t* L) {
    vv_model_t* m = L->m;
    const vv_llm_config_t* c = &m->config.llm;
    char buf[320];
    vv_status_t s;

    snprintf(buf, sizeof(buf), "%sembed_tokens.weight", L->prefix);
    if ((s = need_f16(L, buf, c->vocab_size, c->hidden_size,
                      &m->embed_tokens)) != VV_OK)
        return s;
    snprintf(buf, sizeof(buf), "%snorm.weight", L->prefix);
    if ((s = need_f16(L, buf, c->hidden_size, 0, &m->final_norm)) != VV_OK)
        return s;

    const st_entry_t* head = find(L, "lm_head.weight");
    if (!head) head = find_suffix(L, ".lm_head.weight");
    if (c->tie_word_embeddings) {
        /*
         * Tied: the head is the embedding table. BitNet still ships a
         * byte-identical lm_head.weight; reading it would spend 467 MB of
         * host memory (and as much VRAM) on a copy.
         */
        m->lm_head = m->embed_tokens;
        m->lm_head_tied = true;
        if (head)
            VV_LOG_I("loader: tie_word_embeddings — '%s' ignored, the head "
                     "shares embed_tokens", head->info.name);
        return VV_OK;
    }
    if (!head) {
        VV_LOG_E("loader: no lm_head.weight and tie_word_embeddings is off");
        return VV_ERR_WEIGHT_MISSING;
    }
    if ((s = check_shape(head, c->vocab_size, c->hidden_size)) != VV_OK)
        return s;
    return load_f16(L, head, &m->lm_head);
}

/** @brief fc1 / norm / fc2 of one connector, FP32; biases when present. */
static vv_status_t load_connector(loader_t* L, const char* which, int vae_dim,
                                  vv_weight_t* fc1, vv_weight_t* norm,
                                  vv_weight_t* fc2) {
    const int hs = L->m->config.llm.hidden_size;
    struct { const char* part; vv_tensor_t* t; int64_t d0, d1; bool need; } p[] = {
        { "fc1.weight",  &fc1->tensor,       hs, vae_dim, true  },
        { "fc1.bias",    &fc1->quant.packed, hs, 0,       false },
        { "norm.weight", &norm->tensor,      hs, 0,       true  },
        { "fc2.weight",  &fc2->tensor,       hs, hs,      true  },
        { "fc2.bias",    &fc2->quant.packed, hs, 0,       false },
    };
    char buf[128];
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); i++) {
        snprintf(buf, sizeof(buf), "%s.%s", which, p[i].part);
        const st_entry_t* e = find_suffix(L, buf);
        if (!e) {
            if (!p[i].need) continue;
            VV_LOG_E("loader: '%s' is missing from the checkpoint", buf);
            return VV_ERR_WEIGHT_MISSING;
        }
        vv_status_t s = check_shape(e, p[i].d0, p[i].d1);
        if (s == VV_OK) s = load_f32(L, e, p[i].t);
        if (s != VV_OK) return s;
        if (i == 0) strncpy(fc1->name, e->info.name, sizeof(fc1->name) - 1);
    }
    return VV_OK;
}

/** @brief Every `<which>.encoder.*` tensor as FP32, in index order. */
static vv_status_t load_encoder(loader_t* L, const char* which,
                                vv_weight_t** out, int* n_out) {
    int n = 0;
    for (int i = 0; i < L->n_ent; i++)
        if (strstr(L->ent[i].info.name, which)) n++;
    *out = NULL;
    *n_out = 0;
    if (n == 0) {
        VV_LOG_E("loader: no '%s*' tensors — the speech encoder is missing",
                 which);
        return VV_ERR_WEIGHT_MISSING;
    }
    vv_weight_t* w = (vv_weight_t*)vv_alloc((size_t)n * sizeof(vv_weight_t));
    if (!w) return VV_ERR_OUT_OF_MEMORY;
    memset(w, 0, (size_t)n * sizeof(vv_weight_t));
    *out = w;
    int k = 0;
    for (int i = 0; i < L->n_ent && k < n; i++) {
        const st_entry_t* e = &L->ent[i];
        if (!strstr(e->info.name, which)) continue;
        vv_status_t s = load_f32(L, e, &w[k].tensor);
        if (s != VV_OK) { *n_out = k; return s; }
        strncpy(w[k].name, e->info.name, sizeof(w[k].name) - 1);
        k++;
    }
    *n_out = k;
    return VV_OK;
}

/* ─── Walking a layer ───────────────────────────────────────────────────── */

int vv_layer_projections(vv_layer_weights_t* L, vv_weight_t* out[7]) {
    out[0] = &L->attn.q_proj;   out[1] = &L->attn.k_proj;
    out[2] = &L->attn.v_proj;   out[3] = &L->attn.o_proj;
    out[4] = &L->mlp.gate_proj; out[5] = &L->mlp.up_proj;
    out[6] = &L->mlp.down_proj;
    return 7;
}

int vv_layer_tensors(vv_layer_weights_t* L,
                     vv_tensor_t* out[VV_LAYER_TENSOR_SLOTS]) {
    vv_weight_t* p[7];
    vv_layer_projections(L, p);
    int k = 0;
    out[k++] = &L->input_layernorm;
    out[k++] = &L->post_attn_layernorm;
    for (int i = 0; i < 7; i++) {
        out[k++] = &p[i]->tensor;
        out[k++] = &p[i]->quant.scales;
        out[k++] = &p[i]->mins;
        out[k++] = &p[i]->bias;
    }
    return k;
}

size_t vv_layer_bytes(const vv_layer_weights_t* L) {
    vv_tensor_t* t[VV_LAYER_TENSOR_SLOTS];
    const int n = vv_layer_tensors((vv_layer_weights_t*)L, t);
    size_t total = 0;
    for (int i = 0; i < n; i++)
        if (t[i]->data) total += t[i]->size_bytes;
    return total;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

vv_status_t vv_model_load(const char* model_dir, vv_model_t** out) {
    return vv_model_load_ex(model_dir, NULL, out);
}

vv_status_t vv_model_load_ex(const char* model_dir,
                             const vv_model_load_opts_t* opts,
                             vv_model_t** out) {
    if (!model_dir || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    const vv_model_load_opts_t o = opts ? *opts : vv_model_load_opts_default();
    if (o.quant < 0 || o.quant >= VV_LOAD_QUANT_COUNT) {
        VV_LOG_E("loader: unknown weight quantization %d", o.quant);
        return VV_ERR_INVALID_ARG;
    }

    VV_LOG_I("loader: loading model from '%s' (quant %s)", model_dir,
             vv_load_quant_name((vv_load_quant_t)o.quant));
    const double t0 = vv_time_ms();

    vv_model_t* model = (vv_model_t*)vv_alloc(sizeof(vv_model_t));
    if (!model) return VV_ERR_OUT_OF_MEMORY;
    memset(model, 0, sizeof(*model));

    vv_status_t s = vv_config_load(model_dir, &model->config);
    if (s != VV_OK) {
        VV_LOG_E("loader: failed to parse the model's config");
        vv_free(model);
        return s;
    }

    /* Find and open safetensors files */
    char** st_names = NULL;
    int n_st = 0;
    s = find_safetensors_files(model_dir, &st_names, &n_st);
    if (s != VV_OK) { vv_free(model); return s; }

    model->st_files = (vv_safetensors_t**)vv_alloc(
        (size_t)n_st * sizeof(vv_safetensors_t*));
    if (!model->st_files) {
        for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
        vv_free(st_names);
        vv_free(model);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(model->st_files, 0, (size_t)n_st * sizeof(vv_safetensors_t*));
    model->n_st_files = n_st;
    for (int i = 0; i < n_st && s == VV_OK; i++) {
        char full_path[1024];
        build_path(full_path, sizeof(full_path), model_dir, st_names[i]);
        s = vv_safetensors_open(full_path, &model->st_files[i]);
        if (s != VV_OK) VV_LOG_E("loader: failed to open '%s'", full_path);
    }
    for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
    vv_free(st_names);
    if (s != VV_OK) { vv_model_free(model); return s; }

    const int n_layers = model->config.llm.num_hidden_layers;
    model->num_layers = n_layers;
    model->layers = (vv_layer_weights_t*)vv_alloc(
        (size_t)n_layers * sizeof(vv_layer_weights_t));
    if (!model->layers) { vv_model_free(model); return VV_ERR_OUT_OF_MEMORY; }
    memset(model->layers, 0, (size_t)n_layers * sizeof(vv_layer_weights_t));

    loader_t L;
    memset(&L, 0, sizeof(L));
    L.m = model;
    L.quant = (vv_load_quant_t)o.quant;

    s = build_index(&L);
    if (s == VV_OK) s = find_prefix(&L);
    if (s == VV_OK) s = load_lm_globals(&L);
    if (s == VV_OK) s = load_layers(&L);
    if (s == VV_OK)
        s = load_connector(&L, "acoustic_connector",
                           model->config.acoustic_vae_dim,
                           &model->acoustic_connector_fc1,
                           &model->acoustic_connector_norm,
                           &model->acoustic_connector_fc2);
    if (s == VV_OK)
        s = load_connector(&L, "semantic_connector",
                           model->config.semantic_vae_dim,
                           &model->semantic_connector_fc1,
                           &model->semantic_connector_norm,
                           &model->semantic_connector_fc2);
    if (s == VV_OK)
        s = load_encoder(&L, "acoustic_tokenizer.encoder.",
                         &model->acoustic_weights, &model->n_acoustic_weights);
    if (s == VV_OK)
        s = load_encoder(&L, "semantic_tokenizer.encoder.",
                         &model->semantic_weights, &model->n_semantic_weights);
    vv_free(L.ent);

    if (s != VV_OK) {
        VV_LOG_E("loader: '%s' cannot be loaded: %s", model_dir,
                 vv_status_str(s));
        vv_model_free(model);
        return s;
    }

    int n_bias = 0;
    for (int li = 0; li < n_layers; li++) {
        vv_weight_t* p[7];
        vv_layer_projections(&model->layers[li], p);
        for (int k = 0; k < 7; k++) if (p[k]->bias.data) n_bias++;
    }

    VV_LOG_I("loader: %s model loaded in %.0f ms (%d layers under prefix "
             "'%s'; projections %d nf4 / %d int4g / %d fp16, %d quantized "
             "at load, %.1f MB; head %s; encoders %d+%d tensors; %d biases)",
             vv_model_family_name(model->config.family),
             vv_time_ms() - t0, n_layers, L.prefix, L.n_nf4, L.n_int4g,
             L.n_dense, L.n_converted,
             (double)L.proj_bytes / (1024.0 * 1024.0),
             model->lm_head_tied ? "tied to embed_tokens" : "separate",
             model->n_acoustic_weights, model->n_semantic_weights, n_bias);

    *out = model;
    return VV_OK;
}

vv_status_t vv_model_free(vv_model_t* model) {
    if (!model) return VV_ERR_NULL_PTR;

    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            vv_tensor_t* t[VV_LAYER_TENSOR_SLOTS];
            const int n = vv_layer_tensors(&model->layers[i], t);
            for (int k = 0; k < n; k++) vv_tensor_free(t[k]);
        }
        vv_free(model->layers);
    }

    /* A tied head is the embedding table: freed once, as embed_tokens. */
    if (model->lm_head_tied) memset(&model->lm_head, 0, sizeof(model->lm_head));
    vv_tensor_free(&model->embed_tokens);
    vv_tensor_free(&model->final_norm);
    vv_tensor_free(&model->lm_head);

    /* Connector weights; `.quant.packed` holds the biases. */
    vv_weight_t* conn[6] = {
        &model->acoustic_connector_fc1, &model->acoustic_connector_norm,
        &model->acoustic_connector_fc2, &model->semantic_connector_fc1,
        &model->semantic_connector_norm, &model->semantic_connector_fc2,
    };
    for (int i = 0; i < 6; i++) {
        vv_tensor_free(&conn[i]->tensor);
        vv_tensor_free(&conn[i]->quant.packed);
    }

    if (model->acoustic_weights) {
        for (int i = 0; i < model->n_acoustic_weights; i++)
            vv_tensor_free(&model->acoustic_weights[i].tensor);
        vv_free(model->acoustic_weights);
    }
    if (model->semantic_weights) {
        for (int i = 0; i < model->n_semantic_weights; i++)
            vv_tensor_free(&model->semantic_weights[i].tensor);
        vv_free(model->semantic_weights);
    }

    if (model->st_files) {
        for (int i = 0; i < model->n_st_files; i++)
            if (model->st_files[i]) vv_safetensors_close(model->st_files[i]);
        vv_free(model->st_files);
    }

    vv_free(model);
    return VV_OK;
}
