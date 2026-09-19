/**
 * @file loader.c
 * @brief Orchestrates loading all model components from safetensors + config.
 *
 * What a tensor becomes is decided by what the file holds, not by its name:
 *
 *   projection `.weight` U8 + `.absmax`  → NF4 (bitsandbytes)
 *   projection `.qweight` I32            → INT4G (AWQ / GPTQ, repacked)
 *   projection `.weight` F16/BF16/F32    → dense FP16, or quantized to NF4,
 *                                          INT4G or INT8 while it is read
 *                                          (--quant)
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
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/smooth.h"
#include "vibevoice/bitnet.h"
#include "vibevoice/vae_i8.h"
#include "bitnet_load.h"
#include "cJSON.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─── Tensor index over every shard ─────────────────────────────────────── */

typedef struct {
    vv_st_tensor_info_t info;
    int                 file;
} st_entry_t;

/* ─── compressed-tensors (llm-compressor) ───────────────────────────────── */

/*
 * What llm-compressor writes for a quantized Linear, per projection:
 *
 *   int-quantized   weight            I8   [N, K]  the integer values
 *   pack-quantized  weight_packed     I32  [N, K * bits / 32], element k at
 *                                          bits 4 * (k % 8) of word k / 8,
 *                                          stored as value + 8 (unsigned)
 *                   weight_shape      I64  [2] = { N, K }
 *   both            weight_scale      float [N, 1] (channel) or [N, K/G]
 *                   weight_zero_point asymmetric only: I8 [N, K/G], or
 *                                     packed along N like the weight
 *                                     (I32 [N/8, K/G]) in pack-quantized
 *                   weight_g_idx      act-order "group" only
 *
 * and w = (q - z) * s with q, z signed. The runtime maps 8-bit channel
 * weights onto INT8 (w = q * s: symmetric only) and 4-bit group weights onto
 * INT4G, whose codes and zero points are the same numbers shifted by 8 into
 * 0..15. Nothing is re-rounded except a BF16/F32 group scale to FP16, which
 * is exact for BF16 scales in FP16's normal range.
 */

/** @brief quantization_config of a compressed-tensors checkpoint. */
typedef struct {
    bool present;      /**< quant_method == "compressed-tensors"            */
    int  w_bits;       /**< weights.num_bits of the Linear scheme (0: none) */
    bool act_int8;     /**< input_activations: 8-bit int, dynamic           */
    bool mixed;        /**< config_groups disagree on weight bits or on
                            int8 activations; not supported               */
    char format[32];   /**< "int-quantized", "pack-quantized", ...           */
} ct_cfg_t;

typedef struct {
    vv_model_t*        m;
    st_entry_t*        ent;      /**< sorted by name */
    int                n_ent;
    vv_load_quant_t    quant;
    char               prefix[128];  /**< "model." / "model.language_model." */
    /* What the projections turned out to be, for the summary line. */
    int                n_nf4, n_int4g, n_int8, n_dense, n_converted;
    int                n_ternary;
    /** asr-bitnet from F32 safetensors: ternarize, FP32 norms and biases */
    bool               bitnet;
    int                vae;       /**< vv_vae_numerics_t for bitnet      */
    int                head;      /**< vv_head_format_t for bitnet       */
    /** 1 when GPTQ zeros are stored minus one (all but "gptq_v2") */
    int                gptq_bias;
    size_t             proj_bytes;
    ct_cfg_t           ct;       /**< compressed-tensors quantization_config */
    /** SmoothQuant statistics to fold in (NULL: none), and how. */
    const vv_smooth_stats_t* smooth;
    float              smooth_alpha;
    int                smooth_maps;
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

/** @brief Elements [i0, i1) of `src` to FP32, in the calling thread. */
static void to_f32_range(const void* src, vv_dtype_t dt, float* dst,
                         size_t i0, size_t i1) {
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

static void to_f32(const void* src, vv_dtype_t dt, float* dst, size_t n) {
    const int blocks = (int)((n + CONV_BLOCK - 1) / CONV_BLOCK);
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (b = 0; b < blocks; b++) {
        const size_t i0 = (size_t)b * CONV_BLOCK;
        const size_t i1 = (i0 + CONV_BLOCK < n) ? i0 + CONV_BLOCK : n;
        to_f32_range(src, dt, dst, i0, i1);
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
                dst[i] = vv_float_to_half_rne(vv_bf16_to_float(s[i]));
        } else if (dt == VV_DTYPE_F32) {
            const float* s = (const float*)src;
            for (size_t i = i0; i < i1; i++)
                dst[i] = vv_float_to_half_rne(s[i]);
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

/** @brief An LM norm: FP32 for BitNet (the reference keeps them), else FP16. */
static vv_status_t need_norm(const loader_t* L, const char* name, int64_t d0,
                             vv_tensor_t* t) {
    if (!L->bitnet) return need_f16(L, name, d0, 0, t);
    const st_entry_t* e = find(L, name);
    if (!e) {
        VV_LOG_E("loader: '%s' is missing from the checkpoint", name);
        return VV_ERR_WEIGHT_MISSING;
    }
    vv_status_t s = check_shape(e, d0, 0);
    return s != VV_OK ? s : load_f32(L, e, t);
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
        const int64_t n_super_ll = numel_of(&en->info);
        if (!nested || n_super_ll <= 0 || n_super_ll > (int64_t)n_blocks) {
            VV_LOG_E("loader: '%s' holds %lld scales for %zu blocks", buf,
                     (long long)n_super_ll, n_blocks);
            vv_free(scales);
            return VV_ERR_MODEL_FORMAT;
        }
        const size_t n_super = (size_t)n_super_ll;

        float nested_offset = 0.0f;
        size_t nested_block = 256;
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
                    /* Clamped before the cast: a value past the block
                       count means one superblock, and a double beyond
                       the integer range is undefined to convert. */
                    if (nbs && cJSON_IsNumber(nbs) && nbs->valuedouble >= 1.0)
                        nested_block = nbs->valuedouble >= (double)n_blocks
                                     ? n_blocks : (size_t)nbs->valuedouble;
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

        /* Every block needs its superblock scale. */
        if ((n_blocks + nested_block - 1) / nested_block > n_super) {
            VV_LOG_E("loader: '%s.nested_absmax' has %zu scales; %zu blocks "
                     "of %zu need %zu", wname, n_super, n_blocks,
                     nested_block,
                     (n_blocks + nested_block - 1) / nested_block);
            vv_free(scales);
            return VV_ERR_SHAPE_MISMATCH;
        }

        const uint8_t* codes = (const uint8_t*)abs_src;
        for (size_t i = 0; i < n_blocks; i++) {
            const size_t si = i / nested_block;
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
 *
 * The two formats pack along different axes and are told apart by shape:
 *   AWQ   qweight [K, N/8], scales [K/G, N]   (scales width = 8 x qweight)
 *   GPTQ  qweight [K/8, N], scales [K/G, N]   (scales width = qweight)
 * A GPTQ g_idx is checked: plain k / G loads as is, an act-order one has
 * its input channels sorted by group and keeps the order in w->perm.
 * Every size is checked against the header before anything is indexed:
 * the file is untrusted input.
 */
static vv_status_t load_awq_weight(const loader_t* L, const char* base,
                                   int N_want, int K_want, vv_weight_t* w) {
    char buf[320];
    vv_tensor_t qweight = {0}, qzeros = {0}, scales = {0}, g_idx = {0};
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
    if (!e || e->info.dtype != VV_DTYPE_I32 || e->info.ndim != 2) {
        VV_LOG_E("loader: '%s' is missing or not a 2-D I32", buf);
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
    snprintf(buf, sizeof(buf), "%s.g_idx", base);
    e = find(L, buf);
    const bool have_gidx = e != NULL;
    if (e && (s = load_raw(L, e, &g_idx)) != VV_OK) {
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        vv_tensor_free(&scales);
        return s;
    }

    #define AWQ_FREE_INPUTS() do { vv_tensor_free(&qweight);           \
        vv_tensor_free(&qzeros); vv_tensor_free(&scales);             \
        vv_tensor_free(&g_idx); } while (0)

    const bool gptq_layout = scales.shape[1] == qweight.shape[1] &&
                             scales.shape[1] != qweight.shape[1] * 8;
    const int K = gptq_layout ? (int)qweight.shape[0] * 8
                              : (int)qweight.shape[0];
    const int N = gptq_layout ? (int)qweight.shape[1]
                              : (int)qweight.shape[1] * 8;
    const int n_groups = (int)scales.shape[0];
    if (K != K_want || N != N_want || n_groups <= 0 || (K % n_groups) != 0 ||
        (N % 8) != 0 || scales.shape[1] != N) {
        VV_LOG_E("loader: '%s' %s tensors are [%d x %d] with %d groups; "
                 "the config says [%d x %d]", base,
                 gptq_layout ? "GPTQ" : "AWQ", K, N, n_groups,
                 K_want, N_want);
        AWQ_FREE_INPUTS();
        return VV_ERR_SHAPE_MISMATCH;
    }
    /* The repack indexes all three by these sizes: hold the data to them. */
    if (qweight.size_bytes != (size_t)K * N / 2 ||
        qzeros.size_bytes != (size_t)n_groups * (N / 8) * 4 ||
        scales.size_bytes != (size_t)n_groups * N * 2 ||
        qzeros.shape[0] != n_groups || qzeros.shape[1] != N / 8) {
        VV_LOG_E("loader: '%s' tensor sizes do not match K=%d N=%d "
                 "groups=%d", base, K, N, n_groups);
        AWQ_FREE_INPUTS();
        return VV_ERR_SHAPE_MISMATCH;
    }
    const int group_size = K / n_groups;
    if (have_gidx && (g_idx.dtype != VV_DTYPE_I32 ||
                      g_idx.size_bytes != (size_t)K * 4)) {
        VV_LOG_E("loader: '%s.g_idx' is not int32[%d]", base, K);
        AWQ_FREE_INPUTS();
        return VV_ERR_SHAPE_MISMATCH;
    }

    const size_t packed_bytes = (size_t)N * (K / 2);
    const size_t group_bytes  = (size_t)N * n_groups * sizeof(uint16_t);
    const size_t zero_bytes   = (size_t)N * n_groups;
    uint8_t*  packed = (uint8_t*)vv_alloc(packed_bytes);
    uint16_t* sc     = (uint16_t*)vv_alloc(group_bytes);
    uint16_t* mn     = (uint16_t*)vv_alloc(group_bytes);
    uint8_t*  zr     = (uint8_t*)vv_alloc(zero_bytes);
    int32_t*  perm   = (gptq_layout && have_gidx)
                     ? (int32_t*)vv_alloc((size_t)K * sizeof(int32_t)) : NULL;
    if (!packed || !sc || !mn || !zr || (gptq_layout && have_gidx && !perm)) {
        vv_free(packed); vv_free(sc); vv_free(mn); vv_free(zr); vv_free(perm);
        AWQ_FREE_INPUTS();
        return VV_ERR_OUT_OF_MEMORY;
    }

    /*
     * GPTQ writes zero_point - 1 (unless it is the v2 format); AWQ writes it
     * directly. An AWQ-packed tensor with a g_idx next to it is what earlier
     * versions of this loader called GPTQ, so it keeps that meaning.
     */
    int zero_bias;
    if (gptq_layout) {
        zero_bias = L->gptq_bias;
        s = vv_gptq_repack((const uint32_t*)qweight.data,
                           (const uint32_t*)qzeros.data,
                           (const uint16_t*)scales.data,
                           have_gidx ? (const int32_t*)g_idx.data : NULL,
                           K, N, group_size, zero_bias, packed, sc, mn, zr,
                           perm);
    } else {
        zero_bias = have_gidx ? 1 : 0;
        s = vv_awq_repack((const uint32_t*)qweight.data,
                          (const uint32_t*)qzeros.data,
                          (const uint16_t*)scales.data,
                          K, N, group_size, zero_bias, packed, sc, mn, zr);
    }
    AWQ_FREE_INPUTS();
    #undef AWQ_FREE_INPUTS
    if (s != VV_OK) {
        VV_LOG_E("loader: cannot repack '%s': %s", base, vv_status_str(s));
        vv_free(packed); vv_free(sc); vv_free(mn); vv_free(zr); vv_free(perm);
        return s;
    }
    /* Only an act-order weight needs its order kept. */
    if (perm) {
        bool identity = true;
        for (int k = 0; k < K && identity; k++) identity = perm[k] == k;
        if (identity) { vv_free(perm); perm = NULL; }
    }

    w->quant_kind   = VV_QUANT_INT4G;
    w->is_quantized = true;
    w->group_size   = group_size;
    w->int4g_layout = VV_INT4G_ROWMAJOR;

    const int64_t pshape[2] = { N, K / 2 };
    const int64_t gshape[2] = { N, n_groups };
    set_tensor(&w->tensor, packed, VV_DTYPE_U8, packed_bytes, 2, pshape);
    set_tensor(&w->quant.scales, sc, VV_DTYPE_F16, group_bytes, 2, gshape);
    set_tensor(&w->mins, mn, VV_DTYPE_F16, group_bytes, 2, gshape);

    w->zeros = w->quant.scales;
    w->zeros.data = zr;
    w->zeros.size_bytes = zero_bytes;
    w->zeros.dtype = VV_DTYPE_U8;

    if (perm) {
        w->perm.data = perm;
        w->perm.size_bytes = (size_t)K * sizeof(int32_t);
        w->perm.dtype = VV_DTYPE_I32;
        w->perm.ndim = 1;
        w->perm.shape[0] = K;
    }

    VV_LOG_D("loader: %s '%s' N=%d K=%d group=%d%s%s",
             gptq_layout ? "GPTQ" : "AWQ", base, N, K, group_size,
             zero_bias ? " (zeros stored minus one)" : "",
             perm ? " (act-order)" : "");
    return VV_OK;
}

vv_status_t vv_model_int4g_to_gpu_layout(vv_model_t* model, int* n_converted)
{
    if (n_converted) *n_converted = 0;
    if (!model) return VV_ERR_NULL_PTR;
    const char* legacy = getenv("VV_INT4G_LEGACY");
    if (legacy && legacy[0] && legacy[0] != '0') {
        VV_LOG_I("loader: VV_INT4G_LEGACY set, INT4 weights keep the "
                 "row-major layout and the older kernels");
        return VV_OK;
    }

    const double t0 = vv_time_ms();
    int converted = 0, kept = 0;
    for (int li = 0; li < model->num_layers; li++) {
        vv_weight_t* ws[7];
        const int nw = vv_layer_projections(&model->layers[li], ws);
        for (int wi = 0; wi < nw; wi++) {
            vv_weight_t* w = ws[wi];
            if (w->quant_kind != VV_QUANT_INT4G ||
                w->int4g_layout != VV_INT4G_ROWMAJOR)
                continue;
            const int N = (int)w->tensor.shape[0];
            const int K = (int)w->tensor.shape[1] * 2;
            const int G = w->group_size;
            const bool shape_ok = (K % 64) == 0 && (N % 8) == 0 &&
                                  (G == 32 || G == 64 || G == 128 ||
                                   G == 256);
            if (!w->zeros.data || !shape_ok || w->tensor.on_gpu ||
                w->quant.scales.on_gpu) {
                kept++;
                continue;
            }
            const int n_groups = K / G;
            const size_t sz_bytes = (size_t)N * n_groups * 4;
            uint16_t* sz = (uint16_t*)vv_alloc(sz_bytes);
            if (!sz) return VV_ERR_OUT_OF_MEMORY;
            vv_status_t s = vv_int4g_to_gpu_layout(
                (uint8_t*)w->tensor.data, (const uint16_t*)w->quant.scales.data,
                (const uint8_t*)w->zeros.data, N, K, G, sz);
            if (s != VV_OK) { vv_free(sz); return s; }

            vv_tensor_free(&w->quant.scales);
            vv_tensor_free(&w->mins);
            vv_tensor_free(&w->zeros);
            memset(&w->mins, 0, sizeof(w->mins));
            memset(&w->zeros, 0, sizeof(w->zeros));
            w->quant.scales.data = sz;
            w->quant.scales.size_bytes = sz_bytes;
            w->quant.scales.dtype = VV_DTYPE_F16;
            w->quant.scales.ndim = 2;
            w->quant.scales.shape[0] = N;
            w->quant.scales.shape[1] = n_groups * 2;
            w->quant.scales.on_gpu = false;
            w->int4g_layout = VV_INT4G_GPU;
            converted++;
        }
    }
    if (converted || kept)
        VV_LOG_I("loader: %d INT4 projections in the W4A16 GPU layout "
                 "(%d kept row-major) in %.0f ms", converted, kept,
                 vv_time_ms() - t0);
    if (n_converted) *n_converted = converted;
    return VV_OK;
}

/* ─── SmoothQuant folding (smooth.h) ────────────────────────────────────── */

/** @brief An FP16 vector divided by a per-element factor. */
static void div_f16(vv_tensor_t* t, const float* f) {
    uint16_t* h = (uint16_t*)t->data;
    const size_t n = t->size_bytes / sizeof(uint16_t);
    for (size_t i = 0; i < n; i++)
        h[i] = vv_float_to_half_rne(vv_half_to_float(h[i]) / f[i]);
}

/**
 * @brief acc[k] = max(acc[k], max_r |W[r, k]|) for the dense projection
 *        `<prefix>layers.<layer>.<which>.weight` [N, K].
 *
 * Read straight from the mapping, a few rows per thread; each thread keeps
 * its own maxima and they meet at the end, so the result does not depend on
 * the thread count.
 */
static vv_status_t dense_col_absmax(const loader_t* L, int layer,
                                    const char* which, int N, int K,
                                    float* acc) {
    char buf[320];
    snprintf(buf, sizeof(buf), "%slayers.%d.%s.weight", L->prefix, layer,
             which);
    const st_entry_t* e = find(L, buf);
    if (!e || !is_float_dtype(e->info.dtype)) {
        VV_LOG_E("loader: SmoothQuant folds into dense weights; '%s' is %s",
                 buf, e ? dtype_name(e->info.dtype) : "missing");
        return e ? VV_ERR_UNSUPPORTED : VV_ERR_WEIGHT_MISSING;
    }
    vv_status_t s = check_shape(e, N, K);
    if (s != VV_OK) return s;
    const void* src = data_of(L, e);
    if (!src) return VV_ERR_MODEL_FORMAT;
    const size_t es = vv_dtype_size(e->info.dtype);

    enum { ROWS = 8 };
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    float* part = (float*)vv_alloc((size_t)nt * (size_t)K * sizeof(float));
    float* tmp = (float*)vv_alloc((size_t)nt * ROWS * (size_t)K *
                                  sizeof(float));
    if (!part || !tmp) {
        vv_free(part);
        vv_free(tmp);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(part, 0, (size_t)nt * (size_t)K * sizeof(float));
    const int n_blk = (N + ROWS - 1) / ROWS;
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (b = 0; b < n_blk; b++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float* t = tmp + (size_t)tid * ROWS * (size_t)K;
        float* p = part + (size_t)tid * (size_t)K;
        const int r0 = b * ROWS;
        const int rows = (r0 + ROWS <= N) ? ROWS : N - r0;
        to_f32_range((const uint8_t*)src + (size_t)r0 * K * es,
                     e->info.dtype, t, 0, (size_t)rows * K);
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < K; k++) {
                const float v = fabsf(t[(size_t)r * K + k]);
                if (v > p[k]) p[k] = v;
            }
    }
    for (int i = 0; i < nt; i++)
        for (int k = 0; k < K; k++)
            if (part[(size_t)i * K + k] > acc[k])
                acc[k] = part[(size_t)i * K + k];
    vv_free(part);
    vv_free(tmp);
    return VV_OK;
}

/** @brief Factors of one layer; NULL members are not smoothed. */
typedef struct {
    float* qkv;     /**< [hidden] input_layernorm -> q, k, v           */
    float* gu;      /**< [hidden] post_attention_layernorm -> gate, up */
    float* vo;      /**< [kv_dim] v_proj rows -> o_proj                */
    float* o_cols;  /**< [q_dim]  vo spread over o_proj's columns      */
    float* ud;      /**< [inter]  up_proj rows -> down_proj            */
} smooth_fac_t;

/** @brief Smallest and largest factor seen, for the load log. */
static void smooth_range(const float* f, int n, float* lo, float* hi) {
    for (int i = 0; f && i < n; i++) {
        if (f[i] < *lo) *lo = f[i];
        if (f[i] > *hi) *hi = f[i];
    }
}

/**
 * @brief SmoothQuant factors of layer `li` from the calibration ranges and
 *        the dense weights' column ranges.
 *
 * The members of `f` point into scratch the caller owns; those of maps that
 * are off are set to NULL. `wmax` is scratch of max(hidden, q_dim, inter)
 * floats.
 */
static vv_status_t smooth_layer(const loader_t* L, int li, float* wmax,
                                smooth_fac_t* f) {
    const vv_llm_config_t* c = &L->m->config.llm;
    const vv_smooth_stats_t* st = L->smooth;
    const int hs = c->hidden_size, hd = c->head_dim;
    const int nh = c->num_attention_heads, nkv = c->num_key_value_heads;
    const int qd = nh * hd, kvd = nkv * hd, inter = c->intermediate_size;
    const float a = L->smooth_alpha;
    const float* row = st->absmax + (size_t)li * st->width;
    vv_status_t s;

    if (!(L->smooth_maps & VV_SMOOTH_QKV)) f->qkv = NULL;
    if (!(L->smooth_maps & VV_SMOOTH_GATEUP)) f->gu = NULL;
    if (!(L->smooth_maps & VV_SMOOTH_VO)) f->vo = f->o_cols = NULL;
    if (!(L->smooth_maps & VV_SMOOTH_UPDOWN)) f->ud = NULL;

    if (f->qkv) {
        const float* am = row + vv_smooth_offset(st, VV_SMOOTH_ATTN_IN);
        memset(wmax, 0, (size_t)hs * sizeof(float));
        if ((s = dense_col_absmax(L, li, "self_attn.q_proj", qd, hs,
                                  wmax)) != VV_OK ||
            (s = dense_col_absmax(L, li, "self_attn.k_proj", kvd, hs,
                                  wmax)) != VV_OK ||
            (s = dense_col_absmax(L, li, "self_attn.v_proj", kvd, hs,
                                  wmax)) != VV_OK)
            return s;
        for (int k = 0; k < hs; k++)
            f->qkv[k] = vv_smooth_factor(am[k], wmax[k], a);
    }
    if (f->gu) {
        const float* am = row + vv_smooth_offset(st, VV_SMOOTH_MLP_IN);
        memset(wmax, 0, (size_t)hs * sizeof(float));
        if ((s = dense_col_absmax(L, li, "mlp.gate_proj", inter, hs,
                                  wmax)) != VV_OK ||
            (s = dense_col_absmax(L, li, "mlp.up_proj", inter, hs,
                                  wmax)) != VV_OK)
            return s;
        for (int k = 0; k < hs; k++)
            f->gu[k] = vv_smooth_factor(am[k], wmax[k], a);
    }
    if (f->vo) {
        /* KV channel ch = (h / group) * hd + d feeds o_proj column
           h * hd + d of every query head h of its group: one factor. */
        const float* am = row + vv_smooth_offset(st, VV_SMOOTH_ATTN_OUT);
        const int group = nh / nkv;
        memset(wmax, 0, (size_t)qd * sizeof(float));
        if ((s = dense_col_absmax(L, li, "self_attn.o_proj", hs, qd,
                                  wmax)) != VV_OK)
            return s;
        for (int ch = 0; ch < kvd; ch++) {
            float ra = 0.0f, rw = 0.0f;
            const int g = ch / hd, d = ch % hd;
            for (int h = g * group; h < (g + 1) * group; h++) {
                const int j = h * hd + d;
                if (am[j] > ra) ra = am[j];
                if (wmax[j] > rw) rw = wmax[j];
            }
            f->vo[ch] = vv_smooth_factor(ra, rw, a);
        }
        for (int j = 0; j < qd; j++)
            f->o_cols[j] = f->vo[((j / hd) / group) * hd + j % hd];
    }
    if (f->ud) {
        const float* am = row + vv_smooth_offset(st, VV_SMOOTH_MLP_MID);
        memset(wmax, 0, (size_t)inter * sizeof(float));
        if ((s = dense_col_absmax(L, li, "mlp.down_proj", hs, inter,
                                  wmax)) != VV_OK)
            return s;
        for (int k = 0; k < inter; k++)
            f->ud[k] = vv_smooth_factor(am[k], wmax[k], a);
    }
    return VV_OK;
}

/* ─── Dense projections, kept or quantized on the way in ────────────────── */

/**
 * @brief A dense [N,K] projection: FP16, or NF4 / INT4G / INT8 quantized
 *        straight from the mapping.
 *
 * Quantization goes through a few FP32 rows per thread at a time, so peak
 * memory is the quantized model plus a few MB, never a dense copy of the
 * whole thing. Rows are independent, so the bytes do not depend on the
 * thread count.
 */
static vv_status_t load_dense_weight(loader_t* L, const st_entry_t* e,
                                     int N, int K, const float* col_mul,
                                     const float* row_div, vv_weight_t* w) {
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

    /* W8A8 and W4A8 store exactly what INT8 and INT4 store; only the
       activations they run on differ. */
    const int kind = (q == VV_LOAD_QUANT_NF4) ? VV_QUANT_NF4
                   : (q == VV_LOAD_QUANT_INT8 || q == VV_LOAD_QUANT_W8A8)
                   ? VV_QUANT_INT8 : VV_QUANT_INT4G;
    /* The group a row is cut into: NF4 blocks, INT4G groups, or the whole
       row for per-channel INT8. */
    const int group = kind == VV_QUANT_NF4 ? VV_NF4_BLOCK
                    : kind == VV_QUANT_INT4G ? VV_INT4G_LOAD_GROUP : K;
    if (K % group != 0) {
        VV_LOG_E("loader: '%s' has K=%d, not a multiple of the %s group %d",
                 e->info.name, K, vv_load_quant_name(q), group);
        return VV_ERR_UNSUPPORTED;
    }
    const int n_groups = K / group;
    const size_t code_bytes = kind == VV_QUANT_INT8
                            ? (size_t)N * (size_t)K : (size_t)N * (size_t)(K / 2);
    const size_t scale_bytes = kind == VV_QUANT_INT8
                             ? (size_t)N * sizeof(float)
                             : (size_t)N * (size_t)n_groups * sizeof(uint16_t);

    /*
     * One parallel region per projection. Each worker converts a few rows
     * into its own FP32 buffer and quantizes them there, so the rows go from
     * the mapping to their codes while they are still in cache. Converting a
     * 32 MB slab and then quantizing it opened two regions per slab, and with
     * a thread on every SMT sibling most of the load went to their barriers.
     */
    enum { ROWS = 8 };
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    uint8_t* codes = (uint8_t*)vv_alloc(code_bytes);
    void* sc = vv_alloc(scale_bytes);
    uint16_t* mn = kind == VV_QUANT_INT4G ? (uint16_t*)vv_alloc(scale_bytes)
                                          : NULL;
    /* INT4G keeps its integer zero points for the W4A16 GPU layout. */
    const size_t zero_bytes = (size_t)N * (size_t)n_groups;
    uint8_t* zr = kind == VV_QUANT_INT4G ? (uint8_t*)vv_alloc(zero_bytes)
                                         : NULL;
    float* tmp = (float*)vv_alloc((size_t)nt * ROWS * (size_t)K * sizeof(float));
    if (!codes || !sc || (kind == VV_QUANT_INT4G && (!mn || !zr)) || !tmp) {
        vv_free(codes); vv_free(sc); vv_free(mn); vv_free(zr); vv_free(tmp);
        return VV_ERR_OUT_OF_MEMORY;
    }

    const int n_blk = (N + ROWS - 1) / ROWS;
    vv_status_t first_err = VV_OK;
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (b = 0; b < n_blk; b++) {
#ifdef _OPENMP
        float* t = tmp + (size_t)omp_get_thread_num() * ROWS * (size_t)K;
#else
        float* t = tmp;
#endif
        const int r0 = b * ROWS;
        const int rows = (r0 + ROWS <= N) ? ROWS : N - r0;
        to_f32_range((const uint8_t*)src + (size_t)r0 * K * es, dt, t, 0,
                     (size_t)rows * K);
        /* SmoothQuant: W[r, k] * s_k (input side), / s_r (output side). */
        if (col_mul || row_div)
            for (int r = 0; r < rows; r++) {
                float* tr = t + (size_t)r * K;
                const float rd = row_div ? 1.0f / row_div[r0 + r] : 1.0f;
                if (col_mul)
                    for (int k = 0; k < K; k++) tr[k] *= col_mul[k] * rd;
                else
                    for (int k = 0; k < K; k++) tr[k] *= rd;
            }
        vv_status_t st;
        if (kind == VV_QUANT_NF4)
            st = vv_nf4_quantize(t, rows, K, codes + (size_t)r0 * (K / 2),
                                 (uint16_t*)sc + (size_t)r0 * n_groups);
        else if (kind == VV_QUANT_INT8)
            st = vv_int8_quantize_rows(t, rows, K,
                                       (int8_t*)codes + (size_t)r0 * K,
                                       (float*)sc + r0);
        else
            st = vv_int4g_quantize(t, rows, K, group,
                                   codes + (size_t)r0 * (K / 2),
                                   (uint16_t*)sc + (size_t)r0 * n_groups,
                                   mn + (size_t)r0 * n_groups,
                                   zr + (size_t)r0 * n_groups);
        if (st != VV_OK) {
#ifdef _OPENMP
#pragma omp critical(vv_loader_quant_err)
#endif
            first_err = st;
        }
    }
    vv_free(tmp);
    if (first_err != VV_OK) {
        vv_free(codes); vv_free(sc); vv_free(mn); vv_free(zr);
        return first_err;
    }

    w->is_quantized = true;
    w->quant_kind = kind;
    w->act_int8 = vv_load_quant_act8(q);
    if (kind == VV_QUANT_INT8) {
        const int64_t qshape[2] = { N, K };
        const int64_t sshape[1] = { N };
        set_tensor(&w->tensor, codes, VV_DTYPE_I8, code_bytes, 2, qshape);
        set_tensor(&w->quant.scales, sc, VV_DTYPE_F32, scale_bytes, 1, sshape);
        L->n_int8++;
    } else {
        const int64_t pshape[2] = { N, K / 2 };
        const int64_t gshape[2] = { N, n_groups };
        set_tensor(&w->tensor, codes, VV_DTYPE_U8, code_bytes, 2, pshape);
        set_tensor(&w->quant.scales, sc, VV_DTYPE_F16, scale_bytes, 2, gshape);
        if (kind == VV_QUANT_NF4) {
            w->quant.block_size = VV_NF4_BLOCK;
            w->quant.double_quant = false;
            L->n_nf4++;
        } else {
            set_tensor(&w->mins, mn, VV_DTYPE_F16, scale_bytes, 2, gshape);
            set_tensor(&w->zeros, zr, VV_DTYPE_U8, zero_bytes, 2, gshape);
            w->group_size = group;
            w->int4g_layout = VV_INT4G_ROWMAJOR;
            L->n_int4g++;
        }
    }
    L->n_converted++;
    L->proj_bytes += code_bytes + scale_bytes * (mn ? 2 : 1);
    return VV_OK;
}

/** @brief `n` elements of an integer tensor as int32 (I8 / I32 / I64). */
static bool ct_ints(const void* src, vv_dtype_t dt, size_t n, int32_t* dst) {
    size_t i;
    switch (dt) {
    case VV_DTYPE_I8:
        for (i = 0; i < n; i++) dst[i] = ((const int8_t*)src)[i];
        return true;
    case VV_DTYPE_I32:
        memcpy(dst, src, n * sizeof(int32_t));
        return true;
    case VV_DTYPE_I64:
        for (i = 0; i < n; i++) dst[i] = (int32_t)((const int64_t*)src)[i];
        return true;
    default:
        return false;
    }
}

/**
 * @brief A compressed-tensors projection: `e_int` (int-quantized `.weight`)
 *        or `e_packed` (pack-quantized `.weight_packed`), one of them set.
 */
static vv_status_t load_ct_weight(loader_t* L, const char* base,
                                  const st_entry_t* e_int,
                                  const st_entry_t* e_packed,
                                  int N, int K, vv_weight_t* w) {
    char buf[320];
    vv_status_t s;
    const vv_load_quant_t q = L->quant;

    /* How many bits the stored integers are. */
    int bits;
    if (e_packed) {
        if (e_packed->info.dtype != VV_DTYPE_I32 ||
            (s = check_shape(e_packed, N, (int64_t)K / 8)) != VV_OK) {
            VV_LOG_E("loader: '%s' is not the int32 [%d, %d] of a packed "
                     "4-bit [%d x %d] weight", e_packed->info.name, N, K / 8,
                     N, K);
            return VV_ERR_UNSUPPORTED;
        }
        bits = 4;
        snprintf(buf, sizeof(buf), "%s.weight_shape", base);
        const st_entry_t* sh = find(L, buf);
        int32_t dims[2] = { 0, 0 };
        if (sh && numel_of(&sh->info) == 2 && check_bytes(sh) == VV_OK) {
            const void* d = data_of(L, sh);
            if (d && ct_ints(d, sh->info.dtype, 2, dims) &&
                (dims[0] != N || dims[1] != K)) {
                VV_LOG_E("loader: '%s' says [%d, %d], the config [%d, %d]",
                         buf, dims[0], dims[1], N, K);
                return VV_ERR_SHAPE_MISMATCH;
            }
        }
    } else {
        if ((s = check_shape(e_int, N, K)) != VV_OK) return s;
        bits = L->ct.w_bits ? L->ct.w_bits : 8;
    }
    if (bits != 8 && bits != 4) {
        VV_LOG_E("loader: '%s' is %d-bit; the runtime runs 8- and 4-bit "
                 "compressed-tensors weights", base, bits);
        return VV_ERR_UNSUPPORTED;
    }
    const bool ok_mode = bits == 8
        ? (q == VV_LOAD_QUANT_AUTO || q == VV_LOAD_QUANT_INT8 ||
           q == VV_LOAD_QUANT_W8A8)
        : (q == VV_LOAD_QUANT_AUTO || q == VV_LOAD_QUANT_INT4 ||
           q == VV_LOAD_QUANT_W4A8);
    if (!ok_mode) {
        VV_LOG_E("loader: the checkpoint is %d-bit compressed-tensors; "
                 "--quant %s is not available for it (auto, %s)", bits,
                 vv_load_quant_name(q),
                 bits == 8 ? "int8 or w8a8" : "int4 or w4a8");
        return VV_ERR_UNSUPPORTED;
    }

    /* Scales: one per row (channel) or per group. */
    snprintf(buf, sizeof(buf), "%s.weight_scale", base);
    const st_entry_t* se = find(L, buf);
    if (!se || !is_float_dtype(se->info.dtype) ||
        (s = check_bytes(se)) != VV_OK) {
        VV_LOG_E("loader: '%s' is missing or not a float tensor", buf);
        return VV_ERR_WEIGHT_MISSING;
    }
    const int64_t ns = numel_of(&se->info);
    const int ng = (ns > 0 && ns % N == 0) ? (int)(ns / N) : 0;
    if (ng <= 0 || K % ng != 0) {
        VV_LOG_E("loader: '%s' has %lld scales, not a whole number of "
                 "groups per row of [%d x %d]", buf, (long long)ns, N, K);
        return VV_ERR_SHAPE_MISMATCH;
    }
    const int G = K / ng;
    if (bits == 8 && ng != 1) {
        VV_LOG_E("loader: '%s': 8-bit group-wise weights are not supported "
                 "(per-channel only)", base);
        return VV_ERR_UNSUPPORTED;
    }
    if (bits == 4 && (G % 32) != 0) {
        VV_LOG_E("loader: '%s': group size %d is not a multiple of 32",
                 base, G);
        return VV_ERR_UNSUPPORTED;
    }
    float* scale = (float*)vv_alloc((size_t)ns * sizeof(float));
    if (!scale) return VV_ERR_OUT_OF_MEMORY;
    to_f32(data_of(L, se), se->info.dtype, scale, (size_t)ns);

    /* Zero points (asymmetric only), as signed ints [N][ng]. */
    int32_t* zp = NULL;
    snprintf(buf, sizeof(buf), "%s.weight_zero_point", base);
    const st_entry_t* ze = find(L, buf);
    if (ze) {
        const void* zd = data_of(L, ze);
        const int64_t nz = numel_of(&ze->info);
        zp = (int32_t*)vv_alloc((size_t)N * ng * sizeof(int32_t));
        bool ok = zp && zd && check_bytes(ze) == VV_OK;
        if (ok && nz == (int64_t)N * ng) {
            ok = ct_ints(zd, ze->info.dtype, (size_t)nz, zp);
        } else if (ok && bits == 4 && ze->info.dtype == VV_DTYPE_I32 &&
                   ze->info.ndim == 2 && ze->info.shape[0] == (N + 7) / 8 &&
                   ze->info.shape[1] == ng) {
            /* Packed along N: word (r, g) holds rows 8r..8r+7, + 8. */
            const uint32_t* pz = (const uint32_t*)zd;
            for (int n = 0; n < N; n++)
                for (int g = 0; g < ng; g++)
                    zp[(size_t)n * ng + g] = (int32_t)(
                        (pz[(size_t)(n >> 3) * ng + g] >> (4 * (n & 7))) & 15u)
                        - 8;
        } else {
            ok = false;
        }
        if (!ok) {
            const vv_status_t es = zp ? VV_ERR_SHAPE_MISMATCH
                                      : VV_ERR_OUT_OF_MEMORY;
            VV_LOG_E("loader: '%s' is not a zero point the runtime reads", buf);
            vv_free(scale); vv_free(zp);
            return es;
        }
    }

    /* Act-order: only a g_idx that is plain k / G loads. */
    snprintf(buf, sizeof(buf), "%s.weight_g_idx", base);
    const st_entry_t* gi = find(L, buf);
    if (gi) {
        const void* gd = data_of(L, gi);
        bool plain = gd && numel_of(&gi->info) == K &&
                     check_bytes(gi) == VV_OK;
        int32_t* gv = plain ? (int32_t*)vv_alloc((size_t)K * sizeof(int32_t))
                            : NULL;
        plain = plain && gv && ct_ints(gd, gi->info.dtype, (size_t)K, gv);
        for (int k = 0; plain && k < K; k++) plain = gv[k] == k / G;
        vv_free(gv);
        if (!plain) {
            VV_LOG_E("loader: '%s' is an act-order (group) permutation, which "
                     "compressed-tensors loading does not take", buf);
            vv_free(scale); vv_free(zp);
            return VV_ERR_UNSUPPORTED;
        }
    }

    const void* src = data_of(L, e_packed ? e_packed : e_int);
    if (!src) { vv_free(scale); vv_free(zp); return VV_ERR_MODEL_FORMAT; }
    w->is_quantized = true;
    w->act_int8 = (q == VV_LOAD_QUANT_AUTO) ? L->ct.act_int8
                                            : vv_load_quant_act8(q);

    if (bits == 8) {
        /* INT8 is w = q * s: a zero point would be an offset it lacks. */
        for (int n = 0; zp && n < N; n++)
            if (zp[n] != 0) {
                VV_LOG_E("loader: '%s' is asymmetric int8; only symmetric "
                         "int8 weights load", base);
                vv_free(scale); vv_free(zp);
                return VV_ERR_UNSUPPORTED;
            }
        vv_free(zp);
        int8_t* codes = (int8_t*)vv_alloc((size_t)N * K);
        if (!codes) { vv_free(scale); return VV_ERR_OUT_OF_MEMORY; }
        memcpy(codes, src, (size_t)N * K);
        const int64_t qshape[2] = { N, K };
        const int64_t sshape[1] = { N };
        set_tensor(&w->tensor, codes, VV_DTYPE_I8, (size_t)N * K, 2, qshape);
        set_tensor(&w->quant.scales, scale, VV_DTYPE_F32,
                   (size_t)N * sizeof(float), 1, sshape);
        w->quant_kind = VV_QUANT_INT8;
        L->n_int8++;
        L->proj_bytes += (size_t)N * K + (size_t)N * sizeof(float);
        return VV_OK;
    }

    /* 4-bit: INT4G codes = q + 8 and zeros = z + 8, both 0..15. */
    const size_t packed_bytes = (size_t)N * (K / 2);
    const size_t group_bytes = (size_t)N * ng * sizeof(uint16_t);
    uint8_t*  packed = (uint8_t*)vv_alloc(packed_bytes);
    uint16_t* sc = (uint16_t*)vv_alloc(group_bytes);
    uint16_t* mn = (uint16_t*)vv_alloc(group_bytes);
    uint8_t*  zr = (uint8_t*)vv_alloc((size_t)N * ng);
    if (!packed || !sc || !mn || !zr) {
        vv_free(packed); vv_free(sc); vv_free(mn); vv_free(zr);
        vv_free(scale); vv_free(zp);
        return VV_ERR_OUT_OF_MEMORY;
    }
    int bad = 0;
    int n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:bad)
#endif
    for (n = 0; n < N; n++) {
        uint8_t* prow = packed + (size_t)n * (K / 2);
        for (int k = 0; k < K; k += 2) {
            int c0, c1;
            if (e_packed) {
                const uint32_t* wr = (const uint32_t*)src + (size_t)n * (K / 8);
                c0 = (int)((wr[k >> 3] >> (4 * (k & 7))) & 15u);
                c1 = (int)((wr[(k + 1) >> 3] >> (4 * ((k + 1) & 7))) & 15u);
            } else {
                const int8_t* wr = (const int8_t*)src + (size_t)n * K;
                c0 = wr[k] + 8;
                c1 = wr[k + 1] + 8;
                if (c0 < 0 || c0 > 15 || c1 < 0 || c1 > 15) {
                    bad++;
                    c0 = c0 < 0 ? 0 : (c0 > 15 ? 15 : c0);
                    c1 = c1 < 0 ? 0 : (c1 > 15 ? 15 : c1);
                }
            }
            prow[k >> 1] = (uint8_t)((c0 << 4) | c1);
        }
        for (int g = 0; g < ng; g++) {
            const size_t i = (size_t)n * ng + g;
            int z = 8 + (zp ? zp[i] : 0);
            if (z < 0 || z > 15) { bad++; z = z < 0 ? 0 : 15; }
            const uint16_t sh = vv_float_to_half_rne(scale[i]);
            sc[i] = sh;
            zr[i] = (uint8_t)z;
            mn[i] = vv_float_to_half_rne(-(float)z * vv_half_to_float(sh));
        }
    }
    vv_free(scale);
    vv_free(zp);
    if (bad) {
        VV_LOG_E("loader: '%s' holds %d values outside 4 bits", base, bad);
        vv_free(packed); vv_free(sc); vv_free(mn); vv_free(zr);
        return VV_ERR_MODEL_FORMAT;
    }
    const int64_t pshape[2] = { N, K / 2 };
    const int64_t gshape[2] = { N, ng };
    set_tensor(&w->tensor, packed, VV_DTYPE_U8, packed_bytes, 2, pshape);
    set_tensor(&w->quant.scales, sc, VV_DTYPE_F16, group_bytes, 2, gshape);
    set_tensor(&w->mins, mn, VV_DTYPE_F16, group_bytes, 2, gshape);
    set_tensor(&w->zeros, zr, VV_DTYPE_U8, (size_t)N * ng, 2, gshape);
    w->quant_kind = VV_QUANT_INT4G;
    w->group_size = G;
    w->int4g_layout = VV_INT4G_ROWMAJOR;
    L->n_int4g++;
    L->proj_bytes += packed_bytes + 2 * group_bytes;
    return VV_OK;
}

/**
 * @brief One projection of one layer, in whatever form the file holds it.
 */
static vv_status_t load_projection(loader_t* L, int layer, const char* which,
                                   int N, int K, bool want_bias,
                                   const float* col_mul, const float* row_div,
                                   vv_weight_t* w) {
    char base[256], buf[320];
    snprintf(base, sizeof(base), "%slayers.%d.%s", L->prefix, layer, which);
    strncpy(w->name, base, sizeof(w->name) - 1);
    vv_status_t s;

    snprintf(buf, sizeof(buf), "%s.weight", base);
    const st_entry_t* e = find(L, buf);
    /* compressed-tensors pack-quantized keeps no `.weight` at all. */
    char pbuf[320];
    snprintf(pbuf, sizeof(pbuf), "%s.weight_packed", base);
    const st_entry_t* ep = e ? NULL : find(L, pbuf);
    if (L->bitnet) {
        /*
         * BitNet's safetensors hold the latent F32 weights; the model runs
         * their ternarization (one scale per tensor, VibeASR.cpp's converter
         * formula), never the latent values themselves.
         */
        if (!e) {
            VV_LOG_E("loader: '%s' is missing", buf);
            return VV_ERR_WEIGHT_MISSING;
        }
        if (e->info.dtype != VV_DTYPE_F32) {
            VV_LOG_E("loader: '%s' is %s; BitNet's latent weights are F32",
                     buf, dtype_name(e->info.dtype));
            return VV_ERR_MODEL_FORMAT;
        }
        if ((s = check_shape(e, N, K)) != VV_OK) return s;
        if (K % VV_TERNARY_BLOCK) return VV_ERR_UNSUPPORTED;
        const float* src = (const float*)data_of(L, e);
        if (!src) return VV_ERR_MODEL_FORMAT;
        const size_t nb = (size_t)N * K / 4;
        uint8_t* codes = (uint8_t*)vv_alloc(nb);
        if (!codes) return VV_ERR_OUT_OF_MEMORY;
        float sc = 0.0f;
        s = vv_ternarize_f32(src, N, K, codes, &sc);
        if (s != VV_OK) { vv_free(codes); return s; }
        const int64_t shp[2] = { N, K / 4 };
        set_tensor(&w->tensor, codes, VV_DTYPE_U8, nb, 2, shp);
        w->tscale = sc;
        w->quant_kind = VV_QUANT_TERNARY;
        w->is_quantized = true;
        L->n_ternary++;
        L->n_converted++;
        L->proj_bytes += nb;
        snprintf(buf, sizeof(buf), "%s.bias", base);
        e = find(L, buf);
        if (e) {
            if ((s = check_shape(e, N, 0)) != VV_OK) return s;
            return load_f32(L, e, &w->bias);
        }
        if (want_bias) {
            VV_LOG_E("loader: '%s' is missing (attention_bias is on)", buf);
            return VV_ERR_WEIGHT_MISSING;
        }
        return VV_OK;
    }
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
        if ((s = load_dense_weight(L, e, N, K, col_mul, row_div, w)) != VV_OK)
            return s;
    } else if (e && e->info.dtype == VV_DTYPE_I8) {
        /* compressed-tensors int-quantized: W8A8, or 4-bit in an int8 box */
        if ((s = load_ct_weight(L, base, e, NULL, N, K, w)) != VV_OK) return s;
    } else if (e) {
        VV_LOG_E("loader: '%s' is %s, which no projection format uses",
                 buf, dtype_name(e->info.dtype));
        return VV_ERR_UNSUPPORTED;
    } else if (ep) {
        /* compressed-tensors pack-quantized: W4A16 or W4A8 */
        if ((s = load_ct_weight(L, base, NULL, ep, N, K, w)) != VV_OK) return s;
    } else {
        snprintf(buf, sizeof(buf), "%s.qweight", base);
        if (!find(L, buf)) {
            VV_LOG_E("loader: '%s' has neither .weight nor .qweight", base);
            return VV_ERR_WEIGHT_MISSING;
        }
        if (L->quant != VV_LOAD_QUANT_AUTO && L->quant != VV_LOAD_QUANT_INT4 &&
            L->quant != VV_LOAD_QUANT_W4A8) {
            VV_LOG_E("loader: the checkpoint is AWQ/GPTQ int4; --quant %s "
                     "is not available for it (auto, int4 or w4a8)",
                     vv_load_quant_name(L->quant));
            return VV_ERR_UNSUPPORTED;
        }
        if ((s = load_awq_weight(L, base, N, K, w)) != VV_OK) return s;
        /* W4A8 on AWQ/GPTQ is exact: the same codes and scales, with the
           activations quantized instead. Act-order gathers FP16 inputs,
           which int8 activations do not go through. */
        if (L->quant == VV_LOAD_QUANT_W4A8) {
            if (w->perm.data) {
                VV_LOG_E("loader: '%s' is act-order GPTQ; --quant w4a8 does "
                         "not take a column permutation", base);
                return VV_ERR_UNSUPPORTED;
            }
            w->act_int8 = true;
        }
        L->n_int4g++;
        L->proj_bytes += w->tensor.size_bytes + w->quant.scales.size_bytes
                       + w->mins.size_bytes;
    }

    snprintf(buf, sizeof(buf), "%s.bias", base);
    e = find(L, buf);
    if (e) {
        if ((s = check_shape(e, N, 0)) != VV_OK) return s;
        if ((s = load_f16(L, e, &w->bias)) != VV_OK) return s;
        if (row_div) div_f16(&w->bias, row_div);
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
 * @brief Whether a GPTQ checkpoint stores zero points off by one.
 *
 * The classic AutoGPTQ format writes zero - 1; GPTQModel's "gptq_v2" writes
 * the zero itself. The tensors look the same, so only config.json can tell.
 */
static int gptq_zero_bias(const char* model_dir) {
    char path[512];
    build_path(path, sizeof(path), model_dir, "config.json");
    FILE* f = fopen(path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return 1; }
    char* json = (char*)vv_alloc((size_t)size + 1);
    if (!json) { fclose(f); return 1; }
    const size_t got = fread(json, 1, (size_t)size, f);
    fclose(f);
    json[got] = '\0';

    int bias = 1;
    cJSON* root = cJSON_Parse(json);
    vv_free(json);
    if (!root) return 1;
    const cJSON* qc = cJSON_GetObjectItem(root, "quantization_config");
    const cJSON* fmt = qc ? cJSON_GetObjectItem(qc, "checkpoint_format") : NULL;
    if (cJSON_IsString(fmt) && fmt->valuestring &&
        strcmp(fmt->valuestring, "gptq_v2") == 0)
        bias = 0;
    cJSON_Delete(root);
    return bias;
}

/** @brief Read quantization_config from config.json (absent: all zero). */
static void ct_parse(const char* model_dir, ct_cfg_t* ct) {
    memset(ct, 0, sizeof(*ct));
    char path[512];
    build_path(path, sizeof(path), model_dir, "config.json");
    FILE* f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return; }
    char* json = (char*)vv_alloc((size_t)size + 1);
    if (!json) { fclose(f); return; }
    const size_t got = fread(json, 1, (size_t)size, f);
    fclose(f);
    json[got] = '\0';
    cJSON* root = cJSON_Parse(json);
    vv_free(json);
    if (!root) return;

    const cJSON* qc = cJSON_GetObjectItem(root, "quantization_config");
    const cJSON* qm = qc ? cJSON_GetObjectItem(qc, "quant_method") : NULL;
    if (cJSON_IsString(qm) && qm->valuestring &&
        strcmp(qm->valuestring, "compressed-tensors") == 0) {
        ct->present = true;
        const cJSON* fmt = cJSON_GetObjectItem(qc, "format");
        if (cJSON_IsString(fmt) && fmt->valuestring)
            snprintf(ct->format, sizeof(ct->format), "%s", fmt->valuestring);
        const cJSON* groups = cJSON_GetObjectItem(qc, "config_groups");
        /*
         * One scheme for every Linear: the loader applies it by weight dtype,
         * not by each group's `targets`, so groups that disagree on weight
         * bits or on int8 activations are flagged and refused.
         */
        const cJSON* g;
        int n_groups = 0;
        cJSON_ArrayForEach(g, groups) {
            const cJSON* wa = cJSON_GetObjectItem(g, "weights");
            const cJSON* nb = wa ? cJSON_GetObjectItem(wa, "num_bits") : NULL;
            const int bits = cJSON_IsNumber(nb) ? nb->valueint : 0;
            bool a8 = false;
            const cJSON* ia = cJSON_GetObjectItem(g, "input_activations");
            if (ia && !cJSON_IsNull(ia)) {
                const cJSON* ab = cJSON_GetObjectItem(ia, "num_bits");
                const cJSON* ty = cJSON_GetObjectItem(ia, "type");
                a8 = cJSON_IsNumber(ab) && ab->valueint == 8 &&
                     cJSON_IsString(ty) && ty->valuestring &&
                     strcmp(ty->valuestring, "int") == 0;
            }
            if (n_groups == 0) {
                ct->w_bits = bits;
                ct->act_int8 = a8;
            } else if (bits != ct->w_bits || a8 != ct->act_int8) {
                ct->mixed = true;
            }
            n_groups++;
        }
    }
    cJSON_Delete(root);
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

    /* SmoothQuant scratch: the factors of one layer and one column range. */
    float* sbuf = NULL;
    float* wmax = NULL;
    float lo = 1e30f, hi = 0.0f;
    if (L->smooth) {
        const size_t nf = (size_t)2 * hs + kvd + qd + inter;
        const int wide = inter > qd ? (inter > hs ? inter : hs)
                                    : (qd > hs ? qd : hs);
        sbuf = (float*)vv_alloc(nf * sizeof(float));
        wmax = (float*)vv_alloc((size_t)wide * sizeof(float));
        if (!sbuf || !wmax) {
            vv_free(sbuf);
            vv_free(wmax);
            return VV_ERR_OUT_OF_MEMORY;
        }
    }

    s = VV_OK;
    for (int i = 0; i < m->num_layers && s == VV_OK; i++) {
        vv_layer_weights_t* Ly = &m->layers[i];
        smooth_fac_t f = { NULL, NULL, NULL, NULL, NULL };
        if (sbuf) {
            f.qkv = sbuf;
            f.gu = f.qkv + hs;
            f.vo = f.gu + hs;
            f.o_cols = f.vo + kvd;
            f.ud = f.o_cols + qd;
            if ((s = smooth_layer(L, i, wmax, &f)) != VV_OK) break;
            smooth_range(f.qkv, hs, &lo, &hi);
            smooth_range(f.gu, hs, &lo, &hi);
            smooth_range(f.vo, kvd, &lo, &hi);
            smooth_range(f.ud, inter, &lo, &hi);
        }

        snprintf(buf, sizeof(buf), "%slayers.%d.input_layernorm.weight",
                 L->prefix, i);
        /* F16 norms, except BitNet's F32 ones (SmoothQuant is refused there:
         * it needs --quant, which BitNet refuses). */
        if ((s = need_norm(L, buf, hs, &Ly->input_layernorm)) != VV_OK)
            break;
        snprintf(buf, sizeof(buf), "%slayers.%d.post_attention_layernorm.weight",
                 L->prefix, i);
        if ((s = need_norm(L, buf, hs, &Ly->post_attn_layernorm)) != VV_OK)
            break;
        /* x / s out of the norms, W * s into the projections they feed. */
        if (f.qkv) div_f16(&Ly->input_layernorm, f.qkv);
        if (f.gu) div_f16(&Ly->post_attn_layernorm, f.gu);

        if ((s = load_projection(L, i, "self_attn.q_proj", qd, hs, qkv_bias,
                                 f.qkv, NULL, &Ly->attn.q_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.k_proj", kvd, hs, qkv_bias,
                                 f.qkv, NULL, &Ly->attn.k_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.v_proj", kvd, hs, qkv_bias,
                                 f.qkv, f.vo, &Ly->attn.v_proj)) != VV_OK ||
            (s = load_projection(L, i, "self_attn.o_proj", hs, qd, false,
                                 f.o_cols, NULL, &Ly->attn.o_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.gate_proj", inter, hs, false,
                                 f.gu, NULL, &Ly->mlp.gate_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.up_proj", inter, hs, false,
                                 f.gu, f.ud, &Ly->mlp.up_proj)) != VV_OK ||
            (s = load_projection(L, i, "mlp.down_proj", hs, inter, false,
                                 f.ud, NULL, &Ly->mlp.down_proj)) != VV_OK)
            break;
    }
    vv_free(sbuf);
    vv_free(wmax);
    if (s == VV_OK && L->smooth)
        VV_LOG_I("loader: SmoothQuant folded (alpha %.2f, maps%s%s%s%s), "
                 "factors %.3g..%.3g", (double)L->smooth_alpha,
                 (L->smooth_maps & VV_SMOOTH_QKV) ? " qkv" : "",
                 (L->smooth_maps & VV_SMOOTH_GATEUP) ? " gate/up" : "",
                 (L->smooth_maps & VV_SMOOTH_VO) ? " v->o" : "",
                 (L->smooth_maps & VV_SMOOTH_UPDOWN) ? " up->down" : "",
                 (double)lo, (double)hi);
    return s;
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
    if ((s = need_norm(L, buf, c->hidden_size, &m->final_norm)) != VV_OK)
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
        if (L->bitnet && L->head == VV_HEAD_INT8) {
            /* int8 rows of the F32 table itself, not of its FP16 copy */
            snprintf(buf, sizeof(buf), "%sembed_tokens.weight", L->prefix);
            const st_entry_t* e = find(L, buf);
            const void* src = e ? data_of(L, e) : NULL;
            if (!src || e->info.dtype != VV_DTYPE_F32) return VV_ERR_MODEL_FORMAT;
            return vv_bitnet_head_i8(m, src, 0, c->vocab_size, c->hidden_size);
        }
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

/* ─── BitNet's int8 encoder, quantized from the F32 safetensors ─────────── */

/* One layer: I8_S-quantize the whole F32 tensor (vv_i8s_quantize_f32, which
 * reproduces the shipped GGUF bit for bit), then lay it out as the int8
 * encoder reads it. */
static vv_status_t i8_layer_st(loader_t* L, vv_i8vae_t* v, const char* wname,
                               const char* bname, int stride, int depthwise,
                               vv_i8_layer_t* out) {
    const st_entry_t* ew = find(L, wname);
    const st_entry_t* eb = find(L, bname);
    if (!ew || !eb || ew->info.dtype != VV_DTYPE_F32 ||
        eb->info.dtype != VV_DTYPE_F32) {
        VV_LOG_E("loader: '%s' / '%s' missing or not F32", wname, bname);
        return VV_ERR_WEIGHT_MISSING;
    }
    vv_status_t s = check_bytes(ew);
    if (s == VV_OK) s = check_bytes(eb);
    if (s != VV_OK) return s;
    int o, in, k;
    if (ew->info.ndim == 3) {
        o = (int)ew->info.shape[0]; in = (int)ew->info.shape[1];
        k = (int)ew->info.shape[2];
    } else if (ew->info.ndim == 2) {
        o = (int)ew->info.shape[0]; in = (int)ew->info.shape[1]; k = 1;
    } else {
        return VV_ERR_MODEL_FORMAT;
    }
    if (numel_of(&eb->info) != o) return VV_ERR_SHAPE_MISMATCH;
    const int64_t n = (int64_t)o * in * k;
    int8_t* q = (int8_t*)vv_alloc((size_t)n);
    if (!q) return VV_ERR_OUT_OF_MEMORY;
    float sc = 0.0f;
    s = vv_i8s_quantize_f32((const float*)data_of(L, ew), 1, n, q, &sc);
    if (s == VV_OK)
        s = vv_i8vae_layer_from_q(v, q, sc, o, depthwise ? 1 : in, k, k,
                                  stride, depthwise,
                                  (const float*)data_of(L, eb), out);
    vv_free(q);
    return s;
}

static const float* f32_st(loader_t* L, const char* name, int64_t n) {
    const st_entry_t* e = find(L, name);
    if (!e || e->info.dtype != VV_DTYPE_F32 || numel_of(&e->info) != n ||
        check_bytes(e) != VV_OK) {
        VV_LOG_E("loader: '%s' missing, not F32 or not %lld long", name,
                 (long long)n);
        return NULL;
    }
    return (const float*)data_of(L, e);
}

static vv_status_t load_i8vae_st(loader_t* L) {
    vv_model_t* m = L->m;
    vv_i8vae_t* v = NULL;
    vv_status_t s = vv_i8vae_create(&v);
    for (int t = 0; t < 2 && s == VV_OK; t++) {
        const bool ac = t == 0;
        const char* p = ac ? "acoustic" : "semantic";
        const int* depth = ac ? m->config.acoustic.encoder_depths
                              : m->config.semantic.encoder_depths;
        const int ns = ac ? m->config.acoustic.n_depths : m->config.semantic.n_depths;
        vv_i8_tower_t* tw = &v->tower[t];
        s = vv_i8vae_tower_alloc(v, tw, ns, depth);
        char wn[256], bn[256];
        for (int i = 0; i < ns && s == VV_OK; i++) {
            snprintf(wn, sizeof(wn), "model.%s_tokenizer.encoder.downsample_layers.%d.0.conv.conv.weight", p, i);
            snprintf(bn, sizeof(bn), "model.%s_tokenizer.encoder.downsample_layers.%d.0.conv.conv.bias", p, i);
            s = i8_layer_st(L, v, wn, bn, vv_bitnet_vae_stride(&m->config, ac, i),
                            0, &tw->ds[i]);
            const int C = tw->ds[i].out_ch;
            for (int b = 0; b < depth[i] && s == VV_OK; b++) {
                vv_i8_block_t* B = &tw->blocks[i][b];
                char pre[160];
                snprintf(pre, sizeof(pre), "model.%s_tokenizer.encoder.stages.%d.%d.", p, i, b);
                snprintf(wn, sizeof(wn), "%smixer.conv.conv.conv.weight", pre);
                snprintf(bn, sizeof(bn), "%smixer.conv.conv.conv.bias", pre);
                s = i8_layer_st(L, v, wn, bn, 1, 1, &B->mixer);
                snprintf(wn, sizeof(wn), "%sffn.linear1.weight", pre);
                snprintf(bn, sizeof(bn), "%sffn.linear1.bias", pre);
                if (s == VV_OK) s = i8_layer_st(L, v, wn, bn, 1, 0, &B->fc1);
                snprintf(wn, sizeof(wn), "%sffn.linear2.weight", pre);
                snprintf(bn, sizeof(bn), "%sffn.linear2.bias", pre);
                if (s == VV_OK) s = i8_layer_st(L, v, wn, bn, 1, 0, &B->fc2);
                if (s != VV_OK) break;
                const char* part[4] = { "norm.weight", "gamma", "ffn_norm.weight", "ffn_gamma" };
                float** dst[4] = { &B->norm, &B->gamma, &B->ffn_norm, &B->ffn_gamma };
                for (int q = 0; q < 4 && s == VV_OK; q++) {
                    snprintf(wn, sizeof(wn), "%s%s", pre, part[q]);
                    const float* f = f32_st(L, wn, C);
                    if (!f) { s = VV_ERR_WEIGHT_MISSING; break; }
                    *dst[q] = vv_i8vae_copy_f32(v, f, (size_t)C);
                    if (!*dst[q]) s = VV_ERR_OUT_OF_MEMORY;
                }
            }
        }
        if (s != VV_OK) break;
        snprintf(wn, sizeof(wn), "model.%s_tokenizer.encoder.head.conv.conv.weight", p);
        snprintf(bn, sizeof(bn), "model.%s_tokenizer.encoder.head.conv.conv.bias", p);
        s = i8_layer_st(L, v, wn, bn, 1, 0, &tw->head);
        snprintf(wn, sizeof(wn), "model.%s_connector.fc1.weight", p);
        snprintf(bn, sizeof(bn), "model.%s_connector.fc1.bias", p);
        if (s == VV_OK) s = i8_layer_st(L, v, wn, bn, 1, 0, &tw->cfc1);
        snprintf(wn, sizeof(wn), "model.%s_connector.fc2.weight", p);
        snprintf(bn, sizeof(bn), "model.%s_connector.fc2.bias", p);
        if (s == VV_OK) s = i8_layer_st(L, v, wn, bn, 1, 0, &tw->cfc2);
        if (s == VV_OK) {
            snprintf(wn, sizeof(wn), "model.%s_connector.norm.weight", p);
            const float* f = f32_st(L, wn, tw->cfc1.out_ch);
            tw->cnorm = f ? vv_i8vae_copy_f32(v, f, (size_t)tw->cfc1.out_ch) : NULL;
            if (!tw->cnorm) s = VV_ERR_WEIGHT_MISSING;
        }
    }
    if (s == VV_OK) s = vv_i8vae_prepare(v);
    if (s == VV_OK) m->i8vae = v;
    else vv_i8vae_free(v);
    return s;
}

int vv_bitnet_default_vae(int cpu) {
    /*
     * CPU: the reference's int8 encoder. Its transcripts are VibeASR.cpp's
     * to the token, and it is also the faster one there (int8 GEMMs; the
     * FP32 encoder on the same weights takes about twice as long).
     * GPU: the float encoder on the dequantized I8_S weights, with GELU --
     * the model as trained, streamable and batched by the shared front end
     * in tens of milliseconds, where the int8 one would run on the host.
     * docs/BITNET.md has the measurements behind both.
     */
    return cpu ? VV_VAE_INT8 : VV_VAE_FLOAT;
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
        out[k++] = &p[i]->perm;
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

static vv_status_t model_load_impl(const char* model_dir,
                                   const vv_model_load_opts_t* opts,
                                   vv_model_t** out);

vv_status_t vv_model_load_ex(const char* model_dir,
                             const vv_model_load_opts_t* opts,
                             vv_model_t** out) {
#ifdef _OPENMP
    /*
     * The conversions and quantizers are bound by memory and by the FP
     * units a core's SMT siblings share. OpenMP's default of one thread per
     * logical CPU only adds barrier spinning: on the 5900X a BF16 --quant
     * int4 load took 15 s at 24 threads and 2.7 s at 12. So the load runs on
     * the physical cores unless OMP_NUM_THREADS says otherwise, and puts the
     * caller's setting back afterwards. Nothing is pinned: a GPU run's
     * threads should stay free to move.
     */
    const int prev = omp_get_max_threads();
    if (!getenv("OMP_NUM_THREADS")) {
        const int cores = vv_cpu_physical_cores();
        if (cores > 0 && cores < prev) omp_set_num_threads(cores);
    }
    const vv_status_t s = model_load_impl(model_dir, opts, out);
    omp_set_num_threads(prev);
    return s;
#else
    return model_load_impl(model_dir, opts, out);
#endif
}

static vv_status_t model_load_impl(const char* model_dir,
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

    /*
     * BitNet's own choices: where the weights come from, which encoder, which
     * head. Other families have one answer to each and refuse the others.
     */
    const bool bitnet = model->config.family == VV_FAMILY_ASR_BITNET;
    vv_model_load_opts_t bo = o;
    if (!bitnet) {
        if (o.source != VV_SOURCE_AUTO || o.vae == VV_VAE_INT8 ||
            o.head == VV_HEAD_INT8) {
            VV_LOG_E("loader: a weights source, the int8 encoder and the int8 "
                     "head exist for VibeVoice-ASR-BitNet only");
            vv_free(model);
            return VV_ERR_UNSUPPORTED;
        }
    } else {
        if (o.quant != VV_LOAD_QUANT_AUTO) {
            VV_LOG_E("loader: BitNet's projections are ternary; --quant %s "
                     "does not apply to them",
                     vv_load_quant_name((vv_load_quant_t)o.quant));
            vv_free(model);
            return VV_ERR_UNSUPPORTED;
        }
        if (o.head == VV_HEAD_INT8 && !o.cpu) {
            /* The GPU reads the F16 head at ~900 GB/s; the int8 rows only
             * pay off where decode is bound by host memory. */
            VV_LOG_E("loader: --head int8 is for the CPU path (--cpu)");
            vv_free(model);
            return VV_ERR_UNSUPPORTED;
        }
        if (bo.vae == VV_VAE_AUTO) bo.vae = vv_bitnet_default_vae(o.cpu);
        if (bo.source == VV_SOURCE_AUTO)
            bo.source = vv_model_has_bitnet_gguf(model_dir)
                      ? VV_SOURCE_GGUF : VV_SOURCE_SAFETENSORS;
        model->vae_numerics = bo.vae;
        model->weights_source = bo.source;
    }

    if (bitnet && bo.source == VV_SOURCE_GGUF) {
        const int nl = model->config.llm.num_hidden_layers;
        model->num_layers = nl;
        model->layers = (vv_layer_weights_t*)vv_alloc(
            (size_t)nl * sizeof(vv_layer_weights_t));
        if (!model->layers) { vv_free(model); return VV_ERR_OUT_OF_MEMORY; }
        memset(model->layers, 0, (size_t)nl * sizeof(vv_layer_weights_t));
        s = vv_model_load_bitnet_gguf(model, model_dir, &bo);
        if (s != VV_OK) {
            VV_LOG_E("loader: '%s' cannot be loaded from its GGUF pair: %s",
                     model_dir, vv_status_str(s));
            vv_model_free(model);
            return s;
        }
        if (o.cpu && bo.head == VV_HEAD_AUTO &&
            (s = vv_bitnet_head_filter_prepare(model)) != VV_OK) {
            vv_model_free(model);
            return s;
        }
        size_t pb = 0;
        for (int li = 0; li < nl; li++) pb += vv_layer_bytes(&model->layers[li]);
        VV_LOG_I("loader: asr-bitnet loaded from GGUF in %.0f ms (%d ternary "
                 "layers, %.1f MB; embedding Q6_K, head %s; encoder %s)",
                 vv_time_ms() - t0, nl, (double)pb / (1024.0 * 1024.0),
                 model->head_bound.data ? "F16 behind an exact int8 filter"
                 : model->head_i8.data ? "int8" : "F16",
                 vv_vae_numerics_name((vv_vae_numerics_t)bo.vae));
        *out = model;
        return VV_OK;
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
    L.gptq_bias = gptq_zero_bias(model_dir);
    if (o.smooth) {
        /*
         * Folding changes what the dense weights are before they are
         * quantized; kept dense, they would compute the same thing with
         * extra rounding. So it takes a load-time format and statistics of
         * this model's shape.
         */
        const vv_llm_config_t* c = &model->config.llm;
        const vv_smooth_stats_t* st = o.smooth;
        if (L.quant == VV_LOAD_QUANT_AUTO || L.quant == VV_LOAD_QUANT_NONE) {
            VV_LOG_E("loader: SmoothQuant folds into weights quantized at "
                     "load; pick --quant w8a8, w4a8, int8, int4 or nf4");
            vv_model_free(model);
            return VV_ERR_INVALID_ARG;
        }
        if (st->n_layers != n_layers || st->hidden != c->hidden_size ||
            st->q_dim != c->num_attention_heads * c->head_dim ||
            st->inter != c->intermediate_size) {
            VV_LOG_E("loader: the calibration is of a [%d layers, %d, %d, %d] "
                     "model, this one is [%d, %d, %d, %d]", st->n_layers,
                     st->hidden, st->q_dim, st->inter, n_layers,
                     c->hidden_size, c->num_attention_heads * c->head_dim,
                     c->intermediate_size);
            vv_model_free(model);
            return VV_ERR_SHAPE_MISMATCH;
        }
        L.smooth = st;
        L.smooth_alpha = o.smooth_alpha > 0.0f ? o.smooth_alpha
                                               : VV_SMOOTH_ALPHA_DEFAULT;
        L.smooth_maps = o.smooth_maps > 0 ? (o.smooth_maps & VV_SMOOTH_ALL)
                                          : VV_SMOOTH_MAPS_DEFAULT;
    }
    ct_parse(model_dir, &L.ct);
    if (L.ct.mixed) {
        VV_LOG_E("loader: this compressed-tensors checkpoint mixes schemes "
                 "(config_groups differ in weight bits or activations); "
                 "only one scheme for all Linear layers is supported");
        vv_model_free(model);
        return VV_ERR_UNSUPPORTED;
    }
    if (L.ct.present)
        VV_LOG_I("loader: compressed-tensors checkpoint (%s, %d-bit weights%s)",
                 L.ct.format[0] ? L.ct.format : "?", L.ct.w_bits,
                 L.ct.act_int8 ? ", int8 activations" : "");
    L.bitnet = bitnet;
    L.vae = bo.vae;
    L.head = bo.head;

    s = build_index(&L);
    if (s == VV_OK) s = find_prefix(&L);
    if (s == VV_OK) s = load_lm_globals(&L);
    if (s == VV_OK) s = load_layers(&L);
    if (s == VV_OK && bitnet && bo.vae == VV_VAE_INT8) {
        s = load_i8vae_st(&L);
    } else {
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
    }
    vv_free(L.ent);
    if (s == VV_OK && bitnet && o.cpu && bo.head == VV_HEAD_AUTO)
        s = vv_bitnet_head_filter_prepare(model);

    if (s != VV_OK) {
        VV_LOG_E("loader: '%s' cannot be loaded: %s", model_dir,
                 vv_status_str(s));
        vv_model_free(model);
        return s;
    }

    int n_bias = 0, n_a8 = 0;
    for (int li = 0; li < n_layers; li++) {
        vv_weight_t* p[7];
        vv_layer_projections(&model->layers[li], p);
        for (int k = 0; k < 7; k++) {
            if (p[k]->bias.data) n_bias++;
            if (p[k]->act_int8) n_a8++;
        }
    }

    VV_LOG_I("loader: %s model loaded in %.0f ms (%d layers under prefix "
             "'%s'; projections %d nf4 / %d int4g / %d int8 / %d ternary / "
             "%d fp16, %d quantized at load, %d on int8 activations, %.1f MB; "
             "head %s; encoders %d+%d tensors; %d biases)",
             vv_model_family_name(model->config.family),
             vv_time_ms() - t0, n_layers, L.prefix, L.n_nf4, L.n_int4g,
             L.n_int8, L.n_ternary, L.n_dense, L.n_converted, n_a8,
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
            {
                vv_weight_t* ws[7];
                const int nw = vv_layer_projections(&model->layers[i], ws);
                for (int wi = 0; wi < nw; wi++)
                    vv_tensor_free(&ws[wi]->zeros);
            }
        }
        vv_free(model->layers);
    }

    /* A tied head is the embedding table: freed once, as embed_tokens. */
    if (model->lm_head_tied) memset(&model->lm_head, 0, sizeof(model->lm_head));
    vv_tensor_free(&model->embed_q6k);
    vv_tensor_free(&model->head_i8);
    vv_tensor_free(&model->head_i8_scale);
    vv_tensor_free(&model->head_bound);
    vv_i8vae_free(model->i8vae);
    model->i8vae = NULL;
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
