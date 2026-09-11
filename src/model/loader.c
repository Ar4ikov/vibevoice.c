/**
 * @file loader.c
 * @brief Orchestrates loading all model components from safetensors + config.
 */

#include "vibevoice/model.h"
#include "vibevoice/quant.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/**
 * @brief Check if a tensor name is the main NF4 packed weight (not its metadata).
 *
 * NF4 quantized projections end with _proj.weight and are stored as U8.
 * Their companion tensors (.absmax, .quant_state, .quant_map, .nested_*)
 * are metadata loaded separately.
 */
static bool is_nf4_weight(const char* name) {
    if (strstr(name, ".self_attn.") && strstr(name, "_proj.weight") &&
        !strstr(name, ".absmax") && !strstr(name, ".quant_state") &&
        !strstr(name, ".quant_map") && !strstr(name, ".nested_"))
        return true;
    if (strstr(name, ".mlp.") && strstr(name, "_proj.weight") &&
        !strstr(name, ".absmax") && !strstr(name, ".quant_state") &&
        !strstr(name, ".quant_map") && !strstr(name, ".nested_"))
        return true;
    return false;
}

/**
 * @brief Check if a tensor is NF4 metadata (scale, quant_state, quant_map).
 * These are loaded alongside their parent weight, not independently.
 */
static bool is_nf4_metadata(const char* name) {
    return strstr(name, ".absmax") != NULL ||
           strstr(name, ".nested_absmax") != NULL ||
           strstr(name, ".quant_state") != NULL ||
           strstr(name, ".quant_map") != NULL ||
           strstr(name, ".nested_") != NULL ||
           strstr(name, "SCB") != NULL;
}

/** @brief AWQ / GPTQ store a projection as qweight + qzeros + scales. */
static bool is_awq_weight(const char* name) {
    return (strstr(name, ".self_attn.") || strstr(name, ".mlp.")) &&
           strstr(name, "_proj.qweight") != NULL;
}

/** @brief Companion tensors of an AWQ weight, loaded with their parent. */
static bool is_awq_metadata(const char* name) {
    return strstr(name, "_proj.qzeros") != NULL ||
           strstr(name, "_proj.scales") != NULL ||
           strstr(name, "_proj.g_idx") != NULL;
}

/**
 * @brief Build full path: dir/filename
 */
static void build_path(char* buf, size_t buf_size,
                        const char* dir, const char* filename) {
    size_t dlen = strlen(dir);
    if (dlen > 0 && (dir[dlen-1] == '/' || dir[dlen-1] == '\\')) {
        snprintf(buf, buf_size, "%s%s", dir, filename);
    } else {
        snprintf(buf, buf_size, "%s/%s", dir, filename);
    }
}

/**
 * @brief Parse model.safetensors.index.json to find safetensors file list.
 */
static vv_status_t find_safetensors_files(const char* model_dir,
                                           char*** out_files, int* out_count) {
    char path[512];
    build_path(path, sizeof(path), model_dir, "model.safetensors.index.json");

    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Try single-file model */
        build_path(path, sizeof(path), model_dir, "model.safetensors");
        f = fopen(path, "rb");
        if (f) {
            fclose(f);
            *out_files = (char**)vv_alloc(sizeof(char*));
            (*out_files)[0] = (char*)vv_alloc(strlen("model.safetensors") + 1);
            strcpy((*out_files)[0], "model.safetensors");
            *out_count = 1;
            return VV_OK;
        }
        VV_LOG_E("loader: no safetensors files found in '%s'", model_dir);
        return VV_ERR_NOT_FOUND;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)vv_alloc((size_t)size + 1);
    if (!json) { fclose(f); return VV_ERR_OUT_OF_MEMORY; }
    fread(json, 1, (size_t)size, f);
    fclose(f);
    json[size] = '\0';

    cJSON* root = cJSON_Parse(json);
    vv_free(json);
    if (!root) return VV_ERR_PARSE;

    /* Collect unique filenames from weight_map values */
    cJSON* weight_map = cJSON_GetObjectItem(root, "weight_map");
    if (!weight_map) {
        cJSON_Delete(root);
        return VV_ERR_PARSE;
    }

    /* First pass: count unique filenames */
    char unique_files[16][256];
    int n_unique = 0;

    cJSON* entry;
    cJSON_ArrayForEach(entry, weight_map) {
        if (!cJSON_IsString(entry)) continue;
        const char* fname = entry->valuestring;
        bool found = false;
        for (int i = 0; i < n_unique; i++) {
            if (strcmp(unique_files[i], fname) == 0) {
                found = true;
                break;
            }
        }
        if (!found && n_unique < 16) {
            strncpy(unique_files[n_unique], fname, 255);
            unique_files[n_unique][255] = '\0';
            n_unique++;
        }
    }

    cJSON_Delete(root);

    *out_files = (char**)vv_alloc((size_t)n_unique * sizeof(char*));
    for (int i = 0; i < n_unique; i++) {
        (*out_files)[i] = (char*)vv_alloc(strlen(unique_files[i]) + 1);
        strcpy((*out_files)[i], unique_files[i]);
    }
    *out_count = n_unique;

    VV_LOG_I("loader: found %d safetensors files", n_unique);
    return VV_OK;
}

/* ─── BF16 → FP16 in-place conversion ──────────────────────────────────── */

/**
 * @brief Convert a BF16 value to FP16.
 *
 * BF16 = 1 sign + 8 exponent + 7 mantissa  (same exponent as FP32)
 * FP16 = 1 sign + 5 exponent + 10 mantissa
 *
 * Strategy: BF16 → FP32 (just shift left 16) → FP16.
 */
static uint16_t bf16_to_fp16(uint16_t bf) {
    /* BF16 → FP32: upper 16 bits of IEEE-754 float */
    union { float f; uint32_t u; } u;
    u.u = (uint32_t)bf << 16;
    float f = u.f;

    /* FP32 → FP16 */
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (b >> 13) & 0x03FF;
    if (exp <= 0)       return (uint16_t)sign;            /* underflow → ±0   */
    if (exp >= 0x1F)    return (uint16_t)(sign | 0x7C00); /* overflow  → ±inf */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
}

/**
 * @brief Convert an array of BF16 values to FP16 in-place.
 */
static void bf16_array_to_fp16(uint16_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = bf16_to_fp16(data[i]);
    }
}

/* ─── FP16 tensor → FP32 in-place (for CPU-side weights) ────────────────── */

/**
 * @brief Convert a single FP16 (IEEE-754 half) value to FP32.
 */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;

    if (exp == 0) {
        /* Subnormal or zero */
        if (frac == 0) {
            union { float f; uint32_t u; } u;
            u.u = sign;
            return u.f;
        }
        /* Subnormal: normalize */
        exp = 1;
        while (!(frac & 0x0400)) { frac <<= 1; exp--; }
        frac &= 0x03FF;
        exp = (127 - 15 + exp);
        union { float f; uint32_t u; } u;
        u.u = sign | (exp << 23) | (frac << 13);
        return u.f;
    }
    if (exp == 0x1F) {
        /* Inf / NaN */
        union { float f; uint32_t u; } u;
        u.u = sign | 0x7F800000 | (frac << 13);
        return u.f;
    }
    /* Normal */
    union { float f; uint32_t u; } u;
    u.u = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    return u.f;
}

/**
 * @brief Convert a tensor from FP16 to FP32 in-place.
 *
 * Re-allocates the buffer to 4 bytes per element.
 * Used for connector / Conv-VAE weights that are consumed on CPU as float*.
 */
static void tensor_fp16_to_fp32(vv_tensor_t* t) {
    if (!t || !t->data) return;
    if (t->dtype != VV_DTYPE_F16) return;

    size_t n = t->size_bytes / 2;  /* number of FP16 elements */
    float* fp32 = (float*)vv_alloc(n * sizeof(float));
    if (!fp32) return;

    const uint16_t* fp16 = (const uint16_t*)t->data;
    for (size_t i = 0; i < n; i++) {
        fp32[i] = fp16_to_fp32(fp16[i]);
    }

    vv_free(t->data);
    t->data = fp32;
    t->size_bytes = n * sizeof(float);
    t->dtype = VV_DTYPE_F32;
}

/* ─── Load a single weight tensor ───────────────────────────────────────── */

static vv_status_t load_tensor_from_st(vv_safetensors_t* st,
                                        const char* name,
                                        vv_tensor_t* tensor) {
    vv_st_tensor_info_t info;
    vv_status_t s = vv_safetensors_find(st, name, &info);
    if (s != VV_OK) return s;

    const void* data;
    s = vv_safetensors_get_data(st, &info, &data);
    if (s != VV_OK) return s;

    /* Copy data to our managed memory */
    tensor->data = vv_alloc(info.data_size);
    if (!tensor->data) return VV_ERR_OUT_OF_MEMORY;
    memcpy(tensor->data, data, info.data_size);

    tensor->dtype = info.dtype;
    tensor->ndim = info.ndim;
    memcpy(tensor->shape, info.shape, sizeof(info.shape));
    tensor->size_bytes = info.data_size;
    tensor->on_gpu = false;

    /* Auto-convert BF16 → FP16 so all downstream code uses FP16.
     * Both are 2 bytes per element, so size_bytes stays the same. */
    if (tensor->dtype == VV_DTYPE_BF16) {
        size_t n_elements = tensor->size_bytes / 2;
        bf16_array_to_fp16((uint16_t*)tensor->data, n_elements);
        tensor->dtype = VV_DTYPE_F16;
    }

    return VV_OK;
}

/**
 * @brief Search all safetensors files for a tensor by name and load it.
 */
static vv_status_t load_tensor_any(vv_safetensors_t** st_files, int n_st,
                                    const char* name, vv_tensor_t* tensor) {
    for (int i = 0; i < n_st; i++) {
        vv_status_t s = load_tensor_from_st(st_files[i], name, tensor);
        if (s == VV_OK) return VV_OK;
    }
    return VV_ERR_NOT_FOUND;
}

/* FP16 → FP32 conversion helper */
static float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float r; uint32_t b = sign; memcpy(&r, &b, 4); return r; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF; exp++;
    } else if (exp == 31) {
        uint32_t b = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float r; memcpy(&r, &b, 4); return r;
    }
    exp = exp + (127 - 15);
    uint32_t b = sign | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
    float r; memcpy(&r, &b, 4); return r;
}

/* FP32 → FP16 conversion helper */
static uint16_t f32_to_half(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = b & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/**
 * @brief Load NF4 scales for a quantized weight.
 *
 * Handles bitsandbytes double quantization:
 * 1. Load "foo.weight.absmax" (U8 — quantized per-block scales)
 * 2. Load "foo.weight.nested_absmax" (F32 — per-superblock scale)
 * 3. Load "foo.weight.quant_state.bitsandbytes__nf4" (JSON blob)
 * 4. Parse JSON for nested_offset and nested_quant_map
 * 5. Reconstruct FP16 per-block scales on CPU
 */
static vv_status_t load_nf4_scales(vv_safetensors_t** st_files, int n_st,
                                    const char* weight_name,
                                    vv_weight_t* weight) {
    char buf[512];
    vv_status_t s;

    /* Load absmax */
    vv_tensor_t absmax_tensor = {0};
    snprintf(buf, sizeof(buf), "%s.absmax", weight_name);
    s = load_tensor_any(st_files, n_st, buf, &absmax_tensor);
    if (s != VV_OK) return s;

    /* Check if this is double quantization (absmax dtype == U8) */
    if (absmax_tensor.dtype != VV_DTYPE_U8) {
        /* Simple (non-double) quantization: absmax is already FP16/FP32 scales */
        weight->quant.scales = absmax_tensor;
        weight->quant.block_size = 64;
        weight->quant.double_quant = false;
        VV_LOG_D("loader: loaded NF4 scales for '%s' (simple)", weight_name);
        return VV_OK;
    }

    /* ── Double quantization path ── */
    weight->quant.double_quant = true;
    weight->quant.block_size = 64;

    /* Load nested_absmax (F32 per-superblock scales) */
    vv_tensor_t nested_tensor = {0};
    snprintf(buf, sizeof(buf), "%s.nested_absmax", weight_name);
    s = load_tensor_any(st_files, n_st, buf, &nested_tensor);
    if (s != VV_OK) {
        VV_LOG_E("loader: missing nested_absmax for '%s'", weight_name);
        vv_tensor_free(&absmax_tensor);
        return s;
    }

    /* Load quant_state blob (JSON) */
    vv_tensor_t qs_tensor = {0};
    snprintf(buf, sizeof(buf), "%s.quant_state.bitsandbytes__nf4", weight_name);
    s = load_tensor_any(st_files, n_st, buf, &qs_tensor);

    /* Parse quant_state JSON for nested_offset and nested_quant_map */
    float nested_offset = 0.0f;
    int nested_blocksize = 256;
    bool have_qmap = false;

    /*
     * Default bitsandbytes signed dynamic map: create_dynamic_map(signed=True).
     * 255 entries covering [-0.993, +0.993], symmetric around 0.
     * This is the default code used by bitsandbytes for blockwise quantization
     * of absmax values (centered by offset subtraction).
     * Index 255 padded with 1.0.
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
    const float* nested_qmap = BNB_DEFAULT_CODE;
    float custom_qmap[256];  /* mutable buffer for JSON-provided map */

    if (s == VV_OK && qs_tensor.data && qs_tensor.size_bytes > 0) {
        /* quant_state is a JSON string stored as U8 bytes */
        char* json_str = (char*)vv_alloc(qs_tensor.size_bytes + 1);
        if (json_str) {
            memcpy(json_str, qs_tensor.data, qs_tensor.size_bytes);
            json_str[qs_tensor.size_bytes] = '\0';

            cJSON* root = cJSON_Parse(json_str);
            if (root) {
                /* nested_offset */
                cJSON* off = cJSON_GetObjectItem(root, "nested_offset");
                if (off && cJSON_IsNumber(off))
                    nested_offset = (float)off->valuedouble;

                /* nested_blocksize */
                cJSON* nbs = cJSON_GetObjectItem(root, "nested_blocksize");
                if (nbs && cJSON_IsNumber(nbs))
                    nested_blocksize = (int)nbs->valuedouble;

                /* nested_quant_map — 256-entry lookup table (optional) */
                cJSON* nqm = cJSON_GetObjectItem(root, "nested_quant_map");
                if (nqm && cJSON_IsArray(nqm)) {
                    int nqm_size = cJSON_GetArraySize(nqm);
                    if (nqm_size >= 255 && nqm_size <= 256) {
                        for (int i = 0; i < nqm_size; i++) {
                            cJSON* v = cJSON_GetArrayItem(nqm, i);
                            custom_qmap[i] = v ? (float)v->valuedouble : 0.0f;
                        }
                        if (nqm_size == 255)
                            custom_qmap[255] = 1.0f;  /* pad to 256 */
                        nested_qmap = custom_qmap;
                        have_qmap = true;
                    }
                }
                cJSON_Delete(root);
            }
            vv_free(json_str);
        }
    }
    vv_tensor_free(&qs_tensor);

    /* Reconstruct FP16 per-block scales */
    int n_blocks = (int)absmax_tensor.size_bytes;  /* 1 byte per block (U8) */
    const uint8_t* abs_u8 = (const uint8_t*)absmax_tensor.data;
    const float* nested_f32 = (const float*)nested_tensor.data;
    int n_superblocks = (int)(nested_tensor.size_bytes / sizeof(float));

    size_t scales_bytes = (size_t)n_blocks * sizeof(uint16_t);
    uint16_t* scales_fp16 = (uint16_t*)vv_alloc(scales_bytes);
    if (!scales_fp16) {
        vv_tensor_free(&absmax_tensor);
        vv_tensor_free(&nested_tensor);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < n_blocks; i++) {
        int si = i / nested_blocksize;
        if (si >= n_superblocks) si = n_superblocks - 1;

        /* Dequantize: code[u8_val] * nested_absmax + offset */
        float scale = nested_qmap[abs_u8[i]] * nested_f32[si] + nested_offset;
        scales_fp16[i] = f32_to_half(scale);
    }

    /* Store as the weight's FP16 scales */
    weight->quant.scales.data = scales_fp16;
    weight->quant.scales.size_bytes = scales_bytes;
    weight->quant.scales.dtype = VV_DTYPE_F16;
    weight->quant.scales.ndim = 1;
    weight->quant.scales.shape[0] = n_blocks;
    weight->quant.scales.on_gpu = false;

    vv_tensor_free(&absmax_tensor);
    vv_tensor_free(&nested_tensor);

    VV_LOG_D("loader: loaded NF4 scales for '%s' (double-quant, %d blocks, "
             "offset=%.4f, sb=%d, qmap=%s)",
             weight_name, n_blocks, nested_offset, nested_blocksize,
             have_qmap ? "json-parsed" : "bnb-default");
    return VV_OK;
}

/**
 * @brief Load an AWQ / GPTQ projection and repack it into the INT4G layout.
 *
 * `weight_name` ends in ".qweight"; the zeros and scales sit next to it. The
 * repack is what makes the GEMV coalesce — see src/quant/awq_repack.c.
 */
static vv_status_t load_awq_weight(vv_safetensors_t** st_files, int n_st,
                                   const char* weight_name, vv_weight_t* w)
{
    char base[256];
    size_t blen = strlen(weight_name) - strlen(".qweight");
    if (blen >= sizeof(base)) return VV_ERR_OVERFLOW;
    memcpy(base, weight_name, blen);
    base[blen] = '\0';

    char buf[300];
    vv_tensor_t qweight = {0}, qzeros = {0}, scales = {0};
    vv_status_t s;

    s = load_tensor_any(st_files, n_st, weight_name, &qweight);
    if (s != VV_OK) return s;
    snprintf(buf, sizeof(buf), "%s.qzeros", base);
    s = load_tensor_any(st_files, n_st, buf, &qzeros);
    if (s != VV_OK) { vv_tensor_free(&qweight); return s; }
    snprintf(buf, sizeof(buf), "%s.scales", base);
    s = load_tensor_any(st_files, n_st, buf, &scales);
    if (s != VV_OK) {
        vv_tensor_free(&qweight); vv_tensor_free(&qzeros);
        return s;
    }

    /* qweight is [K, N/8] int32; scales is [K/G, N] fp16. */
    const int K = (int)qweight.shape[0];
    const int N = (int)qweight.shape[1] * 8;
    const int n_groups = (int)scales.shape[0];
    if (K <= 0 || N <= 0 || n_groups <= 0 || (K % n_groups) != 0) {
        VV_LOG_E("loader: '%s' has inconsistent AWQ shapes", weight_name);
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
    vv_tensor_t g_idx = {0};
    const int zero_bias = (load_tensor_any(st_files, n_st, buf, &g_idx) == VV_OK)
                          ? 1 : 0;
    vv_tensor_free(&g_idx);

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
    strncpy(w->name, base, sizeof(w->name) - 1);

    w->tensor.data = packed;
    w->tensor.size_bytes = packed_bytes;
    w->tensor.dtype = VV_DTYPE_U8;
    w->tensor.ndim = 2;
    w->tensor.shape[0] = N;
    w->tensor.shape[1] = K / 2;

    w->quant.scales.data = sc;
    w->quant.scales.size_bytes = group_bytes;
    w->quant.scales.dtype = VV_DTYPE_F16;
    w->quant.scales.ndim = 2;
    w->quant.scales.shape[0] = N;
    w->quant.scales.shape[1] = n_groups;

    w->mins = w->quant.scales;
    w->mins.data = mn;

    VV_LOG_D("loader: AWQ '%s' N=%d K=%d group=%d%s",
             base, N, K, group_size, zero_bias ? " (gptq zeros)" : "");
    return VV_OK;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

vv_status_t vv_model_load(const char* model_dir, vv_model_t** out) {
    if (!model_dir || !out) return VV_ERR_NULL_PTR;

    VV_LOG_I("loader: loading model from '%s'", model_dir);

    /* Allocate model struct */
    vv_model_t* model = (vv_model_t*)vv_alloc(sizeof(vv_model_t));
    if (!model) return VV_ERR_OUT_OF_MEMORY;
    memset(model, 0, sizeof(*model));

    /* Parse config.json */
    char config_path[512];
    build_path(config_path, sizeof(config_path), model_dir, "config.json");
    vv_status_t s = vv_config_parse(config_path, &model->config);
    if (s != VV_OK) {
        VV_LOG_E("loader: failed to parse config.json");
        vv_free(model);
        return s;
    }

    /* Find and open safetensors files */
    char** st_names = NULL;
    int n_st = 0;
    s = find_safetensors_files(model_dir, &st_names, &n_st);
    if (s != VV_OK) {
        vv_free(model);
        return s;
    }

    model->st_files = (vv_safetensors_t**)vv_alloc(
        (size_t)n_st * sizeof(vv_safetensors_t*));
    model->n_st_files = n_st;

    for (int i = 0; i < n_st; i++) {
        char full_path[512];
        build_path(full_path, sizeof(full_path), model_dir, st_names[i]);
        s = vv_safetensors_open(full_path, &model->st_files[i]);
        if (s != VV_OK) {
            VV_LOG_E("loader: failed to open '%s'", full_path);
            /* Clean up opened files */
            for (int j = 0; j < i; j++) {
                vv_safetensors_close(model->st_files[j]);
            }
            for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
            vv_free(st_names);
            vv_free(model->st_files);
            vv_free(model);
            return s;
        }
    }

    for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
    vv_free(st_names);

    /* Allocate LLM layer weights */
    int n_layers = model->config.llm.num_hidden_layers;
    model->num_layers = n_layers;
    model->layers = (vv_layer_weights_t*)vv_alloc(
        (size_t)n_layers * sizeof(vv_layer_weights_t));
    if (!model->layers) {
        vv_model_free(model);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(model->layers, 0, (size_t)n_layers * sizeof(vv_layer_weights_t));

    /*
     * Count tokenizer encoder tensors for pre-allocation.
     */
    int n_acoustic_enc = 0, n_semantic_enc = 0;
    for (int fi = 0; fi < n_st; fi++) {
        int nt = vv_safetensors_num_tensors(model->st_files[fi]);
        for (int ti = 0; ti < nt; ti++) {
            vv_st_tensor_info_t info;
            vv_safetensors_get_info(model->st_files[fi], ti, &info);
            if (strstr(info.name, "acoustic_tokenizer.encoder."))
                n_acoustic_enc++;
            if (strstr(info.name, "semantic_tokenizer.encoder."))
                n_semantic_enc++;
        }
    }

    if (n_acoustic_enc > 0) {
        model->acoustic_weights = (vv_weight_t*)vv_alloc(
            (size_t)n_acoustic_enc * sizeof(vv_weight_t));
        if (model->acoustic_weights)
            memset(model->acoustic_weights, 0,
                   (size_t)n_acoustic_enc * sizeof(vv_weight_t));
    }
    if (n_semantic_enc > 0) {
        model->semantic_weights = (vv_weight_t*)vv_alloc(
            (size_t)n_semantic_enc * sizeof(vv_weight_t));
        if (model->semantic_weights)
            memset(model->semantic_weights, 0,
                   (size_t)n_semantic_enc * sizeof(vv_weight_t));
    }
    int ai = 0, si = 0;  /* Indices for encoder weight arrays */

    /*
     * Load tensors from safetensors.
     * Iterate all tensors in all files and route to the right destination.
     */
    for (int fi = 0; fi < n_st; fi++) {
        vv_safetensors_t* st = model->st_files[fi];
        int nt = vv_safetensors_num_tensors(st);

        for (int ti = 0; ti < nt; ti++) {
            vv_st_tensor_info_t info;
            vv_safetensors_get_info(st, ti, &info);

            /* Skip quantization metadata — loaded with its parent weight */
            if (is_nf4_metadata(info.name)) continue;
            if (is_awq_metadata(info.name)) continue;

            /* ── LLM global weights ── */
            if (strstr(info.name, "embed_tokens.weight")) {
                load_tensor_from_st(st, info.name, &model->embed_tokens);
            }
            else if (strstr(info.name, "language_model.norm.weight") ||
                     (strcmp(info.name, "model.norm.weight") == 0)) {
                load_tensor_from_st(st, info.name, &model->final_norm);
            }
            else if (strcmp(info.name, "lm_head.weight") == 0) {
                load_tensor_from_st(st, info.name, &model->lm_head);
            }

            /* ── Connector weights (BF16→FP16→FP32 for CPU use) ── */
            else if (strstr(info.name, "acoustic_connector.fc1.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc1.tensor);
                tensor_fp16_to_fp32(&model->acoustic_connector_fc1.tensor);
                strncpy(model->acoustic_connector_fc1.name, info.name, 255);
            }
            else if (strstr(info.name, "acoustic_connector.fc1.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc1.quant.packed);
                tensor_fp16_to_fp32(&model->acoustic_connector_fc1.quant.packed);
            }
            else if (strstr(info.name, "acoustic_connector.norm.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_norm.tensor);
                tensor_fp16_to_fp32(&model->acoustic_connector_norm.tensor);
            }
            else if (strstr(info.name, "acoustic_connector.fc2.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc2.tensor);
                tensor_fp16_to_fp32(&model->acoustic_connector_fc2.tensor);
            }
            else if (strstr(info.name, "acoustic_connector.fc2.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc2.quant.packed);
                tensor_fp16_to_fp32(&model->acoustic_connector_fc2.quant.packed);
            }
            else if (strstr(info.name, "semantic_connector.fc1.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc1.tensor);
                tensor_fp16_to_fp32(&model->semantic_connector_fc1.tensor);
                strncpy(model->semantic_connector_fc1.name, info.name, 255);
            }
            else if (strstr(info.name, "semantic_connector.fc1.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc1.quant.packed);
                tensor_fp16_to_fp32(&model->semantic_connector_fc1.quant.packed);
            }
            else if (strstr(info.name, "semantic_connector.norm.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_norm.tensor);
                tensor_fp16_to_fp32(&model->semantic_connector_norm.tensor);
            }
            else if (strstr(info.name, "semantic_connector.fc2.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc2.tensor);
                tensor_fp16_to_fp32(&model->semantic_connector_fc2.tensor);
            }
            else if (strstr(info.name, "semantic_connector.fc2.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc2.quant.packed);
                tensor_fp16_to_fp32(&model->semantic_connector_fc2.quant.packed);
            }

            /* ── Tokenizer encoder weights (BF16→FP16→FP32 for CPU use) ── */
            else if (strstr(info.name, "acoustic_tokenizer.encoder.") &&
                     model->acoustic_weights && ai < n_acoustic_enc) {
                load_tensor_from_st(st, info.name,
                                    &model->acoustic_weights[ai].tensor);
                tensor_fp16_to_fp32(&model->acoustic_weights[ai].tensor);
                strncpy(model->acoustic_weights[ai].name, info.name, 255);
                ai++;
            }
            else if (strstr(info.name, "semantic_tokenizer.encoder.") &&
                     model->semantic_weights && si < n_semantic_enc) {
                load_tensor_from_st(st, info.name,
                                    &model->semantic_weights[si].tensor);
                tensor_fp16_to_fp32(&model->semantic_weights[si].tensor);
                strncpy(model->semantic_weights[si].name, info.name, 255);
                si++;
            }

            /* ── LLM layer weights ── */
            else if (strstr(info.name, ".layers.")) {
                /* Parse layer index — works for both model.layers.N
                 * and model.language_model.layers.N */
                const char* lp = strstr(info.name, ".layers.");
                if (!lp) continue;
                lp += strlen(".layers.");
                int layer_idx = atoi(lp);
                if (layer_idx < 0 || layer_idx >= n_layers) continue;

                vv_layer_weights_t* layer = &model->layers[layer_idx];

                if (strstr(info.name, "input_layernorm.weight")) {
                    load_tensor_from_st(st, info.name,
                                        &layer->input_layernorm);
                }
                else if (strstr(info.name, "post_attention_layernorm.weight")) {
                    load_tensor_from_st(st, info.name,
                                        &layer->post_attn_layernorm);
                }
                /* ── Attention / MLP bias (FP16) ── */
                else if (strstr(info.name, "_proj.bias")) {
                    vv_tensor_t* dst = NULL;
                    if (strstr(info.name, "self_attn.q_proj.bias"))
                        dst = &layer->attn.q_proj.bias;
                    else if (strstr(info.name, "self_attn.k_proj.bias"))
                        dst = &layer->attn.k_proj.bias;
                    else if (strstr(info.name, "self_attn.v_proj.bias"))
                        dst = &layer->attn.v_proj.bias;
                    else if (strstr(info.name, "self_attn.o_proj.bias"))
                        dst = &layer->attn.o_proj.bias;
                    else if (strstr(info.name, "mlp.gate_proj.bias"))
                        dst = &layer->mlp.gate_proj.bias;
                    else if (strstr(info.name, "mlp.up_proj.bias"))
                        dst = &layer->mlp.up_proj.bias;
                    else if (strstr(info.name, "mlp.down_proj.bias"))
                        dst = &layer->mlp.down_proj.bias;

                    if (dst) {
                        load_tensor_from_st(st, info.name, dst);
                    }
                }
                else if (is_nf4_weight(info.name) || is_awq_weight(info.name)) {
                    /* Determine which projection this is */
                    const bool awq = is_awq_weight(info.name);
                    vv_weight_t* w = NULL;
                    if (strstr(info.name, "self_attn.q_proj."))
                        w = &layer->attn.q_proj;
                    else if (strstr(info.name, "self_attn.k_proj."))
                        w = &layer->attn.k_proj;
                    else if (strstr(info.name, "self_attn.v_proj."))
                        w = &layer->attn.v_proj;
                    else if (strstr(info.name, "self_attn.o_proj."))
                        w = &layer->attn.o_proj;
                    else if (strstr(info.name, "mlp.gate_proj."))
                        w = &layer->mlp.gate_proj;
                    else if (strstr(info.name, "mlp.up_proj."))
                        w = &layer->mlp.up_proj;
                    else if (strstr(info.name, "mlp.down_proj."))
                        w = &layer->mlp.down_proj;

                    if (w && awq) {
                        load_awq_weight(model->st_files, n_st, info.name, w);
                    } else if (w) {
                        w->is_quantized = true;
                        w->quant_kind = VV_QUANT_NF4;
                        load_tensor_from_st(st, info.name, &w->tensor);
                        strncpy(w->name, info.name, 255);
                        /* Load NF4 absmax scales */
                        load_nf4_scales(model->st_files, n_st,
                                        info.name, w);
                    }
                }
            }
        }
    }

    model->n_acoustic_weights = ai;
    model->n_semantic_weights = si;

    /* Count bias tensors loaded */
    int n_bias = 0;
    for (int li = 0; li < n_layers; li++) {
        vv_layer_weights_t* L = &model->layers[li];
        if (L->attn.q_proj.bias.data) n_bias++;
        if (L->attn.k_proj.bias.data) n_bias++;
        if (L->attn.v_proj.bias.data) n_bias++;
        if (L->attn.o_proj.bias.data) n_bias++;
    }

    VV_LOG_I("loader: model loaded successfully (%d layers, embed=%s, norm=%s, "
             "connectors=%s, encoders=%d+%d, attn_bias=%d)",
             n_layers,
             model->embed_tokens.data ? "yes" : "no",
             model->final_norm.data ? "yes" : "no",
             model->acoustic_connector_fc1.tensor.data ? "yes" : "no",
             ai, si, n_bias);

    *out = model;
    return VV_OK;
}

vv_status_t vv_model_free(vv_model_t* model) {
    if (!model) return VV_ERR_NULL_PTR;

    /* Free layer weights (including NF4 scales and biases) */
    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            vv_layer_weights_t* l = &model->layers[i];
            vv_tensor_free(&l->input_layernorm);
            vv_tensor_free(&l->post_attn_layernorm);
            vv_tensor_free(&l->attn.q_proj.tensor);
            vv_tensor_free(&l->attn.q_proj.bias);
            vv_tensor_free(&l->attn.q_proj.quant.scales);
            vv_tensor_free(&l->attn.q_proj.mins);
            vv_tensor_free(&l->attn.k_proj.tensor);
            vv_tensor_free(&l->attn.k_proj.bias);
            vv_tensor_free(&l->attn.k_proj.quant.scales);
            vv_tensor_free(&l->attn.k_proj.mins);
            vv_tensor_free(&l->attn.v_proj.tensor);
            vv_tensor_free(&l->attn.v_proj.bias);
            vv_tensor_free(&l->attn.v_proj.quant.scales);
            vv_tensor_free(&l->attn.v_proj.mins);
            vv_tensor_free(&l->attn.o_proj.tensor);
            vv_tensor_free(&l->attn.o_proj.bias);
            vv_tensor_free(&l->attn.o_proj.quant.scales);
            vv_tensor_free(&l->attn.o_proj.mins);
            vv_tensor_free(&l->mlp.gate_proj.tensor);
            vv_tensor_free(&l->mlp.gate_proj.bias);
            vv_tensor_free(&l->mlp.gate_proj.quant.scales);
            vv_tensor_free(&l->mlp.gate_proj.mins);
            vv_tensor_free(&l->mlp.up_proj.tensor);
            vv_tensor_free(&l->mlp.up_proj.bias);
            vv_tensor_free(&l->mlp.up_proj.quant.scales);
            vv_tensor_free(&l->mlp.up_proj.mins);
            vv_tensor_free(&l->mlp.down_proj.tensor);
            vv_tensor_free(&l->mlp.down_proj.bias);
            vv_tensor_free(&l->mlp.down_proj.quant.scales);
            vv_tensor_free(&l->mlp.down_proj.mins);
        }
        vv_free(model->layers);
    }

    /* Free LLM global weights */
    vv_tensor_free(&model->embed_tokens);
    vv_tensor_free(&model->final_norm);
    vv_tensor_free(&model->lm_head);

    /* Free connector weights */
    vv_tensor_free(&model->acoustic_connector_fc1.tensor);
    vv_tensor_free(&model->acoustic_connector_fc1.quant.packed); /* bias */
    vv_tensor_free(&model->acoustic_connector_norm.tensor);
    vv_tensor_free(&model->acoustic_connector_fc2.tensor);
    vv_tensor_free(&model->acoustic_connector_fc2.quant.packed); /* bias */
    vv_tensor_free(&model->semantic_connector_fc1.tensor);
    vv_tensor_free(&model->semantic_connector_fc1.quant.packed); /* bias */
    vv_tensor_free(&model->semantic_connector_norm.tensor);
    vv_tensor_free(&model->semantic_connector_fc2.tensor);
    vv_tensor_free(&model->semantic_connector_fc2.quant.packed); /* bias */

    /* Free tokenizer encoder weights */
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

    /* Close safetensors files */
    if (model->st_files) {
        for (int i = 0; i < model->n_st_files; i++) {
            if (model->st_files[i]) {
                vv_safetensors_close(model->st_files[i]);
            }
        }
        vv_free(model->st_files);
    }

    vv_free(model);
    return VV_OK;
}
