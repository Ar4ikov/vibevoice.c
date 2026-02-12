/**
 * @file sampling.c
 * @brief Token sampling strategies: greedy, top-k.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

/* BF16/FP16 to float conversion (CPU side) */
static float half_to_float_approx(uint16_t h) {
    /* Simplified FP16 to FP32 conversion */
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        /* Subnormal or zero */
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            memcpy(&result, &bits, 4);
            return result;
        }
        /* Subnormal: normalize */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp++;
    } else if (exp == 31) {
        /* Inf/NaN */
        uint32_t bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float result;
        memcpy(&result, &bits, 4);
        return result;
    }

    exp = exp + (127 - 15);
    uint32_t bits = sign | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                              int32_t* token_id) {
    if (!logits_fp16 || !token_id) return VV_ERR_NULL_PTR;

    const uint16_t* logits = (const uint16_t*)logits_fp16;
    float max_val = -FLT_MAX;
    int32_t max_idx = 0;

    for (int i = 0; i < vocab_size; i++) {
        float v = half_to_float_approx(logits[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    *token_id = max_idx;
    return VV_OK;
}

vv_status_t vv_sample_topk(const void* logits_fp16, int vocab_size,
                             int k, float temperature, int32_t* token_id) {
    if (!logits_fp16 || !token_id) return VV_ERR_NULL_PTR;
    if (k <= 0) k = 1;
    if (temperature <= 0.0f) {
        return vv_sample_greedy(logits_fp16, vocab_size, token_id);
    }

    const uint16_t* logits = (const uint16_t*)logits_fp16;

    /* Convert logits to float and find top-k */
    /* Simple partial sort: maintain top-k values */
    int actual_k = k < vocab_size ? k : vocab_size;
    float* top_vals = (float*)vv_alloc((size_t)actual_k * sizeof(float));
    int32_t* top_ids = (int32_t*)vv_alloc((size_t)actual_k * sizeof(int32_t));
    if (!top_vals || !top_ids) {
        if (top_vals) vv_free(top_vals);
        if (top_ids) vv_free(top_ids);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < actual_k; i++) {
        top_vals[i] = -FLT_MAX;
        top_ids[i] = 0;
    }

    for (int i = 0; i < vocab_size; i++) {
        float v = half_to_float_approx(logits[i]);
        /* Check if this value should be in top-k */
        if (v > top_vals[actual_k - 1]) {
            /* Insert into sorted position */
            int pos = actual_k - 1;
            while (pos > 0 && v > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                top_ids[pos] = top_ids[pos - 1];
                pos--;
            }
            top_vals[pos] = v;
            top_ids[pos] = i;
        }
    }

    /* Apply temperature and softmax */
    float max_val = top_vals[0];
    float sum_exp = 0.0f;
    for (int i = 0; i < actual_k; i++) {
        top_vals[i] = expf((top_vals[i] - max_val) / temperature);
        sum_exp += top_vals[i];
    }

    /* Normalize */
    for (int i = 0; i < actual_k; i++) {
        top_vals[i] /= sum_exp;
    }

    /* Sample from distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int32_t selected = top_ids[0];

    for (int i = 0; i < actual_k; i++) {
        cumsum += top_vals[i];
        if (r <= cumsum) {
            selected = top_ids[i];
            break;
        }
    }

    vv_free(top_vals);
    vv_free(top_ids);

    *token_id = selected;
    return VV_OK;
}
