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

/* ─── Sampling from FP32 logits on the host ─────────────────────────────── */

/**
 * @brief Draw one token the way an OpenAI client asks for it.
 *
 * `temperature` <= 0 is greedy and ignores everything else. Otherwise the
 * `top_k` most likely tokens are taken first and nucleus (`top_p`) is applied
 * inside that set: a full sort of 152k logits per token would cost more than
 * the decode step it follows, and past the first few dozen candidates the
 * tail carries no mass worth keeping. `rng` is xorshift64*, so a seed
 * reproduces an answer exactly.
 */
vv_status_t vv_sample_logits_f32(const float* logits, int vocab_size,
                                 float temperature, float top_p, int top_k,
                                 uint64_t* rng, int32_t* token_id) {
    if (!logits || !token_id || vocab_size <= 0) return VV_ERR_NULL_PTR;

    if (!(temperature > 0.0f)) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        *token_id = best;
        return VV_OK;
    }

    int k = top_k > 0 ? top_k : 64;
    if (k > vocab_size) k = vocab_size;
    if (k > 512) k = 512;               /* the tail beyond this is noise */

    /*
     * Selection into a small array kept ascending, so vals[0] is the one to
     * beat and vals[n-1] is the most likely token. One compare rejects the
     * whole tail, which is what makes a pass over 152k logits cheap; no
     * second buffer of that size is ever allocated.
     */
    float vals[512];
    int   ids[512];
    int   n = 0;
    for (int i = 0; i < vocab_size; i++) {
        const float v = logits[i];
        if (n == k) {
            if (v <= vals[0]) continue;      /* worse than everything kept */
        } else {
            /* Grow at the bottom, then insert as if the array were full. */
            for (int j = n; j > 0; j--) { vals[j] = vals[j - 1];
                                          ids[j] = ids[j - 1]; }
            vals[0] = -FLT_MAX; ids[0] = -1;
            n++;
        }
        int j = 0;
        while (j + 1 < n && vals[j + 1] < v) { vals[j] = vals[j + 1];
                                               ids[j] = ids[j + 1]; j++; }
        vals[j] = v; ids[j] = i;
    }

    /* Softmax over the kept candidates, largest last. */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        vals[i] = expf((vals[i] - vals[n - 1]) / temperature);
        sum += vals[i];
    }
    for (int i = 0; i < n; i++) vals[i] /= sum;

    /* Nucleus: keep the most likely tokens up to top_p of the mass. */
    int first = 0;
    if (top_p > 0.0f && top_p < 1.0f) {
        float acc = 0.0f;
        first = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            acc += vals[i];
            first = i;
            if (acc >= top_p) break;
        }
    }

    float mass = 0.0f;
    for (int i = first; i < n; i++) mass += vals[i];

    uint64_t x = *rng ? *rng : 0x9E3779B97F4A7C15ull;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *rng = x;
    const float r = (float)((x * 0x2545F4914F6CDD1Dull) >> 11) /
                    (float)(1ull << 53) * mass;

    float acc = 0.0f;
    for (int i = first; i < n; i++) {
        acc += vals[i];
        if (r <= acc) { *token_id = ids[i]; return VV_OK; }
    }
    *token_id = ids[n - 1];
    return VV_OK;
}
