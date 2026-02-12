/**
 * @file resample.c
 * @brief Audio resampler using windowed sinc interpolation (polyphase).
 *
 * Target: resample any input rate to 24000 Hz for VibeVoice-ASR.
 */

#include "vibevoice/audio.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Sinc filter parameters */
#define SINC_HALF_WIDTH 16  /* Taps on each side */

static float sinc(float x) {
    if (fabsf(x) < 1e-8f) return 1.0f;
    float px = (float)M_PI * x;
    return sinf(px) / px;
}

/** Blackman-Harris window */
static float blackman_harris(float n, float N) {
    float a0 = 0.35875f;
    float a1 = 0.48829f;
    float a2 = 0.14128f;
    float a3 = 0.01168f;
    float x = 2.0f * (float)M_PI * n / (N - 1.0f);
    return a0 - a1 * cosf(x) + a2 * cosf(2.0f * x) - a3 * cosf(3.0f * x);
}

vv_status_t vv_audio_resample(const float* in, int in_sr, int in_len,
                               float** out, int target_sr, int* out_len) {
    if (!in || !out || !out_len) return VV_ERR_NULL_PTR;
    if (in_sr <= 0 || target_sr <= 0 || in_len <= 0) return VV_ERR_INVALID_ARG;

    /* If already at target rate, just copy */
    if (in_sr == target_sr) {
        float* copy = (float*)vv_alloc((size_t)in_len * sizeof(float));
        if (!copy) return VV_ERR_OUT_OF_MEMORY;
        memcpy(copy, in, (size_t)in_len * sizeof(float));
        *out = copy;
        *out_len = in_len;
        return VV_OK;
    }

    double ratio = (double)target_sr / (double)in_sr;
    int n_out = (int)((double)in_len * ratio) + 1;

    float* result = (float*)vv_alloc((size_t)n_out * sizeof(float));
    if (!result) return VV_ERR_OUT_OF_MEMORY;

    /*
     * Windowed sinc interpolation.
     * For each output sample, compute its position in the input timeline,
     * then convolve with a windowed sinc filter.
     */
    float cutoff = (ratio < 1.0) ? (float)ratio : 1.0f;
    int half_w = SINC_HALF_WIDTH;
    float filter_width = (float)(2 * half_w);

    int actual_out = 0;
    for (int i = 0; i < n_out; i++) {
        double src_pos = (double)i / ratio;
        int center = (int)src_pos;
        float frac = (float)(src_pos - (double)center);

        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int j = -half_w; j <= half_w; j++) {
            int idx = center + j;
            if (idx < 0 || idx >= in_len) continue;

            float t = (float)j - frac;
            float w = sinc(t * cutoff) * cutoff;

            /* Apply window */
            float wn = (float)(j - frac + half_w);
            w *= blackman_harris(wn, filter_width + 1.0f);

            sum += in[idx] * w;
            weight_sum += w;
        }

        if (weight_sum > 1e-8f) {
            result[actual_out] = sum / weight_sum;
        } else {
            result[actual_out] = 0.0f;
        }
        actual_out++;
    }

    *out = result;
    *out_len = actual_out;

    VV_LOG_D("resample: %d Hz -> %d Hz, %d -> %d samples",
             in_sr, target_sr, in_len, actual_out);
    return VV_OK;
}
