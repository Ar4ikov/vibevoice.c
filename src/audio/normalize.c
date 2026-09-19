/**
 * @file normalize.c
 * @brief Audio RMS normalization to target dBFS level.
 *
 * VibeVoice-ASR expects: target_dB_FS = -25, eps = 1e-6
 */

#include "vibevoice/audio.h"

#include <string.h>
#include "vibevoice/vibevoice.h"

#include <math.h>

vv_status_t vv_audio_normalize(float* samples, int num_samples,
                                float target_db_fs, float eps) {
    if (!samples) return VV_ERR_NULL_PTR;
    if (num_samples <= 0) return VV_ERR_INVALID_ARG;

    /* Compute RMS */
    double sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
        double s = (double)samples[i];
        sum_sq += s * s;
    }
    double rms = sqrt(sum_sq / (double)num_samples);

    if (rms < (double)eps) {
        VV_LOG_W("normalize: audio is silent (RMS=%.2e), skipping", rms);
        return VV_OK;
    }

    /* Current dBFS = 20 * log10(rms) */
    double current_db = 20.0 * log10(rms);
    double gain_db = (double)target_db_fs - current_db;
    double gain = pow(10.0, gain_db / 20.0);

    /* Apply gain */
    float g = (float)gain;
    for (int i = 0; i < num_samples; i++) {
        samples[i] *= g;
    }

    VV_LOG_D("normalize: RMS %.4f -> target %.1f dBFS (gain: %.4f)",
             (float)rms, target_db_fs, g);
    return VV_OK;
}

/* Full preprocessing pipeline convenience function */
vv_status_t vv_audio_preprocess(const char* wav_path, float** samples,
                                 int* num_samples) {
    if (!wav_path || !samples || !num_samples) return VV_ERR_NULL_PTR;

    /* Step 1: Load WAV */
    float* raw = NULL;
    int raw_len = 0, sr = 0;
    vv_status_t s = vv_audio_load_wav(wav_path, &raw, &raw_len, &sr);
    if (s != VV_OK) return s;

    /* Step 2: Resample to 24kHz if needed */
    float* resampled = NULL;
    int resampled_len = 0;
    if (sr != 24000) {
        s = vv_audio_resample(raw, sr, raw_len, &resampled, 24000,
                               &resampled_len);
        vv_free(raw);
        if (s != VV_OK) return s;
    } else {
        resampled = raw;
        resampled_len = raw_len;
    }

    /* Step 3: Normalize to -25 dBFS */
    s = vv_audio_normalize(resampled, resampled_len, -25.0f, 1e-6f);
    if (s != VV_OK) {
        vv_free(resampled);
        return s;
    }

    *samples = resampled;
    *num_samples = resampled_len;

    VV_LOG_I("audio: preprocessed %s -> %d samples at 24kHz, -25 dBFS",
             wav_path, resampled_len);
    return VV_OK;
}

vv_status_t vv_audio_prepare(const float* pcm, int n_samples, int sample_rate,
                             float** out, int* out_len) {
    return vv_audio_prepare_ex(pcm, n_samples, sample_rate, true,
                               out, out_len);
}

vv_status_t vv_audio_prepare_ex(const float* pcm, int n_samples,
                                int sample_rate, bool normalize,
                                float** out, int* out_len) {
    if (!pcm || !out || !out_len) return VV_ERR_NULL_PTR;
    if (n_samples <= 0 || sample_rate <= 0) return VV_ERR_INVALID_ARG;

    float* buf = NULL;
    int len = 0;

    if (sample_rate != 24000) {
        vv_status_t s = vv_audio_resample(pcm, sample_rate, n_samples,
                                          &buf, 24000, &len);
        if (s != VV_OK) return s;
    } else {
        buf = (float*)vv_alloc((size_t)n_samples * sizeof(float));
        if (!buf) return VV_ERR_OUT_OF_MEMORY;
        memcpy(buf, pcm, (size_t)n_samples * sizeof(float));
        len = n_samples;
    }

    if (normalize) {
        vv_status_t s = vv_audio_normalize(buf, len, -25.0f, 1e-6f);
        if (s != VV_OK) { vv_free(buf); return s; }
    }

    *out = buf;
    *out_len = len;
    return VV_OK;
}

vv_status_t vv_audio_prepare_vibeasr(const float* pcm, int n_samples,
                                     int sample_rate, bool normalize,
                                     float** out, int* out_len) {
    if (!pcm || !out || !out_len) return VV_ERR_NULL_PTR;
    if (n_samples <= 0 || sample_rate <= 0) return VV_ERR_INVALID_ARG;

    /* resample_linear(): the expressions of audio_io.h, in double. */
    const double ratio = (double)sample_rate / 24000.0;
    const size_t len = sample_rate == 24000
                     ? (size_t)n_samples
                     : (size_t)((double)n_samples / ratio);
    if (len == 0 || len > (size_t)0x7fffffff) return VV_ERR_INVALID_ARG;
    float* buf = (float*)vv_alloc(len * sizeof(float));
    if (!buf) return VV_ERR_OUT_OF_MEMORY;
    if (sample_rate == 24000) {
        memcpy(buf, pcm, len * sizeof(float));
    } else {
        for (size_t i = 0; i < len; i++) {
            const double src_pos = (double)i * ratio;
            const size_t idx = (size_t)src_pos;
            const double frac = src_pos - (double)idx;
            if (idx + 1 < (size_t)n_samples)
                buf[i] = (float)((1.0 - frac) * pcm[idx] + frac * pcm[idx + 1]);
            else if (idx < (size_t)n_samples)
                buf[i] = pcm[idx];
            else
                buf[i] = 0.0f;
        }
    }

    if (normalize) {
        /* normalize_audio(): RMS as a float, then one float gain. */
        double sum_sq = 0.0;
        for (size_t i = 0; i < len; i++)
            sum_sq += (double)buf[i] * (double)buf[i];
        const float rms = sqrtf((float)(sum_sq / (double)len));
        if (rms >= 1e-6f) {
            const float scalar = powf(10.0f, -25.0f / 20.0f) / (rms + 1e-6f);
            for (size_t i = 0; i < len; i++) buf[i] *= scalar;
        }
    }

    *out = buf;
    *out_len = (int)len;
    return VV_OK;
}
