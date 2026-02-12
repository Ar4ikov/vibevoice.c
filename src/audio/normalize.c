/**
 * @file normalize.c
 * @brief Audio RMS normalization to target dBFS level.
 *
 * VibeVoice-ASR expects: target_dB_FS = -25, eps = 1e-6
 */

#include "vibevoice/audio.h"
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
