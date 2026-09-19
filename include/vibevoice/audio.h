/**
 * @file audio.h
 * @brief Audio I/O and preprocessing API.
 *
 * Pipeline: WAV → resample 24kHz → normalize to -25 dBFS → raw float32 PCM
 * NO mel spectrogram, NO FFT, NO STFT.
 */
#ifndef VV_AUDIO_H
#define VV_AUDIO_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load a WAV file into float32 mono samples.
 *
 * Supports: PCM 8/16/24/32 bit, IEEE float32, mono and stereo.
 * Stereo is downmixed to mono.
 *
 * @param path         Path to WAV file
 * @param samples      Output: allocated float32 array (caller must vv_free)
 * @param num_samples  Output: number of samples
 * @param sample_rate  Output: original sample rate
 */
vv_status_t vv_audio_load_wav(const char* path, float** samples,
                               int* num_samples, int* sample_rate);

/**
 * @brief Write mono float samples as a 16-bit PCM WAV.
 *
 * @param path         Destination file, overwritten if it exists
 * @param pcm          Mono samples in [-1, 1]; anything outside is clipped
 * @param num_samples  Number of samples
 * @param sample_rate  Sample rate to record in the header
 */
vv_status_t vv_audio_save_wav(const char* path, const float* pcm,
                              int num_samples, int sample_rate);

/**
 * @brief Resample audio to target sample rate using sinc interpolation.
 *
 * @param in         Input samples
 * @param in_sr      Input sample rate
 * @param in_len     Number of input samples
 * @param out        Output: allocated resampled array (caller must vv_free)
 * @param target_sr  Target sample rate (24000 for VibeVoice)
 * @param out_len    Output: number of output samples
 */
vv_status_t vv_audio_resample(const float* in, int in_sr, int in_len,
                               float** out, int target_sr, int* out_len);

/**
 * @brief Fixed-ratio resampler fed piece by piece (live PCM).
 *
 * Pushing a signal in any split and then finishing yields exactly what
 * vv_audio_resample() gives for the whole signal, sample for sample: an
 * output is released as soon as every input tap it reads has arrived
 * (16 input samples of delay), and finish() emits the tail with the same
 * edge handling. Equal rates pass samples through.
 */
typedef struct vv_resampler vv_resampler_t;

vv_status_t vv_resampler_create(int in_sr, int out_sr, vv_resampler_t** out);
void vv_resampler_free(vv_resampler_t* r);

/**
 * @brief Add input; `*out` points at the outputs that became final (owned by
 *        the resampler, valid until the next call), `*n_out` counts them.
 */
vv_status_t vv_resampler_push(vv_resampler_t* r, const float* in, size_t n,
                              const float** out, size_t* n_out);

/** @brief End of input: the remaining outputs. */
vv_status_t vv_resampler_finish(vv_resampler_t* r, const float** out,
                                size_t* n_out);

/**
 * @brief Normalize audio to target dBFS level.
 *
 * Computes RMS, then applies gain to reach target_db_fs.
 * In-place operation.
 *
 * @param samples       Audio samples (modified in-place)
 * @param num_samples   Number of samples
 * @param target_db_fs  Target dBFS level (e.g., -25.0)
 * @param eps           Small epsilon for numerical stability (e.g., 1e-6)
 */
vv_status_t vv_audio_normalize(float* samples, int num_samples,
                                float target_db_fs, float eps);

/**
 * @brief Full audio preprocessing pipeline.
 *
 * load_wav → resample to 24kHz → normalize to -25 dBFS.
 *
 * @param wav_path    Path to input WAV file
 * @param samples     Output: preprocessed float32 samples at 24kHz
 * @param num_samples Output: number of output samples
 */
vv_status_t vv_audio_preprocess(const char* wav_path, float** samples,
                                 int* num_samples);

/**
 * @brief Same pipeline for samples already in memory (mic, HTTP upload).
 *
 * Resamples to 24 kHz if needed and normalizes to -25 dBFS, always returning
 * a fresh buffer the caller frees with vv_free().
 */
vv_status_t vv_audio_prepare(const float* pcm, int n_samples, int sample_rate,
                             float** out, int* out_len);

/**
 * @brief vv_audio_prepare() with the loudness step optional.
 *
 * The streaming model's preprocessor_config.json turns normalization off
 * (`normalize_audio: false`); the batch models keep it on. The model family
 * says which (vv_family_t::normalize_audio).
 */
vv_status_t vv_audio_prepare_ex(const float* pcm, int n_samples,
                                int sample_rate, bool normalize,
                                float** out, int* out_len);

/**
 * @brief vv_audio_prepare_ex() the way VibeASR.cpp prepares audio for
 *        VibeVoice-ASR-BitNet (utils/audio_io.h): linear-interpolation
 *        resampling to 24 kHz (`floor(n / ratio)` samples) and the gain
 *        `10^(-25/20) / (rms + 1e-6)` with the RMS taken as a float.
 *
 * Its int8 speech encoder quantizes the whole clip with one scale, so a
 * gain that differs in the last bit already moves int8 samples; matching
 * the reference's transcripts needs its audio.
 */
vv_status_t vv_audio_prepare_vibeasr(const float* pcm, int n_samples,
                                     int sample_rate, bool normalize,
                                     float** out, int* out_len);

/**
 * @brief Decode any audio file into mono float samples.
 *
 * WAV is parsed directly. Anything else is handed to ffmpeg when it is on
 * PATH, which is how the server accepts mp3/m4a/ogg/flac uploads; without
 * ffmpeg those return VV_ERR_AUDIO_FORMAT.
 */
vv_status_t vv_audio_load_any(const char* path, float** samples,
                              int* num_samples, int* sample_rate);

/**
 * @brief Decode an in-memory audio blob (sniffs WAV, else ffmpeg via a temp
 *        file). Used by the HTTP upload path.
 */
vv_status_t vv_audio_load_memory(const void* data, size_t size,
                                 float** samples, int* num_samples,
                                 int* sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* VV_AUDIO_H */
