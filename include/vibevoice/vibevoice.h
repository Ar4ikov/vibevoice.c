/**
 * @file vibevoice.h
 * @brief Umbrella header for vibevoice.c — Pure C VibeVoice-ASR runtime
 *
 * Include this single header to get access to the full public API.
 */
#ifndef VIBEVOICE_H
#define VIBEVOICE_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Version ───────────────────────────────────────────────────────────── */

#define VV_VERSION_MAJOR 0
#define VV_VERSION_MINOR 1
#define VV_VERSION_PATCH 0
#define VV_VERSION_STRING "0.1.0"

/* ─── Core ──────────────────────────────────────────────────────────────── */

/** @brief Get human-readable string for a status code. */
const char* vv_status_str(vv_status_t status);

/** @brief Set the global log level. Default: VV_LOG_INFO. */
void vv_log_set_level(vv_log_level_t level);

/** @brief Log a message. Prefer the VV_LOG_* macros below. */
void vv_log(vv_log_level_t level, const char* fmt, ...);

#define VV_LOG_E(...) vv_log(VV_LOG_ERROR, __VA_ARGS__)
#define VV_LOG_W(...) vv_log(VV_LOG_WARN,  __VA_ARGS__)
#define VV_LOG_I(...) vv_log(VV_LOG_INFO,  __VA_ARGS__)
#define VV_LOG_D(...) vv_log(VV_LOG_DEBUG, __VA_ARGS__)

/* ─── Memory ────────────────────────────────────────────────────────────── */

/** @brief Allocate CPU memory with tracking. */
void* vv_alloc(size_t size);

/** @brief Reallocate CPU memory with tracking. */
void* vv_realloc(void* ptr, size_t new_size);

/** @brief Free CPU memory. */
void vv_free(void* ptr);

/** @brief Get total CPU memory currently allocated. */
size_t vv_alloc_total(void);

/* ─── Tensor helpers ────────────────────────────────────────────────────── */

/** @brief Calculate total number of elements in a tensor. */
static inline int64_t vv_tensor_numel(const vv_tensor_t* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/** @brief Free tensor data (CPU or GPU). */
vv_status_t vv_tensor_free(vv_tensor_t* t);

/* ─── Forward declarations (detailed APIs in separate headers) ──────────── */

/* Model loading — see model.h */
typedef struct vv_model vv_model_t;
vv_status_t vv_model_load(const char* model_dir, vv_model_t** out);
vv_status_t vv_model_free(vv_model_t* model);

/* Inference — see inference.h */
typedef struct vv_inference_ctx vv_inference_ctx_t;
vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               const vv_init_params_t* params,
                               vv_inference_ctx_t** ctx);
vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result);
vv_status_t vv_transcription_free(vv_transcription_t* result);
vv_status_t vv_inference_free(vv_inference_ctx_t* ctx);
const vv_perf_metrics_t* vv_inference_get_perf(const vv_inference_ctx_t* ctx);

/* Audio — see audio.h */
vv_status_t vv_audio_load_wav(const char* path, float** samples,
                               int* num_samples, int* sample_rate);
vv_status_t vv_audio_resample(const float* in, int in_sr, int in_len,
                               float** out, int target_sr, int* out_len);
vv_status_t vv_audio_normalize(float* samples, int num_samples,
                                float target_db_fs, float eps);

/* Text tokenizer — see text_tokenizer.h */
typedef struct vv_tokenizer vv_tokenizer_t;
vv_status_t vv_tokenizer_load(const char* path, vv_tokenizer_t** out);
vv_status_t vv_tokenizer_encode(const vv_tokenizer_t* tok, const char* text,
                                 int32_t** token_ids, int* num_tokens);
vv_status_t vv_tokenizer_decode(const vv_tokenizer_t* tok,
                                 const int32_t* token_ids, int num_tokens,
                                 char** text);
vv_status_t vv_tokenizer_free(vv_tokenizer_t* tok);

#ifdef __cplusplus
}
#endif

#endif /* VIBEVOICE_H */
