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

/*
 * The release line this source belongs to, and the only place the number is
 * written down. The Release workflow bumps all four together; CMake refuses
 * to configure if they disagree. See docs/RELEASING.md.
 */
#define VV_VERSION_MAJOR 0
#define VV_VERSION_MINOR 3
#define VV_VERSION_PATCH 0
#define VV_VERSION_STRING "0.3.0"

/**
 * @brief Full SemVer 2.0 version of this build.
 *
 * `0.2.0` for a build of the v0.2.0 tag; `0.2.1-dev.5+g1a2b3c4` for a build
 * five commits past it, a prerelease of the next version so that it sorts
 * between the two; `+...dirty` when the tree had uncommitted changes.
 */
const char* vv_version(void);

/**
 * @brief Commit this binary was built from: short hash, "-dirty" when the
 * tree had changes, or "unknown" without git. Reported by `vv_cli --version`,
 * `GET /health` and the `vibevoice_build_info` metric next to the version.
 */
const char* vv_build_ref(void);

/** @brief Compile-time features, space separated: "cuda openmp". */
const char* vv_build_features(void);

/* ─── Core ──────────────────────────────────────────────────────────────── */

/** @brief Get human-readable string for a status code. */
const char* vv_status_str(vv_status_t status);

/** @brief Set the global log level. Default: VV_LOG_INFO. */
void vv_log_set_level(vv_log_level_t level);
vv_log_level_t vv_log_get_level(void);

/** @brief VV_DUMP_DIR if set and non-empty, else NULL. */
const char* vv_debug_dump_dir(void);
/** @brief Write a raw blob to $VV_DUMP_DIR/<name>.bin (no-op when unset). */
void vv_debug_dump(const char* name, const void* data, size_t bytes);

/* ─── Device selection (gpu_select.c) ───────────────────────────────────── */

/** @brief Parse one memory cap: "8G", "18GiB", "8192M", "80%", or bytes. */
vv_status_t vv_parse_mem_cap(const char* text, vv_mem_cap_t* out);

/** @brief Resolve a cap against a device's total size. 0 = uncapped. */
size_t vv_mem_cap_bytes(const vv_mem_cap_t* cap, size_t total);

/** @brief Parse `--gpus`: "0,1", "all", or one id. Caps are left untouched. */
vv_status_t vv_gpu_set_parse(const char* text, vv_gpu_set_t* out);

/** @brief Parse `--gpu-memory`: one cap for every device, or one per device. */
vv_status_t vv_gpu_set_caps(const char* text, vv_gpu_set_t* set);

/**
 * @brief Default an empty set to `fallback_id`, check the ids exist, and log
 *        what each device has and what it is allowed to spend.
 */
vv_status_t vv_gpu_set_resolve(vv_gpu_set_t* set, int fallback_id,
                               bool cpu_only);

/** @brief Bytes the `index`th device may use: free * budget, then the cap. */
size_t vv_gpu_budget(const vv_gpu_set_t* set, int index, float vram_budget);

/** @brief What that device already holds and the budget cannot place. */
size_t vv_gpu_reserved(const vv_gpu_set_t* set, int index);

/** @brief Monotonic clock in milliseconds. */
double vv_time_ms(void);

/** @brief Sleep for the given number of milliseconds. */
void vv_msleep(int ms);

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
