/**
 * @file capture.h
 * @brief Microphone capture and voice activity detection.
 *
 * Capture backends, picked at runtime in this order:
 *
 *   Linux     ALSA, dlopen'd at runtime so the binary still runs without it
 *   fallback  an external recorder on a pipe: arecord on Linux, ffmpeg's
 *             dshow input on Windows, avfoundation on macOS
 *   file      a WAV played back at wall-clock speed, for testing the
 *             streaming path without a sound card
 *
 * Samples always arrive as mono float at the requested rate, through a lock-
 * free ring buffer written by the capture thread.
 */
#ifndef VV_CAPTURE_H
#define VV_CAPTURE_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vv_mic vv_mic_t;

/** @brief One capture device, as reported by the platform's enumerator. */
typedef struct vv_mic_device {
    char id[384];    /**< What to pass as --device. Stable across reboots. */
    char name[256];  /**< Human-readable label, UTF-8.                     */
    bool is_default; /**< The one picked when --device is absent.          */
} vv_mic_device_t;

/**
 * @brief List the capture devices the active backend can open.
 *
 * @param out    vv_alloc'd array the caller releases with vv_free()
 * @param count  number of entries written
 * @return VV_ERR_NOT_FOUND when nothing could be enumerated (no ffmpeg on
 *         Windows/macOS, no libasound on Linux)
 */
vv_status_t vv_mic_list_devices(vv_mic_device_t** out, int* count);

/**
 * @brief Turn a --device argument into a name the backend accepts.
 *
 * Accepts an index from the listing, an exact id or name, or a case-
 * insensitive fragment of either; an empty request means the first device.
 * Unmatched text is passed through so a hand-written device name still works.
 *
 * @return false only when no device could be chosen at all
 */
bool vv_mic_resolve_device(const char* request, char* out, size_t n,
                           char* label, size_t label_n);

/**
 * @brief Open an input device.
 *
 * @param sample_rate  requested rate; 24000 avoids a resample later
 * @param device       index, name or fragment per vv_mic_resolve_device(),
 *                     or NULL for the default
 */
vv_status_t vv_mic_open(int sample_rate, const char* device, vv_mic_t** out);

/**
 * @brief Feed a WAV file through the same interface, in real time.
 *
 * Lets the streaming path be exercised on a machine with no audio hardware:
 * samples are released at wall-clock speed, exactly as a device would.
 */
vv_status_t vv_mic_open_file(const char* wav_path, int sample_rate,
                             bool realtime, vv_mic_t** out);

/** @brief Copy up to `max` new samples out; returns how many. Never blocks. */
int vv_mic_read(vv_mic_t* m, float* dst, int max);

/** @brief True once a file source has been fully consumed. */
bool vv_mic_eof(const vv_mic_t* m);

/** @brief Samples dropped because the reader fell behind. */
uint64_t vv_mic_overruns(const vv_mic_t* m);

/** @brief Name of the backend in use, for logs. */
const char* vv_mic_backend(const vv_mic_t* m);

/** @brief Label of the device in use, for logs. Empty when unknown. */
const char* vv_mic_device_label(const vv_mic_t* m);

/** @brief Samples handed over by the backend so far. Zero means silence. */
uint64_t vv_mic_captured(const vv_mic_t* m);

void vv_mic_close(vv_mic_t* m);

/* ─── Voice activity detection ───────────────────────────────────────────── */

typedef struct vv_vad vv_vad_t;

typedef struct vv_vad_params {
    int   sample_rate;
    float start_db;        /**< Speech starts above this. Default -38 dBFS.  */
    float stop_db;         /**< Speech ends below this. Default -45 dBFS.    */
    float min_speech_ms;   /**< Ignore blips shorter than this. Default 250. */
    float hangover_ms;     /**< Silence needed to close a segment. Def. 700. */
    float pre_roll_ms;     /**< Audio kept before the trigger. Default 300.  */
    float max_segment_s;   /**< Force a cut after this long. Default 30.     */
} vv_vad_params_t;

static inline vv_vad_params_t vv_vad_params_default(void) {
    vv_vad_params_t p;
    p.sample_rate = 24000;
    p.start_db = -38.0f;
    p.stop_db = -45.0f;
    p.min_speech_ms = 250.0f;
    p.hangover_ms = 700.0f;
    p.pre_roll_ms = 300.0f;
    p.max_segment_s = 30.0f;
    return p;
}

vv_status_t vv_vad_create(const vv_vad_params_t* p, vv_vad_t** out);
void vv_vad_free(vv_vad_t* v);

/**
 * @brief Push captured samples; returns a finished utterance, if any.
 *
 * @param out_pcm  set to a buffer the caller frees with vv_free()
 * @param out_len  its length in samples
 * @return true when a segment closed on this call
 */
bool vv_vad_push(vv_vad_t* v, const float* pcm, int n,
                 float** out_pcm, int* out_len);

/**
 * @brief Drain samples already buffered, without pushing more.
 *
 * vv_vad_push returns at most one segment; call this until it returns false
 * so a burst that contains two utterances yields both.
 */
bool vv_vad_drain(vv_vad_t* v, float** out_pcm, int* out_len);

/** @brief Close any open segment, e.g. at end of stream. */
bool vv_vad_flush(vv_vad_t* v, float** out_pcm, int* out_len);

/** @brief True while the detector considers speech to be in progress. */
bool vv_vad_active(const vv_vad_t* v);

/** @brief Level of the most recent frame, in dBFS. */
float vv_vad_level_db(const vv_vad_t* v);

#ifdef __cplusplus
}
#endif

#endif /* VV_CAPTURE_H */
