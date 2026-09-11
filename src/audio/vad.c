/**
 * @file vad.c
 * @brief Energy-gated segmenter with hysteresis, pre-roll and hangover.
 *
 * Deliberately simple. A neural VAD would be better at separating speech from
 * steady noise, but it would also mean a second model to ship and a second
 * thing to keep in sync with the ASR; for deciding when someone stopped
 * talking, a two-threshold energy gate with a pre-roll buffer does the job.
 *
 * The two thresholds matter: one level would chatter around the boundary and
 * cut words in half. Speech opens above `start_db` and only closes after
 * `hangover_ms` continuously below `stop_db`, so pauses inside a sentence
 * stay inside the segment.
 */

#include "vibevoice/capture.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

#define FRAME_MS 20

struct vv_vad {
    vv_vad_params_t p;

    int    frame;          /* samples per analysis frame */
    float* acc;            /* partial frame              */
    int    acc_len;

    /*
     * Input FIFO. A push can close a segment part-way through the samples it
     * was given; the rest belongs to the next utterance, so it waits here
     * rather than being folded into the one just emitted.
     */
    float* in;
    int    in_cap, in_len, in_pos;

    float* pre;            /* pre-roll ring              */
    int    pre_cap, pre_len, pre_head;

    float* seg;            /* the segment being built    */
    int    seg_cap, seg_len;

    bool   active;
    int    silence_run;    /* samples below stop_db      */
    int    speech_run;     /* samples above start_db     */
    float  level_db;
    float  floor_db;       /* tracked noise floor        */
    bool   floor_init;     /* seeded from the first frame */
};

/**
 * @brief Level a frame has to beat to open a segment, right now.
 *
 * A fixed threshold assumes a quiet room. Plug in an audio interface with its
 * preamp up and the noise floor alone sits near -38 dBFS, so every frame
 * looks like speech and the model dutifully transcribes `[Noise]`. Tracking
 * the floor and asking speech to beat it by a margin makes the gate mean the
 * same thing on a laptop mic and on a hot line input; the configured
 * thresholds stay as a lower bound, so this can only make the gate stricter.
 */
static float start_threshold(const vv_vad_t* v) {
    if (!v->p.adapt) return v->p.start_db;
    const float adaptive = v->floor_db + v->p.noise_margin_db;
    return adaptive > v->p.start_db ? adaptive : v->p.start_db;
}

/** @brief Closing threshold, keeping the configured hysteresis gap. */
static float stop_threshold(const vv_vad_t* v) {
    const float gap = v->p.start_db - v->p.stop_db;   /* usually 7 dB */
    return start_threshold(v) - (gap > 0.0f ? gap : 0.0f);
}

static float frame_db(const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const double rms = sqrt(s / (double)(n > 0 ? n : 1));
    return 20.0f * log10f((float)rms + 1e-9f);
}

static bool seg_reserve(vv_vad_t* v, int extra) {
    if (v->seg_len + extra <= v->seg_cap) return true;
    int cap = v->seg_cap ? v->seg_cap : v->p.sample_rate;
    while (cap < v->seg_len + extra) cap *= 2;
    float* nb = (float*)vv_alloc((size_t)cap * sizeof(float));
    if (!nb) return false;
    if (v->seg_len) memcpy(nb, v->seg, (size_t)v->seg_len * sizeof(float));
    vv_free(v->seg);
    v->seg = nb;
    v->seg_cap = cap;
    return true;
}

static void pre_push(vv_vad_t* v, const float* x, int n) {
    for (int i = 0; i < n; i++) {
        v->pre[v->pre_head] = x[i];
        v->pre_head = (v->pre_head + 1) % v->pre_cap;
        if (v->pre_len < v->pre_cap) v->pre_len++;
    }
}

/** @brief Copy the pre-roll ring into the segment, oldest first. */
static void pre_drain(vv_vad_t* v) {
    const int start = (v->pre_head - v->pre_len + v->pre_cap) % v->pre_cap;
    if (!seg_reserve(v, v->pre_len)) return;
    for (int i = 0; i < v->pre_len; i++)
        v->seg[v->seg_len + i] = v->pre[(start + i) % v->pre_cap];
    v->seg_len += v->pre_len;
    v->pre_len = 0;
}

vv_status_t vv_vad_create(const vv_vad_params_t* params, vv_vad_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    vv_vad_params_t p = params ? *params : vv_vad_params_default();
    if (p.sample_rate <= 0) return VV_ERR_INVALID_ARG;

    vv_vad_t* v = (vv_vad_t*)vv_alloc(sizeof(vv_vad_t));
    if (!v) return VV_ERR_OUT_OF_MEMORY;
    memset(v, 0, sizeof(*v));
    v->p = p;
    v->frame = p.sample_rate * FRAME_MS / 1000;
    v->level_db = -120.0f;
    /* Start pessimistic: the first frames raise it to whatever the room is. */
    v->floor_db = -90.0f;

    v->acc = (float*)vv_alloc((size_t)v->frame * sizeof(float));
    v->pre_cap = (int)(p.pre_roll_ms * 0.001f * (float)p.sample_rate) + 1;
    v->pre = (float*)vv_alloc((size_t)v->pre_cap * sizeof(float));
    if (!v->acc || !v->pre) { vv_vad_free(v); return VV_ERR_OUT_OF_MEMORY; }

    *out = v;
    return VV_OK;
}

void vv_vad_free(vv_vad_t* v) {
    if (!v) return;
    vv_free(v->acc);
    vv_free(v->pre);
    vv_free(v->seg);
    vv_free(v->in);
    vv_free(v);
}

bool vv_vad_active(const vv_vad_t* v) { return v && v->active; }
float vv_vad_level_db(const vv_vad_t* v) { return v ? v->level_db : -120.0f; }

float vv_vad_threshold_db(const vv_vad_t* v) {
    return v ? start_threshold(v) : 0.0f;
}

float vv_vad_noise_floor_db(const vv_vad_t* v) {
    return v ? v->floor_db : -120.0f;
}

/** @brief Hand the accumulated segment to the caller and start a new one. */
static bool emit(vv_vad_t* v, float** out_pcm, int* out_len) {
    if (v->seg_len <= 0) { v->active = false; return false; }
    *out_pcm = v->seg;
    *out_len = v->seg_len;
    v->seg = NULL;
    v->seg_cap = v->seg_len = 0;
    v->active = false;
    v->silence_run = v->speech_run = 0;
    return true;
}

static bool process_frame(vv_vad_t* v, const float* f, int n,
                          float** out_pcm, int* out_len) {
    const float db = frame_db(f, n);
    v->level_db = db;

    /*
     * Follow the floor down quickly and up slowly: a pause should re-learn a
     * quieter room within a frame or two, while a burst of speech must not
     * drag the floor up behind it. Only quiet frames are sampled, so the
     * floor tracks the room rather than the talker.
     */
    if (!v->floor_init) {
        /* Seed from the very first frame: a slow climb from silence would
         * leave the gate wide open for the first seconds of a noisy input. */
        v->floor_db = db;
        v->floor_init = true;
    } else if (!v->active && db < start_threshold(v)) {
        const float rate = (db < v->floor_db) ? 0.5f : 0.05f;
        v->floor_db += rate * (db - v->floor_db);
    }
    if (v->floor_db < -90.0f) v->floor_db = -90.0f;

    const int min_speech = (int)(v->p.min_speech_ms * 0.001f
                                 * (float)v->p.sample_rate);
    const int hangover = (int)(v->p.hangover_ms * 0.001f
                               * (float)v->p.sample_rate);
    const int max_seg = (int)(v->p.max_segment_s * (float)v->p.sample_rate);

    if (!v->active) {
        pre_push(v, f, n);
        if (db > start_threshold(v)) {
            v->speech_run += n;
            if (v->speech_run >= min_speech) {
                v->active = true;
                v->silence_run = 0;
                pre_drain(v);
                if (seg_reserve(v, n)) {
                    memcpy(v->seg + v->seg_len, f, (size_t)n * sizeof(float));
                    v->seg_len += n;
                }
            }
        } else {
            v->speech_run = 0;
        }
        return false;
    }

    if (seg_reserve(v, n)) {
        memcpy(v->seg + v->seg_len, f, (size_t)n * sizeof(float));
        v->seg_len += n;
    }

    if (db < stop_threshold(v)) {
        v->silence_run += n;
        if (v->silence_run >= hangover) return emit(v, out_pcm, out_len);
    } else {
        v->silence_run = 0;
    }

    /* A speaker who never pauses still has to be transcribed eventually. */
    if (v->seg_len >= max_seg) return emit(v, out_pcm, out_len);
    return false;
}

/** @brief Append to the input FIFO, compacting what has been consumed. */
static bool in_push(vv_vad_t* v, const float* pcm, int n) {
    if (v->in_pos > 0) {
        const int keep = v->in_len - v->in_pos;
        if (keep > 0) memmove(v->in, v->in + v->in_pos,
                              (size_t)keep * sizeof(float));
        v->in_len = keep;
        v->in_pos = 0;
    }
    if (v->in_len + n > v->in_cap) {
        int cap = v->in_cap ? v->in_cap : 4096;
        while (cap < v->in_len + n) cap *= 2;
        float* nb = (float*)vv_alloc((size_t)cap * sizeof(float));
        if (!nb) return false;
        if (v->in_len) memcpy(nb, v->in, (size_t)v->in_len * sizeof(float));
        vv_free(v->in);
        v->in = nb;
        v->in_cap = cap;
    }
    memcpy(v->in + v->in_len, pcm, (size_t)n * sizeof(float));
    v->in_len += n;
    return true;
}

/** @brief Drain whole frames out of the FIFO until one closes a segment. */
static bool drain(vv_vad_t* v, float** out_pcm, int* out_len) {
    while (v->in_pos < v->in_len) {
        const int need = v->frame - v->acc_len;
        const int avail = v->in_len - v->in_pos;
        const int take = avail < need ? avail : need;
        memcpy(v->acc + v->acc_len, v->in + v->in_pos,
               (size_t)take * sizeof(float));
        v->acc_len += take;
        v->in_pos += take;
        if (v->acc_len < v->frame) break;

        v->acc_len = 0;
        if (process_frame(v, v->acc, v->frame, out_pcm, out_len)) return true;
    }
    return false;
}

bool vv_vad_push(vv_vad_t* v, const float* pcm, int n,
                 float** out_pcm, int* out_len) {
    if (!v || !pcm || n <= 0 || !out_pcm || !out_len) return false;
    if (!in_push(v, pcm, n)) return false;
    return drain(v, out_pcm, out_len);
}

/**
 * @brief Drain samples already buffered, without adding more.
 *
 * A caller that got a segment back has to keep calling this until it returns
 * false, or the tail of a busy stream would never be examined.
 */
bool vv_vad_drain(vv_vad_t* v, float** out_pcm, int* out_len) {
    if (!v || !out_pcm || !out_len) return false;
    return drain(v, out_pcm, out_len);
}

bool vv_vad_flush(vv_vad_t* v, float** out_pcm, int* out_len) {
    if (!v || !out_pcm || !out_len) return false;
    if (v->acc_len > 0 && v->active && seg_reserve(v, v->acc_len)) {
        memcpy(v->seg + v->seg_len, v->acc,
               (size_t)v->acc_len * sizeof(float));
        v->seg_len += v->acc_len;
    }
    v->acc_len = 0;
    if (v->seg_len == 0) return false;
    return emit(v, out_pcm, out_len);
}
