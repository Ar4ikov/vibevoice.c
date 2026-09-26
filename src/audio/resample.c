/**
 * @file resample.c
 * @brief Audio resampler using windowed sinc interpolation (polyphase).
 *
 * Target: resample any input rate to 24000 Hz for VibeVoice-ASR.
 */

#include "vibevoice/audio.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdint.h>
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

/** Tap j's weight for an output at fractional offset `frac`. */
static float tap_weight(int j, float frac, float cutoff, int half_w,
                        float filter_width) {
    float t = (float)j - frac;
    float w = sinc(t * cutoff) * cutoff;

    /* Apply window */
    float wn = (float)(j - frac + half_w);
    w *= blackman_harris(wn, filter_width + 1.0f);
    return w;
}

/*
 * An output's tap weights depend only on its fractional offset: a rational
 * rate pair repeats a handful of offsets (16 kHz -> 24 kHz three, 44.1 kHz
 * eighty), so each offset's weights are computed once -- by tap_weight, the
 * expressions the per-sample loop used, so the output keeps its bits -- and
 * found again by the offset's exact bits. Computing them per sample (33 sinf
 * and 99 cosf an output) was most of preparing a 16 kHz file.
 */
#define RS_TAPS  (2 * SINC_HALF_WIDTH + 1)
#define RS_SLOTS 256                    /* a power of two */

typedef struct rs_cache {
    uint32_t key[RS_SLOTS];             /* frac's bits + 1; 0: empty */
    float    w[RS_SLOTS][RS_TAPS];
} rs_cache_t;

static const float* tap_weights(rs_cache_t* c, float frac, float cutoff,
                                int half_w, float filter_width,
                                float* scratch) {
    uint32_t bits;
    memcpy(&bits, &frac, sizeof(bits));
    float* w = scratch;
    uint32_t slot = 0;
    if (c) {
        slot = (bits * 2654435761u) >> 24 & (RS_SLOTS - 1);
        if (c->key[slot] == bits + 1u) return c->w[slot];
        w = c->w[slot];
    }
    for (int j = -half_w; j <= half_w; j++)
        w[j + half_w] = tap_weight(j, frac, cutoff, half_w, filter_width);
    if (c) c->key[slot] = bits + 1u;
    return w;
}

/*
 * Output sample `i`: the input around i / ratio, windowed-sinc weighted and
 * renormalised. `in` holds absolute input samples from `base` on, and
 * `in_len` is where the input ends (taps at or past it are skipped). The
 * batch and the streaming resampler both come through here, so a stream
 * reproduces a whole-file resample sample for sample. `c` (NULL: none)
 * keeps the weights of the offsets seen so far.
 */
static float resample_one(rs_cache_t* c, const float* in, int64_t base,
                          int64_t in_len, int64_t i, double ratio,
                          float cutoff, int half_w, float filter_width) {
    double src_pos = (double)i / ratio;
    int64_t center = (int64_t)src_pos;
    float frac = (float)(src_pos - (double)center);

    float scratch[RS_TAPS];
    const float* wt = tap_weights(c, frac, cutoff, half_w, filter_width,
                                  scratch);
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int j = -half_w; j <= half_w; j++) {
        int64_t idx = center + j;
        if (idx < 0 || idx >= in_len) continue;
        const float w = wt[j + half_w];
        sum += in[idx - base] * w;
        weight_sum += w;
    }
    return weight_sum > 1e-8f ? sum / weight_sum : 0.0f;
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

    /* Without the cache (no memory for it) every output computes its own
     * weights: slower, the same bits. */
    rs_cache_t* cache = (rs_cache_t*)vv_alloc(sizeof(rs_cache_t));
    if (cache) memset(cache->key, 0, sizeof(cache->key));
    int actual_out = 0;
    for (int i = 0; i < n_out; i++) {
        result[actual_out++] = resample_one(cache, in, 0, in_len, i, ratio,
                                            cutoff, half_w, filter_width);
    }
    vv_free(cache);

    *out = result;
    *out_len = actual_out;

    VV_LOG_D("resample: %d Hz -> %d Hz, %d -> %d samples",
             in_sr, target_sr, in_len, actual_out);
    return VV_OK;
}

/* ─── Streaming ─────────────────────────────────────────────────────────── */

struct vv_resampler {
    int     in_sr, out_sr;
    double  ratio;
    float   cutoff;
    float*  buf;          /* input samples [base, base + len) */
    int64_t base, len, cap;
    int64_t next;         /* next output index */
    float*  out;
    size_t  out_cap;
    bool    finished;
    rs_cache_t* cache;    /* tap weights by offset; NULL: computed each time */
};

vv_status_t vv_resampler_create(int in_sr, int out_sr, vv_resampler_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (in_sr <= 0 || out_sr <= 0) return VV_ERR_INVALID_ARG;
    vv_resampler_t* r = (vv_resampler_t*)vv_alloc(sizeof(*r));
    if (!r) return VV_ERR_OUT_OF_MEMORY;
    memset(r, 0, sizeof(*r));
    r->in_sr = in_sr;
    r->out_sr = out_sr;
    r->ratio = (double)out_sr / (double)in_sr;
    r->cutoff = (r->ratio < 1.0) ? (float)r->ratio : 1.0f;
    if (in_sr != out_sr) {
        r->cache = (rs_cache_t*)vv_alloc(sizeof(rs_cache_t));
        if (r->cache) memset(r->cache->key, 0, sizeof(r->cache->key));
    }
    *out = r;
    return VV_OK;
}

void vv_resampler_free(vv_resampler_t* r) {
    if (!r) return;
    vv_free(r->buf);
    vv_free(r->out);
    vv_free(r->cache);
    vv_free(r);
}

static bool rs_out_reserve(vv_resampler_t* r, size_t n) {
    if (n <= r->out_cap) return true;
    size_t cap = r->out_cap ? r->out_cap : 1024;
    while (cap < n) cap *= 2;
    float* o = (float*)vv_realloc(r->out, cap * sizeof(float));
    if (!o) return false;
    r->out = o;
    r->out_cap = cap;
    return true;
}

/* Emit outputs up to (not including) `limit`, taps bounded by `in_len`. */
static vv_status_t rs_emit(vv_resampler_t* r, int64_t limit, int64_t in_len,
                           size_t* n_out) {
    *n_out = 0;
    if (limit <= r->next) return VV_OK;
    if (!rs_out_reserve(r, (size_t)(limit - r->next)))
        return VV_ERR_OUT_OF_MEMORY;
    size_t k = 0;
    for (int64_t i = r->next; i < limit; i++)
        r->out[k++] = resample_one(r->cache, r->buf, r->base, in_len, i,
                                   r->ratio, r->cutoff, SINC_HALF_WIDTH,
                                   (float)(2 * SINC_HALF_WIDTH));
    r->next = limit;
    *n_out = k;
    /* Drop input no future output reaches. */
    const int64_t keep_from = (int64_t)((double)r->next / r->ratio)
                            - SINC_HALF_WIDTH - 1;
    if (keep_from > r->base) {
        int64_t drop = keep_from - r->base;
        if (drop > r->len) drop = r->len;
        memmove(r->buf, r->buf + drop, (size_t)(r->len - drop) * sizeof(float));
        r->base += drop;
        r->len -= drop;
    }
    return VV_OK;
}

vv_status_t vv_resampler_push(vv_resampler_t* r, const float* in, size_t n,
                              const float** out, size_t* n_out) {
    if (!r || !out || !n_out || (n && !in)) return VV_ERR_NULL_PTR;
    *out = NULL;
    *n_out = 0;
    if (r->finished) return VV_ERR_INVALID_ARG;
    if (r->in_sr == r->out_sr) {
        if (!rs_out_reserve(r, n ? n : 1)) return VV_ERR_OUT_OF_MEMORY;
        if (n) memcpy(r->out, in, n * sizeof(float));
        *out = r->out;
        *n_out = n;
        return VV_OK;
    }
    if (r->len + (int64_t)n > r->cap) {
        int64_t cap = r->cap ? r->cap : 4096;
        while (cap < r->len + (int64_t)n) cap *= 2;
        float* b = (float*)vv_realloc(r->buf, (size_t)cap * sizeof(float));
        if (!b) return VV_ERR_OUT_OF_MEMORY;
        r->buf = b;
        r->cap = cap;
    }
    if (n) memcpy(r->buf + r->len, in, n * sizeof(float));
    r->len += (int64_t)n;
    /* Output i is final once every tap it reads has arrived:
     * (int)(i / ratio) + half_w < samples so far. */
    const int64_t have = r->base + r->len;
    int64_t limit = r->next;
    while ((int64_t)((double)limit / r->ratio) + SINC_HALF_WIDTH < have)
        limit++;
    vv_status_t s = rs_emit(r, limit, have, n_out);
    *out = r->out;
    return s;
}

vv_status_t vv_resampler_finish(vv_resampler_t* r, const float** out,
                                size_t* n_out) {
    if (!r || !out || !n_out) return VV_ERR_NULL_PTR;
    *out = NULL;
    *n_out = 0;
    if (r->finished) return VV_OK;
    r->finished = true;
    if (r->in_sr == r->out_sr) return VV_OK;
    const int64_t in_len = r->base + r->len;
    if (in_len <= 0) return VV_OK;
    /* As many outputs as the whole-file resampler makes. */
    const int64_t total = (int64_t)((double)in_len * r->ratio) + 1;
    vv_status_t s = rs_emit(r, total, in_len, n_out);
    *out = r->out;
    return s;
}
