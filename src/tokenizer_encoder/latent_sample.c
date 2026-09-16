/**
 * @file latent_sample.c
 * @brief Draw the acoustic latent from its distribution instead of its mode.
 *
 * The acoustic tokenizer's encoder does not return a latent, it returns the
 * mean of a Gaussian with a fixed spread (`fix_std`, 0.5 in this checkpoint).
 * What the reference feeds the connector is a sample, and which sample depends
 * on `std_dist_type`:
 *
 *   "fix"       x = mean + fix_std * N(0,1)      elementwise
 *   "gaussian"  s = N(0,1) * fix_std / 0.8       one scalar for the whole clip
 *               x = mean + s * N(0,1)            elementwise
 *
 * The second is what this checkpoint declares, and it is a scale mixture
 * rather than plain noise: the per-clip `s` is itself drawn, is as likely to
 * be negative as positive, and has RMS 0.625 — wider than `fix_std`. So a
 * clip is perturbed by an amount that varies from run to run, which is exactly
 * why the runtime's default is the mode: it is reproducible, and it is the
 * single most likely latent rather than one draw from around it.
 *
 * The draw is available because it is what the reference does, and it stays
 * reproducible: one seed, one fixed traversal order, no dependence on thread
 * count. It cannot be made bit-identical to PyTorch — a different generator
 * produces different numbers from the same seed — so the check that matters is
 * whether the transcript survives noise of this size, not whether one draw
 * matches.
 *
 * The semantic tokenizer is deterministic in the checkpoint (`fix_std` 0,
 * `std_dist_type` "none") and the reference takes its mean directly, so
 * nothing here applies to it.
 */

#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <string.h>

/*
 * splitmix64. Two multiplies and a few shifts for 64 well-mixed bits, which is
 * all a noise source needs, and its state is a single integer so a seed maps
 * to a stream with nothing else to get wrong.
 */
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/** @brief Uniform in (0, 1); never returns 0, which would break the log. */
static double next_open_unit(uint64_t* state) {
    for (;;) {
        const double u = (double)(splitmix64(state) >> 11) / 9007199254740992.0;
        if (u > 0.0) return u;
    }
}

/**
 * @brief Standard normal pairs by Box-Muller.
 *
 * The polar form would be marginally cheaper but rejects samples, which makes
 * the number of draws depend on the values; this way a seed and an element
 * count fix the whole sequence.
 */
typedef struct {
    uint64_t state;
    double   spare;
    bool     has_spare;
} normal_rng_t;

static void normal_rng_init(normal_rng_t* r, uint64_t seed) {
    /* Seed 0 is a legitimate choice, and splitmix64 handles it. */
    r->state = seed;
    r->spare = 0.0;
    r->has_spare = false;
}

static double next_normal(normal_rng_t* r) {
    if (r->has_spare) {
        r->has_spare = false;
        return r->spare;
    }
    const double u1 = next_open_unit(&r->state);
    const double u2 = next_open_unit(&r->state);
    const double mag = sqrt(-2.0 * log(u1));
    const double ang = 6.283185307179586477 * u2;
    r->spare = mag * sin(ang);
    r->has_spare = true;
    return mag * cos(ang);
}

const char* vv_acoustic_sampling_name(vv_acoustic_sampling_t mode) {
    switch (mode) {
        case VV_ACOUSTIC_MODE:     return "mode";
        case VV_ACOUSTIC_FIX:      return "fix";
        case VV_ACOUSTIC_GAUSSIAN: return "gaussian";
        default:                   return "?";
    }
}

vv_acoustic_sampling_t vv_acoustic_sampling_parse(const char* s) {
    if (!s || !s[0]) return VV_ACOUSTIC_MODE;
    if (strcmp(s, "mode") == 0 || strcmp(s, "mean") == 0 ||
        strcmp(s, "off")  == 0 || strcmp(s, "none") == 0)
        return VV_ACOUSTIC_MODE;
    if (strcmp(s, "fix") == 0)      return VV_ACOUSTIC_FIX;
    if (strcmp(s, "gaussian") == 0) return VV_ACOUSTIC_GAUSSIAN;
    return VV_ACOUSTIC_SAMPLING_COUNT;
}

vv_status_t vv_acoustic_sample(float* latents, int n_frames, int vae_dim,
                               float fix_std, vv_acoustic_sampling_t mode,
                               uint64_t seed, float* out_scale) {
    if (!latents) return VV_ERR_NULL_PTR;
    if (n_frames < 0 || vae_dim <= 0) return VV_ERR_INVALID_ARG;
    if (out_scale) *out_scale = 0.0f;
    if (mode == VV_ACOUSTIC_MODE || fix_std == 0.0f || n_frames == 0)
        return VV_OK;
    if (mode != VV_ACOUSTIC_FIX && mode != VV_ACOUSTIC_GAUSSIAN)
        return VV_ERR_INVALID_ARG;

    normal_rng_t rng;
    normal_rng_init(&rng, seed);

    /*
     * The per-clip scale is drawn first, before any element, so that the same
     * seed gives the same scale whatever the clip length.
     */
    double scale = fix_std;
    if (mode == VV_ACOUSTIC_GAUSSIAN)
        scale = next_normal(&rng) * ((double)fix_std / 0.8);

    const size_t n = (size_t)n_frames * (size_t)vae_dim;
    for (size_t i = 0; i < n; i++)
        latents[i] = (float)((double)latents[i] + scale * next_normal(&rng));

    if (out_scale) *out_scale = (float)scale;
    VV_LOG_I("acoustic: sampled %s latent, scale %.4f, seed %llu",
             vv_acoustic_sampling_name(mode), scale,
             (unsigned long long)seed);
    return VV_OK;
}
