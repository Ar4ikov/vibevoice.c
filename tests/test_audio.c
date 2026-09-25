/**
 * @file test_audio.c
 * @brief Unit tests for audio pipeline: WAV loading, resampling, normalization.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
        fprintf(stdout, "  PASS: %s\n", msg); \
    } \
} while(0)

/* ─── Test normalize ────────────────────────────────────────────────────── */

static void test_normalize(void) {
    printf("test_normalize:\n");

    /* Create a simple sine wave */
    int n = 24000;  /* 1 second at 24kHz */
    float* samples = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        samples[i] = 0.5f * sinf(2.0f * 3.14159f * 440.0f * (float)i / 24000.0f);
    }

    /* Compute RMS before */
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (double)samples[i] * (double)samples[i];
    }
    double rms_before = sqrt(sum_sq / n);
    double db_before = 20.0 * log10(rms_before);

    /* Normalize to -25 dBFS */
    vv_status_t s = vv_audio_normalize(samples, n, -25.0f, 1e-6f);
    TEST_ASSERT(s == VV_OK, "normalize returns VV_OK");

    /* Compute RMS after */
    sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (double)samples[i] * (double)samples[i];
    }
    double rms_after = sqrt(sum_sq / n);
    double db_after = 20.0 * log10(rms_after);

    TEST_ASSERT(fabs(db_after - (-25.0)) < 0.5,
                "normalized dBFS is close to -25");

    printf("  dBFS: %.2f -> %.2f (target: -25.0)\n", db_before, db_after);

    vv_free(samples);
}

/* ─── Test resample ─────────────────────────────────────────────────────── */

static void test_resample_identity(void) {
    printf("test_resample_identity:\n");

    /* Resample 24kHz to 24kHz should be identity */
    int n = 1000;
    float* input = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = (float)i / (float)n;
    }

    float* output = NULL;
    int out_len = 0;
    vv_status_t s = vv_audio_resample(input, 24000, n, &output, 24000, &out_len);
    TEST_ASSERT(s == VV_OK, "resample 24k->24k returns VV_OK");
    TEST_ASSERT(out_len == n, "output length matches input");

    if (output && out_len == n) {
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = fabsf(output[i] - input[i]);
            if (diff > max_diff) max_diff = diff;
        }
        TEST_ASSERT(max_diff < 1e-6f, "identity resample has zero error");
    }

    vv_free(input);
    if (output) vv_free(output);
}

static void test_resample_downsample(void) {
    printf("test_resample_downsample:\n");

    /* Resample 48kHz to 24kHz (2:1 ratio) */
    int n = 48000;  /* 1 second at 48kHz */
    float* input = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = sinf(2.0f * 3.14159f * 440.0f * (float)i / 48000.0f);
    }

    float* output = NULL;
    int out_len = 0;
    vv_status_t s = vv_audio_resample(input, 48000, n, &output, 24000, &out_len);
    TEST_ASSERT(s == VV_OK, "resample 48k->24k returns VV_OK");
    TEST_ASSERT(out_len > 20000 && out_len < 30000,
                "output length is reasonable (24k ± margin)");

    printf("  48kHz (%d samples) -> 24kHz (%d samples)\n", n, out_len);

    vv_free(input);
    if (output) vv_free(output);
}

/* ─── Test edge cases ───────────────────────────────────────────────────── */

static void test_normalize_silent(void) {
    printf("test_normalize_silent:\n");

    float samples[100] = {0};
    vv_status_t s = vv_audio_normalize(samples, 100, -25.0f, 1e-6f);
    TEST_ASSERT(s == VV_OK, "normalize silent audio returns VV_OK");

    /* Should remain silent */
    float max_val = 0.0f;
    for (int i = 0; i < 100; i++) {
        if (fabsf(samples[i]) > max_val) max_val = fabsf(samples[i]);
    }
    TEST_ASSERT(max_val < 1e-5f, "silent audio stays silent after normalize");
}

/*
 * The resampler before its weights were cached, verbatim: every output
 * computed its own windowed-sinc taps. The cached one must give its bits.
 */
static float ref_sinc(float x) {
    if (fabsf(x) < 1e-8f) return 1.0f;
    float px = (float)3.14159265358979323846 * x;
    return sinf(px) / px;
}

static float ref_blackman_harris(float n, float N) {
    float a0 = 0.35875f;
    float a1 = 0.48829f;
    float a2 = 0.14128f;
    float a3 = 0.01168f;
    float x = 2.0f * (float)3.14159265358979323846 * n / (N - 1.0f);
    return a0 - a1 * cosf(x) + a2 * cosf(2.0f * x) - a3 * cosf(3.0f * x);
}

static float ref_resample_one(const float* in, int in_len, int i, double ratio,
                              float cutoff, int half_w, float filter_width) {
    double src_pos = (double)i / ratio;
    int center = (int)src_pos;
    float frac = (float)(src_pos - (double)center);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int j = -half_w; j <= half_w; j++) {
        int idx = center + j;
        if (idx < 0 || idx >= in_len) continue;
        float t = (float)j - frac;
        float w = ref_sinc(t * cutoff) * cutoff;
        float wn = (float)(j - frac + half_w);
        w *= ref_blackman_harris(wn, filter_width + 1.0f);
        sum += in[idx] * w;
        weight_sum += w;
    }
    return weight_sum > 1e-8f ? sum / weight_sum : 0.0f;
}

static void test_resample_bits(void) {
    printf("test_resample_bits:\n");
    static const int rates[] = { 8000, 11025, 16000, 22050, 32000, 44100, 48000 };
    static const int lens[] = { 1, 17, 1000, 48017 };
    uint32_t rng = 12345u;
    int bad_cases = 0, cases = 0;
    for (size_t r = 0; r < sizeof(rates) / sizeof(rates[0]); r++) {
        for (size_t l = 0; l < sizeof(lens) / sizeof(lens[0]); l++) {
            const int n = lens[l];
            float* in = (float*)vv_alloc((size_t)n * sizeof(float));
            for (int i = 0; i < n; i++) {
                rng = rng * 1664525u + 1013904223u;
                in[i] = 0.3f * sinf(0.013f * (float)i) +
                        0.2f * ((float)(rng >> 8) / 16777216.0f - 0.5f);
            }
            float* out = NULL;
            int out_len = 0;
            if (vv_audio_resample(in, rates[r], n, &out, 24000, &out_len) != VV_OK) {
                bad_cases++;
                vv_free(in);
                continue;
            }
            const double ratio = 24000.0 / (double)rates[r];
            const float cutoff = ratio < 1.0 ? (float)ratio : 1.0f;
            int diff = (int)((double)n * ratio) + 1 != out_len;
            for (int i = 0; i < out_len && !diff; i++) {
                const float want = ref_resample_one(in, n, i, ratio, cutoff, 16, 32.0f);
                diff = memcmp(&want, &out[i], sizeof(float)) != 0;
            }
            /* The streaming resampler, fed in odd pieces, gives them too. */
            vv_resampler_t* rs = NULL;
            if (!diff && vv_resampler_create(rates[r], 24000, &rs) == VV_OK) {
                int got = 0, off = 0, piece = 1;
                while (off < n && !diff) {
                    const int k = off + piece > n ? n - off : piece;
                    const float* o = NULL;
                    size_t no = 0;
                    diff = vv_resampler_push(rs, in + off, (size_t)k, &o, &no) != VV_OK ||
                           (int)no > out_len - got ||
                           (no && memcmp(o, out + got, no * sizeof(float)) != 0);
                    got += (int)no;
                    off += k;
                    piece = piece * 3 + 1;
                }
                const float* o = NULL;
                size_t no = 0;
                if (!diff)
                    diff = vv_resampler_finish(rs, &o, &no) != VV_OK ||
                           got + (int)no != out_len ||
                           (no && memcmp(o, out + got, no * sizeof(float)) != 0);
                vv_resampler_free(rs);
            }
            if (diff) {
                bad_cases++;
                printf("  %d Hz x %d samples: differs from the per-sample weights\n",
                       rates[r], n);
            }
            cases++;
            vv_free(out);
            vv_free(in);
        }
    }
    TEST_ASSERT(bad_cases == 0,
                "cached tap weights give the per-sample resampler's bits, "
                "whole and streamed");
    printf("  %d rate/length cases\n", cases);
}

static void test_null_args(void) {
    printf("test_null_args:\n");

    TEST_ASSERT(vv_audio_normalize(NULL, 100, -25.0f, 1e-6f) == VV_ERR_NULL_PTR,
                "normalize(NULL) returns NULL_PTR");

    float* out = NULL;
    int out_len = 0;
    TEST_ASSERT(vv_audio_resample(NULL, 24000, 100, &out, 24000, &out_len)
                == VV_ERR_NULL_PTR,
                "resample(NULL) returns NULL_PTR");
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Audio Pipeline Tests ===\n\n");

    test_normalize();
    test_resample_identity();
    test_resample_downsample();
    test_resample_bits();
    test_normalize_silent();
    test_null_args();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
