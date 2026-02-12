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
    test_normalize_silent();
    test_null_args();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
