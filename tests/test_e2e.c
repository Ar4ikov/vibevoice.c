/**
 * @file test_e2e.c
 * @brief End-to-end test: placeholder until model weights are available.
 *
 * This test validates the full pipeline can be initialized and torn down
 * without crashing, even without actual model weights.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

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

static void test_version(void) {
    printf("test_version:\n");
    TEST_ASSERT(VV_VERSION_MAJOR == 0, "version major == 0");
    TEST_ASSERT(VV_VERSION_MINOR == 1, "version minor == 1");
    TEST_ASSERT(strcmp(VV_VERSION_STRING, "0.1.0") == 0,
                "version string == '0.1.0'");
}

static void test_audio_pipeline_synthetic(void) {
    printf("test_audio_pipeline_synthetic:\n");

    /* Create synthetic 1s audio at 48kHz */
    int sr = 48000;
    int n = sr;
    float* audio = (float*)vv_alloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        /* 440Hz sine + some noise */
        audio[i] = 0.3f * sinf(2.0f * 3.14159f * 440.0f * (float)i / (float)sr);
    }

    /* Resample to 24kHz */
    float* resampled = NULL;
    int resampled_len = 0;
    vv_status_t s = vv_audio_resample(audio, sr, n, &resampled, 24000,
                                       &resampled_len);
    TEST_ASSERT(s == VV_OK, "resample 48k->24k OK");
    TEST_ASSERT(resampled_len > 20000, "resampled has > 20k samples");

    /* Normalize */
    if (resampled) {
        s = vv_audio_normalize(resampled, resampled_len, -25.0f, 1e-6f);
        TEST_ASSERT(s == VV_OK, "normalize OK");

        /* Verify RMS is close to -25 dBFS */
        double sum_sq = 0;
        for (int i = 0; i < resampled_len; i++) {
            sum_sq += (double)resampled[i] * (double)resampled[i];
        }
        double rms = sqrt(sum_sq / resampled_len);
        double db = 20.0 * log10(rms + 1e-10);
        TEST_ASSERT(fabs(db - (-25.0)) < 1.0,
                    "normalized dBFS within 1dB of -25");
        printf("  Final dBFS: %.2f\n", db);

        vv_free(resampled);
    }

    vv_free(audio);
}

static void test_memory_tracking(void) {
    printf("test_memory_tracking:\n");

    size_t before = vv_alloc_total();
    void* p1 = vv_alloc(4096);
    void* p2 = vv_alloc(8192);
    size_t after = vv_alloc_total();

    TEST_ASSERT(after >= before + 4096 + 8192,
                "alloc_total increased by at least 12288");

    vv_free(p1);
    vv_free(p2);

    size_t final = vv_alloc_total();
    TEST_ASSERT(final <= before + 16,  /* small rounding tolerance */
                "alloc_total returned to baseline after free");
}

int main(void) {
    printf("=== End-to-End Tests ===\n\n");

    test_version();
    test_audio_pipeline_synthetic();
    test_memory_tracking();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
