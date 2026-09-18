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
#include <stdlib.h>
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

/*
 * One SemVer 2.0 numeric identifier: digits, no leading zero. Returns the
 * number of characters consumed and its value, or 0 when there is none.
 */
static int semver_num(const char* s, long* value) {
    int n = 0;
    while (s[n] >= '0' && s[n] <= '9') n++;
    if (n == 0 || (n > 1 && s[0] == '0')) return 0;
    if (value) *value = strtol(s, NULL, 10);
    return n;
}

/* Dot-separated identifiers from [0-9A-Za-z-], none empty. */
static int semver_idents(const char* s, int len) {
    if (len == 0) return 0;
    int run = 0;
    for (int i = 0; i < len; i++) {
        const char c = s[i];
        if (c == '.') {
            if (run == 0) return 0;
            run = 0;
        } else if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                   (c >= 'A' && c <= 'Z') || c == '-') {
            run++;
        } else {
            return 0;
        }
    }
    return run > 0;
}

/* Whole-string SemVer 2.0 check; fills in major.minor.patch. */
static int is_semver(const char* v, long xyz[3]) {
    const char* p = v;
    for (int i = 0; i < 3; i++) {
        const int n = semver_num(p, &xyz[i]);
        if (n == 0) return 0;
        p += n;
        if (i < 2) {
            if (*p != '.') return 0;
            p++;
        }
    }
    if (*p == '-') {
        const char* end = strchr(p, '+');
        const int len = end ? (int)(end - p - 1) : (int)strlen(p + 1);
        if (!semver_idents(p + 1, len)) return 0;
        p += 1 + len;
    }
    if (*p == '+') {
        if (!semver_idents(p + 1, (int)strlen(p + 1))) return 0;
        p += strlen(p);
    }
    return *p == '\0';
}

static void test_version(void) {
    printf("test_version:\n");

    char parts[32];
    snprintf(parts, sizeof(parts), "%d.%d.%d",
             VV_VERSION_MAJOR, VV_VERSION_MINOR, VV_VERSION_PATCH);
    TEST_ASSERT(strcmp(parts, VV_VERSION_STRING) == 0,
                "header MAJOR.MINOR.PATCH matches VV_VERSION_STRING");

    long xyz[3] = {0, 0, 0};
    const char* v = vv_version();
    printf("  vv_version() = %s, vv_build_ref() = %s\n", v, vv_build_ref());
    TEST_ASSERT(is_semver(v, xyz), "vv_version() is SemVer 2.0");

    /* A build is of the header's release line or of the one after it. */
    const long have = xyz[0] * 1000000L + xyz[1] * 1000L + xyz[2];
    const long want = VV_VERSION_MAJOR * 1000000L + VV_VERSION_MINOR * 1000L +
                      VV_VERSION_PATCH;
    TEST_ASSERT(have >= want, "vv_version() is not older than the header");

    TEST_ASSERT(vv_build_ref() && vv_build_ref()[0],
                "vv_build_ref() is not empty");

    long tmp[3];
    TEST_ASSERT(is_semver("1.0.0-rc.1.dev.3+g1a2b3c4.dirty", tmp) &&
                !is_semver("1.0", tmp) && !is_semver("01.0.0", tmp) &&
                !is_semver("1.0.0-", tmp) && !is_semver("1.0.0+a..b", tmp),
                "the SemVer check itself accepts and rejects correctly");
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
