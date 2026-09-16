/**
 * @file test_conv_vae.c
 * @brief Conv-VAE tokenizer encoder: lifecycle, config, and latent sampling.
 *
 * The encoder itself needs weights to say anything interesting, so what runs
 * here is the init/free lifecycle, config parsing, and the acoustic latent
 * draw, which is self-contained and worth pinning down exactly.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/tokenizer_encoder.h"

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

static void test_conv_vae_init_acoustic(void) {
    printf("test_conv_vae_init_acoustic:\n");

    vv_acoustic_tokenizer_config_t config;
    memset(&config, 0, sizeof(config));
    config.channels = 1;
    config.causal = true;
    config.vae_dim = 64;
    config.fix_std = 0.5f;
    config.encoder_n_filters = 32;
    int ratios[] = {8, 5, 5, 4, 2, 2};
    memcpy(config.encoder_ratios, ratios, sizeof(ratios));
    config.n_ratios = 6;
    int depths[] = {3, 3, 3, 3, 3, 3, 8};
    memcpy(config.encoder_depths, depths, sizeof(depths));
    config.n_depths = 7;

    vv_conv_vae_encoder_t* enc = NULL;
    vv_status_t s = vv_conv_vae_init(NULL, 0, &config, true, &enc);
    TEST_ASSERT(s == VV_OK, "acoustic init returns VV_OK");
    TEST_ASSERT(enc != NULL, "encoder pointer is non-NULL");
    if (!enc) return;
    TEST_ASSERT(enc->vae_dim == 64, "vae_dim == 64");
    TEST_ASSERT(enc->gaussian == true, "gaussian == true for acoustic");

    s = vv_conv_vae_free(enc);
    TEST_ASSERT(s == VV_OK, "free returns VV_OK");
}

/** @brief A config with no stages must be rejected, not half-built. */
static void test_conv_vae_rejects_empty_config(void) {
    printf("test_conv_vae_rejects_empty_config:\n");
    vv_semantic_tokenizer_config_t config;
    memset(&config, 0, sizeof(config));
    config.vae_dim = 128;
    vv_conv_vae_encoder_t* enc = NULL;
    vv_status_t s = vv_conv_vae_init(NULL, 0, &config, false, &enc);
    TEST_ASSERT(s == VV_ERR_INVALID_ARG, "zero-stage config is rejected");
    TEST_ASSERT(enc == NULL, "no encoder is returned");
}

static void test_conv_vae_init_semantic(void) {
    printf("test_conv_vae_init_semantic:\n");

    vv_semantic_tokenizer_config_t config;
    memset(&config, 0, sizeof(config));
    config.channels = 1;
    config.causal = true;
    config.vae_dim = 128;
    config.encoder_n_filters = 32;
    int ratios[] = {8, 5, 5, 4, 2, 2};
    memcpy(config.encoder_ratios, ratios, sizeof(ratios));
    config.n_ratios = 6;
    int depths[] = {3, 3, 3, 3, 3, 3, 8};
    memcpy(config.encoder_depths, depths, sizeof(depths));
    config.n_depths = 7;

    vv_conv_vae_encoder_t* enc = NULL;
    vv_status_t s = vv_conv_vae_init(NULL, 0, &config, false, &enc);
    TEST_ASSERT(s == VV_OK, "semantic init returns VV_OK");
    TEST_ASSERT(enc != NULL, "encoder pointer is non-NULL");
    if (!enc) return;
    TEST_ASSERT(enc->vae_dim == 128, "vae_dim == 128");
    TEST_ASSERT(enc->gaussian == false, "gaussian == false for semantic");
    TEST_ASSERT(enc->n_stages == 7, "7 encoder stages");

    s = vv_conv_vae_free(enc);
    TEST_ASSERT(s == VV_OK, "free returns VV_OK");
}

/**
 * @brief The acoustic latent draw: deterministic per seed, right spread.
 *
 * The reference perturbs the mean before the connector sees it, so what has to
 * hold is that a seed pins the draw exactly, that the mode leaves the mean
 * alone, and that "fix" really does add noise of `fix_std`. The per-clip scale
 * of "gaussian" is itself random, so only its magnitude is checked.
 */
static void test_acoustic_sampling(void) {
    printf("test_acoustic_sampling:\n");

    enum { FRAMES = 512, DIM = 64, N = FRAMES * DIM };
    static float a[N], b[N], base[N];
    for (int i = 0; i < N; i++) base[i] = (float)((i % 17) - 8) * 0.25f;

    /* The mode leaves the mean where it is. */
    memcpy(a, base, sizeof(base));
    vv_status_t s = vv_acoustic_sample(a, FRAMES, DIM, 0.5f,
                                       VV_ACOUSTIC_MODE, 1234, NULL);
    TEST_ASSERT(s == VV_OK, "mode returns VV_OK");
    TEST_ASSERT(memcmp(a, base, sizeof(base)) == 0, "mode leaves the mean alone");

    /* One seed, one latent. */
    float scale_a = 0.0f, scale_b = 0.0f;
    memcpy(a, base, sizeof(base));
    memcpy(b, base, sizeof(base));
    vv_acoustic_sample(a, FRAMES, DIM, 0.5f, VV_ACOUSTIC_GAUSSIAN, 7, &scale_a);
    vv_acoustic_sample(b, FRAMES, DIM, 0.5f, VV_ACOUSTIC_GAUSSIAN, 7, &scale_b);
    TEST_ASSERT(memcmp(a, b, sizeof(base)) == 0, "same seed, same draw");
    TEST_ASSERT(scale_a == scale_b, "same seed, same scale");
    TEST_ASSERT(memcmp(a, base, sizeof(base)) != 0, "gaussian moves the mean");

    /* A different seed is a different latent. */
    memcpy(b, base, sizeof(base));
    vv_acoustic_sample(b, FRAMES, DIM, 0.5f, VV_ACOUSTIC_GAUSSIAN, 8, NULL);
    TEST_ASSERT(memcmp(a, b, sizeof(base)) != 0, "another seed, another draw");

    /* "fix" adds exactly fix_std of noise, elementwise. */
    memcpy(a, base, sizeof(base));
    vv_acoustic_sample(a, FRAMES, DIM, 0.5f, VV_ACOUSTIC_FIX, 42, &scale_a);
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < N; i++) {
        const double d = (double)a[i] - (double)base[i];
        sum += d;
        sq += d * d;
    }
    const double mean = sum / N;
    const double rms = sqrt(sq / N);
    printf("  fix: noise mean %.4f, rms %.4f (want 0 and %.2f)\n",
           mean, rms, 0.5);
    TEST_ASSERT(scale_a == 0.5f, "fix uses fix_std as the scale");
    TEST_ASSERT(fabs(mean) < 0.02, "fix noise is centred");
    TEST_ASSERT(fabs(rms - 0.5) < 0.02, "fix noise has the right spread");

    /* The gaussian scale is drawn from N(0, fix_std / 0.8); it is one number
     * per clip, so only the plausible range is worth asserting. */
    int nonzero = 0;
    double scale_sq = 0.0;
    for (unsigned seed = 1; seed <= 64; seed++) {
        float sc = 0.0f;
        memcpy(a, base, sizeof(base));
        vv_acoustic_sample(a, 1, DIM, 0.5f, VV_ACOUSTIC_GAUSSIAN, seed, &sc);
        if (sc != 0.0f) nonzero++;
        scale_sq += (double)sc * (double)sc;
    }
    const double scale_rms = sqrt(scale_sq / 64.0);
    printf("  gaussian: scale rms over 64 seeds %.4f (want ~%.3f)\n",
           scale_rms, 0.5 / 0.8);
    TEST_ASSERT(nonzero == 64, "every seed draws a scale");
    TEST_ASSERT(scale_rms > 0.3 && scale_rms < 1.1,
                "gaussian scale spread is in range");

    /* Names round-trip, and nonsense is rejected rather than guessed. */
    TEST_ASSERT(vv_acoustic_sampling_parse("gaussian") == VV_ACOUSTIC_GAUSSIAN,
                "parse gaussian");
    TEST_ASSERT(vv_acoustic_sampling_parse("fix") == VV_ACOUSTIC_FIX,
                "parse fix");
    TEST_ASSERT(vv_acoustic_sampling_parse("none") == VV_ACOUSTIC_MODE,
                "parse none as the mode");
    TEST_ASSERT(vv_acoustic_sampling_parse("bogus")
                    == VV_ACOUSTIC_SAMPLING_COUNT, "reject unknown names");
    TEST_ASSERT(strcmp(vv_acoustic_sampling_name(VV_ACOUSTIC_GAUSSIAN),
                       "gaussian") == 0, "name gaussian");
}

int main(void) {
    printf("=== Conv-VAE Encoder Tests ===\n\n");

    test_conv_vae_init_acoustic();
    test_conv_vae_init_semantic();
    test_conv_vae_rejects_empty_config();
    test_acoustic_sampling();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
