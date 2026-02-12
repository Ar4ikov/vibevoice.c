/**
 * @file test_conv_vae.c
 * @brief Placeholder tests for Conv-VAE tokenizer encoder.
 *
 * These tests will be meaningful once weights are loaded.
 * For now, tests the init/free lifecycle and config parsing.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/tokenizer_encoder.h"

#include <stdio.h>
#include <string.h>

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
    TEST_ASSERT(enc->vae_dim == 64, "vae_dim == 64");
    TEST_ASSERT(enc->gaussian == true, "gaussian == true for acoustic");

    s = vv_conv_vae_free(enc);
    TEST_ASSERT(s == VV_OK, "free returns VV_OK");
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

    vv_conv_vae_encoder_t* enc = NULL;
    vv_status_t s = vv_conv_vae_init(NULL, 0, &config, false, &enc);
    TEST_ASSERT(s == VV_OK, "semantic init returns VV_OK");
    TEST_ASSERT(enc != NULL, "encoder pointer is non-NULL");
    TEST_ASSERT(enc->vae_dim == 128, "vae_dim == 128");
    TEST_ASSERT(enc->gaussian == false, "gaussian == false for semantic");

    s = vv_conv_vae_free(enc);
    TEST_ASSERT(s == VV_OK, "free returns VV_OK");
}

int main(void) {
    printf("=== Conv-VAE Encoder Tests ===\n\n");

    test_conv_vae_init_acoustic();
    test_conv_vae_init_semantic();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
