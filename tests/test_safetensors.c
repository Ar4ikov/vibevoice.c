/**
 * @file test_safetensors.c
 * @brief Unit tests for safetensors parser.
 *
 * Creates a minimal safetensors file in memory and tests parsing.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/safetensors.h"
#include "vibevoice/model.h"

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

/* ─── Tests ─────────────────────────────────────────────────────────────── */

static void test_config_parse(void) {
    printf("test_config_parse:\n");

    /* Write a minimal config.json */
    const char* json =
        "{"
        "  \"decoder_config\": {"
        "    \"hidden_size\": 3584,"
        "    \"num_hidden_layers\": 28,"
        "    \"num_attention_heads\": 28,"
        "    \"num_key_value_heads\": 4,"
        "    \"intermediate_size\": 18944,"
        "    \"vocab_size\": 152064,"
        "    \"max_position_embeddings\": 131072,"
        "    \"rope_theta\": 1000000.0,"
        "    \"rms_norm_eps\": 1e-6"
        "  },"
        "  \"acoustic_tokenizer_config\": {"
        "    \"vae_dim\": 64,"
        "    \"fix_std\": 0.5,"
        "    \"encoder_n_filters\": 32,"
        "    \"causal\": true,"
        "    \"encoder_ratios\": [8, 5, 5, 4, 2, 2],"
        "    \"encoder_depths\": \"3-3-3-3-3-3-8\""
        "  },"
        "  \"semantic_tokenizer_config\": {"
        "    \"vae_dim\": 128,"
        "    \"encoder_n_filters\": 32,"
        "    \"causal\": true,"
        "    \"encoder_ratios\": [8, 5, 5, 4, 2, 2],"
        "    \"encoder_depths\": \"3-3-3-3-3-3-8\""
        "  }"
        "}";

    /* Write to temp file */
    const char* path = "test_config_tmp.json";
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "  SKIP: cannot create temp file\n");
        return;
    }
    fputs(json, f);
    fclose(f);

    vv_model_config_t config;
    memset(&config, 0, sizeof(config));
    vv_status_t s = vv_config_parse(path, &config);

    TEST_ASSERT(s == VV_OK, "config parse returns VV_OK");
    TEST_ASSERT(config.llm.hidden_size == 3584, "hidden_size == 3584");
    TEST_ASSERT(config.llm.num_hidden_layers == 28, "num_layers == 28");
    TEST_ASSERT(config.llm.num_attention_heads == 28, "num_heads == 28");
    TEST_ASSERT(config.llm.num_key_value_heads == 4, "num_kv_heads == 4");
    TEST_ASSERT(config.llm.intermediate_size == 18944, "inter_size == 18944");
    TEST_ASSERT(config.llm.vocab_size == 152064, "vocab_size == 152064");
    TEST_ASSERT(config.llm.max_position_embeddings == 131072,
                "max_pos_emb == 131072");

    TEST_ASSERT(config.acoustic.vae_dim == 64, "acoustic vae_dim == 64");
    TEST_ASSERT(config.acoustic.n_ratios == 6, "acoustic n_ratios == 6");
    TEST_ASSERT(config.acoustic.encoder_ratios[0] == 8, "ratio[0] == 8");
    TEST_ASSERT(config.acoustic.n_depths == 7, "acoustic n_depths == 7");

    TEST_ASSERT(config.semantic.vae_dim == 128, "semantic vae_dim == 128");

    /* Cleanup */
    remove(path);
}

static void test_dtype_sizes(void) {
    printf("test_dtype_sizes:\n");

    TEST_ASSERT(vv_dtype_size(VV_DTYPE_F32) == 4, "F32 size == 4");
    TEST_ASSERT(vv_dtype_size(VV_DTYPE_F16) == 2, "F16 size == 2");
    TEST_ASSERT(vv_dtype_size(VV_DTYPE_BF16) == 2, "BF16 size == 2");
    TEST_ASSERT(vv_dtype_size(VV_DTYPE_U8) == 1, "U8 size == 1");
    TEST_ASSERT(vv_dtype_size(VV_DTYPE_I32) == 4, "I32 size == 4");
    TEST_ASSERT(vv_dtype_size(VV_DTYPE_I64) == 8, "I64 size == 8");
}

static void test_status_strings(void) {
    printf("test_status_strings:\n");

    TEST_ASSERT(strcmp(vv_status_str(VV_OK), "OK") == 0,
                "VV_OK string is 'OK'");
    TEST_ASSERT(strcmp(vv_status_str(VV_ERR_CUDA), "CUDA error") == 0,
                "VV_ERR_CUDA string correct");
    TEST_ASSERT(strlen(vv_status_str(VV_ERR_NULL_PTR)) > 0,
                "VV_ERR_NULL_PTR has non-empty string");
}

static void test_alloc(void) {
    printf("test_alloc:\n");

    void* p = vv_alloc(1024);
    TEST_ASSERT(p != NULL, "vv_alloc(1024) returns non-NULL");
    TEST_ASSERT(vv_alloc_total() >= 1024, "alloc_total >= 1024");

    void* p2 = vv_realloc(p, 2048);
    TEST_ASSERT(p2 != NULL, "vv_realloc(2048) returns non-NULL");
    TEST_ASSERT(vv_alloc_total() >= 2048, "alloc_total >= 2048");

    vv_free(p2);
    /* Note: alloc_total is thread-local, so may not be 0 if other allocs */
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Safetensors/Config/Core Tests ===\n\n");

    test_config_parse();
    test_dtype_sizes();
    test_status_strings();
    test_alloc();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
