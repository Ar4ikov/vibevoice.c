/**
 * @file test_nf4.c
 * @brief Unit tests for NF4 dequantization (CPU reference).
 */

#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
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

/* External declarations */
extern const float VV_NF4_TABLE[16];
extern vv_status_t vv_dequant_nf4_cpu(const uint8_t* packed,
                                        const float* scales,
                                        float* output,
                                        int n_elements,
                                        int block_size);

/* ─── Tests ─────────────────────────────────────────────────────────────── */

static void test_nf4_table(void) {
    printf("test_nf4_table:\n");

    /* Table should have 16 entries */
    TEST_ASSERT(VV_NF4_TABLE[0] == -1.0f, "NF4[0] == -1.0");
    TEST_ASSERT(VV_NF4_TABLE[7] == 0.0f, "NF4[7] == 0.0");
    TEST_ASSERT(VV_NF4_TABLE[15] == 1.0f, "NF4[15] == 1.0");

    /* Verify monotonicity */
    bool monotonic = true;
    for (int i = 1; i < 16; i++) {
        if (VV_NF4_TABLE[i] <= VV_NF4_TABLE[i-1]) {
            monotonic = false;
            break;
        }
    }
    TEST_ASSERT(monotonic, "NF4 table is strictly monotonically increasing");
}

static void test_dequant_basic(void) {
    printf("test_dequant_basic:\n");

    /* Create packed data: 4 elements (2 bytes) */
    uint8_t packed[2] = {
        0x0F,  /* high=0 (NF4=-1.0), low=F (NF4=1.0) */
        0x78,  /* high=7 (NF4=0.0), low=8 (NF4=0.0796) */
    };
    float scales[1] = {2.0f};  /* block_size=64, so all use same scale */
    float output[4] = {0};

    vv_status_t s = vv_dequant_nf4_cpu(packed, scales, output, 4, 64);
    TEST_ASSERT(s == VV_OK, "dequant returns VV_OK");

    /* Expected: NF4[0]*2 = -2.0, NF4[15]*2 = 2.0,
     *           NF4[7]*2 = 0.0,  NF4[8]*2 = 0.159 */
    TEST_ASSERT(fabsf(output[0] - (-2.0f)) < 1e-4f,
                "element 0: NF4[0] * 2.0 = -2.0");
    TEST_ASSERT(fabsf(output[1] - 2.0f) < 1e-4f,
                "element 1: NF4[15] * 2.0 = 2.0");
    TEST_ASSERT(fabsf(output[2] - 0.0f) < 1e-4f,
                "element 2: NF4[7] * 2.0 = 0.0");
    TEST_ASSERT(fabsf(output[3] - 0.1592f) < 1e-3f,
                "element 3: NF4[8] * 2.0 ≈ 0.159");

    printf("  output: [%.4f, %.4f, %.4f, %.4f]\n",
           output[0], output[1], output[2], output[3]);
}

static void test_dequant_block_boundary(void) {
    printf("test_dequant_block_boundary:\n");

    /* 128 elements = 2 blocks of 64 = 64 packed bytes */
    int n = 128;
    int n_bytes = n / 2;
    uint8_t* packed = (uint8_t*)vv_alloc(n_bytes);
    float scales[2] = {1.0f, 3.0f};  /* Different scale per block */
    float* output = (float*)vv_alloc(n * sizeof(float));

    memset(packed, 0xFF, n_bytes);  /* All nibbles = 15 (NF4 = 1.0) */

    vv_status_t s = vv_dequant_nf4_cpu(packed, scales, output, n, 64);
    TEST_ASSERT(s == VV_OK, "dequant block boundary returns VV_OK");

    /* First 64 elements: 1.0 * 1.0 = 1.0 */
    TEST_ASSERT(fabsf(output[0] - 1.0f) < 1e-4f,
                "block 0 element: 1.0 * 1.0 = 1.0");
    TEST_ASSERT(fabsf(output[63] - 1.0f) < 1e-4f,
                "block 0 last: 1.0 * 1.0 = 1.0");

    /* Next 64 elements: 1.0 * 3.0 = 3.0 */
    TEST_ASSERT(fabsf(output[64] - 3.0f) < 1e-4f,
                "block 1 element: 1.0 * 3.0 = 3.0");
    TEST_ASSERT(fabsf(output[127] - 3.0f) < 1e-4f,
                "block 1 last: 1.0 * 3.0 = 3.0");

    vv_free(packed);
    vv_free(output);
}

static void test_dequant_null_args(void) {
    printf("test_dequant_null_args:\n");

    float output[4];
    float scales[1] = {1.0f};
    uint8_t packed[2] = {0};

    TEST_ASSERT(vv_dequant_nf4_cpu(NULL, scales, output, 4, 64)
                == VV_ERR_NULL_PTR,
                "NULL packed returns NULL_PTR");
    TEST_ASSERT(vv_dequant_nf4_cpu(packed, NULL, output, 4, 64)
                == VV_ERR_NULL_PTR,
                "NULL scales returns NULL_PTR");
    TEST_ASSERT(vv_dequant_nf4_cpu(packed, scales, NULL, 4, 64)
                == VV_ERR_NULL_PTR,
                "NULL output returns NULL_PTR");
    TEST_ASSERT(vv_dequant_nf4_cpu(packed, scales, output, 0, 64)
                == VV_ERR_INVALID_ARG,
                "n_elements=0 returns INVALID_ARG");
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== NF4 Dequantization Tests ===\n\n");

    test_nf4_table();
    test_dequant_basic();
    test_dequant_block_boundary();
    test_dequant_null_args();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
