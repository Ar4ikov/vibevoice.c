/**
 * @file test_nf4.c
 * @brief Unit tests for NF4 dequantization (CPU reference).
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/quant.h"

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

/**
 * @brief The nearest level as the first version of the load-time quantizer
 *        found it: binary search to a bracket, then its midpoint, ties low.
 */
static int nf4_nearest_bsearch(float x) {
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (x > VV_NF4_TABLE[mid]) lo = mid; else hi = mid;
    }
    const float m = 0.5f * (VV_NF4_TABLE[lo] + VV_NF4_TABLE[hi]);
    return (x > m) ? hi : lo;
}

/**
 * vv_nf4_quantize picks levels by counting midpoints below x. It must give
 * the same code as the binary search it replaced on every value, the exact
 * midpoints (ties) and their neighbours included.
 */
static void test_quantize_nearest(void) {
    printf("test_quantize_nearest:\n");

    enum { K = 64, N = 256 };
    float* w = (float*)vv_alloc((size_t)N * K * sizeof(float));
    uint8_t* packed = (uint8_t*)vv_alloc((size_t)N * K / 2);
    uint16_t* scales = (uint16_t*)vv_alloc((size_t)N * sizeof(uint16_t));
    if (!w || !packed || !scales) {
        TEST_ASSERT(0, "allocation");
        vv_free(w); vv_free(packed); vv_free(scales);
        return;
    }

    /* Element 0 of every block is 1.0, so the block's scale is exactly 1
       and x * (1 / scale) is x itself. */
    int n = 0;
    uint32_t lcg = 12345u;
    for (int r = 0; r < N; r++) {
        w[(size_t)r * K] = 1.0f;
        for (int k = 1; k < K; k++) {
            float v;
            if (n < 15 * 3) {
                const int j = n / 3, d = n % 3;
                const float m = 0.5f * (VV_NF4_TABLE[j] + VV_NF4_TABLE[j + 1]);
                v = d == 0 ? m : d == 1 ? nextafterf(m, -2.0f)
                                        : nextafterf(m, 2.0f);
            } else if (n < 15 * 3 + 16) {
                v = VV_NF4_TABLE[n - 45];
            } else {
                lcg = lcg * 1664525u + 1013904223u;
                v = ((float)(lcg >> 8) / 16777216.0f) * 2.0f - 1.0f;
            }
            w[(size_t)r * K + k] = v;
            n++;
        }
    }

    const vv_status_t s = vv_nf4_quantize(w, N, K, packed, scales);
    TEST_ASSERT(s == VV_OK, "vv_nf4_quantize succeeds");

    int bad = 0, scale_ok = 1;
    for (int r = 0; r < N; r++) {
        if (scales[r] != 0x3C00) scale_ok = 0;
        for (int k = 0; k < K; k++) {
            const uint8_t b = packed[((size_t)r * K + k) / 2];
            const int code = (k & 1) ? (b & 15) : (b >> 4);
            if (code != nf4_nearest_bsearch(w[(size_t)r * K + k])) bad++;
        }
    }
    TEST_ASSERT(scale_ok, "blocks with max 1.0 get scale 1.0 (0x3C00)");
    TEST_ASSERT(bad == 0,
                "codes match the binary search on 16k values, ties included");

    vv_free(w); vv_free(packed); vv_free(scales);
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
    test_quantize_nearest();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
