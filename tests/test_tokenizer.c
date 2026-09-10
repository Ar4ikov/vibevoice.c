/**
 * @file test_tokenizer.c
 * @brief Golden tests for the byte-level BPE tokenizer.
 *
 * The ASR prompt has to tokenise to exactly the same ids the HuggingFace
 * processor produces — a single extra or missing token shifts the audio
 * placeholders and the model answers from the wrong context. The expected
 * ids below were captured from
 * VibeVoiceASRProcessor(tokenizer=VibeVoiceASRTextTokenizerFast) on an
 * 11.00-second clip.
 *
 * Needs tokenizer.json, so the model directory is taken from VV_TEST_MODEL
 * (or ./model_hf). Without it the test reports "skipped" and passes.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/text_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do {                                    \
    tests_run++;                                                       \
    if (!(cond)) { fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
    else { tests_passed++; fprintf(stdout, "  PASS: %s\n", msg); }     \
} while (0)

/* "<|im_start|>system\nYou are a helpful assistant that transcribes audio
 *  input into text output in JSON format.<|im_end|>\n<|im_start|>user\n" */
static const int32_t EXPECT_PREFIX[] = {
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 429, 1356, 55136,
    7699, 1946, 1119, 1467, 2550, 304, 4718, 3561, 13, 151645, 198,
    151644, 872, 198
};

/* "\nThis is a 11.00 seconds audio, please transcribe it with these keys:
 *  Start time, End time, Speaker ID, Content<|im_end|>\n" */
static const int32_t EXPECT_SUFFIX[] = {
    198, 1986, 374, 264, 220, 16, 16, 13, 15, 15, 6486, 7699, 11, 4486,
    1356, 3114, 432, 448, 1493, 6894, 25, 5145, 882, 11, 3972, 882, 11,
    29073, 3034, 11, 8883, 151645, 198
};

static void check_ids(const char* what, const vv_tokenizer_t* tok,
                      const char* text,
                      const int32_t* expect, int n_expect) {
    int32_t* ids = NULL;
    int n = 0;
    vv_status_t s = vv_tokenizer_encode(tok, text, &ids, &n);
    if (s != VV_OK || !ids) {
        tests_run++;
        fprintf(stderr, "  FAIL: %s encode failed\n", what);
        return;
    }

    bool ok = (n == n_expect);
    if (ok) {
        for (int i = 0; i < n; i++)
            if (ids[i] != expect[i]) { ok = false; break; }
    }
    if (!ok) {
        fprintf(stderr, "  got %d ids (expected %d):", n, n_expect);
        for (int i = 0; i < n && i < 48; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
    }
    TEST_ASSERT(ok, what);
    vv_free(ids);
}

static void test_roundtrip(const vv_tokenizer_t* tok) {
    static const char* SAMPLES[] = {
        "Hello, world!",
        "  leading and trailing  ",
        "don't split contractions",
        "numbers 1234 and 11.00 seconds",
        "unicode: \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xe4\xbd\xa0\xe5\xa5\xbd",
        "line\nbreaks\n\nand\ttabs",
    };
    for (size_t i = 0; i < sizeof(SAMPLES) / sizeof(SAMPLES[0]); i++) {
        int32_t* ids = NULL;
        int n = 0;
        char* back = NULL;
        if (vv_tokenizer_encode(tok, SAMPLES[i], &ids, &n) != VV_OK) continue;
        vv_tokenizer_decode(tok, ids, n, &back);
        bool ok = back && strcmp(back, SAMPLES[i]) == 0;
        if (!ok) fprintf(stderr, "  round-trip got %s\n", back ? back : "(null)");
        TEST_ASSERT(ok, SAMPLES[i]);
        vv_free(ids);
        if (back) vv_free(back);
    }
}

int main(void) {
    printf("=== Tokenizer Tests ===\n\n");

    const char* dir = getenv("VV_TEST_MODEL");
    if (!dir || !dir[0]) dir = "model_hf";

    vv_tokenizer_t* tok = NULL;
    if (vv_tokenizer_load(dir, &tok) != VV_OK || !tok) {
        printf("  SKIP: no tokenizer.json under '%s' "
               "(set VV_TEST_MODEL to run)\n", dir);
        return 0;
    }

    printf("prompt prefix:\n");
    check_ids("system + user header tokenises to the reference ids", tok,
              "<|im_start|>system\n"
              "You are a helpful assistant that transcribes audio input into "
              "text output in JSON format.<|im_end|>\n"
              "<|im_start|>user\n",
              EXPECT_PREFIX,
              (int)(sizeof(EXPECT_PREFIX) / sizeof(EXPECT_PREFIX[0])));

    printf("prompt suffix:\n");
    check_ids("instruction tokenises to the reference ids", tok,
              "\nThis is a 11.00 seconds audio, please transcribe it with "
              "these keys: Start time, End time, Speaker ID, Content"
              "<|im_end|>\n",
              EXPECT_SUFFIX,
              (int)(sizeof(EXPECT_SUFFIX) / sizeof(EXPECT_SUFFIX[0])));

    printf("special tokens:\n");
    TEST_ASSERT(vv_tokenizer_special_id(tok, "<|im_start|>") == 151644,
                "<|im_start|> == 151644");
    TEST_ASSERT(vv_tokenizer_special_id(tok, "<|object_ref_start|>") == 151646,
                "speech_start == 151646");
    TEST_ASSERT(vv_tokenizer_special_id(tok, "<|object_ref_end|>") == 151647,
                "speech_end == 151647");
    TEST_ASSERT(vv_tokenizer_special_id(tok, "<|box_start|>") == 151648,
                "speech_pad == 151648");

    printf("round-trip:\n");
    test_roundtrip(tok);

    vv_tokenizer_free(tok);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
