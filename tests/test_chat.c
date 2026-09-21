/**
 * @file test_chat.c
 * @brief What /v1/chat/completions accepts, and the prompt it renders.
 *
 * No model, no socket: a request body in, a ChatML prompt and the sampling
 * settings out. The template is the contract with the checkpoint — a turn
 * that opens or closes differently is a different model as far as the
 * weights are concerned — so it is compared byte for byte.
 */

#include "vibevoice/server.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void ok_prompt(const char* name, const char* body, const char* want) {
    vv_chat_request_t r;
    const char* err = NULL;
    const vv_status_t s = vv_chat_request_parse(body, strlen(body), &r, &err);
    const int pass = (s == VV_OK) && r.prompt && strcmp(r.prompt, want) == 0;
    printf("  %-22s %s\n", name, pass ? "ok" : "FAIL");
    if (!pass) {
        printf("      status %d (%s)\n", (int)s, err ? err : "-");
        printf("      got    |%s|\n", r.prompt ? r.prompt : "(null)");
        printf("      wanted |%s|\n", want);
        failures++;
    }
    vv_chat_request_free(&r);
}

static void bad_request(const char* name, const char* body) {
    vv_chat_request_t r;
    const char* err = NULL;
    const vv_status_t s = vv_chat_request_parse(body, strlen(body), &r, &err);
    const int pass = (s != VV_OK) && err != NULL;
    printf("  %-22s %s%s\n", name, pass ? "ok" : "FAIL",
           pass ? "" : " (accepted)");
    if (!pass) failures++;
    vv_chat_request_free(&r);
}

int main(void) {
    printf("chat: the ChatML prompt\n");

    /* Without a system turn the model keeps the transcription instruction it
     * was fine-tuned on, so one is supplied. */
    ok_prompt("user only",
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n");

    ok_prompt("system kept",
        "{\"messages\":[{\"role\":\"system\",\"content\":\"be terse\"},"
        "{\"role\":\"user\",\"content\":\"hi\"}]}",
        "<|im_start|>system\nbe terse<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n");

    ok_prompt("a whole exchange",
        "{\"messages\":[{\"role\":\"system\",\"content\":\"s\"},"
        "{\"role\":\"user\",\"content\":\"a\"},"
        "{\"role\":\"assistant\",\"content\":\"b\"},"
        "{\"role\":\"user\",\"content\":\"c\"}]}",
        "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\na<|im_end|>\n"
        "<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n"
        "<|im_start|>assistant\n");

    /* Clients built for vision models send content as parts. */
    ok_prompt("content parts",
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"one \"},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}},"
        "{\"type\":\"text\",\"text\":\"two\"}]}]}",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\none two<|im_end|>\n"
        "<|im_start|>assistant\n");

    printf("chat: what is refused\n");
    bad_request("not JSON", "not json at all");
    bad_request("no messages", "{\"model\":\"m\"}");
    bad_request("messages empty", "{\"messages\":[]}");
    bad_request("n > 1", "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":2}");

    printf("chat: sampling settings\n");
    {
        vv_chat_request_t r;
        const char* err = NULL;
        const char* body =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
            "\"max_completion_tokens\":32,\"temperature\":0.7,\"top_p\":0.9,"
            "\"seed\":42,\"stream\":true,\"stop\":[\"\\nUser:\",\"END\"]}";
        const vv_status_t s = vv_chat_request_parse(body, strlen(body), &r, &err);
        const int pass = s == VV_OK && r.max_tokens == 32 &&
                         r.temperature > 0.69f && r.temperature < 0.71f &&
                         r.top_p > 0.89f && r.top_p < 0.91f &&
                         r.seed == 42 && r.stream && r.n_stop == 2 &&
                         strcmp(r.stop[0], "\nUser:") == 0 &&
                         strcmp(r.stop[1], "END") == 0;
        printf("  %-22s %s\n", "parsed", pass ? "ok" : "FAIL");
        if (!pass) failures++;
        vv_chat_request_free(&r);
    }
    {
        /* The old spelling still works, and greedy is the default. */
        vv_chat_request_t r;
        const char* body = "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
                           "\"max_tokens\":7}";
        const vv_status_t s = vv_chat_request_parse(body, strlen(body), &r, NULL);
        const int pass = s == VV_OK && r.max_tokens == 7 &&
                         r.temperature == 0.0f && r.top_p == 1.0f &&
                         !r.stream && r.n_stop == 0;
        printf("  %-22s %s\n", "defaults", pass ? "ok" : "FAIL");
        if (!pass) failures++;
        vv_chat_request_free(&r);
    }

    printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
