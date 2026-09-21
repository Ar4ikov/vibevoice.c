/**
 * @file chat_prompt.c
 * @brief An OpenAI chat request → the ChatML prompt this model reads.
 *
 * Kept apart from the endpoint so that the shape of a request — what a
 * client may send, what it turns into, and what is refused — can be checked
 * without a model, a socket or a GPU (`tests/test_chat.c`).
 */

#include "vibevoice/server.h"
#include "vibevoice/vibevoice.h"

#include "cJSON.h"

#include <stdlib.h>
#include <string.h>

/* ChatML, the template Qwen2 (and this fine-tune of it) was trained on. */
#define TURN_OPEN  "<|im_start|>"
#define TURN_CLOSE "<|im_end|>\n"

typedef struct buf {
    char*  s;
    size_t n, cap;
} buf_t;

static bool buf_add(buf_t* b, const char* s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 512;
        while (cap < b->n + n + 1) cap *= 2;
        char* p = (char*)vv_realloc(b->s, cap);
        if (!p) return false;
        b->s = p; b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = '\0';
    return true;
}

static bool buf_puts(buf_t* b, const char* s) { return buf_add(b, s, strlen(s)); }

/**
 * @brief The text of one message, which may be a string or content parts.
 *
 * Clients that talk to vision models send `content` as an array of parts;
 * the text ones are all this model can read, and an image part is dropped
 * rather than refused, because a client that sends one still wants an answer
 * to the text beside it.
 */
static bool append_content(buf_t* b, const cJSON* content) {
    if (cJSON_IsString(content)) return buf_puts(b, content->valuestring);
    if (!cJSON_IsArray(content)) return true;       /* null, or a number */

    const cJSON* part = NULL;
    cJSON_ArrayForEach(part, content) {
        const cJSON* text = cJSON_GetObjectItemCaseSensitive(part, "text");
        if (cJSON_IsString(text) && !buf_puts(b, text->valuestring))
            return false;
    }
    return true;
}

static char* dup_string(const char* s) {
    const size_t n = strlen(s) + 1;
    char* p = (char*)vv_alloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static float number_or(const cJSON* o, const char* key, float fallback) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, key);
    return cJSON_IsNumber(v) ? (float)v->valuedouble : fallback;
}

static int int_or(const cJSON* o, const char* key, int fallback) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, key);
    return cJSON_IsNumber(v) ? (int)v->valuedouble : fallback;
}

void vv_chat_request_free(vv_chat_request_t* r) {
    if (!r) return;
    vv_free(r->prompt);
    vv_free(r->model);
    for (int i = 0; i < r->n_stop; i++) vv_free(r->stop[i]);
    memset(r, 0, sizeof(*r));
}

vv_status_t vv_chat_request_parse(const char* body, size_t len,
                                  vv_chat_request_t* out, const char** err) {
    if (err) *err = NULL;
    if (!body || !out) return VV_ERR_NULL_PTR;
    memset(out, 0, sizeof(*out));
    out->max_tokens = 512;
    out->top_p = 1.0f;
    out->top_k = 64;

    cJSON* root = cJSON_ParseWithLength(body, len);
    if (!root) {
        if (err) *err = "body is not JSON";
        return VV_ERR_PARSE;
    }

    const cJSON* messages = cJSON_GetObjectItemCaseSensitive(root, "messages");
    if (!cJSON_IsArray(messages) || cJSON_GetArraySize(messages) == 0) {
        if (err) *err = "'messages' must be a non-empty array";
        cJSON_Delete(root);
        return VV_ERR_INVALID_ARG;
    }
    /*
     * One answer per request. `n` > 1 would run the whole prompt again per
     * choice on a slot another request is waiting for, so it is refused
     * rather than silently served as one.
     */
    const int n_choices = int_or(root, "n", 1);
    if (n_choices != 1) {
        if (err) *err = "'n' must be 1";
        cJSON_Delete(root);
        return VV_ERR_UNSUPPORTED;
    }

    buf_t b = {0};
    bool ok = true;
    bool have_system = false;
    const cJSON* m = NULL;
    cJSON_ArrayForEach(m, messages) {
        const cJSON* role = cJSON_GetObjectItemCaseSensitive(m, "role");
        const cJSON* content = cJSON_GetObjectItemCaseSensitive(m, "content");
        const char* r = cJSON_IsString(role) ? role->valuestring : "user";
        if (strcmp(r, "system") == 0) have_system = true;
        ok = ok && buf_puts(&b, TURN_OPEN) && buf_puts(&b, r) &&
             buf_puts(&b, "\n") && append_content(&b, content) &&
             buf_puts(&b, TURN_CLOSE);
        if (!ok) break;
    }
    /*
     * Without a system turn the model is left with the one it was fine-tuned
     * on, which tells it to answer in transcription JSON. Qwen2's own default
     * is the closest thing to "no instruction" it will listen to.
     */
    if (ok && !have_system) {
        buf_t head = {0};
        ok = buf_puts(&head, TURN_OPEN "system\nYou are a helpful assistant."
                             TURN_CLOSE) &&
             buf_add(&head, b.s ? b.s : "", b.n);
        if (ok) { vv_free(b.s); b = head; } else { vv_free(head.s); }
    }
    ok = ok && buf_puts(&b, TURN_OPEN "assistant\n");
    if (!ok) {
        vv_free(b.s);
        cJSON_Delete(root);
        if (err) *err = "out of memory";
        return VV_ERR_OUT_OF_MEMORY;
    }
    out->prompt = b.s;

    const cJSON* model = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (cJSON_IsString(model)) out->model = dup_string(model->valuestring);

    /* `max_completion_tokens` is the current spelling; `max_tokens` the old. */
    int max_tokens = int_or(root, "max_completion_tokens",
                            int_or(root, "max_tokens", 512));
    if (max_tokens <= 0) max_tokens = 512;
    out->max_tokens = max_tokens;
    out->temperature = number_or(root, "temperature", 0.0f);
    out->top_p = number_or(root, "top_p", 1.0f);
    out->top_k = int_or(root, "top_k", 64);
    out->seed = (uint64_t)int_or(root, "seed", 0);
    const cJSON* stream = cJSON_GetObjectItemCaseSensitive(root, "stream");
    out->stream = cJSON_IsTrue(stream);

    const cJSON* stop = cJSON_GetObjectItemCaseSensitive(root, "stop");
    if (cJSON_IsString(stop)) {
        out->stop[out->n_stop] = dup_string(stop->valuestring);
        if (out->stop[out->n_stop]) out->n_stop++;
    } else if (cJSON_IsArray(stop)) {
        const cJSON* it = NULL;
        cJSON_ArrayForEach(it, stop) {
            if (out->n_stop >= (int)(sizeof(out->stop) / sizeof(out->stop[0])))
                break;
            if (!cJSON_IsString(it)) continue;
            out->stop[out->n_stop] = dup_string(it->valuestring);
            if (out->stop[out->n_stop]) out->n_stop++;
        }
    }

    cJSON_Delete(root);
    return VV_OK;
}
