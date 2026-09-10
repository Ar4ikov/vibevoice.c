/**
 * @file postprocess.c
 * @brief Post-processing: generated token stream → structured transcription.
 *
 * VibeVoice-ASR is prompted to answer in JSON, so the assistant turn looks
 * like
 *
 *   <|im_start|>assistant
 *   [{"Start":0.0,"End":11.1,"Speaker":0,"Content":"..."}, ...]
 *
 * The keys mirror the ones requested in the user turn ("Start time",
 * "End time", "Speaker ID", "Content"), and the model abbreviates them, so
 * several spellings are accepted — the same set the reference processor's
 * post_process_transcription maps.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static char* dup_cstr(const char* s, size_t n) {
    char* p = (char*)vv_alloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/** @brief First member found among a NULL-terminated list of key spellings. */
static cJSON* pick(cJSON* obj, const char* const* keys) {
    for (int i = 0; keys[i]; i++) {
        cJSON* v = cJSON_GetObjectItemCaseSensitive(obj, keys[i]);
        if (v) return v;
    }
    return NULL;
}

static float as_float(cJSON* v, float dflt) {
    if (!v) return dflt;
    if (cJSON_IsNumber(v)) return (float)v->valuedouble;
    if (cJSON_IsString(v) && v->valuestring) return (float)atof(v->valuestring);
    return dflt;
}

/**
 * @brief Locate the JSON array/object inside the model's answer.
 *
 * Handles a bare array, a ```json fenced block, and a truncated tail (when
 * generation hit the token budget mid-array) by closing the brackets that
 * are still open.
 */
static char* extract_json(const char* text) {
    if (!text) return NULL;

    const char* start = strstr(text, "```json");
    if (start) start += 7;
    else start = text;

    const char* open = strpbrk(start, "[{");
    if (!open) return NULL;

    int depth = 0;
    bool in_str = false, esc = false;
    const char* p = open;
    for (; *p; p++) {
        char c = *p;
        if (esc) { esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') depth++;
        else if (c == ']' || c == '}') {
            depth--;
            if (depth == 0) { p++; break; }
        }
    }

    size_t len = (size_t)(p - open);
    if (depth == 0) return dup_cstr(open, len);

    /* Truncated output: drop the partial trailing object and close up. */
    const char* last = open + len;
    while (last > open && *(last - 1) != '}') last--;
    if (last <= open) return NULL;
    size_t keep = (size_t)(last - open);

    char* out = (char*)vv_alloc(keep + 8);
    if (!out) return NULL;
    memcpy(out, open, keep);
    size_t w = keep;
    if (*open == '[') out[w++] = ']';
    out[w] = '\0';
    return out;
}

/* ─── Build transcription from decoded text ─────────────────────────────── */

vv_status_t vv_postprocess_text(const char* text, float audio_duration,
                                vv_transcription_t** result) {
    if (!text || !result) return VV_ERR_NULL_PTR;

    vv_transcription_t* tr = (vv_transcription_t*)vv_alloc(sizeof(*tr));
    if (!tr) return VV_ERR_OUT_OF_MEMORY;
    memset(tr, 0, sizeof(*tr));
    tr->duration = audio_duration;
    tr->language = "unknown";

    char* json_str = extract_json(text);
    cJSON* root = json_str ? cJSON_Parse(json_str) : NULL;
    if (json_str) vv_free(json_str);

    if (!root) {
        /* Not JSON after all — hand back the raw answer. */
        tr->full_text = dup_cstr(text, strlen(text));
        *result = tr;
        return VV_OK;
    }

    cJSON* arr = cJSON_IsArray(root) ? root : NULL;
    int n = arr ? cJSON_GetArraySize(arr) : 1;
    if (n < 0) n = 0;

    tr->segments = (vv_segment_t*)vv_alloc((size_t)(n > 0 ? n : 1) *
                                           sizeof(vv_segment_t));
    if (!tr->segments) { cJSON_Delete(root); vv_free(tr); return VV_ERR_OUT_OF_MEMORY; }
    memset(tr->segments, 0, (size_t)(n > 0 ? n : 1) * sizeof(vv_segment_t));

    static const char* K_START[]   = {"Start time", "Start", "start_time", "start", NULL};
    static const char* K_END[]     = {"End time", "End", "end_time", "end", NULL};
    static const char* K_SPEAKER[] = {"Speaker ID", "Speaker", "speaker_id", "speaker", NULL};
    static const char* K_TEXT[]    = {"Content", "Text", "content", "text", NULL};

    size_t cap = 1024, len = 0;
    char* full = (char*)vv_alloc(cap);
    if (!full) { cJSON_Delete(root); vv_free(tr->segments); vv_free(tr); return VV_ERR_OUT_OF_MEMORY; }
    full[0] = '\0';

    for (int i = 0; i < n; i++) {
        cJSON* item = arr ? cJSON_GetArrayItem(arr, i) : root;
        if (!item || !cJSON_IsObject(item)) continue;

        vv_segment_t* seg = &tr->segments[tr->num_segments];
        seg->start_time = as_float(pick(item, K_START), 0.0f);
        seg->end_time   = as_float(pick(item, K_END), 0.0f);

        cJSON* spk = pick(item, K_SPEAKER);
        char name[64];
        if (spk && cJSON_IsString(spk) && spk->valuestring)
            snprintf(name, sizeof(name), "%s", spk->valuestring);
        else if (spk && cJSON_IsNumber(spk))
            snprintf(name, sizeof(name), "Speaker %d", (int)spk->valuedouble);
        else
            snprintf(name, sizeof(name), "Speaker 0");
        seg->speaker = dup_cstr(name, strlen(name));

        cJSON* txt = pick(item, K_TEXT);
        const char* body = (txt && cJSON_IsString(txt) && txt->valuestring)
                           ? txt->valuestring : "";
        seg->text = dup_cstr(body, strlen(body));

        size_t blen = strlen(body);
        if (blen) {
            if (len + blen + 2 > cap) {
                cap = (len + blen + 2) * 2;
                char* nf = (char*)vv_realloc(full, cap);
                if (!nf) break;
                full = nf;
            }
            if (len) full[len++] = ' ';
            memcpy(full + len, body, blen);
            len += blen;
            full[len] = '\0';
        }
        tr->num_segments++;
    }

    cJSON_Delete(root);
    tr->full_text = full;
    *result = tr;
    return VV_OK;
}

/**
 * @brief Legacy entry point: joins the per-token strings, then parses.
 */
vv_status_t vv_postprocess_tokens(
    const char** token_texts, int n_tokens,
    vv_transcription_t** result)
{
    if (!token_texts || !result) return VV_ERR_NULL_PTR;

    size_t total = 1;
    for (int i = 0; i < n_tokens; i++)
        if (token_texts[i]) total += strlen(token_texts[i]);

    char* joined = (char*)vv_alloc(total);
    if (!joined) return VV_ERR_OUT_OF_MEMORY;
    size_t w = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (!token_texts[i]) continue;
        size_t l = strlen(token_texts[i]);
        memcpy(joined + w, token_texts[i], l);
        w += l;
    }
    joined[w] = '\0';

    vv_status_t s = vv_postprocess_text(joined, 0.0f, result);
    vv_free(joined);
    return s;
}

/* ─── JSON output ───────────────────────────────────────────────────────── */

vv_status_t vv_transcription_to_json(const vv_transcription_t* tr,
                                      char** json_str) {
    if (!tr || !json_str) return VV_ERR_NULL_PTR;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "text",
                             tr->full_text ? tr->full_text : "");
    cJSON_AddNumberToObject(root, "duration", (double)tr->duration);
    cJSON_AddStringToObject(root, "language",
                             tr->language ? tr->language : "unknown");

    cJSON* segments = cJSON_AddArrayToObject(root, "segments");
    for (int i = 0; i < tr->num_segments; i++) {
        cJSON* seg = cJSON_CreateObject();
        cJSON_AddStringToObject(seg, "speaker",
                                 tr->segments[i].speaker ?
                                 tr->segments[i].speaker : "Unknown");
        cJSON_AddNumberToObject(seg, "start",
                                 (double)tr->segments[i].start_time);
        cJSON_AddNumberToObject(seg, "end",
                                 (double)tr->segments[i].end_time);
        cJSON_AddStringToObject(seg, "text",
                                 tr->segments[i].text ?
                                 tr->segments[i].text : "");
        cJSON_AddItemToArray(segments, seg);
    }

    *json_str = cJSON_Print(root);
    cJSON_Delete(root);

    return VV_OK;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

vv_status_t vv_transcription_free(vv_transcription_t* tr) {
    if (!tr) return VV_ERR_NULL_PTR;

    if (tr->segments) {
        for (int i = 0; i < tr->num_segments; i++) {
            if (tr->segments[i].speaker) vv_free((void*)tr->segments[i].speaker);
            if (tr->segments[i].text) vv_free((void*)tr->segments[i].text);
        }
        vv_free(tr->segments);
    }
    if (tr->full_text) vv_free((void*)tr->full_text);

    vv_free(tr);
    return VV_OK;
}
