/**
 * @file postprocess.c
 * @brief Post-processing: token stream → structured JSON transcription.
 *
 * Parse special tokens like <|speaker_N|>, <|timestamp_X.XX|>,
 * <|startoftranscript|>, <|endoftranscript|> from the generated
 * token stream and produce the final vv_transcription_t result.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── Token classification ──────────────────────────────────────────────── */

static bool is_timestamp_token(const char* text, float* time_val) {
    if (!text) return false;
    if (strncmp(text, "<|timestamp_", 12) != 0) return false;
    const char* p = text + 12;
    char* end;
    *time_val = strtof(p, &end);
    return (end != p && *end == '|');
}

static bool is_speaker_token(const char* text, int* speaker_id) {
    if (!text) return false;
    if (strncmp(text, "<|speaker_", 10) != 0) return false;
    *speaker_id = atoi(text + 10);
    return true;
}

/* ─── Build transcription from decoded text ─────────────────────────────── */

vv_status_t vv_postprocess_tokens(
    const char** token_texts, int n_tokens,
    vv_transcription_t** result)
{
    if (!token_texts || !result) return VV_ERR_NULL_PTR;

    vv_transcription_t* tr = (vv_transcription_t*)vv_alloc(
        sizeof(vv_transcription_t));
    if (!tr) return VV_ERR_OUT_OF_MEMORY;
    memset(tr, 0, sizeof(*tr));

    /* First pass: count segments (each speaker_token starts a new segment) */
    int n_segs = 0;
    for (int i = 0; i < n_tokens; i++) {
        int sid;
        if (is_speaker_token(token_texts[i], &sid)) n_segs++;
    }
    if (n_segs == 0) n_segs = 1;

    tr->segments = (vv_segment_t*)vv_alloc(
        (size_t)n_segs * sizeof(vv_segment_t));
    if (!tr->segments) {
        vv_free(tr);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(tr->segments, 0, (size_t)n_segs * sizeof(vv_segment_t));
    tr->num_segments = 0;

    /* Build full text and parse segments */
    size_t full_text_cap = 4096;
    char* full_text = (char*)vv_alloc(full_text_cap);
    if (!full_text) {
        vv_free(tr->segments);
        vv_free(tr);
        return VV_ERR_OUT_OF_MEMORY;
    }
    full_text[0] = '\0';
    size_t full_text_len = 0;

    int cur_speaker = 0;
    float cur_start = 0.0f;
    float cur_end = 0.0f;
    size_t seg_text_cap = 1024;
    char* seg_text = (char*)vv_alloc(seg_text_cap);
    if (!seg_text) {
        vv_free(full_text);
        vv_free(tr->segments);
        vv_free(tr);
        return VV_ERR_OUT_OF_MEMORY;
    }
    seg_text[0] = '\0';
    size_t seg_text_len = 0;
    bool in_segment = false;

    for (int i = 0; i < n_tokens; i++) {
        const char* tok = token_texts[i];
        if (!tok) continue;

        /* Skip transcript markers */
        if (strcmp(tok, "<|startoftranscript|>") == 0) continue;
        if (strcmp(tok, "<|endoftranscript|>") == 0) break;
        if (strcmp(tok, "<|nospeech|>") == 0) continue;

        int sid;
        float ts;

        if (is_speaker_token(tok, &sid)) {
            /* Flush previous segment */
            if (in_segment && tr->num_segments < n_segs) {
                vv_segment_t* seg = &tr->segments[tr->num_segments];
                char speaker_name[32];
                snprintf(speaker_name, sizeof(speaker_name),
                         "Speaker %d", cur_speaker);
                seg->speaker = (char*)vv_alloc(strlen(speaker_name) + 1);
                if (seg->speaker) strcpy((char*)seg->speaker, speaker_name);
                seg->start_time = cur_start;
                seg->end_time = cur_end;
                seg->text = (char*)vv_alloc(seg_text_len + 1);
                if (seg->text) {
                    memcpy((char*)seg->text, seg_text, seg_text_len);
                    ((char*)seg->text)[seg_text_len] = '\0';
                }
                tr->num_segments++;
            }
            cur_speaker = sid;
            seg_text[0] = '\0';
            seg_text_len = 0;
            in_segment = true;
        }
        else if (is_timestamp_token(tok, &ts)) {
            if (!in_segment) {
                cur_start = ts;
                in_segment = true;
            } else {
                cur_end = ts;
            }
        }
        else {
            /* Regular text token */
            size_t tlen = strlen(tok);

            /* Append to segment text */
            if (seg_text_len + tlen + 1 > seg_text_cap) {
                seg_text_cap = seg_text_cap * 2 + tlen;
                seg_text = (char*)vv_realloc(seg_text, seg_text_cap);
            }
            memcpy(seg_text + seg_text_len, tok, tlen);
            seg_text_len += tlen;
            seg_text[seg_text_len] = '\0';

            /* Append to full text */
            if (full_text_len + tlen + 1 > full_text_cap) {
                full_text_cap = full_text_cap * 2 + tlen;
                full_text = (char*)vv_realloc(full_text, full_text_cap);
            }
            memcpy(full_text + full_text_len, tok, tlen);
            full_text_len += tlen;
            full_text[full_text_len] = '\0';
        }
    }

    /* Flush last segment */
    if (in_segment && seg_text_len > 0 && tr->num_segments < n_segs) {
        vv_segment_t* seg = &tr->segments[tr->num_segments];
        char speaker_name[32];
        snprintf(speaker_name, sizeof(speaker_name),
                 "Speaker %d", cur_speaker);
        seg->speaker = (char*)vv_alloc(strlen(speaker_name) + 1);
        if (seg->speaker) strcpy((char*)seg->speaker, speaker_name);
        seg->start_time = cur_start;
        seg->end_time = cur_end;
        seg->text = (char*)vv_alloc(seg_text_len + 1);
        if (seg->text) {
            memcpy((char*)seg->text, seg_text, seg_text_len);
            ((char*)seg->text)[seg_text_len] = '\0';
        }
        tr->num_segments++;
    }

    tr->full_text = full_text;
    tr->duration = cur_end;
    tr->language = "en"; /* TODO: detect from tokens */

    vv_free(seg_text);

    *result = tr;
    return VV_OK;
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
