/**
 * @file bpe.c
 * @brief BPE text tokenizer implementation.
 *
 * Loads tokenizer.json (HuggingFace format) and performs:
 * - BPE encoding: text → token IDs
 * - Decoding: token IDs → text
 * - Special token handling for VibeVoice-ASR
 */

#include "vibevoice/text_tokenizer.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ─── Internal structures ───────────────────────────────────────────────── */

typedef struct {
    char* token;
    int   id;
} vv_vocab_entry_t;

typedef struct {
    char* first;
    char* second;
} vv_merge_t;

struct vv_tokenizer {
    /* Vocabulary: token string → ID */
    vv_vocab_entry_t* vocab;
    int               vocab_size;

    /* Reverse vocabulary: ID → token string */
    char**            id_to_token;
    int               id_to_token_size;

    /* BPE merge rules (ordered by priority) */
    vv_merge_t*       merges;
    int               n_merges;

    /* Special tokens */
    vv_vocab_entry_t* special_tokens;
    int               n_special;
};

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static int find_token_id(const vv_tokenizer_t* tok, const char* token) {
    /* Linear search (could be optimized with hash table) */
    for (int i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->vocab[i].token, token) == 0) {
            return tok->vocab[i].id;
        }
    }
    return -1;
}

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return NULL; }

    char* buf = (char*)vv_alloc((size_t)size + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[size] = '\0';
    return buf;
}

/* ─── Load ──────────────────────────────────────────────────────────────── */

vv_status_t vv_tokenizer_load(const char* dir_path, vv_tokenizer_t** out) {
    if (!dir_path || !out) return VV_ERR_NULL_PTR;

    /* Build path to tokenizer.json */
    char path[512];
    size_t dlen = strlen(dir_path);
    if (dlen > 0 && (dir_path[dlen-1] == '/' || dir_path[dlen-1] == '\\')) {
        snprintf(path, sizeof(path), "%stokenizer.json", dir_path);
    } else {
        snprintf(path, sizeof(path), "%s/tokenizer.json", dir_path);
    }

    char* json_str = read_file(path);
    if (!json_str) {
        VV_LOG_E("tokenizer: cannot read '%s'", path);
        return VV_ERR_IO;
    }

    cJSON* root = cJSON_Parse(json_str);
    vv_free(json_str);
    if (!root) {
        VV_LOG_E("tokenizer: JSON parse failed");
        return VV_ERR_PARSE;
    }

    vv_tokenizer_t* tok = (vv_tokenizer_t*)vv_alloc(sizeof(vv_tokenizer_t));
    if (!tok) { cJSON_Delete(root); return VV_ERR_OUT_OF_MEMORY; }
    memset(tok, 0, sizeof(*tok));

    /* Parse model.vocab */
    cJSON* model = cJSON_GetObjectItem(root, "model");
    if (model) {
        cJSON* vocab = cJSON_GetObjectItem(model, "vocab");
        if (vocab) {
            int n = cJSON_GetArraySize(vocab);
            tok->vocab = (vv_vocab_entry_t*)vv_alloc(
                (size_t)n * sizeof(vv_vocab_entry_t));
            if (!tok->vocab) {
                cJSON_Delete(root);
                vv_free(tok);
                return VV_ERR_OUT_OF_MEMORY;
            }
            tok->vocab_size = 0;

            int max_id = 0;
            cJSON* entry;
            cJSON_ArrayForEach(entry, vocab) {
                if (tok->vocab_size >= n) break;
                vv_vocab_entry_t* e = &tok->vocab[tok->vocab_size];
                e->token = (char*)vv_alloc(strlen(entry->string) + 1);
                if (e->token) strcpy(e->token, entry->string);
                e->id = (int)entry->valuedouble;
                if (e->id > max_id) max_id = e->id;
                tok->vocab_size++;
            }

            /* Build reverse mapping */
            tok->id_to_token_size = max_id + 1;
            tok->id_to_token = (char**)vv_alloc(
                (size_t)(max_id + 1) * sizeof(char*));
            if (tok->id_to_token) {
                memset(tok->id_to_token, 0,
                       (size_t)(max_id + 1) * sizeof(char*));
                for (int i = 0; i < tok->vocab_size; i++) {
                    int id = tok->vocab[i].id;
                    if (id >= 0 && id <= max_id) {
                        tok->id_to_token[id] = tok->vocab[i].token;
                    }
                }
            }
        }

        /* Parse merges — use cJSON_ArrayForEach for O(n) linked-list walk
         * instead of cJSON_GetArrayItem(i) which is O(n²) */
        cJSON* merges = cJSON_GetObjectItem(model, "merges");
        if (merges && cJSON_IsArray(merges)) {
            int nm = cJSON_GetArraySize(merges);
            tok->merges = (vv_merge_t*)vv_alloc(
                (size_t)nm * sizeof(vv_merge_t));
            if (tok->merges) {
                tok->n_merges = 0;
                cJSON* m;
                cJSON_ArrayForEach(m, merges) {
                    if (!cJSON_IsString(m)) continue;

                    const char* s = m->valuestring;
                    /* Format: "token1 token2" */
                    const char* space = strchr(s, ' ');
                    if (!space) continue;

                    size_t len1 = (size_t)(space - s);
                    size_t len2 = strlen(space + 1);

                    tok->merges[tok->n_merges].first =
                        (char*)vv_alloc(len1 + 1);
                    tok->merges[tok->n_merges].second =
                        (char*)vv_alloc(len2 + 1);

                    if (tok->merges[tok->n_merges].first &&
                        tok->merges[tok->n_merges].second) {
                        memcpy(tok->merges[tok->n_merges].first, s, len1);
                        tok->merges[tok->n_merges].first[len1] = '\0';
                        strcpy(tok->merges[tok->n_merges].second, space + 1);
                        tok->n_merges++;
                    }
                }
            }
        }
    }

    /* Parse added_tokens (special tokens) */
    cJSON* added = cJSON_GetObjectItem(root, "added_tokens");
    if (added && cJSON_IsArray(added)) {
        int na = cJSON_GetArraySize(added);
        tok->special_tokens = (vv_vocab_entry_t*)vv_alloc(
            (size_t)na * sizeof(vv_vocab_entry_t));
        if (tok->special_tokens) {
            tok->n_special = 0;
            cJSON* item;
            cJSON_ArrayForEach(item, added) {
                cJSON* content = cJSON_GetObjectItem(item, "content");
                cJSON* id_val = cJSON_GetObjectItem(item, "id");
                if (!content || !id_val) continue;

                vv_vocab_entry_t* e = &tok->special_tokens[tok->n_special];
                e->token = (char*)vv_alloc(
                    strlen(content->valuestring) + 1);
                if (e->token) {
                    strcpy(e->token, content->valuestring);
                    e->id = (int)id_val->valuedouble;
                    tok->n_special++;

                    /* Also add to reverse mapping */
                    if (tok->id_to_token && e->id >= 0 &&
                        e->id < tok->id_to_token_size) {
                        tok->id_to_token[e->id] = e->token;
                    }
                }
            }
        }
    }

    cJSON_Delete(root);

    VV_LOG_I("tokenizer: loaded %d vocab entries, %d merges, %d special tokens",
             tok->vocab_size, tok->n_merges, tok->n_special);

    *out = tok;
    return VV_OK;
}

/* ─── Encode ────────────────────────────────────────────────────────────── */

vv_status_t vv_tokenizer_encode(const vv_tokenizer_t* tok, const char* text,
                                 int32_t** ids, int* n_tokens) {
    if (!tok || !text || !ids || !n_tokens) return VV_ERR_NULL_PTR;

    size_t text_len = strlen(text);
    if (text_len == 0) {
        *ids = NULL;
        *n_tokens = 0;
        return VV_OK;
    }

    /*
     * Simplified BPE encoding:
     * 1. Split text into initial characters (UTF-8 aware)
     * 2. Repeatedly apply the highest-priority merge
     */

    /* Initial split: one token per byte (simplified) */
    int max_tokens = (int)text_len + 16;
    char** tokens = (char**)vv_alloc((size_t)max_tokens * sizeof(char*));
    if (!tokens) return VV_ERR_OUT_OF_MEMORY;

    int nt = 0;
    size_t pos = 0;
    while (pos < text_len && nt < max_tokens) {
        /* Get UTF-8 character length */
        unsigned char c = (unsigned char)text[pos];
        int char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        if (pos + char_len > text_len) char_len = (int)(text_len - pos);

        tokens[nt] = (char*)vv_alloc((size_t)char_len + 1);
        if (!tokens[nt]) break;
        memcpy(tokens[nt], text + pos, (size_t)char_len);
        tokens[nt][char_len] = '\0';
        nt++;
        pos += char_len;
    }

    /* Apply BPE merges (greedy: find highest-priority applicable merge) */
    bool changed = true;
    while (changed && nt > 1) {
        changed = false;
        int best_merge = -1;
        int best_pos = -1;

        /* Find the merge with lowest index (highest priority) that applies */
        for (int m = 0; m < tok->n_merges; m++) {
            for (int p = 0; p < nt - 1; p++) {
                if (strcmp(tokens[p], tok->merges[m].first) == 0 &&
                    strcmp(tokens[p + 1], tok->merges[m].second) == 0) {
                    best_merge = m;
                    best_pos = p;
                    goto found_merge;
                }
            }
        }
        found_merge:

        if (best_merge >= 0) {
            /* Merge tokens[best_pos] and tokens[best_pos+1] */
            size_t len1 = strlen(tokens[best_pos]);
            size_t len2 = strlen(tokens[best_pos + 1]);
            char* merged = (char*)vv_alloc(len1 + len2 + 1);
            if (!merged) break;
            memcpy(merged, tokens[best_pos], len1);
            memcpy(merged + len1, tokens[best_pos + 1], len2);
            merged[len1 + len2] = '\0';

            vv_free(tokens[best_pos]);
            vv_free(tokens[best_pos + 1]);
            tokens[best_pos] = merged;

            /* Shift remaining tokens */
            for (int i = best_pos + 1; i < nt - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            nt--;
            changed = true;
        }
    }

    /* Convert tokens to IDs */
    int32_t* result_ids = (int32_t*)vv_alloc((size_t)nt * sizeof(int32_t));
    if (!result_ids) {
        for (int i = 0; i < nt; i++) vv_free(tokens[i]);
        vv_free(tokens);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < nt; i++) {
        int id = find_token_id(tok, tokens[i]);
        result_ids[i] = (id >= 0) ? id : 0; /* Unknown → 0 */
        vv_free(tokens[i]);
    }
    vv_free(tokens);

    *ids = result_ids;
    *n_tokens = nt;
    return VV_OK;
}

/* ─── Decode ────────────────────────────────────────────────────────────── */

vv_status_t vv_tokenizer_decode(const vv_tokenizer_t* tok,
                                 const int32_t* ids, int n_tokens,
                                 char** text) {
    if (!tok || !ids || !text) return VV_ERR_NULL_PTR;

    size_t total_len = 0;
    for (int i = 0; i < n_tokens; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->id_to_token_size && tok->id_to_token[id]) {
            total_len += strlen(tok->id_to_token[id]);
        }
    }

    char* result = (char*)vv_alloc(total_len + 1);
    if (!result) return VV_ERR_OUT_OF_MEMORY;
    result[0] = '\0';

    size_t pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->id_to_token_size && tok->id_to_token[id]) {
            const char* t = tok->id_to_token[id];
            size_t tlen = strlen(t);
            memcpy(result + pos, t, tlen);
            pos += tlen;
        }
    }
    result[pos] = '\0';

    *text = result;
    return VV_OK;
}

/* ─── Special tokens ────────────────────────────────────────────────────── */

int vv_tokenizer_special_id(const vv_tokenizer_t* tok, const char* name) {
    if (!tok || !name) return -1;

    for (int i = 0; i < tok->n_special; i++) {
        if (strcmp(tok->special_tokens[i].token, name) == 0) {
            return tok->special_tokens[i].id;
        }
    }

    /* Also check regular vocab */
    return find_token_id(tok, name);
}

int vv_tokenizer_vocab_size(const vv_tokenizer_t* tok) {
    return tok ? tok->vocab_size : 0;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

vv_status_t vv_tokenizer_free(vv_tokenizer_t* tok) {
    if (!tok) return VV_ERR_NULL_PTR;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i].token) vv_free(tok->vocab[i].token);
        }
        vv_free(tok->vocab);
    }

    if (tok->id_to_token) vv_free(tok->id_to_token);

    if (tok->merges) {
        for (int i = 0; i < tok->n_merges; i++) {
            if (tok->merges[i].first) vv_free(tok->merges[i].first);
            if (tok->merges[i].second) vv_free(tok->merges[i].second);
        }
        vv_free(tok->merges);
    }

    if (tok->special_tokens) {
        for (int i = 0; i < tok->n_special; i++) {
            if (tok->special_tokens[i].token)
                vv_free(tok->special_tokens[i].token);
        }
        vv_free(tok->special_tokens);
    }

    vv_free(tok);
    return VV_OK;
}
