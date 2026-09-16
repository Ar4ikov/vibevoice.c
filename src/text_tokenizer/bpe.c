/**
 * @file bpe.c
 * @brief Byte-level BPE text tokenizer (Qwen2 / GPT-2 family).
 *
 * Loads tokenizer.json (HuggingFace `tokenizers` format) and reproduces the
 * exact pipeline declared there for the VibeVoice-ASR checkpoint:
 *
 *   normalizer     : NFC              (input is assumed to be NFC already)
 *   pre_tokenizer  : Split(Regex, Isolated) -> ByteLevel(add_prefix_space=false)
 *   model          : BPE (no dropout, no byte_fallback)
 *   decoder        : ByteLevel
 *
 * The split regex is
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * and is implemented directly (no regex engine) in next_piece().
 */

#include "vibevoice/text_tokenizer.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vv_thread.h"

/* ─── Byte-level alphabet (GPT-2) ───────────────────────────────────────── */

/** Codepoint each raw byte maps to in the byte-level alphabet. */
static uint32_t g_byte_to_cp[256];
/** Reverse map, only valid for codepoints produced by g_byte_to_cp. */
static int      g_cp_to_byte[324];
static vv_once_t g_bytelevel_once = VV_ONCE_INIT;

static void bytelevel_build(void) {
    bool printable[256];
    for (int b = 0; b < 256; b++) printable[b] = false;
    for (int b = '!';  b <= '~';  b++) printable[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) printable[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) printable[b] = true;

    for (int i = 0; i < 324; i++) g_cp_to_byte[i] = -1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (printable[b]) {
            g_byte_to_cp[b] = (uint32_t)b;
        } else {
            g_byte_to_cp[b] = (uint32_t)(256 + n);
            n++;
        }
        if (g_byte_to_cp[b] < 324) g_cp_to_byte[g_byte_to_cp[b]] = b;
    }
}

/** @brief Build the byte alphabet once, however many threads ask for it. */
static void bytelevel_init(void) { vv_once(&g_bytelevel_once, bytelevel_build); }

/** @brief Append codepoint @p cp to @p out as UTF-8. Returns bytes written. */
static int utf8_put(char* out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/** @brief Decode one UTF-8 codepoint. Returns bytes consumed (>=1). */
static int utf8_get(const char* s, size_t len, uint32_t* cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && len >= 2) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && len >= 3) {
        *cp = ((uint32_t)(c & 0x0F) << 12) |
              (((unsigned char)s[1] & 0x3F) << 6) |
              ((unsigned char)s[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && len >= 4) {
        *cp = ((uint32_t)(c & 0x07) << 18) |
              (((unsigned char)s[1] & 0x3F) << 12) |
              (((unsigned char)s[2] & 0x3F) << 6) |
              ((unsigned char)s[3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

/* ─── Unicode class approximations for the split regex ──────────────────── */

static bool cp_is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == 0x0B || c == 0x0C || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

static bool cp_is_digit(uint32_t c) {
    return (c >= '0' && c <= '9') ||
           (c >= 0x0660 && c <= 0x0669) || (c >= 0x06F0 && c <= 0x06F9) ||
           (c >= 0x0966 && c <= 0x096F) || (c >= 0x09E6 && c <= 0x09EF) ||
           (c >= 0x0E50 && c <= 0x0E59) || (c >= 0xFF10 && c <= 0xFF19) ||
           c == 0xB2 || c == 0xB3 || c == 0xB9 ||
           (c >= 0xBC && c <= 0xBE);
}

/** Ranges that are punctuation/symbols rather than letters. */
static bool cp_is_sym(uint32_t c) {
    if (c < 0x80) return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    if (c >= 0xA0 && c <= 0xBF) return true;
    if (c == 0xD7 || c == 0xF7) return true;
    if (c >= 0x2000 && c <= 0x2BFF) return true;
    if (c >= 0x3000 && c <= 0x303F) return true;
    if (c >= 0xFE00 && c <= 0xFE6F) return true;
    if (c >= 0xFF00 && c <= 0xFF0F) return true;
    if (c >= 0xFF1A && c <= 0xFF20) return true;
    if (c >= 0xFF3B && c <= 0xFF40) return true;
    if (c >= 0xFF5B && c <= 0xFF65) return true;
    if (c >= 0xFFF0) return true;
    if (c >= 0x1F000 && c <= 0x1FAFF) return true;
    return false;
}

/** @brief Approximates \p{L}: anything that is not space, digit or symbol. */
static bool cp_is_letter(uint32_t c) {
    if (cp_is_space(c) || cp_is_digit(c)) return false;
    return !cp_is_sym(c);
}

/* ─── Hash maps ─────────────────────────────────────────────────────────── */

static uint64_t fnv1a(const char* s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

typedef struct {
    char*    str;   /**< owned token text */
    uint32_t len;
    int32_t  id;
} vocab_entry_t;

typedef struct {
    char*    a;     /**< owned left side */
    uint32_t alen;
    char*    b;     /**< owned right side */
    uint32_t blen;
    int32_t  rank;
} merge_entry_t;

struct vv_tokenizer {
    /* vocab: open-addressed hash table of indices into `ventry` */
    vocab_entry_t* ventry;
    int            n_vocab;
    int32_t*       vbucket;      /**< -1 = empty, else index into ventry */
    uint32_t       vmask;

    /* merges */
    merge_entry_t* mentry;
    int            n_merges;
    int32_t*       mbucket;
    uint32_t       mmask;

    /* id -> token text (borrowed pointers into ventry / sentry) */
    char**         id_to_token;
    uint32_t*      id_to_len;
    int            id_capacity;

    /* special / added tokens, longest-first for greedy matching */
    vocab_entry_t* sentry;
    int            n_special;
    bool*          s_is_special; /**< true when HF marks it "special" */
};

static uint32_t next_pow2(uint32_t v) {
    uint32_t p = 16;
    while (p < v) p <<= 1;
    return p;
}

static void vocab_put(vv_tokenizer_t* t, int idx) {
    uint64_t h = fnv1a(t->ventry[idx].str, t->ventry[idx].len);
    uint32_t i = (uint32_t)h & t->vmask;
    while (t->vbucket[i] >= 0) i = (i + 1) & t->vmask;
    t->vbucket[i] = idx;
}

static int32_t vocab_get(const vv_tokenizer_t* t, const char* s, size_t n) {
    uint64_t h = fnv1a(s, n);
    uint32_t i = (uint32_t)h & t->vmask;
    while (t->vbucket[i] >= 0) {
        const vocab_entry_t* e = &t->ventry[t->vbucket[i]];
        if (e->len == n && memcmp(e->str, s, n) == 0) return e->id;
        i = (i + 1) & t->vmask;
    }
    return -1;
}

static uint64_t pair_hash(const char* a, size_t an, const char* b, size_t bn) {
    return fnv1a(a, an) * 1000003ULL ^ fnv1a(b, bn);
}

static void merge_put(vv_tokenizer_t* t, int idx) {
    const merge_entry_t* m = &t->mentry[idx];
    uint64_t h = pair_hash(m->a, m->alen, m->b, m->blen);
    uint32_t i = (uint32_t)h & t->mmask;
    while (t->mbucket[i] >= 0) i = (i + 1) & t->mmask;
    t->mbucket[i] = idx;
}

static int32_t merge_rank(const vv_tokenizer_t* t,
                          const char* a, size_t an,
                          const char* b, size_t bn) {
    uint64_t h = pair_hash(a, an, b, bn);
    uint32_t i = (uint32_t)h & t->mmask;
    while (t->mbucket[i] >= 0) {
        const merge_entry_t* m = &t->mentry[t->mbucket[i]];
        if (m->alen == an && m->blen == bn &&
            memcmp(m->a, a, an) == 0 && memcmp(m->b, b, bn) == 0) {
            return m->rank;
        }
        i = (i + 1) & t->mmask;
    }
    return -1;
}

/* ─── Loading ───────────────────────────────────────────────────────────── */

static char* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return NULL; }

    char* buf = (char*)vv_alloc((size_t)size + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fclose(f); vv_free(buf); return NULL;
    }
    fclose(f);
    buf[size] = '\0';
    if (out_size) *out_size = (size_t)size;
    return buf;
}

static char* dup_str(const char* s, size_t n) {
    char* p = (char*)vv_alloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

vv_status_t vv_tokenizer_load(const char* dir_path, vv_tokenizer_t** out) {
    if (!dir_path || !out) return VV_ERR_NULL_PTR;
    bytelevel_init();

    char path[512];
    size_t dlen = strlen(dir_path);
    if (dlen > 0 && (dir_path[dlen - 1] == '/' || dir_path[dlen - 1] == '\\'))
        snprintf(path, sizeof(path), "%stokenizer.json", dir_path);
    else
        snprintf(path, sizeof(path), "%s/tokenizer.json", dir_path);

    char* json_str = read_file(path, NULL);
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

    vv_tokenizer_t* t = (vv_tokenizer_t*)vv_alloc(sizeof(*t));
    if (!t) { cJSON_Delete(root); return VV_ERR_OUT_OF_MEMORY; }
    memset(t, 0, sizeof(*t));

    /* ── model.vocab ── */
    cJSON* model = cJSON_GetObjectItem(root, "model");
    cJSON* vocab = model ? cJSON_GetObjectItem(model, "vocab") : NULL;
    if (!vocab) {
        VV_LOG_E("tokenizer: model.vocab missing");
        cJSON_Delete(root); vv_free(t);
        return VV_ERR_PARSE;
    }

    int nv = cJSON_GetArraySize(vocab);
    t->ventry = (vocab_entry_t*)vv_alloc((size_t)nv * sizeof(vocab_entry_t));
    if (!t->ventry) goto oom;
    memset(t->ventry, 0, (size_t)nv * sizeof(vocab_entry_t));

    int max_id = 0;
    {
        cJSON* item = vocab->child;
        int i = 0;
        while (item && i < nv) {
            size_t klen = strlen(item->string);
            t->ventry[i].str = dup_str(item->string, klen);
            t->ventry[i].len = (uint32_t)klen;
            t->ventry[i].id  = (int32_t)item->valuedouble;
            if (t->ventry[i].id > max_id) max_id = t->ventry[i].id;
            i++;
            item = item->next;
        }
        t->n_vocab = i;
    }

    t->vmask = next_pow2((uint32_t)t->n_vocab * 2) - 1;
    t->vbucket = (int32_t*)vv_alloc(((size_t)t->vmask + 1) * sizeof(int32_t));
    if (!t->vbucket) goto oom;
    for (uint32_t i = 0; i <= t->vmask; i++) t->vbucket[i] = -1;
    for (int i = 0; i < t->n_vocab; i++) vocab_put(t, i);

    /* ── model.merges ── */
    cJSON* merges = cJSON_GetObjectItem(model, "merges");
    if (merges) {
        int nm = cJSON_GetArraySize(merges);
        t->mentry = (merge_entry_t*)vv_alloc((size_t)nm * sizeof(merge_entry_t));
        if (!t->mentry) goto oom;
        memset(t->mentry, 0, (size_t)nm * sizeof(merge_entry_t));

        /*
         * Walk the sibling list directly: cJSON arrays are linked lists, so
         * indexing with cJSON_GetArrayItem() inside the loop is quadratic —
         * 151 k merges turned tokenizer loading into a 22-second stall.
         */
        int k = 0, i = 0;
        for (cJSON* m = merges->child; m; m = m->next, i++) {
            const char *a = NULL, *b = NULL;
            size_t an = 0, bn = 0;
            if (cJSON_IsString(m)) {
                const char* s = m->valuestring;
                const char* sp = strchr(s, ' ');
                if (!sp) continue;
                a = s;      an = (size_t)(sp - s);
                b = sp + 1; bn = strlen(sp + 1);
            } else if (cJSON_IsArray(m) && cJSON_GetArraySize(m) == 2) {
                cJSON* x = cJSON_GetArrayItem(m, 0);
                cJSON* y = cJSON_GetArrayItem(m, 1);
                if (!cJSON_IsString(x) || !cJSON_IsString(y)) continue;
                a = x->valuestring; an = strlen(a);
                b = y->valuestring; bn = strlen(b);
            } else {
                continue;
            }
            t->mentry[k].a    = dup_str(a, an);
            t->mentry[k].alen = (uint32_t)an;
            t->mentry[k].b    = dup_str(b, bn);
            t->mentry[k].blen = (uint32_t)bn;
            t->mentry[k].rank = i;
            k++;
        }
        t->n_merges = k;

        t->mmask = next_pow2((uint32_t)(k > 0 ? k : 1) * 2) - 1;
        t->mbucket = (int32_t*)vv_alloc(((size_t)t->mmask + 1) * sizeof(int32_t));
        if (!t->mbucket) goto oom;
        for (uint32_t i = 0; i <= t->mmask; i++) t->mbucket[i] = -1;
        for (int i = 0; i < k; i++) merge_put(t, i);
    }

    /* ── added_tokens ── */
    cJSON* added = cJSON_GetObjectItem(root, "added_tokens");
    if (added) {
        int na = cJSON_GetArraySize(added);
        t->sentry = (vocab_entry_t*)vv_alloc((size_t)na * sizeof(vocab_entry_t));
        t->s_is_special = (bool*)vv_alloc((size_t)na * sizeof(bool));
        if (!t->sentry || !t->s_is_special) goto oom;
        memset(t->sentry, 0, (size_t)na * sizeof(vocab_entry_t));

        int k = 0;
        for (int i = 0; i < na; i++) {
            cJSON* it = cJSON_GetArrayItem(added, i);
            cJSON* c  = cJSON_GetObjectItem(it, "content");
            cJSON* id = cJSON_GetObjectItem(it, "id");
            if (!c || !id || !cJSON_IsString(c)) continue;
            size_t n = strlen(c->valuestring);
            t->sentry[k].str = dup_str(c->valuestring, n);
            t->sentry[k].len = (uint32_t)n;
            t->sentry[k].id  = (int32_t)id->valuedouble;
            cJSON* sp = cJSON_GetObjectItem(it, "special");
            t->s_is_special[k] = sp ? cJSON_IsTrue(sp) : true;
            if (t->sentry[k].id > max_id) max_id = t->sentry[k].id;
            k++;
        }
        t->n_special = k;

        /* longest-first so that overlapping literals match greedily */
        for (int i = 1; i < k; i++) {
            vocab_entry_t e = t->sentry[i];
            bool sflag = t->s_is_special[i];
            int j = i - 1;
            while (j >= 0 && t->sentry[j].len < e.len) {
                t->sentry[j + 1] = t->sentry[j];
                t->s_is_special[j + 1] = t->s_is_special[j];
                j--;
            }
            t->sentry[j + 1] = e;
            t->s_is_special[j + 1] = sflag;
        }
    }

    /* ── id -> token ── */
    t->id_capacity  = max_id + 1;
    t->id_to_token  = (char**)vv_alloc((size_t)t->id_capacity * sizeof(char*));
    t->id_to_len    = (uint32_t*)vv_alloc((size_t)t->id_capacity * sizeof(uint32_t));
    if (!t->id_to_token || !t->id_to_len) goto oom;
    memset(t->id_to_token, 0, (size_t)t->id_capacity * sizeof(char*));
    memset(t->id_to_len, 0, (size_t)t->id_capacity * sizeof(uint32_t));
    for (int i = 0; i < t->n_vocab; i++) {
        int32_t id = t->ventry[i].id;
        if (id >= 0 && id < t->id_capacity) {
            t->id_to_token[id] = t->ventry[i].str;
            t->id_to_len[id]   = t->ventry[i].len;
        }
    }
    for (int i = 0; i < t->n_special; i++) {
        int32_t id = t->sentry[i].id;
        if (id >= 0 && id < t->id_capacity) {
            t->id_to_token[id] = t->sentry[i].str;
            t->id_to_len[id]   = t->sentry[i].len;
        }
    }

    cJSON_Delete(root);
    VV_LOG_I("tokenizer: %d vocab, %d merges, %d added tokens (max id %d)",
             t->n_vocab, t->n_merges, t->n_special, max_id);
    *out = t;
    return VV_OK;

oom:
    cJSON_Delete(root);
    vv_tokenizer_free(t);
    return VV_ERR_OUT_OF_MEMORY;
}

/* ─── Pre-tokenizer split ───────────────────────────────────────────────── */

/**
 * @brief Length of the next pre-token starting at @p s (never 0).
 *
 * Implements the Qwen2 split regex alternation in source order.
 */
static size_t next_piece(const char* s, size_t len) {
    uint32_t cp;
    int adv = utf8_get(s, len, &cp);

    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (cp == '\'' && len >= 2) {
        char c1 = s[1] | 0x20;
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
        if (len >= 3) {
            char c2 = s[2] | 0x20;
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                (c1 == 'l' && c2 == 'l')) return 3;
        }
    }

    /* [^\r\n\p{L}\p{N}]?\p{L}+ */
    {
        size_t p = 0;
        if (cp != '\r' && cp != '\n' && !cp_is_letter(cp) && !cp_is_digit(cp))
            p = (size_t)adv;                 /* tentatively take the prefix */
        size_t q = p;
        while (q < len) {
            uint32_t c2;
            int a2 = utf8_get(s + q, len - q, &c2);
            if (!cp_is_letter(c2)) break;
            q += (size_t)a2;
        }
        if (q > p) return q;                 /* prefix (maybe) + letters */
    }

    /* \p{N} — a single digit */
    if (cp_is_digit(cp)) return (size_t)adv;

    /* ' ?[^\s\p{L}\p{N}]+[\r\n]*' */
    {
        size_t p = (cp == ' ') ? 1 : 0;
        size_t q = p;
        while (q < len) {
            uint32_t c2;
            int a2 = utf8_get(s + q, len - q, &c2);
            if (cp_is_space(c2) || cp_is_letter(c2) || cp_is_digit(c2)) break;
            q += (size_t)a2;
        }
        if (q > p) {
            while (q < len && (s[q] == '\r' || s[q] == '\n')) q++;
            return q;
        }
    }

    /* whitespace run — shared by the last three alternatives */
    if (cp_is_space(cp)) {
        size_t end = 0, last_nl = (size_t)-1;
        while (end < len) {
            uint32_t c2;
            int a2 = utf8_get(s + end, len - end, &c2);
            if (!cp_is_space(c2)) break;
            if (c2 == '\r' || c2 == '\n') last_nl = end;
            end += (size_t)a2;
        }
        /* \s*[\r\n]+ */
        if (last_nl != (size_t)-1) return last_nl + 1;
        /* \s+(?!\S) */
        if (end >= len) return end;
        if (end >= 2) return end - 1;
        /* \s+ */
        return end;
    }

    return (size_t)adv;   /* defensive: never return 0 */
}

/* ─── BPE over one pre-token ────────────────────────────────────────────── */

typedef struct { uint32_t start, len; } sym_t;

static vv_status_t bpe_piece(const vv_tokenizer_t* t,
                             const char* buf, size_t blen,
                             int32_t** ids, int* n_ids, int* cap_ids) {
    /* one symbol per UTF-8 character of the byte-level string */
    size_t max_syms = blen + 1;
    sym_t* sym = (sym_t*)vv_alloc(max_syms * sizeof(sym_t));
    if (!sym) return VV_ERR_OUT_OF_MEMORY;

    int ns = 0;
    for (size_t p = 0; p < blen; ) {
        uint32_t cp;
        int a = utf8_get(buf + p, blen - p, &cp);
        sym[ns].start = (uint32_t)p;
        sym[ns].len   = (uint32_t)a;
        ns++;
        p += (size_t)a;
    }

    while (ns > 1) {
        int32_t best_rank = -1;
        int     best_i    = -1;
        for (int i = 0; i + 1 < ns; i++) {
            int32_t r = merge_rank(t, buf + sym[i].start, sym[i].len,
                                      buf + sym[i + 1].start, sym[i + 1].len);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        sym[best_i].len += sym[best_i + 1].len;
        for (int i = best_i + 1; i + 1 < ns; i++) sym[i] = sym[i + 1];
        ns--;
    }

    for (int i = 0; i < ns; i++) {
        int32_t id = vocab_get(t, buf + sym[i].start, sym[i].len);
        if (id < 0) continue;               /* unreachable: byte alphabet covers all */
        if (*n_ids >= *cap_ids) {
            int nc = (*cap_ids) * 2 + 64;
            int32_t* np = (int32_t*)vv_alloc((size_t)nc * sizeof(int32_t));
            if (!np) { vv_free(sym); return VV_ERR_OUT_OF_MEMORY; }
            memcpy(np, *ids, (size_t)(*n_ids) * sizeof(int32_t));
            vv_free(*ids);
            *ids = np;
            *cap_ids = nc;
        }
        (*ids)[(*n_ids)++] = id;
    }
    vv_free(sym);
    return VV_OK;
}

/** @brief Encode a plain-text run (no special tokens inside). */
static vv_status_t encode_text_run(const vv_tokenizer_t* t,
                                   const char* text, size_t len,
                                   int32_t** ids, int* n_ids, int* cap_ids) {
    /* worst case 2 UTF-8 bytes per input byte in the byte-level alphabet */
    size_t cap = len * 2 + 8;
    char* enc = (char*)vv_alloc(cap);
    if (!enc) return VV_ERR_OUT_OF_MEMORY;

    size_t pos = 0;
    while (pos < len) {
        size_t plen = next_piece(text + pos, len - pos);
        if (plen == 0) plen = 1;

        size_t e = 0;
        for (size_t i = 0; i < plen; i++) {
            unsigned char b = (unsigned char)text[pos + i];
            e += (size_t)utf8_put(enc + e, g_byte_to_cp[b]);
        }
        vv_status_t s = bpe_piece(t, enc, e, ids, n_ids, cap_ids);
        if (s != VV_OK) { vv_free(enc); return s; }
        pos += plen;
    }
    vv_free(enc);
    return VV_OK;
}

/* ─── Public encode / decode ────────────────────────────────────────────── */

vv_status_t vv_tokenizer_encode(const vv_tokenizer_t* t, const char* text,
                                int32_t** out_ids, int* out_n) {
    if (!t || !text || !out_ids || !out_n) return VV_ERR_NULL_PTR;

    size_t len = strlen(text);
    int cap = (int)(len / 2) + 32;
    int32_t* ids = (int32_t*)vv_alloc((size_t)cap * sizeof(int32_t));
    if (!ids) return VV_ERR_OUT_OF_MEMORY;
    int n = 0;

    size_t pos = 0, run_start = 0;
    while (pos < len) {
        int hit = -1;
        if (text[pos] == '<') {
            for (int i = 0; i < t->n_special; i++) {
                size_t sl = t->sentry[i].len;
                if (pos + sl <= len &&
                    memcmp(text + pos, t->sentry[i].str, sl) == 0) {
                    hit = i;
                    break;
                }
            }
        }
        if (hit < 0) { pos++; continue; }

        if (pos > run_start) {
            vv_status_t s = encode_text_run(t, text + run_start,
                                            pos - run_start, &ids, &n, &cap);
            if (s != VV_OK) { vv_free(ids); return s; }
        }
        if (n >= cap) {
            int nc = cap * 2 + 64;
            int32_t* np = (int32_t*)vv_alloc((size_t)nc * sizeof(int32_t));
            if (!np) { vv_free(ids); return VV_ERR_OUT_OF_MEMORY; }
            memcpy(np, ids, (size_t)n * sizeof(int32_t));
            vv_free(ids);
            ids = np; cap = nc;
        }
        ids[n++] = t->sentry[hit].id;
        pos += t->sentry[hit].len;
        run_start = pos;
    }
    if (len > run_start) {
        vv_status_t s = encode_text_run(t, text + run_start,
                                        len - run_start, &ids, &n, &cap);
        if (s != VV_OK) { vv_free(ids); return s; }
    }

    *out_ids = ids;
    *out_n = n;
    return VV_OK;
}

vv_status_t vv_tokenizer_decode_ex(const vv_tokenizer_t* t,
                                   const int32_t* ids, int n_tokens,
                                   bool skip_special, char** out_text) {
    if (!t || !ids || !out_text) return VV_ERR_NULL_PTR;
    bytelevel_init();

    size_t total = 1;
    for (int i = 0; i < n_tokens; i++) {
        int32_t id = ids[i];
        if (id >= 0 && id < t->id_capacity && t->id_to_token[id])
            total += t->id_to_len[id];
    }

    char* res = (char*)vv_alloc(total + 1);
    if (!res) return VV_ERR_OUT_OF_MEMORY;
    size_t w = 0;

    for (int i = 0; i < n_tokens; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= t->id_capacity || !t->id_to_token[id]) continue;

        bool is_added = false, is_spec = false;
        for (int k = 0; k < t->n_special; k++) {
            if (t->sentry[k].id == id) {
                is_added = true;
                is_spec = t->s_is_special[k];
                break;
            }
        }
        if (is_added) {
            if (skip_special && is_spec) continue;
            /* added tokens are stored verbatim, not byte-level encoded */
            memcpy(res + w, t->id_to_token[id], t->id_to_len[id]);
            w += t->id_to_len[id];
            continue;
        }

        const char* tokstr = t->id_to_token[id];
        size_t tl = t->id_to_len[id];
        for (size_t p = 0; p < tl; ) {
            uint32_t cp;
            int a = utf8_get(tokstr + p, tl - p, &cp);
            int b = (cp < 324) ? g_cp_to_byte[cp] : -1;
            if (b >= 0) res[w++] = (char)b;
            else        w += (size_t)utf8_put(res + w, cp);
            p += (size_t)a;
        }
    }
    res[w] = '\0';
    *out_text = res;
    return VV_OK;
}

vv_status_t vv_tokenizer_decode(const vv_tokenizer_t* t,
                                const int32_t* ids, int n_tokens,
                                char** text) {
    return vv_tokenizer_decode_ex(t, ids, n_tokens, false, text);
}

/* ─── Lookups ───────────────────────────────────────────────────────────── */

int vv_tokenizer_special_id(const vv_tokenizer_t* t, const char* name) {
    if (!t || !name) return -1;
    size_t n = strlen(name);
    for (int i = 0; i < t->n_special; i++) {
        if (t->sentry[i].len == n && memcmp(t->sentry[i].str, name, n) == 0)
            return t->sentry[i].id;
    }
    return vocab_get(t, name, n);
}

const char* vv_tokenizer_id_to_token(const vv_tokenizer_t* t, int32_t id) {
    if (!t || id < 0 || id >= t->id_capacity) return NULL;
    return t->id_to_token[id];
}

int vv_tokenizer_vocab_size(const vv_tokenizer_t* t) {
    return t ? t->n_vocab + t->n_special : 0;
}

/* ─── Teardown ──────────────────────────────────────────────────────────── */

vv_status_t vv_tokenizer_free(vv_tokenizer_t* t) {
    if (!t) return VV_ERR_NULL_PTR;

    if (t->ventry) {
        for (int i = 0; i < t->n_vocab; i++) vv_free(t->ventry[i].str);
        vv_free(t->ventry);
    }
    if (t->vbucket) vv_free(t->vbucket);

    if (t->mentry) {
        for (int i = 0; i < t->n_merges; i++) {
            vv_free(t->mentry[i].a);
            vv_free(t->mentry[i].b);
        }
        vv_free(t->mentry);
    }
    if (t->mbucket) vv_free(t->mbucket);

    if (t->sentry) {
        for (int i = 0; i < t->n_special; i++) vv_free(t->sentry[i].str);
        vv_free(t->sentry);
    }
    if (t->s_is_special) vv_free(t->s_is_special);
    if (t->id_to_token)  vv_free(t->id_to_token);
    if (t->id_to_len)    vv_free(t->id_to_len);

    vv_free(t);
    return VV_OK;
}
