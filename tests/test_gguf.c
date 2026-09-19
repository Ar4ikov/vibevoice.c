/**
 * @file test_gguf.c
 * @brief GGUF reader: synthetic files, malformed files, and (when
 *        VV_BITNET_MODEL names the VibeVoice-ASR-BitNet directory) the real
 *        GGUF pair checked against the F32 safetensors it was made from.
 *
 * The real-file checks are the ground truth for the BitNet formats: they
 * ternarize a projection from the F32 latent weights with
 * vv_ternarize_f32 and compare it byte for byte with the I2_S tensor that
 * VibeASR.cpp's converter wrote, and do the same for an I8_S encoder weight.
 */

#include "vibevoice/bitnet.h"
#include "vibevoice/gguf.h"
#include "vibevoice/safetensors.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                        \
    do {                                        \
        if (!(cond)) {                          \
            printf("  FAIL: " __VA_ARGS__);     \
            printf("\n");                       \
            failures++;                         \
        }                                       \
    } while (0)

/* ─── A tiny GGUF writer ────────────────────────────────────────────────── */

typedef struct {
    uint8_t* p;
    size_t n, cap;
} buf_t;

static void put(buf_t* b, const void* d, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2 + 1024;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void put_u32(buf_t* b, uint32_t v) { put(b, &v, 4); }
static void put_u64(buf_t* b, uint64_t v) { put(b, &v, 8); }
static void put_str(buf_t* b, const char* s) { put_u64(b, strlen(s)); put(b, s, strlen(s)); }
static void pad_to(buf_t* b, size_t a) {
    static const uint8_t z[64] = { 0 };
    while (b->n % a) put(b, z, 1);
}

typedef struct {
    const char* name;
    int nd;
    uint64_t ne[4];
    uint32_t type;
    const void* data;
    size_t nbytes;
} wt_t;

static buf_t write_gguf(const wt_t* ts, int nt, uint32_t alignment) {
    buf_t b = { 0 };
    put_u32(&b, 0x46554747u);
    put_u32(&b, 3);
    put_u64(&b, (uint64_t)nt);
    put_u64(&b, 6); /* kv count */
    put_str(&b, "general.architecture"); put_u32(&b, 8); put_str(&b, "qwen2");
    put_str(&b, "qwen2.block_count"); put_u32(&b, 4); put_u32(&b, 28);
    put_str(&b, "qwen2.rope.freq_base"); put_u32(&b, 6);
    { float f = 1000000.0f; put(&b, &f, 4); }
    put_str(&b, "general.alignment"); put_u32(&b, 4); put_u32(&b, alignment);
    put_str(&b, "tokenizer.ggml.tokens"); put_u32(&b, 9); put_u32(&b, 8); put_u64(&b, 3);
    put_str(&b, "hello"); put_str(&b, "<|im_end|>"); put_str(&b, "");
    put_str(&b, "vae.depths"); put_u32(&b, 9); put_u32(&b, 5); put_u64(&b, 2);
    put_u32(&b, 3); put_u32(&b, 8);

    uint64_t off = 0;
    for (int i = 0; i < nt; i++) {
        put_str(&b, ts[i].name);
        put_u32(&b, (uint32_t)ts[i].nd);
        for (int d = 0; d < ts[i].nd; d++) put_u64(&b, ts[i].ne[d]);
        put_u32(&b, ts[i].type);
        put_u64(&b, off);
        off += (ts[i].nbytes + alignment - 1) / alignment * alignment;
    }
    pad_to(&b, alignment);
    for (int i = 0; i < nt; i++) {
        put(&b, ts[i].data, ts[i].nbytes);
        pad_to(&b, alignment);
    }
    return b;
}

/* Pack known Q6_K values (independently of the decoder): q in [-32, 31]. */
static void pack_q6k(const int8_t* q, const int8_t* sc, float d, uint8_t* blk) {
    memset(blk, 0, 210);
    uint8_t* ql = blk;
    uint8_t* qh = blk + 128;
    for (int n = 0; n < 2; n++) /* two halves of 128 */
        for (int l = 0; l < 32; l++) {
            const int base = n * 128;
            const int v0 = q[base + l] + 32, v1 = q[base + l + 32] + 32;
            const int v2 = q[base + l + 64] + 32, v3 = q[base + l + 96] + 32;
            ql[n * 64 + l] = (uint8_t)((v0 & 0xF) | ((v2 & 0xF) << 4));
            ql[n * 64 + l + 32] = (uint8_t)((v1 & 0xF) | ((v3 & 0xF) << 4));
            qh[n * 32 + l] = (uint8_t)((v0 >> 4) | ((v1 >> 4) << 2) | ((v2 >> 4) << 4) | ((v3 >> 4) << 6));
        }
    memcpy(blk + 192, sc, 16);
    const uint16_t h = vv_float_to_half(d);
    memcpy(blk + 208, &h, 2);
}

static void test_synthetic(void) {
    printf("synthetic GGUF\n");
    float f32[12];
    for (int i = 0; i < 12; i++) f32[i] = (float)i * 0.5f - 2.0f;
    uint16_t f16[8];
    for (int i = 0; i < 8; i++) f16[i] = vv_float_to_half((float)i - 3.25f);
    /* Q8_0, 2 rows of 32 */
    uint8_t q80[2 * 34];
    for (int r = 0; r < 2; r++) {
        const uint16_t d = vv_float_to_half(0.5f * (r + 1));
        memcpy(q80 + r * 34, &d, 2);
        for (int l = 0; l < 32; l++) q80[r * 34 + 2 + l] = (uint8_t)(int8_t)(l - 16);
    }
    /* Q6_K, 2 rows of 256 */
    int8_t q6[512], sc6[32];
    for (int i = 0; i < 512; i++) q6[i] = (int8_t)((i * 7) % 64 - 32);
    for (int i = 0; i < 32; i++) sc6[i] = (int8_t)(i * 5 - 70);
    uint8_t q6k[2 * 210];
    pack_q6k(q6, sc6, 0.125f, q6k);
    pack_q6k(q6 + 256, sc6 + 16, 0.25f, q6k + 210);
    /* I2_S 4 rows x 128 */
    int8_t tv[512];
    for (int i = 0; i < 512; i++) tv[i] = (int8_t)(i % 3 - 1);
    uint8_t i2s[128 + 32];
    memset(i2s, 0, sizeof(i2s));
    vv_ternary_pack(tv, 4, 128, i2s);
    const float i2scale = 0.0375f;
    memcpy(i2s + 128, &i2scale, 4);
    /* I8_S 2 x 16 */
    uint8_t i8s[32 + 32];
    memset(i8s, 0, sizeof(i8s));
    for (int i = 0; i < 32; i++) i8s[i] = (uint8_t)(int8_t)(i * 8 - 127);
    const float i8scale = 0.01f;
    memcpy(i8s + 32, &i8scale, 4);

    const wt_t ts[] = {
        { "a.f32", 2, { 4, 3 }, VV_GGML_F32, f32, sizeof(f32) },
        { "b.f16", 1, { 8 }, VV_GGML_F16, f16, sizeof(f16) },
        { "c.q8_0", 2, { 32, 2 }, VV_GGML_Q8_0, q80, sizeof(q80) },
        { "d.q6_k", 2, { 256, 2 }, VV_GGML_Q6_K, q6k, sizeof(q6k) },
        { "e.i2_s", 2, { 128, 4 }, VV_GGML_I2_S, i2s, sizeof(i2s) },
        { "f.i8_s", 2, { 16, 2 }, VV_GGML_I8_S, i8s, sizeof(i8s) },
    };
    const int nt = (int)(sizeof(ts) / sizeof(ts[0]));
    for (int ai = 0; ai < 2; ai++) {
        const uint32_t align = ai ? 64 : 32;
        buf_t b = write_gguf(ts, nt, align);
        vv_gguf_t* g = NULL;
        CHECK(vv_gguf_open_memory(b.p, b.n, &g) == VV_OK, "open (align %u)", align);
        if (!g) { free(b.p); continue; }
        CHECK(vv_gguf_version(g) == 3 && vv_gguf_n_tensors(g) == nt && vv_gguf_n_kv(g) == 6, "counts");
        CHECK(vv_gguf_alignment(g) == align && vv_gguf_data_offset(g) % align == 0, "alignment");
        CHECK(strcmp(vv_gguf_get_str(g, "general.architecture"), "qwen2") == 0, "string kv");
        CHECK(vv_gguf_get_int(g, "qwen2.block_count", -1) == 28, "u32 kv");
        CHECK(vv_gguf_get_float(g, "qwen2.rope.freq_base", 0) == 1000000.0, "f32 kv");
        CHECK(vv_gguf_get_int(g, "missing", -7) == -7, "default");
        const vv_gguf_kv_t* tok = vv_gguf_find_kv(g, "tokenizer.ggml.tokens");
        size_t len = 0;
        const char* s1 = vv_gguf_arr_str(tok, 1, &len);
        CHECK(tok && tok->arr_n == 3 && s1 && len == 10 && memcmp(s1, "<|im_end|>", 10) == 0, "string array");
        const char* s2 = vv_gguf_arr_str(tok, 2, &len);
        CHECK(s2 && len == 0, "empty string element");
        const vv_gguf_kv_t* dep = vv_gguf_find_kv(g, "vae.depths");
        CHECK(dep && dep->arr_type == VV_GGUF_I32 && dep->arr_n == 2, "i32 array");

        float out[512];
        const vv_gguf_tensor_t* t;
        t = vv_gguf_find_tensor(g, "a.f32");
        CHECK(t && t->ne[0] == 4 && t->ne[1] == 3 && vv_gguf_dequant_rows_f32(t, 1, 2, out) == VV_OK &&
              memcmp(out, f32 + 4, 8 * 4) == 0, "f32 rows");
        t = vv_gguf_find_tensor(g, "b.f16");
        CHECK(t && vv_gguf_dequant_rows_f32(t, 0, 1, out) == VV_OK && out[7] == 3.75f, "f16");
        t = vv_gguf_find_tensor(g, "c.q8_0");
        CHECK(t && vv_gguf_dequant_rows_f32(t, 1, 1, out) == VV_OK && out[0] == -16.0f && out[31] == 15.0f,
              "q8_0");
        t = vv_gguf_find_tensor(g, "d.q6_k");
        CHECK(t && t->nbytes == 420 && vv_gguf_dequant_rows_f32(t, 0, 2, out) == VV_OK, "q6_k decode");
        int bad = 0;
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 256; i++) {
                const float d = r ? 0.25f : 0.125f;
                const float want = d * sc6[r * 16 + i / 16] * q6[r * 256 + i];
                if (out[r * 256 + i] != want) bad = 1;
            }
        CHECK(!bad, "q6_k values");
        t = vv_gguf_find_tensor(g, "e.i2_s");
        const uint8_t* codes = NULL;
        float sc = 0;
        CHECK(t && t->nbytes == 160 && vv_gguf_i2s_view(t, &codes, &sc) == VV_OK && sc == i2scale &&
              memcmp(codes, i2s, 128) == 0, "i2_s view");
        CHECK(vv_gguf_dequant_rows_f32(t, 2, 1, out) == VV_OK && out[0] == i2scale * tv[256] &&
              out[127] == i2scale * tv[383], "i2_s dequant");
        t = vv_gguf_find_tensor(g, "f.i8_s");
        const int8_t* q = NULL;
        CHECK(t && t->nbytes == 64 && vv_gguf_i8s_view(t, &q, &sc) == VV_OK && sc == i8scale &&
              q[3] == -103, "i8_s view");
        uint16_t h[16];
        CHECK(vv_gguf_dequant_rows_f16(t, 1, 1, h) == VV_OK, "to f16");

        /* every truncation must be refused, never read past the end */
        int refused = 0, tried = 0;
        for (size_t cut = 0; cut < b.n; cut += (cut < 512 ? 1 : 61)) {
            vv_gguf_t* g2 = NULL;
            uint8_t* copy = malloc(cut ? cut : 1);
            memcpy(copy, b.p, cut);
            tried++;
            if (vv_gguf_open_memory(copy, cut, &g2) != VV_OK) refused++;
            else vv_gguf_close(g2);
            free(copy);
        }
        CHECK(refused == tried, "truncated files: %d of %d refused", refused, tried);
        vv_gguf_close(g);

        /* corrupt: bad magic, absurd counts, a tensor offset past the end */
        uint8_t* c = malloc(b.n);
        memcpy(c, b.p, b.n);
        c[0] ^= 0xFF;
        CHECK(vv_gguf_open_memory(c, b.n, &g) != VV_OK, "bad magic");
        memcpy(c, b.p, b.n);
        const uint64_t huge = (uint64_t)1 << 60;
        memcpy(c + 8, &huge, 8);
        CHECK(vv_gguf_open_memory(c, b.n, &g) != VV_OK, "huge tensor count");
        memcpy(c, b.p, b.n);
        memcpy(c + 16, &huge, 8);
        CHECK(vv_gguf_open_memory(c, b.n, &g) != VV_OK, "huge kv count");
        free(c);
        free(b.p);
    }
    {
        /* a tensor that claims more bytes than the file holds */
        float big[4] = { 0 };
        wt_t lie = { "x", 2, { 4096, 4096 }, VV_GGML_F32, big, sizeof(big) };
        buf_t b = write_gguf(&lie, 1, 32);
        vv_gguf_t* g = NULL;
        CHECK(vv_gguf_open_memory(b.p, b.n, &g) != VV_OK, "oversized tensor");
        free(b.p);
    }
}

/* ─── Real files ────────────────────────────────────────────────────────── */

/** Find a tensor in any of the three BitNet safetensors shards. */
static const float* st_find(vv_safetensors_t** sts, int nst, const char* name,
                            vv_st_tensor_info_t* info) {
    for (int i = 0; i < nst; i++) {
        if (sts[i] && vv_safetensors_find(sts[i], name, info) == VV_OK) {
            const void* p = NULL;
            if (info->dtype != VV_DTYPE_F32) return NULL;
            vv_safetensors_get_data(sts[i], info, &p);
            return (const float*)p;
        }
    }
    return NULL;
}

static void test_real(const char* dir) {
    char path[1024];
    printf("real BitNet GGUF files in %s\n", dir);
    snprintf(path, sizeof(path), "%s/vibeasr-lm-i2_s-embed-q6_k.gguf", dir);
    vv_gguf_t* lm = NULL;
    if (vv_gguf_open(path, &lm) != VV_OK) {
        CHECK(0, "cannot open %s", path);
        return;
    }
    snprintf(path, sizeof(path), "%s/vibeasr-vae-encoder-i8_s.gguf", dir);
    vv_gguf_t* vae = NULL;
    CHECK(vv_gguf_open(path, &vae) == VV_OK, "open VAE gguf");

    CHECK(vv_gguf_n_tensors(lm) == 339, "LM tensors %d", vv_gguf_n_tensors(lm));
    CHECK(strcmp(vv_gguf_get_str(lm, "general.architecture"), "qwen2") == 0, "arch");
    CHECK(vv_gguf_get_int(lm, "qwen2.embedding_length", 0) == 1536 &&
          vv_gguf_get_int(lm, "qwen2.attention.head_count", 0) == 12 &&
          vv_gguf_get_int(lm, "qwen2.attention.head_count_kv", 0) == 2 &&
          vv_gguf_get_int(lm, "qwen2.feed_forward_length", 0) == 8960, "LM hparams");
    int n_i2s = 0, n_q6k = 0;
    for (int i = 0; i < vv_gguf_n_tensors(lm); i++) {
        const vv_gguf_tensor_t* t = vv_gguf_tensor(lm, i);
        n_i2s += t->type == VV_GGML_I2_S;
        n_q6k += t->type == VV_GGML_Q6_K;
        CHECK(t->data != NULL, "tensor %s has no data", t->name);
    }
    CHECK(n_i2s == 196 && n_q6k == 1, "types: %d I2_S, %d Q6_K", n_i2s, n_q6k);
    const vv_gguf_tensor_t* out = vv_gguf_find_tensor(lm, "output.weight");
    CHECK(out && out->type == VV_GGML_F16, "output.weight is F16");

    /* the F32 latent weights the GGUF was converted from */
    vv_safetensors_t* sts[3] = { NULL, NULL, NULL };
    int have_st = 0;
    for (int i = 0; i < 3; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-00003.safetensors", dir, i + 1);
        if (vv_safetensors_open(path, &sts[i]) == VV_OK) have_st++;
    }
    if (have_st < 3) {
        printf("  SKIP safetensors cross-check: shards not found\n");
    } else {
        static const char* PROJ[][2] = {
            { "self_attn.q_proj", "attn_q" }, { "self_attn.k_proj", "attn_k" },
            { "self_attn.v_proj", "attn_v" }, { "self_attn.o_proj", "attn_output" },
            { "mlp.gate_proj", "ffn_gate" },  { "mlp.up_proj", "ffn_up" },
            { "mlp.down_proj", "ffn_down" },
        };
        static const int LAYERS[] = { 0, 13, 27 };
        int64_t total = 0, mism = 0, scale_eq = 0, n_checked = 0;
        for (int li = 0; li < 3; li++)
            for (int p = 0; p < 7; p++) {
                char sname[256], gname[256];
                snprintf(sname, sizeof(sname), "model.language_model.layers.%d.%s.weight", LAYERS[li], PROJ[p][0]);
                snprintf(gname, sizeof(gname), "blk.%d.%s.weight", LAYERS[li], PROJ[p][1]);
                vv_st_tensor_info_t info;
                const float* w = st_find(sts, 3, sname, &info);
                const vv_gguf_tensor_t* gt = vv_gguf_find_tensor(lm, gname);
                if (!w || !gt) { CHECK(0, "missing %s / %s", sname, gname); continue; }
                const int64_t N = info.shape[0], K = info.shape[1];
                CHECK(gt->ne[0] == K && gt->ne[1] == N, "%s shape", gname);
                uint8_t* codes = malloc((size_t)(N * K / 4));
                float sc = 0;
                vv_ternarize_f32(w, N, K, codes, &sc);
                const uint8_t* gc;
                float gs;
                vv_gguf_i2s_view(gt, &gc, &gs);
                int64_t m = 0;
                for (int64_t i = 0; i < N * K / 4; i++) {
                    const uint8_t x = codes[i] ^ gc[i];
                    for (int f = 0; f < 4; f++) m += ((x >> (2 * f)) & 3) != 0;
                }
                total += N * K;
                mism += m;
                scale_eq += sc == gs;
                n_checked++;
                if (m || sc != gs)
                    printf("  %s: %lld of %lld codes differ, scale %.9g vs gguf %.9g\n", gname,
                           (long long)m, (long long)(N * K), sc, gs);
                free(codes);
            }
        printf("  ternarize vs GGUF: %lld / %lld codes differ over %lld tensors, %lld scales equal\n",
               (long long)mism, (long long)total, (long long)n_checked, (long long)scale_eq);
        /* the formula must be the converter's: allow a handful of codes that
         * sit exactly on the rounding boundary (torch's FP32 mean vs ours) */
        CHECK(mism * 100000 <= total, "too many ternary mismatches");

        /* Q6_K embedding vs the F32 table */
        vv_st_tensor_info_t info;
        const float* emb = st_find(sts, 3, "model.language_model.embed_tokens.weight", &info);
        const vv_gguf_tensor_t* te = vv_gguf_find_tensor(lm, "token_embd.weight");
        if (emb && te) {
            float row[1536];
            double err = 0, mag = 0;
            static const int ROWS[] = { 0, 198, 151643, 151648, 100000 };
            for (int r = 0; r < 5; r++) {
                vv_gguf_dequant_rows_f32(te, ROWS[r], 1, row);
                for (int k = 0; k < 1536; k++) {
                    const double d = row[k] - emb[(size_t)ROWS[r] * 1536 + k];
                    err += d * d;
                    mag += (double)emb[(size_t)ROWS[r] * 1536 + k] * emb[(size_t)ROWS[r] * 1536 + k];
                }
            }
            printf("  Q6_K embedding relative RMS error %.3e\n", sqrt(err / mag));
            CHECK(sqrt(err / mag) < 0.02, "Q6_K embedding too far from F32");
            /* output.weight is an F16 copy of the same table */
            uint16_t h[1536];
            vv_gguf_dequant_rows_f16(out, 151648, 1, h);
            int same = 1;
            for (int k = 0; k < 1536; k++)
                same &= h[k] == vv_float_to_half(emb[(size_t)151648 * 1536 + k]);
            printf("  output.weight row 151648 %s the F32 embedding rounded to F16\n",
                   same ? "equals" : "differs from");
        }

        /* I8_S: one encoder FFN weight, quantized here vs in the GGUF */
        static const char* VPAIR[][2] = {
            { "model.acoustic_tokenizer.encoder.stages.0.0.ffn.linear1.weight", "acoustic.stages.0.0.ffn.linear1.weight" },
            { "model.semantic_tokenizer.encoder.stages.6.7.ffn.linear2.weight", "semantic.stages.6.7.ffn.linear2.weight" },
            { "model.acoustic_connector.fc2.weight", "acoustic_connector.fc2.weight" },
        };
        for (int v = 0; v < 3 && vae; v++) {
            const float* w = st_find(sts, 3, VPAIR[v][0], &info);
            const vv_gguf_tensor_t* gt = vv_gguf_find_tensor(vae, VPAIR[v][1]);
            if (!w || !gt) { CHECK(0, "missing %s", VPAIR[v][0]); continue; }
            const int64_t n = vv_gguf_nelements(gt);
            int8_t* q = malloc((size_t)n);
            float sc = 0;
            vv_i8s_quantize_f32(w, 1, n, q, &sc);
            const int8_t* gq;
            float gs;
            vv_gguf_i8s_view(gt, &gq, &gs);
            int64_t m = 0;
            for (int64_t i = 0; i < n; i++) m += q[i] != gq[i];
            printf("  I8_S %s: %lld / %lld differ, scale %.9g vs %.9g\n", VPAIR[v][1],
                   (long long)m, (long long)n, sc, gs);
            CHECK(m == 0 && sc == gs, "I8_S quantization differs from the GGUF");
            free(q);
        }
    }
    for (int i = 0; i < 3; i++)
        if (sts[i]) vv_safetensors_close(sts[i]);
    vv_gguf_close(vae);
    vv_gguf_close(lm);
}

int main(void) {
    test_synthetic();
    const char* dir = getenv("VV_BITNET_MODEL");
    if (dir && *dir) test_real(dir);
    else printf("SKIP real files: set VV_BITNET_MODEL to the VibeVoice-ASR-BitNet directory\n");
    if (failures) {
        printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("all gguf checks passed\n");
    return 0;
}
