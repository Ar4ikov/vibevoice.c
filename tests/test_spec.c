/**
 * @file test_spec.c
 * @brief Speculative decoding (spec.h): the kernels that check a drafted
 *        block must give every row exactly what its own decode step would,
 *        and the drafter's kernels must do what the trainer does.
 *
 * Bit for bit, against the one-token kernels:
 *   - vv_w4a16_gemv_rows_dev against vv_w4a16_gemv_multi_dev row by row, at
 *     the 7B and 1.5B projection shapes (q/k/v fused, gate/up, down), M 1..16;
 *   - vv_attn_decode_rows against vv_attn_decode at each row's own length,
 *     for fa1, fa2 and flashinfer, FP16 and TurboQuant caches, lengths that
 *     cross a split-count step inside the block, contiguous and paged;
 *   - vv_lm_head_rows_dev / vv_argmax_rows_dev against the one-row head;
 *   - the BitNet projection a checked block runs (per-token int8
 *     quantization, then the ternary product) against one row at a time.
 * Against CPU references: the in-block convolution, top-k, the selector walk.
 * Without a GPU: the drafter's config.json is read and checked (what the
 * runtime refuses, and why).
 *
 * With VV_TEST_MODEL (a checkpoint) and VV_TEST_DRAFT (a drafter for it)
 * set, the jfk clip (VV_TEST_AUDIO, else tests/data/jfk.wav) is transcribed
 * with and without the drafter and the generated ids must be identical.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/quant.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/inference.h"
#include "vibevoice/spec.h"
#include "vibevoice/audio.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 0x12345u;
static uint32_t urand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float frand(void) { return ((float)(urand() >> 8) / 8388608.0f) - 1.0f; }

static int failures = 0;
#define CHECK(cond, ...) do {                                                 \
        if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n");          \
                       failures++; }                                          \
    } while (0)

static void* dev_upload(const void* h, size_t n) {
    void* d = NULL;
    if (vv_dev_alloc(&d, n) != VV_OK) return NULL;
    vv_dev_memcpy_h2d(d, h, n, NULL);
    return d;
}

static void fill_half(uint16_t* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = vv_float_to_half(frand() * scale);
}

/* ─── Drafter config (no GPU) ────────────────────────────────────────────── */

static const char* k_cfg =
    "{\"architectures\":[\"DFlash2DraftModel\"],\"hidden_size\":1536,"
    "\"intermediate_size\":4480,\"num_hidden_layers\":5,"
    "\"num_attention_heads\":12,\"num_key_value_heads\":2,\"head_dim\":128,"
    "\"rms_norm_eps\":1e-06,\"rope_theta\":1000000.0,\"vocab_size\":151936,"
    "\"num_target_layers\":28,\"layer_types\":[\"%s\"],\"sliding_window\":64,"
    "\"dflash_config\":{\"block_size\":8,\"mask_token_id\":151662,"
    "\"target_layer_ids\":[1,7,13,19,25],\"conv_kernel_size\":2,"
    "\"conv_group_size\":16,\"selector_rank\":256,\"selector_top_k\":16,"
    "\"draft_vocab_size\":%s}%s}";

static vv_status_t cfg_try(const char* dir, const char* vocab,
                           const char* layers, vv_drafter_config_t* c) {
    char path[1024], body[2048];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    snprintf(body, sizeof(body), k_cfg, layers, vocab, "");
    FILE* f = fopen(path, "wb");
    if (!f) return VV_ERR_IO;
    fputs(body, f);
    fclose(f);
    return vv_drafter_config_load(dir, c);
}

static void test_drafter_config(void) {
    const char* tmp = getenv("TMPDIR");
    char dir[1024];
#ifdef _WIN32
    snprintf(dir, sizeof(dir), "%s", getenv("TEMP") ? getenv("TEMP") : ".");
#else
    snprintf(dir, sizeof(dir), "%s", tmp && tmp[0] ? tmp : "/tmp");
#endif
    vv_drafter_config_t c;
    vv_status_t s = cfg_try(dir, "28917", "full_attention", &c);
    CHECK(s == VV_OK, "drafter config: %s", vv_status_str(s));
    if (s == VV_OK) {
        CHECK(c.hidden_size == 1536 && c.num_layers == 5 && c.num_heads == 12 &&
              c.num_kv_heads == 2 && c.head_dim == 128 &&
              c.intermediate_size == 4480 && c.vocab_size == 151936 &&
              c.block_size == 8 && c.mask_token_id == 151662 &&
              c.n_taps == 5 && c.target_layer_ids[4] == 25 &&
              c.conv_kernel == 2 && c.conv_group == 16 &&
              c.selector_rank == 256 && c.selector_top_k == 16 &&
              c.draft_vocab_size == 28917 &&
              c.weight_quant == VV_DRAFTER_INT4,
              "drafter config: fields read back wrong");
        /* INT4 holds a quarter of FP16's projection bytes, plus scales. */
        const size_t b4 = vv_drafter_weight_bytes(&c);
        c.weight_quant = VV_DRAFTER_F16;
        const size_t b16 = vv_drafter_weight_bytes(&c);
        CHECK(b4 < b16, "drafter bytes: int4 %zu !< f16 %zu", b4, b16);
    }
    /* What the kernels cannot run is refused at load, whole. */
    CHECK(cfg_try(dir, "8", "full_attention", &c) == VV_ERR_MODEL_FORMAT,
          "a draft vocabulary smaller than top_k must be refused");
    CHECK(cfg_try(dir, "200000", "full_attention", &c) == VV_ERR_MODEL_FORMAT,
          "a draft vocabulary larger than the vocabulary must be refused");
    CHECK(cfg_try(dir, "0", "sliding_attention", &c) == VV_ERR_MODEL_FORMAT,
          "a sliding-window drafter must be refused");
    CHECK(vv_drafter_quant_parse("int4") == VV_DRAFTER_INT4 &&
          vv_drafter_quant_parse("f16") == VV_DRAFTER_F16 &&
          vv_drafter_quant_parse("int8") == VV_DRAFTER_QUANT_COUNT,
          "--draft-quant names");
    char path[1100];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    remove(path);
    if (!failures) printf("  ok   drafter config.json: read, sized, refused\n");
}

/* ─── W4A16 rows ─────────────────────────────────────────────────────────── */

typedef struct { void* packed; void* sz; void* bias; int N; } gw_t;

static int make_w4(gw_t* w, int N, int K, int G, bool bias) {
    float* f = (float*)malloc((size_t)N * K * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc((size_t)N * K / 2);
    uint16_t* sc = (uint16_t*)malloc((size_t)N * (K / G) * 2);
    uint16_t* mn = (uint16_t*)malloc((size_t)N * (K / G) * 2);
    uint8_t* zr = (uint8_t*)malloc((size_t)N * (K / G));
    uint16_t* sz = (uint16_t*)malloc((size_t)N * (K / G) * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) f[i] = frand() * 0.05f;
    int ok = vv_int4g_quantize(f, N, K, G, packed, sc, mn, zr) == VV_OK &&
             vv_int4g_to_gpu_layout(packed, sc, zr, N, K, G, sz) == VV_OK;
    w->N = N;
    w->packed = ok ? dev_upload(packed, (size_t)N * K / 2) : NULL;
    w->sz = ok ? dev_upload(sz, (size_t)N * (K / G) * 4) : NULL;
    w->bias = NULL;
    if (ok && bias) {
        uint16_t* b = (uint16_t*)malloc((size_t)N * 2);
        fill_half(b, (size_t)N, 0.5f);
        w->bias = dev_upload(b, (size_t)N * 2);
        free(b);
    }
    free(f); free(packed); free(sc); free(mn); free(zr); free(sz);
    return ok && w->packed && w->sz;
}

static void test_w4a16_rows(const char* name, const int* Ns, int n, int K,
                            int G, bool bias) {
    gw_t w[3];
    for (int i = 0; i < n; i++)
        if (!make_w4(&w[i], Ns[i], K, G, bias)) {
            printf("  SKIP %s: no GPU weights\n", name);
            return;
        }
    const int MMAX = 16;
    uint16_t* xh = (uint16_t*)malloc((size_t)MMAX * K * 2);
    fill_half(xh, (size_t)MMAX * K, 2.0f);
    void* x = dev_upload(xh, (size_t)MMAX * K * 2);
    void *ya[3], *yb[3];
    size_t nb[3];
    for (int i = 0; i < n; i++) {
        nb[i] = (size_t)MMAX * Ns[i] * 2;
        vv_dev_alloc(&ya[i], nb[i]);
        vv_dev_alloc(&yb[i], nb[i]);
    }
    const int Ms[] = { 1, 2, 3, 7, 8, 9, 16 };
    int bad = 0;
    for (size_t mi = 0; mi < sizeof(Ms) / sizeof(Ms[0]); mi++) {
        const int M = Ms[mi];
        vv_w4a16_proj_t p[3];
        for (int i = 0; i < n; i++) {
            p[i].packed = w[i].packed; p[i].sz = w[i].sz; p[i].bias = w[i].bias;
            p[i].N = Ns[i]; p[i].y = ya[i];
        }
        vv_status_t s = vv_w4a16_gemv_rows_dev(x, M, p, n, K, G, NULL);
        for (int m = 0; s == VV_OK && m < M; m++) {
            for (int i = 0; i < n; i++)
                p[i].y = (uint8_t*)yb[i] + (size_t)m * Ns[i] * 2;
            s = vv_w4a16_gemv_multi_dev((const uint8_t*)x + (size_t)m * K * 2,
                                        p, n, K, G, NULL);
        }
        vv_dev_stream_sync(NULL);
        CHECK(s == VV_OK, "%s M=%d: %s", name, M, vv_status_str(s));
        for (int i = 0; s == VV_OK && i < n; i++) {
            uint16_t* a = (uint16_t*)malloc((size_t)M * Ns[i] * 2);
            uint16_t* b = (uint16_t*)malloc((size_t)M * Ns[i] * 2);
            vv_dev_memcpy_d2h(a, ya[i], (size_t)M * Ns[i] * 2, NULL);
            vv_dev_memcpy_d2h(b, yb[i], (size_t)M * Ns[i] * 2, NULL);
            if (memcmp(a, b, (size_t)M * Ns[i] * 2) != 0) bad++;
            free(a); free(b);
        }
    }
    CHECK(bad == 0, "%s: %d (M, projection) pairs differ from the one-row GEMV",
          name, bad);
    if (!bad) printf("  ok   %s: rows == one-row GEMV, M 1..16\n", name);
    /* VV_SPEC_BENCH=1: one token against a block of 8, per launch. */
    if (getenv("VV_SPEC_BENCH") && getenv("VV_SPEC_BENCH")[0] == '1') {
        void *e0 = NULL, *e1 = NULL;
        vv_dev_event_create(&e0);
        vv_dev_event_create(&e1);
        vv_w4a16_proj_t p[3];
        for (int i = 0; i < n; i++) {
            p[i].packed = w[i].packed; p[i].sz = w[i].sz; p[i].bias = w[i].bias;
            p[i].N = Ns[i]; p[i].y = ya[i];
        }
        const int bm[4] = { 2, 3, 4, 8 };
        double t1 = 0.0, tm[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int rep = 0; rep < 2; rep++) {
            const int it = 50;
            double t0 = vv_time_ms();
            for (int k = 0; k < it; k++)
                vv_w4a16_gemv_multi_dev(x, p, n, K, G, NULL);
            vv_dev_stream_sync(NULL);
            t1 = (vv_time_ms() - t0) * 1000.0 / it;
            for (int b = 0; b < 4; b++) {
                t0 = vv_time_ms();
                for (int k = 0; k < it; k++)
                    vv_w4a16_gemv_rows_dev(x, bm[b], p, n, K, G, NULL);
                vv_dev_stream_sync(NULL);
                tm[b] = (vv_time_ms() - t0) * 1000.0 / it;
            }
        }
        printf("  bench %s: 1 row %.1f us; rows 2/3/4/8: %.2fx %.2fx %.2fx %.2fx\n",
               name, t1, tm[0] / t1, tm[1] / t1, tm[2] / t1, tm[3] / t1);
        vv_dev_event_destroy(e0);
        vv_dev_event_destroy(e1);
    }
    for (int i = 0; i < n; i++) {
        vv_dev_free(w[i].packed); vv_dev_free(w[i].sz);
        if (w[i].bias) vv_dev_free(w[i].bias);
        vv_dev_free(ya[i]); vv_dev_free(yb[i]);
    }
    vv_dev_free(x);
    free(xh);
}

/* ─── Decode attention rows ──────────────────────────────────────────────── */

static void test_attn_rows(int backend, int fmt, bool paged, int n_q, int n_kv,
                           int L0, int M) {
    const int D = 128;
    const int cap = ((L0 + M + 63) / 64 + 1) * 64;
    const int bpv = vv_kv_bytes_per_vec((vv_kv_format_t)fmt, D);
    const bool meta = vv_kv_has_meta((vv_kv_format_t)fmt);
    char name[128];
    snprintf(name, sizeof(name), "attn rows %s %s%s H%d/%d L%d+%d",
             vv_attn_backend_name((vv_attn_backend_t)backend),
             vv_kv_format_name((vv_kv_format_t)fmt), paged ? " paged" : "",
             n_q, n_kv, L0, M);
    const int b = vv_attn_resolve(backend, fmt, paged, n_q, n_kv, D);
    if (b != backend) { printf("  SKIP %s: resolves to another backend\n", name); return; }

    uint16_t* kh = (uint16_t*)malloc((size_t)cap * n_kv * D * 2);
    uint16_t* vh = (uint16_t*)malloc((size_t)cap * n_kv * D * 2);
    fill_half(kh, (size_t)cap * n_kv * D, 1.5f);
    fill_half(vh, (size_t)cap * n_kv * D, 1.0f);
    void* kf = dev_upload(kh, (size_t)cap * n_kv * D * 2);
    void* vf = dev_upload(vh, (size_t)cap * n_kv * D * 2);
    void *ks = NULL, *vs = NULL, *km = NULL, *vm = NULL, *kref = NULL;
    vv_dev_alloc(&ks, (size_t)cap * n_kv * bpv);
    vv_dev_alloc(&vs, (size_t)cap * n_kv * bpv);
    if (meta) {
        vv_dev_alloc(&km, (size_t)cap * n_kv * 2);
        vv_dev_alloc(&vm, (size_t)cap * n_kv * 2);
    }
    vv_dev_alloc(&kref, (size_t)n_kv * D * 2);
    vv_dev_memset(kref, 0, (size_t)n_kv * D * 2);
    /* Pages in scrambled order, so a table that is ignored shows. */
    int* pages = NULL;
    int* table = NULL;
    const int n_pages = cap / 64;
    if (paged) {
        pages = (int*)malloc(sizeof(int) * (size_t)n_pages);
        for (int i = 0; i < n_pages; i++) pages[i] = n_pages - 1 - i;
        vv_dev_alloc((void**)&table, sizeof(int) * (size_t)n_pages);
        vv_kv_page_map_dev(table, 0, n_pages, pages, NULL);
    }
    vv_status_t s = vv_kv_store_dev(kf, vf, ks, vs, km, vm, kref,
                                    !vv_kv_is_raw((vv_kv_format_t)fmt), n_kv, D,
                                    0, NULL, cap, fmt, table, NULL);
    vv_kv_view_t view;
    memset(&view, 0, sizeof(view));
    view.k = ks; view.v = vs; view.k_meta = km; view.v_meta = vm;
    view.page_table = table; view.format = fmt;
    view.n_kv_heads = n_kv; view.head_dim = D;

    uint16_t* qh = (uint16_t*)malloc((size_t)M * n_q * D * 2);
    fill_half(qh, (size_t)M * n_q * D, 1.0f);
    void* q = dev_upload(qh, (size_t)M * n_q * D * 2);
    void *oa = NULL, *ob = NULL, *scr = NULL;
    vv_dev_alloc(&oa, (size_t)M * n_q * D * 2);
    vv_dev_alloc(&ob, (size_t)M * n_q * D * 2);
    size_t sb = vv_attn_rows_scratch_bytes(n_q, D, M);
    const size_t s1 = vv_attn_scratch_bytes(n_q, n_kv, D);
    if (sb < s1) sb = s1;
    vv_dev_alloc(&scr, sb);
    if (s == VV_OK)
        s = vv_attn_decode_rows(b, q, &view, oa, n_q, M, L0 + 1, NULL, scr, NULL);
    for (int r = 0; s == VV_OK && r < M; r++)
        s = vv_attn_decode(b, (const uint8_t*)q + (size_t)r * n_q * D * 2,
                           &view, (uint8_t*)ob + (size_t)r * n_q * D * 2, n_q,
                           L0 + 1 + r, NULL, scr, NULL);
    vv_dev_stream_sync(NULL);
    if (s != VV_OK) {
        printf("  SKIP %s: %s\n", name, vv_status_str(s));
    } else {
        uint16_t* a = (uint16_t*)malloc((size_t)M * n_q * D * 2);
        uint16_t* c = (uint16_t*)malloc((size_t)M * n_q * D * 2);
        vv_dev_memcpy_d2h(a, oa, (size_t)M * n_q * D * 2, NULL);
        vv_dev_memcpy_d2h(c, ob, (size_t)M * n_q * D * 2, NULL);
        int bad_rows = 0;
        for (int r = 0; r < M; r++)
            if (memcmp(a + (size_t)r * n_q * D, c + (size_t)r * n_q * D,
                       (size_t)n_q * D * 2) != 0)
                bad_rows++;
        CHECK(bad_rows == 0, "%s: %d of %d rows differ from their own decode",
              name, bad_rows, M);
        if (!bad_rows) printf("  ok   %s\n", name);
        free(a); free(c);
    }
    vv_dev_free(kf); vv_dev_free(vf); vv_dev_free(ks); vv_dev_free(vs);
    if (km) vv_dev_free(km);
    if (vm) vv_dev_free(vm);
    vv_dev_free(kref); vv_dev_free(q); vv_dev_free(oa); vv_dev_free(ob);
    vv_dev_free(scr);
    if (table) vv_dev_free(table);
    free(pages); free(kh); free(vh); free(qh);
}

/* ─── Ternary rows (BitNet) ──────────────────────────────────────────────── */

/*
 * A BitNet projection of a checked block quantizes each row to int8 on its
 * own and runs the dp4a product for up to 8 rows; its integers are exact in
 * any order and the scaling is per element, so every row must come out as
 * the one-row call gives it. Shapes of the 1.5B model's projections.
 */
static void test_ternary_rows(int N, int K) {
    const int MAXM = 8;
    uint8_t* codes = (uint8_t*)malloc((size_t)N * K / 4);
    for (size_t i = 0; i < (size_t)N * K / 4; i++) {
        uint8_t b = 0;
        for (int j = 0; j < 4; j++) b |= (uint8_t)((urand() % 3) << (2 * j));
        codes[i] = b;
    }
    uint16_t* xh = (uint16_t*)malloc((size_t)MAXM * K * 2);
    fill_half(xh, (size_t)MAXM * K, 3.0f);
    float* bias = (float*)malloc((size_t)N * 4);
    for (int i = 0; i < N; i++) bias[i] = frand();
    void* dc = dev_upload(codes, (size_t)N * K / 4);
    void* dx = dev_upload(xh, (size_t)MAXM * K * 2);
    void* db = dev_upload(bias, (size_t)N * 4);
    void *q = NULL, *sc = NULL, *sum = NULL, *ya = NULL, *yb = NULL;
    vv_dev_alloc(&q, (size_t)MAXM * K);
    vv_dev_alloc(&sc, (size_t)MAXM * 4);
    vv_dev_alloc(&sum, (size_t)MAXM * 4);
    vv_dev_alloc(&ya, (size_t)MAXM * N * 2);
    vv_dev_alloc(&yb, (size_t)MAXM * N * 2);
    const float w_scale = 0.0123f;
    int bad = 0;
    vv_status_t s = VV_OK;
    for (int M = 2; s == VV_OK && M <= MAXM; M++) {
        s = vv_act_quant_i8_dev(dx, 1, M, K, (int8_t*)q, (float*)sc,
                                (int32_t*)sum, NULL);
        if (s == VV_OK)
            s = vv_ternary_gemm_dev((const int8_t*)q, (const int32_t*)sum,
                                    (const float*)sc, (const uint8_t*)dc,
                                    w_scale, (const float*)db, NULL, ya, 1,
                                    M, N, K, NULL);
        for (int m = 0; s == VV_OK && m < M; m++) {
            s = vv_act_quant_i8_dev((const uint8_t*)dx + (size_t)m * K * 2, 1,
                                    1, K, (int8_t*)q, (float*)sc,
                                    (int32_t*)sum, NULL);
            if (s == VV_OK)
                s = vv_ternary_gemm_dev((const int8_t*)q, (const int32_t*)sum,
                                        (const float*)sc, (const uint8_t*)dc,
                                        w_scale, (const float*)db, NULL,
                                        (uint8_t*)yb + (size_t)m * N * 2, 1,
                                        1, N, K, NULL);
        }
        vv_dev_stream_sync(NULL);
        if (s == VV_OK) {
            uint16_t* a = (uint16_t*)malloc((size_t)M * N * 2);
            uint16_t* b = (uint16_t*)malloc((size_t)M * N * 2);
            vv_dev_memcpy_d2h(a, ya, (size_t)M * N * 2, NULL);
            vv_dev_memcpy_d2h(b, yb, (size_t)M * N * 2, NULL);
            if (memcmp(a, b, (size_t)M * N * 2) != 0) bad++;
            free(a); free(b);
        }
    }
    CHECK(s == VV_OK, "ternary rows N%d K%d: %s", N, K, vv_status_str(s));
    CHECK(bad == 0, "ternary rows N%d K%d: %d block sizes differ from one row "
          "at a time", N, K, bad);
    if (s == VV_OK && !bad)
        printf("  ok   ternary rows N%d K%d == one-row, M 2..8\n", N, K);
    vv_dev_free(dc); vv_dev_free(dx); vv_dev_free(db); vv_dev_free(q);
    vv_dev_free(sc); vv_dev_free(sum); vv_dev_free(ya); vv_dev_free(yb);
    free(codes); free(xh); free(bias);
}

/* ─── Head rows ──────────────────────────────────────────────────────────── */

static void test_head_rows(int V, int K, int M) {
    uint16_t* wh = (uint16_t*)malloc((size_t)V * K * 2);
    uint16_t* xh = (uint16_t*)malloc((size_t)M * K * 2);
    fill_half(wh, (size_t)V * K, 0.05f);
    fill_half(xh, (size_t)M * K, 2.0f);
    void* W = dev_upload(wh, (size_t)V * K * 2);
    void* x = dev_upload(xh, (size_t)M * K * 2);
    void *la = NULL, *lb = NULL, *scr = NULL, *sv = NULL, *si = NULL;
    int32_t *ta = NULL, *tb = NULL;
    vv_dev_alloc(&la, (size_t)M * V * 4);
    vv_dev_alloc(&lb, (size_t)M * V * 4);
    vv_dev_alloc(&scr, vv_argmax_rows_scratch_bytes(M));
    vv_dev_alloc(&sv, 256 * 4);
    vv_dev_alloc(&si, 256 * 4);
    vv_dev_alloc((void**)&ta, (size_t)M * 4);
    vv_dev_alloc((void**)&tb, (size_t)M * 4);
    vv_status_t s = vv_lm_head_rows_dev(x, W, la, M, V, K, NULL);
    if (s == VV_OK) s = vv_argmax_rows_dev(la, M, V, scr, ta, NULL);
    for (int m = 0; s == VV_OK && m < M; m++) {
        s = vv_lm_head_gemv_dev((const uint8_t*)x + (size_t)m * K * 2, W,
                                (uint8_t*)lb + (size_t)m * V * 4, V, K, NULL);
        if (s == VV_OK)
            s = vv_argmax_dev((const uint8_t*)lb + (size_t)m * V * 4, V, sv, si,
                              tb + m, NULL, NULL);
    }
    vv_dev_stream_sync(NULL);
    CHECK(s == VV_OK, "head rows: %s", vv_status_str(s));
    if (s == VV_OK) {
        float* a = (float*)malloc((size_t)M * V * 4);
        float* b = (float*)malloc((size_t)M * V * 4);
        int32_t at[16], bt[16];
        vv_dev_memcpy_d2h(a, la, (size_t)M * V * 4, NULL);
        vv_dev_memcpy_d2h(b, lb, (size_t)M * V * 4, NULL);
        vv_dev_memcpy_d2h(at, ta, (size_t)M * 4, NULL);
        vv_dev_memcpy_d2h(bt, tb, (size_t)M * 4, NULL);
        CHECK(memcmp(a, b, (size_t)M * V * 4) == 0,
              "head rows V%d K%d M%d: logits differ from the one-row head", V, K, M);
        CHECK(memcmp(at, bt, (size_t)M * 4) == 0,
              "head rows V%d K%d M%d: argmax differs", V, K, M);
        if (!memcmp(a, b, (size_t)M * V * 4) && !memcmp(at, bt, (size_t)M * 4))
            printf("  ok   head + argmax rows V%d K%d M%d\n", V, K, M);
    if (getenv("VV_SPEC_BENCH") && getenv("VV_SPEC_BENCH")[0] == '1') {
        double t1 = 0.0, tm = 0.0;
        for (int rep = 0; rep < 2; rep++) {
            const int it = 20;
            double t0 = vv_time_ms();
            for (int k = 0; k < it; k++)
                vv_lm_head_gemv_dev(x, W, lb, V, K, NULL);
            vv_dev_stream_sync(NULL);
            t1 = (vv_time_ms() - t0) * 1000.0 / it;
            t0 = vv_time_ms();
            for (int k = 0; k < it; k++)
                vv_lm_head_rows_dev(x, W, la, M, V, K, NULL);
            vv_dev_stream_sync(NULL);
            tm = (vv_time_ms() - t0) * 1000.0 / it;
        }
        printf("  bench head V%d K%d: 1 row %.1f us, %d rows %.1f us (%.2fx)\n",
               V, K, t1, M, tm, tm / t1);
    }
        free(a); free(b);
    }
    vv_dev_free(W); vv_dev_free(x); vv_dev_free(la); vv_dev_free(lb);
    vv_dev_free(scr); vv_dev_free(sv); vv_dev_free(si);
    vv_dev_free(ta); vv_dev_free(tb);
    free(wh); free(xh);
}

/* ─── The drafter's own kernels against CPU references ───────────────────── */

static void test_conv(void) {
    const int M = 8, H = 256, T = 2, gs = 16, G = H / gs, ld = 2 * T * G;
    uint16_t* xh = (uint16_t*)malloc((size_t)M * H * 2);
    uint16_t* dh = (uint16_t*)malloc((size_t)M * ld * 2);
    float* base = (float*)malloc((size_t)T * H * 4);
    fill_half(xh, (size_t)M * H, 1.0f);
    fill_half(dh, (size_t)M * ld, 0.3f);
    for (int i = 0; i < T * H; i++) base[i] = frand();
    void* x = dev_upload(xh, (size_t)M * H * 2);
    void* dy = dev_upload(dh, (size_t)M * ld * 2);
    float* bd = (float*)dev_upload(base, (size_t)T * H * 4);
    void* out = NULL;
    vv_dev_alloc(&out, (size_t)M * H * 2);
    /* Second half of the coefficients, blocks of 4: the finish conv. */
    const int block = 4;
    vv_status_t s = vv_dflash_conv_dev(x, (const uint8_t*)dy + (size_t)T * G * 2,
                                       ld, bd, out, M, H, T, gs, block, NULL);
    uint16_t* oh = (uint16_t*)malloc((size_t)M * H * 2);
    vv_dev_memcpy_d2h(oh, out, (size_t)M * H * 2, NULL);
    double worst = 0.0;
    for (int t = 0; t < M; t++)
        for (int c = 0; c < H; c++) {
            double acc = 0.0;
            for (int o = 0; o < T && t - o >= (t / block) * block; o++) {
                const double xv = vv_half_to_float(xh[(size_t)(t - o) * H + c]);
                const double dv = vv_half_to_float(dh[(size_t)t * ld + T * G + o * G + c / gs]);
                acc += (base[o * H + c] + dv) * xv;
            }
            const double e = fabs(acc - vv_half_to_float(oh[(size_t)t * H + c]));
            if (e > worst) worst = e;
        }
    CHECK(s == VV_OK && worst < 4e-3, "conv: max error %.3g", worst);
    if (s == VV_OK && worst < 4e-3) printf("  ok   in-block conv (max error %.2g)\n", worst);
    vv_dev_free(x); vv_dev_free(dy); vv_dev_free(bd); vv_dev_free(out);
    free(xh); free(dh); free(base); free(oh);
}

static void test_topk_walk(void) {
    const int M = 7, V = 50000, k = 16, rank = 64, VC = 1000;
    float* lg = (float*)malloc((size_t)M * V * 4);
    for (size_t i = 0; i < (size_t)M * V; i++) lg[i] = frand() * 10.0f;
    float* ld = (float*)dev_upload(lg, (size_t)M * V * 4);
    float* tv = NULL;
    int32_t* ti = NULL;
    vv_dev_alloc((void**)&tv, (size_t)M * k * 4);
    vv_dev_alloc((void**)&ti, (size_t)M * k * 4);
    vv_status_t s = vv_topk_rows_dev(ld, M, V, k, tv, ti, NULL);
    float hv[7 * 16];
    int32_t hi[7 * 16];
    vv_dev_memcpy_d2h(hv, tv, sizeof(hv), NULL);
    vv_dev_memcpy_d2h(hi, ti, sizeof(hi), NULL);
    int bad = s != VV_OK;
    for (int m = 0; !bad && m < M; m++) {
        /* reference: k best by value, ties to the lower id */
        const float* r = lg + (size_t)m * V;
        int ref[16];
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int i = 0; i < V; i++) {
                bool used = false;
                for (int u = 0; u < j; u++) used |= ref[u] == i;
                if (used) continue;
                if (best < 0 || r[i] > r[best]) best = i;
            }
            ref[j] = best;
        }
        for (int j = 0; j < k; j++)
            if (hi[m * k + j] != ref[j] || hv[m * k + j] != r[ref[j]]) bad = 1;
    }
    CHECK(!bad, "top-k differs from the reference");
    if (!bad) printf("  ok   top-%d over %d\n", k, V);

    /* The walk over small codebooks: ids below VC. */
    for (int i = 0; i < M * k; i++) hi[i] = (int32_t)(urand() % VC);
    uint16_t* pc = (uint16_t*)malloc((size_t)VC * rank * 2);
    uint16_t* sc = (uint16_t*)malloc((size_t)VC * rank * 2);
    uint16_t* hp = (uint16_t*)malloc((size_t)M * rank * 2);
    fill_half(pc, (size_t)VC * rank, 1.0f);
    fill_half(sc, (size_t)VC * rank, 1.0f);
    fill_half(hp, (size_t)M * rank, 1.0f);
    void* pcd = dev_upload(pc, (size_t)VC * rank * 2);
    void* scd = dev_upload(sc, (size_t)VC * rank * 2);
    void* hpd = dev_upload(hp, (size_t)M * rank * 2);
    int32_t anchor = 7;
    int32_t* ad = (int32_t*)dev_upload(&anchor, 4);
    int32_t* ci = (int32_t*)dev_upload(hi, sizeof(hi));
    int32_t* od = NULL;
    vv_dev_alloc((void**)&od, (size_t)M * 4);
    s = vv_dflash_walk_dev(hpd, tv, ci, pcd, scd, ad, M, k, rank, od, NULL);
    int32_t path[7];
    vv_dev_memcpy_d2h(path, od, sizeof(path), NULL);
    bad = s != VV_OK;
    int prev = anchor;
    for (int t = 0; !bad && t < M; t++) {
        double best = -1e300, chosen = 0;
        for (int j = 0; j < k; j++) {
            double sc2 = hv[t * k + j];
            for (int r = 0; r < rank; r++)
                sc2 += (double)vv_half_to_float(pc[(size_t)prev * rank + r]) *
                       vv_half_to_float(hp[(size_t)t * rank + r]) *
                       vv_half_to_float(sc[(size_t)hi[t * k + j] * rank + r]);
            if (sc2 > best) best = sc2;
            if (hi[t * k + j] == path[t]) chosen = sc2;
        }
        /* The GPU sums in FP32 in its own order: its pick must be a max
         * within that rounding. */
        if (chosen < best - 1e-3 * (fabs(best) + 1.0)) bad = 1;
        prev = path[t];
    }
    CHECK(!bad, "selector walk picked a non-maximal candidate");
    if (!bad) printf("  ok   selector walk\n");
    vv_dev_free(ld); vv_dev_free(tv); vv_dev_free(ti); vv_dev_free(pcd);
    vv_dev_free(scd); vv_dev_free(hpd); vv_dev_free(ad); vv_dev_free(ci);
    vv_dev_free(od);
    free(lg); free(pc); free(sc); free(hp);
}

/* ─── A whole transcription, with and without the drafter ────────────────── */

static int run_ids(const char* model, const char* draft, const float* pcm,
                   int n, int32_t** ids, int* n_ids,
                   vv_spec_stats_t* stats) {
    vv_init_params_t p = vv_init_params_default();
    p.draft_dir = draft;
    vv_inference_ctx_t* ctx = NULL;
    if (vv_inference_init(model, 0, &p, &ctx) != VV_OK) return 0;
    vv_transcription_t* tr = NULL;
    vv_inference_params_t ip;
    memset(&ip, 0, sizeof(ip));
    int ok = vv_inference_transcribe(ctx, pcm, n, &ip, &tr) == VV_OK && tr;
    if (ok) {
        *n_ids = tr->num_tokens;
        *ids = (int32_t*)malloc(sizeof(int32_t) * (size_t)(tr->num_tokens + 1));
        memcpy(*ids, tr->tokens, sizeof(int32_t) * (size_t)tr->num_tokens);
        const vv_spec_stats_t* st = vv_inference_spec_stats(ctx);
        if (stats) {
            if (st) *stats = *st;
            else memset(stats, 0, sizeof(*stats));
        }
        vv_transcription_free(tr);
    }
    vv_inference_free(ctx);
    return ok;
}

static void test_transcript(void) {
    const char* model = getenv("VV_TEST_MODEL");
    const char* draft = getenv("VV_TEST_DRAFT");
    if (!model || !draft) {
        printf("  SKIP transcript: set VV_TEST_MODEL and VV_TEST_DRAFT\n");
        return;
    }
    const char* audio = getenv("VV_TEST_AUDIO");
    if (!audio) audio = VV_TEST_DATA_DIR "/jfk.wav";
    float* raw = NULL;
    int rn = 0, sr = 0;
    if (vv_audio_load_any(audio, &raw, &rn, &sr) != VV_OK) {
        printf("  SKIP transcript: cannot read %s\n", audio);
        return;
    }
    float* pcm = NULL;
    int n = 0;
    vv_audio_prepare_ex(raw, rn, sr, true, &pcm, &n);
    vv_free(raw);
    int32_t *a = NULL, *b = NULL;
    int na = 0, nb = 0;
    vv_spec_stats_t st;
    const int ok = run_ids(model, NULL, pcm, n, &a, &na, NULL) &&
                   run_ids(model, draft, pcm, n, &b, &nb, &st);
    CHECK(ok, "transcript: a run failed");
    if (ok) {
        const int same = na == nb && !memcmp(a, b, sizeof(int32_t) * (size_t)na);
        CHECK(same, "transcript: %d ids plain, %d with the drafter, not equal",
              na, nb);
        CHECK(st.cycles > 0, "transcript: the drafter never ran");
        if (same && st.cycles > 0)
            printf("  ok   transcript: %d ids identical, %.2f tokens per "
                   "cycle\n", na, (double)st.tokens / (double)st.cycles);
    }
    free(a); free(b);
    vv_free(pcm);
}

int main(void) {
    printf("test_spec\n");
    test_drafter_config();
    if (vv_dev_device_count() <= 0) {
        printf("  SKIP: the kernels (no GPU)\n");
        return failures ? 1 : 0;
    }
    vv_dev_set_device(0);
    {
        const int qkv7[3] = { 3584, 512, 512 }, gu7[2] = { 18944, 18944 };
        const int down7[1] = { 3584 }, o7[1] = { 3584 };
        const int qkv15[3] = { 1536, 256, 256 }, gu15[2] = { 8960, 8960 };
        const int down15[1] = { 1536 };
        test_w4a16_rows("w4a16 rows 7B q/k/v", qkv7, 3, 3584, 128, true);
        test_w4a16_rows("w4a16 rows 7B o", o7, 1, 3584, 128, false);
        test_w4a16_rows("w4a16 rows 7B gate/up", gu7, 2, 3584, 128, false);
        test_w4a16_rows("w4a16 rows 7B down", down7, 1, 18944, 128, false);
        test_w4a16_rows("w4a16 rows 1.5B q/k/v", qkv15, 3, 1536, 128, true);
        test_w4a16_rows("w4a16 rows 1.5B gate/up", gu15, 2, 1536, 64, false);
        test_w4a16_rows("w4a16 rows 1.5B down", down15, 1, 8960, 128, false);
    }
    {
        const int backends[3] = { VV_ATTN_FA1, VV_ATTN_FA2, VV_ATTN_FLASHINFER };
        const int fmts[2] = { VV_KV_FP16, VV_KV_TQ4 };
        for (int bi = 0; bi < 3; bi++)
            for (int fi = 0; fi < 2; fi++) {
                /* 1020 + 8 crosses a flashinfer split step; 250 + 8 an fa2 one */
                test_attn_rows(backends[bi], fmts[fi], false, 28, 4, 1020, 8);
                test_attn_rows(backends[bi], fmts[fi], false, 28, 4, 250, 8);
                test_attn_rows(backends[bi], fmts[fi], false, 12, 2, 3000, 16);
            }
        test_attn_rows(VV_ATTN_FA2, VV_KV_FP16, true, 28, 4, 700, 8);
        test_attn_rows(VV_ATTN_FLASHINFER, VV_KV_TQ4, true, 28, 4, 2040, 8);
    }
    test_ternary_rows(1536, 1536);
    test_ternary_rows(8960, 1536);
    test_ternary_rows(1536, 8960);
    test_head_rows(152064, 3584, 8);
    test_head_rows(151936, 1536, 7);
    test_head_rows(152064, 3584, 16);
    test_head_rows(1000, 512, 5);       /* a last warp with rows past V */
    test_conv();
    test_topk_walk();
    test_transcript();
    printf(failures ? "test_spec: %d FAILED\n" : "test_spec: ok\n", failures);
    return failures ? 1 : 0;
}
