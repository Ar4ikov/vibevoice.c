/**
 * @file test_vae_stream.c
 * @brief The Conv-VAE encoder's contracts, on a tiny random-weight model.
 *
 * No checkpoint needed: a three-stage encoder (8 → 16 → 32 channels, strides
 * 2 and 3, 6x compression) built from random tensors under the real weight
 * names exercises every code path the 7-stage one does, in milliseconds.
 *
 *   - the CPU encoder against a naive whole-signal FP32 reference
 *   - GPU against the same reference (FP16 tolerance)
 *   - batched == one by one, bit for bit
 *   - chunks of any length, across launch boundaries == the whole signal
 *   - a stateless window == a fresh encode of that window
 *   - arena high-water <= the bound it was sized with
 *   - four threads through one front end at once == one at a time
 *   - the device connectors against the CPU ones, and nothing written past
 *     a job's last row
 *   - the conv-as-GEMM and both FFN GEMM tiles against the direct kernels,
 *     bit for bit
 *   - a short job next to a long one leaves before the long one does
 *
 * The GPU cases skip cleanly without a device.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"
#include "vibevoice/frontend.h"
#include "vibevoice/device.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vv_thread.h"

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
} while (0)

/* ─── A tiny model ──────────────────────────────────────────────────────── */

#define NF      8
#define VAE     8
#define HS      16
#define N_W_MAX 128

typedef struct {
    vv_weight_t w[N_W_MAX];
    int         n;
    float*      mem[N_W_MAX];
    uint64_t    rng;
} tiny_t;

static float frand(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)(*s >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static void add_t(tiny_t* t, const char* name, int ndim, int d0, int d1, int d2,
                  float scale, float offset) {
    vv_weight_t* w = &t->w[t->n];
    memset(w, 0, sizeof(*w));
    snprintf(w->name, sizeof(w->name), "%s", name);
    const size_t n = (size_t)d0 * (ndim > 1 ? d1 : 1) * (ndim > 2 ? d2 : 1);
    float* f = (float*)vv_alloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) f[i] = offset + scale * frand(&t->rng);
    t->mem[t->n] = f;
    w->tensor.data = f;
    w->tensor.ndim = ndim;
    w->tensor.shape[0] = d0;
    w->tensor.shape[1] = d1;
    w->tensor.shape[2] = d2;
    w->tensor.dtype = VV_DTYPE_F32;
    w->tensor.size_bytes = n * sizeof(float);
    t->n++;
}

static const int DEPTHS[3] = { 1, 1, 2 };
static const int RATIOS[2] = { 3, 2 };   /* listed large → small, as config.json */

static void tiny_build(tiny_t* t, const char* which, uint64_t seed) {
    memset(t, 0, sizeof(*t));
    t->rng = seed;
    char nm[256];
    const char* pre = which;
    snprintf(nm, sizeof(nm), "%sdownsample_layers.0.0.conv.conv.weight", pre);
    add_t(t, nm, 3, NF, 1, 7, 0.4f, 0.0f);
    snprintf(nm, sizeof(nm), "%sdownsample_layers.0.0.conv.conv.bias", pre);
    add_t(t, nm, 1, NF, 0, 0, 0.1f, 0.0f);
    int ch = NF;
    for (int s = 0; s < 3; s++) {
        for (int b = 0; b < DEPTHS[s]; b++) {
            const float sc = 1.0f / sqrtf((float)ch);
#define BLK(sfx, nd, a, bb, c, scale, off) do { \
            snprintf(nm, sizeof(nm), "%sstages.%d.%d." sfx, pre, s, b); \
            add_t(t, nm, nd, a, bb, c, scale, off); } while (0)
            BLK("norm.weight", 1, ch, 0, 0, 0.2f, 1.0f);
            BLK("mixer.conv.conv.conv.weight", 3, ch, 1, 7, 0.4f, 0.0f);
            BLK("mixer.conv.conv.conv.bias", 1, ch, 0, 0, 0.1f, 0.0f);
            BLK("gamma", 1, ch, 0, 0, 0.3f, 0.3f);
            BLK("ffn_norm.weight", 1, ch, 0, 0, 0.2f, 1.0f);
            BLK("ffn.linear1.weight", 2, 4 * ch, ch, 0, 1.7f * sc, 0.0f);
            BLK("ffn.linear1.bias", 1, 4 * ch, 0, 0, 0.1f, 0.0f);
            BLK("ffn.linear2.weight", 2, ch, 4 * ch, 0, 0.85f * sc, 0.0f);
            BLK("ffn.linear2.bias", 1, ch, 0, 0, 0.1f, 0.0f);
            BLK("ffn_gamma", 1, ch, 0, 0, 0.3f, 0.3f);
#undef BLK
        }
        if (s < 2) {
            const int r = RATIOS[1 - s];
            snprintf(nm, sizeof(nm), "%sdownsample_layers.%d.0.conv.conv.weight",
                     pre, s + 1);
            add_t(t, nm, 3, 2 * ch, ch, 2 * r, 1.0f / sqrtf((float)(ch * 2 * r)),
                  0.0f);
            snprintf(nm, sizeof(nm), "%sdownsample_layers.%d.0.conv.conv.bias",
                     pre, s + 1);
            add_t(t, nm, 1, 2 * ch, 0, 0, 0.1f, 0.0f);
            ch *= 2;
        }
    }
    snprintf(nm, sizeof(nm), "%shead.conv.conv.weight", pre);
    add_t(t, nm, 3, VAE, ch, 7, 1.0f / sqrtf((float)(ch * 7)), 0.0f);
    snprintf(nm, sizeof(nm), "%shead.conv.conv.bias", pre);
    add_t(t, nm, 1, VAE, 0, 0, 0.1f, 0.0f);
}

static void tiny_free(tiny_t* t) {
    for (int i = 0; i < t->n; i++) vv_free(t->mem[i]);
}

static vv_conv_vae_encoder_t* tiny_encoder(tiny_t* t, bool acoustic) {
    vv_acoustic_tokenizer_config_t ac;
    vv_semantic_tokenizer_config_t se;
    memset(&ac, 0, sizeof(ac));
    memset(&se, 0, sizeof(se));
    ac.causal = se.causal = true;
    ac.vae_dim = se.vae_dim = VAE;
    ac.n_ratios = se.n_ratios = 2;
    ac.n_depths = se.n_depths = 3;
    memcpy(ac.encoder_ratios, RATIOS, sizeof(RATIOS));
    memcpy(se.encoder_ratios, RATIOS, sizeof(RATIOS));
    memcpy(ac.encoder_depths, DEPTHS, sizeof(DEPTHS));
    memcpy(se.encoder_depths, DEPTHS, sizeof(DEPTHS));
    ac.layernorm_eps = se.layernorm_eps = 1e-5f;
    vv_conv_vae_encoder_t* e = NULL;
    vv_conv_vae_init(t->w, t->n, acoustic ? (const void*)&ac : (const void*)&se,
                     acoustic, &e);
    return e;
}

/* ─── Naive whole-signal reference (FP32, channel-first) ────────────────── */

static const float* T(const vv_tensor_t* t) { return (const float*)t->data; }

/** Causal conv over a whole signal: left pad k - s, output ceil(L / s). */
static float* ref_conv(const float* x, int ic, int L, const vv_conv1d_weights_t* c,
                       int* out_len) {
    const int oc = (int)c->weight.shape[0], k = c->kernel_size, s = c->stride;
    const int ol = (L + s - 1) / s, pad = k - s;
    float* y = (float*)vv_alloc((size_t)oc * ol * sizeof(float));
    for (int o = 0; o < oc; o++)
        for (int t = 0; t < ol; t++) {
            float acc = c->bias.data ? T(&c->bias)[o] : 0.0f;
            for (int i = 0; i < ic; i++)
                for (int kk = 0; kk < k; kk++) {
                    const int p = t * s + kk - pad;
                    if (p >= 0 && p < L)
                        acc += T(&c->weight)[((size_t)o * ic + i) * k + kk] *
                               x[(size_t)i * L + p];
                }
            y[(size_t)o * ol + t] = acc;
        }
    *out_len = ol;
    return y;
}

static void ref_norm(const float* x, const float* w, float* y, int C, int L,
                     float eps) {
    for (int t = 0; t < L; t++) {
        float ss = 0.0f;
        for (int c = 0; c < C; c++) ss += x[(size_t)c * L + t] * x[(size_t)c * L + t];
        const float inv = 1.0f / sqrtf(ss / (float)C + eps);
        for (int c = 0; c < C; c++)
            y[(size_t)c * L + t] = x[(size_t)c * L + t] * inv * w[c];
    }
}

/** [frames][VAE] latents of the whole signal. */
static float* ref_encode(const vv_conv_vae_encoder_t* e, const float* audio,
                         int N, int* frames) {
    int L = 0;
    float* x = ref_conv(audio, 1, N, &e->input_conv, &L);
    int C = (int)e->input_conv.weight.shape[0];
    for (int s = 0; s < e->n_stages; s++) {
        for (int b = 0; b < e->stages[s].n_blocks; b++) {
            const vv_encoder_block_t* k = &e->stages[s].blocks[b];
            float* n = (float*)vv_alloc((size_t)C * L * sizeof(float));
            ref_norm(x, T(&k->mixer_norm_weight), n, C, L, e->eps);
            const int kk = k->mixer_conv.kernel_size;
            for (int c = 0; c < C; c++)
                for (int t = L - 1; t >= 0; t--) {
                    float acc = T(&k->mixer_conv.bias)[c];
                    for (int j = 0; j < kk; j++) {
                        const int p = t + j - (kk - 1);
                        if (p >= 0) acc += T(&k->mixer_conv.weight)[c * kk + j] *
                                           n[(size_t)c * L + p];
                    }
                    x[(size_t)c * L + t] += T(&k->mixer_layer_scale)[c] * acc;
                }
            ref_norm(x, T(&k->ffn_norm_weight), n, C, L, e->eps);
            const int H = (int)k->ffn_linear1_weight.shape[0];
            float* h = (float*)vv_alloc((size_t)H * sizeof(float));
            for (int t = 0; t < L; t++) {
                for (int o = 0; o < H; o++) {
                    float a = T(&k->ffn_linear1_bias)[o];
                    for (int c = 0; c < C; c++)
                        a += T(&k->ffn_linear1_weight)[(size_t)o * C + c] *
                             n[(size_t)c * L + t];
                    h[o] = 0.5f * a * (1.0f + erff(a * 0.7071067811865476f));
                }
                for (int c = 0; c < C; c++) {
                    float a = T(&k->ffn_linear2_bias)[c];
                    for (int o = 0; o < H; o++)
                        a += T(&k->ffn_linear2_weight)[(size_t)c * H + o] * h[o];
                    x[(size_t)c * L + t] += T(&k->ffn_layer_scale)[c] * a;
                }
            }
            vv_free(h);
            vv_free(n);
        }
        if (e->stages[s].downsample.weight.data) {
            int L2 = 0;
            float* y = ref_conv(x, C, L, &e->stages[s].downsample, &L2);
            vv_free(x);
            x = y;
            L = L2;
            C = (int)e->stages[s].downsample.weight.shape[0];
        }
    }
    int F = 0;
    float* z = ref_conv(x, C, L, &e->proj_mean, &F);
    vv_free(x);
    float* out = (float*)vv_alloc((size_t)F * VAE * sizeof(float));
    for (int f = 0; f < F; f++)
        for (int d = 0; d < VAE; d++) out[(size_t)f * VAE + d] = z[(size_t)d * F + f];
    vv_free(z);
    *frames = F;
    return out;
}

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static float* make_audio(int N, uint64_t seed) {
    float* a = (float*)vv_alloc((size_t)N * sizeof(float));
    uint64_t s = seed;
    for (int i = 0; i < N; i++)
        a[i] = 0.3f * sinf(0.013f * (float)i) + 0.2f * frand(&s);
    return a;
}

static void rel_err(const float* a, const float* b, size_t n, double* rel,
                    double* cosv) {
    double num = 0, den = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        den += (double)b[i] * b[i];
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    *rel = sqrt(num / (den > 0 ? den : 1));
    *cosv = dot / sqrt((na > 0 ? na : 1) * (nb > 0 ? nb : 1));
}

static uint16_t* upload_audio(const float* a, int N, void* stream) {
    void* f32 = NULL;
    void* f16 = NULL;
    vv_dev_alloc(&f32, (size_t)(N > 0 ? N : 1) * 4);
    vv_dev_alloc(&f16, (size_t)(N > 0 ? N : 1) * 2);
    if (N > 0) {
        vv_dev_memcpy_h2d(f32, a, (size_t)N * 4, stream);
        vv_vae_f32_to_f16_dev((const float*)f32, f16, N, stream);
    }
    vv_dev_stream_sync(stream);
    vv_dev_free(f32);
    return (uint16_t*)f16;
}

static uint16_t* download(const void* d, size_t n, void* stream) {
    uint16_t* h = (uint16_t*)vv_alloc(n * 2 + 2);
    vv_dev_stream_sync(stream);
    if (n) vv_dev_memcpy_d2h(h, d, n * 2, stream);
    vv_dev_stream_sync(stream);
    return h;
}

/* ─── CPU ───────────────────────────────────────────────────────────────── */

static void test_frames(vv_conv_vae_encoder_t* e) {
    printf("test_frames:\n");
    int ok = 1;
    for (int n = 1; n < 200; n++) {
        const int want = (n + 5) / 6;
        if (vv_conv_vae_frames(e, n) != want) ok = 0;
    }
    TEST_ASSERT(ok, "frames == ceil(N / 6) for N in 1..199");
}

static void test_cpu_vs_reference(vv_conv_vae_encoder_t* e) {
    printf("test_cpu_vs_reference:\n");
    /* Longer than one CPU tile, so the carried context is exercised. */
    const int N = 100003;
    float* audio = make_audio(N, 7);
    int fr_ref = 0, fr = 0;
    float* ref = ref_encode(e, audio, N, &fr_ref);
    float* got = NULL;
    vv_status_t s = vv_conv_vae_encode_cpu(e, audio, N, &got, &fr);
    TEST_ASSERT(s == VV_OK, "CPU encode succeeds");
    TEST_ASSERT(fr == fr_ref && fr == (N + 5) / 6, "CPU frame count");
    if (s == VV_OK && fr == fr_ref) {
        double rel, cs;
        rel_err(got, ref, (size_t)fr * VAE, &rel, &cs);
        printf("    rel=%.2e cos=%.8f\n", rel, cs);
        /* FP16 weights in the GEMMs; FP32 activations. */
        TEST_ASSERT(rel < 5e-3 && cs > 0.99999, "CPU matches the FP32 reference");
    }
    vv_free(got);
    vv_free(ref);
    vv_free(audio);
}

/* ─── GPU ───────────────────────────────────────────────────────────────── */

typedef struct {
    vv_conv_vae_encoder_t* e;
    vv_vae_weights_t*      w;
    vv_vae_arena_t*        a;
    void*                  stream;
} gpu_t;

/** Encode `audio` through a state in the given chunk lengths (0-terminated,
 *  cycled); returns [frames][VAE] FP16 on the host. */
static uint16_t* gpu_chunked(gpu_t* g, const float* audio, int N,
                             const int* chunks, int* frames) {
    vv_vae_state_t* st = NULL;
    vv_vae_state_create(g->w, &st);
    const int total_frames = vv_vae_frames(g->w, NULL, N, true);
    void* out = NULL;
    vv_dev_alloc(&out, (size_t)(total_frames + 1) * VAE * 2);
    uint16_t* dev_audio = upload_audio(audio, N, g->stream);
    int off = 0, got = 0, ci = 0;
    while (off < N) {
        int len = chunks[ci];
        if (!chunks[++ci]) ci = 0;
        if (len > N - off) len = N - off;
        vv_vae_item_t it;
        memset(&it, 0, sizeof(it));
        it.audio = dev_audio + off;
        it.n_samples = len;
        it.state = st;
        it.is_final = off + len == N;
        it.out = (uint16_t*)out + (size_t)got * VAE;
        it.out_ld = VAE;
        if (vv_vae_encode(g->w, g->a, &it, 1, g->stream) != VV_OK) break;
        got += it.n_frames;
        off += len;
    }
    uint16_t* h = download(out, (size_t)got * VAE, g->stream);
    vv_dev_free(out);
    vv_dev_free(dev_audio);
    vv_vae_state_free(st);
    *frames = off == N ? got : -1;
    return h;
}

static void test_gpu_vs_reference(gpu_t* g) {
    printf("test_gpu_vs_reference:\n");
    const int N = 40001;
    float* audio = make_audio(N, 11);
    int fr_ref = 0, fr = 0;
    float* ref = ref_encode(g->e, audio, N, &fr_ref);
    const int whole[] = { N, 0 };
    uint16_t* h = gpu_chunked(g, audio, N, whole, &fr);
    TEST_ASSERT(fr == fr_ref, "GPU frame count matches");
    if (fr == fr_ref) {
        float* f = (float*)vv_alloc((size_t)fr * VAE * sizeof(float));
        for (size_t i = 0; i < (size_t)fr * VAE; i++) f[i] = vv_half_to_float(h[i]);
        double rel, cs;
        rel_err(f, ref, (size_t)fr * VAE, &rel, &cs);
        printf("    rel=%.2e cos=%.8f\n", rel, cs);
        TEST_ASSERT(rel < 3e-2 && cs > 0.9995, "GPU matches the FP32 reference");
        /* And the CPU encoder, while we are here: the two paths agree. */
        float* c = NULL;
        int cf = 0;
        if (vv_conv_vae_encode_cpu(g->e, audio, N, &c, &cf) == VV_OK && cf == fr) {
            rel_err(f, c, (size_t)fr * VAE, &rel, &cs);
            printf("    gpu vs cpu rel=%.2e cos=%.8f\n", rel, cs);
            TEST_ASSERT(rel < 3e-2 && cs > 0.9995, "GPU matches the CPU encoder");
        }
        vv_free(c);
        vv_free(f);
    }
    vv_free(h);
    vv_free(ref);
    vv_free(audio);
}

static void test_gpu_chunking(gpu_t* g) {
    printf("test_gpu_chunking:\n");
    const int N = 30011;
    float* audio = make_audio(N, 21);
    const int whole[] = { N, 0 };
    /* Every stride alignment and then some, including 1-sample chunks. */
    const int odd[] = { 1, 7, 13, 2, 5, 3001, 6, 997, 1, 4096, 0 };
    const int aligned[] = { 600, 0 };
    int f0 = 0, f1 = 0, f2 = 0;
    uint16_t* a = gpu_chunked(g, audio, N, whole, &f0);
    uint16_t* b = gpu_chunked(g, audio, N, odd, &f1);
    uint16_t* c = gpu_chunked(g, audio, N, aligned, &f2);
    TEST_ASSERT(f0 == (N + 5) / 6 && f1 == f0 && f2 == f0,
                "chunked frame counts equal the whole signal's");
    TEST_ASSERT(f1 == f0 && memcmp(a, b, (size_t)f0 * VAE * 2) == 0,
                "odd-length chunks == whole signal, bit for bit");
    TEST_ASSERT(f2 == f0 && memcmp(a, c, (size_t)f0 * VAE * 2) == 0,
                "aligned chunks == whole signal, bit for bit");
    vv_free(a);
    vv_free(b);
    vv_free(c);
    vv_free(audio);
}

static void test_gpu_batching(gpu_t* g) {
    printf("test_gpu_batching:\n");
    enum { NI = 5 };
    const int lens[NI] = { 12000, 1, 7919, 0, 30000 };
    float* audio[NI];
    uint16_t* single[NI];
    int fr[NI];
    for (int i = 0; i < NI; i++) {
        audio[i] = make_audio(lens[i] > 0 ? lens[i] : 1, 100 + i);
        const int whole[] = { lens[i] > 0 ? lens[i] : 1, 0 };
        if (lens[i] > 0) single[i] = gpu_chunked(g, audio[i], lens[i], whole, &fr[i]);
        else { single[i] = NULL; fr[i] = 0; }
    }
    /* All five in one launch: fresh states, some stateless, all final. */
    vv_vae_state_t* st[NI];
    uint16_t* dev[NI];
    void* out[NI];
    vv_vae_item_t it[NI];
    memset(it, 0, sizeof(it));
    for (int i = 0; i < NI; i++) {
        st[i] = NULL;
        if (i % 2 == 0) vv_vae_state_create(g->w, &st[i]);
        dev[i] = upload_audio(audio[i], lens[i], g->stream);
        vv_dev_alloc(&out[i], (size_t)(fr[i] + 1) * VAE * 2);
        it[i].audio = dev[i];
        it[i].n_samples = lens[i];
        it[i].state = st[i];
        it[i].is_final = true;
        it[i].out = out[i];
        it[i].out_ld = VAE;
    }
    vv_status_t s = vv_vae_encode(g->w, g->a, it, NI, g->stream);
    TEST_ASSERT(s == VV_OK, "batched encode succeeds");
    int same = 1;
    for (int i = 0; i < NI; i++) {
        if (it[i].n_frames != fr[i]) same = 0;
        if (fr[i] > 0) {
            uint16_t* h = download(out[i], (size_t)fr[i] * VAE, g->stream);
            if (memcmp(h, single[i], (size_t)fr[i] * VAE * 2) != 0) same = 0;
            vv_free(h);
        }
    }
    TEST_ASSERT(same, "batched == one by one, bit for bit");
    TEST_ASSERT(vv_vae_arena_high_water(g->a) <=
                vv_vae_arena_bytes(g->e, vv_vae_arena_max_items(g->a),
                                   vv_vae_arena_max_samples(g->a)),
                "arena high-water within its bound");
    for (int i = 0; i < NI; i++) {
        vv_vae_state_free(st[i]);
        vv_dev_free(dev[i]);
        vv_dev_free(out[i]);
        vv_free(single[i]);
        vv_free(audio[i]);
    }
}

static void test_gpu_window(gpu_t* g) {
    printf("test_gpu_window:\n");
    /* A long stream, then a window cut out of its middle encoded stateless:
       the same as a fresh stream of just that window, and its leading
       frames can be dropped on the device. */
    const int N = 24000, W0 = 6000, WL = 7200;
    float* audio = make_audio(N, 31);
    const int whole[] = { WL, 0 };
    int fr = 0;
    uint16_t* fresh = gpu_chunked(g, audio + W0, WL, whole, &fr);

    uint16_t* dev = upload_audio(audio + W0, WL, g->stream);
    void* out = NULL;
    vv_dev_alloc(&out, (size_t)(fr + 1) * VAE * 2);
    vv_vae_item_t it;
    memset(&it, 0, sizeof(it));
    it.audio = dev;
    it.n_samples = WL;
    it.state = NULL;
    it.out = out;
    it.out_ld = VAE;
    it.skip_frames = 4;
    vv_status_t s = vv_vae_encode(g->w, g->a, &it, 1, g->stream);
    TEST_ASSERT(s == VV_OK && it.n_frames == fr, "stateless window encodes");
    uint16_t* h = download(out, (size_t)(fr - 4) * VAE, g->stream);
    TEST_ASSERT(memcmp(h, fresh + 4 * VAE, (size_t)(fr - 4) * VAE * 2) == 0,
                "window == fresh encode of the window (after skip)");
    vv_free(h);
    vv_free(fresh);
    vv_dev_free(out);
    vv_dev_free(dev);
    vv_free(audio);
}

/* ─── Front end: launches, connectors, concurrency ─────────────────────── */

typedef struct {
    tiny_t           conn_w;
    vv_connector_t   conn[2];
} tiny_conn_t;

static void tiny_connectors(tiny_conn_t* c) {
    memset(c, 0, sizeof(*c));
    c->conn_w.rng = 99;
    for (int e = 0; e < 2; e++) {
        vv_connector_init(&c->conn[e], VAE, HS);
        add_t(&c->conn_w, "fc1", 2, HS, VAE, 0, 0.5f, 0.0f);
        add_t(&c->conn_w, "fc1b", 1, HS, 0, 0, 0.1f, 0.0f);
        add_t(&c->conn_w, "norm", 1, HS, 0, 0, 0.2f, 1.0f);
        add_t(&c->conn_w, "fc2", 2, HS, HS, 0, 0.25f, 0.0f);
        add_t(&c->conn_w, "fc2b", 1, HS, 0, 0, 0.1f, 0.0f);
        const int b = e * 5;
        c->conn[e].fc1_weight = c->conn_w.w[b + 0].tensor;
        c->conn[e].fc1_bias = c->conn_w.w[b + 1].tensor;
        c->conn[e].norm_weight = c->conn_w.w[b + 2].tensor;
        c->conn[e].fc2_weight = c->conn_w.w[b + 3].tensor;
        c->conn[e].fc2_bias = c->conn_w.w[b + 4].tensor;
    }
}

typedef struct {
    vv_frontend_t* fe;
    const float*   audio;
    int            n;
    int            frames;
    uint16_t*      result;     /* host copy of the rows */
    vv_status_t    status;
    int            canary;     /* the row past the last one is untouched */
    double         t_start, t_done;
} fe_thread_t;

#define CANARY_BYTE 0x5A

/**
 * Encode one clip into `frames` rows of a buffer one row longer, filled
 * with a canary first; *canary_ok says whether that last row survived.
 */
static uint16_t* fe_encode(vv_frontend_t* fe, const float* audio, int n,
                           int frames, vv_status_t* st_out, int* canary_ok) {
    vv_frontend_stream_t* st = NULL;
    void* rows = NULL;
    void* ev = NULL;
    void* stream = NULL;
    uint16_t* h = NULL;
    vv_status_t s = vv_frontend_stream_create(fe, &st);
    if (s == VV_OK) s = vv_dev_alloc(&rows, (size_t)(frames + 1) * HS * 2);
    if (s == VV_OK) s = vv_dev_event_create(&ev);
    if (s == VV_OK) s = vv_dev_stream_create(&stream);
    if (s == VV_OK)
        s = vv_dev_memset_async(rows, CANARY_BYTE, (size_t)(frames + 1) * HS * 2,
                                stream);
    if (s == VV_OK) s = vv_dev_stream_sync(stream);
    if (canary_ok) *canary_ok = 0;
    if (s == VV_OK) {
        vv_frontend_job_t job;
        memset(&job, 0, sizeof(job));
        job.audio = audio;
        job.n_samples = n;
        job.stream = st;
        job.is_final = true;
        job.rows = rows;
        job.rows_ld = HS;
        job.done_event = ev;
        s = vv_frontend_submit(fe, &job);
        if (s == VV_OK && job.n_frames != frames) s = VV_ERR_SHAPE_MISMATCH;
        if (s == VV_OK) s = vv_dev_stream_wait_event(stream, ev);
        if (s == VV_OK) h = download(rows, (size_t)(frames + 1) * HS, stream);
        if (h && canary_ok) {
            const unsigned char* b = (const unsigned char*)(h + (size_t)frames * HS);
            int ok = 1;
            for (int i = 0; i < HS * 2; i++) ok &= b[i] == CANARY_BYTE;
            *canary_ok = ok;
        }
    }
    if (stream) vv_dev_stream_destroy(stream);
    if (ev) vv_dev_event_destroy(ev);
    if (rows) vv_dev_free(rows);
    vv_frontend_stream_free(st);
    *st_out = s;
    return h;
}

static VV_THREAD_RET fe_worker(void* arg) {
    fe_thread_t* t = (fe_thread_t*)arg;
    vv_dev_set_device(0);
    t->t_start = vv_time_ms();
    t->result = fe_encode(t->fe, t->audio, t->n, t->frames, &t->status,
                          &t->canary);
    t->t_done = vv_time_ms();
    VV_THREAD_RETURN;
}

/**
 * The rows against a host reference of the path they replace: the same
 * clip's latents (from the front end, raw), each connector on the CPU, the
 * two summed in FP32 and cut to FP16 -- what the pipeline did before the
 * connectors moved to the device. The device GEMMs round through FP16
 * between fc1, the norm and fc2, so the match is to FP16 precision.
 */
static void check_connectors(vv_frontend_t* fe, tiny_conn_t* tc,
                             const float* audio, int n, int frames,
                             const uint16_t* rows) {
    void *dl[2] = { NULL, NULL }, *ev = NULL, *stream = NULL;
    vv_frontend_stream_t* st = NULL;
    vv_status_t s = vv_frontend_stream_create(fe, &st);
    for (int e = 0; e < 2 && s == VV_OK; e++)
        s = vv_dev_alloc(&dl[e], (size_t)frames * VAE * 2);
    if (s == VV_OK) s = vv_dev_event_create(&ev);
    if (s == VV_OK) s = vv_dev_stream_create(&stream);
    uint16_t* lh[2] = { NULL, NULL };
    if (s == VV_OK) {
        vv_frontend_job_t job;
        memset(&job, 0, sizeof(job));
        job.audio = audio;
        job.n_samples = n;
        job.stream = st;
        job.is_final = true;
        job.ac_latents = dl[0];
        job.sem_latents = dl[1];
        job.done_event = ev;
        s = vv_frontend_run(fe, &job, 1);
        if (s == VV_OK && job.n_frames != frames) s = VV_ERR_SHAPE_MISMATCH;
        if (s == VV_OK) s = vv_dev_stream_wait_event(stream, ev);
        for (int e = 0; e < 2 && s == VV_OK; e++)
            lh[e] = download(dl[e], (size_t)frames * VAE, stream);
    }
    double num = 0.0, den = 0.0, worst = 0.0;
    if (s == VV_OK && lh[0] && lh[1]) {
        float* out[2] = { NULL, NULL };
        float* lf = (float*)vv_alloc((size_t)frames * VAE * sizeof(float));
        for (int e = 0; e < 2 && s == VV_OK; e++) {
            for (size_t i = 0; i < (size_t)frames * VAE; i++)
                lf[i] = vv_half_to_float(lh[e][i]);
            s = vv_connector_forward_cpu(&tc->conn[e], lf, frames, &out[e]);
        }
        for (size_t i = 0; s == VV_OK && i < (size_t)frames * HS; i++) {
            const double want = vv_half_to_float(vv_float_to_half(out[0][i] + out[1][i]));
            const double got = vv_half_to_float(rows[i]);
            const double d = fabs(got - want);
            num += d * d;
            den += want * want;
            if (d > worst) worst = d;
        }
        vv_free(out[0]);
        vv_free(out[1]);
        vv_free(lf);
    }
    const double rel = sqrt(num / (den > 0 ? den : 1));
    printf("    connectors vs host: %d frames rel=%.2e max abs=%.2e\n", frames,
           rel, worst);
    TEST_ASSERT(s == VV_OK && den > 0 && rel < 4e-3,
                "device connectors == CPU connectors + FP32 sum (FP16 tolerance)");
    vv_free(lh[0]);
    vv_free(lh[1]);
    if (stream) vv_dev_stream_destroy(stream);
    if (ev) vv_dev_event_destroy(ev);
    for (int e = 0; e < 2; e++) if (dl[e]) vv_dev_free(dl[e]);
    vv_frontend_stream_free(st);
}

static void test_frontend(vv_conv_vae_encoder_t* ea, vv_conv_vae_encoder_t* es) {
    printf("test_frontend:\n");
    tiny_conn_t tc;
    tiny_connectors(&tc);

    /* Small launches (1 s of the tiny model) so a clip spans several. */
    vv_frontend_params_t small = vv_frontend_params_default();
    small.max_items = 4;
    small.max_samples = 9000;
    vv_frontend_params_t big = small;
    big.max_samples = 200000;

    vv_frontend_t *fs = NULL, *fb = NULL;
    vv_status_t s1 = vv_frontend_create(ea, es, &tc.conn[0], &tc.conn[1], &small, &fs);
    vv_status_t s2 = vv_frontend_create(ea, es, &tc.conn[0], &tc.conn[1], &big, &fb);
    TEST_ASSERT(s1 == VV_OK && s2 == VV_OK, "front ends come up");
    if (s1 != VV_OK || s2 != VV_OK) {
        vv_frontend_free(fs);
        vv_frontend_free(fb);
        tiny_free(&tc.conn_w);
        return;
    }

    enum { NT = 4 };
    const int lens[NT] = { 31000, 17, 9000, 25003 };
    float* audio[NT];
    uint16_t* ref[NT];
    int frames[NT];
    int ok_split = 1, ok_canary = 1;
    for (int i = 0; i < NT; i++) {
        audio[i] = make_audio(lens[i], 500 + i);
        frames[i] = vv_frontend_frames(fb, lens[i]);
        vv_status_t s;
        int c1 = 0, c2 = 0;
        ref[i] = fe_encode(fb, audio[i], lens[i], frames[i], &s, &c1);
        uint16_t* sp = fe_encode(fs, audio[i], lens[i], frames[i], &s, &c2);
        if (!ref[i] || !sp || memcmp(ref[i], sp, (size_t)frames[i] * HS * 2) != 0)
            ok_split = 0;
        ok_canary &= c1 & c2;
        vv_free(sp);
    }
    TEST_ASSERT(ok_split, "a clip split over several launches == one launch");
    TEST_ASSERT(ok_canary, "nothing is written past a job's last row");
    for (int i = 0; i < NT; i++)
        if (lens[i] > 1000) check_connectors(fb, &tc, audio[i], lens[i],
                                             frames[i], ref[i]);

    /* Four threads at once through the service, then without it. */
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0) vv_frontend_service_start(fs, 0);
        fe_thread_t th[NT];
        vv_thread_t tid[NT];
        for (int round = 0; round < 3; round++) {
            for (int i = 0; i < NT; i++) {
                th[i].fe = fs;
                th[i].audio = audio[i];
                th[i].n = lens[i];
                th[i].frames = frames[i];
                th[i].result = NULL;
                th[i].status = VV_ERR_CUDA;
                vv_thread_start(&tid[i], fe_worker, &th[i]);
            }
            int ok = 1;
            for (int i = 0; i < NT; i++) {
                vv_thread_join(tid[i]);
                if (th[i].status != VV_OK || !th[i].result || !th[i].canary ||
                    memcmp(th[i].result, ref[i], (size_t)frames[i] * HS * 2) != 0)
                    ok = 0;
                vv_free(th[i].result);
            }
            if (!ok) {
                TEST_ASSERT(0, pass == 0 ? "4 threads through the service"
                                         : "4 threads, direct");
                break;
            }
            if (round == 2)
                TEST_ASSERT(1, pass == 0 ? "4 threads through the service == alone"
                                         : "4 threads, direct == alone");
        }
        if (pass == 0) {
            uint64_t launches = 0, jobs = 0;
            vv_frontend_stats(fs, &launches, &jobs);
            printf("    %llu launches for %llu jobs\n",
                   (unsigned long long)launches, (unsigned long long)jobs);
            vv_frontend_service_stop(fs);
        }
    }

    /* A short job submitted while a long one is running leaves first: the
       service schedules launches, not jobs. Both still match alone. */
    {
        const int LN = 600000;   /* ~67 launches of the small front end */
        float* la = make_audio(LN, 900);
        const int lf = vv_frontend_frames(fb, LN);
        vv_status_t s;
        uint16_t* lref = fe_encode(fb, la, LN, lf, &s, NULL);
        vv_frontend_service_start(fs, 0);
        fe_thread_t tl, ts;
        memset(&tl, 0, sizeof(tl));
        memset(&ts, 0, sizeof(ts));
        tl.fe = ts.fe = fs;
        tl.audio = la; tl.n = LN; tl.frames = lf;
        ts.audio = audio[2]; ts.n = lens[2]; ts.frames = frames[2];
        uint64_t l0 = 0, l1 = 0;
        vv_frontend_stats(fs, &l0, NULL);
        vv_thread_t htl, hts;
        vv_thread_start(&htl, fe_worker, &tl);
        for (int spin = 0; spin < 2000; spin++) {       /* long one under way */
            vv_frontend_stats(fs, &l1, NULL);
            if (l1 >= l0 + 2) break;
            vv_sleep_ms(1);
        }
        vv_thread_start(&hts, fe_worker, &ts);
        vv_thread_join(hts);
        vv_thread_join(htl);
        vv_frontend_service_stop(fs);
        printf("    short done at %.1f ms, long at %.1f ms\n",
               ts.t_done - tl.t_start, tl.t_done - tl.t_start);
        TEST_ASSERT(ts.status == VV_OK && tl.status == VV_OK &&
                    ts.t_done < tl.t_done,
                    "a short job next to a long one finishes first");
        TEST_ASSERT(lref && ts.result && tl.result &&
                    memcmp(ts.result, ref[2], (size_t)frames[2] * HS * 2) == 0 &&
                    memcmp(tl.result, lref, (size_t)lf * HS * 2) == 0,
                    "short and long jobs sharing launches == alone");
        vv_free(ts.result);
        vv_free(tl.result);
        vv_free(lref);
        vv_free(la);
    }

    for (int i = 0; i < NT; i++) {
        vv_free(ref[i]);
        vv_free(audio[i]);
    }
    vv_frontend_free(fs);
    vv_frontend_free(fb);
    tiny_free(&tc.conn_w);
}

/* ─── GEMM tiles ───────────────────────────────────────────────────────── */

/**
 * The FFN GEMMs pick a 64 or 128 tile from the grid size, so an item's
 * result must not depend on it: both tiles, and the same columns in two
 * narrower calls (another batch), agree bit for bit, and all of it matches
 * an FP64 reference with every epilogue.
 */
static void test_gemm_tiles(void* stream) {
    const int M = 96, K = 2048, P = 45, P1 = 19;
    uint64_t rng = 99;
    uint16_t* A = (uint16_t*)vv_alloc((size_t)M * K * 2);
    uint16_t* B = (uint16_t*)vv_alloc((size_t)K * P * 2);
    uint16_t* X = (uint16_t*)vv_alloc((size_t)M * P * 2);
    uint16_t bias[96], gam[96];
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = vv_float_to_half(frand(&rng) * 0.05f);
    for (size_t i = 0; i < (size_t)K * P; i++) B[i] = vv_float_to_half(frand(&rng));
    for (size_t i = 0; i < (size_t)M * P; i++) X[i] = vv_float_to_half(frand(&rng));
    for (int i = 0; i < M; i++) {
        bias[i] = vv_float_to_half(frand(&rng) * 0.1f);
        gam[i] = vv_float_to_half(0.5f + frand(&rng) * 0.1f);
    }
    void *dA, *dB, *dC, *dC2, *dC3, *db, *dg;
    vv_dev_alloc(&dA, (size_t)M * K * 2);
    vv_dev_alloc(&dB, (size_t)K * P * 2);
    vv_dev_alloc(&dC, (size_t)M * P * 2);
    vv_dev_alloc(&dC2, (size_t)M * P * 2);
    vv_dev_alloc(&dC3, (size_t)M * P * 2);
    vv_dev_alloc(&db, (size_t)M * 2);
    vv_dev_alloc(&dg, (size_t)M * 2);
    vv_dev_memcpy_h2d(dA, A, (size_t)M * K * 2, stream);
    vv_dev_memcpy_h2d(dB, B, (size_t)K * P * 2, stream);
    vv_dev_memcpy_h2d(db, bias, (size_t)M * 2, stream);
    vv_dev_memcpy_h2d(dg, gam, (size_t)M * 2, stream);

    const int epis[3] = { VV_VAE_EPI_BIAS, VV_VAE_EPI_BIAS_GELU,
                          VV_VAE_EPI_BIAS_RESID };
    for (int e = 0; e < 3; e++) {
        vv_dev_memcpy_h2d(dC, X, (size_t)M * P * 2, stream);
        vv_dev_memcpy_h2d(dC2, X, (size_t)M * P * 2, stream);
        vv_dev_memcpy_h2d(dC3, X, (size_t)M * P * 2, stream);
        vv_status_t st = vv_vae_gemm_nn_dev(epis[e], dA, dB, P, dC, P, M, K, P,
                                            db, dg, NULL, NULL, 128, stream);
        if (st == VV_OK)
            st = vv_vae_gemm_nn_dev(epis[e], dA, dB, P, dC3, P, M, K, P,
                                    db, dg, NULL, NULL, 64, stream);
        /* The same columns as two calls of 19 and 26, tile chosen freely. */
        if (st == VV_OK)
            st = vv_vae_gemm_nn_dev(epis[e], dA, dB, P, dC2, P, M, K, P1, db,
                                    dg, NULL, NULL, 0, stream);
        if (st == VV_OK)
            st = vv_vae_gemm_nn_dev(epis[e], dA, (uint16_t*)dB + P1, P,
                                    (uint16_t*)dC2 + P1, P, M, K, P - P1, db,
                                    dg, NULL, NULL, 0, stream);
        TEST_ASSERT(st == VV_OK, "GEMM runs with both tiles");
        uint16_t* c1 = download(dC, (size_t)M * P, stream);
        uint16_t* c2 = download(dC2, (size_t)M * P, stream);
        uint16_t* c3 = download(dC3, (size_t)M * P, stream);
        TEST_ASSERT(memcmp(c1, c3, (size_t)M * P * 2) == 0,
                    "GEMM: 64 tile == 128 tile, bit for bit");
        TEST_ASSERT(memcmp(c1, c2, (size_t)M * P * 2) == 0,
                    "GEMM: one call == two narrower calls, bit for bit");
        double num = 0.0, den = 0.0;
        for (int r = 0; r < M; r++)
            for (int c = 0; c < P; c++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++)
                    acc += (double)vv_half_to_float(A[(size_t)r * K + k]) *
                           (double)vv_half_to_float(B[(size_t)k * P + c]);
                acc += vv_half_to_float(bias[r]);
                double want;
                if (epis[e] == VV_VAE_EPI_BIAS) want = acc;
                else if (epis[e] == VV_VAE_EPI_BIAS_GELU)
                    want = 0.5 * acc * (1.0 + erf(acc / sqrt(2.0)));
                else
                    want = vv_half_to_float(X[(size_t)r * P + c]) +
                           acc * vv_half_to_float(gam[r]);
                const double got = vv_half_to_float(c1[(size_t)r * P + c]);
                num += (got - want) * (got - want);
                den += want * want;
            }
        const double rel = sqrt(num / (den > 0 ? den : 1));
        printf("    epilogue %d rel=%.2e\n", epis[e], rel);
        TEST_ASSERT(rel < 2e-3, "GEMM matches the FP64 reference");
        vv_free(c1);
        vv_free(c2);
        vv_free(c3);
    }
    vv_dev_stream_sync(stream);
    vv_dev_free(dA); vv_dev_free(dB); vv_dev_free(dC); vv_dev_free(dC2);
    vv_dev_free(dC3); vv_dev_free(db); vv_dev_free(dg);
    vv_free(A); vv_free(B); vv_free(X);
}

/* ─── Conv as im2col GEMM ──────────────────────────────────────────────── */

/**
 * The downsample and head convs run as im2col + an FP32 GEMM that sums in
 * the direct kernel's order: both must give the direct kernel's bits, for
 * packed and transposed outputs, with a carried context, a fresh one, the
 * zero padding past a final chunk, and a column window (p0) mid-item.
 */
static void test_conv_gemm(void* stream) {
    const int IC = 24, OC = 80, KW = 10, SD = 5, CAP = 8;
    const int in_len[2] = { 203, 97 };
    const int have[2] = { 5, 5 };
    const int out_len[2] = { 42, 21 };
    const int T_in = in_len[0] + in_len[1], T_out = out_len[0] + out_len[1];
    const int64_t ld_in = 304, ld_out = 64;
    const int Kc = IC * KW;
    const int skip0 = 2, OLD = OC + 3;
    uint64_t rng = 7;

    uint16_t* x = (uint16_t*)vv_alloc((size_t)IC * ld_in * 2);
    uint16_t* w = (uint16_t*)vv_alloc((size_t)OC * Kc * 2);
    uint16_t* tl = (uint16_t*)vv_alloc((size_t)IC * CAP * 2);
    uint16_t b[80];
    for (size_t i = 0; i < (size_t)IC * ld_in; i++) x[i] = vv_float_to_half(frand(&rng));
    for (size_t i = 0; i < (size_t)OC * Kc; i++)
        w[i] = vv_float_to_half(frand(&rng) * 0.1f);
    for (size_t i = 0; i < (size_t)IC * CAP; i++) tl[i] = vv_float_to_half(frand(&rng));
    for (int i = 0; i < OC; i++) b[i] = vv_float_to_half(frand(&rng) * 0.1f);

    void *dx, *dw, *dtl, *db, *dy1, *dy2, *dcol, *dr1, *dr2;
    const size_t rows_elems = (size_t)(out_len[0] + out_len[1]) * OLD;
    vv_dev_alloc(&dx, (size_t)IC * ld_in * 2);
    vv_dev_alloc(&dw, (size_t)OC * Kc * 2);
    vv_dev_alloc(&dtl, (size_t)IC * CAP * 2);
    vv_dev_alloc(&db, (size_t)OC * 2);
    vv_dev_alloc(&dy1, (size_t)OC * ld_out * 2);
    vv_dev_alloc(&dy2, (size_t)OC * ld_out * 2);
    vv_dev_alloc(&dcol, (size_t)Kc * 64 * 2);
    vv_dev_alloc(&dr1, rows_elems * 2);
    vv_dev_alloc(&dr2, rows_elems * 2);
    vv_dev_memcpy_h2d(dx, x, (size_t)IC * ld_in * 2, stream);
    vv_dev_memcpy_h2d(dw, w, (size_t)OC * Kc * 2, stream);
    vv_dev_memcpy_h2d(dtl, tl, (size_t)IC * CAP * 2, stream);
    vv_dev_memcpy_h2d(db, b, (size_t)OC * 2, stream);
    vv_dev_memset_async(dy1, 0, (size_t)OC * ld_out * 2, stream);
    vv_dev_memset_async(dy2, 0, (size_t)OC * ld_out * 2, stream);
    vv_dev_memset_async(dr1, 0, rows_elems * 2, stream);
    vv_dev_memset_async(dr2, 0, rows_elems * 2, stream);

    vv_vae_conv_desc_t d;
    memset(&d, 0, sizeof(d));
    d.n = 2;
    d.cap = CAP;
    int64_t io = 0, oo = 0;
    for (int i = 0; i < 2; i++) {
        d.it[i].in_off = io;
        d.it[i].out_off = oo;
        d.it[i].tail_in = i == 0 ? dtl : NULL;   /* carried, then fresh */
        d.it[i].in_len = in_len[i];
        d.it[i].out_len = out_len[i];
        d.it[i].have = have[i];
        io += in_len[i];
        oo += out_len[i];
    }
    (void)T_in;

    /* Packed: direct vs two im2col windows (41 + 22 columns). */
    vv_status_t st = vv_vae_conv_dev(&d, dx, ld_in, dw, db, dy1, ld_out, IC, OC,
                                     KW, SD, false, stream);
    const int cut = 41;
    for (int part = 0; part < 2 && st == VV_OK; part++) {
        const int64_t p0 = part ? cut : 0;
        const int pc = part ? T_out - cut : cut;
        st = vv_vae_im2col_dev(&d, dx, ld_in, IC, KW, SD, p0, pc, dcol, 64,
                               stream);
        if (st == VV_OK)
            st = vv_vae_conv_gemm_dev(NULL, dw, dcol, 64, db,
                                      (uint16_t*)dy2 + p0, ld_out, OC, Kc, p0,
                                      pc, stream);
    }
    TEST_ASSERT(st == VV_OK, "conv GEMM runs (packed)");
    uint16_t* y1 = download(dy1, (size_t)OC * ld_out, stream);
    uint16_t* y2 = download(dy2, (size_t)OC * ld_out, stream);
    TEST_ASSERT(memcmp(y1, y2, (size_t)OC * ld_out * 2) == 0,
                "conv GEMM == direct conv, packed, bit for bit");

    /* Transposed rows with a skipped head frame: the head's layout. */
    vv_vae_conv_desc_t d1 = d, d2 = d;
    for (int i = 0; i < 2; i++) {
        const size_t r0 = i ? (size_t)out_len[0] : 0;
        d1.it[i].out_ptr = (uint16_t*)dr1 + r0 * OLD;
        d2.it[i].out_ptr = (uint16_t*)dr2 + r0 * OLD;
        d1.it[i].out_ld = d2.it[i].out_ld = OLD;
        d1.it[i].skip = d2.it[i].skip = i == 0 ? skip0 : 0;
    }
    st = vv_vae_conv_dev(&d1, dx, ld_in, dw, db, NULL, 0, IC, OC, KW, SD, true,
                         stream);
    for (int part = 0; part < 2 && st == VV_OK; part++) {
        const int64_t p0 = part ? cut : 0;
        const int pc = part ? T_out - cut : cut;
        st = vv_vae_im2col_dev(&d2, dx, ld_in, IC, KW, SD, p0, pc, dcol, 64,
                               stream);
        if (st == VV_OK)
            st = vv_vae_conv_gemm_dev(&d2, dw, dcol, 64, db, NULL, 0, OC, Kc,
                                      p0, pc, stream);
    }
    TEST_ASSERT(st == VV_OK, "conv GEMM runs (transposed)");
    uint16_t* r1 = download(dr1, rows_elems, stream);
    uint16_t* r2 = download(dr2, rows_elems, stream);
    TEST_ASSERT(memcmp(r1, r2, rows_elems * 2) == 0,
                "conv GEMM == direct conv, transposed with skip, bit for bit");

    vv_free(y1); vv_free(y2); vv_free(r1); vv_free(r2);
    vv_dev_stream_sync(stream);
    vv_dev_free(dx); vv_dev_free(dw); vv_dev_free(dtl); vv_dev_free(db);
    vv_dev_free(dy1); vv_dev_free(dy2); vv_dev_free(dcol);
    vv_dev_free(dr1); vv_dev_free(dr2);
    vv_free(x); vv_free(w); vv_free(tl);
}

int main(void) {
    printf("=== VibeVoice Conv-VAE Streaming Tests ===\n\n");
    vv_log_set_level(VV_LOG_WARN);

    tiny_t ta, ts;
    tiny_build(&ta, "model.acoustic_tokenizer.encoder.", 1);
    tiny_build(&ts, "model.semantic_tokenizer.encoder.", 2);
    vv_conv_vae_encoder_t* ea = tiny_encoder(&ta, true);
    vv_conv_vae_encoder_t* es = tiny_encoder(&ts, false);
    TEST_ASSERT(ea && vv_conv_vae_complete(ea), "tiny acoustic encoder binds");
    TEST_ASSERT(es && vv_conv_vae_complete(es), "tiny semantic encoder binds");
    if (!ea || !es) return 1;

    test_frames(ea);
    test_cpu_vs_reference(ea);

    if (vv_dev_device_count() > 0 && vv_dev_set_device(0) == VV_OK) {
        gpu_t g;
        memset(&g, 0, sizeof(g));
        g.e = ea;
        vv_dev_stream_create(&g.stream);
        vv_status_t s = vv_vae_weights_upload(ea, g.stream, &g.w);
        if (s == VV_OK) s = vv_vae_arena_create(ea, 8, 100000, &g.a);
        TEST_ASSERT(s == VV_OK, "GPU weights and arena");
        if (s == VV_OK) {
            test_gemm_tiles(g.stream);
            test_conv_gemm(g.stream);
            test_gpu_vs_reference(&g);
            test_gpu_chunking(&g);
            test_gpu_batching(&g);
            test_gpu_window(&g);
            test_frontend(ea, es);
        }
        vv_vae_arena_free(g.a);
        vv_vae_weights_free(g.w);
        vv_dev_stream_destroy(g.stream);
    } else {
        printf("  SKIP: no GPU — GPU cases not run\n");
    }

    vv_conv_vae_free(ea);
    vv_conv_vae_free(es);
    tiny_free(&ta);
    tiny_free(&ts);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
