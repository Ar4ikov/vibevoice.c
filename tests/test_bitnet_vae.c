/**
 * @file test_bitnet_vae.c
 * @brief VibeASR.cpp's int8 speech encoder (vae_i8.c) against a plain
 *        transcription of the reference's ops, exactly.
 *
 * Part 1 builds a tiny random two-stage encoder and runs it through
 * vv_i8vae_encode at several thread counts; the reference below is written
 * op by op the obvious way (no tiling, no packing, no SIMD, one thread),
 * with the expressions the reference's GCC build computes -- fused
 * multiply-adds in the mul_mat_add epilogue and in add_scaled, and
 * add_scaled's per-thread scalar head and tail. Every feature must be equal
 * to the bit.
 *
 * Part 2, with VV_BITNET_MODEL pointing at the VibeVoice-ASR-BitNet
 * directory, encodes a synthetic 3 s clip with the shipped I8_S encoder at
 * four threads and compares a hash of the features with the one VibeASR.cpp
 * produced for the same clip (`asr_infer -t 4`, features dumped before the
 * prefill). VV_BITNET_WRITE_WAV=<path> writes that clip as a WAV so the
 * reference can be rerun.
 */

#include "vibevoice/vae_i8.h"
#include "vibevoice/audio.h"
#include "vibevoice/model.h"
#include "vibevoice/types.h"
#include "vibevoice/vibevoice.h"
#include "../src/model/bitnet_load.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static uint32_t rng = 20260919u;

static uint32_t urand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static float frand(void) { return (float)(urand() >> 8) / 16777216.0f * 2.0f - 1.0f; }

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("  FAIL: " __VA_ARGS__);               \
            printf("\n");                                 \
            failures++;                                   \
        }                                                 \
    } while (0)

/* ─── The reference, op by op ───────────────────────────────────────────── */

typedef struct {
    int8_t* q;
    int T, C;
    float s;
} ract_t;

static int rne_i(float f) {
    float v = f + 12582912.0f;
    int32_t i;
    memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}

static void rfree(ract_t* a) { free(a->q); a->q = NULL; }

/* global max -> id, then requantize y (ggml_i8_s_quantize_range) */
static void requant_all(const float* y, int n, float lo, ract_t* out) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(y[i]) > m) m = fabsf(y[i]);
    const float id = m != 0.0f ? 127.0f / m : 0.0f;
    for (int i = 0; i < n; i++) {
        float v = y[i] * id;
        v = v > lo ? v : lo;
        v = v < 127.0f ? v : 127.0f;
        out->q[i] = (int8_t)rne_i(v);
    }
    out->s = id;
}

/* conv / linear / depthwise, causal, left pad k - stride, no right pad */
static ract_t rconv(const vv_i8_layer_t* L, const ract_t* x, int relu) {
    const int k = L->k, s = L->stride, lp = k - s;
    const int M = (x->T + lp < k) ? 0 : (x->T + lp - k) / s + 1;
    const int N = L->out_ch;
    ract_t y;
    y.T = M;
    y.C = N;
    y.q = (int8_t*)malloc((size_t)M * N + 1);
    float* f = (float*)malloc(sizeof(float) * ((size_t)M * N + 1));
    const float d = L->w_scale / x->s;
    for (int t = 0; t < M; t++)
        for (int o = 0; o < N; o++) {
            int32_t acc = 0;
            for (int j = 0; j < k; j++) {
                const int ti = t * s - lp + j;
                if (ti < 0) continue;
                if (L->depthwise) {
                    acc += (int32_t)L->w[(size_t)j * N + o] * x->q[(size_t)ti * x->C + o];
                } else {
                    for (int i = 0; i < x->C; i++)
                        acc += (int32_t)L->w[((size_t)o * k + j) * x->C + i] *
                               x->q[(size_t)ti * x->C + i];
                }
            }
            f[(size_t)t * N + o] = fmaf((float)acc, d, L->bias[o]);
        }
    requant_all(f, M * N, relu ? 0.0f : -127.0f, &y);
    free(f);
    return y;
}

static ract_t rnorm(const float* g, float eps, const ract_t* x) {
    ract_t y;
    y.T = x->T;
    y.C = x->C;
    y.q = (int8_t*)malloc((size_t)x->T * x->C + 1);
    float* f = (float*)malloc(sizeof(float) * ((size_t)x->T * x->C + 1));
    const float es = (float)((double)eps * (double)x->s * (double)x->s);
    for (int t = 0; t < x->T; t++) {
        int32_t ss = 0;
        for (int c = 0; c < x->C; c++) ss += x->q[t * x->C + c] * x->q[t * x->C + c];
        const float r = 1.0f / sqrtf((float)ss / (float)x->C + es);
        for (int c = 0; c < x->C; c++)
            f[t * x->C + c] = ((float)x->q[t * x->C + c] * r) * g[c];
    }
    requant_all(f, x->T * x->C, -127.0f, &y);
    free(f);
    return y;
}

/* ggml_compute_forward_add_scaled over nth equal flat ranges, into b */
static void radd(const ract_t* a, const float* g, ract_t* b, int nth) {
    const int n = a->T * a->C, C = a->C;
    float* f = (float*)malloc(sizeof(float) * ((size_t)n + 1));
    const float ia = 1.0f / a->s, ib = 1.0f / b->s;
    const int dr = (n + nth - 1) / nth;
    for (int th = 0; th < nth; th++) {
        const int i0 = dr * th < n ? dr * th : n;
        const int i1 = i0 + dr < n ? i0 + dr : n;
        int i = i0;
        if (i0 % C) {
            int e = i0 + (C - i0 % C);
            if (e > i1) e = i1;
            for (; i < e; i++)
                f[i] = fmaf((float)a->q[i] / a->s, g[i % C], (float)b->q[i] / b->s);
        }
        for (; i + 7 < i1; i += 8)
            for (int j = 0; j < 8; j++)
                f[i + j] = fmaf((float)b->q[i + j], ib,
                                ((float)a->q[i + j] * ia) * g[(i + j) % C]);
        for (; i < i1; i++)
            f[i] = fmaf((float)a->q[i] / a->s, g[i % C], (float)b->q[i] / b->s);
    }
    float m = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(f[i]) > m) m = fabsf(f[i]);
    const float id = m != 0.0f ? 127.0f / m : 0.0f;
    for (int th = 0; th < nth; th++) {
        const int i0 = dr * th < n ? dr * th : n;
        const int i1 = i0 + dr < n ? i0 + dr : n;
        int i = i0;
        for (; i + 7 < i1; i += 8)
            for (int j = 0; j < 8; j++) {
                float v = f[i + j] * id;
                v = v > -127.0f ? v : -127.0f;
                v = v < 127.0f ? v : 127.0f;
                b->q[i + j] = (int8_t)rne_i(v);
            }
        for (; i < i1; i++) {
            float v = f[i] * id;
            v = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
            b->q[i] = (int8_t)roundf(v);
        }
    }
    b->s = id;
    free(f);
}

static void rtower(const vv_i8vae_t* v, const vv_i8_tower_t* tw,
                   const float* audio, int n, int nth, float* out, int* nf) {
    float amax = 0.00001f;
    for (int i = 0; i < n; i++) if (fabsf(audio[i]) > amax) amax = fabsf(audio[i]);
    ract_t x;
    x.T = n;
    x.C = 1;
    x.s = 127.0f / amax;
    x.q = (int8_t*)malloc((size_t)n);
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(audio[i] * x.s);
        x.q[i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
    ract_t y = rconv(&tw->ds[0], &x, 0);
    rfree(&x);
    x = y;
    for (int st = 0; st < tw->n_stages; st++) {
        if (st > 0) {
            y = rconv(&tw->ds[st], &x, 0);
            rfree(&x);
            x = y;
        }
        for (int b = 0; b < tw->depth[st]; b++) {
            const vv_i8_block_t* B = &tw->blocks[st][b];
            ract_t h = rnorm(B->norm, v->block_eps, &x);
            ract_t m = rconv(&B->mixer, &h, 0);
            rfree(&h);
            radd(&m, B->gamma, &x, nth);
            rfree(&m);
            h = rnorm(B->ffn_norm, v->block_eps, &x);
            ract_t f1 = rconv(&B->fc1, &h, 1);
            rfree(&h);
            ract_t f2 = rconv(&B->fc2, &f1, 0);
            rfree(&f1);
            radd(&f2, B->ffn_gamma, &x, nth);
            rfree(&f2);
        }
    }
    ract_t h = rconv(&tw->head, &x, 0);
    rfree(&x);
    ract_t c1 = rconv(&tw->cfc1, &h, 0);
    rfree(&h);
    ract_t cn = rnorm(tw->cnorm, v->conn_eps, &c1);
    rfree(&c1);
    ract_t c2 = rconv(&tw->cfc2, &cn, 0);
    rfree(&cn);
    const float dq = 1.0f / c2.s;
    for (int i = 0; i < c2.T * c2.C; i++) out[i] = (float)c2.q[i] * dq;
    *nf = c2.T;
    rfree(&c2);
}

/* ─── A tiny random encoder ─────────────────────────────────────────────── */

static vv_status_t rand_layer(vv_i8vae_t* v, int out, int in, int k, int stride,
                              int dw, vv_i8_layer_t* L) {
    const int n = dw ? out * k : out * in * k;
    int8_t* q = (int8_t*)malloc((size_t)n);
    float* b = (float*)malloc(sizeof(float) * (size_t)out);
    for (int i = 0; i < n; i++) q[i] = (int8_t)((int)(urand() % 255) - 127);
    for (int i = 0; i < out; i++) b[i] = frand() * 0.05f;
    vv_status_t s = vv_i8vae_layer_from_q(v, q, 0.01f + 0.02f * fabsf(frand()),
                                          out, dw ? 1 : in, k, k, stride, dw, b, L);
    free(q);
    free(b);
    return s;
}

static float* rand_vec(vv_i8vae_t* v, int n, float base, float spread) {
    float* f = (float*)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++) f[i] = base + spread * frand();
    float* d = vv_i8vae_copy_f32(v, f, (size_t)n);
    free(f);
    return d;
}

static vv_status_t build_tiny(vv_i8vae_t** out) {
    vv_i8vae_t* v = NULL;
    vv_status_t s = vv_i8vae_create(&v);
    if (s != VV_OK) return s;
    static const int depth[2] = { 1, 2 };
    for (int t = 0; t < 2 && s == VV_OK; t++) {
        vv_i8_tower_t* tw = &v->tower[t];
        s = vv_i8vae_tower_alloc(v, tw, 2, depth);
        const int C[2] = { 8, 16 };
        if (s == VV_OK) s = rand_layer(v, C[0], 1, 7, 1, 0, &tw->ds[0]);
        if (s == VV_OK) s = rand_layer(v, C[1], C[0], 4, 2, 0, &tw->ds[1]);
        for (int st = 0; st < 2 && s == VV_OK; st++)
            for (int b = 0; b < depth[st] && s == VV_OK; b++) {
                vv_i8_block_t* B = &tw->blocks[st][b];
                s = rand_layer(v, C[st], C[st], 7, 1, 1, &B->mixer);
                if (s == VV_OK) s = rand_layer(v, 4 * C[st], C[st], 1, 1, 0, &B->fc1);
                if (s == VV_OK) s = rand_layer(v, C[st], 4 * C[st], 1, 1, 0, &B->fc2);
                B->norm = rand_vec(v, C[st], 1.0f, 0.5f);
                B->ffn_norm = rand_vec(v, C[st], 1.0f, 0.5f);
                /* layer scales large enough that the branch moves bits */
                B->gamma = rand_vec(v, C[st], 0.0f, 0.5f);
                B->ffn_gamma = rand_vec(v, C[st], 0.0f, 0.5f);
            }
        const int vd = t == 0 ? 4 : 8;
        if (s == VV_OK) s = rand_layer(v, vd, C[1], 7, 1, 0, &tw->head);
        if (s == VV_OK) s = rand_layer(v, 16, vd, 1, 1, 0, &tw->cfc1);
        if (s == VV_OK) s = rand_layer(v, 16, 16, 1, 1, 0, &tw->cfc2);
        tw->cnorm = rand_vec(v, 16, 1.0f, 0.5f);
    }
    if (s == VV_OK) s = vv_i8vae_prepare(v);
    if (s != VV_OK) { vv_i8vae_free(v); return s; }
    *out = v;
    return VV_OK;
}

static void test_tiny(void) {
    printf("tiny encoder vs the op-by-op reference\n");
    vv_i8vae_t* v = NULL;
    vv_status_t s = build_tiny(&v);
    CHECK(s == VV_OK, "build: %s", vv_status_str(s));
    if (s != VV_OK) return;
    /* lengths that leave ragged thread ranges and odd frame counts */
    const int lens[3] = { 2 * 1500 + 7, 4099, 811 };
    for (int li = 0; li < 3; li++) {
        const int n = lens[li];
        float* audio = (float*)malloc(sizeof(float) * (size_t)n);
        for (int i = 0; i < n; i++)
            audio[i] = 0.3f * sinf((float)i * 0.013f) + 0.05f * frand();
        const int cap = vv_i8vae_frames(v, n) + 1;
        float* got = (float*)calloc((size_t)cap * 16, sizeof(float));
        float* want = (float*)calloc((size_t)cap * 16, sizeof(float));
        float* w2 = (float*)calloc((size_t)cap * 16, sizeof(float));
        const int threads[4] = { 1, 3, 5, 12 };
        for (int ti = 0; ti < 4; ti++) {
            v->n_threads = threads[ti];
            int nf = 0, na = 0, ns = 0;
            s = vv_i8vae_encode(v, audio, n, 0, got, &nf);
            CHECK(s == VV_OK, "encode n=%d t=%d: %s", n, threads[ti], vv_status_str(s));
            rtower(v, &v->tower[0], audio, n, threads[ti], want, &na);
            rtower(v, &v->tower[1], audio, n, threads[ti], w2, &ns);
            const int f = na < ns ? na : ns;
            CHECK(nf == f && nf == vv_i8vae_frames(v, n),
                  "frames n=%d: got %d, reference %d/%d", n, nf, na, ns);
            int bad = 0;
            for (int i = 0; i < f * 16; i++) {
                const float r = want[i] + w2[i];
                if (memcmp(&r, &got[i], 4) != 0) bad++;
            }
            CHECK(bad == 0, "n=%d t=%d: %d of %d features differ", n, threads[ti],
                  bad, f * 16);
        }
        free(audio); free(got); free(want); free(w2);
    }
    vv_i8vae_free(v);
}

/* ─── The shipped encoder against VibeASR.cpp ───────────────────────────── */

/* 3 s at 24 kHz: two chirps, a burst and noise, as 16-bit PCM. */
static void synth_clip(int16_t* pcm, int n) {
    uint32_t r = 7u;
    for (int i = 0; i < n; i++) {
        const double t = (double)i / 24000.0;
        double v = 0.30 * sin(2 * 3.141592653589793 * (180.0 + 90.0 * t) * t)
                 + 0.15 * sin(2 * 3.141592653589793 * (1200.0 - 200.0 * t) * t);
        if (t > 1.2 && t < 1.5) v *= 2.2;
        r = r * 1664525u + 1013904223u;
        v += 0.02 * ((double)(r >> 8) / 16777216.0 - 0.5);
        long q = lround(v * 32767.0);
        pcm[i] = (int16_t)(q > 32767 ? 32767 : (q < -32768 ? -32768 : q));
    }
}

static void put32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }

static void write_wav(const char* path, const int16_t* pcm, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("  cannot write %s\n", path); return; }
    fwrite("RIFF", 1, 4, f); put32(f, 36 + (uint32_t)n * 2);
    fwrite("WAVEfmt ", 1, 8, f); put32(f, 16); put16(f, 1); put16(f, 1);
    put32(f, 24000); put32(f, 48000); put16(f, 2); put16(f, 16);
    fwrite("data", 1, 4, f); put32(f, (uint32_t)n * 2);
    fwrite(pcm, 2, (size_t)n, f);
    fclose(f);
    printf("  wrote %s\n", path);
}

static uint64_t fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/*
 * FNV-1a over the float bits of VibeASR.cpp's features for synth_clip
 * (acoustic + semantic, [frames][1536]), from `asr_infer -t 4 --greedy`
 * with the features dumped before the prefill (22 frames).
 */
#define REF_FEATURE_HASH 0x7bb2de8527a12e56ull

static void test_model(const char* dir) {
    printf("shipped I8_S encoder vs VibeASR.cpp (%s)\n", dir);
    const int n = 72000;
    int16_t* pcm = (int16_t*)malloc(sizeof(int16_t) * (size_t)n);
    synth_clip(pcm, n);
    const char* wav = getenv("VV_BITNET_WRITE_WAV");
    if (wav && wav[0]) write_wav(wav, pcm, n);
    float* raw = (float*)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++) raw[i] = (float)pcm[i] / 32768.0f;   /* dr_wav */
    free(pcm);
    float* audio = NULL;
    int len = 0;
    vv_status_t s = vv_audio_prepare_vibeasr(raw, n, 24000, true, &audio, &len);
    free(raw);
    CHECK(s == VV_OK && len == n, "audio prep: %s", vv_status_str(s));
    if (s != VV_OK) return;

    vv_model_load_opts_t o = vv_model_load_opts_default();
    o.vae = VV_VAE_INT8;
    o.cpu = 1;
    vv_model_t* m = NULL;
    s = vv_model_load_ex(dir, &o, &m);
    CHECK(s == VV_OK && m && m->i8vae, "load: %s", vv_status_str(s));
    if (s != VV_OK || !m || !m->i8vae) { vv_free(audio); return; }
    m->i8vae->n_threads = 4;
    const int hs = m->i8vae->tower[0].hidden;
    const int cap = vv_i8vae_frames(m->i8vae, len);
    float* feat = (float*)vv_alloc((size_t)cap * hs * sizeof(float));
    int nf = 0;
    s = vv_i8vae_encode(m->i8vae, audio, len, 0, feat, &nf);
    CHECK(s == VV_OK && nf == 22, "encode: %s, %d frames", vv_status_str(s), nf);
    if (s == VV_OK) {
        const uint64_t h = fnv(feat, (size_t)nf * hs * sizeof(float));
        printf("  features: %d x %d, hash %016llx\n", nf, hs, (unsigned long long)h);
        CHECK(h == REF_FEATURE_HASH, "hash %016llx, VibeASR.cpp's %016llx",
              (unsigned long long)h, (unsigned long long)REF_FEATURE_HASH);
    }
    vv_free(feat);
    vv_free(audio);
    vv_model_free(m);
}

int main(void) {
    test_tiny();
    const char* dir = getenv("VV_BITNET_MODEL");
    if (dir && dir[0]) test_model(dir);
    else printf("SKIP model check: VV_BITNET_MODEL not set\n");
    if (failures) {
        printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("all int8 encoder checks passed\n");
    return 0;
}
