/**
 * @file test_w4a16_model.c
 * @brief The W4A16 path on a real AWQ / GPTQ checkpoint, through the loader.
 *
 * test_w4a16 exercises the repacks and kernels on synthetic tensors; this
 * one goes through what the pipeline does with a checkpoint: vv_model_load
 * (AWQ vs GPTQ told apart by shape, gptq_v2 zeros from config.json,
 * act-order channels sorted into w->perm), then vv_model_int4g_to_gpu_layout
 * (scales, zeros and mins replaced by the half2 {scale, zero} tensor, in
 * place). For the first, second and last layer every projection is checked
 * on the GPU (GEMV, fused q/k/v and gate/up GEMV, tensor-core GEMM, with the
 * act-order gather where the weight has one) against vv_int4g_gemm_cpu on a
 * copy of the row-major form taken before the conversion.
 *
 * Runs when VV_TEST_MODEL names an AWQ or GPTQ checkpoint; otherwise (no
 * model, an NF4 or dense one, no GPU) it reports SKIP and passes.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"
#include "vibevoice/model.h"
#include "vibevoice/quant.h"
#include "vibevoice/cpu_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;

static uint32_t rng_state = 0x2545F491u;
static float frand(void) {                       /* [-1, 1) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;
}

/** @brief Host copy of one projection in the loader's row-major form. */
typedef struct {
    const vv_weight_t* w;
    const char* what;
    int N, K, G;
    uint8_t*  packed;
    uint16_t* scales;
    uint16_t* mins;
    int32_t*  perm;       /* NULL unless act-order */
} rowmajor_t;

static void* dup(const void* p, size_t n) {
    if (!p || !n) return NULL;
    void* q = malloc(n);
    if (q) memcpy(q, p, n);
    return q;
}

static int snapshot(const vv_weight_t* w, const char* what, rowmajor_t* r) {
    memset(r, 0, sizeof(*r));
    r->w = w; r->what = what;
    r->N = (int)w->tensor.shape[0];
    r->K = (int)w->tensor.shape[1] * 2;
    r->G = w->group_size;
    r->packed = (uint8_t*)dup(w->tensor.data, w->tensor.size_bytes);
    r->scales = (uint16_t*)dup(w->quant.scales.data, w->quant.scales.size_bytes);
    r->mins   = (uint16_t*)dup(w->mins.data, w->mins.size_bytes);
    r->perm   = (int32_t*)dup(w->perm.data, w->perm.size_bytes);
    return !r->packed || !r->scales || !r->mins ||
           (w->perm.data && !r->perm);
}

static void rowmajor_free(rowmajor_t* r) {
    free(r->packed); free(r->scales); free(r->mins); free(r->perm);
}

/** @brief CPU reference for x [M][K]: gather through perm, then the kernel. */
static void cpu_ref(const rowmajor_t* r, const float* x, int M, float* y) {
    const int K = r->K;
    float* xg = NULL;
    if (r->perm) {
        xg = (float*)malloc((size_t)M * K * sizeof(float));
        for (int m = 0; m < M; m++)
            for (int j = 0; j < K; j++)
                xg[(size_t)m * K + j] = x[(size_t)m * K + r->perm[j]];
        x = xg;
    }
    vv_int4g_gemm_cpu(x, r->packed, r->scales, r->mins, r->w->bias.data, y,
                      M, r->N, K, r->G);
    free(xg);
}

static void compare(const char* tag, const uint16_t* hy, const float* ref,
                    size_t n) {
    double num = 0, da = 0, db = 0, worst = 0;
    for (size_t i = 0; i < n; i++) {
        const double g = vv_half_to_float(hy[i]), r = ref[i];
        num += g * r; da += g * g; db += r * r;
        const double e = fabs(g - r) / (fabs(r) + 1.0);
        if (e > worst || e != e) worst = (e != e) ? 1e9 : e;
    }
    const double cosv = num / (sqrt(da) * sqrt(db) + 1e-300);
    const bool ok = cosv > 0.99999 && worst < 0.05;
    printf("  %s %-34s cos %.8f  worst %.2e\n", ok ? "ok  " : "FAIL", tag,
           cosv, worst);
    if (!ok) failures++;
}

/** @brief Device copies of a converted weight. */
typedef struct { void *p, *sz, *bias, *perm; } dev_w_t;

static int upload(const vv_weight_t* w, dev_w_t* d) {
    memset(d, 0, sizeof(*d));
    if (vv_dev_alloc(&d->p, w->tensor.size_bytes) ||
        vv_dev_alloc(&d->sz, w->quant.scales.size_bytes))
        return 1;
    vv_dev_memcpy_h2d(d->p, w->tensor.data, w->tensor.size_bytes, NULL);
    vv_dev_memcpy_h2d(d->sz, w->quant.scales.data, w->quant.scales.size_bytes,
                      NULL);
    if (w->bias.data) {
        if (vv_dev_alloc(&d->bias, w->bias.size_bytes)) return 1;
        vv_dev_memcpy_h2d(d->bias, w->bias.data, w->bias.size_bytes, NULL);
    }
    if (w->perm.data) {
        if (vv_dev_alloc(&d->perm, w->perm.size_bytes)) return 1;
        vv_dev_memcpy_h2d(d->perm, w->perm.data, w->perm.size_bytes, NULL);
    }
    return 0;
}

static void dev_free(dev_w_t* d) {
    vv_dev_free(d->p); vv_dev_free(d->sz); vv_dev_free(d->bias);
    vv_dev_free(d->perm);
}

/**
 * @brief GPU y = x W^T for M rows, x already on the device; the act-order
 *        gather goes into `xg` first.
 */
static vv_status_t gpu_run(const rowmajor_t* r, const dev_w_t* d,
                           const void* dx, void* xg, void* dy, int M,
                           void* scratch, size_t scratch_bytes, void* st) {
    const void* xin = dx;
    if (d->perm) {
        vv_status_t s = vv_w4a16_gather_dev(dx, (const int32_t*)d->perm, xg,
                                            M, r->K, st);
        if (s != VV_OK) return s;
        xin = xg;
    }
    if (M == 1)
        return vv_w4a16_gemv_dev(xin, d->p, d->sz, d->bias, dy, r->N, r->K,
                                 r->G, st);
    return vv_w4a16_gemm_dev(xin, d->p, d->sz, d->bias, dy, scratch,
                             scratch_bytes, M, r->N, r->K, r->G, st);
}

int main(void) {
    const char* dir = getenv("VV_TEST_MODEL");
    if (!dir || !dir[0]) {
        printf("SKIP: VV_TEST_MODEL not set\n");
        return 0;
    }
#ifndef VV_HAS_ACCEL
    printf("SKIP: built without an accelerator backend\n");
    return 0;
#else
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(0, &total, &freem, NULL) != VV_OK || total == 0) {
        printf("SKIP: no device available\n");
        return 0;
    }
    vv_dev_set_device(0);

    vv_model_t* model = NULL;
    if (vv_model_load(dir, &model) != VV_OK || !model) {
        printf("FAIL: cannot load '%s'\n", dir);
        return 1;
    }
    int n_int4 = 0, n_perm = 0;
    for (int li = 0; li < model->num_layers; li++) {
        const vv_layer_weights_t* L = &model->layers[li];
        const vv_weight_t* ws[7] = { &L->attn.q_proj, &L->attn.k_proj,
            &L->attn.v_proj, &L->attn.o_proj, &L->mlp.gate_proj,
            &L->mlp.up_proj, &L->mlp.down_proj };
        for (int i = 0; i < 7; i++) {
            if (ws[i]->quant_kind == VV_QUANT_INT4G) n_int4++;
            if (ws[i]->perm.data) n_perm++;
        }
    }
    if (n_int4 == 0) {
        printf("SKIP: '%s' has no AWQ/GPTQ projections\n", dir);
        vv_model_free(model);
        return 0;
    }
    printf("=== %s: %d INT4 projections, %d act-order ===\n", dir, n_int4,
           n_perm);

    /* Snapshot the row-major form of the layers under test. */
    const int nl = model->num_layers;
    const int layers[3] = { 0, nl > 1 ? 1 : 0, nl - 1 };
    const char* names[7] = { "q", "k", "v", "o", "gate", "up", "down" };
    rowmajor_t rm[3][7];
    memset(rm, 0, sizeof(rm));
    for (int t = 0; t < 3; t++) {
        const vv_layer_weights_t* L = &model->layers[layers[t]];
        const vv_weight_t* ws[7] = { &L->attn.q_proj, &L->attn.k_proj,
            &L->attn.v_proj, &L->attn.o_proj, &L->mlp.gate_proj,
            &L->mlp.up_proj, &L->mlp.down_proj };
        for (int i = 0; i < 7; i++) {
            if (ws[i]->quant_kind != VV_QUANT_INT4G) continue;
            if (ws[i]->int4g_layout != VV_INT4G_ROWMAJOR ||
                snapshot(ws[i], names[i], &rm[t][i])) {
                printf("FAIL: layer %d %s: not row-major after load\n",
                       layers[t], names[i]);
                failures++;
            }
        }
    }

    int converted = 0;
    if (vv_model_int4g_to_gpu_layout(model, &converted) != VV_OK) {
        printf("FAIL: GPU layout conversion\n");
        return 1;
    }
    printf("  %s %d of %d projections converted\n",
           converted == n_int4 ? "ok  " : "FAIL", converted, n_int4);
    if (converted != n_int4) failures++;

    void* st = NULL;
    vv_dev_stream_create(&st);
    const int MG = 37;                       /* one GEMM M, not a tile size */
    const size_t scratch_bytes = (size_t)64 << 20;
    void* scratch = NULL;
    vv_dev_alloc(&scratch, scratch_bytes);

    for (int t = 0; t < 3; t++) {
        dev_w_t d[7];
        memset(d, 0, sizeof(d));
        int K_in[7];
        for (int i = 0; i < 7; i++) {
            K_in[i] = rm[t][i].K;
            if (!rm[t][i].w) continue;
            const vv_weight_t* w = rm[t][i].w;
            char tag[96];
            if (w->int4g_layout != VV_INT4G_GPU || w->mins.data ||
                w->zeros.data ||
                w->quant.scales.size_bytes !=
                    (size_t)rm[t][i].N * (rm[t][i].K / rm[t][i].G) * 4) {
                printf("  FAIL layer %d %s: GPU layout bookkeeping\n",
                       layers[t], names[i]);
                failures++;
                rm[t][i].w = NULL;
                continue;
            }
            if (upload(w, &d[i])) {
                printf("  FAIL layer %d %s: upload\n", layers[t], names[i]);
                failures++;
                continue;
            }
            for (int pass = 0; pass < 2; pass++) {
                const int M = pass ? MG : 1;
                const int K = rm[t][i].K, N = rm[t][i].N;
                float* x = (float*)malloc((size_t)M * K * sizeof(float));
                uint16_t* hx = (uint16_t*)malloc((size_t)M * K * 2);
                for (size_t e = 0; e < (size_t)M * K; e++) {
                    hx[e] = vv_float_to_half(frand());
                    x[e] = vv_half_to_float(hx[e]);
                }
                float* ref = (float*)malloc((size_t)M * N * sizeof(float));
                cpu_ref(&rm[t][i], x, M, ref);
                void *dx = NULL, *dxg = NULL, *dy = NULL;
                vv_dev_alloc(&dx, (size_t)M * K * 2);
                vv_dev_alloc(&dxg, (size_t)M * K * 2);
                vv_dev_alloc(&dy, (size_t)M * N * 2);
                vv_dev_memcpy_h2d(dx, hx, (size_t)M * K * 2, NULL);
                vv_status_t s = gpu_run(&rm[t][i], &d[i], dx, dxg, dy, M,
                                        scratch, scratch_bytes, st);
                vv_dev_stream_sync(st);
                snprintf(tag, sizeof(tag), "layer %d %s%s M=%d", layers[t],
                         names[i], w->perm.data ? " (act-order)" : "", M);
                if (s != VV_OK) {
                    printf("  FAIL %s: returned %d\n", tag, s);
                    failures++;
                } else {
                    uint16_t* hy = (uint16_t*)malloc((size_t)M * N * 2);
                    vv_dev_memcpy_d2h(hy, dy, (size_t)M * N * 2, NULL);
                    compare(tag, hy, ref, (size_t)M * N);
                    free(hy);
                }
                vv_dev_free(dx); vv_dev_free(dxg); vv_dev_free(dy);
                free(x); free(hx); free(ref);
            }
        }

        /* The fused launches the decoder uses for q/k/v and gate/up. */
        const int groups[2][4] = { { 0, 1, 2, 3 }, { 4, 5, 2, -1 } };
        for (int gi = 0; gi < 2; gi++) {
            const int n = gi == 0 ? 3 : 2;
            bool ok = true;
            for (int j = 0; j < n; j++) {
                const rowmajor_t* r = &rm[t][groups[gi][j]];
                ok = ok && r->w && !r->perm && r->G == rm[t][groups[gi][0]].G &&
                     K_in[groups[gi][j]] == K_in[groups[gi][0]];
            }
            if (!ok) continue;                /* act-order: not fused */
            const int K = K_in[groups[gi][0]];
            uint16_t* hx = (uint16_t*)malloc((size_t)K * 2);
            float* x = (float*)malloc((size_t)K * sizeof(float));
            for (int e = 0; e < K; e++) {
                hx[e] = vv_float_to_half(frand());
                x[e] = vv_half_to_float(hx[e]);
            }
            void* dx = NULL;
            vv_dev_alloc(&dx, (size_t)K * 2);
            vv_dev_memcpy_h2d(dx, hx, (size_t)K * 2, NULL);
            vv_w4a16_proj_t p[3];
            void* dy[3] = { NULL, NULL, NULL };
            for (int j = 0; j < n; j++) {
                const int i = groups[gi][j];
                vv_dev_alloc(&dy[j], (size_t)rm[t][i].N * 2);
                p[j].packed = d[i].p; p[j].sz = d[i].sz; p[j].bias = d[i].bias;
                p[j].y = dy[j]; p[j].N = rm[t][i].N;
            }
            vv_status_t s = vv_w4a16_gemv_multi_dev(dx, p, n, K,
                                                    rm[t][groups[gi][0]].G, st);
            vv_dev_stream_sync(st);
            for (int j = 0; j < n; j++) {
                const int i = groups[gi][j];
                char tag[96];
                snprintf(tag, sizeof(tag), "layer %d %s fused %d/%d",
                         layers[t], names[i], j + 1, n);
                if (s != VV_OK) {
                    printf("  FAIL %s: returned %d\n", tag, s);
                    failures++;
                    continue;
                }
                float* ref = (float*)malloc((size_t)rm[t][i].N * sizeof(float));
                uint16_t* hy = (uint16_t*)malloc((size_t)rm[t][i].N * 2);
                cpu_ref(&rm[t][i], x, 1, ref);
                vv_dev_memcpy_d2h(hy, dy[j], (size_t)rm[t][i].N * 2, NULL);
                compare(tag, hy, ref, (size_t)rm[t][i].N);
                free(ref); free(hy);
            }
            for (int j = 0; j < n; j++) vv_dev_free(dy[j]);
            vv_dev_free(dx);
            free(hx); free(x);
        }
        for (int i = 0; i < 7; i++) dev_free(&d[i]);
    }

    for (int t = 0; t < 3; t++)
        for (int i = 0; i < 7; i++) rowmajor_free(&rm[t][i]);
    vv_dev_free(scratch);
    vv_dev_stream_destroy(st);
    vv_model_free(model);

    if (failures) {
        printf("\n=== %d check(s) failed ===\n", failures);
        return 1;
    }
    printf("\n=== the loaded checkpoint runs the W4A16 kernels correctly ===\n");
    return 0;
#endif
}
