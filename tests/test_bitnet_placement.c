/**
 * @file test_bitnet_placement.c
 * @brief A BitNet run that placement sends to the CPU gives the --cpu run.
 *
 * With a GPU present the model is loaded for the GPU (F16 norms, the GPU's
 * encoder and head); a budget too small for anything then places it on the
 * CPU. That run must print exactly what a --cpu run prints. Needs
 * VV_BITNET_MODEL and a device; skips cleanly without either.
 */

#include "vibevoice/inference.h"
#include "vibevoice/device.h"
#include "vibevoice/types.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } \
                         else printf("  ok: %s\n", m); } while (0)

/* A few seconds of gliding tones: the text does not matter, only that
 * both runs agree on it. */
static float* make_clip(int n) {
    float* x = (float*)malloc(sizeof(float) * (size_t)n);
    double ph = 0.0;
    for (int i = 0; i < n; i++) {
        const double t = (double)i / 24000.0;
        ph += 2.0 * 3.14159265358979 * (180.0 + 60.0 * sin(3.0 * t)) / 24000.0;
        x[i] = (float)(0.2 * sin(ph) * (0.6 + 0.4 * sin(1.7 * t)));
    }
    return x;
}

static char* run(const char* dir, const vv_init_params_t* ip,
                 const float* x, int n, int* placement) {
    vv_inference_ctx_t* ctx = NULL;
    if (vv_inference_init(dir, 0, ip, &ctx) != VV_OK) return NULL;
    *placement = (int)ctx->placement;
    vv_inference_params_t p;
    memset(&p, 0, sizeof(p));
    p.max_new_tokens = 256;
    p.top_k = 1;
    vv_transcription_t* r = NULL;
    char* out = NULL;
    if (vv_inference_transcribe(ctx, x, n, &p, &r) == VV_OK && r) {
        const char* t = r->full_text ? r->full_text : "";
        out = (char*)malloc(strlen(t) + 1);
        strcpy(out, t);
        vv_transcription_free(r);
    }
    vv_inference_free(ctx);
    return out;
}

int main(void) {
    const char* dir = getenv("VV_BITNET_MODEL");
    if (!dir || !*dir) {
        printf("SKIP: VV_BITNET_MODEL not set\n");
        return 0;
    }
    if (vv_dev_device_count() <= 0) {
        printf("SKIP: no device\n");
        return 0;
    }
    const int n = 4 * 24000;
    float* x = make_clip(n);

    vv_init_params_t cpu = vv_init_params_default();
    cpu.cpu_only = true;
    cpu.max_seq_len = 2048;
    int pc = -1, pf = -1;
    char* a = run(dir, &cpu, x, n, &pc);

    vv_init_params_t tiny = vv_init_params_default();
    tiny.vram_budget = 0.0001f;   /* a GPU run with no room on the GPU */
    tiny.max_seq_len = 2048;
    char* b = run(dir, &tiny, x, n, &pf);

    CHECK(a && b, "both runs transcribe");
    CHECK(pf == VV_PLACE_CPU_ONLY, "the tiny budget places on the CPU");
    if (a && b) {
        printf("  --cpu:   \"%s\"\n  budget:  \"%s\"\n", a, b);
        CHECK(strcmp(a, b) == 0, "same transcript as --cpu");
    }
    free(a);
    free(b);
    free(x);
    if (failures) printf("FAILED (%d)\n", failures);
    else printf("PASSED\n");
    return failures ? 1 : 0;
}
