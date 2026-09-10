/**
 * @file conv_vae.c
 * @brief Conv-VAE speech tokenizer encoder — CPU reference implementation.
 *
 * Architecture (from VibeVoice config.json):
 *   - Input: raw 24kHz mono PCM [1, 1, N]
 *   - Input conv: Conv1d(1, 32, kernel=7, stride=1)
 *   - 7 stages, each with:
 *     - Downsample: Conv1d(in_ch, out_ch, kernel=2*ratio, stride=ratio)
 *     - D blocks, each containing:
 *       - norm → depthwise_conv → residual + layer_scale
 *       - norm → ffn(linear1→act→linear2) → residual + ffn_layer_scale
 *   - Final projection: Conv1d(ch, vae_dim*2, kernel=3) for acoustic (mean+logvar)
 *                        Conv1d(ch, vae_dim, kernel=3) for semantic (mean only)
 *   - Gaussian sample: mean + exp(logvar*0.5) * eps for acoustic (fix_std=0.5)
 *   - Deterministic: just mean for semantic
 *
 *   encoder_ratios: [8, 5, 5, 4, 2, 2] → total compression 3200
 *   encoder_depths: "3-3-3-3-3-3-8" → 7 stage groups
 *   encoder_n_filters: 32 (base, doubled each stage)
 *   causal: true
 *   layernorm: RMSNorm
 */

#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
static double vv_time_ms_enc(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart * 1000.0;
}
#else
#include <time.h>
static double vv_time_ms_enc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ─── CUDA function declarations ────────────────────────────────────────── */

extern vv_status_t vv_cuda_alloc(void** ptr, size_t size);
extern vv_status_t vv_cuda_free(void* ptr);
extern vv_status_t vv_cuda_memcpy_h2d(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_memcpy_d2h(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_stream_sync(void* stream);
extern vv_status_t vv_cuda_stream_create(void** stream);
extern vv_status_t vv_cuda_stream_destroy(void* stream);

extern vv_status_t vv_gemm_fp16_nn_cuda(
    const void* A, const void* B, void* C,
    int M, int K, int P,
    float alpha, float beta,
    void* stream);

extern vv_status_t vv_silu_cuda(void* data, int total, void* stream);
extern vv_status_t vv_gelu_cuda(void* data, int total, void* stream);
extern vv_status_t vv_cuda_memset(void* ptr, int value, size_t size);
extern vv_status_t vv_conv1d_raw_cuda(
    const void* input_fp16, const void* weight_fp16, const void* bias_fp16,
    void* output_fp16,
    int in_channels, int in_length, int out_channels, int kernel_size,
    int stride, int groups, int pad_left, int out_length, void* stream);
extern vv_status_t vv_channel_bias_add_cuda(void* output, const void* bias,
                                              int channels, int length,
                                              void* stream);
extern vv_status_t vv_conv1d_cuda(
    const void* input_fp16, const void* weight_fp16, const void* bias_fp16,
    void* output_fp16,
    int in_channels, int in_length, int out_channels, int kernel_size,
    int stride, int groups, bool causal,
    int* out_length, void* stream);
extern vv_status_t vv_rmsnorm_channel_first_cuda(
    const void* input, const void* weight, void* output,
    int channels, int length, float eps, void* stream);
extern vv_status_t vv_residual_add_scaled_cuda(
    void* x, const void* y, const void* scale,
    int channels, int length, void* stream);
extern vv_status_t vv_residual_add_cuda(void* x, const void* y, int total,
                                          void* stream);
extern vv_status_t vv_fp32_to_fp16_cuda(const void* in_fp32, void* out_fp16,
                                          int n, void* stream);
extern vv_status_t vv_fp16_to_fp32_cuda(const void* in_fp16, void* out_fp32,
                                          int n, void* stream);
extern vv_status_t vv_gather_tile_cuda(const void* src, void* dst,
                                        int channels, int full_len,
                                        int tile_offset, int tile_len, void* stream);
extern vv_status_t vv_scatter_tile_cuda(const void* src, void* dst,
                                         int channels, int full_len,
                                         int tile_offset, int tile_len, void* stream);

/* ─── FP32 ↔ FP16 conversion (CPU-side) ────────────────────────────────── */

static uint16_t fp32_to_fp16(float f) {
    /* IEEE-754 single → half conversion with rounding */
    union { float f; uint32_t u; } bits;
    bits.f = f;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t exponent = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits.u & 0x7FFFFF;

    if (exponent <= 0) {
        /* subnormal or zero */
        if (exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000;
        int shift = 14 - exponent;
        uint32_t round_bit = (mantissa >> (shift - 1)) & 1;
        mantissa >>= shift;
        mantissa += round_bit;
        return (uint16_t)(sign | mantissa);
    } else if (exponent >= 31) {
        /* overflow → infinity */
        return (uint16_t)(sign | 0x7C00);
    }
    /* round to nearest even */
    uint32_t round_bit = (mantissa >> 12) & 1;
    mantissa >>= 13;
    mantissa += round_bit;
    if (mantissa & 0x400) { mantissa = 0; exponent++; }
    if (exponent >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | mantissa);
}

static float fp16_to_fp32_val(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;

    if (exponent == 0) {
        if (mantissa == 0) { f = sign; }
        else {
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            f = sign | ((uint32_t)(127 - 15 + exponent) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f = sign | ((uint32_t)(exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    union { uint32_t u; float fv; } r;
    r.u = f;
    return r.fv;
}

/* Flag: whether GPU is available for FFN offload */
static int s_gpu_ffn_available = -1; /* -1 = untested, 0 = no, 1 = yes */

static int check_gpu_ffn(void) {
    if (s_gpu_ffn_available >= 0) return s_gpu_ffn_available;
    void* test_ptr = NULL;
    if (vv_cuda_alloc(&test_ptr, 256) == VV_OK) {
        vv_cuda_free(test_ptr);
        s_gpu_ffn_available = 1;
    } else {
        s_gpu_ffn_available = 0;
    }
    return s_gpu_ffn_available;
}

/* ─── GPU tensor helpers ───────────────────────────────────────────────── */

/** Upload a FP32 CPU tensor to GPU as FP16. Caller must vv_cuda_free the result. */
static vv_status_t upload_fp32_as_fp16(const float* cpu, size_t n_elements,
                                        void** out_gpu, void* stream) {
    if (!cpu || n_elements == 0) { *out_gpu = NULL; return VV_OK; }
    vv_status_t s;
    void* fp32_gpu = NULL;
    void* fp16_gpu = NULL;

    s = vv_cuda_alloc(&fp32_gpu, n_elements * sizeof(float));
    if (s != VV_OK) return s;
    s = vv_cuda_alloc(&fp16_gpu, n_elements * 2);
    if (s != VV_OK) { vv_cuda_free(fp32_gpu); return s; }

    vv_cuda_memcpy_h2d(fp32_gpu, cpu, n_elements * sizeof(float), stream);
    vv_fp32_to_fp16_cuda(fp32_gpu, fp16_gpu, (int)n_elements, stream);
    vv_cuda_stream_sync(stream);
    vv_cuda_free(fp32_gpu);

    *out_gpu = fp16_gpu;
    return VV_OK;
}

/* ─── GPU weight cache ─────────────────────────────────────────────────── */

#define UP(cpu_ptr, n, dst)                                                   \
    do {                                                                      \
        if ((cpu_ptr)) {                                                      \
            vv_status_t _s = upload_fp32_as_fp16((const float*)(cpu_ptr),     \
                                                 (size_t)(n), &(dst), stream);\
            if (_s != VV_OK) return _s;                                       \
        }                                                                     \
    } while (0)

/**
 * @brief Upload every encoder weight to the GPU as FP16, once per session.
 */
static vv_status_t ensure_gpu_weights(vv_conv_vae_encoder_t* enc, void* stream) {
    if (enc->gpu_weights_ready) return VV_OK;

    if (enc->input_conv.weight.data) {
        int out_ch = (int)enc->input_conv.weight.shape[0];
        size_t wn = (size_t)out_ch * 1 * enc->input_conv.kernel_size;
        UP(enc->input_conv.weight.data, wn, enc->input_w_gpu);
        UP(enc->input_conv.bias.data, out_ch, enc->input_b_gpu);
    }

    enc->gpu_stages = (vv_encoder_stage_gpu_t*)vv_alloc(
        sizeof(vv_encoder_stage_gpu_t) * (size_t)enc->n_stages);
    if (!enc->gpu_stages) return VV_ERR_OUT_OF_MEMORY;
    memset(enc->gpu_stages, 0,
           sizeof(vv_encoder_stage_gpu_t) * (size_t)enc->n_stages);

    for (int st = 0; st < enc->n_stages; st++) {
        vv_encoder_stage_t* stage = &enc->stages[st];
        vv_encoder_stage_gpu_t* g = &enc->gpu_stages[st];

        g->blocks = (vv_encoder_block_gpu_t*)vv_alloc(
            sizeof(vv_encoder_block_gpu_t) * (size_t)stage->n_blocks);
        if (!g->blocks) return VV_ERR_OUT_OF_MEMORY;
        memset(g->blocks, 0,
               sizeof(vv_encoder_block_gpu_t) * (size_t)stage->n_blocks);

        for (int b = 0; b < stage->n_blocks; b++) {
            vv_encoder_block_t* blk = &stage->blocks[b];
            vv_encoder_block_gpu_t* gb = &g->blocks[b];

            int ch = (blk->mixer_conv.weight.ndim >= 1)
                     ? (int)blk->mixer_conv.weight.shape[0] : 0;
            int hidden = (blk->ffn_linear1_weight.ndim >= 1)
                         ? (int)blk->ffn_linear1_weight.shape[0] : ch * 4;
            gb->channels   = ch;
            gb->ffn_hidden = hidden;

            UP(blk->mixer_norm_weight.data, ch, gb->norm_w);
            UP(blk->mixer_conv.weight.data,
               (size_t)ch * blk->mixer_conv.kernel_size, gb->conv_w);
            UP(blk->mixer_conv.bias.data, ch, gb->conv_b);
            UP(blk->mixer_layer_scale.data, ch, gb->gamma);
            UP(blk->ffn_norm_weight.data, ch, gb->ffn_norm_w);
            UP(blk->ffn_layer_scale.data, ch, gb->ffn_gamma);
            UP(blk->ffn_linear1_weight.data, (size_t)hidden * ch, gb->l1_w);
            UP(blk->ffn_linear1_bias.data, hidden, gb->l1_b);
            UP(blk->ffn_linear2_weight.data, (size_t)ch * hidden, gb->l2_w);
            UP(blk->ffn_linear2_bias.data, ch, gb->l2_b);
        }

        if (stage->downsample.weight.data) {
            int oc = (int)stage->downsample.weight.shape[0];
            int ic = (int)stage->downsample.weight.shape[1];
            UP(stage->downsample.weight.data,
               (size_t)oc * ic * stage->downsample.kernel_size, g->ds_w);
            UP(stage->downsample.bias.data, oc, g->ds_b);
        }
    }

    if (enc->proj_mean.weight.data) {
        int oc = (int)enc->proj_mean.weight.shape[0];
        int ic = (int)enc->proj_mean.weight.shape[1];
        UP(enc->proj_mean.weight.data,
           (size_t)oc * ic * enc->proj_mean.kernel_size, enc->proj_w_gpu);
        UP(enc->proj_mean.bias.data, oc, enc->proj_b_gpu);
    }

    enc->gpu_weights_ready = true;
    VV_LOG_I("conv_vae: encoder weights resident on GPU (vae_dim=%d)",
             enc->vae_dim);
    return VV_OK;
}

#undef UP

/** @brief Release the GPU weight mirrors. */
static void free_gpu_weights(vv_conv_vae_encoder_t* enc) {
    if (!enc->gpu_weights_ready) return;
    if (enc->input_w_gpu) vv_cuda_free(enc->input_w_gpu);
    if (enc->input_b_gpu) vv_cuda_free(enc->input_b_gpu);
    if (enc->proj_w_gpu)  vv_cuda_free(enc->proj_w_gpu);
    if (enc->proj_b_gpu)  vv_cuda_free(enc->proj_b_gpu);
    if (enc->input_cache) vv_cuda_free(enc->input_cache);
    if (enc->proj_cache)  vv_cuda_free(enc->proj_cache);
    if (enc->gpu_stages) {
        for (int st = 0; st < enc->n_stages; st++) {
            vv_encoder_stage_gpu_t* g = &enc->gpu_stages[st];
            if (g->ds_w) vv_cuda_free(g->ds_w);
            if (g->ds_b) vv_cuda_free(g->ds_b);
            if (g->ds_cache) vv_cuda_free(g->ds_cache);
            if (g->blocks) {
                for (int b = 0; b < enc->stages[st].n_blocks; b++) {
                    vv_encoder_block_gpu_t* gb = &g->blocks[b];
                    if (gb->norm_w) vv_cuda_free(gb->norm_w);
                    if (gb->conv_w) vv_cuda_free(gb->conv_w);
                    if (gb->conv_b) vv_cuda_free(gb->conv_b);
                    if (gb->gamma)  vv_cuda_free(gb->gamma);
                    if (gb->ffn_norm_w) vv_cuda_free(gb->ffn_norm_w);
                    if (gb->ffn_gamma)  vv_cuda_free(gb->ffn_gamma);
                    if (gb->l1_w) vv_cuda_free(gb->l1_w);
                    if (gb->l1_b) vv_cuda_free(gb->l1_b);
                    if (gb->l2_w) vv_cuda_free(gb->l2_w);
                    if (gb->l2_b) vv_cuda_free(gb->l2_b);
                    if (gb->cache) vv_cuda_free(gb->cache);
                }
                vv_free(g->blocks);
            }
        }
        vv_free(enc->gpu_stages);
        enc->gpu_stages = NULL;
    }
    enc->gpu_weights_ready = false;
}

/* ─── Streaming causal convolution ─────────────────────────────────────── */

#define FFN_TILE_GPU 131072
/** Segment length used by the reference encoder for long audio (60 s). */
#define VV_STREAM_SEGMENT_SAMPLES (60 * 24000)

/** @brief Reset every streaming conv cache before a fresh utterance. */
static void reset_stream_caches(vv_conv_vae_encoder_t* enc) {
    enc->input_cache_len = 0;
    enc->proj_cache_len = 0;
    if (!enc->gpu_stages) return;
    for (int st = 0; st < enc->n_stages; st++) {
        enc->gpu_stages[st].ds_cache_len = 0;
        if (!enc->gpu_stages[st].blocks) continue;
        for (int b = 0; b < enc->stages[st].n_blocks; b++)
            enc->gpu_stages[st].blocks[b].cache_len = 0;
    }
}

/**
 * @brief One causal SConv1d step over a chunk, with a streaming context.
 *
 * Mirrors SConv1d._forward_streaming: the previous chunk's tail (or zeros for
 * the first chunk) is prepended, the convolution runs with no left padding of
 * its own, and the final chunk gets the ceil-alignment zeros on the right.
 * Because the kernel reads out-of-range positions as zero, that right padding
 * is expressed purely as a longer output length.
 *
 * @param cache      In/out: device buffer holding the previous tail
 * @param cache_len  In/out: number of samples currently in @p cache
 * @param out        Out: freshly allocated [out_ch, *out_len] FP16 device buffer
 */
static vv_status_t sconv_chunk(
    void** cache, int* cache_len,
    const void* x, int in_ch, int in_len,
    const void* w, const void* b,
    int out_ch, int k, int stride, int groups,
    bool is_final,
    void** out, int* out_len, void* stream)
{
    const int ctx = (k - stride > 0) ? (k - stride) : 0;
    vv_status_t s;

    /* First chunk starts from zeros — same as the non-streaming left pad. */
    if (ctx > 0 && *cache_len == 0) {
        if (!*cache) {
            s = vv_cuda_alloc(cache, (size_t)in_ch * ctx * 2);
            if (s != VV_OK) return s;
        }
        vv_cuda_memset(*cache, 0, (size_t)in_ch * ctx * 2);
        *cache_len = ctx;
    }

    const int have  = (ctx > 0) ? *cache_len : 0;
    const int total = have + in_len;

    void* cat = NULL;
    if (have > 0) {
        s = vv_cuda_alloc(&cat, (size_t)in_ch * total * 2);
        if (s != VV_OK) return s;
        vv_scatter_tile_cuda(*cache, cat, in_ch, total, 0, have, stream);
        vv_scatter_tile_cuda(x, cat, in_ch, total, have, in_len, stream);
    }
    const void* src = have > 0 ? cat : x;

    int ol;
    if (is_final) {
        const int padded = ((total + stride - 1) / stride) * stride;
        ol = (padded - k) / stride + 1;
    } else {
        ol = (total - k) / stride + 1;
    }
    if (ol < 0) ol = 0;

    s = vv_cuda_alloc(out, (size_t)out_ch * (size_t)(ol > 0 ? ol : 1) * 2);
    if (s != VV_OK) { if (cat) vv_cuda_free(cat); return s; }

    if (ol > 0) {
        s = vv_conv1d_raw_cuda(src, w, b, *out, in_ch, total, out_ch,
                               k, stride, groups, 0, ol, stream);
        if (s != VV_OK) {
            vv_cuda_free(*out); *out = NULL;
            if (cat) vv_cuda_free(cat);
            return s;
        }
    }
    *out_len = ol;

    if (ctx > 0) {
        const int keep = total < ctx ? total : ctx;
        if (!*cache) {
            s = vv_cuda_alloc(cache, (size_t)in_ch * ctx * 2);
            if (s != VV_OK) { if (cat) vv_cuda_free(cat); return s; }
        }
        vv_gather_tile_cuda(src, *cache, in_ch, total, total - keep, keep, stream);
        *cache_len = keep;
    }

    if (cat) vv_cuda_free(cat);
    return VV_OK;
}

/* ─── Full GPU Conv-VAE encode ─────────────────────────────────────────── */

/**
 * @brief Encode one audio segment on the GPU, keeping streaming state.
 *
 * All data stays FP16 on the device; only the latent frames leave.
 */
static vv_status_t encode_gpu_chunk(
    vv_conv_vae_encoder_t* encoder, void* stream,
    const void* audio_gpu, int n_samples, bool is_final,
    void** out_gpu, int* out_frames)
{
    vv_status_t s;
    int cur_ch = 1, cur_len = n_samples;
    void* cur_gpu = NULL;

    /* Stem convolution */
    {
        const int out_ch = (int)encoder->input_conv.weight.shape[0];
        void* conv_out = NULL;
        int   conv_len = 0;
        s = sconv_chunk(&encoder->input_cache, &encoder->input_cache_len,
                        audio_gpu, cur_ch, cur_len,
                        encoder->input_w_gpu, encoder->input_b_gpu,
                        out_ch, encoder->input_conv.kernel_size,
                        encoder->input_conv.stride, 1, is_final,
                        &conv_out, &conv_len, stream);
        if (s != VV_OK) return s;
        cur_gpu = conv_out;
        cur_ch  = out_ch;
        cur_len = conv_len;
    }

    const float eps = 1e-5f;

    for (int st = 0; st < encoder->n_stages; st++) {
        const double stage_t0 = vv_time_ms_enc();
        vv_encoder_stage_t* stage = &encoder->stages[st];

        const size_t buf_elems = (size_t)cur_ch * (size_t)cur_len;
        const size_t buf_bytes = buf_elems * 2;
        void *work1 = NULL, *work2 = NULL;
        s = vv_cuda_alloc(&work1, buf_bytes);
        if (s != VV_OK) goto fail;
        s = vv_cuda_alloc(&work2, buf_bytes);
        if (s != VV_OK) { vv_cuda_free(work1); goto fail; }

        for (int b = 0; b < stage->n_blocks; b++) {
            vv_encoder_block_gpu_t* gb = &encoder->gpu_stages[st].blocks[b];
            vv_encoder_block_t*    blk = &stage->blocks[b];
            const int ffn_hidden = gb->ffn_hidden;

            /* ── Mixer: RMSNorm → depthwise conv → γ·residual ── */
            vv_rmsnorm_channel_first_cuda(cur_gpu, gb->norm_w, work1,
                                          cur_ch, cur_len, eps, stream);
            {
                void* mix = NULL;
                int   mix_len = 0;
                s = sconv_chunk(&gb->cache, &gb->cache_len,
                                work1, cur_ch, cur_len,
                                gb->conv_w, gb->conv_b,
                                cur_ch, blk->mixer_conv.kernel_size,
                                1, cur_ch, is_final,
                                &mix, &mix_len, stream);
                if (s != VV_OK) { vv_cuda_free(work1); vv_cuda_free(work2); goto fail; }
                if (gb->gamma)
                    vv_residual_add_scaled_cuda(cur_gpu, mix, gb->gamma,
                                                cur_ch, cur_len, stream);
                else
                    vv_residual_add_cuda(cur_gpu, mix, (int)buf_elems, stream);
                vv_cuda_free(mix);
            }

            /* ── FFN: RMSNorm → linear → GELU → linear → γ·residual ── */
            vv_rmsnorm_channel_first_cuda(cur_gpu, gb->ffn_norm_w, work1,
                                          cur_ch, cur_len, eps, stream);
            {
                void* ffn = NULL;
                s = vv_cuda_alloc(&ffn, (size_t)ffn_hidden * cur_len * 2);
                if (s == VV_OK) {
                    vv_gemm_fp16_nn_cuda(gb->l1_w, work1, ffn,
                                         ffn_hidden, cur_ch, cur_len,
                                         1.0f, 0.0f, stream);
                    if (gb->l1_b)
                        vv_channel_bias_add_cuda(ffn, gb->l1_b, ffn_hidden,
                                                 cur_len, stream);
                    vv_gelu_cuda(ffn, ffn_hidden * cur_len, stream);
                    vv_gemm_fp16_nn_cuda(gb->l2_w, ffn, work2,
                                         cur_ch, ffn_hidden, cur_len,
                                         1.0f, 0.0f, stream);
                    if (gb->l2_b)
                        vv_channel_bias_add_cuda(work2, gb->l2_b, cur_ch,
                                                 cur_len, stream);
                    vv_cuda_free(ffn);
                } else {
                    /* Not enough VRAM for the whole segment — tile over time. */
                    int tile = FFN_TILE_GPU;
                    if (tile > cur_len) tile = cur_len;
                    void *tin = NULL, *tff = NULL, *tout = NULL;
                    s = vv_cuda_alloc(&tin, (size_t)cur_ch * tile * 2);
                    if (s == VV_OK) s = vv_cuda_alloc(&tff, (size_t)ffn_hidden * tile * 2);
                    if (s == VV_OK) s = vv_cuda_alloc(&tout, (size_t)cur_ch * tile * 2);
                    if (s != VV_OK) {
                        if (tin) vv_cuda_free(tin);
                        if (tff) vv_cuda_free(tff);
                        vv_cuda_free(work1); vv_cuda_free(work2);
                        goto fail;
                    }
                    for (int off = 0; off < cur_len; off += tile) {
                        const int tl = (off + tile <= cur_len) ? tile : (cur_len - off);
                        vv_gather_tile_cuda(work1, tin, cur_ch, cur_len, off, tl, stream);
                        vv_gemm_fp16_nn_cuda(gb->l1_w, tin, tff,
                                             ffn_hidden, cur_ch, tl, 1.0f, 0.0f, stream);
                        if (gb->l1_b)
                            vv_channel_bias_add_cuda(tff, gb->l1_b, ffn_hidden, tl, stream);
                        vv_gelu_cuda(tff, ffn_hidden * tl, stream);
                        vv_gemm_fp16_nn_cuda(gb->l2_w, tff, tout,
                                             cur_ch, ffn_hidden, tl, 1.0f, 0.0f, stream);
                        if (gb->l2_b)
                            vv_channel_bias_add_cuda(tout, gb->l2_b, cur_ch, tl, stream);
                        vv_scatter_tile_cuda(tout, work2, cur_ch, cur_len, off, tl, stream);
                    }
                    vv_cuda_free(tin); vv_cuda_free(tff); vv_cuda_free(tout);
                }
                if (gb->ffn_gamma)
                    vv_residual_add_scaled_cuda(cur_gpu, work2, gb->ffn_gamma,
                                                cur_ch, cur_len, stream);
                else
                    vv_residual_add_cuda(cur_gpu, work2, (int)buf_elems, stream);
            }
        }
        vv_cuda_free(work1);
        vv_cuda_free(work2);

        /* Downsample into the next stage */
        if (stage->downsample.weight.data) {
            const int out_ch = (int)stage->downsample.weight.shape[0];
            void* ds_out = NULL;
            int   ds_len = 0;
            s = sconv_chunk(&encoder->gpu_stages[st].ds_cache,
                            &encoder->gpu_stages[st].ds_cache_len,
                            cur_gpu, cur_ch, cur_len,
                            encoder->gpu_stages[st].ds_w,
                            encoder->gpu_stages[st].ds_b,
                            out_ch, stage->downsample.kernel_size,
                            stage->downsample.stride, 1, is_final,
                            &ds_out, &ds_len, stream);
            if (s != VV_OK) goto fail;
            vv_cuda_free(cur_gpu);
            cur_gpu = ds_out;
            cur_ch  = out_ch;
            cur_len = ds_len;
        }

        vv_cuda_stream_sync(stream);
        VV_LOG_D("conv_vae_gpu: stage %d/%d (%d blocks, %d ch x %d len) %.0f ms",
                 st + 1, encoder->n_stages, stage->n_blocks,
                 cur_ch, cur_len, vv_time_ms_enc() - stage_t0);
    }

    /* Head projection to the VAE latent dimension */
    {
        void* proj = NULL;
        int   proj_len = 0;
        s = sconv_chunk(&encoder->proj_cache, &encoder->proj_cache_len,
                        cur_gpu, cur_ch, cur_len,
                        encoder->proj_w_gpu, encoder->proj_b_gpu,
                        encoder->vae_dim, encoder->proj_mean.kernel_size,
                        1, 1, is_final, &proj, &proj_len, stream);
        if (s != VV_OK) goto fail;
        vv_cuda_free(cur_gpu);
        *out_gpu = proj;
        *out_frames = proj_len;
    }
    return VV_OK;

fail:
    if (cur_gpu) vv_cuda_free(cur_gpu);
    return s;
}

/**
 * @brief Encode a full utterance, segmenting long audio like the reference.
 *
 * Audio longer than 60 s is processed in 60 s segments with the streaming
 * conv caches carried across them, exactly as
 * VibeVoiceASRForConditionalGeneration.encode_speech does. Chunking is what
 * keeps activation memory flat: a single-shot 10-minute encode would need
 * several GB just for the first stage.
 */
static vv_status_t vv_conv_vae_encode_gpu_full(
    vv_conv_vae_encoder_t* encoder,
    const float* audio_cpu, int n_samples,
    float** output_cpu, int* n_frames)
{
    vv_status_t s;
    void* stream = NULL;
    s = vv_cuda_stream_create(&stream);
    if (s != VV_OK) return s;

    s = ensure_gpu_weights(encoder, stream);
    if (s != VV_OK) { vv_cuda_stream_destroy(stream); return s; }
    reset_stream_caches(encoder);

    const double t0 = vv_time_ms_enc();
    const int vae_dim = encoder->vae_dim;
    const int seg = VV_STREAM_SEGMENT_SAMPLES;
    const int n_seg = (n_samples + seg - 1) / seg;

    /* ceil() over the whole signal is what the processor assumes */
    const int cap_frames = (n_samples + 3199) / 3200 + 8;
    float* result = (float*)vv_alloc((size_t)cap_frames * vae_dim * sizeof(float));
    if (!result) { vv_cuda_stream_destroy(stream); return VV_ERR_OUT_OF_MEMORY; }
    int total_frames = 0;

    void* audio_gpu = NULL;
    void* fp32_gpu = NULL;
    float* ch_first = NULL;

    for (int i = 0; i < n_seg; i++) {
        const int start = i * seg;
        const int len   = (start + seg <= n_samples) ? seg : (n_samples - start);
        const bool is_final = (i == n_seg - 1);

        s = vv_cuda_alloc(&fp32_gpu, (size_t)len * 4);
        if (s != VV_OK) goto cleanup;
        s = vv_cuda_alloc(&audio_gpu, (size_t)len * 2);
        if (s != VV_OK) goto cleanup;
        vv_cuda_memcpy_h2d(fp32_gpu, audio_cpu + start, (size_t)len * 4, stream);
        vv_fp32_to_fp16_cuda(fp32_gpu, audio_gpu, len, stream);
        vv_cuda_stream_sync(stream);
        vv_cuda_free(fp32_gpu);
        fp32_gpu = NULL;

        void* lat_gpu = NULL;
        int   lat_len = 0;
        s = encode_gpu_chunk(encoder, stream, audio_gpu, len, is_final,
                             &lat_gpu, &lat_len);
        vv_cuda_free(audio_gpu);
        audio_gpu = NULL;
        if (s != VV_OK) goto cleanup;

        /* Download [vae_dim, lat_len] and transpose into [frame, vae_dim] */
        const size_t n_elem = (size_t)vae_dim * (size_t)lat_len;
        s = vv_cuda_alloc(&fp32_gpu, n_elem * 4);
        if (s != VV_OK) { vv_cuda_free(lat_gpu); goto cleanup; }
        vv_fp16_to_fp32_cuda(lat_gpu, fp32_gpu, (int)n_elem, stream);
        vv_cuda_stream_sync(stream);
        vv_cuda_free(lat_gpu);

        ch_first = (float*)vv_alloc(n_elem * 4);
        if (!ch_first) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup; }
        vv_cuda_memcpy_d2h(ch_first, fp32_gpu, n_elem * 4, NULL);
        vv_cuda_free(fp32_gpu);
        fp32_gpu = NULL;

        if (total_frames + lat_len > cap_frames) lat_len = cap_frames - total_frames;
        for (int t = 0; t < lat_len; t++)
            for (int d = 0; d < vae_dim; d++)
                result[(size_t)(total_frames + t) * vae_dim + d] =
                    ch_first[(size_t)d * lat_len + t];
        total_frames += lat_len;
        vv_free(ch_first);
        ch_first = NULL;
    }

    *output_cpu = result;
    *n_frames = total_frames;
    result = NULL;

    VV_LOG_I("conv_vae_gpu: encoded %d samples -> %d frames in %d segment(s), "
             "vae_dim=%d, %.0f ms",
             n_samples, total_frames, n_seg, vae_dim, vv_time_ms_enc() - t0);
    s = VV_OK;

cleanup:
    if (fp32_gpu)  vv_cuda_free(fp32_gpu);
    if (audio_gpu) vv_cuda_free(audio_gpu);
    if (ch_first)  vv_free(ch_first);
    if (result)    vv_free(result);
    vv_cuda_stream_destroy(stream);
    return s;
}

/* ─── Helper: 1D causal convolution (CPU reference) ─────────────────────── */

/**
 * @brief 1D convolution with optional causal padding and stride.
 *
 * Input:  [in_channels, in_len]
 * Weight: [out_channels, in_channels/groups, kernel_size]
 * Bias:   [out_channels] or NULL
 * Output: [out_channels, out_len]
 */
static void conv1d_forward(
    const float* input,  int in_channels, int in_len,
    const float* weight, int out_channels, int kernel_size,
    const float* bias,
    int stride, int groups, bool causal,
    float* output, int* out_len)
{
    /* See vv_conv1d_cuda: causal pad is (k - stride), output is ceil(L/s). */
    int pad;
    if (causal) {
        pad = kernel_size - stride;
        if (pad < 0) pad = 0;
        *out_len = (in_len + stride - 1) / stride;
    } else {
        pad = (kernel_size - 1) / 2;
        *out_len = (in_len + (kernel_size - 1) - kernel_size) / stride + 1;
    }
    if (*out_len < 1) *out_len = 1;

    int oL = *out_len;
    int ch_per_group_in  = in_channels / groups;
    int ch_per_group_out = out_channels / groups;

    /* Depthwise fast path: groups == in_channels == out_channels, ch_per_group_in == 1 */
    if (groups == in_channels && groups == out_channels && ch_per_group_in == 1) {
        for (int c = 0; c < out_channels; c++) {
            float b = bias ? bias[c] : 0.0f;
            const float* w_ptr = weight + (size_t)c * kernel_size;
            const float* x_ptr = input  + (size_t)c * in_len;
            float*       o_ptr = output + (size_t)c * oL;

            for (int t = 0; t < oL; t++) {
                float sum = b;
                int base = t * stride - pad;
                for (int k = 0; k < kernel_size; k++) {
                    int in_t = base + k;
                    if (in_t >= 0 && in_t < in_len)
                        sum += x_ptr[in_t] * w_ptr[k];
                }
                o_ptr[t] = sum;
            }
        }
        return;
    }

    /* General convolution (optimized loop order) */
    for (int oc = 0; oc < out_channels; oc++) {
        int g = oc / ch_per_group_out;
        float b = bias ? bias[oc] : 0.0f;
        float* o_ptr = output + (size_t)oc * oL;

        /* Initialize output with bias */
        for (int t = 0; t < oL; t++) o_ptr[t] = b;

        /* Accumulate weighted input (ic, k, t) order for better cache */
        for (int ic = 0; ic < ch_per_group_in; ic++) {
            int abs_ic = g * ch_per_group_in + ic;
            const float* x_row = input + (size_t)abs_ic * in_len;
            for (int k = 0; k < kernel_size; k++) {
                float w_val = weight[oc * (ch_per_group_in * kernel_size)
                                     + ic * kernel_size + k];
                for (int t = 0; t < oL; t++) {
                    int in_t = t * stride + k - pad;
                    if (in_t >= 0 && in_t < in_len)
                        o_ptr[t] += w_val * x_row[in_t];
                }
            }
        }
    }
}

/* ─── Helper: RMSNorm ───────────────────────────────────────────────────── */

static void rmsnorm_1d(const float* input, const float* weight,
                        float* output, int channels, int length,
                        float eps) {
    for (int t = 0; t < length; t++) {
        /* Compute RMS over channels at time step t */
        float sum_sq = 0.0f;
        for (int c = 0; c < channels; c++) {
            float v = input[c * length + t];
            sum_sq += v * v;
        }
        float rms = sqrtf(sum_sq / (float)channels + eps);
        float inv_rms = 1.0f / rms;

        for (int c = 0; c < channels; c++) {
            output[c * length + t] = input[c * length + t] * inv_rms *
                                      weight[c];
        }
    }
}

/* ─── Helper: SiLU activation ───────────────────────────────────────────── */

/** @brief Exact GELU, matching ACT2FN["gelu"] used by the tokenizer FFN. */
static void gelu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 0.5f * data[i] * (1.0f + erff(data[i] * 0.7071067811865476f));
}

static void silu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

/* ─── Helper: Linear (as 1x1 conv) — cache-optimized ──────────────────── */

static void linear_1d(const float* input, int in_ch, int length,
                       const float* weight, const float* bias, int out_ch,
                       float* output) {
    /*
     * weight: [out_ch, in_ch], input: [in_ch, length], output: [out_ch, length]
     *
     * Loop order: (oc, ic, t)  ← inner loop over time for sequential memory access.
     * This allows the compiler to auto-vectorize the inner loop (SSE/AVX).
     */
    for (int oc = 0; oc < out_ch; oc++) {
        float b = bias ? bias[oc] : 0.0f;
        float* out_row = output + (size_t)oc * length;
        /* Initialize output row with bias */
        for (int t = 0; t < length; t++) {
            out_row[t] = b;
        }
        /* Accumulate weighted input rows */
        const float* w_row = weight + (size_t)oc * in_ch;
        for (int ic = 0; ic < in_ch; ic++) {
            float w = w_row[ic];
            const float* in_row = input + (size_t)ic * length;
            for (int t = 0; t < length; t++) {
                out_row[t] += w * in_row[t];
            }
        }
    }
}

/* ─── GPU-offloaded FFN forward (tiled) ─────────────────────────────────── */

/**
 * @brief Run FFN (linear1 → SiLU → linear2) on GPU in tiles.
 *
 * input:   [in_ch, length]  FP32 on CPU
 * w1:      [ffn_hidden, in_ch] FP32 on CPU
 * b1:      [ffn_hidden] FP32 on CPU  (or NULL)
 * w2:      [in_ch, ffn_hidden] FP32 on CPU
 * b2:      [in_ch] FP32 on CPU  (or NULL)
 * output:  [in_ch, length]  FP32 on CPU
 *
 * Processes along the time axis in tiles to limit GPU memory usage.
 * Returns VV_OK on success, or a fallback-triggering error code.
 */
#define FFN_TILE_SIZE 131072  /* timesteps per tile — keeps GPU buffers ~256 MB */

static vv_status_t ffn_forward_gpu(
    const float* input_cpu,  int in_ch, int length,
    const float* w1, const float* b1, int ffn_hidden,
    const float* w2, const float* b2,
    float* output_cpu)
{
    vv_status_t s;
    void* stream = NULL;

    /* Create a temporary CUDA stream */
    s = vv_cuda_stream_create(&stream);
    if (s != VV_OK) return s;

    /* --- Upload weights (once, persist across all tiles) --- */
    size_t w1_n = (size_t)ffn_hidden * in_ch;
    size_t w2_n = (size_t)in_ch * ffn_hidden;

    size_t w1_gpu_bytes = w1_n * 2;
    size_t w2_gpu_bytes = w2_n * 2;
    void *w1_gpu = NULL, *w2_gpu = NULL;
    void *b1_gpu = NULL, *b2_gpu = NULL;

    /* Convert FP32 weights → FP16 temp buffer, upload */
    uint16_t* w1_h = (uint16_t*)vv_alloc(w1_gpu_bytes);
    uint16_t* w2_h = (uint16_t*)vv_alloc(w2_gpu_bytes);
    if (!w1_h || !w2_h) {
        if (w1_h) vv_free(w1_h);
        if (w2_h) vv_free(w2_h);
        vv_cuda_stream_destroy(stream);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < w1_n; i++) w1_h[i] = fp32_to_fp16(w1[i]);
    for (size_t i = 0; i < w2_n; i++) w2_h[i] = fp32_to_fp16(w2[i]);

    s = vv_cuda_alloc(&w1_gpu, w1_gpu_bytes);
    if (s != VV_OK) goto fail_weights;
    s = vv_cuda_alloc(&w2_gpu, w2_gpu_bytes);
    if (s != VV_OK) goto fail_weights;

    vv_cuda_memcpy_h2d(w1_gpu, w1_h, w1_gpu_bytes, stream);
    vv_cuda_memcpy_h2d(w2_gpu, w2_h, w2_gpu_bytes, stream);
    vv_free(w1_h); w1_h = NULL;
    vv_free(w2_h); w2_h = NULL;

    /* Upload biases if present */
    if (b1) {
        size_t b1_bytes = (size_t)ffn_hidden * 2;
        uint16_t* b1_h = (uint16_t*)vv_alloc(b1_bytes);
        if (!b1_h) goto fail_weights;
        for (int i = 0; i < ffn_hidden; i++) b1_h[i] = fp32_to_fp16(b1[i]);
        s = vv_cuda_alloc(&b1_gpu, b1_bytes);
        if (s != VV_OK) { vv_free(b1_h); goto fail_weights; }
        vv_cuda_memcpy_h2d(b1_gpu, b1_h, b1_bytes, stream);
        vv_free(b1_h);
    }
    if (b2) {
        size_t b2_bytes = (size_t)in_ch * 2;
        uint16_t* b2_h = (uint16_t*)vv_alloc(b2_bytes);
        if (!b2_h) goto fail_weights;
        for (int i = 0; i < in_ch; i++) b2_h[i] = fp32_to_fp16(b2[i]);
        s = vv_cuda_alloc(&b2_gpu, b2_bytes);
        if (s != VV_OK) { vv_free(b2_h); goto fail_weights; }
        vv_cuda_memcpy_h2d(b2_gpu, b2_h, b2_bytes, stream);
        vv_free(b2_h);
    }

    /* --- Allocate tile buffers on GPU --- */
    int tile_len = length < FFN_TILE_SIZE ? length : FFN_TILE_SIZE;

    void *in_tile_gpu = NULL, *ffn_tile_gpu = NULL, *out_tile_gpu = NULL;
    size_t in_tile_bytes  = (size_t)in_ch    * tile_len * 2;
    size_t ffn_tile_bytes = (size_t)ffn_hidden * tile_len * 2;
    size_t out_tile_bytes = (size_t)in_ch    * tile_len * 2;

    s = vv_cuda_alloc(&in_tile_gpu,  in_tile_bytes);
    if (s != VV_OK) goto fail_weights;
    s = vv_cuda_alloc(&ffn_tile_gpu, ffn_tile_bytes);
    if (s != VV_OK) goto fail_tiles;
    s = vv_cuda_alloc(&out_tile_gpu, out_tile_bytes);
    if (s != VV_OK) goto fail_tiles;

    /* CPU staging buffers for FP32↔FP16 conversion */
    uint16_t* stage_in  = (uint16_t*)vv_alloc(in_tile_bytes);
    uint16_t* stage_out = (uint16_t*)vv_alloc(out_tile_bytes);
    if (!stage_in || !stage_out) {
        if (stage_in)  vv_free(stage_in);
        if (stage_out) vv_free(stage_out);
        s = VV_ERR_OUT_OF_MEMORY;
        goto fail_tiles;
    }

    /* --- Process tiles --- */
    for (int t_off = 0; t_off < length; t_off += tile_len) {
        int cur_tile = (t_off + tile_len <= length) ? tile_len : (length - t_off);
        size_t cur_in_bytes  = (size_t)in_ch     * cur_tile * 2;
        size_t cur_out_bytes = (size_t)in_ch      * cur_tile * 2;

        /* Convert input tile: [in_ch, cur_tile] FP32 → FP16 */
        for (int c = 0; c < in_ch; c++) {
            const float* src = input_cpu + (size_t)c * length + t_off;
            uint16_t*    dst = stage_in  + (size_t)c * cur_tile;
            for (int t = 0; t < cur_tile; t++) dst[t] = fp32_to_fp16(src[t]);
        }

        /* Upload input tile to GPU */
        vv_cuda_memcpy_h2d(in_tile_gpu, stage_in, cur_in_bytes, stream);

        /* GEMM1: ffn_tile[ffn_hidden, cur_tile] = w1[ffn_hidden, in_ch] @ in_tile[in_ch, cur_tile] */
        s = vv_gemm_fp16_nn_cuda(w1_gpu, in_tile_gpu, ffn_tile_gpu,
                                  ffn_hidden, in_ch, cur_tile,
                                  1.0f, 0.0f, stream);
        if (s != VV_OK) goto fail_process;

        /* Bias add (per-channel) */
        if (b1_gpu) {
            s = vv_channel_bias_add_cuda(ffn_tile_gpu, b1_gpu,
                                          ffn_hidden, cur_tile, stream);
            if (s != VV_OK) goto fail_process;
        }

        /* SiLU activation in-place */
        s = vv_gelu_cuda(ffn_tile_gpu, ffn_hidden * cur_tile, stream);
        if (s != VV_OK) goto fail_process;

        /* GEMM2: out_tile[in_ch, cur_tile] = w2[in_ch, ffn_hidden] @ ffn_tile[ffn_hidden, cur_tile] */
        s = vv_gemm_fp16_nn_cuda(w2_gpu, ffn_tile_gpu, out_tile_gpu,
                                  in_ch, ffn_hidden, cur_tile,
                                  1.0f, 0.0f, stream);
        if (s != VV_OK) goto fail_process;

        /* Bias add (per-channel) */
        if (b2_gpu) {
            s = vv_channel_bias_add_cuda(out_tile_gpu, b2_gpu,
                                          in_ch, cur_tile, stream);
            if (s != VV_OK) goto fail_process;
        }

        /* Download output tile */
        vv_cuda_stream_sync(stream);
        vv_cuda_memcpy_d2h(stage_out, out_tile_gpu, cur_out_bytes, NULL);

        /* Convert output tile: FP16 → FP32 into final output */
        for (int c = 0; c < in_ch; c++) {
            const uint16_t* src = stage_out + (size_t)c * cur_tile;
            float*          dst = output_cpu + (size_t)c * length + t_off;
            for (int t = 0; t < cur_tile; t++) dst[t] = fp16_to_fp32_val(src[t]);
        }
    }

    s = VV_OK;

fail_process:
    vv_free(stage_in);
    vv_free(stage_out);
fail_tiles:
    if (in_tile_gpu)  vv_cuda_free(in_tile_gpu);
    if (ffn_tile_gpu) vv_cuda_free(ffn_tile_gpu);
    if (out_tile_gpu) vv_cuda_free(out_tile_gpu);
fail_weights:
    if (w1_h) vv_free(w1_h);
    if (w2_h) vv_free(w2_h);
    if (w1_gpu) vv_cuda_free(w1_gpu);
    if (w2_gpu) vv_cuda_free(w2_gpu);
    if (b1_gpu) vv_cuda_free(b1_gpu);
    if (b2_gpu) vv_cuda_free(b2_gpu);
    vv_cuda_stream_destroy(stream);
    return s;
}

/* ─── Helper: element-wise ops ──────────────────────────────────────────── */

static void residual_add(float* residual, const float* delta,
                          int n, const float* layer_scale, int channels,
                          int length) {
    if (layer_scale) {
        /* layer_scale is per-channel [channels], broadcast over time */
        for (int c = 0; c < channels; c++) {
            float s = layer_scale[c];
            for (int t = 0; t < length; t++) {
                int idx = c * length + t;
                residual[idx] += delta[idx] * s;
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            residual[i] += delta[i];
        }
    }
}

/* ─── Encoder block forward ─────────────────────────────────────────────── */

static vv_status_t encoder_block_forward(
    const vv_encoder_block_t* block,
    float* x,              /* [channels, length], modified in-place */
    int channels, int length,
    float* work1, float* work2,
    float eps)
{
    /* Defensive NULL checks for critical weights */
    if (!block->mixer_norm_weight.data || !block->mixer_conv.weight.data ||
        !block->ffn_norm_weight.data || !block->ffn_linear1_weight.data ||
        !block->ffn_linear2_weight.data) {
        VV_LOG_E("conv_vae: block has NULL weight(s): norm=%p mixer=%p ffn_norm=%p l1=%p l2=%p",
                 block->mixer_norm_weight.data, block->mixer_conv.weight.data,
                 block->ffn_norm_weight.data, block->ffn_linear1_weight.data,
                 block->ffn_linear2_weight.data);
        return VV_ERR_WEIGHT_MISSING;
    }

    /* === Mixer path: norm → depthwise_conv → residual + layer_scale === */

    /* RMSNorm */
    rmsnorm_1d(x, (const float*)block->mixer_norm_weight.data,
               work1, channels, length, eps);

    /* Depthwise conv */
    int conv_out_len;
    conv1d_forward(work1, channels, length,
                    (const float*)block->mixer_conv.weight.data,
                    channels,
                    block->mixer_conv.kernel_size,
                    block->mixer_conv.bias.data ?
                        (const float*)block->mixer_conv.bias.data : NULL,
                    1, /* stride=1 for mixer */
                    channels, /* groups = channels (depthwise) */
                    block->mixer_conv.causal,
                    work2, &conv_out_len);

    /* Residual + layer scale */
    residual_add(x, work2, channels * length,
                  (const float*)block->mixer_layer_scale.data,
                  channels, length);

    /* === FFN path: norm → linear1 → SiLU → linear2 → residual + scale === */

    /* RMSNorm */
    rmsnorm_1d(x, (const float*)block->ffn_norm_weight.data,
               work1, channels, length, eps);

    /* FFN expand: linear1 (channels → channels*4 typically) */
    int ffn_hidden = channels * 4; /* typical expansion factor */
    if (block->ffn_linear1_weight.ndim >= 1) {
        ffn_hidden = (int)block->ffn_linear1_weight.shape[0];
    }

    /* Try GPU-offloaded FFN (linear1 → SiLU → linear2 fused on GPU) */
    int used_gpu = 0;
    if (check_gpu_ffn()) {
        vv_status_t gs = ffn_forward_gpu(
            work1, channels, length,
            (const float*)block->ffn_linear1_weight.data,
            block->ffn_linear1_bias.data ?
                (const float*)block->ffn_linear1_bias.data : NULL,
            ffn_hidden,
            (const float*)block->ffn_linear2_weight.data,
            block->ffn_linear2_bias.data ?
                (const float*)block->ffn_linear2_bias.data : NULL,
            work2);
        if (gs == VV_OK) used_gpu = 1;
    }

    if (!used_gpu) {
        /* CPU fallback: separate linear1 → gelu → linear2 */
        float* ffn_buf = (float*)vv_alloc(
            (size_t)ffn_hidden * (size_t)length * sizeof(float));
        if (!ffn_buf) return VV_ERR_OUT_OF_MEMORY;

        linear_1d(work1, channels, length,
                  (const float*)block->ffn_linear1_weight.data,
                  block->ffn_linear1_bias.data ?
                      (const float*)block->ffn_linear1_bias.data : NULL,
                  ffn_hidden, ffn_buf);

        /* SiLU activation */
        gelu_inplace(ffn_buf, ffn_hidden * length);

        /* FFN contract: linear2 (ffn_hidden → channels) */
        linear_1d(ffn_buf, ffn_hidden, length,
                  (const float*)block->ffn_linear2_weight.data,
                  block->ffn_linear2_bias.data ?
                      (const float*)block->ffn_linear2_bias.data : NULL,
                  channels, work2);

        vv_free(ffn_buf);
    }

    /* Residual + layer scale */
    residual_add(x, work2, channels * length,
                  (const float*)block->ffn_layer_scale.data,
                  channels, length);

    return VV_OK;
}

/* ─── Public API: CPU encode ────────────────────────────────────────────── */

vv_status_t vv_conv_vae_encode_cpu(vv_conv_vae_encoder_t* encoder,
                                    const float* audio, int n_samples,
                                    float** output, int* n_frames) {
    if (!encoder || !audio || !output || !n_frames) return VV_ERR_NULL_PTR;
    if (n_samples <= 0) return VV_ERR_INVALID_ARG;

    /* Try full-GPU path first */
    if (check_gpu_ffn()) {
        vv_status_t gs = vv_conv_vae_encode_gpu_full(encoder, audio, n_samples,
                                                      output, n_frames);
        if (gs == VV_OK) return VV_OK;
        VV_LOG_W("conv_vae: GPU encode failed (status=%d), falling back to CPU", gs);
    }

    VV_LOG_D("conv_vae: encoding %d samples (%.2f sec) [CPU]",
             n_samples, (float)n_samples / 24000.0f);

    /* Input is [1, n_samples], treat as [1_channel, n_samples] */
    int cur_channels = 1;
    int cur_len = n_samples;

    /* Allocate working buffer (input copy as channel-first) */
    float* cur = (float*)vv_alloc((size_t)cur_len * sizeof(float));
    if (!cur) return VV_ERR_OUT_OF_MEMORY;
    memcpy(cur, audio, (size_t)cur_len * sizeof(float));
    VV_LOG_D("conv_vae: cur buffer allocated (%d floats)", cur_len);

    /* Input convolution */
    if (encoder->input_conv.weight.data) {
        int out_ch = (encoder->input_conv.weight.ndim >= 1)
                     ? (int)encoder->input_conv.weight.shape[0] : 32;
        VV_LOG_D("conv_vae: input_conv: %d→%d ch, k=%d, s=%d, len=%d",
                 cur_channels, out_ch,
                 encoder->input_conv.kernel_size,
                 encoder->input_conv.stride, cur_len);

        int out_len;
        size_t conv_alloc = (size_t)out_ch * (size_t)(cur_len + 16) * sizeof(float);
        float* conv_out = (float*)vv_alloc(conv_alloc);
        if (!conv_out) { vv_free(cur); return VV_ERR_OUT_OF_MEMORY; }
        VV_LOG_D("conv_vae: conv_out allocated (%.1f MB)", (double)conv_alloc / (1024*1024));

        conv1d_forward(cur, cur_channels, cur_len,
                        (const float*)encoder->input_conv.weight.data,
                        out_ch, encoder->input_conv.kernel_size,
                        encoder->input_conv.bias.data ?
                            (const float*)encoder->input_conv.bias.data : NULL,
                        encoder->input_conv.stride, 1,
                        encoder->causal,
                        conv_out, &out_len);
        VV_LOG_D("conv_vae: input_conv done, out_len=%d", out_len);

        vv_free(cur);
        cur = conv_out;
        cur_channels = out_ch;
        cur_len = out_len;
    } else {
        VV_LOG_W("conv_vae: input_conv weight is NULL, skipping");
    }

    /* Process each encoder stage */
    float eps = 1e-5f; /* RMSNorm epsilon for tokenizer */

    VV_LOG_D("conv_vae: input_conv → %d ch, %d samples (gpu_ffn=%s)",
             cur_channels, cur_len, check_gpu_ffn() ? "yes" : "no");

    for (int s = 0; s < encoder->n_stages; s++) {
        double stage_t0 = vv_time_ms_enc();
        vv_encoder_stage_t* stage = &encoder->stages[s];

        /* Process blocks in this stage */
        size_t work_size = (size_t)cur_channels * (size_t)cur_len * sizeof(float);
        float* work1 = (float*)vv_alloc(work_size);
        float* work2 = (float*)vv_alloc(work_size);
        if (!work1 || !work2) {
            vv_free(cur);
            if (work1) vv_free(work1);
            if (work2) vv_free(work2);
            return VV_ERR_OUT_OF_MEMORY;
        }

        for (int b = 0; b < stage->n_blocks; b++) {
            VV_LOG_D("conv_vae: stage %d block %d/%d (%d ch × %d len) ...",
                     s + 1, b + 1, stage->n_blocks, cur_channels, cur_len);
            vv_status_t st = encoder_block_forward(
                &stage->blocks[b], cur, cur_channels, cur_len,
                work1, work2, eps);
            if (st != VV_OK) {
                VV_LOG_E("conv_vae: block forward failed (stage %d, block %d): %d",
                         s + 1, b + 1, (int)st);
                vv_free(work1); vv_free(work2); vv_free(cur);
                return st;
            }
        }
        vv_free(work1);
        vv_free(work2);

        /* Downsample convolution */
        if (stage->downsample.weight.data) {
            int out_ch = (int)stage->downsample.weight.shape[0];
            int new_len;
            float* ds_out = (float*)vv_alloc(
                (size_t)out_ch * ((size_t)cur_len / stage->downsample.stride + 2)
                * sizeof(float));
            if (!ds_out) { vv_free(cur); return VV_ERR_OUT_OF_MEMORY; }

            conv1d_forward(cur, cur_channels, cur_len,
                            (const float*)stage->downsample.weight.data,
                            out_ch, stage->downsample.kernel_size,
                            stage->downsample.bias.data ?
                                (const float*)stage->downsample.bias.data : NULL,
                            stage->downsample.stride,
                            1, /* groups = 1 */
                            encoder->causal,
                            ds_out, &new_len);

            vv_free(cur);
            cur = ds_out;
            cur_channels = out_ch;
            cur_len = new_len;
        }

        double stage_ms = vv_time_ms_enc() - stage_t0;
        VV_LOG_D("conv_vae: stage %d/%d done (%d blocks, %d ch × %d len) in %.1f ms",
                 s + 1, encoder->n_stages, stage->n_blocks,
                 cur_channels, cur_len, stage_ms);
    }

    /* Final projection to VAE latent space */
    int vae_dim = encoder->vae_dim;

    if (encoder->gaussian && encoder->proj_mean.weight.data &&
        encoder->proj_logvar.weight.data) {
        /* Acoustic: project to mean and logvar, then sample */
        float* mean = (float*)vv_alloc(
            (size_t)vae_dim * (size_t)cur_len * sizeof(float));
        float* logvar = (float*)vv_alloc(
            (size_t)vae_dim * (size_t)cur_len * sizeof(float));
        if (!mean || !logvar) {
            vv_free(cur);
            if (mean) vv_free(mean);
            if (logvar) vv_free(logvar);
            return VV_ERR_OUT_OF_MEMORY;
        }

        int mean_len, var_len;
        conv1d_forward(cur, cur_channels, cur_len,
                        (const float*)encoder->proj_mean.weight.data,
                        vae_dim, encoder->proj_mean.kernel_size,
                        encoder->proj_mean.bias.data ?
                            (const float*)encoder->proj_mean.bias.data : NULL,
                        1, 1, encoder->causal, mean, &mean_len);

        conv1d_forward(cur, cur_channels, cur_len,
                        (const float*)encoder->proj_logvar.weight.data,
                        vae_dim, encoder->proj_logvar.kernel_size,
                        encoder->proj_logvar.bias.data ?
                            (const float*)encoder->proj_logvar.bias.data : NULL,
                        1, 1, encoder->causal, logvar, &var_len);

        vv_free(cur);

        /* Gaussian sampling: z = mean + fix_std * exp(0.5 * logvar) * eps
         * For inference: just use mean + fix_std (deterministic approx) */
        int total = vae_dim * mean_len;
        float* z = (float*)vv_alloc((size_t)total * sizeof(float));
        if (!z) {
            vv_free(mean); vv_free(logvar);
            return VV_ERR_OUT_OF_MEMORY;
        }

        /* In inference mode, we use the mean directly (no random sampling) */
        memcpy(z, mean, (size_t)total * sizeof(float));

        vv_free(mean);
        vv_free(logvar);

        /* Transpose from [vae_dim, n_frames] to [n_frames, vae_dim] */
        float* result = (float*)vv_alloc(
            (size_t)total * sizeof(float));
        if (!result) { vv_free(z); return VV_ERR_OUT_OF_MEMORY; }

        for (int t = 0; t < mean_len; t++) {
            for (int d = 0; d < vae_dim; d++) {
                result[t * vae_dim + d] = z[d * mean_len + t];
            }
        }

        vv_free(z);
        *output = result;
        *n_frames = mean_len;
    }
    else if (encoder->proj_mean.weight.data) {
        /* Semantic: project to mean only, deterministic */
        float* mean = (float*)vv_alloc(
            (size_t)vae_dim * (size_t)cur_len * sizeof(float));
        if (!mean) { vv_free(cur); return VV_ERR_OUT_OF_MEMORY; }

        int mean_len;
        conv1d_forward(cur, cur_channels, cur_len,
                        (const float*)encoder->proj_mean.weight.data,
                        vae_dim, encoder->proj_mean.kernel_size,
                        encoder->proj_mean.bias.data ?
                            (const float*)encoder->proj_mean.bias.data : NULL,
                        1, 1, encoder->causal, mean, &mean_len);

        vv_free(cur);

        /* Transpose [vae_dim, n_frames] → [n_frames, vae_dim] */
        int total = vae_dim * mean_len;
        float* result = (float*)vv_alloc((size_t)total * sizeof(float));
        if (!result) { vv_free(mean); return VV_ERR_OUT_OF_MEMORY; }

        for (int t = 0; t < mean_len; t++) {
            for (int d = 0; d < vae_dim; d++) {
                result[t * vae_dim + d] = mean[d * mean_len + t];
            }
        }

        vv_free(mean);
        *output = result;
        *n_frames = mean_len;
    }
    else {
        vv_free(cur);
        VV_LOG_E("conv_vae: no projection weights loaded");
        return VV_ERR_WEIGHT_MISSING;
    }

    VV_LOG_I("conv_vae: encoded %d samples → %d frames (vae_dim=%d)",
             n_samples, *n_frames, vae_dim);
    return VV_OK;
}

/* ─── Init / Free ───────────────────────────────────────────────────────── */

/**
 * @brief Initialize a Conv-VAE encoder from a flat weight array.
 *
 * Parses weight names (e.g. "model.acoustic_tokenizer.encoder.stages.0.1.norm.weight")
 * and assigns them to the hierarchical encoder structure.
 *
 * Architecture (from safetensors):
 *   downsample_layers.0  = input conv (1→base_ch, kernel=7, stride=1)
 *   downsample_layers.{1..n_ratios} = per-stage strided downsample convs
 *   stages.{S}.{B}.*     = encoder blocks (norm+mixer+ffn per block)
 *   head.conv.conv.*      = final projection to vae_dim
 *
 * Tensor data is BORROWED — the model loader owns it.
 * vv_conv_vae_free only frees structural memory (stages/blocks arrays).
 */
vv_status_t vv_conv_vae_init(const vv_weight_t* model_weights, int n_weights,
                              const void* config,
                              bool is_acoustic,
                              vv_conv_vae_encoder_t** encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;

    vv_conv_vae_encoder_t* enc = (vv_conv_vae_encoder_t*)vv_alloc(
        sizeof(vv_conv_vae_encoder_t));
    if (!enc) return VV_ERR_OUT_OF_MEMORY;
    memset(enc, 0, sizeof(*enc));

    /* ── Parse config ── */
    int n_stages, n_ratios;
    int depths[8], ratios[8];

    if (is_acoustic) {
        const vv_acoustic_tokenizer_config_t* ac =
            (const vv_acoustic_tokenizer_config_t*)config;
        enc->vae_dim   = ac->vae_dim;
        enc->fix_std   = ac->fix_std;
        enc->gaussian  = (ac->fix_std > 0.0f);
        enc->causal    = ac->causal;
        n_ratios       = ac->n_ratios;          /* 6 */
        n_stages       = ac->n_depths;           /* 7 */
        memcpy(depths, ac->encoder_depths, sizeof(int) * n_stages);
        memcpy(ratios, ac->encoder_ratios, sizeof(int) * n_ratios);
    } else {
        const vv_semantic_tokenizer_config_t* sem =
            (const vv_semantic_tokenizer_config_t*)config;
        enc->vae_dim   = sem->vae_dim;
        enc->fix_std   = 0.0f;
        enc->gaussian  = false;
        enc->causal    = sem->causal;
        n_ratios       = sem->n_ratios;
        n_stages       = sem->n_depths;
        memcpy(depths, sem->encoder_depths, sizeof(int) * n_stages);
        memcpy(ratios, sem->encoder_ratios, sizeof(int) * n_ratios);
    }
    if (n_stages < 1 || n_stages > 8 || n_ratios < 1 || n_ratios > 8 ||
        enc->vae_dim < 1) {
        VV_LOG_E("conv_vae: invalid config (stages=%d, ratios=%d, vae_dim=%d)",
                 n_stages, n_ratios, enc->vae_dim);
        vv_free(enc);
        return VV_ERR_INVALID_ARG;
    }
    enc->n_stages = n_stages;

    /* ── Allocate stages and blocks ── */
    enc->stages = (vv_encoder_stage_t*)vv_alloc(
        sizeof(vv_encoder_stage_t) * n_stages);
    if (!enc->stages) { vv_free(enc); return VV_ERR_OUT_OF_MEMORY; }
    memset(enc->stages, 0, sizeof(vv_encoder_stage_t) * n_stages);

    for (int s = 0; s < n_stages; s++) {
        enc->stages[s].n_blocks = depths[s];
        enc->stages[s].blocks = (vv_encoder_block_t*)vv_alloc(
            sizeof(vv_encoder_block_t) * depths[s]);
        if (!enc->stages[s].blocks) {
            /* Cleanup on failure */
            for (int k = 0; k < s; k++) vv_free(enc->stages[k].blocks);
            vv_free(enc->stages); vv_free(enc);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memset(enc->stages[s].blocks, 0,
               sizeof(vv_encoder_block_t) * depths[s]);

        /* Pre-fill downsample metadata for stages that have downsampling.
         * Config ratios are listed large→small [8,5,5,4,2,2] but the
         * encoder processes them reversed (stage 0 uses ratio[n-1]=2). */
        if (s < n_ratios) {
            int ratio = ratios[n_ratios - 1 - s];
            enc->stages[s].downsample.stride      = ratio;
            enc->stages[s].downsample.kernel_size  = 2 * ratio;
            enc->stages[s].downsample.causal       = enc->causal;
        }
    }

    /* ── Determine prefix to strip from weight names ── */
    const char* prefix = is_acoustic
        ? "model.acoustic_tokenizer.encoder."
        : "model.semantic_tokenizer.encoder.";
    size_t prefix_len = strlen(prefix);

    int n_assigned = 0;

    /* ── Parse each weight name and assign to the encoder ── */
    for (int w = 0; w < n_weights; w++) {
        const char* name = model_weights[w].name;
        vv_tensor_t t    = model_weights[w].tensor;

        if (!t.data) continue;
        if (strncmp(name, prefix, prefix_len) != 0) continue;
        const char* sfx = name + prefix_len;   /* suffix after prefix */

        int idx;
        int s_idx, b_idx;

        /* ── downsample_layers.N.0.conv.conv.{weight|bias} ── */
        if (sscanf(sfx, "downsample_layers.%d.", &idx) == 1) {
            if (idx == 0) {
                /* Input conv (1 → base_ch) */
                if (strstr(sfx, ".weight")) {
                    enc->input_conv.weight      = t;
                    enc->input_conv.kernel_size  = t.ndim >= 3 ? (int)t.shape[2] : 7;
                    enc->input_conv.stride       = 1;
                    enc->input_conv.causal       = enc->causal;
                    n_assigned++;
                } else if (strstr(sfx, ".bias")) {
                    enc->input_conv.bias = t;
                    n_assigned++;
                }
            } else if (idx >= 1 && idx <= n_ratios) {
                int stage = idx - 1;
                if (strstr(sfx, ".weight")) {
                    enc->stages[stage].downsample.weight = t;
                    if (t.ndim >= 3)
                        enc->stages[stage].downsample.kernel_size = (int)t.shape[2];
                    n_assigned++;
                } else if (strstr(sfx, ".bias")) {
                    enc->stages[stage].downsample.bias = t;
                    n_assigned++;
                }
            }
        }
        /* ── stages.S.B.{component} ── */
        else if (sscanf(sfx, "stages.%d.%d.", &s_idx, &b_idx) == 2) {
            if (s_idx < 0 || s_idx >= n_stages) continue;
            if (b_idx < 0 || b_idx >= enc->stages[s_idx].n_blocks) continue;

            vv_encoder_block_t* blk = &enc->stages[s_idx].blocks[b_idx];

            /* Compute pointer to the part after "stages.S.B." */
            char pat[32];
            int plen = snprintf(pat, sizeof(pat), "stages.%d.%d.", s_idx, b_idx);
            const char* rest = sfx + plen;

            if (strcmp(rest, "norm.weight") == 0) {
                blk->mixer_norm_weight = t;
            } else if (strcmp(rest, "mixer.conv.conv.conv.weight") == 0) {
                blk->mixer_conv.weight      = t;
                blk->mixer_conv.kernel_size  = t.ndim >= 3 ? (int)t.shape[2] : 7;
                blk->mixer_conv.stride       = 1;
                blk->mixer_conv.causal       = enc->causal;
                blk->mixer_conv.depthwise    = true;
            } else if (strcmp(rest, "mixer.conv.conv.conv.bias") == 0) {
                blk->mixer_conv.bias = t;
            } else if (strcmp(rest, "gamma") == 0) {
                blk->mixer_layer_scale = t;
            } else if (strcmp(rest, "ffn_norm.weight") == 0) {
                blk->ffn_norm_weight = t;
            } else if (strcmp(rest, "ffn.linear1.weight") == 0) {
                blk->ffn_linear1_weight = t;
            } else if (strcmp(rest, "ffn.linear1.bias") == 0) {
                blk->ffn_linear1_bias = t;
            } else if (strcmp(rest, "ffn.linear2.weight") == 0) {
                blk->ffn_linear2_weight = t;
            } else if (strcmp(rest, "ffn.linear2.bias") == 0) {
                blk->ffn_linear2_bias = t;
            } else if (strcmp(rest, "ffn_gamma") == 0) {
                blk->ffn_layer_scale = t;
            } else {
                continue;   /* Unknown component */
            }
            n_assigned++;
        }
        /* ── head.conv.conv.{weight|bias} — final projection ── */
        else if (strstr(sfx, "head.conv.conv.")) {
            if (strstr(sfx, ".weight")) {
                enc->proj_mean.weight      = t;
                enc->proj_mean.kernel_size  = t.ndim >= 3 ? (int)t.shape[2] : 7;
                enc->proj_mean.stride       = 1;
                enc->proj_mean.causal       = enc->causal;
                n_assigned++;
            } else if (strstr(sfx, ".bias")) {
                enc->proj_mean.bias = t;
                n_assigned++;
            }
        }
    }

    *encoder = enc;
    VV_LOG_I("conv_vae: initialized %s encoder (vae_dim=%d, gaussian=%d, "
             "%d stages, %d/%d weights assigned)",
             is_acoustic ? "acoustic" : "semantic",
             enc->vae_dim, enc->gaussian,
             n_stages, n_assigned, n_weights);
    return VV_OK;
}

/**
 * @brief Free Conv-VAE encoder structural memory.
 *
 * Tensor data is owned by the model loader (vv_model_free handles it),
 * so we only free the stages/blocks arrays and the encoder struct itself.
 */
vv_status_t vv_conv_vae_warmup(vv_conv_vae_encoder_t* encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;
    if (!check_gpu_ffn()) return VV_OK;

    void* stream = NULL;
    vv_status_t s = vv_cuda_stream_create(&stream);
    if (s != VV_OK) return s;
    s = ensure_gpu_weights(encoder, stream);
    vv_cuda_stream_sync(stream);
    vv_cuda_stream_destroy(stream);
    return s;
}

vv_status_t vv_conv_vae_free(vv_conv_vae_encoder_t* encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;

    free_gpu_weights(encoder);

    if (encoder->stages) {
        for (int s = 0; s < encoder->n_stages; s++) {
            if (encoder->stages[s].blocks)
                vv_free(encoder->stages[s].blocks);
        }
        vv_free(encoder->stages);
    }

    vv_free(encoder);
    return VV_OK;
}
