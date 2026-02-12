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
#include <string.h>
#include <stdlib.h>

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
    int pad;
    if (causal) {
        /* Causal: pad left by (kernel_size - 1) */
        pad = kernel_size - 1;
    } else {
        pad = (kernel_size - 1) / 2;
    }

    int padded_len = in_len + pad;
    if (causal) {
        /* Only left padding for causal */
        *out_len = (padded_len - kernel_size) / stride + 1;
    } else {
        int total_pad = kernel_size - 1;
        *out_len = (in_len + total_pad - kernel_size) / stride + 1;
    }

    int ch_per_group_in  = in_channels / groups;
    int ch_per_group_out = out_channels / groups;

    for (int oc = 0; oc < out_channels; oc++) {
        int g = oc / ch_per_group_out;
        for (int t = 0; t < *out_len; t++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < ch_per_group_in; ic++) {
                int abs_ic = g * ch_per_group_in + ic;
                for (int k = 0; k < kernel_size; k++) {
                    int in_t = t * stride + k - pad;
                    if (causal) {
                        in_t = t * stride + k - (kernel_size - 1);
                    }
                    if (in_t < 0 || in_t >= in_len) continue;

                    int w_idx = oc * (ch_per_group_in * kernel_size) +
                                ic * kernel_size + k;
                    int x_idx = abs_ic * in_len + in_t;
                    sum += input[x_idx] * weight[w_idx];
                }
            }
            output[oc * (*out_len) + t] = sum;
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

static void silu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

/* ─── Helper: Linear (as 1x1 conv) ─────────────────────────────────────── */

static void linear_1d(const float* input, int in_ch, int length,
                       const float* weight, const float* bias, int out_ch,
                       float* output) {
    /* weight: [out_ch, in_ch], input: [in_ch, length] */
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < length; t++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < in_ch; ic++) {
                sum += weight[oc * in_ch + ic] * input[ic * length + t];
            }
            output[oc * length + t] = sum;
        }
    }
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

    float* ffn_buf = (float*)vv_alloc(
        (size_t)ffn_hidden * (size_t)length * sizeof(float));
    if (!ffn_buf) return VV_ERR_OUT_OF_MEMORY;

    linear_1d(work1, channels, length,
              (const float*)block->ffn_linear1_weight.data,
              block->ffn_linear1_bias.data ?
                  (const float*)block->ffn_linear1_bias.data : NULL,
              ffn_hidden, ffn_buf);

    /* SiLU activation */
    silu_inplace(ffn_buf, ffn_hidden * length);

    /* FFN contract: linear2 (ffn_hidden → channels) */
    linear_1d(ffn_buf, ffn_hidden, length,
              (const float*)block->ffn_linear2_weight.data,
              block->ffn_linear2_bias.data ?
                  (const float*)block->ffn_linear2_bias.data : NULL,
              channels, work2);

    vv_free(ffn_buf);

    /* Residual + layer scale */
    residual_add(x, work2, channels * length,
                  (const float*)block->ffn_layer_scale.data,
                  channels, length);

    return VV_OK;
}

/* ─── Public API: CPU encode ────────────────────────────────────────────── */

vv_status_t vv_conv_vae_encode_cpu(const vv_conv_vae_encoder_t* encoder,
                                    const float* audio, int n_samples,
                                    float** output, int* n_frames) {
    if (!encoder || !audio || !output || !n_frames) return VV_ERR_NULL_PTR;
    if (n_samples <= 0) return VV_ERR_INVALID_ARG;

    VV_LOG_D("conv_vae: encoding %d samples (%.2f sec)",
             n_samples, (float)n_samples / 24000.0f);

    /* Input is [1, n_samples], treat as [1_channel, n_samples] */
    int cur_channels = 1;
    int cur_len = n_samples;

    /* Allocate working buffer (input copy as channel-first) */
    float* cur = (float*)vv_alloc((size_t)cur_len * sizeof(float));
    if (!cur) return VV_ERR_OUT_OF_MEMORY;
    memcpy(cur, audio, (size_t)cur_len * sizeof(float));

    /* Input convolution */
    if (encoder->input_conv.weight.data) {
        int out_ch = (int)encoder->input_conv.weight.shape[0];
        int out_len;
        float* conv_out = (float*)vv_alloc(
            (size_t)out_ch * (size_t)(cur_len + 16) * sizeof(float));
        if (!conv_out) { vv_free(cur); return VV_ERR_OUT_OF_MEMORY; }

        conv1d_forward(cur, cur_channels, cur_len,
                        (const float*)encoder->input_conv.weight.data,
                        out_ch, encoder->input_conv.kernel_size,
                        encoder->input_conv.bias.data ?
                            (const float*)encoder->input_conv.bias.data : NULL,
                        encoder->input_conv.stride, 1,
                        encoder->causal,
                        conv_out, &out_len);

        vv_free(cur);
        cur = conv_out;
        cur_channels = out_ch;
        cur_len = out_len;
    }

    /* Process each encoder stage */
    float eps = 1e-5f; /* RMSNorm epsilon for tokenizer */

    for (int s = 0; s < encoder->n_stages; s++) {
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
            vv_status_t st = encoder_block_forward(
                &stage->blocks[b], cur, cur_channels, cur_len,
                work1, work2, eps);
            if (st != VV_OK) {
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

vv_status_t vv_conv_vae_init(const vv_weight_t* model_weights, int n_weights,
                              const void* config,
                              bool is_acoustic,
                              vv_conv_vae_encoder_t** encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;

    vv_conv_vae_encoder_t* enc = (vv_conv_vae_encoder_t*)vv_alloc(
        sizeof(vv_conv_vae_encoder_t));
    if (!enc) return VV_ERR_OUT_OF_MEMORY;
    memset(enc, 0, sizeof(*enc));

    if (is_acoustic) {
        const vv_acoustic_tokenizer_config_t* ac =
            (const vv_acoustic_tokenizer_config_t*)config;
        enc->vae_dim = ac->vae_dim;
        enc->fix_std = ac->fix_std;
        enc->gaussian = (ac->fix_std > 0.0f);
        enc->causal = ac->causal;
        enc->n_stages = ac->n_ratios + 1; /* ratios stages + final */
    } else {
        const vv_semantic_tokenizer_config_t* sem =
            (const vv_semantic_tokenizer_config_t*)config;
        enc->vae_dim = sem->vae_dim;
        enc->fix_std = 0.0f;
        enc->gaussian = false;
        enc->causal = sem->causal;
        enc->n_stages = sem->n_ratios + 1;
    }

    /* Weight loading is done by the model loader, which populates the
     * encoder stage/block structures. This init just creates the skeleton.
     * Actual weight assignment is handled in vv_model_load. */

    *encoder = enc;
    VV_LOG_I("conv_vae: initialized %s encoder (vae_dim=%d, gaussian=%d)",
             is_acoustic ? "acoustic" : "semantic",
             enc->vae_dim, enc->gaussian);
    return VV_OK;
}

vv_status_t vv_conv_vae_free(vv_conv_vae_encoder_t* encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;

    if (encoder->stages) {
        for (int s = 0; s < encoder->n_stages; s++) {
            vv_encoder_stage_t* stage = &encoder->stages[s];
            if (stage->blocks) {
                for (int b = 0; b < stage->n_blocks; b++) {
                    vv_tensor_free(&stage->blocks[b].mixer_norm_weight);
                    vv_tensor_free(&stage->blocks[b].mixer_conv.weight);
                    vv_tensor_free(&stage->blocks[b].mixer_conv.bias);
                    vv_tensor_free(&stage->blocks[b].mixer_layer_scale);
                    vv_tensor_free(&stage->blocks[b].ffn_norm_weight);
                    vv_tensor_free(&stage->blocks[b].ffn_linear1_weight);
                    vv_tensor_free(&stage->blocks[b].ffn_linear1_bias);
                    vv_tensor_free(&stage->blocks[b].ffn_linear2_weight);
                    vv_tensor_free(&stage->blocks[b].ffn_linear2_bias);
                    vv_tensor_free(&stage->blocks[b].ffn_layer_scale);
                }
                vv_free(stage->blocks);
            }
            vv_tensor_free(&stage->downsample.weight);
            vv_tensor_free(&stage->downsample.bias);
        }
        vv_free(encoder->stages);
    }

    vv_tensor_free(&encoder->input_conv.weight);
    vv_tensor_free(&encoder->input_conv.bias);
    vv_tensor_free(&encoder->proj_mean.weight);
    vv_tensor_free(&encoder->proj_mean.bias);
    vv_tensor_free(&encoder->proj_logvar.weight);
    vv_tensor_free(&encoder->proj_logvar.bias);

    vv_free(encoder);
    return VV_OK;
}
