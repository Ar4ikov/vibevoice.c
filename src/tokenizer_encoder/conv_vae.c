/**
 * @file conv_vae.c
 * @brief Conv-VAE speech tokenizer encoder — structure and weight binding.
 *
 * Architecture (from VibeVoice config.json):
 *   - Input: raw 24kHz mono PCM [1, 1, N]
 *   - Stem: Conv1d(1, 32, kernel=7, stride=1)
 *   - 7 stages, each: D blocks, then (except the last) a downsample conv
 *       block = norm → depthwise_conv → residual + gamma
 *               norm → ffn(linear1 → GELU → linear2) → residual + ffn_gamma
 *       downsample = Conv1d(C, 2C, kernel=2*ratio, stride=ratio)
 *   - Head: Conv1d(2048, vae_dim, kernel=7)
 *
 *   encoder_ratios: [8, 5, 5, 4, 2, 2] → total compression 3200
 *   encoder_depths: "3-3-3-3-3-3-8"
 *   causal: true, layernorm: RMSNorm (eps = layernorm_eps)
 *
 * The forward passes live elsewhere: vae_gpu.c (batched, streaming, CUDA)
 * and vae_cpu.c (time-tiled, OpenMP). Both walk the layer plan built here.
 */

#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/vibevoice.h"
#include "vae_plan.h"

#include <stdio.h>
#include <string.h>

/* ─── Layer plan ────────────────────────────────────────────────────────── */

static int tensor_dim(const vv_tensor_t* t, int i) {
    return (t->data && t->ndim > i) ? (int)t->shape[i] : 0;
}

static vv_status_t plan_add(vae_plan_t* p, int kind, int stage, int block,
                            int in_ch, int out_ch, int k, int stride) {
    if (p->n_layers >= VAE_MAX_LAYERS) return VV_ERR_OVERFLOW;
    if (in_ch <= 0 || out_ch <= 0 || k <= 0 || stride <= 0 || stride > k)
        return VV_ERR_SHAPE_MISMATCH;
    vae_layer_t* L = &p->L[p->n_layers++];
    L->kind = kind;
    L->stage = stage;
    L->block = block;
    L->in_ch = in_ch;
    L->out_ch = out_ch;
    L->k = k;
    L->stride = stride;
    L->cap = k - 1;
    L->tail_off = p->tail_elems;
    p->tail_elems += (size_t)in_ch * (size_t)(k > 1 ? k - 1 : 0);
    if (L->cap > p->max_cap) p->max_cap = L->cap;
    return VV_OK;
}

vv_status_t vae_plan_build(const vv_conv_vae_encoder_t* enc, vae_plan_t* p) {
    if (!enc || !p) return VV_ERR_NULL_PTR;
    memset(p, 0, sizeof(*p));
    if (enc->n_stages < 1 || enc->n_stages > VAE_MAX_STAGES)
        return VV_ERR_INVALID_ARG;
    p->n_stages = enc->n_stages;
    p->vae_dim = enc->vae_dim;

    vv_status_t s;
    int ch = tensor_dim(&enc->input_conv.weight, 0);
    s = plan_add(p, VAE_L_STEM, -1, -1, 1, ch,
                 enc->input_conv.kernel_size, 1);
    if (s != VV_OK) return s;

    int64_t div = 1;
    for (int st = 0; st < enc->n_stages; st++) {
        const vv_encoder_stage_t* stage = &enc->stages[st];
        p->C[st] = ch;
        p->div[st] = div;
        for (int b = 0; b < stage->n_blocks; b++) {
            const vv_encoder_block_t* blk = &stage->blocks[b];
            if (tensor_dim(&blk->mixer_conv.weight, 0) != ch)
                return VV_ERR_SHAPE_MISMATCH;
            const int hidden = tensor_dim(&blk->ffn_linear1_weight, 0);
            if (hidden > p->hidden[st]) p->hidden[st] = hidden;
            s = plan_add(p, VAE_L_MIXER, st, b, ch, ch,
                         blk->mixer_conv.kernel_size, 1);
            if (s != VV_OK) return s;
        }
        if (stage->downsample.weight.data) {
            const int oc = tensor_dim(&stage->downsample.weight, 0);
            if (tensor_dim(&stage->downsample.weight, 1) != ch)
                return VV_ERR_SHAPE_MISMATCH;
            s = plan_add(p, VAE_L_DS, st, -1, ch, oc,
                         stage->downsample.kernel_size,
                         stage->downsample.stride);
            if (s != VV_OK) return s;
            div *= stage->downsample.stride;
            ch = oc;
        }
    }
    if (tensor_dim(&enc->proj_mean.weight, 1) != ch)
        return VV_ERR_SHAPE_MISMATCH;
    return plan_add(p, VAE_L_HEAD, -1, -1, ch, enc->vae_dim,
                    enc->proj_mean.kernel_size, 1);
}

int vv_conv_vae_frames(const vv_conv_vae_encoder_t* enc, int64_t n_samples) {
    vae_plan_t p;
    if (vae_plan_build(enc, &p) != VV_OK || n_samples < 0) return 0;
    int64_t len = n_samples;
    for (int i = 0; i < p.n_layers; i++) {
        int64_t ol, keep;
        vae_layer_step(&p.L[i], vae_layer_fresh_have(&p.L[i]), len, true,
                       &ol, &keep);
        len = ol;
    }
    return (int)len;
}

bool vv_conv_vae_complete(const vv_conv_vae_encoder_t* enc) {
    if (!enc || !enc->stages || !enc->input_conv.weight.data ||
        !enc->proj_mean.weight.data)
        return false;
    for (int s = 0; s < enc->n_stages; s++) {
        const vv_encoder_stage_t* st = &enc->stages[s];
        if (s + 1 < enc->n_stages && !st->downsample.weight.data) return false;
        for (int b = 0; b < st->n_blocks; b++) {
            const vv_encoder_block_t* blk = &st->blocks[b];
            if (!blk->mixer_norm_weight.data || !blk->mixer_conv.weight.data ||
                !blk->ffn_norm_weight.data || !blk->ffn_linear1_weight.data ||
                !blk->ffn_linear2_weight.data)
                return false;
        }
    }
    vae_plan_t p;
    return vae_plan_build(enc, &p) == VV_OK;
}

/* ─── Init / Free ───────────────────────────────────────────────────────── */

/**
 * @brief Initialize a Conv-VAE encoder from a flat weight array.
 *
 * Weight names, after the "model.{acoustic,semantic}_tokenizer.encoder."
 * prefix:
 *   downsample_layers.0.0.conv.conv.{weight,bias}   stem (1 → base_ch)
 *   downsample_layers.N.0.conv.conv.{weight,bias}   downsample before stage N
 *   stages.S.B.{norm,mixer.conv.conv.conv,gamma,ffn_norm,ffn.linear1,
 *               ffn.linear2,ffn_gamma}              encoder blocks
 *   head.conv.conv.{weight,bias}                    projection to vae_dim
 *
 * Names are matched exactly: a loose ".weight" match would also bind a
 * quantized checkpoint's ".weight_scale".
 *
 * Tensor data is BORROWED — the model loader owns it.
 */
vv_status_t vv_conv_vae_init(const vv_weight_t* model_weights, int n_weights,
                              const void* config,
                              bool is_acoustic,
                              vv_conv_vae_encoder_t** encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;
    *encoder = NULL;

    vv_conv_vae_encoder_t* enc = (vv_conv_vae_encoder_t*)vv_alloc(
        sizeof(vv_conv_vae_encoder_t));
    if (!enc) return VV_ERR_OUT_OF_MEMORY;
    memset(enc, 0, sizeof(*enc));

    int n_stages, n_ratios;
    int depths[8], ratios[8];
    float eps;

    if (is_acoustic) {
        const vv_acoustic_tokenizer_config_t* ac =
            (const vv_acoustic_tokenizer_config_t*)config;
        enc->vae_dim   = ac->vae_dim;
        enc->fix_std   = ac->fix_std;
        enc->gaussian  = (ac->fix_std > 0.0f);
        enc->causal    = ac->causal;
        n_ratios       = ac->n_ratios;
        n_stages       = ac->n_depths;
        eps            = ac->layernorm_eps;
        if (n_stages >= 1 && n_stages <= 8)
            memcpy(depths, ac->encoder_depths, sizeof(int) * n_stages);
        if (n_ratios >= 1 && n_ratios <= 8)
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
        eps            = sem->layernorm_eps;
        if (n_stages >= 1 && n_stages <= 8)
            memcpy(depths, sem->encoder_depths, sizeof(int) * n_stages);
        if (n_ratios >= 1 && n_ratios <= 8)
            memcpy(ratios, sem->encoder_ratios, sizeof(int) * n_ratios);
    }
    if (n_stages < 1 || n_stages > 8 || n_ratios < 1 || n_ratios > 8 ||
        enc->vae_dim < 1) {
        VV_LOG_E("conv_vae: invalid config (stages=%d, ratios=%d, vae_dim=%d)",
                 n_stages, n_ratios, enc->vae_dim);
        vv_free(enc);
        return VV_ERR_INVALID_ARG;
    }
    /* config.json has carried 1e-5 for every released checkpoint; a config
       without the key must not turn into eps = 0. */
    enc->eps = eps > 0.0f ? eps : 1e-5f;
    enc->n_stages = n_stages;

    enc->stages = (vv_encoder_stage_t*)vv_alloc(
        sizeof(vv_encoder_stage_t) * n_stages);
    if (!enc->stages) { vv_free(enc); return VV_ERR_OUT_OF_MEMORY; }
    memset(enc->stages, 0, sizeof(vv_encoder_stage_t) * n_stages);

    for (int s = 0; s < n_stages; s++) {
        enc->stages[s].n_blocks = depths[s];
        enc->stages[s].blocks = (vv_encoder_block_t*)vv_alloc(
            sizeof(vv_encoder_block_t) * (depths[s] > 0 ? depths[s] : 1));
        if (!enc->stages[s].blocks) {
            for (int k = 0; k < s; k++) vv_free(enc->stages[k].blocks);
            vv_free(enc->stages); vv_free(enc);
            return VV_ERR_OUT_OF_MEMORY;
        }
        memset(enc->stages[s].blocks, 0,
               sizeof(vv_encoder_block_t) * (depths[s] > 0 ? depths[s] : 1));

        /* Config ratios are listed large→small [8,5,5,4,2,2] but the
         * encoder applies them reversed (stage 0 uses ratio[n-1]=2). */
        if (s < n_ratios) {
            int ratio = ratios[n_ratios - 1 - s];
            enc->stages[s].downsample.stride      = ratio;
            enc->stages[s].downsample.kernel_size = 2 * ratio;
            enc->stages[s].downsample.causal      = enc->causal;
        }
    }

    const char* prefix = is_acoustic
        ? "model.acoustic_tokenizer.encoder."
        : "model.semantic_tokenizer.encoder.";
    size_t prefix_len = strlen(prefix);

    int n_assigned = 0;

    for (int w = 0; w < n_weights; w++) {
        const char* name = model_weights[w].name;
        vv_tensor_t t    = model_weights[w].tensor;

        if (!t.data) continue;
        if (strncmp(name, prefix, prefix_len) != 0) continue;
        const char* sfx = name + prefix_len;

        int idx, s_idx, b_idx, used = 0;
        char tail[64];

        if (sscanf(sfx, "downsample_layers.%d.0.conv.conv.%63s", &idx, tail) == 2) {
            const bool is_w = strcmp(tail, "weight") == 0;
            const bool is_b = strcmp(tail, "bias") == 0;
            if (idx == 0) {
                if (is_w) {
                    enc->input_conv.weight      = t;
                    enc->input_conv.kernel_size = t.ndim >= 3 ? (int)t.shape[2] : 7;
                    enc->input_conv.stride      = 1;
                    enc->input_conv.causal      = enc->causal;
                    used = 1;
                } else if (is_b) {
                    enc->input_conv.bias = t;
                    used = 1;
                }
            } else if (idx >= 1 && idx <= n_ratios && idx - 1 < n_stages) {
                vv_conv1d_weights_t* ds = &enc->stages[idx - 1].downsample;
                if (is_w) {
                    ds->weight = t;
                    if (t.ndim >= 3) ds->kernel_size = (int)t.shape[2];
                    used = 1;
                } else if (is_b) {
                    ds->bias = t;
                    used = 1;
                }
            }
        } else if (sscanf(sfx, "stages.%d.%d.", &s_idx, &b_idx) == 2) {
            if (s_idx < 0 || s_idx >= n_stages) continue;
            if (b_idx < 0 || b_idx >= enc->stages[s_idx].n_blocks) continue;

            vv_encoder_block_t* blk = &enc->stages[s_idx].blocks[b_idx];
            char pat[32];
            int plen = snprintf(pat, sizeof(pat), "stages.%d.%d.", s_idx, b_idx);
            const char* rest = sfx + plen;
            used = 1;

            if (strcmp(rest, "norm.weight") == 0) {
                blk->mixer_norm_weight = t;
            } else if (strcmp(rest, "mixer.conv.conv.conv.weight") == 0) {
                blk->mixer_conv.weight      = t;
                blk->mixer_conv.kernel_size = t.ndim >= 3 ? (int)t.shape[2] : 7;
                blk->mixer_conv.stride      = 1;
                blk->mixer_conv.causal      = enc->causal;
                blk->mixer_conv.depthwise   = true;
                enc->stages[s_idx].channels = (int)t.shape[0];
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
                used = 0;
            }
        } else if (strcmp(sfx, "head.conv.conv.weight") == 0) {
            enc->proj_mean.weight      = t;
            enc->proj_mean.kernel_size = t.ndim >= 3 ? (int)t.shape[2] : 7;
            enc->proj_mean.stride      = 1;
            enc->proj_mean.causal      = enc->causal;
            used = 1;
        } else if (strcmp(sfx, "head.conv.conv.bias") == 0) {
            enc->proj_mean.bias = t;
            used = 1;
        }
        n_assigned += used;
    }

    /*
     * The CUDA and CPU forward passes are causal-only. A non-causal
     * tokenizer (centre padding) would run without error and produce
     * shifted latents, so refuse it here instead.
     */
    if (!enc->causal && n_weights > 0) {
        VV_LOG_E("conv_vae: non-causal tokenizers are not supported");
        vv_conv_vae_free(enc);
        return VV_ERR_UNSUPPORTED;
    }

    *encoder = enc;
    VV_LOG_I("conv_vae: initialized %s encoder (vae_dim=%d, gaussian=%d, "
             "%d stages, %d/%d weights assigned)",
             is_acoustic ? "acoustic" : "semantic",
             enc->vae_dim, enc->gaussian,
             n_stages, n_assigned, n_weights);
    return VV_OK;
}

vv_status_t vv_conv_vae_free(vv_conv_vae_encoder_t* encoder) {
    if (!encoder) return VV_ERR_NULL_PTR;
    if (encoder->cpu) vv_vae_cpu_weights_free(encoder->cpu);
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
