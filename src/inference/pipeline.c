/**
 * @file pipeline.c
 * @brief End-to-end inference pipeline: WAV -> transcription JSON.
 *
 * Supports multiple placement strategies based on VRAM budget:
 *   - VV_PLACE_ALL_GPU:       All layers + embed + lm_head on GPU.
 *   - VV_PLACE_STREAM_FULL:   Stream layers, embed + lm_head on GPU.
 *   - VV_PLACE_STREAM_EMBED:  Stream layers, embed on GPU, lm_head CPU.
 *   - VV_PLACE_STREAM_ALL:    Stream layers, embed + lm_head on CPU.
 *   - VV_PLACE_CPU_ONLY:      Everything on CPU, no CUDA at all.
 */

#include "vibevoice/inference.h"
#include "vibevoice/audio.h"
#include "vibevoice/text_tokenizer.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/connector.h"
#include "vibevoice/frontend.h"
#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/bitnet.h"
#include "vibevoice/vae_i8.h"
#include "vibevoice/gguf.h"
#include "cJSON.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/cpu_kernels.h"

#include "vv_thread.h"
#include "pipeline_internal.h"
#include "vibevoice/stream.h"
#include "vibevoice/spec.h"
#include "spec_internal.h"

/* Forward declarations — CUDA helpers */


extern vv_status_t vv_postprocess_tokens(
    const char** token_texts, int n_tokens,
    vv_transcription_t** result);

extern vv_status_t vv_transcription_to_json(const vv_transcription_t* tr,
                                              char** json_str);

extern vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                                      int32_t* token_id);

/* Prompts and stop tokens come from the model family (family.h). */

/* ─── FP32 <-> FP16 conversion helpers ──────────────────────────────────── */

static void float_to_half(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } v;
        v.f = src[i];
        uint32_t sign = (v.u >> 16) & 0x8000;
        int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (v.u >> 13) & 0x3FF;
        if (exp <= 0)      dst[i] = (uint16_t)sign;
        else if (exp >= 0x1F) dst[i] = (uint16_t)(sign | 0x7C00);
        else               dst[i] = (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
    }
}

static float half_to_float_single(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
    uint32_t frac = (uint32_t)h & 0x03FF;
    union { float f; uint32_t u; } u;
    if (expo == 0) u.u = sign;
    else if (expo == 0x1F) u.u = sign | 0x7F800000 | (frac << 13);
    else u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
    return u.f;
}

/* ─── Upload tensor to GPU helper (for all-on-GPU path) ─────────────────── */

static vv_status_t upload_tensor_to_gpu(vv_tensor_t* t, void* stream) {
    if (!t || !t->data || t->on_gpu) return VV_OK;
    void* gpu_ptr = NULL;
    vv_status_t s = vv_dev_alloc(&gpu_ptr, t->size_bytes);
    if (s != VV_OK) return s;
    s = vv_dev_memcpy_h2d(gpu_ptr, t->data, t->size_bytes, stream);
    if (s != VV_OK) { vv_dev_free(gpu_ptr); return s; }
    vv_free(t->data);
    t->data = gpu_ptr;
    t->on_gpu = true;
    return VV_OK;
}

/**
 * @brief Upload all NF4 layer weights (packed + scales + norms) to GPU.
 * Used only for VV_PLACE_ALL_GPU mode.
 */
static void attach_frontend(vv_inference_ctx_t* c);
static void attach_spec(vv_inference_ctx_t* c, const char* dir, int quant,
                        int verify_rows, int check,
                        const vv_inference_ctx_t* parent);

/** @brief `<model_dir>/drafter` into `out` when it holds a drafter's
 *         config.json and model.safetensors. */
static bool bundled_drafter(const char* model_dir, char* out, size_t cap) {
    const int n = snprintf(out, cap, "%s/drafter", model_dir);
    if (n <= 0 || (size_t)n >= cap) return false;
    static const char* const need[2] = { "config.json", "model.safetensors" };
    for (int i = 0; i < 2; i++) {
        char probe[1100];
        snprintf(probe, sizeof(probe), "%s/%s", out, need[i]);
        FILE* f = fopen(probe, "rb");
        if (!f) return false;
        fclose(f);
    }
    return true;
}

/**
 * @brief Upload the first `n_layers` transformer layers to the GPU.
 *
 * Partial residency is the point: on a card that cannot hold all 28, the
 * layers that fit stay put and only the remainder is streamed per token, so
 * the PCIe cost scales with what is missing rather than with the whole model.
 */
static vv_status_t upload_layer_range(vv_model_t* model, int first,
                                      int count, void* stream) {
    size_t total_bytes = 0;
    if (first < 0) first = 0;
    if (first + count > model->num_layers) count = model->num_layers - first;
    if (count <= 0) return VV_OK;
    for (int i = first; i < first + count; i++) {
        vv_tensor_t* ts[VV_LAYER_TENSOR_SLOTS];
        const int nt = vv_layer_tensors(&model->layers[i], ts);
        for (int k = 0; k < nt; k++) {
            if (!ts[k]->data) continue;
            vv_status_t s = upload_tensor_to_gpu(ts[k], stream);
            if (s != VV_OK) return s;
            total_bytes += ts[k]->size_bytes;
        }
    }
    if (count == model->num_layers)
        VV_LOG_I("inference: %d/%d layers resident on GPU (%.1f MB)",
                 count, model->num_layers,
                 (double)total_bytes / (1024.0 * 1024.0));
    else
        VV_LOG_I("inference: layers %d..%d resident on GPU (%.1f MB)",
                 first, first + count - 1,
                 (double)total_bytes / (1024.0 * 1024.0));
    return VV_OK;
}

static vv_status_t upload_layer_weights(vv_model_t* model, int n_layers,
                                        void* stream) {
    return upload_layer_range(model, 0, n_layers, stream);
}

/**
 * @brief Free GPU memory for layer tensors (call before vv_model_free).
 */
static void free_layer_gpu_weights(vv_model_t* model) {
    if (!model || !model->layers) return;
    for (int i = 0; i < model->num_layers; i++) {
        vv_tensor_t* ts[VV_LAYER_TENSOR_SLOTS];
        const int nt = vv_layer_tensors(&model->layers[i], ts);
        for (int k = 0; k < nt; k++) {
            if (!ts[k]->on_gpu || !ts[k]->data) continue;
            vv_dev_free(ts[k]->data);
            ts[k]->data = NULL;
            ts[k]->on_gpu = false;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VRAM budget decision logic
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief GPU bytes of the largest layer, as the loader left it.
 *
 * The largest rather than layer 0's: a checkpoint quantized unevenly would
 * otherwise be budgeted by whichever layer happened to come first.
 */
static size_t calc_per_layer_gpu_bytes(const vv_model_t* model) {
    size_t most = 0;
    for (int i = 0; i < model->num_layers; i++) {
        const size_t b = vv_layer_bytes(&model->layers[i]);
        if (b > most) most = b;
    }
    return most;
}

/** @brief Bytes the head adds on its own: none when it is the embedding. */
static size_t lm_head_own_bytes(const vv_model_t* model) {
    return model->lm_head_tied ? 0 : model->lm_head.size_bytes;
}

/**
 * @brief Build the host descriptions of both encoders and both connectors.
 *
 * Host-only and cheap (the tensors are borrowed from the model); done before
 * placement so the budget can size the front end from the real layer plan.
 */
static void build_frontend_host(vv_inference_ctx_t* c) {
    const vv_llm_config_t* llm = &c->model->config.llm;
    vv_status_t s;
    if (c->model->n_acoustic_weights > 0 && !c->acoustic_encoder) {
        s = vv_conv_vae_init(c->model->acoustic_weights,
                              c->model->n_acoustic_weights,
                              &c->model->config.acoustic,
                              true, &c->acoustic_encoder);
        if (s != VV_OK) VV_LOG_W("inference: failed to init acoustic encoder");
    }
    if (c->model->n_semantic_weights > 0 && !c->semantic_encoder) {
        s = vv_conv_vae_init(c->model->semantic_weights,
                              c->model->n_semantic_weights,
                              &c->model->config.semantic,
                              false, &c->semantic_encoder);
        if (s != VV_OK) VV_LOG_W("inference: failed to init semantic encoder");
    }
    if (c->acoustic_encoder && !vv_conv_vae_complete(c->acoustic_encoder)) {
        VV_LOG_W("inference: acoustic encoder is missing weights; not used");
        vv_conv_vae_free(c->acoustic_encoder);
        c->acoustic_encoder = NULL;
    }
    if (c->semantic_encoder && !vv_conv_vae_complete(c->semantic_encoder)) {
        VV_LOG_W("inference: semantic encoder is missing weights; not used");
        vv_conv_vae_free(c->semantic_encoder);
        c->semantic_encoder = NULL;
    }

    if (c->model->acoustic_connector_fc1.tensor.data && !c->acoustic_connector) {
        c->acoustic_connector = (vv_connector_t*)vv_alloc(sizeof(vv_connector_t));
        if (c->acoustic_connector) {
            vv_connector_init(c->acoustic_connector,
                               c->model->config.acoustic_vae_dim,
                               llm->hidden_size);
            c->acoustic_connector->fc1_weight = c->model->acoustic_connector_fc1.tensor;
            c->acoustic_connector->fc1_bias   = c->model->acoustic_connector_fc1.quant.packed;
            c->acoustic_connector->norm_weight = c->model->acoustic_connector_norm.tensor;
            c->acoustic_connector->fc2_weight = c->model->acoustic_connector_fc2.tensor;
            c->acoustic_connector->fc2_bias   = c->model->acoustic_connector_fc2.quant.packed;
        }
    }
    if (c->model->semantic_connector_fc1.tensor.data && !c->semantic_connector) {
        c->semantic_connector = (vv_connector_t*)vv_alloc(sizeof(vv_connector_t));
        if (c->semantic_connector) {
            vv_connector_init(c->semantic_connector,
                               c->model->config.semantic_vae_dim,
                               llm->hidden_size);
            c->semantic_connector->fc1_weight = c->model->semantic_connector_fc1.tensor;
            c->semantic_connector->fc1_bias   = c->model->semantic_connector_fc1.quant.packed;
            c->semantic_connector->norm_weight = c->model->semantic_connector_norm.tensor;
            c->semantic_connector->fc2_weight = c->model->semantic_connector_fc2.tensor;
            c->semantic_connector->fc2_bias   = c->model->semantic_connector_fc2.quant.packed;
        }
    }
}

/**
 * @brief On unified memory, stop holding what the device already holds.
 *
 * Where the GPU's memory is the machine's RAM (Apple Silicon), an uploaded
 * tensor's host copy is the same bytes a second time. The embedding table
 * and the head (2 GB of the 7B) are pointed at their device copies, which
 * the CPU reads directly, so the host paths that use them still work; the
 * speech encoders' FP32 host weights (2.8 GB), which only the CPU encoder
 * and the upload read, are freed once the front end has its FP16 ones.
 * A discrete GPU keeps everything as before.
 */
static void share_uploaded(vv_tensor_t* t, void* dev) {
    if (!t || !dev || !t->data || t->on_gpu) return;
    vv_free(t->data);
    t->data = dev;
    t->on_gpu = true;           /* the context frees it, not vv_tensor_free */
}

/** @brief Host bytes release_encoder_host_weights() will hand back. */
static size_t encoder_host_bytes(const vv_model_t* m) {
    size_t n = 0;
    for (int i = 0; i < m->n_acoustic_weights; i++)
        if (m->acoustic_weights[i].tensor.data &&
            !m->acoustic_weights[i].tensor.on_gpu)
            n += m->acoustic_weights[i].tensor.size_bytes;
    for (int i = 0; i < m->n_semantic_weights; i++)
        if (m->semantic_weights[i].tensor.data &&
            !m->semantic_weights[i].tensor.on_gpu)
            n += m->semantic_weights[i].tensor.size_bytes;
    return n;
}

static void release_encoder_host_weights(vv_inference_ctx_t* c) {
    vv_model_t* m = c->model;
    if (c->acoustic_encoder) vv_conv_vae_forget_host_weights(c->acoustic_encoder);
    if (c->semantic_encoder) vv_conv_vae_forget_host_weights(c->semantic_encoder);
    size_t freed = 0;
    for (int i = 0; i < m->n_acoustic_weights; i++) {
        freed += m->acoustic_weights[i].tensor.size_bytes;
        vv_tensor_free(&m->acoustic_weights[i].tensor);
    }
    for (int i = 0; i < m->n_semantic_weights; i++) {
        freed += m->semantic_weights[i].tensor.size_bytes;
        vv_tensor_free(&m->semantic_weights[i].tensor);
    }
    if (freed)
        VV_LOG_I("inference: unified memory -- freed the speech encoders' "
                 "host copies (%.1f MB)", (double)freed / (1024.0 * 1024.0));
}

static void free_frontend_host(vv_inference_ctx_t* c) {
    if (c->acoustic_encoder) vv_conv_vae_free(c->acoustic_encoder);
    if (c->semantic_encoder) vv_conv_vae_free(c->semantic_encoder);
    if (c->acoustic_connector) {
        vv_connector_free(c->acoustic_connector);
        vv_free(c->acoustic_connector);
    }
    if (c->semantic_connector) {
        vv_connector_free(c->semantic_connector);
        vv_free(c->semantic_connector);
    }
    c->acoustic_encoder = c->semantic_encoder = NULL;
    c->acoustic_connector = c->semantic_connector = NULL;
}

/**
 * @brief Bytes the speech front end will take on the device.
 *
 * Both encoders' and both connectors' FP16 weights, and the arena their
 * launches run in, sized from the largest launch rather than guessed. Held
 * once per device: every slot on it shares them. The old reservation was a
 * flat 256 MB against a real peak of ~640 MB per 60 s segment, plus an
 * uncounted 1.4 GB copy of the weights for every extra slot.
 */
static size_t calc_frontend_gpu_bytes(const vv_inference_ctx_t* c) {
    const vv_frontend_params_t fp = vv_frontend_params_default();
    return vv_frontend_device_bytes(c->acoustic_encoder, c->semantic_encoder,
                                    c->acoustic_connector,
                                    c->semantic_connector, &fp)
         + vv_frontend_stream_bytes(c->acoustic_encoder, c->semantic_encoder);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Layer shards
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_split_mode_t vv_split_mode_parse(const char* name) {
    if (!name) return VV_SPLIT_MODE_COUNT;
    if (strcmp(name, "auto") == 0)    return VV_SPLIT_AUTO;
    if (strcmp(name, "replica") == 0) return VV_SPLIT_REPLICA;
    if (strcmp(name, "layer") == 0)   return VV_SPLIT_LAYER;
    return VV_SPLIT_MODE_COUNT;
}

const char* vv_split_mode_name(vv_split_mode_t mode) {
    switch (mode) {
        case VV_SPLIT_AUTO:    return "auto";
        case VV_SPLIT_REPLICA: return "replica";
        case VV_SPLIT_LAYER:   return "layer";
        default:               return "?";
    }
}

/**
 * @brief Split `n_layers` by how much each device can give to layers.
 *
 * Proportional to `room`, not even, and `room` is what is left of a device's
 * budget after everything that is not a layer. Two reasons that matters. A
 * 24 GB and a 12 GB card should not be handed fourteen layers each, or the
 * smaller one streams while the larger one sits half empty. And the primary
 * is never the equal of the others even when the cards are identical: it
 * also carries the embedding table, the head and the speech encoder, which
 * on this model is 3.4 GB before a single layer lands.
 *
 * Every device gets at least one layer, so a device named on the command
 * line is always a device that does some work.
 */
static void split_layers(const size_t* room, int n_dev, int n_layers,
                         int* out) {
    double total = 0.0;
    for (int i = 0; i < n_dev; i++) total += (double)room[i] + 1.0;

    int assigned = 0;
    for (int i = 0; i < n_dev; i++) {
        out[i] = (int)((double)n_layers * ((double)room[i] + 1.0) / total + 0.5);
        if (out[i] < 1) out[i] = 1;
        assigned += out[i];
    }
    /* Rounding never lands exactly; settle it against the largest share. */
    while (assigned != n_layers) {
        int pick = 0;
        for (int i = 1; i < n_dev; i++)
            if (assigned > n_layers ? out[i] > out[pick] : out[i] < out[pick])
                pick = i;
        if (assigned > n_layers) {
            if (out[pick] <= 1) break;
            out[pick]--; assigned--;
        } else {
            out[pick]++; assigned++;
        }
    }
}

/**
 * @brief Bring up one shard: its streams, KV slice, weights and workspace.
 */
static vv_status_t shard_init(vv_shard_t* sh, vv_model_t* model,
                              const vv_llm_config_t* llm, int max_seq,
                              int kv_format, size_t ws_target) {
    vv_status_t s = vv_dev_set_device(sh->gpu_id);
    if (s != VV_OK) return s;

    s = vv_dev_stream_create(&sh->compute_stream);
    if (s != VV_OK) return s;
    s = vv_dev_stream_create(&sh->transfer_stream);
    if (s != VV_OK) return s;
    s = vv_dev_event_create(&sh->done);
    if (s != VV_OK) return s;

    s = vv_kv_cache_create_range(&sh->kv_cache, llm->num_hidden_layers,
                                 sh->first_layer, sh->n_layers,
                                 llm->num_key_value_heads, llm->head_dim,
                                 max_seq, kv_format, false);
    if (s != VV_OK) return s;

    s = upload_layer_range(model, sh->first_layer, sh->n_resident,
                           sh->transfer_stream);
    if (s != VV_OK) return s;
    vv_dev_stream_sync(sh->transfer_stream);

    for (size_t want = ws_target; ; want /= 2) {
        if (want < (size_t)64 * 1024 * 1024) return VV_ERR_CUDA_OOM;
        if (vv_dev_alloc(&sh->workspace, want) == VV_OK) {
            sh->workspace_size = want;
            break;
        }
    }

    VV_LOG_I("shard: gpu %d holds layers %d..%d (%d resident), workspace %zu MB",
             sh->gpu_id, sh->first_layer, sh->first_layer + sh->n_layers - 1,
             sh->n_resident, sh->workspace_size / (1024 * 1024));
    return VV_OK;
}

static void shard_free(vv_shard_t* sh) {
    if (!sh || sh->gpu_id < 0) return;
    vv_dev_set_device(sh->gpu_id);
    if (sh->kv_cache) vv_kv_cache_free(sh->kv_cache);
    if (sh->layer_pool) vv_layer_pool_free(sh->layer_pool);
    if (sh->workspace) vv_dev_free(sh->workspace);
    if (sh->hidden) vv_dev_free(sh->hidden);
    if (sh->done) vv_dev_event_destroy(sh->done);
    if (sh->compute_stream) vv_dev_stream_destroy(sh->compute_stream);
    if (sh->transfer_stream) vv_dev_stream_destroy(sh->transfer_stream);
    memset(sh, 0, sizeof(*sh));
    sh->gpu_id = -1;
}

/** @brief Grow the shard's landing buffer to hold `bytes` of hidden state. */
static vv_status_t shard_hidden(vv_shard_t* sh, size_t bytes) {
    if (sh->hidden && sh->hidden_bytes >= bytes) return VV_OK;
    if (sh->hidden) { vv_dev_free(sh->hidden); sh->hidden = NULL; }
    const vv_status_t s = vv_dev_alloc(&sh->hidden, bytes);
    sh->hidden_bytes = (s == VV_OK) ? bytes : 0;
    return s;
}

/**
 * @brief Hand the hidden state to `dst_gpu` and make its stream wait for it.
 *
 * The receiving stream cannot start before the sending device has finished
 * writing, and the two are on different devices, so the ordering goes
 * through an event rather than through the stream itself. Leaves `dst_gpu`
 * current, which is what the caller wants next.
 */
static vv_status_t hand_off(int dst_gpu, void* dst, void* dst_stream,
                            int src_gpu, const void* src, void* src_stream,
                            void* src_done, size_t bytes) {
    vv_status_t s = vv_dev_event_record(src_done, src_stream);
    if (s != VV_OK) return s;
    s = vv_dev_set_device(dst_gpu);
    if (s != VV_OK) return s;
    s = vv_dev_stream_wait_event(dst_stream, src_done);
    if (s != VV_OK) return s;
    return vv_dev_memcpy_peer(dst, dst_gpu, src, src_gpu, bytes, dst_stream);
}

/**
 * @brief Prefill every layer, walking the shards in order.
 *
 * With no shards this is the single call it always was.
 */
static vv_status_t prefill_all_shards(vv_inference_ctx_t* ctx, void* hidden,
                                      int seq_len) {
    const size_t bytes = (size_t)seq_len *
                         (size_t)ctx->model->config.llm.hidden_size * 2;
    /* Taps read every layer on one device; a split model has no such place. */
    if (ctx->taps && ctx->n_shards > 0) return VV_ERR_UNSUPPORTED;
    vv_status_t s = vv_decoder_prefill_taps(ctx->model, hidden, seq_len,
                                            ctx->kv_cache, ctx->layer_pool,
                                            ctx->workspace,
                                            ctx->workspace_size,
                                            ctx->compute_stream,
                                            ctx->transfer_stream,
                                            0, ctx->primary_layers,
                                            ctx->taps);
    if (s != VV_OK || ctx->n_shards == 0) return s;

    int src_gpu = ctx->gpu_id;
    void *src = hidden, *src_stream = ctx->compute_stream,
         *src_done = ctx->shard_done;

    for (int i = 0; i < ctx->n_shards; i++) {
        vv_shard_t* sh = &ctx->shards[i];
        s = vv_dev_set_device(sh->gpu_id);
        if (s == VV_OK) s = shard_hidden(sh, bytes);
        if (s == VV_OK)
            s = hand_off(sh->gpu_id, sh->hidden, sh->compute_stream,
                         src_gpu, src, src_stream, src_done, bytes);
        if (s == VV_OK)
            s = vv_decoder_prefill(ctx->model, sh->hidden, seq_len,
                                   sh->kv_cache, sh->layer_pool,
                                   sh->workspace, sh->workspace_size,
                                   sh->compute_stream, sh->transfer_stream,
                                   sh->first_layer, sh->n_layers);
        if (s != VV_OK) { vv_dev_set_device(ctx->gpu_id); return s; }
        src_gpu = sh->gpu_id; src = sh->hidden;
        src_stream = sh->compute_stream; src_done = sh->done;
    }

    /* Back to the primary: the final norm and the head live there. */
    s = hand_off(ctx->gpu_id, hidden, ctx->compute_stream,
                 src_gpu, src, src_stream, src_done, bytes);
    return s;
}

static vv_placement_t decide_placement(
    size_t available, bool cpu_only,
    size_t all_layers_bytes,   /* total for 28 layers */
    size_t embed_bytes,
    size_t lm_head_bytes,
    size_t per_layer_bytes,
    size_t kv_cache_bytes,
    size_t workspace_bytes)
{
    if (cpu_only || available == 0) return VV_PLACE_CPU_ONLY;

    size_t base = workspace_bytes + kv_cache_bytes;
    /* 2 buffers, each with the pool's own alignment slack per tensor. */
    size_t staging_2 = (per_layer_bytes
                        + (size_t)VV_LAYER_TENSORS_PER_LAYER * 256) * 2;

    VV_LOG_D("budget: available %.1f MB, base %.1f MB (ws+kv), "
             "all_layers %.1f MB, embed %.1f MB, lm_head %.1f MB",
             (double)available / (1024.0*1024.0),
             (double)base / (1024.0*1024.0),
             (double)all_layers_bytes / (1024.0*1024.0),
             (double)embed_bytes / (1024.0*1024.0),
             (double)lm_head_bytes / (1024.0*1024.0));

    if (available < base) return VV_PLACE_CPU_ONLY;
    size_t remaining = available - base;

    if (remaining >= all_layers_bytes + embed_bytes + lm_head_bytes) {
        return VV_PLACE_ALL_GPU;
    }
    if (remaining >= staging_2 + embed_bytes + lm_head_bytes) {
        return VV_PLACE_STREAM_FULL;
    }
    if (remaining >= staging_2 + embed_bytes) {
        return VV_PLACE_STREAM_EMBED;
    }
    if (remaining >= staging_2) {
        return VV_PLACE_STREAM_ALL;
    }
    /* Not even enough for staging buffers */
    return VV_PLACE_CPU_ONLY;
}

static const char* placement_str(vv_placement_t p) {
    switch (p) {
        case VV_PLACE_ALL_GPU:      return "ALL_GPU";
        case VV_PLACE_STREAM_FULL:  return "STREAM_FULL";
        case VV_PLACE_STREAM_EMBED: return "STREAM_EMBED";
        case VV_PLACE_STREAM_ALL:   return "STREAM_ALL";
        case VV_PLACE_CPU_ONLY:     return "CPU_ONLY";
        default:                    return "?";
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Init
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               const vv_init_params_t* params,
                               vv_inference_ctx_t** ctx) {
    if (!model_dir || !ctx) return VV_ERR_NULL_PTR;

    /* Resolve params (NULL → defaults) */
    vv_init_params_t p = params ? *params : vv_init_params_default();
    bool cpu_only = p.cpu_only || p.vram_budget <= 0.0f;

    /* A model directory may carry its drafter (`<model>/drafter/`, as the
     * Hub bundles publish one); it is used when the caller named none. An
     * empty draft_dir (`--draft none`) means no drafter at all. */
    char bundled[1024];
    if (!p.draft_dir && bundled_drafter(model_dir, bundled, sizeof(bundled))) {
        p.draft_dir = bundled;
        VV_LOG_I("spec: the drafter in %s", bundled);
    }

    if (p.kv_format < 0 || p.kv_format >= VV_KV_FORMAT_COUNT) {
        VV_LOG_E("inference: unknown KV-cache format");
        return VV_ERR_INVALID_ARG;
    }
    /* Quantized stores need a device; the CPU path keeps its FP32 values. */
    if (cpu_only && !vv_kv_is_raw((vv_kv_format_t)p.kv_format)) {
        VV_LOG_W("inference: KV format %s needs a device; "
                 "falling back to fp16 on CPU",
                 vv_kv_format_name((vv_kv_format_t)p.kv_format));
        p.kv_format = VV_KV_FP32;
    }

    VV_LOG_I("inference: initializing from '%s' %s (budget=%.0f%%)",
             model_dir,
             cpu_only ? "CPU-only" : "on GPU",
             cpu_only ? 0.0 : (double)p.vram_budget * 100.0);

    /* Set GPU (skip if CPU-only) */
    vv_status_t s = VV_OK;
    if (!cpu_only) {
        s = vv_dev_set_device(gpu_id);
        if (s != VV_OK) {
            VV_LOG_W("inference: failed to set GPU %d, falling back to CPU", gpu_id);
            cpu_only = true;
        }
    }

    /* Allocate context */
    vv_inference_ctx_t* c = (vv_inference_ctx_t*)vv_alloc(
        sizeof(vv_inference_ctx_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));
    /*
     * An empty set means the caller only named a device; give it one entry
     * so everything downstream reads the same structure. Validation and the
     * one-line-per-device log belong to whoever parsed the flags.
     */
    if (p.gpus.n == 0) { p.gpus.n = 1; p.gpus.id[0] = gpu_id; }
    c->gpus = p.gpus;
    c->gpu_id = gpu_id;
    c->gpu_index = 0;
    for (int i = 0; i < p.gpus.n; i++)
        if (p.gpus.id[i] == gpu_id) c->gpu_index = i;
    c->use_gpu = !cpu_only;
    strncpy(c->model_dir, model_dir, sizeof(c->model_dir) - 1);

    /* Load model — quantized now if asked, so every size below is final. */
    vv_model_load_opts_t lo = vv_model_load_opts_default();
    lo.quant = p.weight_quant;
    lo.smooth = p.smooth;
    lo.smooth_alpha = p.smooth_alpha;
    lo.smooth_maps = p.smooth_maps;
    if (p.smooth) {
        /* Tuning knobs for the fold, read only when there is one. */
        const char* ea = getenv("VV_SMOOTH_ALPHA");
        const char* em = getenv("VV_SMOOTH_MAPS");
        if (lo.smooth_alpha <= 0.0f && ea && ea[0])
            lo.smooth_alpha = (float)atof(ea);
        if (lo.smooth_maps <= 0 && em && em[0])
            lo.smooth_maps = atoi(em);
    }
    lo.source = p.weights_source;
    lo.head = p.head_format;
    lo.vae = p.vae_numerics;
    lo.cpu = cpu_only ? 1 : 0;
    s = vv_model_load_ex(model_dir, &lo, &c->model);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to load model: %s", vv_status_str(s));
        vv_free(c);
        return s;
    }
    if (!cpu_only) {
        /* BitNet's FP32 norms and Q6_K table, in the forms the GPU reads. */
        s = vv_model_bitnet_prepare_gpu(c->model);
        if (s != VV_OK) {
            vv_model_free(c->model);
            vv_free(c);
            return s;
        }
    }

    const vv_llm_config_t* llm = &c->model->config.llm;
    int max_seq = (p.max_seq_len > 0) ? p.max_seq_len : 32768;
    build_frontend_host(c);

    /*
     * How the model is spread. Layer sharding only makes sense with more
     * than one device, and only for the layers: the embedding table, the
     * final norm and the head stay on the primary, because moving them
     * would buy nothing and cost a round trip per token.
     */
    c->primary_layers = llm->num_hidden_layers;
    int split[VV_MAX_GPUS] = {0};
    const int gpu_layers_total = p.gpu_layers;   /* before it is shared out */
    const bool sharding = !cpu_only && p.gpus.n > 1 &&
                          p.split_mode == VV_SPLIT_LAYER;
    if (sharding) {
        if (c->gpu_index != 0) {
            VV_LOG_W("shard: layers are laid out from gpu %d onwards; "
                     "starting there rather than on gpu %d",
                     p.gpus.id[0], gpu_id);
            c->gpu_index = 0;
            c->gpu_id = gpu_id = p.gpus.id[0];
        }

        const size_t ws = (size_t)512 * 1024 * 1024;
        const size_t head = c->model->embed_tokens.size_bytes
                          + lm_head_own_bytes(c->model)
                          + calc_frontend_gpu_bytes(c);
        size_t room[VV_MAX_GPUS];
        for (int i = 0; i < p.gpus.n; i++) {
            size_t b = vv_gpu_budget(&p.gpus, i, p.vram_budget);
            const size_t held = vv_gpu_reserved(&p.gpus, i);
            b = b > held ? b - held : 0;
            const size_t fixed = ws + (i == 0 ? head : 0);
            room[i] = b > fixed ? b - fixed : 0;
        }
        split_layers(room, p.gpus.n, llm->num_hidden_layers, split);
        c->primary_layers = split[0];
        if (p.gpu_layers >= 0) {
            /* --gpu-layers counts the whole model; share it out. */
            p.gpu_layers = (int)((double)gpu_layers_total * c->primary_layers
                                 / llm->num_hidden_layers + 0.5);
        }
    }

    /*
     * Attention backend and KV paging. Paging lets the slots of one device
     * share a pool; it needs kernels that read a page table (fa2 on FP16,
     * flashinfer on any format) and a single device holding every layer's KV.
     */
    const int n_slots = p.n_slots > 0 ? p.n_slots : 1;
    vv_attn_backend_t attn_want =
        vv_attn_backend_from_env((vv_attn_backend_t)p.attn_backend);
    /*
     * The streaming model prefills 29 rows at a time on top of a cache that
     * keeps growing. fa2 runs those as one row kernel -- 16 blocks on an
     * 82-SM card, each walking the whole cache -- while flashinfer splits
     * the cache across the idle SMs. `auto` is fa2 for the batch model to
     * keep its transcripts bit-identical to the old kernels; this family
     * has no such history, and with flashinfer it still matches upstream
     * chunk for chunk (jfk, test30, test120), so `auto` takes the faster
     * one here. `--attn fa2` still gets fa2.
     */
    if (attn_want == VV_ATTN_AUTO &&
        c->model->config.family == VV_FAMILY_ASR_STREAMING_7B)
        attn_want = VV_ATTN_FLASHINFER;
    /*
     * With a drafter `auto` is flashinfer as well. A checked block runs each
     * row's attention as its own decode step; fa2's walk over the cache is
     * one dependent reduction per position, so eight rows cost eight walks,
     * while flashinfer packs two or three rows to a fragment (AWQ 7B,
     * 32-minute file: drafted 1.06x the plain decode with fa2, 1.47x with
     * flashinfer). The drafted transcript is then byte for byte that of
     * `--attn flashinfer` without a drafter; `--attn fa2 --draft` keeps fa2.
     * Not where the drafter is declined for sure (CPU, layer shards); where
     * it is declined later, a slab goes back to `auto`'s choice (below).
     */
    const bool attn_for_draft = attn_want == VV_ATTN_AUTO && !cpu_only &&
                                !sharding && p.draft_dir && p.draft_dir[0];
    if (attn_for_draft)
        attn_want = VV_ATTN_FLASHINFER;
    const int attn_slab = vv_attn_resolve(
        (int)attn_want, p.kv_format, false, llm->num_attention_heads,
        llm->num_key_value_heads, llm->head_dim);
    const int attn_paged = vv_attn_resolve(
        (int)attn_want, p.kv_format, true, llm->num_attention_heads,
        llm->num_key_value_heads, llm->head_dim);
    /*
     * `auto` pages only where that changes nothing but memory: several slots
     * whose kernels read pages as they are. Forcing it on may pick other
     * kernels for `auto`, but never overrides a backend asked for by name.
     */
    const bool pages_readable =
        attn_paged == VV_ATTN_FLASHINFER ||
        (attn_paged == VV_ATTN_FA2 && p.kv_format == VV_KV_FP16);
    bool paged = !cpu_only && !sharding && pages_readable &&
                 (p.kv_paging == VV_KV_PAGED_ON ||
                  (p.kv_paging == VV_KV_PAGED_AUTO && n_slots > 1 &&
                   attn_paged == attn_slab));
    if (paged && attn_want != VV_ATTN_AUTO && attn_paged != (int)attn_want)
        paged = false;
    if (!paged && p.kv_paging == VV_KV_PAGED_ON)
        VV_LOG_W("kv: paging needs --attn flashinfer, or fa2 on an fp16 "
                 "cache, on one device; using one slab per slot");
    int pool_tokens = max_seq;

    /* ── Decide placement strategy ── */
    if (cpu_only) {
        c->placement = VV_PLACE_CPU_ONLY;
    } else {
        /*
         * What is free, scaled by --vram-budget, and then never more than
         * --gpu-memory allows. Everything below works off `available`, so
         * the cap is honoured by the placement decision itself rather than
         * checked once the weights are already uploaded.
         */
        size_t per_layer = calc_per_layer_gpu_bytes(c->model);
        size_t all_layers = per_layer * (size_t)c->primary_layers;
        size_t embed_sz = c->model->embed_tokens.size_bytes;
        size_t lm_head_sz = lm_head_own_bytes(c->model);
        /*
         * On unified memory those three are host bytes that placing them
         * gives back, so they are part of what the budget can spend; on a
         * discrete card they are a second copy across a bus, and are not.
         *
         * This counts what an all-resident placement frees. Should the
         * budget land on a streaming one instead, the layers left on the
         * host keep their copies and the estimate is that much generous --
         * but the loop below shrinks the window to the floor before it
         * gives up on residency, and streaming is the wrong answer here
         * anyway: the bytes are in the same RAM either way.
         */
        const size_t reclaimable = vv_dev_host_shares_memory(c->gpu_id)
            ? all_layers + embed_sz + lm_head_sz
              + encoder_host_bytes(c->model) : 0;
        size_t available = vv_gpu_budget_reclaim(&p.gpus, c->gpu_index,
                                                 p.vram_budget, reclaimable);
        /*
         * Three things come out of the budget before anything is placed,
         * because all three are spent on the device and none of them are
         * decided here: what the device already holds (this process's CUDA
         * context, the driver, anyone else), the speech front end, and the
         * scratch its encoder needs for the longest segment it will see.
         * Leave them out and a cap overshoots by about 2 GB, which makes it
         * no cap at all.
         */
        const size_t frontend = calc_frontend_gpu_bytes(c);
        size_t reserve = vv_gpu_reserved(&p.gpus, c->gpu_index)
                       + frontend;
        /* A drafter: its weights once, its state once per slot. */
        if (p.draft_dir && p.draft_dir[0]) {
            vv_drafter_config_t dc;
            if (vv_drafter_config_load(p.draft_dir, &dc) == VV_OK) {
                dc.weight_quant = p.draft_quant;
                const size_t sb = vv_drafter_weight_bytes(&dc) +
                                  (size_t)n_slots *
                                  vv_spec_context_bytes(&dc, max_seq);
                reserve += sb;
                VV_LOG_I("budget: %.1f MB for the drafter",
                         (double)sb / (1024.0 * 1024.0));
            }
        }
        VV_LOG_I("budget: reserving %.1f MB (in use %.1f, speech front end "
                 "%.1f incl. its arena)",
                 (double)reserve / (1024.0*1024.0),
                 (double)vv_gpu_reserved(&p.gpus, c->gpu_index) / (1024.0*1024.0),
                 (double)frontend / (1024.0*1024.0));
        available = available > reserve ? available - reserve : 0;

        size_t kv_per_token = vv_kv_cache_bytes(
            c->primary_layers, llm->num_key_value_heads,
            llm->head_dim, 1, p.kv_format);
        /*
         * A streaming model's context starts at the 256 MB workspace (see
         * the allocation below), and so does every clone of it; budgeting
         * 512 MB per slot there would starve the KV pool the sessions share.
         */
        size_t ws_target =
            c->model->config.family == VV_FAMILY_ASR_STREAMING_7B
            ? (size_t)256 * 1024 * 1024 : (size_t)512 * 1024 * 1024;

        /*
         * The context is the first thing to give up. A shorter window costs
         * one long recording; streaming the layers costs every token of
         * every recording. So trim it — to whatever the budget leaves when
         * the weights stay put, and failing that by halving until they fit,
         * down to a floor below which a transcription is not worth starting.
         */
        const int kv_floor = 8192;
        /* A shared pool also has to leave room for the other slots'
         * workspaces, which clones allocate after this. */
        const size_t clone_ws = paged ? (size_t)(n_slots - 1) * ws_target : 0;
        const size_t resident = ws_target + all_layers + embed_sz + lm_head_sz
                              + clone_ws;
        const int max_seq_asked = max_seq;
        if (available > resident) {
            size_t kv_room = (available - resident) / kv_per_token;
            if (kv_room < (size_t)max_seq) {
                int fitted = (int)(kv_room & ~(size_t)255);
                if (fitted >= 2048) max_seq = fitted;
            }
        }

        /*
         * A pool is sized for every slot's window when that fits, and for
         * whatever does fit otherwise, but never below one window: a slot
         * running alone may always use all of it.
         */
        pool_tokens = max_seq;
        if (paged) {
            size_t want = (size_t)max_seq * (size_t)n_slots;
            const size_t room = available > resident
                              ? (available - resident) / kv_per_token : 0;
            if (room < want) want = room > (size_t)max_seq ? room : (size_t)max_seq;
            pool_tokens = (int)(want / VV_KV_PAGE_SIZE * VV_KV_PAGE_SIZE);
            if (pool_tokens < max_seq) pool_tokens = max_seq;
        }
        size_t kv_total = (size_t)pool_tokens * kv_per_token;
        for (;;) {
            c->placement = decide_placement(
                available, false,
                all_layers, embed_sz, lm_head_sz,
                per_layer, kv_total, ws_target);
            if (c->placement == VV_PLACE_ALL_GPU || max_seq <= kv_floor) break;
            max_seq = max_seq / 2 < kv_floor ? kv_floor : max_seq / 2;
            pool_tokens = max_seq;
            kv_total = (size_t)max_seq * kv_per_token;
        }
        /* One line at the end, not one per attempt around the loop. */
        VV_LOG_I("budget: %.1f MB for %.1f MB of layers, %.1f embed, "
                 "%.1f lm_head, %.1f KV and %.1f workspace",
                 (double)available / (1024.0*1024.0),
                 (double)all_layers / (1024.0*1024.0),
                 (double)embed_sz / (1024.0*1024.0),
                 (double)lm_head_sz / (1024.0*1024.0),
                 (double)kv_total / (1024.0*1024.0),
                 (double)ws_target / (1024.0*1024.0));
        if (max_seq != max_seq_asked)
            VV_LOG_W("inference: KV window trimmed %d -> %d tokens to keep "
                     "%s", max_seq_asked, max_seq,
                     c->placement == VV_PLACE_ALL_GPU
                     ? "the weights resident" : "as many layers resident as fit");

        /*
         * When not everything fits, work out how many layers do. Two staging
         * buffers must be kept back for the ones that don't, plus embed and
         * lm_head if this placement keeps them on the GPU.
         */
        size_t fixed = ws_target + kv_total + 2 * per_layer;
        if (c->placement == VV_PLACE_STREAM_FULL)
            fixed += embed_sz + lm_head_sz;
        else if (c->placement == VV_PLACE_STREAM_EMBED)
            fixed += embed_sz;
        if (available > fixed) {
            size_t fit = (available - fixed) / per_layer;
            c->auto_resident_layers = fit > (size_t)llm->num_hidden_layers
                                      ? llm->num_hidden_layers : (int)fit;
        } else {
            c->auto_resident_layers = 0;
        }
    }

    /*
     * How many layers stay on the GPU. The placement enum only says whether
     * embed and lm_head fit; this is the finer knob, and the one that
     * actually decides the per-token PCIe bill.
     */
    if (c->placement == VV_PLACE_CPU_ONLY) {
        c->n_resident_layers = 0;
    } else if (p.gpu_layers >= 0) {
        c->n_resident_layers = p.gpu_layers < c->primary_layers
                               ? p.gpu_layers : c->primary_layers;
    } else if (c->placement == VV_PLACE_ALL_GPU) {
        c->n_resident_layers = c->primary_layers;
    } else {
        c->n_resident_layers = c->auto_resident_layers;
    }

    VV_LOG_I("inference: placement strategy = %s, %d/%d layers resident",
             placement_str(c->placement), c->n_resident_layers,
             c->primary_layers);

    /*
     * BitNet loads differently per backend: the GPU preparation above turned
     * the norms into F16, and the encoder and head defaults were picked for
     * the GPU. When the budget sends the model to the CPU anyway, load it
     * again the way --cpu does, so the result is the --cpu transcript.
     */
    if (c->placement == VV_PLACE_CPU_ONLY && !lo.cpu &&
        c->model->config.family == VV_FAMILY_ASR_BITNET) {
        VV_LOG_W("inference: nothing fits on the GPU; reloading BitNet "
                 "for the CPU");
        free_frontend_host(c);
        vv_model_free(c->model);
        c->model = NULL;
        lo.cpu = 1;
        s = vv_model_load_ex(model_dir, &lo, &c->model);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to reload model: %s",
                     vv_status_str(s));
            vv_free(c);
            return s;
        }
        llm = &c->model->config.llm;
        build_frontend_host(c);
    }

    /* ── CPU-only path: skip all CUDA ── */
    if (c->placement == VV_PLACE_CPU_ONLY) {
        c->use_gpu = false;
        s = vv_kv_cache_create(&c->kv_cache,
                                llm->num_hidden_layers,
                                llm->num_key_value_heads,
                                llm->head_dim,
                                max_seq, VV_KV_FP32, true);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to create CPU KV-cache");
            vv_model_free(c->model);
            vv_free(c);
            return s;
        }

        /* CPU workspace: FP32 intermediates */
        int hs = llm->hidden_size;
        int inter = llm->intermediate_size;
        int n_h = llm->num_attention_heads;
        int n_kv = llm->num_key_value_heads;
        int hd = llm->head_dim;
        /* Max per-layer: seq_len * (hs + n_h*hd + n_kv*hd + n_kv*hd + hs
         *                           + inter + inter + hs) + N*K for dequant */
        /* For decode (seq=1): relatively small. Allocate 512 MB CPU workspace. */
        c->workspace_size = (size_t)512 * 1024 * 1024;
        c->workspace = vv_alloc(c->workspace_size);
        if (!c->workspace) {
            c->workspace_size = (size_t)256 * 1024 * 1024;
            c->workspace = vv_alloc(c->workspace_size);
        }
        if (!c->workspace) {
            VV_LOG_E("inference: cannot allocate CPU workspace");
            vv_kv_cache_free(c->kv_cache);
            vv_model_free(c->model);
            vv_free(c);
            return VV_ERR_OUT_OF_MEMORY;
        }

        /* Calling this once here also applies the physical-core default. */
        VV_LOG_I("inference: CPU kernels use %s on %d thread(s)",
                 vv_cpu_simd_name(), vv_cpu_threads());
        goto init_common;
    }

    /*
     * INT4 (AWQ / GPTQ) weights take the W4A16 kernels' layout here, before
     * anything is uploaded or pinned, so resident and streamed layers carry
     * the same bytes and the host copy is never kept twice.
     */
    s = vv_model_int4g_to_gpu_layout(c->model, NULL);
    if (s != VV_OK) {
        VV_LOG_E("inference: INT4 weight layout failed: %s",
                 vv_status_str(s));
        vv_model_free(c->model); vv_free(c); return s;
    }

    /* ── GPU path: create streams ── */
    s = vv_dev_stream_create(&c->compute_stream);
    if (s != VV_OK) { vv_model_free(c->model); vv_free(c); return s; }
    s = vv_dev_stream_create(&c->transfer_stream);
    if (s != VV_OK) {
        vv_dev_stream_destroy(c->compute_stream);
        vv_model_free(c->model); vv_free(c); return s;
    }

    /* ── KV-cache on GPU ── */
    if (paged) {
        vv_kv_pool_t* kv_pool = NULL;
        s = vv_kv_pool_create(&kv_pool, llm->num_hidden_layers, 0,
                              c->primary_layers, llm->num_key_value_heads,
                              llm->head_dim,
                              (pool_tokens + VV_KV_PAGE_SIZE - 1)
                                  / VV_KV_PAGE_SIZE,
                              p.kv_format);
        if (s == VV_OK) {
            s = vv_kv_cache_create_paged(&c->kv_cache, kv_pool, max_seq);
            vv_kv_pool_release(kv_pool);        /* the cache holds it now */
        }
    } else {
        s = vv_kv_cache_create_range(&c->kv_cache,
                                     llm->num_hidden_layers, 0,
                                     c->primary_layers,
                                     llm->num_key_value_heads,
                                     llm->head_dim,
                                     max_seq, p.kv_format, false);
    }
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to create KV-cache");
        goto fail_gpu;
    }
    c->kv_cache->attn_backend = paged ? attn_paged : attn_slab;
    VV_LOG_I("inference: attention %s (asked %s), KV %s%s",
             vv_attn_backend_name((vv_attn_backend_t)c->kv_cache->attn_backend),
             vv_attn_backend_name(attn_want),
             vv_kv_format_name((vv_kv_format_t)p.kv_format),
             paged ? ", paged" : "");

    /* ── Layer weights ── */
    if (c->n_resident_layers > 0) {
        s = upload_layer_weights(c->model, c->n_resident_layers,
                                 c->transfer_stream);
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload layer weights");
            goto fail_gpu;
        }
    }
    s = vv_layer_pool_create(&c->layer_pool, c->model,
                             c->n_resident_layers >= c->primary_layers);
    if (s == VV_OK && c->n_resident_layers < c->primary_layers)
        vv_layer_pool_pin_range(c->model, c->n_resident_layers,
                                c->primary_layers - c->n_resident_layers);
    if (s != VV_OK) {
        VV_LOG_E("inference: failed to create layer pool");
        goto fail_gpu;
    }

    /* ── GPU workspace ── */
    {
        static const size_t ws_sizes[] = {
            (size_t)512 * 1024 * 1024,
            (size_t)384 * 1024 * 1024,
            (size_t)256 * 1024 * 1024,
        };
        s = VV_ERR_CUDA_OOM;
        /*
         * A streaming model never prefills more than one chunk (29 rows) or
         * its prompt at a time: the weight scratch plus a few MB of
         * activations. The smallest size leaves room for more slots.
         */
        const int ws_first =
            c->model->config.family == VV_FAMILY_ASR_STREAMING_7B ? 2 : 0;
        for (int wi = ws_first; wi < 3; wi++) {
            c->workspace_size = ws_sizes[wi];
            s = vv_dev_alloc(&c->workspace, c->workspace_size);
            if (s == VV_OK) break;
            VV_LOG_W("inference: failed to alloc %zu MB workspace, trying smaller",
                     c->workspace_size / (1024 * 1024));
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: cannot allocate workspace");
            goto fail_gpu;
        }
    }

    /* ── The rest of the layers, on the other devices ── */
    if (sharding) {
        c->shards = (vv_shard_t*)vv_alloc(
            (size_t)(p.gpus.n - 1) * sizeof(vv_shard_t));
        if (!c->shards) { s = VV_ERR_OUT_OF_MEMORY; goto fail_gpu; }
        memset(c->shards, 0, (size_t)(p.gpus.n - 1) * sizeof(vv_shard_t));
        s = vv_dev_event_create(&c->shard_done);
        if (s != VV_OK) goto fail_gpu;

        int next = c->primary_layers;
        for (int i = 1; i < p.gpus.n; i++) {
            vv_shard_t* sh = &c->shards[c->n_shards];
            sh->gpu_id = p.gpus.id[i];
            sh->first_layer = next;
            sh->n_layers = split[i];
            next += split[i];

            /*
             * Each device sizes its own residency from its own budget, the
             * same ladder the primary just went through, only for its slice.
             */
            size_t budget = vv_gpu_budget(&p.gpus, i, p.vram_budget);
            const size_t held = vv_gpu_reserved(&p.gpus, i);
            budget = budget > held ? budget - held : 0;
            const size_t per_layer = calc_per_layer_gpu_bytes(c->model);
            const size_t kv = vv_kv_cache_bytes(sh->n_layers,
                                                llm->num_key_value_heads,
                                                llm->head_dim, max_seq,
                                                p.kv_format);
            size_t ws = (size_t)512 * 1024 * 1024;
            size_t room = budget > kv + ws ? budget - kv - ws : 0;
            sh->n_resident = (int)(room / per_layer);
            if (sh->n_resident > sh->n_layers) sh->n_resident = sh->n_layers;
            if (gpu_layers_total >= 0) {
                const int want = (int)((double)gpu_layers_total * sh->n_layers
                                       / llm->num_hidden_layers + 0.5);
                if (want < sh->n_resident) sh->n_resident = want;
            }

            s = shard_init(sh, c->model, llm, max_seq, p.kv_format, ws);
            if (s == VV_OK)
                sh->kv_cache->attn_backend = attn_slab;
            if (s != VV_OK) {
                VV_LOG_E("shard: gpu %d unusable (%s)", sh->gpu_id,
                         vv_status_str(s));
                shard_free(sh);
                goto fail_gpu;
            }
            s = vv_layer_pool_create(&sh->layer_pool, c->model,
                                     sh->n_resident >= sh->n_layers);
            if (s != VV_OK) { shard_free(sh); goto fail_gpu; }
            if (sh->n_resident < sh->n_layers)
                vv_layer_pool_pin_range(c->model,
                                        sh->first_layer + sh->n_resident,
                                        sh->n_layers - sh->n_resident);
            c->n_shards++;
        }
        vv_dev_set_device(c->gpu_id);

        /* Direct peer copies where the pair allows it; through the host
           where it does not, which is slower but not different. */
        for (int i = 0; i < p.gpus.n; i++)
            for (int j = 0; j < p.gpus.n; j++)
                if (i != j) vv_dev_enable_peer(p.gpus.id[i], p.gpus.id[j]);

        VV_LOG_I("shard: %d devices, layers 0..%d on gpu %d and %d more shard(s)",
                 p.gpus.n, c->primary_layers - 1, c->gpu_id, c->n_shards);
    }

    /* ── Embed table: on GPU unless STREAM_ALL ── */
    if (c->placement != VV_PLACE_STREAM_ALL &&
        c->model->embed_tokens.data) {
        size_t sz = c->model->embed_tokens.size_bytes;
        s = vv_dev_alloc(&c->embed_table_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->embed_table_gpu,
                                    c->model->embed_tokens.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_W("inference: embed_tokens GPU upload failed, using CPU path");
            if (c->embed_table_gpu) { vv_dev_free(c->embed_table_gpu); c->embed_table_gpu = NULL; }
        } else {
            VV_LOG_I("inference: embed_tokens uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    /* ── LM head: on GPU unless STREAM_EMBED or STREAM_ALL ── */
    if (c->model->lm_head_tied) {
        /* The head is the embedding table: wherever that went, it went. */
        c->lm_head_gpu = c->embed_table_gpu;
        if (c->lm_head_gpu)
            VV_LOG_I("inference: lm_head is tied to embed_tokens (shared)");
    } else if (c->placement != VV_PLACE_STREAM_EMBED &&
        c->placement != VV_PLACE_STREAM_ALL &&
        c->model->lm_head.data) {
        size_t sz = c->model->lm_head.size_bytes;
        s = vv_dev_alloc(&c->lm_head_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->lm_head_gpu,
                                    c->model->lm_head.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_W("inference: lm_head GPU upload failed, using CPU path");
            if (c->lm_head_gpu) { vv_dev_free(c->lm_head_gpu); c->lm_head_gpu = NULL; }
        } else {
            VV_LOG_I("inference: lm_head uploaded (%.1f MB)",
                     (double)sz / (1024.0 * 1024.0));
        }
    }

    if (vv_dev_host_shares_memory(c->gpu_id)) {
        share_uploaded(&c->model->embed_tokens, c->embed_table_gpu);
        if (c->model->lm_head_tied) c->model->lm_head = c->model->embed_tokens;
        else share_uploaded(&c->model->lm_head, c->lm_head_gpu);
    }

    /* ── Final norm: always on GPU for GPU mode ── */
    if (c->model->final_norm.data) {
        size_t sz = c->model->final_norm.size_bytes;
        s = vv_dev_alloc(&c->final_norm_gpu, sz);
        if (s == VV_OK) {
            s = vv_dev_memcpy_h2d(c->final_norm_gpu,
                                    c->model->final_norm.data, sz,
                                    c->transfer_stream);
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: failed to upload final_norm to GPU");
        }
    }

    /* Sync transfers */
    vv_dev_stream_sync(c->transfer_stream);
    goto init_common;

fail_gpu:
    free_frontend_host(c);
    if (c->layer_pool) vv_layer_pool_free(c->layer_pool);
    if (c->kv_cache) vv_kv_cache_free(c->kv_cache);
    if (c->workspace && c->use_gpu) vv_dev_free(c->workspace);
    if (c->compute_stream) vv_dev_stream_destroy(c->compute_stream);
    if (c->transfer_stream) vv_dev_stream_destroy(c->transfer_stream);
    vv_model_free(c->model);
    vv_free(c);
    return s;

init_common:
    attach_frontend(c);
    attach_spec(c, p.draft_dir, p.draft_quant, p.draft_block, p.draft_check,
                NULL);
    /* `--draft` made `auto` flashinfer for the rows the drafter's blocks
     * check. Without the drafter, decode as `auto` does: a slab's backend is
     * only which kernels read it (the scratch fits every backend); a paged
     * pool keeps the backend its layout was chosen for. */
    if (attn_for_draft && !c->drafter && c->kv_cache && !paged) {
        const int plain = vv_attn_resolve(VV_ATTN_AUTO, p.kv_format, false,
                                          llm->num_attention_heads,
                                          llm->num_key_value_heads,
                                          llm->head_dim);
        if (plain != c->kv_cache->attn_backend) {
            c->kv_cache->attn_backend = plain;
            VV_LOG_I("inference: no drafter; attention %s",
                     vv_attn_backend_name((vv_attn_backend_t)plain));
        }
    }

    VV_LOG_I("inference: initialized (%s, workspace=%zu MB, tokenizer=%s)",
             placement_str(c->placement),
             c->workspace_size / (1024 * 1024),
             c->tokenizer ? "yes" : "no");
    *ctx = c;
    return VV_OK;
}


/**
 * @brief Give a context its drafter: the parent loads it from `dir`, a
 *        clone borrows the parent's. Speculative decoding needs the whole
 *        model, its embedding, head and final norm on one device; anything
 *        else decodes without it, with a warning, rather than failing.
 */
static void attach_spec(vv_inference_ctx_t* c, const char* dir, int quant,
                        int verify_rows, int check,
                        const vv_inference_ctx_t* parent) {
    if (!parent && (!dir || !dir[0])) return;
    c->draft_block = verify_rows;
    c->draft_check = check;
    if (parent && !parent->drafter) return;
    const char* why = NULL;
    if (!c->use_gpu || c->placement == VV_PLACE_CPU_ONLY)
        why = "the drafter runs on the GPU only";
    else if (c->n_shards > 0)
        why = "the model is split across devices";
    else if (!c->embed_table_gpu || !c->lm_head_gpu || !c->final_norm_gpu)
        why = "the embedding, LM head and final norm must be on the GPU";
    else if (!c->kv_cache || c->kv_cache->on_cpu)
        why = "the KV cache is not on the GPU";
    else if (check != VV_DRAFT_CHECK_FAST && vv_decoder_verify_per_row(c->model))
        /* The 7B drafter on the NF4 checkpoint, test120: 124 tok/s plain,
         * 112 with exact blocks (paused as soon as they were measured);
         * INT8, 96 and 88. */
        why = "these weights (NF4, INT8) check a block exactly one row at a "
              "time, which never beats plain decoding; --draft-check fast "
              "checks it in one pass";
    else {
        /* The drafter's kernels are CUDA's; a backend without them declines
         * each one (VV_ERR_UNSUPPORTED). Ask before loading anything. */
        int32_t none = 0;
        if (vv_gather_i32_dev(&none, 1, &none, 0, NULL) == VV_ERR_UNSUPPORTED)
            why = "this backend has no speculative decoding kernels";
    }
    if (why) {
        VV_LOG_W("spec: %s; decoding without the drafter", why);
        if (!parent) return;
        c->drafter = NULL;
        return;
    }
    const int rows_max = vv_decoder_verify_rows_max(c->model);
    if (verify_rows > rows_max) {
        if (!parent)
            VV_LOG_W("spec: this model checks at most %d rows exactly "
                     "(W4A8); --draft-block %d -> %d", rows_max, verify_rows,
                     rows_max);
        verify_rows = rows_max;
    }
    vv_status_t s = VV_OK;
    if (!parent)
        s = vv_drafter_load(dir, c->model, c->gpu_id, quant, &c->drafter);
    if (s == VV_OK)
        s = vv_spec_create(c->drafter, c->model->config.llm.hidden_size,
                           c->kv_cache->max_seq_len, c->kv_cache->attn_backend,
                           verify_rows, check, c->compute_stream, &c->spec);
    if (s != VV_OK) {
        VV_LOG_W("spec: no drafter (%s); decoding without it",
                 vv_status_str(s));
        if (!parent && c->drafter) vv_drafter_free(c->drafter);
        c->drafter = NULL;
        c->spec = NULL;
        return;
    }
    vv_spec_limit_rows(c->spec, rows_max);
}

const vv_spec_stats_t* vv_inference_spec_stats(const vv_inference_ctx_t* ctx) {
    return ctx && ctx->spec ? vv_spec_get_stats(ctx->spec) : NULL;
}

/**
 * @brief Attach the tokenizer and the speech front end to a context.
 *
 * Shared by vv_inference_init and vv_inference_clone. The parent brings the
 * front end up on its device — the one upload of the encoder and connector
 * weights; a clone borrows it (set by the caller) and gets only its own
 * streaming state and events.
 */
static void attach_frontend(vv_inference_ctx_t* c) {
    vv_status_t s = vv_tokenizer_load(c->model_dir, &c->tokenizer);
    if (s != VV_OK)
        VV_LOG_W("inference: failed to load tokenizer from '%s'", c->model_dir);

    c->family_ok = c->tokenizer &&
        vv_family_init(&c->family, &c->model->config, c->tokenizer) == VV_OK;
    if (c->tokenizer && !c->family_ok)
        VV_LOG_W("inference: the tokenizer lacks tokens %s needs",
                 vv_model_family_name(c->model->config.family));

    if (!c->is_clone) build_frontend_host(c);
    if (!c->use_gpu) {
        /* Repack the CPU encoders' FP16 weights now, not in the first
           request, whose time it would otherwise be charged to. */
        const double t_w = vv_time_ms();
        vv_status_t sa = VV_OK, ss = VV_OK;
        if (c->acoustic_encoder) sa = vv_conv_vae_prepare_cpu(c->acoustic_encoder);
        if (c->semantic_encoder) ss = vv_conv_vae_prepare_cpu(c->semantic_encoder);
        if (sa != VV_OK || ss != VV_OK)
            VV_LOG_W("inference: CPU speech encoder weights not prepared");
        else if (c->acoustic_encoder || c->semantic_encoder)
            VV_LOG_I("inference: CPU speech encoder ready (%.0f ms)",
                     vv_time_ms() - t_w);
        return;
    }

    if (!c->is_clone && (c->acoustic_encoder || c->semantic_encoder)) {
        const vv_frontend_params_t fp = vv_frontend_params_default();
        double t_w = vv_time_ms();
        s = vv_frontend_create(c->acoustic_encoder, c->semantic_encoder,
                               c->acoustic_connector, c->semantic_connector,
                               &fp, &c->frontend);
        if (s != VV_OK) {
            VV_LOG_E("inference: speech front end unavailable on gpu %d: %s",
                     c->gpu_id, vv_status_str(s));
            c->frontend = NULL;
            return;
        }
        VV_LOG_I("inference: speech front end ready (%.0f ms, %.1f MB)",
                 vv_time_ms() - t_w,
                 (double)vv_frontend_bytes(c->frontend) / (1024.0 * 1024.0));
        if (vv_dev_host_shares_memory(c->gpu_id)) release_encoder_host_weights(c);
    }
    if (!c->frontend) return;
    s = vv_frontend_stream_create(c->frontend, &c->fe_stream);
    if (s == VV_OK) s = vv_dev_event_create(&c->fe_ready);
    if (s == VV_OK) s = vv_dev_event_create(&c->fe_done);
    if (s != VV_OK)
        VV_LOG_E("inference: no speech encoder state: %s", vv_status_str(s));
}

vv_status_t vv_inference_clone(const vv_inference_ctx_t* parent,
                                const vv_init_params_t* params,
                                vv_inference_ctx_t** out) {
    if (!parent || !out) return VV_ERR_NULL_PTR;
    if (!parent->use_gpu || parent->placement != VV_PLACE_ALL_GPU) {
        VV_LOG_E("inference: clone needs a parent with all layers resident");
        return VV_ERR_UNSUPPORTED;
    }
    for (int i = 0; i < parent->n_shards; i++) {
        if (parent->shards[i].n_resident < parent->shards[i].n_layers) {
            VV_LOG_E("inference: clone needs every shard resident; gpu %d "
                     "streams %d of its %d layers", parent->shards[i].gpu_id,
                     parent->shards[i].n_layers
                     - parent->shards[i].n_resident,
                     parent->shards[i].n_layers);
            return VV_ERR_UNSUPPORTED;
        }
    }

    vv_init_params_t p = params ? *params : vv_init_params_default();
    const vv_llm_config_t* llm = &parent->model->config.llm;
    const int max_seq = (p.max_seq_len > 0) ? p.max_seq_len : 32768;

    vv_inference_ctx_t* c =
        (vv_inference_ctx_t*)vv_alloc(sizeof(vv_inference_ctx_t));
    if (!c) return VV_ERR_OUT_OF_MEMORY;
    memset(c, 0, sizeof(*c));

    c->is_clone  = true;
    c->gpu_id    = parent->gpu_id;
    c->gpu_index = parent->gpu_index;
    c->gpus      = parent->gpus;
    c->primary_layers = parent->primary_layers;
    c->use_gpu   = true;
    c->placement = parent->placement;
    c->model     = parent->model;
    c->layer_pool = parent->layer_pool;      /* all_resident: stateless */
    c->embed_table_gpu = parent->embed_table_gpu;
    c->lm_head_gpu     = parent->lm_head_gpu;
    c->final_norm_gpu  = parent->final_norm_gpu;
    c->frontend        = parent->frontend;     /* one per device, shared */
    memcpy(c->model_dir, parent->model_dir, sizeof(c->model_dir));

    vv_status_t s = vv_dev_stream_create(&c->compute_stream);
    if (s != VV_OK) { vv_free(c); return s; }
    s = vv_dev_stream_create(&c->transfer_stream);
    if (s != VV_OK) {
        vv_dev_stream_destroy(c->compute_stream);
        vv_free(c);
        return s;
    }

    if (parent->kv_cache->pool) {
        /* Same pool, own page table: the whole point of paging. */
        s = vv_kv_cache_create_paged(&c->kv_cache, parent->kv_cache->pool,
                                     parent->kv_cache->max_seq_len);
    } else {
        s = vv_kv_cache_create_range(&c->kv_cache, llm->num_hidden_layers,
                                     0, c->primary_layers,
                                     llm->num_key_value_heads, llm->head_dim,
                                     max_seq, p.kv_format, false);
    }
    if (s != VV_OK) goto fail;
    c->kv_cache->attn_backend = parent->kv_cache->attn_backend;

    c->workspace_size = parent->workspace_size;
    s = vv_dev_alloc(&c->workspace, c->workspace_size);
    if (s != VV_OK) {
        c->workspace_size = (size_t)256 * 1024 * 1024;
        s = vv_dev_alloc(&c->workspace, c->workspace_size);
    }
    if (s != VV_OK) goto fail;

    /*
     * A sharded parent hands out its layer layout and its pools; what a
     * clone needs of its own on each device is a KV cache, a workspace, a
     * pair of streams and somewhere for the hidden state to land.
     */
    if (parent->n_shards > 0) {
        c->shards = (vv_shard_t*)vv_alloc(
            (size_t)parent->n_shards * sizeof(vv_shard_t));
        if (!c->shards) { s = VV_ERR_OUT_OF_MEMORY; goto fail; }
        memset(c->shards, 0, (size_t)parent->n_shards * sizeof(vv_shard_t));
        s = vv_dev_event_create(&c->shard_done);
        if (s != VV_OK) goto fail;

        for (int i = 0; i < parent->n_shards; i++) {
            const vv_shard_t* ps = &parent->shards[i];
            vv_shard_t* sh = &c->shards[i];
            sh->gpu_id      = ps->gpu_id;
            sh->first_layer = ps->first_layer;
            sh->n_layers    = ps->n_layers;
            sh->n_resident  = ps->n_resident;
            sh->layer_pool  = ps->layer_pool;   /* all resident: stateless */

            s = vv_dev_set_device(sh->gpu_id);
            if (s == VV_OK) s = vv_dev_stream_create(&sh->compute_stream);
            if (s == VV_OK) s = vv_dev_stream_create(&sh->transfer_stream);
            if (s == VV_OK) s = vv_dev_event_create(&sh->done);
            if (s == VV_OK)
                s = vv_kv_cache_create_range(&sh->kv_cache,
                                             llm->num_hidden_layers,
                                             sh->first_layer, sh->n_layers,
                                             llm->num_key_value_heads,
                                             llm->head_dim, max_seq,
                                             p.kv_format, false);
            if (s == VV_OK) {
                sh->kv_cache->attn_backend = ps->kv_cache->attn_backend;
                sh->workspace_size = ps->workspace_size;
                s = vv_dev_alloc(&sh->workspace, sh->workspace_size);
            }
            if (s != VV_OK) {
                sh->layer_pool = NULL;
                shard_free(sh);
                vv_dev_set_device(c->gpu_id);
                goto fail;
            }
            c->n_shards++;
        }
        vv_dev_set_device(c->gpu_id);
    }

    attach_frontend(c);
    c->drafter = parent->drafter;
    attach_spec(c, NULL, 0, parent->draft_block, parent->draft_check, parent);

    if (c->kv_cache->pool)
        VV_LOG_I("inference: cloned context (workspace=%zu MB, kv shared "
                 "from the pool)", c->workspace_size / (1024 * 1024));
    else
        VV_LOG_I("inference: cloned context (workspace=%zu MB, kv=%.0f MB)",
                 c->workspace_size / (1024 * 1024),
                 (double)c->kv_cache->bytes_total / (1024.0 * 1024.0));
    *out = c;
    return VV_OK;

fail:
    for (int i = 0; i < c->n_shards; i++) {
        c->shards[i].layer_pool = NULL;
        shard_free(&c->shards[i]);
    }
    if (c->shards) vv_free(c->shards);
    if (c->shard_done) vv_dev_event_destroy(c->shard_done);
    vv_dev_set_device(c->gpu_id);
    if (c->kv_cache) vv_kv_cache_free(c->kv_cache);
    if (c->workspace) vv_dev_free(c->workspace);
    vv_dev_stream_destroy(c->compute_stream);
    vv_dev_stream_destroy(c->transfer_stream);
    vv_free(c);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Debug tensor dumping
 *
 * Set VV_DUMP_DIR to a directory and every intermediate below is written as a
 * raw little-endian binary blob, so tools/compare_ref.py can diff them against
 * the PyTorch reference dump element by element.
 * ═══════════════════════════════════════════════════════════════════════════ */


/* VV_ARGMAX_PARTIALS: pipeline_internal.h, shared with the streaming loop. */

static const char* s_dump_dir = NULL;
static vv_once_t s_dump_once = VV_ONCE_INIT;
static void dump_dir_probe(void) { s_dump_dir = getenv("VV_DUMP_DIR"); }

static const char* dump_dir(void) {
    vv_once(&s_dump_once, dump_dir_probe);
    return (s_dump_dir && s_dump_dir[0]) ? s_dump_dir : NULL;
}

/**
 * @brief Whether to time the decode phases separately.
 *
 * Splitting a token into embed / 28 layers / LM head means draining the stream
 * between them, which is two extra pipeline stalls per token and costs real
 * throughput. That is a price worth paying while deciding what to optimise and
 * not otherwise, so the split is opt-in and the default reports the total.
 */
static bool s_profile_decode = false;
static vv_once_t s_profile_once = VV_ONCE_INIT;
static void profile_decode_probe(void) {
    const char* e = getenv("VV_PROFILE_DECODE");
    s_profile_decode = e && e[0] && e[0] != '0';
}

static bool profile_decode(void) {
    vv_once(&s_profile_once, profile_decode_probe);
    return s_profile_decode;
}

static void dump_raw(const char* name, const void* data, size_t bytes) {
    const char* d = dump_dir();
    if (!d || !data || bytes == 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", d, name);
    FILE* f = fopen(path, "wb");
    if (!f) { VV_LOG_W("dump: cannot write %s", path); return; }
    fwrite(data, 1, bytes, f);
    fclose(f);
    VV_LOG_D("dump: wrote %s (%zu bytes)", path, bytes);
}

static void dump_f32(const char* name, const float* v, size_t n) {
    dump_raw(name, v, n * sizeof(float));
}

static void dump_i32(const char* name, const int32_t* v, size_t n) {
    dump_raw(name, v, n * sizeof(int32_t));
}

static void dump_f16_as_f32(const char* name, const uint16_t* v, size_t n) {
    if (!dump_dir() || !v || n == 0) return;
    float* tmp = (float*)vv_alloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < n; i++) tmp[i] = half_to_float_single(v[i]);
    dump_f32(name, tmp, n);
    vv_free(tmp);
}

/**
 * @brief Join hotwords into the processor's `context_info` string.
 *
 * The reference processor embeds it as "with extra info: <...>"; passing the
 * comma-joined hotword list there is how the model is steered towards rare
 * names in the upstream demo.
 */
/**
 * @brief Draw the acoustic latent, if the caller asked for a draw.
 *
 * The reference samples once over the whole clip, after the streaming segments
 * are concatenated, so this sits between the encoder and the connector rather
 * than inside the segment loop. A seed of 0 means "pick one and say which",
 * which keeps an unseeded run reproducible after the fact.
 */
static void sample_acoustic(vv_inference_ctx_t* ctx,
                            const vv_inference_params_t* params,
                            float* latents, int n_frames) {
    if (!latents || n_frames <= 0) return;

    vv_acoustic_sampling_t mode = params
        ? (vv_acoustic_sampling_t)params->acoustic_sampling
        : VV_ACOUSTIC_MODE;
    if (mode == VV_ACOUSTIC_MODE) return;

    uint64_t seed = params ? params->acoustic_seed : 0;
    if (seed == 0) seed = (uint64_t)(vv_time_ms() * 1000.0);

    vv_acoustic_sample(latents, n_frames,
                       ctx->model->config.acoustic.vae_dim,
                       ctx->model->config.acoustic.fix_std,
                       mode, seed, NULL);
}

size_t vv_hotwords_join(const char* const* words, int n, char* buf,
                        size_t cap) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    size_t w = 0;
    for (int i = 0; words && i < n; i++) {
        const char* hw = words[i];
        if (!hw || !hw[0]) continue;
        /* A whole word or none: a cut one would be a different hotword. */
        const int k = snprintf(buf + w, cap - w, "%s%s", w ? ", " : "", hw);
        if (k < 0 || (size_t)k >= cap - w) { buf[w] = '\0'; break; }
        w += (size_t)k;
    }
    return w;
}

size_t vv_hotwords_join_csv(const char* csv, char* buf, size_t cap) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    if (!csv) return 0;
    char tmp[VV_HOTWORDS_MAX];
    const char* words[VV_HOTWORDS_MAX / 2];
    snprintf(tmp, sizeof(tmp), "%s", csv);
    int n = 0;
    for (char* t = tmp; t && n < (int)(sizeof(words) / sizeof(words[0])); ) {
        char* next = strchr(t, ',');
        if (next) *next++ = '\0';
        while (*t == ' ') t++;
        size_t len = strlen(t);
        while (len > 0 && t[len - 1] == ' ') t[--len] = '\0';
        if (*t) words[n++] = t;
        t = next;
    }
    return vv_hotwords_join(words, n, buf, cap);
}

static void build_context_info(const vv_inference_params_t* params,
                               char* buf, size_t buf_size) {
    buf[0] = '\0';
    if (!params || !params->hotwords || params->num_hotwords <= 0) return;
    vv_hotwords_join(params->hotwords, params->num_hotwords, buf, buf_size);
}

/**
 * @brief Whether to replay decode steps from a captured graph.
 *
 * On by default; VV_CUDA_GRAPH=0 launches every kernel instead, which is how
 * the two are compared.
 */
static bool s_graph_on = true;
static vv_once_t s_graph_once = VV_ONCE_INIT;
static void graph_probe(void) {
    const char* e = getenv("VV_CUDA_GRAPH");
    s_graph_on = !(e && e[0] == '0');
}

static bool decode_graph_enabled(void) {
    vv_once(&s_graph_once, graph_probe);
    return s_graph_on;
}

/**
 * @brief One decode step, replayed from a capture when that is possible.
 *
 * The 28 layers are about 450 kernel launches and consecutive kernels on a
 * stream cannot overlap, so each pays a dispatch gap; submitting the lot as
 * one graph takes roughly 300 us off an 8 ms token here. That only works
 * because nothing in the step carries the position as a kernel argument any
 * more — see the device-side length on vv_kv_cache_t.
 *
 * What does change is the decode attention's launch shape, which steps up
 * every 1024 cached positions. `shape` tracks it and the capture is redone
 * when it moves: two dozen times over a 24K decode, against thousands of
 * replays.
 *
 * Capture issues no work, so the host-side length the step advances has to be
 * put back afterwards; every token, captured or not, advances it exactly once.
 *
 * @param shape  The launch shape the current capture was made for, or
 *               VV_GRAPH_GAVE_UP once a capture has failed and this request
 *               has stopped trying. Start it at -1.
 */
#define VV_GRAPH_GAVE_UP (-2)

/** @brief One captured step, and the launch shape it was captured for. */
typedef vv_graph_slot_t graph_slot_t;

/**
 * @brief One shard's slice of a decode step, replayed when it can be.
 *
 * Every argument that used to come from the context is passed instead,
 * because with a sharded model each device has its own streams, workspace,
 * KV cache and captured graph, and the loop below walks them in turn.
 */
static vv_status_t step_slice(vv_model_t* model, void* hidden,
                              vv_kv_cache_t* kv, vv_layer_pool_t* pool,
                              void* ws, size_t ws_size,
                              void* compute, void* xfer,
                              int first_layer, int n_layers,
                              bool graph_ok, graph_slot_t* g,
                              const vv_taps_t* taps)
{
    #define VV_STEP_DIRECT()                                                 \
        vv_decoder_step_taps(model, hidden, kv, pool, ws, ws_size,           \
                             compute, xfer, first_layer, n_layers, taps)

    /*
     * The capacity check lives in vv_kv_cache_append, and a replayed step
     * never runs it: past the window a replay would write K and V past the
     * end of the cache and attend over memory that is not the cache. Every
     * path, direct or replayed, on every shard, stops here instead.
     */
    if (kv->current_len >= kv->max_seq_len) return VV_ERR_OVERFLOW;

    /*
     * A paged cache maps its pages here, outside any capture: the step
     * writes at the device-side position and reads the table from device
     * memory, so a replay needs only this token's page to be mapped. Admission
     * reserved the request's expected length up front; past that, pages come
     * one at a time (never a look-ahead, which would sit on pages another
     * slot needs), and a pool that is short waits for another slot to finish
     * rather than ending this transcript early.
     */
    if (kv->pool) {
        const vv_status_t rs = vv_kv_cache_reserve_wait(
            kv, kv->current_len + 1, compute);
        if (rs != VV_OK) return rs;
    }

    if (!graph_ok || g->shape == VV_GRAPH_GAVE_UP) return VV_STEP_DIRECT();

    const vv_llm_config_t* llm = &model->config.llm;
    const int want = vv_attn_decode_shape(kv->attn_backend,
                                          llm->num_attention_heads,
                                          llm->num_key_value_heads,
                                          kv->current_len + 1);
    if (want != g->shape) {
        if (g->exec) { vv_dev_graph_destroy(g->exec); g->exec = NULL; }

        const int len = kv->current_len;
        vv_status_t cs = vv_dev_graph_begin(compute);
        if (cs == VV_OK) {
            void* got = NULL;
            cs = VV_STEP_DIRECT();
            const vv_status_t es = vv_dev_graph_end(compute, &got);
            if (cs == VV_OK && es == VV_OK && got) g->exec = got;
            else if (got)                          vv_dev_graph_destroy(got);
        }
        kv->current_len = len;

        if (!g->exec) {
            VV_LOG_W("decode: cannot capture the step, launching each kernel");
            g->shape = VV_GRAPH_GAVE_UP;
            return VV_STEP_DIRECT();
        }
        VV_LOG_D("decode: captured the step at %d cached positions", len);
        g->shape = want;
    }

    const vv_status_t s = vv_dev_graph_launch(g->exec, compute);
    if (s == VV_OK) kv->current_len++;
    return s;
    #undef VV_STEP_DIRECT
}

/**
 * @brief A whole decode step: the primary's layers, then each shard's.
 *
 * `graphs` holds one slot for the primary and one per shard. With no shards
 * this is exactly the single captured step it was before.
 */
static vv_status_t decoder_step_graphed_taps(vv_inference_ctx_t* ctx,
                                             void* hidden, bool graph_ok,
                                             graph_slot_t* graphs,
                                             const vv_taps_t* taps)
{
    /* Taps are for a speculative context, which is never sharded. */
    if (taps && ctx->n_shards > 0) return VV_ERR_UNSUPPORTED;
    vv_status_t s = step_slice(ctx->model, hidden, ctx->kv_cache,
                               ctx->layer_pool, ctx->workspace,
                               ctx->workspace_size, ctx->compute_stream,
                               ctx->transfer_stream, 0, ctx->primary_layers,
                               graph_ok, &graphs[0], taps);
    if (s != VV_OK || ctx->n_shards == 0) return s;

    const size_t bytes = (size_t)ctx->model->config.llm.hidden_size * 2;
    int src_gpu = ctx->gpu_id;
    void *src = hidden, *src_stream = ctx->compute_stream,
         *src_done = ctx->shard_done;

    for (int i = 0; i < ctx->n_shards; i++) {
        vv_shard_t* sh = &ctx->shards[i];
        s = hand_off(sh->gpu_id, sh->hidden, sh->compute_stream,
                     src_gpu, src, src_stream, src_done, bytes);
        if (s == VV_OK)
            s = step_slice(ctx->model, sh->hidden, sh->kv_cache,
                           sh->layer_pool, sh->workspace, sh->workspace_size,
                           sh->compute_stream, sh->transfer_stream,
                           sh->first_layer, sh->n_layers,
                           graph_ok, &graphs[i + 1], NULL);
        if (s != VV_OK) { vv_dev_set_device(ctx->gpu_id); return s; }
        src_gpu = sh->gpu_id; src = sh->hidden;
        src_stream = sh->compute_stream; src_done = sh->done;
    }

    return hand_off(ctx->gpu_id, hidden, ctx->compute_stream,
                    src_gpu, src, src_stream, src_done, bytes);
}

static vv_status_t decoder_step_graphed(vv_inference_ctx_t* ctx, void* hidden,
                                        bool graph_ok, graph_slot_t* graphs)
{
    return decoder_step_graphed_taps(ctx, hidden, graph_ok, graphs, NULL);
}

/**
 * @brief Whether a one-shot transcription stops at `token_id`.
 *
 * The ids were resolved once with the family, so this is two compares, not
 * two linear scans of the tokenizer's special tokens per generated token.
 */
static bool token_ends(const vv_inference_ctx_t* ctx, int32_t token_id) {
    return vv_family_stop(&ctx->family, token_id) != VV_STOP_CONTINUE;
}

/* ─── Teacher forcing (accuracy measurement) ────────────────────────────── */

/**
 * @brief A reference token sequence to decode against.
 *
 * With VV_TEACHER_TOKENS=<file> (token ids, whitespace-separated, as
 * VV_SAVE_TOKENS writes them) every decode step still computes its own
 * greedy token, which is compared with the reference and then replaced by
 * it, so the model always continues from the reference prefix. The share of
 * steps whose own token matched is the teacher-forced top-1 agreement: a
 * per-token measure of how far a quantized model is from the one that wrote
 * the reference, free of the divergence a single early flip causes in free
 * decoding. The step after the last reference token is scored on whether
 * the model, too, stops there. Debugging aid for one request at a time on
 * the GPU path: the CPU path ignores both variables (and says so), and
 * concurrent slots would all write the same VV_SAVE_TOKENS file.
 */
typedef struct teacher {
    int32_t* ids;
    int      n;
    int      pos;       /**< reference tokens consumed                     */
    int      agree;     /**< own greedy token == reference                  */
    int      scored;
} teacher_t;

static void teacher_load(teacher_t* t) {
    memset(t, 0, sizeof(*t));
#ifndef VV_TEACHER_FORCING
    /*
     * Off unless the build asked for it (-DVV_TEACHER_FORCING=ON). The file
     * this reads replaces the tokens the model would have produced, and a
     * server hands those to a client -- an environment variable that rewrites
     * answers has no business in a release binary, and CodeQL is right to
     * call it an exposure (cpp/system-data-exposure). Measuring agreement is
     * a developer's job on a developer's build.
     */
    if (getenv("VV_TEACHER_TOKENS"))
        VV_LOG_W("teacher: VV_TEACHER_TOKENS is set but this build has no "
                 "teacher forcing; configure with -DVV_TEACHER_FORCING=ON");
    return;
#else
    const char* path = getenv("VV_TEACHER_TOKENS");
    if (!path || !path[0]) return;
    FILE* f = fopen(path, "rb");
    if (!f) { VV_LOG_W("teacher: cannot read '%s'", path); return; }
    int cap = 4096;
    t->ids = (int32_t*)vv_alloc((size_t)cap * sizeof(int32_t));
    long v;
    while (t->ids && fscanf(f, "%ld", &v) == 1) {
        if (t->n == cap) {
            int32_t* grown = (int32_t*)vv_realloc(
                t->ids, (size_t)cap * 2 * sizeof(int32_t));
            if (!grown) { vv_free(t->ids); t->ids = NULL; break; }
            t->ids = grown;
            cap *= 2;
        }
        t->ids[t->n++] = (int32_t)v;
    }
    fclose(f);
    if (!t->ids) { t->n = 0; return; }
    VV_LOG_I("teacher: forcing %d reference tokens from '%s'", t->n, path);
#endif
}

/**
 * @brief Score the model's own `*token` and swap in the reference one.
 * @return true when the reference is exhausted (the decode stops).
 */
static bool teacher_step(const vv_inference_ctx_t* ctx, teacher_t* t,
                         int32_t* token) {
    if (!t->ids) return false;
    t->scored++;
    if (t->pos >= t->n) {                  /* the reference stopped here */
        t->agree += token_ends(ctx, *token);
        return true;
    }
    t->agree += *token == t->ids[t->pos];
    *token = t->ids[t->pos++];
    return false;
}

static void teacher_report(teacher_t* t) {
    if (t->ids && t->scored > 0)
        VV_LOG_I("teacher: top-1 agreement %d / %d = %.2f%%", t->agree,
                 t->scored, 100.0 * t->agree / t->scored);
    vv_free(t->ids);
    t->ids = NULL;
}

/** @brief VV_SAVE_TOKENS=<file>: the generated ids, for VV_TEACHER_TOKENS. */
static void save_tokens(const int32_t* ids, int n) {
    const char* path = getenv("VV_SAVE_TOKENS");
    if (!path || !path[0]) return;
    FILE* f = fopen(path, "wb");
    if (!f) { VV_LOG_W("save tokens: cannot write '%s'", path); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", (int)ids[i]);
    fclose(f);
}

/* ─── Host-side embedding, final norm and head ──────────────────────────────
 *
 * Most models: an FP16 table, FP16 norm weights, an FP16 head. asr-bitnet
 * from its GGUF pair embeds from the Q6_K table the reference looks rows up
 * in, keeps its final norm in FP32 and computes the head in the reference's
 * order, or from int8 rows when --head int8 asked for them.
 */

static vv_status_t embed_host(const vv_model_t* m, const int32_t* ids, int n,
                              float* out) {
    const int hs = m->config.llm.hidden_size;
    const int V = m->config.llm.vocab_size;
    if (!m->embed_q6k.data)
        return vv_embedding_cpu(m->embed_tokens.data, ids, out, n, hs);
    const size_t row = (size_t)m->embed_q6k.shape[1];
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= V) return VV_ERR_INVALID_ARG;
        vv_q6k_dequant_row((const uint8_t*)m->embed_q6k.data + (size_t)ids[i] * row,
                           out + (size_t)i * hs, hs);
    }
    return VV_OK;
}

static void final_norm_host(const vv_model_t* m, const float* x, float* y) {
    const vv_llm_config_t* c = &m->config.llm;
    if (m->final_norm.dtype == VV_DTYPE_F32)
        vv_bitnet_rmsnorm_cpu(x, (const float*)m->final_norm.data, y, 1,
                              c->hidden_size, c->rms_norm_eps);
    else
        vv_rmsnorm_cpu(x, m->final_norm.data, y, 1, c->hidden_size,
                       c->rms_norm_eps);
}

static vv_status_t head_host(const vv_model_t* m, const float* x,
                             void* scratch, size_t scratch_bytes,
                             int32_t* token) {
    const vv_llm_config_t* c = &m->config.llm;
    const int hs = c->hidden_size, V = c->vocab_size;
    if (m->head_bound.data && m->lm_head.data && m->head_i8.data) {
        /* the F16 head's own argmax, read through its int8 filter */
        float v = 0.0f;
        int n = 0;
        vv_status_t s = vv_bitnet_head_filtered_argmax_cpu(
            x, (const uint16_t*)m->lm_head.data, (const int8_t*)m->head_i8.data,
            (const float*)m->head_i8_scale.data, (const float*)m->head_bound.data,
            V, hs, scratch, scratch_bytes, token, &v, &n);
        /* the logit, to diff against VibeASR.cpp's (bit for bit so far) */
        VV_LOG_D("head: token %d logit %.6f (%d rows scored in F16)",
                 (int)*token, (double)v, n);
        return s;
    }
    if (m->head_i8.data) {
        int8_t q[8192];
        float sc;
        if (hs > 8192) return VV_ERR_UNSUPPORTED;
        vv_status_t s = vv_act_quant_i8_cpu(x, 1, hs, q, &sc, NULL);
        if (s != VV_OK) return s;
        return vv_i8_head_argmax_cpu(q, sc, (const int8_t*)m->head_i8.data,
                                     (const float*)m->head_i8_scale.data, V,
                                     hs, token, NULL);
    }
    if (m->config.family == VV_FAMILY_ASR_BITNET)
        return vv_bitnet_head_f16_argmax_cpu(x, (const uint16_t*)m->lm_head.data,
                                             V, hs, token, NULL);
    return vv_lm_head_argmax_cpu(x, m->lm_head.data, V, hs, token, NULL);
}

/**
 * @brief Greedy token from a head that stays on the host.
 *
 * Placements that keep the LM head in system memory land here once per
 * token: the normalized row comes down (7 KB), the fused FP16 GEMV + argmax
 * reads the head in place. `h` and `f` are one hidden row each, allocated
 * by the caller once per request.
 */
static vv_status_t cpu_head_argmax(vv_inference_ctx_t* ctx,
                                   const void* normed_gpu,
                                   uint16_t* h, float* f, int32_t* token) {
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    const int hs = llm->hidden_size;
    vv_status_t s = vv_dev_stream_sync(ctx->compute_stream);
    if (s == VV_OK)
        s = vv_dev_memcpy_d2h(h, normed_gpu, (size_t)hs * 2,
                              ctx->compute_stream);
    if (s != VV_OK) return s;
    for (int d = 0; d < hs; d++) f[d] = vv_half_to_float(h[d]);
    return vv_lm_head_argmax_cpu(f, ctx->model->lm_head.data,
                                 llm->vocab_size, hs, token, NULL);
}

vv_status_t vv_pipeline_head_argmax(vv_inference_ctx_t* ctx, const void* row,
                                    void* normed, void* logits, void* am_v,
                                    void* am_i, void* tok_dev,
                                    int32_t* tok_host, uint16_t* h, float* f,
                                    int32_t* token) {
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    vv_status_t s = vv_rmsnorm_dev(row, ctx->final_norm_gpu, normed, 1,
                                   llm->hidden_size, llm->rms_norm_eps,
                                   ctx->compute_stream);
    if (s != VV_OK) return s;
    if (!ctx->lm_head_gpu) return cpu_head_argmax(ctx, normed, h, f, token);
    s = vv_lm_head_gemv_dev(normed, ctx->lm_head_gpu, logits, llm->vocab_size,
                            llm->hidden_size, ctx->compute_stream);
    if (s == VV_OK)
        s = vv_argmax_dev(logits, llm->vocab_size, am_v, am_i, tok_dev, NULL,
                          ctx->compute_stream);
    /* Into pinned memory the copy is asynchronous: wait for it, not just
     * for the argmax before it. */
    if (s == VV_OK)
        s = vv_dev_memcpy_d2h(tok_host, tok_dev, sizeof(int32_t),
                              ctx->compute_stream);
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    if (s == VV_OK) *token = *tok_host;
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Speech front end — GPU path
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Latents through the host: the acoustic draw, and tensor dumps.
 *
 * The draw is taken once over the whole clip, after the segments are
 * concatenated, which is where the reference takes it; so the latents have
 * to exist in full before the connectors run. Off the default path. The
 * latents and the FP32 staging live in a per-context buffer that only grows
 * (a longer clip than any before), so in serve this mode does not cudaFree
 * -- and stall the other slots -- per request. The dump path (VV_DUMP_DIR,
 * debugging only) still allocates its connector scratch per call.
 */
static vv_status_t encode_audio_host(vv_inference_ctx_t* ctx,
                                     const vv_inference_params_t* params,
                                     const float* audio, int n_samples,
                                     int frames, void* rows) {
    const int hs = ctx->model->config.llm.hidden_size;
    const int vd[2] = { ctx->model->config.acoustic.vae_dim,
                        ctx->model->config.semantic.vae_dim };
    const char* names[2] = { "c_ac_mean", "c_sem_mean" };
    void* lat[2] = { NULL, NULL };
    void* f32 = NULL;
    uint16_t* h16 = NULL;
    float* h32 = NULL;
    vv_status_t s = VV_OK;
    const size_t fr = (size_t)(frames > 0 ? frames : 1);
    const size_t maxv = (size_t)(vd[0] > vd[1] ? vd[0] : vd[1]);

    {
        const size_t al = 256;
        const size_t b0 = (fr * (size_t)vd[0] * 2 + al - 1) / al * al;
        const size_t b1 = (fr * (size_t)vd[1] * 2 + al - 1) / al * al;
        const size_t need = b0 + b1 + fr * maxv * sizeof(float);
        if (ctx->fe_lat_bytes < need) {
            if (ctx->fe_lat_buf) {
                vv_dev_stream_sync(ctx->compute_stream);
                vv_dev_free(ctx->fe_lat_buf);
            }
            ctx->fe_lat_buf = NULL;
            ctx->fe_lat_bytes = 0;
            s = vv_dev_alloc(&ctx->fe_lat_buf, need);
            if (s == VV_OK) ctx->fe_lat_bytes = need;
        }
        if (s == VV_OK) {
            lat[0] = ctx->fe_lat_buf;
            lat[1] = (char*)ctx->fe_lat_buf + b0;
            f32 = (char*)ctx->fe_lat_buf + b0 + b1;
        }
    }
    h16 = (uint16_t*)vv_alloc(fr * maxv * 2);
    h32 = (float*)vv_alloc(fr * maxv * sizeof(float));
    if (s == VV_OK && (!h16 || !h32)) s = VV_ERR_OUT_OF_MEMORY;

    if (s == VV_OK) {
        vv_frontend_job_t job;
        memset(&job, 0, sizeof(job));
        vv_frontend_stream_reset(ctx->fe_stream);
        job.audio = audio;
        job.n_samples = n_samples;
        job.stream = ctx->fe_stream;
        job.is_final = true;
        job.ac_latents = lat[0];
        job.sem_latents = lat[1];
        job.done_event = ctx->fe_done;
        s = vv_frontend_submit(ctx->frontend, &job);
        if (s == VV_OK) s = vv_dev_event_sync(ctx->fe_done);
        if (s == VV_OK && job.n_frames != frames) s = VV_ERR_SHAPE_MISMATCH;
    }

    for (int e = 0; e < 2 && s == VV_OK; e++) {
        const size_t n = (size_t)frames * (size_t)vd[e];
        s = vv_dev_memcpy_d2h(h16, lat[e], n * 2, ctx->compute_stream);
        if (s != VV_OK) break;
        for (size_t i = 0; i < n; i++) h32[i] = vv_half_to_float(h16[i]);
        dump_f32(names[e], h32, n);
        if (e != 0 || !params ||
            params->acoustic_sampling == VV_ACOUSTIC_MODE)
            continue;
        sample_acoustic(ctx, params, h32, frames);
        /* Back to FP16 the way the old path did it: on the device. */
        s = vv_dev_memcpy_h2d(f32, h32, n * sizeof(float), ctx->compute_stream);
        if (s == VV_OK)
            s = vv_vae_f32_to_f16_dev((const float*)f32, lat[0], (int64_t)n,
                                      ctx->compute_stream);
        if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    }

    if (s == VV_OK && dump_dir()) {
        /* Each connector alone, for tools/compare_ref.py. */
        const char* fn[2] = { "c_ac_feat", "c_sem_feat" };
        void* tmp = NULL;
        uint16_t* th = (uint16_t*)vv_alloc(fr * (size_t)hs * 2);
        float* tf = (float*)vv_alloc(fr * (size_t)hs * sizeof(float));
        if (th && tf && vv_dev_alloc(&tmp, fr * (size_t)hs * 2) == VV_OK) {
            for (int e = 0; e < 2; e++) {
                vv_dev_memset_async(tmp, 0, fr * (size_t)hs * 2,
                                    ctx->compute_stream);
                vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);
                if (vv_frontend_connect(ctx->frontend, e == 0 ? lat[0] : NULL,
                                        e == 1 ? lat[1] : NULL, frames, tmp, hs,
                                        ctx->fe_ready, ctx->fe_done) != VV_OK)
                    continue;
                vv_dev_event_sync(ctx->fe_done);
                vv_dev_memcpy_d2h(th, tmp, (size_t)frames * hs * 2,
                                  ctx->compute_stream);
                for (size_t i = 0; i < (size_t)frames * hs; i++)
                    tf[i] = vv_half_to_float(th[i]);
                dump_f32(fn[e], tf, (size_t)frames * hs);
            }
            vv_dev_free(tmp);
        }
        vv_free(th);
        vv_free(tf);
        /* The prompt's rows must still wait for its own embedding. */
        vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);
    }

    if (s == VV_OK)
        s = vv_frontend_connect(ctx->frontend, lat[0], lat[1], frames, rows, hs,
                                ctx->fe_ready, ctx->fe_done);
    if (s == VV_OK) s = vv_dev_event_sync(ctx->fe_done);

    vv_free(h16);
    vv_free(h32);
    return s;
}

/**
 * @brief Encode the clip into the prompt's speech rows, asynchronously.
 *
 * On return the work is enqueued and ctx->fe_done will fire when the rows
 * are written. Row writes wait for ctx->fe_ready (the embedding).
 */
static vv_status_t encode_audio_gpu(vv_inference_ctx_t* ctx,
                                    const vv_inference_params_t* params,
                                    const float* audio, int n_samples,
                                    int frames, void* rows) {
    const bool sampling = params &&
                          params->acoustic_sampling != VV_ACOUSTIC_MODE;
    if (sampling || dump_dir())
        return encode_audio_host(ctx, params, audio, n_samples, frames, rows);

    vv_frontend_job_t job;
    memset(&job, 0, sizeof(job));
    vv_frontend_stream_reset(ctx->fe_stream);
    job.audio = audio;
    job.n_samples = n_samples;
    job.stream = ctx->fe_stream;
    job.is_final = true;
    job.rows = rows;
    job.rows_ld = ctx->model->config.llm.hidden_size;
    job.wait_event = ctx->fe_ready;
    job.done_event = ctx->fe_done;
    vv_status_t s = vv_frontend_submit(ctx->frontend, &job);
    if (s == VV_OK && job.n_frames != frames) {
        VV_LOG_E("inference: %d speech frames written, the prompt has %d",
                 job.n_frames, frames);
        s = VV_ERR_SHAPE_MISMATCH;
    }
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — GPU path
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Positions a request reserves before it starts: the prompt plus the decode
 * it is expected to run. Transcripts come to about 5 tokens per second of
 * audio (jfk 4.5, test120 5.0, the 32-minute file 5.0); 8 a second plus a
 * little for the JSON framing leaves room for dense speech. A request that
 * outruns it takes further pages one at a time.
 */
#define VV_KV_ADMIT_TOKENS_PER_SEC 8
#define VV_KV_ADMIT_BASE           256

static int kv_admit_positions(const vv_kv_cache_t* kv, int prompt_len,
                              float audio_sec, int max_new_tokens) {
    double decode = (double)audio_sec * VV_KV_ADMIT_TOKENS_PER_SEC
                  + VV_KV_ADMIT_BASE;
    if (decode > (double)max_new_tokens) decode = (double)max_new_tokens;
    double want = (double)prompt_len + decode;
    if (want > (double)kv->max_seq_len) want = (double)kv->max_seq_len;
    return (int)want;
}

static vv_status_t transcribe_gpu(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    vv_perf_metrics_t* perf = &ctx->last_perf;
    double t_total_start = vv_time_ms();
    vv_status_t s;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    int hs = llm->hidden_size;
    int vocab_size = llm->vocab_size;
    int max_new_tokens = params ? params->max_new_tokens : 64000;
    if (max_new_tokens <= 0) max_new_tokens = 64000;

    perf->audio_duration_sec = (float)num_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = hs;
    perf->kv_format = ctx->kv_cache ? ctx->kv_cache->format : VV_KV_FP16;
    perf->attn_backend = ctx->kv_cache ? ctx->kv_cache->attn_backend
                                       : VV_ATTN_FA1;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    VV_LOG_I("inference: GPU transcribe %d samples (%.2f sec)",
             num_samples, perf->audio_duration_sec);

    if (!ctx->final_norm_gpu || !ctx->tokenizer) {
        VV_LOG_E("inference: missing critical components");
        return VV_ERR_NULL_PTR;
    }

    /* Check: need either GPU embed or CPU fallback */
    bool embed_on_cpu = (ctx->embed_table_gpu == NULL);
    bool lm_head_on_cpu = (ctx->lm_head_gpu == NULL);

    vv_kv_cache_reset(ctx->kv_cache, ctx->compute_stream);
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_dev_set_device(ctx->shards[i].gpu_id);
        vv_kv_cache_reset(ctx->shards[i].kv_cache,
                          ctx->shards[i].compute_stream);
    }
    vv_dev_set_device(ctx->gpu_id);

    /* ═══ STEP 1: Prompt ═══
     *
     * The frame count is a function of the sample count alone, so the
     * prompt and its embedding exist before a sample is encoded, and the
     * front end writes the audio rows straight into them: no latents or
     * features on the host, no combined buffer, no overlay copy.
     */
    double t_step = vv_time_ms();
    dump_f32("c_audio24k", audio_samples, (size_t)num_samples);

    const vv_i8vae_t* i8vae = ctx->model->i8vae;
    if (!i8vae && (!ctx->frontend || !ctx->fe_stream)) {
        VV_LOG_E("inference: no speech encoder on the device");
        return VV_ERR_WEIGHT_MISSING;
    }
    const int n_audio_frames = i8vae ? vv_i8vae_frames(i8vae, num_samples)
                                     : vv_frontend_frames(ctx->frontend, num_samples);
    perf->audio_frames = n_audio_frames;

    int32_t* input_ids = NULL;
    int seq_len = 0;
    int audio_offset = 0;  /* index where box_start tokens (= audio frames) begin */

    char ctx_info[VV_HOTWORDS_MAX];
    build_context_info(params, ctx_info, sizeof(ctx_info));

    {
        vv_prompt_t pr;
        s = vv_family_build_prompt(&ctx->family, ctx->tokenizer,
                                   n_audio_frames, perf->audio_duration_sec,
                                   ctx_info, &pr);
        if (s != VV_OK) return s;
        input_ids = pr.ids;
        seq_len = pr.n;
        audio_offset = pr.audio_offset;
    }

    /*
     * Admission. A paged cache shares its pool with the other slots on this
     * device, and the pool may be smaller than all their windows together.
     * Take the pages this request is expected to need before any work is
     * done on the device, waiting here while other requests hold them, so
     * that a request rarely has to wait halfway through its decode (and
     * never gets cut short: see step_slice).
     */
    if (ctx->kv_cache && ctx->kv_cache->pool) {
        const int admit = kv_admit_positions(ctx->kv_cache, seq_len,
                                             perf->audio_duration_sec,
                                             max_new_tokens);
        const double t_admit = vv_time_ms();
        s = vv_kv_cache_reserve_wait(ctx->kv_cache, admit,
                                     ctx->compute_stream);
        const double waited = vv_time_ms() - t_admit;
        if (waited > 50.0)
            VV_LOG_I("inference: waited %.0f ms for %d KV positions",
                     waited, admit);
        if (s != VV_OK) {
            VV_LOG_E("inference: no KV pages for this request: %s",
                     vv_status_str(s));
            vv_free(input_ids);
            return s;
        }
    }

    perf->prefill_tokens = seq_len;
    dump_i32("c_input_ids", input_ids, (size_t)seq_len);

    size_t hidden_bytes = (size_t)seq_len * (size_t)hs * 2;
    void* hidden_states_gpu = NULL;
    int32_t* input_ids_gpu = NULL;

    s = vv_dev_alloc(&hidden_states_gpu, hidden_bytes);
    if (s != VV_OK) { vv_free(input_ids); return s; }

    /* Embedding: all tokens (text + speech_pad placeholders) */
    if (embed_on_cpu) {
        float* embed_fp32 = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
        if (!embed_fp32) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); return VV_ERR_OUT_OF_MEMORY; }
        vv_embedding_cpu(ctx->model->embed_tokens.data, input_ids, embed_fp32, seq_len, hs);
        uint16_t* embed_fp16 = (uint16_t*)vv_alloc(hidden_bytes);
        if (!embed_fp16) { vv_free(embed_fp32); vv_dev_free(hidden_states_gpu); vv_free(input_ids); return VV_ERR_OUT_OF_MEMORY; }
        float_to_half(embed_fp32, embed_fp16, seq_len * hs);
        vv_free(embed_fp32);
        vv_dev_memcpy_h2d(hidden_states_gpu, embed_fp16, hidden_bytes, ctx->compute_stream);
        vv_free(embed_fp16);
    } else {
        s = vv_dev_alloc((void**)&input_ids_gpu, (size_t)seq_len * sizeof(int32_t));
        if (s != VV_OK) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); return s; }
        vv_dev_memcpy_h2d(input_ids_gpu, input_ids, (size_t)seq_len * sizeof(int32_t), ctx->compute_stream);
        s = vv_embedding_dev(ctx->embed_table_gpu, input_ids_gpu, hidden_states_gpu, seq_len, hs, ctx->compute_stream);
        vv_dev_free(input_ids_gpu);
        if (s != VV_OK) { vv_dev_free(hidden_states_gpu); vv_free(input_ids); return s; }
    }
    vv_free(input_ids);
    /* The front end's row writes wait for this, not for the host. */
    if (ctx->fe_ready) vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);
    perf->sequence_build_ms = vv_time_ms() - t_step;

    /* ═══ STEP 2: Audio → the prompt's speech rows ═══ */
    t_step = vv_time_ms();
    VV_LOG_I("inference: step 1 — encoding audio");
    if (i8vae) {
        /*
         * VibeASR.cpp's int8 encoder runs on the host (it is not streamable:
         * one scale per whole tensor), and its summed features go into the
         * prompt's rows as FP16.
         */
        void* audio_rows = (uint8_t*)hidden_states_gpu
                         + (size_t)audio_offset * (size_t)hs * 2;
        float* feat = (float*)vv_alloc((size_t)(n_audio_frames > 0 ? n_audio_frames : 1)
                                       * hs * sizeof(float));
        uint16_t* h16 = (uint16_t*)vv_alloc((size_t)(n_audio_frames > 0 ? n_audio_frames : 1)
                                            * hs * 2);
        int nf = 0;
        s = (feat && h16) ? VV_OK : VV_ERR_OUT_OF_MEMORY;
        if (s == VV_OK)
            s = vv_i8vae_encode(i8vae, audio_samples, num_samples,
                                VV_I8VAE_WINDOW_SAMPLES, feat, &nf);
        if (s == VV_OK && nf != n_audio_frames) s = VV_ERR_SHAPE_MISMATCH;
        if (s == VV_OK) {
            for (size_t i = 0; i < (size_t)nf * hs; i++)
                h16[i] = vv_float_to_half_rne(feat[i]);
            s = vv_dev_memcpy_h2d(audio_rows, h16, (size_t)nf * hs * 2,
                                  ctx->compute_stream);
        }
        if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
        vv_free(feat);
        vv_free(h16);
        if (s != VV_OK) {
            VV_LOG_E("inference: int8 speech encoder failed: %s", vv_status_str(s));
            vv_dev_free(hidden_states_gpu);
            return s;
        }
    } else {
        void* audio_rows = (uint8_t*)hidden_states_gpu
                         + (size_t)audio_offset * (size_t)hs * 2;
        s = encode_audio_gpu(ctx, params, audio_samples, num_samples,
                             n_audio_frames, audio_rows);
        if (s == VV_OK) s = vv_dev_stream_wait_event(ctx->compute_stream, ctx->fe_done);
        /* Only to time it: prefill cannot start before this anyway. */
        if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
        if (s != VV_OK) {
            VV_LOG_E("inference: speech encoder failed: %s", vv_status_str(s));
            vv_dev_free(hidden_states_gpu);
            return s;
        }
    }
    perf->audio_encode_ms = vv_time_ms() - t_step;
    VV_LOG_D("inference: step 1 complete — audio encoded + connectors done (%.0f ms)",
             perf->audio_encode_ms);

    /* ── Diagnostic: compare text-embedding vs audio-feature magnitudes ── */
    if (vv_log_get_level() >= VV_LOG_DEBUG) {
        const int sample_dim = (hs < 32) ? hs : 32;  /* sample first 32 dims */
        uint16_t diag_buf[32];
        float sum_sq;

        vv_dev_memcpy_d2h(diag_buf, hidden_states_gpu,
                            (size_t)sample_dim * 2, ctx->compute_stream);
        sum_sq = 0.0f;
        for (int d = 0; d < sample_dim; d++) {
            float v = half_to_float_single(diag_buf[d]);
            sum_sq += v * v;
        }
        VV_LOG_D("diag: text embed[0] L2(first %d dims) = %.6f", sample_dim, sqrtf(sum_sq));

        if (n_audio_frames > 0) {
            void* audio_pos = (uint8_t*)hidden_states_gpu
                            + (size_t)audio_offset * (size_t)hs * 2;
            vv_dev_memcpy_d2h(diag_buf, audio_pos,
                                (size_t)sample_dim * 2, ctx->compute_stream);
            sum_sq = 0.0f;
            for (int d = 0; d < sample_dim; d++) {
                float v = half_to_float_single(diag_buf[d]);
                sum_sq += v * v;
            }
            VV_LOG_D("diag: audio feat[0] L2(first %d dims) = %.6f", sample_dim, sqrtf(sum_sq));
        }
    }

    if (dump_dir()) {
        size_t n = (size_t)seq_len * (size_t)hs;
        uint16_t* h = (uint16_t*)vv_alloc(n * 2);
        if (h) {
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(h, hidden_states_gpu, n * 2,
                              ctx->compute_stream);
            dump_f16_as_f32("c_embeds", h, n);
            {
                const size_t off = (size_t)audio_offset * (size_t)hs;
                dump_f16_as_f32("c_combined", h + off,
                                (size_t)n_audio_frames * (size_t)hs);
            }
            vv_free(h);
        }
    }

    /* ═══ STEP 3: LLM Prefill ═══ */
    t_step = vv_time_ms();
    VV_LOG_I("inference: prefill %d tokens, %d layers", seq_len, ctx->model->num_layers);

    /*
     * With a drafter the prompt's taps become its context as each prefill
     * chunk finishes. Not while forcing tokens (the reference decides) or
     * splitting the decode into timed phases.
     */
    const bool spec_on = ctx->spec && !lm_head_on_cpu && !embed_on_cpu &&
                         !profile_decode();
    if (spec_on) {
        vv_spec_reset(ctx->spec);
        ctx->taps = vv_spec_prefill_taps(ctx->spec);
    }
    s = prefill_all_shards(ctx, hidden_states_gpu, seq_len);
    ctx->taps = NULL;
    if (s != VV_OK) {
        VV_LOG_E("inference: prefill failed: %s", vv_status_str(s));
        vv_dev_free(hidden_states_gpu);
        return s;
    }

    if (dump_dir()) {
        size_t n = (size_t)seq_len * (size_t)hs;
        uint16_t* h = (uint16_t*)vv_alloc(n * 2);
        if (h) {
            vv_dev_stream_sync(ctx->compute_stream);
            vv_dev_memcpy_d2h(h, hidden_states_gpu, n * 2,
                              ctx->compute_stream);
            dump_f16_as_f32("c_prefill_hidden", h, n);
            vv_free(h);
        }
    }

    /* RMSNorm → lm_head → first token */
    size_t one_hidden = (size_t)hs * 2;
    void* last_hidden_gpu = (uint8_t*)hidden_states_gpu + (size_t)(seq_len-1) * one_hidden;

    void* normed_gpu = NULL;
    void* logits_gpu = NULL;
    void* hidden_one_gpu = NULL;
    /*
     * Everything the decode loop touches is allocated once here: cudaMalloc
     * serialises against the stream, so even a 4-byte allocation per token
     * would cost more than the layer it feeds.
     */
    void* logits_f32_gpu = NULL;
    void* argmax_v_gpu   = NULL;
    void* argmax_i_gpu   = NULL;
    void* token_out_gpu  = NULL;
    /* Host side of a head that stayed on the CPU: one hidden row each. */
    uint16_t* host_normed_h = NULL;
    float*    host_normed_f = NULL;
    s = vv_dev_alloc(&normed_gpu, one_hidden);
    if (s != VV_OK) { vv_dev_free(hidden_states_gpu); return s; }
    s = vv_dev_alloc(&hidden_one_gpu, one_hidden);
    if (s != VV_OK) { vv_dev_free(normed_gpu); vv_dev_free(hidden_states_gpu); return s; }
    /* Holds the current token: the argmax writes it, the embedding reads it. */
    if (vv_dev_alloc(&token_out_gpu, sizeof(int32_t)) != VV_OK) {
        s = VV_ERR_CUDA_OOM;
        goto cleanup_decode;
    }
    if (lm_head_on_cpu) {
        host_normed_h = (uint16_t*)vv_alloc(one_hidden);
        host_normed_f = (float*)vv_alloc((size_t)hs * sizeof(float));
        if (!host_normed_h || !host_normed_f || !ctx->model->lm_head.data) {
            s = ctx->model->lm_head.data ? VV_ERR_OUT_OF_MEMORY
                                         : VV_ERR_WEIGHT_MISSING;
            goto cleanup_decode;
        }
    } else {
        if (vv_dev_alloc(&logits_f32_gpu, (size_t)vocab_size * sizeof(float)) != VV_OK ||
            vv_dev_alloc(&argmax_v_gpu, VV_ARGMAX_PARTIALS * sizeof(float)) != VV_OK ||
            vv_dev_alloc(&argmax_i_gpu, VV_ARGMAX_PARTIALS * sizeof(int32_t)) != VV_OK) {
            s = VV_ERR_CUDA_OOM;
            goto cleanup_decode;
        }
    }

    s = vv_rmsnorm_dev(last_hidden_gpu, ctx->final_norm_gpu,
                         normed_gpu, 1, hs, llm->rms_norm_eps,
                         ctx->compute_stream);
    if (s != VV_OK) goto cleanup_decode;

    /* LM head — GPU or CPU */
    int32_t token_id;
    if (lm_head_on_cpu) {
        /*
         * The head stays where it is, FP16 on the host, and the fused CPU
         * GEMV + argmax reads it once. This used to widen the whole head to
         * FP32 first — 2.2 GB for the 7B — and then walk it single-threaded.
         */
        s = cpu_head_argmax(ctx, normed_gpu, host_normed_h, host_normed_f,
                            &token_id);
        if (s != VV_OK) goto cleanup_decode;
    } else {
        s = vv_lm_head_gemv_dev(normed_gpu, ctx->lm_head_gpu, logits_f32_gpu,
                                  vocab_size, hs, ctx->compute_stream);
        if (s != VV_OK) goto cleanup_decode;
        s = vv_argmax_dev(logits_f32_gpu, vocab_size, argmax_v_gpu,
                            argmax_i_gpu, token_out_gpu, NULL,
                            ctx->compute_stream);
        if (s != VV_OK) goto cleanup_decode;
        vv_dev_stream_sync(ctx->compute_stream);
        vv_dev_memcpy_d2h(&token_id, token_out_gpu, sizeof(int32_t),
                          ctx->compute_stream);
        if (dump_dir()) {
            float* lg = (float*)vv_alloc((size_t)vocab_size * sizeof(float));
            if (lg) {
                vv_dev_memcpy_d2h(lg, logits_f32_gpu,
                                    (size_t)vocab_size * sizeof(float),
                                    ctx->compute_stream);
                dump_f32("c_prefill_logits", lg, (size_t)vocab_size);
                vv_free(lg);
            }
        }
    }

    vv_dev_free(hidden_states_gpu); hidden_states_gpu = NULL;

    perf->prefill_ms = vv_time_ms() - t_step;
    perf->ttft_ms = vv_time_ms() - t_total_start;
    (void)0;
    perf->prefill_tok_per_sec = (perf->prefill_ms > 0.001)
        ? (double)seq_len / perf->prefill_ms * 1000.0 : 0.0;
    VV_LOG_I("inference: prefill %.1f ms (%.0f tok/s), TTFT=%.1f ms",
             perf->prefill_ms, perf->prefill_tok_per_sec, perf->ttft_ms);

    /* ═══ STEP 4: Autoregressive decode ═══ */
    t_step = vv_time_ms();
    int out_cap = max_new_tokens < 8192 ? max_new_tokens : 8192;
    int32_t* output_tokens = (int32_t*)vv_alloc((size_t)out_cap * sizeof(int32_t));
    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
    int n_generated = 0;
    s = VV_OK;          /* the loop below leaves it set only on failure */

    /* Split the decode cost so the next optimisation targets the real hot
     * spot: 28 transformer layers vs the 152k-row LM head + sampling. */
    double t_layers_ms = 0.0, t_head_ms = 0.0, t_embed_ms = 0.0;
    const bool prof = profile_decode();

    /*
     * Prefill advanced the length on the host; from here the kernels read it
     * on the device, so hand it over.
     */
    vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_dev_set_device(ctx->shards[i].gpu_id);
        vv_kv_cache_publish_len(ctx->shards[i].kv_cache,
                                ctx->shards[i].compute_stream);
    }
    vv_dev_set_device(ctx->gpu_id);

    /*
     * Replaying the step needs every layer resident (a streaming pool patches
     * host pointers between layers, which no graph can record) and the
     * device-side length that makes the step position-invariant. Per-phase
     * timing wants a drained stream between phases, which a single submission
     * does not give, so the two are mutually exclusive.
     */
    graph_slot_t graphs[VV_MAX_GPUS];
    for (int i = 0; i <= ctx->n_shards; i++) { graphs[i].exec = NULL;
                                               graphs[i].shape = -1; }
    const bool graph_ok = decode_graph_enabled() &&
                          ctx->layer_pool && ctx->layer_pool->all_resident &&
                          ctx->kv_cache && ctx->kv_cache->d_len && !prof;

    /*
     * Live token echo. Worth watching during a long transcription, pure
     * noise inside the chat and mic loops, so it follows the log level.
     */
    const bool echo_tokens = !ctx->quiet &&
                             vv_log_get_level() >= VV_LOG_INFO;
    if (echo_tokens) fprintf(stderr, "\n--- token stream ---\n");
    teacher_t teacher;
    teacher_load(&teacher);
    bool teacher_done = teacher_step(ctx, &teacher, &token_id);
    if (teacher.ids && !teacher_done)      /* the embedding reads it there */
        vv_dev_memcpy_h2d(token_out_gpu, &token_id, sizeof(int32_t),
                          ctx->compute_stream);
    if (!teacher_done && !token_ends(ctx, token_id)) {
        output_tokens[n_generated++] = token_id;
        /* Stream first token */
        char* first_text = NULL;
        vv_tokenizer_decode(ctx->tokenizer, &token_id, 1, &first_text);
        if (first_text) {
            if (echo_tokens) fprintf(stderr, "%s", first_text);
            fflush(stderr);
            vv_free(first_text);
        }
    }

    /*
     * Speculative decoding: each cycle drafts a block and keeps what the
     * model agrees with (spec.h). It stops at a stop token, at the token
     * cap, or when a block no longer fits the window -- the plain steps
     * below then take the last few positions.
     */
    if (spec_on && !teacher.ids) {
        vv_spec_stats_reset(ctx->spec);
        /* Plain steps between blocks tap their row for the drafter: their
         * captures are not the untapped step's (graphs, below). */
        graph_slot_t graphs_taps[VV_MAX_GPUS];
        for (int i = 0; i < VV_MAX_GPUS; i++) { graphs_taps[i].exec = NULL;
                                                graphs_taps[i].shape = -1; }
        /* Whether the device holds the length and the token a step reads:
         * a step leaves them there for the next, a block does not. */
        bool dev_fresh = false;
        while (n_generated < max_new_tokens && !token_ends(ctx, token_id)) {
            int32_t outs[VV_SPEC_MAX_BLOCK];
            int n_out = 0;
            const double t0 = vv_time_ms();
            if (vv_spec_want_block(ctx->spec)) {
                s = vv_spec_cycle(ctx, ctx->spec, token_id, outs, &n_out);
                if (s == VV_ERR_KV_POOL_EXHAUSTED) {
                    /* No pages for a whole block: plain steps for a while. */
                    s = VV_OK;
                    vv_spec_pause(ctx->spec);
                    continue;
                }
                if (s == VV_ERR_OVERFLOW) { s = VV_OK; break; }
                if (s != VV_OK) {
                    VV_LOG_E("inference: speculative step at %d failed: %s",
                             n_generated, vv_status_str(s));
                    break;
                }
                dev_fresh = false;
                vv_spec_note_block(ctx->spec, n_out, vv_time_ms() - t0);
            } else {
                /* A plain step, seen by the drafter as well. It reads the
                 * length and the token on the device. */
                if (!dev_fresh) {
                    vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
                    s = vv_dev_memcpy_h2d(token_out_gpu, &token_id,
                                          sizeof(int32_t), ctx->compute_stream);
                }
                if (s == VV_OK)
                    s = vv_embedding_dev(ctx->embed_table_gpu,
                                         (const int32_t*)token_out_gpu,
                                         hidden_one_gpu, 1, hs,
                                         ctx->compute_stream);
                if (s == VV_OK)
                    s = decoder_step_graphed_taps(ctx, hidden_one_gpu, graph_ok,
                                                  graphs_taps,
                                                  vv_spec_step_taps(ctx->spec));
                if (s == VV_ERR_OVERFLOW) { s = VV_OK; break; }
                if (s == VV_OK) {
                    int32_t tok_host = 0;
                    s = vv_pipeline_head_argmax(ctx, hidden_one_gpu, normed_gpu,
                                                logits_f32_gpu, argmax_v_gpu,
                                                argmax_i_gpu, token_out_gpu,
                                                &tok_host, host_normed_h,
                                                host_normed_f, &outs[0]);
                }
                if (s == VV_OK) s = vv_spec_push_step(ctx->spec);
                if (s != VV_OK) break;
                dev_fresh = true;
                n_out = 1;
                vv_spec_note_step(ctx->spec, vv_time_ms() - t0);
            }
            for (int i = 0; i < n_out; i++) {
                token_id = outs[i];
                if (token_ends(ctx, token_id)) break;
                if (n_generated >= out_cap) {
                    out_cap *= 2;
                    output_tokens = (int32_t*)vv_realloc(output_tokens,
                        (size_t)out_cap * sizeof(int32_t));
                    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
                }
                output_tokens[n_generated++] = token_id;
                if (echo_tokens) {
                    char* tok_text = NULL;
                    vv_tokenizer_decode(ctx->tokenizer, &token_id, 1, &tok_text);
                    if (tok_text) {
                        fprintf(stderr, "%s", tok_text);
                        fflush(stderr);
                        vv_free(tok_text);
                    }
                }
                if (n_generated >= max_new_tokens) break;
            }
            if (s != VV_OK) break;
        }
        for (int i = 0; i < VV_MAX_GPUS; i++)
            if (graphs_taps[i].exec) vv_dev_graph_destroy(graphs_taps[i].exec);
        {
            const vv_spec_stats_t* st = vv_spec_get_stats(ctx->spec);
            if (st && st->cycles > 0) {
                /* A captured cycle has no drafting time of its own. */
                char split[48] = "";
                if (st->draft_ms > 0.0)
                    snprintf(split, sizeof(split), ", drafting %.0f ms",
                             st->draft_ms);
                VV_LOG_I("spec: %lld cycles, %.2f tokens per cycle, %lld of "
                         "%lld drafts kept, %lld plain steps, %lld pauses "
                         "(blocks %.0f ms%s)",
                         (long long)st->cycles,
                         (double)st->tokens / (double)st->cycles,
                         (long long)st->accepted, (long long)st->drafted,
                         (long long)st->steps, (long long)st->fallbacks,
                         st->draft_ms + st->verify_ms, split);
            }
        }
        /* The plain steps read the length and the token on the device. */
        if (s == VV_OK && n_generated < max_new_tokens &&
            !token_ends(ctx, token_id)) {
            vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
            vv_dev_memcpy_h2d(token_out_gpu, &token_id, sizeof(int32_t),
                              ctx->compute_stream);
        }
    }

    while (s == VV_OK && n_generated < max_new_tokens && !teacher_done &&
           !token_ends(ctx, token_id)) {

        double t_tok = vv_time_ms();

        /* Embed token */
        if (embed_on_cpu) {
            /* One FP16 row of the table is exactly the embedding. */
            if (token_id < 0 || token_id >= vocab_size) {
                s = VV_ERR_INVALID_ARG;
                break;
            }
            s = vv_dev_memcpy_h2d(hidden_one_gpu,
                                  (const uint8_t*)ctx->model->embed_tokens.data
                                  + (size_t)token_id * one_hidden,
                                  one_hidden, ctx->compute_stream);
            if (s != VV_OK) break;
        } else {
            /*
             * The token to embed is the one the previous step's argmax left in
             * `token_out_gpu`, so it never has to travel. Copying it up from
             * the host was not just a transfer: a pageable H2D drains the
             * stream first, which is a stall per token for four bytes that
             * were already on the card.
             */
            if (lm_head_on_cpu) {
                /* Sampled on the host, so this one does have to travel. */
                vv_dev_memcpy_h2d(token_out_gpu, &token_id, sizeof(int32_t),
                                  ctx->compute_stream);
            }
            s = vv_embedding_dev(ctx->embed_table_gpu,
                                   (const int32_t*)token_out_gpu,
                                   hidden_one_gpu, 1, hs, ctx->compute_stream);
            if (s != VV_OK) break;
        }

        if (prof) {
            vv_dev_stream_sync(ctx->compute_stream);
            t_embed_ms += vv_time_ms() - t_tok;
            t_tok = vv_time_ms();
        }

        /* Decoder step */
        s = decoder_step_graphed(ctx, hidden_one_gpu, graph_ok, graphs);
        if (s == VV_ERR_OVERFLOW) {
            /* This request's own window, and only that: a shared pool that
             * runs dry reports VV_ERR_KV_POOL_EXHAUSTED and fails below. */
            VV_LOG_W("inference: the KV window is full (%d positions); the "
                     "transcript stops here -- raise --max-seq-len",
                     ctx->kv_cache->max_seq_len);
            s = VV_OK;
            break;
        }
        if (s != VV_OK) {
            VV_LOG_E("inference: decode step %d failed: %s", n_generated,
                     vv_status_str(s));
            break;
        }

        if (prof) {
            vv_dev_stream_sync(ctx->compute_stream);
            t_layers_ms += vv_time_ms() - t_tok;
            t_tok = vv_time_ms();
        }

        /* RMSNorm + LM head + sample */
        {
            int32_t tok_host = 0;
            s = vv_pipeline_head_argmax(ctx, hidden_one_gpu, normed_gpu,
                                        logits_f32_gpu, argmax_v_gpu,
                                        argmax_i_gpu, token_out_gpu, &tok_host,
                                        host_normed_h, host_normed_f,
                                        &token_id);
            if (s != VV_OK) break;
        }

        if (prof) t_head_ms += vv_time_ms() - t_tok;

        if (teacher.ids) {
            if (teacher_step(ctx, &teacher, &token_id)) break;
            vv_dev_memcpy_h2d(token_out_gpu, &token_id, sizeof(int32_t),
                              ctx->compute_stream);
        }
        if (token_ends(ctx, token_id)) break;

        if (n_generated >= out_cap) {
            out_cap *= 2;
            output_tokens = (int32_t*)vv_realloc(output_tokens,
                (size_t)out_cap * sizeof(int32_t));
            if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
        }
        output_tokens[n_generated++] = token_id;

        /*
         * Stream the token text for live monitoring. Only when somebody is
         * watching: detokenizing here costs an allocation and a second pass
         * over the merges for text that post-processing decodes again anyway,
         * and the flush is a write syscall per token.
         */
        if (echo_tokens) {
            char* tok_text = NULL;
            vv_tokenizer_decode(ctx->tokenizer, &token_id, 1, &tok_text);
            if (tok_text) {
                fprintf(stderr, "%s", tok_text);
                fflush(stderr);
                vv_free(tok_text);
            }
        }

        if ((n_generated % 500) == 0) {
            double elapsed = vv_time_ms() - t_step;
            VV_LOG_I("inference: decoded %d tokens (%.1f tok/s)...",
                     n_generated, (elapsed > 0.001) ? (double)n_generated / elapsed * 1000.0 : 0.0);
        }
    }
    if (echo_tokens)
        fprintf(stderr, "\n--- end stream (%d tokens) ---\n",
                n_generated);
    fflush(stderr);
    teacher_report(&teacher);
    if (s == VV_OK) save_tokens(output_tokens, n_generated);

    for (int i = 0; i <= ctx->n_shards; i++)
        if (graphs[i].exec) vv_dev_graph_destroy(graphs[i].exec);

    /*
     * A step that failed fails the request. Post-processing what came out
     * so far would hand back a truncated transcript as a success.
     */
    if (s != VV_OK) {
        vv_free(output_tokens);
        goto cleanup_decode;
    }

    /* Zero unless VV_PROFILE_DECODE asked for the extra stalls. */
    perf->decode_layers_ms = t_layers_ms;
    perf->decode_head_ms   = t_head_ms;
    perf->decode_embed_ms  = t_embed_ms;
    perf->decode_ms = vv_time_ms() - t_step;
    perf->decode_tokens = n_generated;
    perf->decode_tok_per_sec = (perf->decode_ms > 0.001)
        ? (double)n_generated / perf->decode_ms * 1000.0 : 0.0;

    /* ═══ STEP 5: Post-processing ═══ */
    t_step = vv_time_ms();
    const char** token_texts = (const char**)vv_alloc((size_t)n_generated * sizeof(char*));
    if (!token_texts) { vv_free(output_tokens); s = VV_ERR_OUT_OF_MEMORY; goto cleanup_decode; }
    for (int i = 0; i < n_generated; i++) {
        char* decoded = NULL;
        vv_tokenizer_decode(ctx->tokenizer, &output_tokens[i], 1, &decoded);
        token_texts[i] = decoded ? decoded : "";
    }
    s = vv_postprocess_tokens(token_texts, n_generated, result);
    if (s == VV_OK && *result) (*result)->duration = perf->audio_duration_sec;
    if (s == VV_OK && *result) {
        const int32_t stop = token_ends(ctx, token_id) ? token_id : -1;
        s = vv_transcription_set_tokens(*result, output_tokens, n_generated,
                                        &n_generated, &stop, 1);
        if (s != VV_OK) { vv_transcription_free(*result); *result = NULL; }
    }
    for (int i = 0; i < n_generated; i++)
        if (token_texts[i] && token_texts[i][0] != '\0') vv_free((void*)token_texts[i]);
    vv_free((void*)token_texts);
    vv_free(output_tokens);
    perf->postprocess_ms = vv_time_ms() - t_step;

    /* Finalize metrics */
    perf->total_ms = vv_time_ms() - t_total_start;
    perf->rtf = (perf->audio_duration_sec > 0.001)
        ? (perf->total_ms / 1000.0) / perf->audio_duration_sec : 0.0;
    if (ctx->kv_cache) {
        perf->kv_cache_used = ctx->kv_cache->current_len;
        perf->kv_cache_max  = ctx->kv_cache->max_seq_len;
        perf->kv_cache_pct  = (perf->kv_cache_max > 0)
            ? 100.0f * (float)perf->kv_cache_used / (float)perf->kv_cache_max : 0.0f;
    }
    {
        size_t vfree = 0, vtotal = 0;
        if (vv_dev_get_device_info(ctx->gpu_id, &vtotal, &vfree, NULL) == VV_OK) {
            perf->vram_total_bytes = vtotal;
            perf->vram_free_bytes  = vfree;
            perf->vram_used_bytes  = vtotal - vfree;
        }
    }

cleanup_decode:
    if (hidden_states_gpu) vv_dev_free(hidden_states_gpu);
    if (normed_gpu) vv_dev_free(normed_gpu);
    if (logits_gpu) vv_dev_free(logits_gpu);
    if (hidden_one_gpu) vv_dev_free(hidden_one_gpu);
    if (logits_f32_gpu) vv_dev_free(logits_f32_gpu);
    if (argmax_v_gpu) vv_dev_free(argmax_v_gpu);
    if (argmax_i_gpu) vv_dev_free(argmax_i_gpu);
    if (token_out_gpu) vv_dev_free(token_out_gpu);
    vv_free(host_normed_h);
    vv_free(host_normed_f);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — CPU path
 * ═══════════════════════════════════════════════════════════════════════════ */

static vv_status_t transcribe_cpu(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    vv_perf_metrics_t* perf = &ctx->last_perf;
    double t_total_start = vv_time_ms();
    vv_status_t s;
    const vv_llm_config_t* llm = &ctx->model->config.llm;
    int hs = llm->hidden_size;
    int max_new_tokens = params ? params->max_new_tokens : 64000;
    if (max_new_tokens <= 0) max_new_tokens = 64000;

    perf->audio_duration_sec = (float)num_samples / 24000.0f;
    perf->num_layers = ctx->model->num_layers;
    perf->hidden_size = hs;
    perf->kv_format = ctx->kv_cache ? ctx->kv_cache->format : VV_KV_FP32;
    perf->workspace_mb = ctx->workspace_size / (1024 * 1024);

    VV_LOG_I("inference: CPU transcribe %d samples (%.2f sec)",
             num_samples, perf->audio_duration_sec);
    {
        const char* sv = getenv("VV_SAVE_TOKENS");
        const char* te = getenv("VV_TEACHER_TOKENS");
        if ((sv && sv[0]) || (te && te[0]))
            VV_LOG_W("inference: VV_SAVE_TOKENS / VV_TEACHER_TOKENS work on "
                     "the GPU path only; ignored here");
    }

    if (!ctx->tokenizer) { VV_LOG_E("inference: no tokenizer"); return VV_ERR_NULL_PTR; }
    vv_kv_cache_reset(ctx->kv_cache, ctx->compute_stream);

    /* ═══ STEP 1: Audio encoding (same as GPU path) ═══ */
    double t_step = vv_time_ms();
    float* acoustic_latents = NULL, *semantic_latents = NULL;
    float* acoustic_features = NULL, *semantic_features = NULL;
    int n_acoustic_frames = 0, n_semantic_frames = 0;

    if (ctx->model->i8vae) {
        /* VibeASR.cpp's int8 encoder: features come out already summed. */
        int nf = 0;
        const int cap = vv_i8vae_frames(ctx->model->i8vae, num_samples);
        float* feat = (float*)vv_alloc((size_t)(cap > 0 ? cap : 1) * hs * sizeof(float));
        if (!feat) return VV_ERR_OUT_OF_MEMORY;
        s = vv_i8vae_encode(ctx->model->i8vae, audio_samples, num_samples,
                            VV_I8VAE_WINDOW_SAMPLES, feat, &nf);
        if (s != VV_OK) { vv_free(feat); return s; }
        if (dump_dir()) {
            /* the towers one by one, for a diff against VibeASR.cpp's */
            float* t = (float*)vv_alloc((size_t)(cap > 0 ? cap : 1) * hs * sizeof(float));
            int tn = 0;
            if (t && vv_i8vae_encode_tower(ctx->model->i8vae, 0, audio_samples,
                                           num_samples, t, &tn) == VV_OK)
                dump_f32("c_i8_acoustic", t, (size_t)tn * hs);
            if (t && vv_i8vae_encode_tower(ctx->model->i8vae, 1, audio_samples,
                                           num_samples, t, &tn) == VV_OK)
                dump_f32("c_i8_semantic", t, (size_t)tn * hs);
            vv_free(t);
        }
        acoustic_features = feat;
        n_acoustic_frames = n_semantic_frames = nf;
    }
    if (ctx->acoustic_encoder) {
        vv_conv_vae_encode_cpu(ctx->acoustic_encoder, audio_samples, num_samples,
                                &acoustic_latents, &n_acoustic_frames);
    }
    if (ctx->semantic_encoder) {
        vv_conv_vae_encode_cpu(ctx->semantic_encoder, audio_samples, num_samples,
                                &semantic_latents, &n_semantic_frames);
    }

    int n_audio_frames;
    if (n_acoustic_frames > 0 && n_semantic_frames > 0)
        n_audio_frames = n_acoustic_frames < n_semantic_frames ? n_acoustic_frames : n_semantic_frames;
    else if (n_acoustic_frames > 0) n_audio_frames = n_acoustic_frames;
    else if (n_semantic_frames > 0) n_audio_frames = n_semantic_frames;
    else { n_audio_frames = num_samples / 3200; if (n_audio_frames < 1) n_audio_frames = 1; }
    perf->audio_frames = n_audio_frames;

    sample_acoustic(ctx, params, acoustic_latents, n_audio_frames);

    if (acoustic_latents && ctx->acoustic_connector)
        vv_connector_forward_cpu(ctx->acoustic_connector, acoustic_latents, n_audio_frames, &acoustic_features);
    if (semantic_latents && ctx->semantic_connector)
        vv_connector_forward_cpu(ctx->semantic_connector, semantic_latents, n_audio_frames, &semantic_features);
    if (acoustic_latents) vv_free(acoustic_latents);
    if (semantic_latents) vv_free(semantic_latents);

    int combined_elems = n_audio_frames * hs;
    float* combined = (float*)vv_alloc((size_t)combined_elems * sizeof(float));
    if (!combined) {
        if (acoustic_features) vv_free(acoustic_features);
        if (semantic_features) vv_free(semantic_features);
        return VV_ERR_OUT_OF_MEMORY;
    }
    if (acoustic_features && semantic_features)
        for (int i = 0; i < combined_elems; i++) combined[i] = acoustic_features[i] + semantic_features[i];
    else if (acoustic_features) memcpy(combined, acoustic_features, (size_t)combined_elems * sizeof(float));
    else if (semantic_features) memcpy(combined, semantic_features, (size_t)combined_elems * sizeof(float));
    else memset(combined, 0, (size_t)combined_elems * sizeof(float));
    if (acoustic_features) vv_free(acoustic_features);
    if (semantic_features) vv_free(semantic_features);
    dump_f32("c_speech_features", combined, (size_t)combined_elems);

    perf->audio_encode_ms = vv_time_ms() - t_step;

    /* ═══ STEP 2: Build ChatML input sequence (FP32) ═══ */
    t_step = vv_time_ms();
    int32_t* prompt_ids = NULL;
    int seq_len = 0;
    int audio_offset = 0;

    char ctx_info[VV_HOTWORDS_MAX];
    build_context_info(params, ctx_info, sizeof(ctx_info));

    {
        vv_prompt_t pr;
        s = vv_family_build_prompt(&ctx->family, ctx->tokenizer,
                                   n_audio_frames, perf->audio_duration_sec,
                                   ctx_info, &pr);
        if (s != VV_OK) { vv_free(combined); return s; }
        prompt_ids = pr.ids;
        seq_len = pr.n;
        audio_offset = pr.audio_offset;
    }
    perf->prefill_tokens = seq_len;
    dump_i32("c_prompt_ids", prompt_ids, (size_t)seq_len);

    /* Allocate FP32 hidden states on CPU */
    float* hidden = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
    if (!hidden) { vv_free(prompt_ids); vv_free(combined); return VV_ERR_OUT_OF_MEMORY; }

    /* Embed all prompt tokens */
    s = embed_host(ctx->model, prompt_ids, seq_len, hidden);
    vv_free(prompt_ids);
    if (s != VV_OK) { vv_free(hidden); vv_free(combined); return s; }

    /* Overlay audio features at speech_pad positions */
    if (n_audio_frames > 0) {
        memcpy(hidden + (size_t)audio_offset * hs,
               combined, (size_t)combined_elems * sizeof(float));
    }
    vv_free(combined);

    perf->sequence_build_ms = vv_time_ms() - t_step;

    /* ═══ STEP 3: Prefill ═══ */
    t_step = vv_time_ms();
    s = vv_decoder_prefill_cpu(ctx->model, hidden, seq_len, ctx->kv_cache,
                                (float*)ctx->workspace, ctx->workspace_size);
    if (s != VV_OK) { vv_free(hidden); return s; }
    if (ctx->model->config.family == VV_FAMILY_ASR_BITNET &&
        ctx->family.frame_samples > 0) {
        /*
         * VibeASR.cpp reserves ceil(n/3200) pads, feeds the floor(n/3200)
         * frames its convolutions produce and decodes from the reserved
         * length: generated tokens sit that many positions further on.
         */
        const int fs = ctx->family.frame_samples;
        const int reserved = (int)(((int64_t)num_samples + fs - 1) / fs);
        if (reserved > n_audio_frames)
            ctx->kv_cache->rope_gap = reserved - n_audio_frames;
    }

    /* Final norm + lm_head + sample (CPU FP32) */
    float* normed = (float*)vv_alloc((size_t)hs * sizeof(float));
    float* hidden_one = (float*)vv_alloc((size_t)hs * sizeof(float));
    float* logits = NULL;   /* the fused head never materialises them */
    if (!normed || !hidden_one) {
        if (normed) vv_free(normed);
        if (hidden_one) vv_free(hidden_one);
        vv_free(hidden);
        return VV_ERR_OUT_OF_MEMORY;
    }

    float* last_hidden = hidden + (size_t)(seq_len - 1) * hs;
    final_norm_host(ctx->model, last_hidden, normed);

    int32_t token_id = 0;
    s = head_host(ctx->model, normed, ctx->workspace, ctx->workspace_size, &token_id);
    vv_free(hidden); hidden = NULL;
    if (s != VV_OK) {
        VV_LOG_E("inference: CPU LM head failed: %s", vv_status_str(s));
        vv_free(normed);
        vv_free(hidden_one);
        return s;
    }

    perf->prefill_ms = vv_time_ms() - t_step;
    perf->ttft_ms = vv_time_ms() - t_total_start;
    perf->prefill_tok_per_sec = (perf->prefill_ms > 0.001)
        ? (double)seq_len / perf->prefill_ms * 1000.0 : 0.0;

    /* ═══ STEP 4: Decode loop (CPU) ═══ */
    t_step = vv_time_ms();
    int out_cap = max_new_tokens < 8192 ? max_new_tokens : 8192;
    int32_t* output_tokens = (int32_t*)vv_alloc((size_t)out_cap * sizeof(int32_t));
    if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; goto cpu_cleanup; }
    int n_generated = 0;
    if (!token_ends(ctx, token_id))
        output_tokens[n_generated++] = token_id;

    while (n_generated < max_new_tokens && !token_ends(ctx, token_id)) {
        /* Embed */
        s = embed_host(ctx->model, &token_id, 1, hidden_one);
        if (s != VV_OK) break;
        /* Decoder step */
        s = vv_decoder_step_cpu(ctx->model, hidden_one, ctx->kv_cache,
                                 (float*)ctx->workspace, ctx->workspace_size);
        if (s != VV_OK) break;

        /* Norm + LM head */
        final_norm_host(ctx->model, hidden_one, normed);
        s = head_host(ctx->model, normed, ctx->workspace, ctx->workspace_size, &token_id);
        if (s != VV_OK) break;
        if (token_ends(ctx, token_id)) break;

        if (n_generated >= out_cap) {
            out_cap *= 2;
            output_tokens = (int32_t*)vv_realloc(output_tokens, (size_t)out_cap * sizeof(int32_t));
            if (!output_tokens) { s = VV_ERR_OUT_OF_MEMORY; break; }
        }
        output_tokens[n_generated++] = token_id;

        if ((n_generated % 100) == 0) {
            double elapsed = vv_time_ms() - t_step;
            VV_LOG_I("inference: CPU decoded %d tokens (%.1f tok/s)...",
                     n_generated, (elapsed > 0.001) ? (double)n_generated / elapsed * 1000.0 : 0.0);
        }
    }

    perf->decode_ms = vv_time_ms() - t_step;
    perf->decode_tokens = n_generated;
    perf->decode_tok_per_sec = (perf->decode_ms > 0.001)
        ? (double)n_generated / perf->decode_ms * 1000.0 : 0.0;
    if (output_tokens) dump_i32("c_tokens", output_tokens, (size_t)n_generated);

    /* ═══ STEP 5: Post-processing ═══ */
    t_step = vv_time_ms();
    {
        const char** token_texts = (const char**)vv_alloc((size_t)n_generated * sizeof(char*));
        if (token_texts) {
            for (int i = 0; i < n_generated; i++) {
                char* decoded = NULL;
                vv_tokenizer_decode(ctx->tokenizer, &output_tokens[i], 1, &decoded);
                token_texts[i] = decoded ? decoded : "";
            }
            s = vv_postprocess_tokens(token_texts, n_generated, result);
            if (s == VV_OK && *result) (*result)->duration = perf->audio_duration_sec;
            if (s == VV_OK && *result) {
                const int32_t stop = token_ends(ctx, token_id) ? token_id : -1;
                s = vv_transcription_set_tokens(*result, output_tokens,
                                                n_generated, &n_generated,
                                                &stop, 1);
                if (s != VV_OK) {
                    vv_transcription_free(*result);
                    *result = NULL;
                }
            }
            for (int i = 0; i < n_generated; i++)
                if (token_texts[i] && token_texts[i][0] != '\0') vv_free((void*)token_texts[i]);
            vv_free((void*)token_texts);
        }
    }
    vv_free(output_tokens);
    perf->postprocess_ms = vv_time_ms() - t_step;

    perf->total_ms = vv_time_ms() - t_total_start;
    perf->rtf = (perf->audio_duration_sec > 0.001)
        ? (perf->total_ms / 1000.0) / perf->audio_duration_sec : 0.0;
    if (ctx->kv_cache) {
        perf->kv_cache_used = ctx->kv_cache->current_len;
        perf->kv_cache_max  = ctx->kv_cache->max_seq_len;
        perf->kv_cache_pct  = (perf->kv_cache_max > 0)
            ? 100.0f * (float)perf->kv_cache_used / (float)perf->kv_cache_max : 0.0f;
    }

cpu_cleanup:
    vv_free(normed);
    vv_free(logits);
    vv_free(hidden_one);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Trace — a transcription replayed as prefills, taps kept (spec.h)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * The prompt with its audio and then every generated token, all in one
 * prefill: the cache ends up as the transcription left it, and each row's
 * taps are the hidden states the drafter would have read there.
 */
static vv_status_t trace_oneshot_gpu(vv_inference_ctx_t* ctx,
                                     const float* audio, int n_samples,
                                     const char* context_info,
                                     const vv_transcription_t* tr,
                                     vv_spec_trace_t* t)
{
    const int hs = ctx->model->config.llm.hidden_size;
    const vv_i8vae_t* i8vae = ctx->model->i8vae;
    if (!i8vae && (!ctx->frontend || !ctx->fe_stream))
        return VV_ERR_WEIGHT_MISSING;
    if (!ctx->embed_table_gpu || !ctx->kv_cache) return VV_ERR_UNSUPPORTED;

    const int frames = i8vae ? vv_i8vae_frames(i8vae, n_samples)
                             : vv_frontend_frames(ctx->frontend, n_samples);
    vv_prompt_t pr;
    vv_status_t s = vv_family_build_prompt(&ctx->family, ctx->tokenizer,
                                           frames, (float)n_samples / 24000.0f,
                                           context_info, &pr);
    if (s != VV_OK) return s;

    const int n_gen = tr->num_tokens;
    const int32_t stop = tr->num_chunks > 0 && tr->chunk_stops
                       ? tr->chunk_stops[0] : -1;
    const int n_fed = pr.n + n_gen;
    t->n = n_fed + (stop >= 0 ? 1 : 0);
    if (t->n > t->cap || n_fed > t->taps.rows ||
        n_fed > ctx->kv_cache->max_seq_len) {
        vv_prompt_free(&pr);
        return VV_ERR_OVERFLOW;
    }
    for (int i = 0; i < pr.n; i++) {
        t->ids[i] = pr.ids[i];
        t->kind[i] = VV_TRACE_CONTEXT;
    }
    for (int i = 0; i < n_gen; i++) {
        t->ids[pr.n + i] = tr->tokens[i];
        t->kind[pr.n + i] = VV_TRACE_GEN;
    }
    if (stop >= 0) {
        t->ids[n_fed] = stop;
        t->kind[n_fed] = VV_TRACE_LABEL;
    }
    const int audio_offset = pr.audio_offset;
    vv_prompt_free(&pr);

    vv_pipeline_kv_reset(ctx);
    s = vv_pipeline_kv_reserve(ctx, n_fed);
    if (s != VV_OK) return s;

    void* hidden = NULL;
    int32_t* ids_dev = NULL;
    s = vv_dev_alloc(&hidden, (size_t)n_fed * hs * 2);
    if (s == VV_OK)
        s = vv_dev_alloc((void**)&ids_dev, (size_t)n_fed * sizeof(int32_t));
    if (s == VV_OK)
        s = vv_dev_memcpy_h2d(ids_dev, t->ids, (size_t)n_fed * sizeof(int32_t),
                              ctx->compute_stream);
    if (s == VV_OK)
        s = vv_embedding_dev(ctx->embed_table_gpu, ids_dev, hidden, n_fed, hs,
                             ctx->compute_stream);
    if (s == VV_OK && ctx->fe_ready)
        s = vv_dev_event_record(ctx->fe_ready, ctx->compute_stream);

    void* rows = (uint8_t*)hidden + (size_t)audio_offset * hs * 2;
    if (s == VV_OK && i8vae) {
        float* feat = (float*)vv_alloc((size_t)(frames > 0 ? frames : 1) * hs *
                                       sizeof(float));
        uint16_t* h16 = (uint16_t*)vv_alloc((size_t)(frames > 0 ? frames : 1) *
                                            hs * 2);
        int nf = 0;
        s = (feat && h16) ? VV_OK : VV_ERR_OUT_OF_MEMORY;
        if (s == VV_OK)
            s = vv_i8vae_encode(i8vae, audio, n_samples,
                                VV_I8VAE_WINDOW_SAMPLES, feat, &nf);
        if (s == VV_OK && nf != frames) s = VV_ERR_SHAPE_MISMATCH;
        if (s == VV_OK) {
            for (size_t i = 0; i < (size_t)nf * hs; i++)
                h16[i] = vv_float_to_half_rne(feat[i]);
            s = vv_dev_memcpy_h2d(rows, h16, (size_t)nf * hs * 2,
                                  ctx->compute_stream);
        }
        if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
        vv_free(feat);
        vv_free(h16);
    } else if (s == VV_OK) {
        s = encode_audio_gpu(ctx, NULL, audio, n_samples, frames, rows);
        if (s == VV_OK)
            s = vv_dev_stream_wait_event(ctx->compute_stream, ctx->fe_done);
    }

    if (s == VV_OK) {
        ctx->taps = &t->taps;
        s = prefill_all_shards(ctx, hidden, n_fed);
        ctx->taps = NULL;
    }
    if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    if (ids_dev) vv_dev_free(ids_dev);
    if (hidden) vv_dev_free(hidden);
    if (ctx->kv_cache->pool) {
        vv_dev_stream_sync(ctx->compute_stream);
        vv_kv_cache_release(ctx->kv_cache);
    }
    return s;
}

vv_status_t vv_spec_trace(vv_inference_ctx_t* ctx, const float* pcm24k,
                          int n_samples, const char* context_info,
                          const vv_transcription_t* tr, vv_spec_trace_t* t) {
    if (!ctx || !pcm24k || !tr || !t || !t->ids || !t->kind || !t->taps.buf)
        return VV_ERR_NULL_PTR;
    if (t->taps.on_chunk || t->taps.row_base != 0 || t->taps.n <= 0 ||
        t->taps.n > VV_TAPS_MAX)
        return VV_ERR_INVALID_ARG;
    if (tr->num_tokens > 0 && !tr->tokens) return VV_ERR_INVALID_ARG;
    t->n = 0;
    if (!ctx->tokenizer || !ctx->family_ok) return VV_ERR_MODEL_FORMAT;
    if (!ctx->use_gpu || ctx->placement == VV_PLACE_CPU_ONLY ||
        ctx->n_shards > 0 || !ctx->layer_pool ||
        !ctx->layer_pool->all_resident)
        return VV_ERR_UNSUPPORTED;
    for (int i = 0; i < t->taps.n; i++)
        if (t->taps.layers[i] < 0 ||
            t->taps.layers[i] >= ctx->model->num_layers)
            return VV_ERR_INVALID_ARG;
    vv_status_t s = vv_dev_set_device(ctx->gpu_id);
    if (s != VV_OK) return s;
    if (ctx->family.mode != VV_GEN_ONE_SHOT)
        return vv_stream_trace(ctx, pcm24k, n_samples, context_info, tr, t);
    return trace_oneshot_gpu(ctx, pcm24k, n_samples, context_info, tr, t);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transcribe — dispatcher
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result)
{
    if (!ctx || !audio_samples || !result) return VV_ERR_NULL_PTR;

    memset(&ctx->last_perf, 0, sizeof(ctx->last_perf));

    if (!ctx->tokenizer || !ctx->family_ok) {
        VV_LOG_E("inference: no usable tokenizer for this model");
        return VV_ERR_MODEL_FORMAT;
    }

    /*
     * CUDA's current device is per-thread and this context was created on
     * another one — the server hands each request to a worker from its
     * pool. Without this the worker runs on device 0, and every kernel
     * launched there with a pointer from `gpu_id` fails; it showed up as the
     * speech encoder silently falling back to the CPU.
     */
    if (ctx->use_gpu) {
        vv_status_t bind = vv_dev_set_device(ctx->gpu_id);
        if (bind != VV_OK) {
            VV_LOG_E("inference: cannot bind GPU %d on this thread: %s",
                     ctx->gpu_id, vv_status_str(bind));
            return bind;
        }
    }

    if (ctx->family.mode != VV_GEN_ONE_SHOT) {
        /*
         * The streaming model is prompted without audio and fed chunk by
         * chunk on one KV cache, so a whole clip goes through a session:
         * every window, then the tail, text joined at the end.
         */
        char ctx_info[VV_HOTWORDS_MAX];
        build_context_info(params, ctx_info, sizeof(ctx_info));
        return vv_stream_transcribe(ctx, audio_samples, num_samples,
                                    ctx_info[0] ? ctx_info : NULL, NULL,
                                    NULL, result);
    }

    if (ctx->placement == VV_PLACE_CPU_ONLY) {
        return transcribe_cpu(ctx, audio_samples, num_samples, params, result);
    } else {
        const vv_status_t s = transcribe_gpu(ctx, audio_samples, num_samples,
                                             params, result);
        /*
         * An idle slot must not sit on pages another slot could use. The
         * request is over, but its kernels may still be queued; once the
         * stream drains, nothing reads them.
         */
        if (ctx->kv_cache && ctx->kv_cache->pool) {
            vv_dev_stream_sync(ctx->compute_stream);
            vv_kv_cache_release(ctx->kv_cache);
        }
        return s;
    }
}

/* ─── Perf getter ───────────────────────────────────────────────────────── */

const vv_perf_metrics_t* vv_inference_get_perf(
    const vv_inference_ctx_t* ctx) {
    return ctx ? &ctx->last_perf : NULL;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

vv_status_t vv_inference_free(vv_inference_ctx_t* ctx) {
    if (!ctx) return VV_ERR_NULL_PTR;

    vv_stream_ctx_drop_cache(ctx);
    if (ctx->spec) { vv_spec_free(ctx->spec); ctx->spec = NULL; }
    if (ctx->drafter && !ctx->is_clone) vv_drafter_free(ctx->drafter);
    ctx->drafter = NULL;
    if (ctx->use_gpu) {
        if (ctx->workspace) vv_dev_free(ctx->workspace);
        if (!ctx->is_clone) {
            if (ctx->embed_table_gpu) vv_dev_free(ctx->embed_table_gpu);
            if (ctx->lm_head_gpu && ctx->lm_head_gpu != ctx->embed_table_gpu)
                vv_dev_free(ctx->lm_head_gpu);
            if (ctx->final_norm_gpu) vv_dev_free(ctx->final_norm_gpu);
            if (ctx->layer_pool) vv_layer_pool_free(ctx->layer_pool);
        }
    } else {
        if (ctx->workspace) vv_free(ctx->workspace);
    }

    if (ctx->shards) {
        for (int i = 0; i < ctx->n_shards; i++) {
            /* A clone borrows the parent's weights and pool, but owns its
               own cache, workspace, streams and landing buffer. */
            if (ctx->is_clone) ctx->shards[i].layer_pool = NULL;
            shard_free(&ctx->shards[i]);
        }
        vv_free(ctx->shards);
        vv_dev_set_device(ctx->gpu_id);
    }
    if (ctx->shard_done) vv_dev_event_destroy(ctx->shard_done);

    if (ctx->kv_cache) vv_kv_cache_free(ctx->kv_cache);
    if (ctx->use_gpu) {
        if (ctx->compute_stream) vv_dev_stream_destroy(ctx->compute_stream);
        if (ctx->transfer_stream) vv_dev_stream_destroy(ctx->transfer_stream);
    }

    if (ctx->tokenizer) vv_tokenizer_free(ctx->tokenizer);
    if (ctx->fe_stream) vv_frontend_stream_free(ctx->fe_stream);
    if (ctx->fe_ready) vv_dev_event_destroy(ctx->fe_ready);
    if (ctx->fe_done) vv_dev_event_destroy(ctx->fe_done);
    if (ctx->fe_lat_buf) vv_dev_free(ctx->fe_lat_buf);
    /* Clones borrow the device's front end; the parent goes last. */
    if (!ctx->is_clone && ctx->frontend) vv_frontend_free(ctx->frontend);
    free_frontend_host(ctx);

    if (ctx->model && !ctx->is_clone) {
        if (ctx->use_gpu && ctx->placement == VV_PLACE_ALL_GPU)
            free_layer_gpu_weights(ctx->model);
        vv_model_free(ctx->model);
    }


    if (ctx->use_gpu && !ctx->is_clone) vv_gemm_cleanup();
    vv_free(ctx);

    VV_LOG_I("inference: context freed");
    return VV_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Exports for the streaming backend (pipeline_internal.h)
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_pipeline_prefill(vv_inference_ctx_t* ctx, void* hidden,
                                int seq_len) {
    return prefill_all_shards(ctx, hidden, seq_len);
}

vv_status_t vv_pipeline_step(vv_inference_ctx_t* ctx, void* hidden,
                             bool graph_ok, vv_graph_slot_t* graphs) {
    return decoder_step_graphed(ctx, hidden, graph_ok, graphs);
}

vv_status_t vv_pipeline_step_taps(vv_inference_ctx_t* ctx, void* hidden,
                                  bool graph_ok, vv_graph_slot_t* graphs,
                                  const vv_taps_t* taps) {
    return decoder_step_graphed_taps(ctx, hidden, graph_ok, graphs, taps);
}

bool vv_pipeline_graph_ok(const vv_inference_ctx_t* ctx) {
    return decode_graph_enabled() && ctx->layer_pool &&
           ctx->layer_pool->all_resident && ctx->kv_cache &&
           ctx->kv_cache->d_len && !profile_decode();
}

void vv_pipeline_graphs_free(const vv_inference_ctx_t* ctx,
                             vv_graph_slot_t* graphs) {
    for (int i = 0; i <= ctx->n_shards; i++) {
        if (graphs[i].exec) vv_dev_graph_destroy(graphs[i].exec);
        graphs[i].exec = NULL;
        graphs[i].shape = -1;
    }
}

void vv_pipeline_kv_reset(vv_inference_ctx_t* ctx) {
    vv_kv_cache_reset(ctx->kv_cache, ctx->compute_stream);
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_dev_set_device(ctx->shards[i].gpu_id);
        vv_kv_cache_reset(ctx->shards[i].kv_cache,
                          ctx->shards[i].compute_stream);
    }
    if (ctx->use_gpu) vv_dev_set_device(ctx->gpu_id);
}

void vv_pipeline_kv_publish(vv_inference_ctx_t* ctx) {
    vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_dev_set_device(ctx->shards[i].gpu_id);
        vv_kv_cache_publish_len(ctx->shards[i].kv_cache,
                                ctx->shards[i].compute_stream);
    }
    if (ctx->use_gpu) vv_dev_set_device(ctx->gpu_id);
}

vv_status_t vv_pipeline_kv_reserve(vv_inference_ctx_t* ctx, int n_positions) {
    vv_status_t s = VV_OK;
    if (ctx->kv_cache && ctx->kv_cache->pool)
        s = vv_kv_cache_reserve_wait(ctx->kv_cache, n_positions,
                                     ctx->compute_stream);
    for (int i = 0; s == VV_OK && i < ctx->n_shards; i++) {
        vv_shard_t* sh = &ctx->shards[i];
        if (!sh->kv_cache || !sh->kv_cache->pool) continue;
        vv_dev_set_device(sh->gpu_id);
        s = vv_kv_cache_reserve_wait(sh->kv_cache, n_positions,
                                     sh->compute_stream);
    }
    if (ctx->use_gpu) vv_dev_set_device(ctx->gpu_id);
    return s;
}

void vv_pipeline_kv_release(vv_inference_ctx_t* ctx) {
    if (ctx->kv_cache && ctx->kv_cache->pool) {
        vv_dev_stream_sync(ctx->compute_stream);
        vv_kv_cache_release(ctx->kv_cache);
    }
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_shard_t* sh = &ctx->shards[i];
        if (!sh->kv_cache || !sh->kv_cache->pool) continue;
        vv_dev_set_device(sh->gpu_id);
        vv_dev_stream_sync(sh->compute_stream);
        vv_kv_cache_release(sh->kv_cache);
    }
    if (ctx->use_gpu) vv_dev_set_device(ctx->gpu_id);
}

vv_status_t vv_pipeline_cpu_head_argmax(vv_inference_ctx_t* ctx,
                                        const void* normed_gpu,
                                        uint16_t* h, float* f,
                                        int32_t* token) {
    return cpu_head_argmax(ctx, normed_gpu, h, f, token);
}

/* ─── Text generation: the language model without the audio ─────────────── */

/* Defined in special_tokens.c, like the family's other token names. */
extern const char* VV_TOKEN_IM_START;

/**
 * @brief Emit the part of `pending` that is whole UTF-8, keep the rest.
 *
 * A byte-level BPE token can end in the middle of a character — Cyrillic and
 * emoji are two tokens more often than not — so a delta cut at a token
 * boundary would put half a code point on the wire and clients would render
 * a replacement character that never goes away.
 */
static size_t utf8_complete(const char* s, size_t n) {
    size_t cut = n;
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;          /* continuation byte */
        const size_t need = (c & 0x80) == 0    ? 1 :
                            (c & 0xE0) == 0xC0 ? 2 :
                            (c & 0xF0) == 0xE0 ? 3 :
                            (c & 0xF8) == 0xF0 ? 4 : 1;
        cut = (back >= need) ? n : n - back;
        break;
    }
    return cut;
}

/** @brief The text ends with one of the caller's stop strings. */
static size_t stop_cut(const char* text, size_t len,
                       const char* const* stop, int n_stop) {
    for (int i = 0; i < n_stop; i++) {
        if (!stop[i] || !stop[i][0]) continue;
        const size_t sl = strlen(stop[i]);
        if (sl <= len && memcmp(text + len - sl, stop[i], sl) == 0)
            return len - sl;
    }
    return (size_t)-1;
}

/** @brief Growing text buffer for the answer. */
typedef struct gen_buf {
    char*  s;
    size_t n, cap;
} gen_buf_t;

static bool gen_push(gen_buf_t* b, const char* s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 1024;
        while (cap < b->n + n + 1) cap *= 2;
        char* p = (char*)vv_realloc(b->s, cap);
        if (!p) return false;
        b->s = p; b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = '\0';
    return true;
}

void vv_generation_free(vv_generation_t* g) {
    if (!g) return;
    vv_free(g->text);
    vv_free(g);
}

/**
 * @brief Prompt in, answer out, on whichever device this context placed.
 *
 * The decode loop is the transcription one minus the audio: same prefill,
 * same captured step, same stop tokens (`<|im_end|>` and `<|endoftext|>`
 * are what the ChatML family already ends on). Sampling is the one
 * addition — a transcript is decoded greedily on purpose, an answer usually
 * is not — and it needs the logits on the host, so `temperature > 0` asks
 * for a copy per token that greedy does not pay for.
 */
vv_status_t vv_inference_generate(vv_inference_ctx_t* ctx,
                                  const vv_generate_params_t* params,
                                  vv_generation_t** out) {
    if (!ctx || !params || !params->prompt || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!ctx->tokenizer) return VV_ERR_NULL_PTR;

    const vv_llm_config_t* llm = &ctx->model->config.llm;
    const int hs = llm->hidden_size;
    const int vocab_size = llm->vocab_size;
    const bool cpu_only = (ctx->placement == VV_PLACE_CPU_ONLY);
    const bool sampling = params->temperature > 0.0f;

    int32_t* ids = NULL;
    int seq_len = 0;
    vv_status_t s = vv_tokenizer_encode(ctx->tokenizer, params->prompt,
                                        &ids, &seq_len);
    if (s != VV_OK) return s;
    if (seq_len <= 0) { vv_free(ids); return VV_ERR_INVALID_ARG; }

    int max_new = params->max_tokens > 0 ? params->max_tokens : 512;
    const int window = ctx->kv_cache ? ctx->kv_cache->max_seq_len : 0;
    if (window > 0 && seq_len >= window) {
        VV_LOG_E("generate: the prompt is %d tokens and the KV window is %d; "
                 "raise --max-seq-len", seq_len, window);
        vv_free(ids);
        return VV_ERR_OVERFLOW;
    }
    if (window > 0 && seq_len + max_new > window) max_new = window - seq_len;

    vv_generation_t* g = (vv_generation_t*)vv_alloc(sizeof(*g));
    if (!g) { vv_free(ids); return VV_ERR_OUT_OF_MEMORY; }
    memset(g, 0, sizeof(*g));
    g->prompt_tokens = seq_len;
    g->finish_reason = "stop";

    gen_buf_t text = {0};
    gen_buf_t pending = {0};
    uint64_t rng = params->seed ? params->seed
                                : (uint64_t)(vv_time_ms() * 1000.0) | 1ull;
    size_t emitted = 0;
    bool cancelled = false;

    /* A slot runs one request at a time, so the cache is this one's alone. */
    vv_kv_cache_reset(ctx->kv_cache, ctx->compute_stream);
    for (int i = 0; i < ctx->n_shards; i++) {
        vv_dev_set_device(ctx->shards[i].gpu_id);
        vv_kv_cache_reset(ctx->shards[i].kv_cache,
                          ctx->shards[i].compute_stream);
    }
    if (!cpu_only) vv_dev_set_device(ctx->gpu_id);

    if (ctx->kv_cache && ctx->kv_cache->pool) {
        s = vv_kv_cache_reserve_wait(ctx->kv_cache, seq_len + max_new,
                                     ctx->compute_stream);
        if (s != VV_OK) {
            VV_LOG_E("generate: no KV pages for this request: %s",
                     vv_status_str(s));
            vv_free(ids); vv_generation_free(g);
            return s;
        }
    }

    const double t_prefill = vv_time_ms();
    int32_t token_id = 0;
    int n_generated = 0;
    /*
     * The answer ends at <|im_end|> (vv_family_stop), but a model fine-tuned
     * on one task also likes to open a turn of its own and keep talking:
     * <|im_start|> ends it too, or the answer would carry a whole imagined
     * conversation.
     */
    const int32_t im_start = vv_tokenizer_special_id(ctx->tokenizer,
                                                     VV_TOKEN_IM_START);

    /* ── The host path: the same steps, all in FP32 on the CPU ── */
    if (cpu_only) {
        if (sampling)
            VV_LOG_W("generate: the CPU path decodes greedily; temperature "
                     "and top_p are ignored");
        float* hidden = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
        float* one    = (float*)vv_alloc((size_t)hs * sizeof(float));
        float* normed = (float*)vv_alloc((size_t)hs * sizeof(float));
        if (!hidden || !one || !normed) {
            vv_free(hidden); vv_free(one); vv_free(normed);
            vv_free(ids); vv_generation_free(g);
            return VV_ERR_OUT_OF_MEMORY;
        }
        s = embed_host(ctx->model, ids, seq_len, hidden);
        if (s == VV_OK)
            s = vv_decoder_prefill_cpu(ctx->model, hidden, seq_len,
                                       ctx->kv_cache, (float*)ctx->workspace,
                                       ctx->workspace_size);
        if (s == VV_OK) {
            final_norm_host(ctx->model, hidden + (size_t)(seq_len - 1) * hs,
                            normed);
            s = head_host(ctx->model, normed, ctx->workspace,
                          ctx->workspace_size, &token_id);
        }
        vv_free(hidden);

        while (s == VV_OK && n_generated < max_new &&
               !token_ends(ctx, token_id) && token_id != im_start) {
            char piece[256];
            size_t pn = 0;
            if (vv_tokenizer_decode_into(ctx->tokenizer, &token_id, 1, true,
                                         piece, sizeof(piece), &pn) != VV_OK)
                pn = 0;
            if (pn > 0 && !gen_push(&pending, piece, pn)) {
                s = VV_ERR_OUT_OF_MEMORY; break;
            }
            n_generated++;

            const size_t whole = utf8_complete(pending.s ? pending.s : "",
                                               pending.n);
            if (whole > 0) {
                if (!gen_push(&text, pending.s, whole)) { s = VV_ERR_OUT_OF_MEMORY; break; }
                memmove(pending.s, pending.s + whole, pending.n - whole);
                pending.n -= whole;
                if (pending.s) pending.s[pending.n] = '\0';
            }
            const size_t cut = stop_cut(text.s ? text.s : "", text.n,
                                        params->stop, params->n_stop);
            if (cut != (size_t)-1) { text.n = cut; if (text.s) text.s[cut] = '\0'; break; }
            if (params->on_text && text.n > emitted) {
                if (!params->on_text(params->user, text.s + emitted)) {
                    cancelled = true; break;
                }
                emitted = text.n;
            }

            s = embed_host(ctx->model, &token_id, 1, one);
            if (s == VV_OK)
                s = vv_decoder_step_cpu(ctx->model, one, ctx->kv_cache,
                                        (float*)ctx->workspace,
                                        ctx->workspace_size);
            if (s == VV_OK) {
                final_norm_host(ctx->model, one, normed);
                s = head_host(ctx->model, normed, ctx->workspace,
                              ctx->workspace_size, &token_id);
            }
        }
        vv_free(one); vv_free(normed);
        goto done;
    }

    /* ── The device path ── */
    {
        const bool embed_on_cpu   = (ctx->embed_table_gpu == NULL);
        const bool lm_head_on_cpu = (ctx->lm_head_gpu == NULL);
        if (sampling && lm_head_on_cpu)
            VV_LOG_W("generate: the head is on the host, which decodes "
                     "greedily; temperature and top_p are ignored");

        void* hidden_gpu = NULL;
        void* one_gpu = NULL;
        void* normed_gpu = NULL;
        void* logits_gpu = NULL;
        void* am_v = NULL;
        void* am_i = NULL;
        void* tok_dev = NULL;
        uint16_t* host_h = NULL;
        float* host_f = NULL;
        float* logits_host = NULL;
        const size_t one_hidden = (size_t)hs * 2;

        s = vv_dev_alloc(&hidden_gpu, (size_t)seq_len * one_hidden);
        if (s == VV_OK) s = vv_dev_alloc(&one_gpu, one_hidden);
        if (s == VV_OK) s = vv_dev_alloc(&normed_gpu, one_hidden);
        if (s == VV_OK) s = vv_dev_alloc(&tok_dev, sizeof(int32_t));
        if (s == VV_OK && !lm_head_on_cpu) {
            s = vv_dev_alloc(&logits_gpu, (size_t)vocab_size * sizeof(float));
            if (s == VV_OK) s = vv_dev_alloc(&am_v, VV_ARGMAX_PARTIALS * sizeof(float));
            if (s == VV_OK) s = vv_dev_alloc(&am_i, VV_ARGMAX_PARTIALS * sizeof(int32_t));
        }
        if (s == VV_OK && lm_head_on_cpu) {
            host_h = (uint16_t*)vv_alloc(one_hidden);
            host_f = (float*)vv_alloc((size_t)hs * sizeof(float));
            if (!host_h || !host_f) s = VV_ERR_OUT_OF_MEMORY;
        }
        if (s == VV_OK && sampling && !lm_head_on_cpu) {
            logits_host = (float*)vv_alloc((size_t)vocab_size * sizeof(float));
            if (!logits_host) s = VV_ERR_OUT_OF_MEMORY;
        }

        /* Prompt → hidden states. */
        if (s == VV_OK) {
            if (embed_on_cpu) {
                float* f32 = (float*)vv_alloc((size_t)seq_len * hs * sizeof(float));
                uint16_t* f16 = (uint16_t*)vv_alloc((size_t)seq_len * one_hidden);
                if (f32 && f16) {
                    vv_embedding_cpu(ctx->model->embed_tokens.data, ids, f32,
                                     seq_len, hs);
                    float_to_half(f32, f16, seq_len * hs);
                    s = vv_dev_memcpy_h2d(hidden_gpu, f16,
                                          (size_t)seq_len * one_hidden,
                                          ctx->compute_stream);
                } else {
                    s = VV_ERR_OUT_OF_MEMORY;
                }
                vv_free(f32); vv_free(f16);
            } else {
                int32_t* ids_gpu = NULL;
                s = vv_dev_alloc((void**)&ids_gpu, (size_t)seq_len * sizeof(int32_t));
                if (s == VV_OK) {
                    s = vv_dev_memcpy_h2d(ids_gpu, ids,
                                          (size_t)seq_len * sizeof(int32_t),
                                          ctx->compute_stream);
                    if (s == VV_OK)
                        s = vv_embedding_dev(ctx->embed_table_gpu, ids_gpu,
                                             hidden_gpu, seq_len, hs,
                                             ctx->compute_stream);
                    vv_dev_free(ids_gpu);
                }
            }
        }

        /* Greedy answers can go a drafted block at a time (spec.h): the
         * drafter reads the prompt through the target's taps. */
        bool spec_on = ctx->spec && !sampling && !embed_on_cpu &&
                       !lm_head_on_cpu && ctx->n_shards == 0;
        if (spec_on) {
            vv_spec_reset(ctx->spec);
            vv_spec_stats_reset(ctx->spec);
            ctx->taps = vv_spec_prefill_taps(ctx->spec);
        }
        if (s == VV_OK) s = prefill_all_shards(ctx, hidden_gpu, seq_len);
        ctx->taps = NULL;

        /* The first token comes off the last prompt row. */
        if (s == VV_OK) {
            void* last = (uint8_t*)hidden_gpu + (size_t)(seq_len - 1) * one_hidden;
            int32_t tok_host = 0;
            s = vv_pipeline_head_argmax(ctx, last, normed_gpu, logits_gpu,
                                        am_v, am_i, tok_dev, &tok_host,
                                        host_h, host_f, &token_id);
            if (s == VV_OK && sampling && logits_host) {
                s = vv_dev_memcpy_d2h(logits_host, logits_gpu,
                                      (size_t)vocab_size * sizeof(float),
                                      ctx->compute_stream);
                if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
                if (s == VV_OK)
                    s = vv_sample_logits_f32(logits_host, vocab_size,
                                             params->temperature,
                                             params->top_p, params->top_k,
                                             &rng, &token_id);
                if (s == VV_OK)
                    s = vv_dev_memcpy_h2d(tok_dev, &token_id, sizeof(int32_t),
                                          ctx->compute_stream);
            }
        }
        vv_dev_free(hidden_gpu); hidden_gpu = NULL;

        graph_slot_t graphs[VV_MAX_GPUS];
        for (int i = 0; i <= ctx->n_shards; i++) { graphs[i].exec = NULL;
                                                   graphs[i].shape = -1; }
        const bool graph_ok = !sampling && decode_graph_enabled() &&
                              ctx->layer_pool && ctx->layer_pool->all_resident &&
                              ctx->kv_cache && ctx->kv_cache->d_len;

        vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
        for (int i = 0; i < ctx->n_shards; i++) {
            vv_dev_set_device(ctx->shards[i].gpu_id);
            vv_kv_cache_publish_len(ctx->shards[i].kv_cache,
                                    ctx->shards[i].compute_stream);
        }
        vv_dev_set_device(ctx->gpu_id);

        /*
         * A block returns several tokens at once, all but the last of them
         * already fed. They wait here and come out one per iteration, so the
         * text, the stop strings and the cap see each of them exactly as
         * they would see a step's; only when none is left is anything fed.
         */
        int32_t queued[VV_SPEC_MAX_BLOCK];
        int n_queued = 0, q_at = 0;
        bool dev_fresh = false;     /* device length and token: a step's */
        graph_slot_t graphs_taps[VV_MAX_GPUS];
        for (int i = 0; i < VV_MAX_GPUS; i++) { graphs_taps[i].exec = NULL;
                                                graphs_taps[i].shape = -1; }

        while (s == VV_OK && n_generated < max_new &&
               !token_ends(ctx, token_id) && token_id != im_start) {
            char piece[256];
            size_t pn = 0;
            if (vv_tokenizer_decode_into(ctx->tokenizer, &token_id, 1, true,
                                         piece, sizeof(piece), &pn) != VV_OK)
                pn = 0;
            if (pn > 0 && !gen_push(&pending, piece, pn)) {
                s = VV_ERR_OUT_OF_MEMORY; break;
            }
            n_generated++;

            const size_t whole = utf8_complete(pending.s ? pending.s : "",
                                               pending.n);
            if (whole > 0) {
                if (!gen_push(&text, pending.s, whole)) { s = VV_ERR_OUT_OF_MEMORY; break; }
                memmove(pending.s, pending.s + whole, pending.n - whole);
                pending.n -= whole;
                if (pending.s) pending.s[pending.n] = '\0';
            }
            const size_t cut = stop_cut(text.s ? text.s : "", text.n,
                                        params->stop, params->n_stop);
            if (cut != (size_t)-1) { text.n = cut; if (text.s) text.s[cut] = '\0'; break; }
            if (params->on_text && text.n > emitted) {
                if (!params->on_text(params->user, text.s + emitted)) {
                    cancelled = true; break;
                }
                emitted = text.n;
            }

            /* A token a block has fed already: the next one is known. */
            if (q_at < n_queued) {
                token_id = queued[q_at++];
                continue;
            }
            if (spec_on && vv_spec_want_block(ctx->spec)) {
                int n_out = 0;
                const double t0 = vv_time_ms();
                s = vv_spec_cycle(ctx, ctx->spec, token_id, queued, &n_out);
                if (s == VV_OK) {
                    vv_spec_note_block(ctx->spec, n_out, vv_time_ms() - t0);
                    dev_fresh = false;
                    n_queued = n_out;
                    q_at = 0;
                    token_id = queued[q_at++];
                    continue;
                }
                if (s == VV_ERR_KV_POOL_EXHAUSTED) {
                    /* No pages for a whole block: a plain step, below. */
                    s = VV_OK;
                    vv_spec_pause(ctx->spec);
                } else if (s != VV_ERR_OVERFLOW) {
                    break;
                } else {
                    /* The window has no room for a block: plain steps from
                     * here, which read the length and the token on the
                     * device. */
                    s = VV_OK;
                    spec_on = false;
                    vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
                    s = vv_dev_memcpy_h2d(tok_dev, &token_id, sizeof(int32_t),
                                          ctx->compute_stream);
                    if (s != VV_OK) break;
                }
            }
            if (spec_on) {
                /* Blocks are not paying here (or found no pages): a plain
                 * step the drafter sees as well, so they can resume later. */
                const double t0 = vv_time_ms();
                if (!dev_fresh) {
                    vv_kv_cache_publish_len(ctx->kv_cache, ctx->compute_stream);
                    s = vv_dev_memcpy_h2d(tok_dev, &token_id, sizeof(int32_t),
                                          ctx->compute_stream);
                }
                if (s == VV_OK)
                    s = vv_embedding_dev(ctx->embed_table_gpu,
                                         (const int32_t*)tok_dev, one_gpu, 1,
                                         hs, ctx->compute_stream);
                if (s == VV_OK)
                    s = decoder_step_graphed_taps(ctx, one_gpu, graph_ok,
                                                  graphs_taps,
                                                  vv_spec_step_taps(ctx->spec));
                if (s == VV_ERR_OVERFLOW) {
                    VV_LOG_W("generate: the KV window is full (%d positions); "
                             "the answer stops here -- raise --max-seq-len",
                             ctx->kv_cache->max_seq_len);
                    s = VV_OK;
                    g->finish_reason = "length";
                    break;
                }
                if (s == VV_OK) {
                    int32_t tok_host = 0;
                    s = vv_pipeline_head_argmax(ctx, one_gpu, normed_gpu,
                                                logits_gpu, am_v, am_i, tok_dev,
                                                &tok_host, host_h, host_f,
                                                &token_id);
                }
                if (s == VV_OK) s = vv_spec_push_step(ctx->spec);
                if (s != VV_OK) break;
                dev_fresh = true;
                vv_spec_note_step(ctx->spec, vv_time_ms() - t0);
                continue;
            }

            /* Embed the token that is already on the device, step, sample. */
            if (embed_on_cpu) {
                if (token_id < 0 || token_id >= vocab_size) { s = VV_ERR_INVALID_ARG; break; }
                s = vv_dev_memcpy_h2d(one_gpu,
                                      (const uint8_t*)ctx->model->embed_tokens.data
                                      + (size_t)token_id * one_hidden,
                                      one_hidden, ctx->compute_stream);
            } else {
                if (lm_head_on_cpu || sampling)
                    vv_dev_memcpy_h2d(tok_dev, &token_id, sizeof(int32_t),
                                      ctx->compute_stream);
                s = vv_embedding_dev(ctx->embed_table_gpu,
                                     (const int32_t*)tok_dev, one_gpu, 1, hs,
                                     ctx->compute_stream);
            }
            if (s != VV_OK) break;

            s = decoder_step_graphed(ctx, one_gpu, graph_ok, graphs);
            if (s == VV_ERR_OVERFLOW) {
                VV_LOG_W("generate: the KV window is full (%d positions); the "
                         "answer stops here -- raise --max-seq-len",
                         ctx->kv_cache->max_seq_len);
                s = VV_OK;
                g->finish_reason = "length";
                break;
            }
            if (s != VV_OK) break;

            int32_t tok_host = 0;
            s = vv_pipeline_head_argmax(ctx, one_gpu, normed_gpu, logits_gpu,
                                        am_v, am_i, tok_dev, &tok_host,
                                        host_h, host_f, &token_id);
            if (s == VV_OK && sampling && logits_host) {
                s = vv_dev_memcpy_d2h(logits_host, logits_gpu,
                                      (size_t)vocab_size * sizeof(float),
                                      ctx->compute_stream);
                if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
                if (s == VV_OK)
                    s = vv_sample_logits_f32(logits_host, vocab_size,
                                             params->temperature,
                                             params->top_p, params->top_k,
                                             &rng, &token_id);
            }
        }

        for (int i = 0; i < VV_MAX_GPUS; i++)
            if (graphs_taps[i].exec) vv_dev_graph_destroy(graphs_taps[i].exec);
        if (ctx->spec && !sampling) {
            const vv_spec_stats_t* st = vv_spec_get_stats(ctx->spec);
            if (st && st->cycles > 0)
                VV_LOG_I("generate: %lld drafted blocks, %.2f tokens each, "
                         "%lld plain steps", (long long)st->cycles,
                         (double)st->tokens / (double)st->cycles,
                         (long long)st->steps);
        }

        for (int i = 0; i <= ctx->n_shards; i++)
            if (graphs[i].exec) vv_dev_graph_destroy(graphs[i].exec);

        vv_dev_free(one_gpu); vv_dev_free(normed_gpu); vv_dev_free(tok_dev);
        if (logits_gpu) vv_dev_free(logits_gpu);
        if (am_v) vv_dev_free(am_v);
        if (am_i) vv_dev_free(am_i);
        vv_free(host_h); vv_free(host_f); vv_free(logits_host);
    }

done:
    vv_free(ids);
    /* Whatever is left in `pending` is a broken character, not text. */
    vv_free(pending.s);

    if (ctx->kv_cache && ctx->kv_cache->pool) vv_pipeline_kv_release(ctx);

    if (s != VV_OK) {
        vv_free(text.s);
        vv_generation_free(g);
        return s;
    }

    if (cancelled)                    g->finish_reason = "cancelled";
    else if (n_generated >= max_new)  g->finish_reason = "length";

    g->text = text.s ? text.s : (char*)vv_alloc(1);
    if (!g->text) { vv_generation_free(g); return VV_ERR_OUT_OF_MEMORY; }
    if (!text.s) g->text[0] = '\0';
    g->completion_tokens = n_generated;

    vv_perf_metrics_t* perf = &ctx->last_perf;
    perf->prefill_tokens = seq_len;
    perf->decode_tokens = n_generated;
    perf->total_ms = vv_time_ms() - t_prefill;
    VV_LOG_I("generate: %d prompt + %d new tokens in %.0f ms (%s)",
             seq_len, n_generated, perf->total_ms, g->finish_reason);

    *out = g;
    return VV_OK;
}
