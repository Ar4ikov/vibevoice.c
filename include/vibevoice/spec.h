/**
 * @file spec.h
 * @brief Speculative decoding with a DFlash 2 drafter, and the traces its
 *        training reads.
 *
 * A DFlash 2 drafter (https://inco.ai/blog/dflash2/) is a small
 * block-diffusion model that proposes a whole block of tokens in one forward
 * pass, conditioned on the target model's own hidden states at a few layers
 * (its "taps"); the target then checks the block in one pass and keeps the
 * longest prefix it agrees with. docs/DFLASH.md has the design.
 *
 * This header has two halves: the trace -- what tools/dflash trains a
 * drafter from -- and the drafter itself.
 */
#ifndef VV_SPEC_H
#define VV_SPEC_H

#include "vibevoice/types.h"
#include "vibevoice/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Traces (training data) ────────────────────────────────────────────── */

/** @brief What a position of a trace is to the drafter's training. */
typedef enum vv_trace_kind {
    VV_TRACE_CONTEXT = 0,  /**< prompt or chunk row: context only           */
    VV_TRACE_GEN     = 1,  /**< a generated token: anchor and label         */
    VV_TRACE_LABEL   = 2,  /**< a stop token: a label, never an anchor      */
} vv_trace_kind_t;

/**
 * @brief A transcription replayed with the target's hidden states kept.
 *
 * The replay feeds the model exactly what the transcription fed it -- the
 * prompt with the audio, and for a streaming model every chunk and its text
 * in turn -- but as prefills of known tokens instead of decode steps, so it
 * runs at prefill speed. Every fed position leaves the outputs of
 * `taps.layers` in `taps.buf` at its row (position p at row p: `row_base`
 * 0, no `on_chunk`), FP16 on the device. The token that stopped each chunk
 * is not fed; it follows its chunk as a VV_TRACE_LABEL position (the last
 * one's row has no features).
 */
typedef struct vv_spec_trace {
    vv_taps_t taps;       /**< buf: device FP16 [cap][taps.n][hidden]       */
    int32_t*  ids;        /**< host [cap]: the token at each position; the
                               speech pad id under an audio row            */
    uint8_t*  kind;       /**< host [cap]: vv_trace_kind_t                  */
    int       cap;        /**< positions the three buffers hold            */
    int       n;          /**< out: positions written                      */
} vv_spec_trace_t;

/**
 * @brief Replay `tr` (a transcription of `pcm24k` on this context's model,
 *        with its `tokens`) and fill `t`.
 *
 * `context_info` must be the hotwords the transcription had. GPU contexts
 * whose layers all sit on one device only.
 *
 * @return VV_ERR_OVERFLOW (with `t->n` = the positions needed) when the
 *         trace does not fit `t->cap` or the KV window.
 */
vv_status_t vv_spec_trace(vv_inference_ctx_t* ctx, const float* pcm24k,
                          int n_samples, const char* context_info,
                          const vv_transcription_t* tr, vv_spec_trace_t* t);

/* ─── The drafter ────────────────────────────────────────────────────────── */

/**
 * @brief A drafter's config.json: a Qwen3-style stack plus `dflash_config`
 *        (the names the published DFlash 2 drafters use).
 */
typedef struct vv_drafter_config {
    int   hidden_size;         /**< = the target's                          */
    int   num_layers;
    int   num_heads;
    int   num_kv_heads;
    int   head_dim;            /**< 128                                     */
    int   intermediate_size;
    int   vocab_size;          /**< = the target's                          */
    int   num_target_layers;   /**< the target's depth it was trained on    */
    float rms_norm_eps;
    float rope_theta;
    int   sliding_window;      /**< 0: full attention (the only kind yet)   */
    int   block_size;          /**< rows per draft, the last token included */
    int   mask_token_id;
    int   n_taps;
    int   target_layer_ids[VV_TAPS_MAX];
    int   conv_kernel;         /**< taps of the in-block convolution (2)    */
    int   conv_group;          /**< channels per dynamic correction (16)    */
    int   selector_rank;
    int   selector_top_k;
    /** Ids the drafter proposes from: `draft_vocab_size` rows of the
     *  target's head, listed by the `draft_vocab` tensor. 0: all of them. */
    int   draft_vocab_size;
    /** vv_drafter_quant_t of the projections on the device. */
    int   weight_quant;
    /** `vv_fp16` in config.json, 0 for the defaults: how far the runtime
     *  scales the MLP activation and the sublayer outputs down to keep
     *  them in FP16 range (spec.c). */
    float fp16_mlp_div;
    float fp16_out_div;
} vv_drafter_config_t;

/** @brief A loaded drafter: its weights on one device, shared by every
 *         context (slot) of that device. */
typedef struct vv_drafter vv_drafter_t;

/** @brief Read and check `dir`/config.json. */
vv_status_t vv_drafter_config_load(const char* dir, vv_drafter_config_t* c);

/** @brief Device bytes of a drafter's weights, from its config and its
 *         `weight_quant` (set by the caller; config_load leaves int4). */
size_t vv_drafter_weight_bytes(const vv_drafter_config_t* c);

/** @brief Device bytes one context needs to run it: its KV cache for
 *         `max_pos` positions, the target's taps and the draft pass. */
size_t vv_spec_context_bytes(const vv_drafter_config_t* c, int max_pos);

/**
 * @brief Load `dir` (config.json + model.safetensors, BF16/F16/F32) onto
 *        `gpu_id` for `target`, whose width, vocabulary and depth it must
 *        have been trained on, its projections in `quant`
 *        (vv_drafter_quant_t).
 */
vv_status_t vv_drafter_load(const char* dir, const vv_model_t* target,
                            int gpu_id, int quant, vv_drafter_t** out);

void vv_drafter_free(vv_drafter_t* d);
const vv_drafter_config_t* vv_drafter_get_config(const vv_drafter_t* d);
size_t vv_drafter_device_bytes(const vv_drafter_t* d);

/** @brief How speculation went for a context since its counters were
 *         last reset. Acceptance length = tokens / cycles. */
typedef struct vv_spec_stats {
    int64_t cycles;     /**< draft + verify passes                        */
    int64_t drafted;    /**< tokens proposed                              */
    int64_t accepted;   /**< of them kept                                 */
    int64_t tokens;     /**< tokens produced (kept + the target's own)    */
    double  draft_ms;   /**< host wall time of the draft passes           */
    double  verify_ms;  /**< and of the verification and bookkeeping      */
} vv_spec_stats_t;

/** @brief The context's counters, or NULL when it has no drafter. */
const vv_spec_stats_t* vv_inference_spec_stats(const vv_inference_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VV_SPEC_H */
