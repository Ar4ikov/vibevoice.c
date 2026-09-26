/**
 * @file pipeline_internal.h
 * @brief The pieces of pipeline.c that a streaming session reuses.
 *
 * Not public API: the streaming backend (stream_ctx.c) runs the same
 * prefill, the same captured decode step and the same host-side head as a
 * one-shot transcription, only chunk by chunk on one cache. These are thin
 * exports of what transcribe_gpu uses, so there is one implementation of
 * each.
 */
#ifndef VV_PIPELINE_INTERNAL_H
#define VV_PIPELINE_INTERNAL_H

#include "vibevoice/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief One captured decode step and the launch shape it was made for.
 *  Start `shape` at -1 and `exec` at NULL. */
typedef struct vv_graph_slot { void* exec; int shape; } vv_graph_slot_t;

/** @brief Prefill `seq_len` rows at the cache's current length, through
 *         the primary's layers and every shard's; `hidden` ends up holding
 *         the last layer's output on the primary. */
vv_status_t vv_pipeline_prefill(vv_inference_ctx_t* ctx, void* hidden,
                                int seq_len);

/** @brief One decode step over every shard, replayed from `graphs` (one per
 *         device: 1 + n_shards) when `graph_ok`. */
vv_status_t vv_pipeline_step(vv_inference_ctx_t* ctx, void* hidden,
                             bool graph_ok, vv_graph_slot_t* graphs);

/** @brief vv_pipeline_step that also leaves its row of the tapped layers in
 *         row 0 of `taps` (a speculative context's plain step). Its captures
 *         belong in their own `graphs`, apart from the untapped step's. */
vv_status_t vv_pipeline_step_taps(vv_inference_ctx_t* ctx, void* hidden,
                                  bool graph_ok, vv_graph_slot_t* graphs,
                                  const vv_taps_t* taps);

/** @brief Whether decode steps on `ctx` can be captured and replayed. */
bool vv_pipeline_graph_ok(const vv_inference_ctx_t* ctx);

/** @brief Destroy the captures in `graphs` (1 + n_shards entries). */
void vv_pipeline_graphs_free(const vv_inference_ctx_t* ctx,
                             vv_graph_slot_t* graphs);

/** @brief Empty the cache of the primary and of every shard. */
void vv_pipeline_kv_reset(vv_inference_ctx_t* ctx);

/** @brief Hand the host lengths to the device on every shard (before decode
 *         steps that follow a prefill). */
void vv_pipeline_kv_publish(vv_inference_ctx_t* ctx);

/**
 * @brief Map pages for positions [0, n_positions) on every paged cache,
 *        waiting while other slots hold them. A no-op without a pool.
 */
vv_status_t vv_pipeline_kv_reserve(vv_inference_ctx_t* ctx, int n_positions);

/** @brief Give every page back (after the stream is drained). */
void vv_pipeline_kv_release(vv_inference_ctx_t* ctx);

/** @brief Free the device buffers a closed streaming session parked on
 *         `ctx` (stream_ctx.c). Called by vv_inference_free(). */
void vv_stream_ctx_drop_cache(vv_inference_ctx_t* ctx);

struct vv_spec_trace;
struct vv_transcription;

/** @brief vv_spec_trace() for a chunked model: a replayed session
 *         (stream_ctx.c). The device is bound and `t` checked. */
vv_status_t vv_stream_trace(vv_inference_ctx_t* ctx, const float* pcm24k,
                            int n_samples, const char* context_info,
                            const struct vv_transcription* tr,
                            struct vv_spec_trace* t);

/** @brief Partials the device argmax reduces through, per call. */
#define VV_ARGMAX_PARTIALS 256

/**
 * @brief Final norm, LM head and greedy argmax of one FP16 hidden row on
 *        the device -- the tail of every decode step, batch or streaming.
 *
 * With the head on the device the token is left in `tok_dev` (so the next
 * step can embed it without a round trip) and copied to `tok_host`; with it
 * in host memory `logits`, `am_v`, `am_i` and `tok_dev` are unused and `h`,
 * `f` (one row each) stage the host GEMV. Ends in a sync of the compute
 * stream either way.
 */
vv_status_t vv_pipeline_head_argmax(vv_inference_ctx_t* ctx, const void* row,
                                    void* normed, void* logits, void* am_v,
                                    void* am_i, void* tok_dev,
                                    int32_t* tok_host, uint16_t* h, float* f,
                                    int32_t* token);

/** @brief Greedy token from an LM head that stayed in host memory; `h`
 *         and `f` are one hidden row each. */
vv_status_t vv_pipeline_cpu_head_argmax(vv_inference_ctx_t* ctx,
                                        const void* normed_gpu,
                                        uint16_t* h, float* f,
                                        int32_t* token);

#ifdef __cplusplus
}
#endif

#endif /* VV_PIPELINE_INTERNAL_H */
