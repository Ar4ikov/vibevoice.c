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
