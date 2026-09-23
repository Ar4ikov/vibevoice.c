/**
 * @file spec_internal.h
 * @brief The per-context half of speculative decoding (spec.c), for the
 *        decode loops of pipeline.c and stream_ctx.c.
 */
#ifndef VV_SPEC_INTERNAL_H
#define VV_SPEC_INTERNAL_H

#include "vibevoice/spec.h"
#include "pipeline_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Rows of target taps a context keeps: a prefill chunk's worth. */
#define VV_SPEC_TAP_ROWS 2048

/** @brief Largest draft block (block_size) a drafter may have. */
#define VV_SPEC_MAX_BLOCK 16

/** @brief One context's drafter state: its KV cache, taps, scratch. */
typedef struct vv_spec vv_spec_t;

/**
 * @brief State for one context of `d`, whose target rows are
 *        `target_hidden` wide, for `max_pos` positions, on `stream`.
 */
vv_status_t vv_spec_create(const vv_drafter_t* d, int target_hidden,
                           int max_pos, int attn_backend, void* stream,
                           vv_spec_t** out);
void vv_spec_free(vv_spec_t* s);

/** @brief Empty the drafter's context (a new sequence). */
void vv_spec_reset(vv_spec_t* s);

/** @brief Draft block rows B (the last token included). */
int vv_spec_block(const vv_spec_t* s);

/** @brief Positions the drafter's context holds. */
int vv_spec_context_len(const vv_spec_t* s);

/**
 * @brief Taps for a prefill of the target: every chunk's rows become the
 *        drafter's context at their positions as the chunk completes. Set
 *        ctx->taps to it for the prefill, NULL afterwards.
 */
const vv_taps_t* vv_spec_prefill_taps(vv_spec_t* s);

/**
 * @brief One draft-and-verify cycle (see spec.c).
 *
 * On entry the target's cache and the drafter's context both end at p and
 * `anchor` is the unfed token at p. On return `out` holds `*n_out` (1..B)
 * tokens -- the drafts the target kept, then its own next token -- both end
 * at p + *n_out, and out[*n_out - 1] is the new anchor. A token of `out`
 * that ends generation leaves tokens after it fed; vv_spec_rewind() undoes
 * them when the cache lives on (a streaming chunk).
 *
 * @return VV_ERR_OVERFLOW when a block no longer fits the window.
 */
vv_status_t vv_spec_cycle(vv_inference_ctx_t* ctx, vv_spec_t* s,
                          int32_t anchor, int32_t* out, int* n_out);

/** @brief Cut both caches back to `len` positions. */
vv_status_t vv_spec_rewind(vv_inference_ctx_t* ctx, vv_spec_t* s, int len);

/** @brief Drop the drafter's context past `len`. */
vv_status_t vv_spec_truncate(vv_spec_t* s, int len);

const vv_spec_stats_t* vv_spec_get_stats(const vv_spec_t* s);
void vv_spec_stats_reset(vv_spec_t* s);

#ifdef __cplusplus
}
#endif

#endif /* VV_SPEC_INTERNAL_H */
