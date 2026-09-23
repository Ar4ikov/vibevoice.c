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
                           int max_pos, int attn_backend, int verify_rows,
                           int check, void* stream, vv_spec_t** out);
void vv_spec_free(vv_spec_t* s);

/** @brief Empty the drafter's context (a new sequence). */
void vv_spec_reset(vv_spec_t* s);

/*
 * Blocks or steps. A block pays only while its tokens per millisecond beat a
 * plain step's, and how many drafts are kept depends on the audio: on
 * speech like the drafter's corpus a block brings 2-3 tokens, on a language
 * it barely saw ~1, at the price of about two steps. So the decode loops
 * ask before every position, and tell afterwards what it cost:
 *
 *   vv_spec_want_block()   true: a block (vv_spec_cycle); false: a plain
 *                          step through vv_spec_step_taps(), then
 *                          vv_spec_push_step() so the drafter sees it too;
 *   vv_spec_note_block() / vv_spec_note_step()   the wall time it took.
 *
 * The first positions of a sequence are plain steps (a measurement of the
 * step); then blocks, while their running rate stays above the steps'; below
 * it, a run of plain steps before blocks are tried again, the runs doubling
 * while blocks keep losing. Either way the tokens are the same.
 */
bool vv_spec_want_block(vv_spec_t* s);
void vv_spec_note_block(vv_spec_t* s, int tokens, double ms);
void vv_spec_note_step(vv_spec_t* s, double ms);

/** @brief A block found no room (VV_ERR_KV_POOL_EXHAUSTED: the shared page
 *         pool cannot give a whole block's pages): a run of plain steps,
 *         which need one page at most. vv_spec_cycle changes nothing before
 *         it asks for the pages, so a step can follow at once. */
void vv_spec_pause(vv_spec_t* s);

/** @brief Taps for a plain step: its one row lands in row 0. */
const vv_taps_t* vv_spec_step_taps(vv_spec_t* s);

/** @brief The plain step just taken (at the drafter's context length) into
 *         the drafter's context, from vv_spec_step_taps()' row. */
vv_status_t vv_spec_push_step(vv_spec_t* s);

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
