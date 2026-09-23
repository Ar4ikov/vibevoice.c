/**
 * @file inference.h
 * @brief Inference pipeline API: end-to-end audio → transcription.
 */
#ifndef VV_INFERENCE_H
#define VV_INFERENCE_H

#include "vibevoice/types.h"
#include "vibevoice/model.h"
#include "vibevoice/family.h"
#include "vibevoice/device.h"

#include <string.h>          /* the inline params defaults memset themselves */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── KV-Cache ──────────────────────────────────────────────────────────── */

/**
 * @brief A device's worth of KV pages, shared by the caches of its slots.
 *
 * Each layer holds [n_pages][VV_KV_PAGE_SIZE][n_kv_heads][bytes_per_vec]
 * (plus FP16 scales for TurboQuant). A cache maps logical pages onto pool
 * pages through its own table, so N slots need memory for the positions they
 * actually hold rather than N full windows. Opaque; thread-safe.
 */
typedef struct vv_kv_pool vv_kv_pool_t;

/**
 * @brief KV-cache for Qwen2 transformer.
 *
 * 28 layers × 2 (K+V) × 4 KV heads × head_dim=128
 * FP16 by default. FP8 option for memory savings.
 */
typedef struct vv_kv_cache {
    void**   k_cache;         /**< [num_layers] pointers to K store          */
    void**   v_cache;         /**< [num_layers] pointers to V store          */
    void**   k_meta;          /**< [num_layers] per-vector scales, or NULL   */
    void**   v_meta;
    void**   k_ref;           /**< [num_layers] reference key, or NULL       */
    bool*    ref_ready;       /**< [num_layers] reference computed yet       */
    int      num_layers;
    /*
     * The slice of those layers this cache actually holds. A cache that is
     * not sharded holds all of them, so these are 0 and num_layers - 1 and
     * every index below stays global — which is the point: nothing that
     * walks layers has to know whether the model was split.
     */
    int      first_layer;
    int      last_layer;
    int      n_kv_heads;
    int      head_dim;
    int      max_seq_len;     /**< Maximum cache capacity */
    int      current_len;     /**< Current number of cached positions */
    int      format;          /**< vv_kv_format_t storage format             */
    int      bytes_per_vec;   /**< Store bytes for one head's vector         */
    bool     on_cpu;          /**< true = CPU RAM, false = GPU VRAM */
    size_t   bytes_total;     /**< Store + metadata, all layers              */
    /*
     * `current_len` again, on the device, plus that value + 1. A decode step
     * is issued once and replayed for every token, so the kernels that move
     * with the position -- RoPE, the cache write, how far attention walks --
     * read it from here rather than from a kernel argument the replay would
     * reuse. `d_len_next` is what attention covers: the cache as it will be
     * once this token's K and V are in it.
     */
    void*    d_len;           /**< device int, GPU caches only               */
    void*    d_len_next;      /**< device int = *d_len + 1                   */

    /** vv_attn_backend_t this cache is read with, already resolved. */
    int      attn_backend;
    /*
     * Paging. With a pool, the per-layer pointers above are the pool's and
     * position t lives at page_table[t / 64] * 64 + t % 64. The table is on
     * the device (the kernels read it, so a captured graph follows it) and
     * `pages` is its host mirror.
     */
    vv_kv_pool_t* pool;       /**< NULL for a contiguous slab                */
    int*     page_table;      /**< device [max_pages]                        */
    int*     pages;           /**< host [max_pages]                          */
    int      n_pages;         /**< pages mapped so far                       */
    int      max_pages;
    /** vv_kv_cache_reserve_wait() does not wait: a short pool is
     *  VV_ERR_KV_POOL_EXHAUSTED at once (a live stream sets it). */
    bool     no_wait;
    /*
     * Positions RoPE skips from here on: a row appended at cache index i is
     * rotated as position i + rope_gap. asr-bitnet's reference feeds the
     * floor(n/3200) frames it has but starts decoding at the prompt length
     * with ceil(n/3200) pads, so its first generated token sits one or more
     * positions past the last row it cached. Zero everywhere else; reset
     * clears it. Only the CPU BitNet layer reads it.
     */
    int      rope_gap;
} vv_kv_cache_t;

/**
 * @brief Allocate KV-cache (GPU or CPU).
 * @param format  vv_kv_format_t: FP16, emulated FP8, or TurboQuant.
 * @param on_cpu  If true, allocate in CPU RAM instead of GPU VRAM.
 */
vv_status_t vv_kv_cache_create(vv_kv_cache_t** cache,
                                int num_layers, int n_kv_heads,
                                int head_dim, int max_seq_len,
                                int format, bool on_cpu);

/**
 * @brief A cache for layers [first, first + count) of a `num_layers` model.
 *
 * Allocates only that slice; the pointer arrays still have one entry per
 * model layer, so callers index by the layer number they already have. This
 * is what puts each shard's KV on its own device.
 */
vv_status_t vv_kv_cache_create_range(vv_kv_cache_t** cache,
                                     int num_layers, int first, int count,
                                     int n_kv_heads, int head_dim,
                                     int max_seq_len, int format, bool on_cpu);

/** @brief Bytes a cache of this shape and format would occupy. */
size_t vv_kv_cache_bytes(int num_layers, int n_kv_heads, int head_dim,
                         int max_seq_len, int format);

/**
 * @brief Append new K, V to cache at current position.
 *
 * With `use_device_pos` the write lands at the position held in `d_len`
 * instead of the host's `current_len`, which is what lets a decode step be
 * captured once and replayed. Only valid for a single position on a GPU
 * cache; prefill passes false.
 */
vv_status_t vv_kv_cache_append(vv_kv_cache_t* cache, int layer,
                                const void* k, const void* v,
                                int seq_len, bool use_device_pos,
                                void* stream);

/**
 * @brief Publish the host's `current_len` to the device copy.
 *
 * Prefill advances the length on the host; decode reads it on the device.
 * This is the handover, called once before the decode loop.
 */
vv_status_t vv_kv_cache_publish_len(vv_kv_cache_t* cache, void* stream);

/**
 * @brief Get K, V pointers for a layer (for attention).
 */
vv_status_t vv_kv_cache_get(const vv_kv_cache_t* cache, int layer,
                              const void** k, const void** v, int* len);

/** @brief Metadata pointers for a layer; both NULL unless the format has any. */
vv_status_t vv_kv_cache_get_meta(const vv_kv_cache_t* cache, int layer,
                                 const void** k_meta, const void** v_meta);

/**
 * @brief Reset cache (for new inference), ordered on `stream`.
 */
vv_status_t vv_kv_cache_reset(vv_kv_cache_t* cache, void* stream);

/**
 * @brief Free KV-cache.
 */
vv_status_t vv_kv_cache_free(vv_kv_cache_t* cache);

/**
 * @brief Allocate a page pool for layers [first, first + count).
 * @param n_pages  Pages of VV_KV_PAGE_SIZE positions, shared by all users.
 */
vv_status_t vv_kv_pool_create(vv_kv_pool_t** pool, int num_layers,
                              int first, int count, int n_kv_heads,
                              int head_dim, int n_pages, int format);

/** @brief Drop one reference; the last one frees the device memory. */
void vv_kv_pool_release(vv_kv_pool_t* pool);

/** @brief Pages in the pool, and how many are free right now. */
int vv_kv_pool_pages(const vv_kv_pool_t* pool, int* n_free);

/** @brief Device bytes the pool holds. */
size_t vv_kv_pool_bytes(const vv_kv_pool_t* pool);

/**
 * @brief A cache that takes its pages from `pool` (and holds a reference).
 *
 * Nothing is mapped until vv_kv_cache_reserve; `max_seq_len` only bounds
 * the table.
 */
vv_status_t vv_kv_cache_create_paged(vv_kv_cache_t** cache,
                                     vv_kv_pool_t* pool, int max_seq_len);

/**
 * @brief Make sure positions [0, n_positions) have pages.
 *
 * Takes pages from the pool and writes their ids into the device table on
 * `stream`. A no-op for a slab or when they are already mapped. Must not be
 * called inside a graph capture unless it is known to be a no-op.
 * @return VV_ERR_OVERFLOW past max_seq_len (the window is full),
 *         VV_ERR_KV_POOL_EXHAUSTED when the pool has too few free pages.
 */
vv_status_t vv_kv_cache_reserve(vv_kv_cache_t* cache, int n_positions,
                                void* stream);

/**
 * @brief vv_kv_cache_reserve that waits for pages instead of failing.
 *
 * A pool shared by several slots can run short while another slot still
 * holds pages it will return. This blocks until they come back, and returns
 * VV_ERR_KV_POOL_EXHAUSTED only when that cannot happen: the pool is smaller
 * than the request, or every other cache holding pages is itself waiting
 * (then exactly one of the waiters fails, and its pages free the others).
 * Never call it inside a graph capture.
 */
vv_status_t vv_kv_cache_reserve_wait(vv_kv_cache_t* cache, int n_positions,
                                     void* stream);

/** @brief Free pages, caches holding pages, and how many of those are
 *         blocked in vv_kv_cache_reserve_wait. Any pointer may be NULL. */
void vv_kv_pool_stats(const vv_kv_pool_t* pool, int* n_free, int* n_holders,
                      int* n_stalled);

/**
 * @brief Give every page back to the pool. The caller guarantees no kernel
 *        still reads them (the stream is idle). Also done by reset.
 */
void vv_kv_cache_release(vv_kv_cache_t* cache);

/**
 * @brief The view attention kernels take for one layer of `cache`.
 */
vv_kv_view_t vv_kv_cache_view(const vv_kv_cache_t* cache, int layer);

/* ─── Layer weight streaming pool ───────────────────────────────────────── */

/**
 * @brief Double-buffered GPU staging pool for layer-by-layer weight upload.
 *
 * When VRAM is too small to hold all 28 layers at once, we keep NF4
 * weights on CPU and upload one layer's worth to a reusable GPU buffer
 * before processing it.  Double-buffering allows overlapping transfer
 * of layer i+1 while layer i is computing.
 */
#define VV_LAYER_POOL_SLOTS 2
/* Every slot vv_layer_tensors() lists (model.h), biases of all 7 included. */
#define VV_LAYER_TENSORS_PER_LAYER VV_LAYER_TENSOR_SLOTS

typedef struct vv_layer_pool {
    void*  gpu_buf[VV_LAYER_POOL_SLOTS]; /**< Pre-allocated GPU staging  */
    size_t buf_size;                      /**< Size of each GPU buffer    */
    int    loaded[VV_LAYER_POOL_SLOTS];   /**< Layer idx in slot (-1=empty) */

    /* Saved CPU pointers for restoring after unstage */
    void*  saved_ptrs[VV_LAYER_POOL_SLOTS][VV_LAYER_TENSORS_PER_LAYER];

    bool   all_resident;  /**< true = all layers already on GPU, pool unused */

    /*
     * Ordering between the copy stream and the compute stream. `ready` fires
     * when a slot's weights have landed; `done` fires when the compute that
     * used them has finished, which is what makes it safe to overwrite the
     * slot with the next layer.
     */
    void*  ready_ev[VV_LAYER_POOL_SLOTS];
    void*  done_ev[VV_LAYER_POOL_SLOTS];
    bool   done_valid[VV_LAYER_POOL_SLOTS];
    int    n_bufs;
    int    n_resident;   /**< Layers permanently on the GPU          */
} vv_layer_pool_t;

vv_status_t vv_layer_pool_create(vv_layer_pool_t** pool,
                                  const vv_model_t* model, bool all_resident);
vv_status_t vv_layer_pool_stage(vv_layer_pool_t* pool,
                                 vv_model_t* model, int layer_idx,
                                 void* stream);
vv_status_t vv_layer_pool_unstage(vv_layer_pool_t* pool,
                                   vv_model_t* model, int layer_idx);
vv_status_t vv_layer_pool_free(vv_layer_pool_t* pool);

/**
 * @brief Start uploading layer `layer_idx` on the copy stream.
 *
 * With two staging buffers this runs while the previous layer is still on
 * the tensor cores, which is the whole point of streaming weights: the PCIe
 * transfer for layer i+1 overlaps the compute of layer i.
 */
vv_status_t vv_layer_prefetch_begin(vv_layer_pool_t* pool, vv_model_t* model,
                                    int layer_idx, void* xfer_stream);

/** @brief Block the compute stream until layer `layer_idx` has landed. */
vv_status_t vv_layer_prefetch_wait(vv_layer_pool_t* pool, vv_model_t* model,
                                   int layer_idx, void* compute_stream);

/** @brief Mark a layer's compute finished so its slot can be reused. */
vv_status_t vv_layer_prefetch_done(vv_layer_pool_t* pool, int layer_idx,
                                   void* compute_stream);

/** @brief Page-lock the host copies of the layers that will be streamed. */
vv_status_t vv_layer_pool_pin_host(vv_model_t* model, int first_streamed);

/**
 * @brief The same, for one range of layers.
 *
 * A shard must not page-lock the layers another device is about to upload
 * and free: registration outlives the free, and the driver is then holding
 * a mapping of memory nobody owns.
 */
vv_status_t vv_layer_pool_pin_range(vv_model_t* model, int first, int count);

/* ─── Hidden-state taps ─────────────────────────────────────────────────── */

/** @brief Most layers one set of taps copies. */
#define VV_TAPS_MAX 8

/**
 * @brief Copies of chosen layers' outputs, taken as a forward pass goes by.
 *
 * A speculative drafter (spec.h) reads the target model's hidden states at a
 * few layers; the layers write them here as they finish. The buffer holds
 * `rows` positions of `n` interleaved layer outputs: position r, slot i at
 * element (r * n + i) * hidden_size -- FP16 on the device, FP32 on the CPU
 * -- which is the drafter's input layout. `layers[i]` fills slot i.
 *
 * A prefill larger than the buffer is handed over chunk by chunk: with
 * `on_chunk` set, each chunk's rows land at 0.. and `on_chunk` runs once the
 * chunk is through every layer (`pos0` is its first cache position);
 * without it, cache position p lands at row p - `row_base`, which must fit.
 */
typedef struct vv_taps {
    void* buf;
    int   n;
    int   layers[VV_TAPS_MAX];
    int   rows;
    int   row_base;
    vv_status_t (*on_chunk)(void* user, int pos0, int n_rows, void* stream);
    void* user;
} vv_taps_t;

/* ─── Decoder ───────────────────────────────────────────────────────────── */

/**
 * @brief Per-layer forward pass on GPU.
 */
vv_status_t vv_decoder_layer_forward(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,       /**< [seq_len, hidden_size] FP16, in/out */
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    void* temp_workspace,
    size_t workspace_size,
    void* stream);

/**
 * @brief Prefill through layers [first_layer, first_layer + n_layers).
 *
 * The range is the whole model unless it has been sharded across devices,
 * in which case each shard runs its own slice over the whole sequence and
 * hands the hidden state on. Running a layer for every position before
 * moving to the next layer is the same computation as running every layer
 * for one chunk of positions: causality lives inside a layer, and the cache
 * a layer reads is the one it just wrote.
 *
 * The sequence is appended at `kv_cache->current_len`: on an empty cache
 * that is a prompt, after decode steps it is the next chunk of a streaming
 * session. The device-side length the decode step reads is not touched;
 * call vv_kv_cache_publish_len() before decoding again.
 *
 * @param pool  Optional layer pool for streaming (NULL = weights on GPU).
 * @param xfer  Transfer stream for async upload (NULL = use compute).
 */
vv_status_t vv_decoder_prefill(
    vv_model_t* model,
    void* hidden_states,       /**< [seq_len, hidden_size] FP16, in/out */
    int seq_len,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    int first_layer,
    int n_layers);

/**
 * @brief vv_decoder_prefill that also fills `taps` (NULL: none) with the
 *        outputs of the tapped layers in [first_layer, first_layer +
 *        n_layers) for every row it prefills.
 */
vv_status_t vv_decoder_prefill_taps(
    vv_model_t* model,
    void* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    int first_layer,
    int n_layers,
    const vv_taps_t* taps);

/**
 * @brief `rows` (<= 16) tokens at the cache's current length, every row
 *        computed exactly as the decode step at its position would.
 *
 * What checks a drafted block (spec.h): the rows go through the same kernels
 * a decode step runs -- multi-row versions where a weight should be read
 * once -- so their logits carry the bits `rows` decode steps would have.
 * The whole model on one device (no shards). Appends `rows` positions;
 * `taps` (NULL: none) receive them at position - taps->row_base.
 */
vv_status_t vv_decoder_verify(
    vv_model_t* model,
    void* hidden_states,       /**< [rows, hidden_size] FP16, in/out */
    int rows,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    const vv_taps_t* taps);

/**
 * @brief The most rows vv_decoder_verify keeps exact on `model`.
 *
 * 16, except where layers run W4A8: past the rows its GEMV takes (8), the
 * activations go to the GEMM, which adds the weight groups up in another
 * order than the decode step does.
 */
int vv_decoder_verify_rows_max(const vv_model_t* model);

/**
 * @brief One decode step through layers [first_layer, first_layer+n_layers).
 * @param pool  Optional layer pool for streaming (NULL = weights on GPU).
 * @param xfer  Transfer stream for async upload (NULL = use compute).
 */
vv_status_t vv_decoder_step(
    vv_model_t* model,
    void* hidden_state,        /**< [1, hidden_size] FP16, in/out */
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    int first_layer,
    int n_layers);

/**
 * @brief vv_decoder_step that also copies the tapped layers' outputs for its
 *        one position into row 0 of `taps` (a speculative decoder's plain
 *        step, which its drafter must see like any other). The destination
 *        does not depend on the position, so the step stays capturable.
 */
vv_status_t vv_decoder_step_taps(
    vv_model_t* model,
    void* hidden_state,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream,
    int first_layer,
    int n_layers,
    const vv_taps_t* taps);

/* ─── CPU-mode decoder (Phase 3) ───────────────────────────────────────── */

/**
 * @brief Prefill on the CPU, appended to what the cache already holds.
 *
 * Like the GPU prefill, positions start at `kv_cache->current_len`, and the
 * sequence runs in chunks sized to the workspace, so the prompt length is
 * bounded by the KV window rather than by 512 MB of activations.
 */
vv_status_t vv_decoder_prefill_cpu(
    vv_model_t* model,
    float* hidden_states,      /**< [seq_len, hidden_size] FP32, in/out */
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size);

/** @brief vv_decoder_prefill_cpu that also fills `taps` (FP32 rows). */
vv_status_t vv_decoder_prefill_cpu_taps(
    vv_model_t* model,
    float* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size,
    const vv_taps_t* taps);

vv_status_t vv_decoder_step_cpu(
    vv_model_t* model,
    float* hidden_state,       /**< [1, hidden_size] FP32, in/out */
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size);

/* ─── asr-bitnet on the CPU (bitnet_lm.c) ──────────────────────────────── */

/**
 * @brief One ternary (BitNet) transformer layer over T rows at positions
 *        pos0.., computed as VibeASR.cpp's ggml graph does (see the file
 *        comment for the operation order). FP32 norms and biases, an FP32
 *        cache holding F16-rounded values.
 */
vv_status_t vv_bitnet_layer_cpu(const vv_layer_weights_t* L,
                                const vv_llm_config_t* c, float* h,
                                vv_kv_cache_t* kv, int layer, int pos0, int T,
                                void* ws, size_t ws_bytes);

/** @brief Workspace bytes vv_bitnet_layer_cpu needs for T rows over a cache
 *         of up to kv_max positions. */
size_t vv_bitnet_layer_cpu_bytes(const vv_llm_config_t* c, int T, int kv_max);

/** @brief ggml's rms_norm then mul: sum of squares in double, FP32 weight. */
void vv_bitnet_rmsnorm_cpu(const float* x, const float* w, float* y, int T,
                           int n, float eps);

/**
 * @brief Greedy token from an F16 head, as the reference's F16 mul_mat
 *        computes the logits: the activation rounded to F16, then
 *        ggml_vec_dot_f16 (4 x 8 FMA lanes, its reduction); ties go to the
 *        lowest id, as llama.cpp's greedy sampler. K % 32 == 0, K <= 8192.
 */
vv_status_t vv_bitnet_head_f16_argmax_cpu(const float* x, const uint16_t* w,
                                          int V, int K, int32_t* token,
                                          float* value);

/**
 * @brief Row-quantize an F16 head to int8 and record, per row, the bounds
 *        vv_bitnet_head_filtered_argmax_cpu() needs: ||w - s q||_2,
 *        s ||q||_2 and ||w||_2 (rounded up), into bound[V][3].
 */
vv_status_t vv_bitnet_head_filter_build(const uint16_t* w, int V, int K,
                                        int8_t* q, float* scale, float* bound);

/**
 * @brief The same token as vv_bitnet_head_f16_argmax_cpu(), reading the
 *        int8 copy of the head instead of all of the F16 one.
 *
 * Every row's int8 logit comes with a rigorous bound on its distance to the
 * F16 logit the reference computes (weight and activation rounding by
 * Cauchy-Schwarz, plus the float error of the F16 dot's FMA lanes). Rows whose
 * upper bound falls below the best lower bound cannot win; the rest -- a
 * handful per token -- are scored exactly as the full F16 scan scores them.
 * So the argmax, ties included, is the full scan's, and a decode step reads
 * 233 MB of head instead of 467 MB.
 *
 * @param scratch  at least vv_bitnet_head_filter_scratch(V, K) bytes
 * @param n_exact  out, rows scored in F16; may be NULL
 */
vv_status_t vv_bitnet_head_filtered_argmax_cpu(
    const float* x, const uint16_t* w, const int8_t* q, const float* scale,
    const float* bound, int V, int K, void* scratch, size_t scratch_bytes,
    int32_t* token, float* value, int* n_exact);

/** @brief Scratch bytes vv_bitnet_head_filtered_argmax_cpu() needs. */
size_t vv_bitnet_head_filter_scratch(int V, int K);

/* ─── Sampling ──────────────────────────────────────────────────────────── */

/**
 * @brief Greedy decode: argmax of logits.
 */
vv_status_t vv_sample_greedy(const void* logits_fp16, int vocab_size,
                              int32_t* token_id);

/**
 * @brief Top-k sampling.
 */
vv_status_t vv_sample_topk(const void* logits_fp16, int vocab_size,
                             int k, float temperature, int32_t* token_id);

/**
 * @brief Draw one token from FP32 logits: temperature, top_k, then nucleus.
 *
 * `temperature` <= 0 is greedy. `rng` carries the stream, so the same seed
 * gives the same answer. Nucleus is applied inside the `top_k` candidates.
 */
vv_status_t vv_sample_logits_f32(const float* logits, int vocab_size,
                                 float temperature, float top_p, int top_k,
                                 uint64_t* rng, int32_t* token_id);

/* ─── Layer shards ──────────────────────────────────────────────────────── */

/**
 * @brief A slice of the transformer living on a device other than the primary.
 *
 * The primary device keeps the embedding table, the final norm, the LM head
 * and the first slice of layers, and it is where a request starts and ends.
 * Each further shard owns its own layers, the KV cache for exactly those
 * layers, its own workspace and streams, and a buffer for the hidden state
 * that arrives from the shard before it.
 *
 * What crosses a boundary is one hidden state — 3584 halves, 7 KB — per
 * token per boundary, against the ~8 ms of compute that token costs. That
 * is why this is worth doing at all: the devices take turns rather than
 * working at once, so it buys capacity, not speed.
 */
typedef struct vv_shard {
    int              gpu_id;
    int              first_layer;
    int              n_layers;
    int              n_resident;      /**< of those, held in VRAM          */
    void*            compute_stream;
    void*            transfer_stream;
    void*            workspace;
    size_t           workspace_size;
    vv_kv_cache_t*   kv_cache;        /**< this shard's layers only        */
    vv_layer_pool_t* layer_pool;
    void*            hidden;          /**< the state handed to this shard  */
    size_t           hidden_bytes;
    void*            done;            /**< event: its layers have finished */
} vv_shard_t;

/* ─── Full inference context ────────────────────────────────────────────── */

/* Forward declarations for opaque types */
struct vv_tokenizer;
struct vv_conv_vae_encoder;
struct vv_frontend;
struct vv_frontend_stream;
struct vv_connector;

typedef struct vv_inference_ctx {
    vv_model_t*    model;
    vv_kv_cache_t* kv_cache;
    void*          compute_stream;
    void*          transfer_stream;
    void*          workspace;
    size_t         workspace_size;
    int            gpu_id;
    int            gpu_index;         /**< this device's slot in the set  */
    vv_gpu_set_t   gpus;              /**< devices and their memory caps  */

    /*
     * Layers 0..primary_layers-1 run on `gpu_id` with the fields above;
     * `shards` covers every device after that. With one device n_shards is
     * 0, primary_layers is every layer, and not one line below changes.
     */
    vv_shard_t*    shards;
    int            n_shards;
    int            primary_layers;
    void*          shard_done;        /**< event on the primary's stream  */

    /* Placement strategy (auto-selected from VRAM budget) */
    vv_placement_t placement;
    bool           use_gpu;           /**< false for CPU-only mode */
    int            n_resident_layers; /**< Layers held on the GPU          */
    int            auto_resident_layers; /**< What the budget allowed      */

    /* GPU-resident weight buffers (NULL if offloaded to CPU) */
    void*          embed_table_gpu;
    void*          lm_head_gpu;
    void*          final_norm_gpu;

    /* Layer weight streaming pool */
    vv_layer_pool_t* layer_pool;

    /* Text tokenizer (cached for decode loop) */
    struct vv_tokenizer* tokenizer;

    /*
     * Prompt, stop tokens and generation mode for this model, resolved
     * against the tokenizer once. `family_ok` is false when the tokenizer
     * could not supply what the family needs; transcription then refuses.
     */
    vv_family_t    family;
    bool           family_ok;

    /* Audio encoding components (host descriptions; owned by the parent) */
    struct vv_conv_vae_encoder* acoustic_encoder;
    struct vv_conv_vae_encoder* semantic_encoder;
    struct vv_connector*        acoustic_connector;
    struct vv_connector*        semantic_connector;

    /*
     * The device's speech front end — encoder and connector weights, the
     * arena, the batching service — is one per device: the parent owns it
     * and clones borrow it. What each context owns is its streaming state
     * and the two events that order the front end against its own stream.
     */
    struct vv_frontend*        frontend;
    struct vv_frontend_stream* fe_stream;
    void*                      fe_ready;   /**< event: prompt rows embedded */
    void*                      fe_done;    /**< event: audio rows written   */
    /** Latents + FP32 staging for the acoustic draw and dumps, grown to the
        longest clip seen and kept, so those modes do not allocate per request. */
    void*                      fe_lat_buf;
    size_t                     fe_lat_bytes;
    /** Device buffers a closed streaming session left for the next one on
        this context (stream_ctx.c), so opening and closing a live session
        allocates and frees nothing on the device. */
    void*                      stream_bufs;

    /* Model directory path (for loading tokenizer) */
    char           model_dir[512];

    /* Performance metrics from last transcribe() call */
    vv_perf_metrics_t last_perf;

    /*
     * A clone borrows the parent's weights (model, embed, lm_head, norm) and
     * owns only its own KV cache, workspace and streams, so several requests
     * can run concurrently against one copy of the 3.2 GB of weights.
     */
    bool           is_clone;

    /*
     * Live token echo is worth watching on a long file and unreadable the
     * moment two requests share the terminal, so a pool of more than one slot
     * turns it off.
     */
    bool           quiet;

    /*
     * Hidden-state taps every prefill of this context fills while set: the
     * drafter's context features, or a trace (spec.h). NULL otherwise. Not
     * supported on a model split across devices.
     */
    const vv_taps_t* taps;

    /*
     * Speculative decoding (spec.h): the drafter's weights, loaded by the
     * parent and borrowed by its clones, and this context's own drafter
     * state. Both NULL without --draft.
     */
    struct vv_drafter* drafter;
    struct vv_spec*    spec;
    int                draft_block;   /**< rows checked per block, for clones */
    int                draft_check;   /**< vv_draft_check_t, for clones       */
} vv_inference_ctx_t;

/** Bytes of joined hotwords a prompt takes, on every entry point. */
#define VV_HOTWORDS_MAX 1024

/**
 * @brief Hotwords joined into the prompt's context ("A, B, C"), the one way
 *        every path (file, SSE, WebSocket, CLI) does it.
 *
 * Whole words only: one that does not fit `cap` (at most VV_HOTWORDS_MAX)
 * is dropped with everything after it. Empty entries are skipped.
 *
 * @return bytes written, without the terminator
 */
size_t vv_hotwords_join(const char* const* words, int n, char* buf,
                        size_t cap);

/** @brief vv_hotwords_join() over a comma-separated list ("A,B, C"),
 *         entries trimmed of surrounding spaces. */
size_t vv_hotwords_join_csv(const char* csv, char* buf, size_t cap);

vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               const vv_init_params_t* params,
                               vv_inference_ctx_t** ctx);

/**
 * @brief Create a context sharing `parent`'s GPU weights.
 *
 * Only valid when the parent placed every layer on the GPU; a streaming
 * parent has one staging pool that two threads cannot share. The clone gets
 * its own KV cache, workspace, streams, tokenizer and speech encoders.
 * Freeing a clone leaves the parent intact; free clones before the parent.
 */
vv_status_t vv_inference_clone(const vv_inference_ctx_t* parent,
                                const vv_init_params_t* params,
                                vv_inference_ctx_t** out);
vv_status_t vv_inference_transcribe(
    vv_inference_ctx_t* ctx,
    const float* audio_samples, int num_samples,
    const vv_inference_params_t* params,
    vv_transcription_t** result);
vv_status_t vv_inference_free(vv_inference_ctx_t* ctx);
vv_status_t vv_transcription_free(vv_transcription_t* result);

/**
 * @brief Attach the generated ids to a transcription (copied): `n` ids in
 *        `n_chunks` runs of `chunk_n[i]`, run i stopped by `chunk_stop[i]`
 *        (-1: by the token cap).
 */
vv_status_t vv_transcription_set_tokens(vv_transcription_t* tr,
                                        const int32_t* ids, int n,
                                        const int* chunk_n,
                                        const int32_t* chunk_stop,
                                        int n_chunks);

/* ─── Text generation (the LM without the audio) ────────────────────────── */

/**
 * @brief What to generate from a prompt that is already text.
 *
 * The caller has applied whatever template the model expects; this is the
 * plain prompt → tokens → text path the chat endpoint runs on. `on_text` is
 * called with each new piece of text as it is decoded, for streaming; it may
 * return false to stop the generation (a client that went away).
 */
typedef struct vv_generate_params {
    const char* prompt;          /**< Prompt text, already templated       */
    int         max_tokens;      /**< Cap on new tokens. Default: 512      */
    float       temperature;     /**< 0 = greedy (the default)             */
    float       top_p;           /**< Nucleus, within top_k. Default: 1    */
    int         top_k;           /**< Candidates kept. Default: 64         */
    uint64_t    seed;            /**< 0 = pick one                         */
    const char* const* stop;     /**< Stop strings, beyond the model's own */
    int         n_stop;
    bool      (*on_text)(void* user, const char* delta);
    void*       user;
} vv_generate_params_t;

static inline vv_generate_params_t vv_generate_params_default(void) {
    vv_generate_params_t p;
    memset(&p, 0, sizeof(p));
    p.max_tokens = 512;
    p.top_p = 1.0f;
    p.top_k = 64;
    return p;
}

/** @brief What a generation produced. Free with vv_generation_free(). */
typedef struct vv_generation {
    char*       text;
    int         prompt_tokens;
    int         completion_tokens;
    const char* finish_reason;   /**< "stop" | "length" | "cancelled"      */
} vv_generation_t;

/**
 * @brief Run the language model on text alone: prompt in, answer out.
 *
 * Shares the context's KV cache with transcription, so a slot does one or
 * the other at a time. The speech front end is untouched.
 */
vv_status_t vv_inference_generate(vv_inference_ctx_t* ctx,
                                  const vv_generate_params_t* params,
                                  vv_generation_t** out);

void vv_generation_free(vv_generation_t* g);

/**
 * @brief Parse the model's JSON answer into a structured transcription.
 *
 * Falls back to returning the raw text with no segments when the answer is
 * not the expected JSON array.
 */
vv_status_t vv_postprocess_text(const char* text, float audio_duration,
                                 vv_transcription_t** result);

/**
 * @brief Get performance metrics from the last transcribe() call.
 */
const vv_perf_metrics_t* vv_inference_get_perf(const vv_inference_ctx_t* ctx);

struct vv_smooth_stats;

/**
 * @brief SmoothQuant calibration: transcribe `pcm` with the dense model and
 *        collect the per-channel range of every projection input.
 *
 * Loads `model_dir` unquantized on one device (or the CPU, as `params`
 * says), runs the clip greedily and frees the model again; pass the result
 * as vv_init_params_t.smooth to a quantized load of the same checkpoint.
 * `pcm` is mono at `sample_rate`, prepared like any input. A dense 7B needs
 * ~16 GB of VRAM (or streams its layers); the clip can be short: ranges
 * settle within a minute of speech.
 *
 * @param out  statistics; free with vv_smooth_stats_free
 */
vv_status_t vv_smooth_calibrate(const char* model_dir, int gpu_id,
                                const vv_init_params_t* params,
                                const float* pcm, int n_samples,
                                int sample_rate,
                                struct vv_smooth_stats** out);

/**
 * @brief What the CLI's --calib / --calib-stats ask for.
 *
 * Neither: `*out` = NULL. `stats_path` alone: read it. `calib_audio`:
 * calibrate on that file, and write the statistics to `stats_path` when
 * one is given, so the next load can skip the dense pass.
 */
vv_status_t vv_smooth_from_args(const char* model_dir, int gpu_id,
                                const vv_init_params_t* params,
                                const char* calib_audio,
                                const char* stats_path,
                                struct vv_smooth_stats** out);

#ifdef __cplusplus
}
#endif

#endif /* VV_INFERENCE_H */
