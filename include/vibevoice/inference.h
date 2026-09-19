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

vv_status_t vv_decoder_step_cpu(
    vv_model_t* model,
    float* hidden_state,       /**< [1, hidden_size] FP32, in/out */
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size);

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
} vv_inference_ctx_t;

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

#ifdef __cplusplus
}
#endif

#endif /* VV_INFERENCE_H */
