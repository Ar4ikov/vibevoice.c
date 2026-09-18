/**
 * @file device.h
 * @brief The accelerator op set — one name per kernel, one backend per build.
 *
 * Every GPU kernel the pipeline needs is declared here exactly once. A build
 * links exactly one implementation of these symbols:
 *
 *   src/cuda/*.cu    NVIDIA, via CUDA          (VV_HAS_CUDA)
 *   src/metal/*.m    Apple Silicon, via Metal  (VV_HAS_METAL)
 *
 * There is no dispatch table: the two backends never coexist in one binary,
 * so the linker picks the implementation and the call costs nothing. A build
 * with neither backend still works — the pipeline falls back to the CPU
 * kernels in cpu_kernels.h, as it does at runtime when no device is present.
 *
 * Conventions shared by every op:
 *   - `void* stream` is a CUDA stream or a Metal command queue; NULL = default
 *   - tensors are FP16 unless the name says otherwise, row-major, contiguous
 *   - weights are [out_features, in_features], matching PyTorch
 *   - returns VV_OK or a vv_status_t error; no op throws or aborts
 */
#ifndef VV_DEVICE_H
#define VV_DEVICE_H

#include "vibevoice/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(VV_HAS_CUDA) || defined(VV_HAS_METAL)
#  define VV_HAS_ACCEL 1
#endif

/* ─── Device / memory / streams ──────────────────────────────────────────── */

vv_status_t vv_dev_alloc(void** ptr, size_t size);
vv_status_t vv_dev_alloc_pinned(void** ptr, size_t size);
vv_status_t vv_dev_free(void* ptr);
vv_status_t vv_dev_free_pinned(void* ptr);
vv_status_t vv_dev_memcpy_h2d(void* dst, const void* src, size_t size, void* stream);
vv_status_t vv_dev_memcpy_d2h(void* dst, const void* src, size_t size, void* stream);
vv_status_t vv_dev_memcpy_d2d(void* dst, const void* src, size_t size, void* stream);

/**
 * @brief Copy one row into `dst_base + (*d_index) * size`.
 *
 * The same copy as vv_dev_memcpy_d2d with the destination index read on the
 * device, so that a graph replay writes to this token's slot rather than to
 * the one its capture happened to see.
 */
vv_status_t vv_dev_memcpy_d2d_at(void* dst_base, const void* src, size_t size,
                                 const int* d_index, void* stream);
vv_status_t vv_dev_memset(void* ptr, int value, size_t size);

/**
 * @brief vv_dev_memset, ordered on a stream rather than device-wide.
 *
 * The unstreamed form runs on the legacy default stream, which synchronises
 * with every blocking stream in the process: on anything with more than one
 * request in flight it stalls the others and invalidates a graph capture.
 */
vv_status_t vv_dev_memset_async(void* ptr, int value, size_t size,
                                void* stream);
vv_status_t vv_dev_stream_create(void** stream);
vv_status_t vv_dev_stream_destroy(void* stream);
vv_status_t vv_dev_stream_sync(void* stream);
vv_status_t vv_dev_set_device(int device_id);
vv_status_t vv_dev_get_device_info(int device_id, size_t* total_mem,
                                   size_t* free_mem, int* sm_count);

/**
 * @brief Devices this process can use, or 0 when there is no usable driver.
 *
 * Counts what CUDA reports, which is already narrowed by
 * CUDA_VISIBLE_DEVICES, so device 0 here is device 0 everywhere else in the
 * process and not necessarily the first card in the machine.
 */
int vv_dev_device_count(void);

/** @brief The device's name, e.g. "NVIDIA GeForce RTX 3090". */
vv_status_t vv_dev_get_device_name(int device_id, char* buf, size_t buf_size);

/**
 * @brief Copy between two devices, ordered on `stream`.
 *
 * Works whether or not the two can address each other: without peer access
 * the driver stages through host memory, which is slower but not wrong. The
 * stream belongs to whichever device is current; ordering against work on
 * the *other* device is the caller's business, through an event.
 */
vv_status_t vv_dev_memcpy_peer(void* dst, int dst_device,
                               const void* src, int src_device,
                               size_t size, void* stream);

/**
 * @brief Let `device` read `peer` directly, if the pair supports it.
 *
 * Returns VV_ERR_UNSUPPORTED when they do not, which is not a failure —
 * vv_dev_memcpy_peer still works, just through the host.
 */
vv_status_t vv_dev_enable_peer(int device, int peer);

/* Events order the copy stream against the compute stream when weights are
 * streamed layer by layer; host registration makes those copies DMA. */
vv_status_t vv_dev_event_create(void** ev);
vv_status_t vv_dev_event_destroy(void* ev);
vv_status_t vv_dev_event_record(void* ev, void* stream);
vv_status_t vv_dev_stream_wait_event(void* stream, void* ev);
vv_status_t vv_dev_host_register(void* p, size_t n);
vv_status_t vv_dev_host_unregister(void* p);

/** @brief Short name of the compiled-in backend ("CUDA", "Metal", "none"). */
const char* vv_dev_backend_name(void);

/* ─── Graph capture ──────────────────────────────────────────────────────── */

/*
 * A decode token issues around 450 kernels, and consecutive kernels on one
 * stream cannot overlap, so each pays a dispatch gap whether or not the host
 * keeps up. Recording the sequence once and replaying it submits the whole
 * thing at once.
 *
 * Between begin and end the stream may receive only capturable work: no
 * synchronisation, no allocation, nothing read back to the host, and no
 * pageable host memory (which drains the stream behind your back). Anything
 * whose kernel arguments change from one token to the next has to read them
 * from device memory instead, because a replay reuses the arguments it
 * recorded.
 */

/** @brief Start recording work issued to `stream` instead of running it. */
vv_status_t vv_dev_graph_begin(void* stream);

/** @brief Stop recording and instantiate what was recorded. */
vv_status_t vv_dev_graph_end(void* stream, void** graph_exec);

/** @brief Submit a recorded graph to `stream`. */
vv_status_t vv_dev_graph_launch(void* graph_exec, void* stream);

void vv_dev_graph_destroy(void* graph_exec);

/** @brief `*dst = *src + delta`, on the stream. Both are device ints. */
vv_status_t vv_pos_add_dev(int* dst, const int* src, int delta, void* stream);

/* ─── Quantized linear ───────────────────────────────────────────────────── */

/** @brief NF4 dequant fused into a GEMV (M = 1). Bias may be NULL. */
vv_status_t vv_nf4_gemv_dev(
    const void* x, const uint8_t* packed, const void* scales,
    const void* bias, void* y, int N, int K, void* stream);

/** @brief NF4 dequant to `temp_weight`, then FP16 GEMM. */
vv_status_t vv_nf4_gemm_dev(
    const void* input_fp16, const uint8_t* weight_packed,
    const void* weight_scales_fp16, void* output_fp16,
    void* temp_weight_fp16, int M, int N, int K,
    int block_size, void* stream);

/** @brief Standalone NF4 → FP16 dequantization. */
vv_status_t vv_dequant_nf4_dev(
    const uint8_t* packed, const void* scales_fp16,
    void* output_fp16, int n_elements, int block_size, void* stream);

/*
 * INT4 group-affine weights (the runtime form of an AWQ or GPTQ checkpoint,
 * repacked at load time by vv_awq_repack):
 *   packed [N][K/2] uint8, scales [N][K/G] FP16, mins [N][K/G] FP16
 * with w = q * scale + min, so dequantisation is a single FMA.
 */

/** @brief INT4 dequant fused into a GEMV (M = 1). Bias may be NULL. */
vv_status_t vv_awq_gemv_dev(
    const void* x, const uint32_t* packed, const uint32_t* mins,
    const void* scales, const void* bias, void* y,
    int N, int K, int group_size, void* stream);

/** @brief INT4 GEMM: a direct kernel for M <= 8, else dequant + FP16 GEMM. */
vv_status_t vv_awq_gemm_dev(
    const void* input_fp16, const uint32_t* packed, const uint32_t* mins,
    const void* scales, void* output_fp16, void* temp_weight_fp16,
    int M, int N, int K, int group_size, void* stream);

/** @brief Standalone INT4 → FP16 dequantization into [N, K]. */
vv_status_t vv_dequant_awq_dev(
    const uint32_t* packed, const uint32_t* mins, const void* scales,
    void* output_fp16, int N, int K, int group_size, void* stream);

/*
 * INT8 per-channel weights (`--quant int8`), run on FP16 activations:
 *   q [N][K] int8, scales [N] FP32, w = q * scale. K must be a multiple of 16.
 */

/** @brief INT8 GEMV (M = 1), scale and bias applied once per row. Bias may
 *         be NULL. */
vv_status_t vv_int8_gemv_dev(
    const void* x, const int8_t* q, const float* scales, const void* bias,
    void* y, int N, int K, void* stream);

/** @brief INT8 GEMM without bias: a direct kernel for M <= 8, else dequant
 *         into `temp_weight_fp16` [N, K] + FP16 GEMM. */
vv_status_t vv_int8_gemm_dev(
    const void* input_fp16, const int8_t* q, const float* scales,
    void* output_fp16, void* temp_weight_fp16,
    int M, int N, int K, void* stream);

/** @brief Standalone INT8 → FP16 dequantization into [N, K]. */
vv_status_t vv_dequant_int8_dev(
    const int8_t* q, const float* scales, void* output_fp16,
    int N, int K, void* stream);

/*
 * INT4 group-affine weights in the GPU layout (vv_int4g_to_gpu_layout):
 *   packed [N][K/2] bytes, rows contiguous in k, nibbles permuted inside
 *          each 16-byte chunk so one LOP3 yields MMA-ready half2 pairs;
 *   sz     [N][K/G] half2 = { scale, exact integer zero point }.
 * Dequantization is w = (q - z) * s in FP16, the reference's formula.
 * K must be a multiple of 32 (64 for the tensor-core path), N of 8, and
 * G one of 32, 64, 128, 256.
 */

/**
 * @brief y[N] = W . x (+ bias) for one token. No allocation, no host-side
 *        per-token arguments: safe to capture into the decode graph.
 */
vv_status_t vv_w4a16_gemv_dev(
    const void* x, const void* packed, const void* sz, const void* bias,
    void* y, int N, int K, int group_size, void* stream);

/** @brief One projection of a fused GEMV (see vv_w4a16_gemv_multi_dev). */
typedef struct vv_w4a16_proj {
    const void* packed;   /**< GPU-layout codes [N][K/2]                   */
    const void* sz;       /**< half2 {scale, zero} [N][K/G]                */
    const void* bias;     /**< FP16 [N] or NULL                            */
    void*       y;        /**< FP16 [N] output                             */
    int         N;        /**< output rows                                 */
} vv_w4a16_proj_t;

/**
 * @brief Up to three GEMVs of the same x (and K, group size) in one launch,
 *        e.g. q/k/v or gate/up. Each row is computed exactly as a single
 *        vv_w4a16_gemv_dev over the combined row count would compute it.
 *        Graph-capturable like vv_w4a16_gemv_dev.
 */
vv_status_t vv_w4a16_gemv_multi_dev(
    const void* x, const vv_w4a16_proj_t* projs, int n_proj,
    int K, int group_size, void* stream);

/**
 * @brief out[m][j] = x[m][perm[j]] (FP16, K % 8 == 0, perm 16-byte aligned).
 *        Puts activations in the column order of an act-order (desc_act)
 *        GPTQ weight, whose input channels were sorted by group at load.
 */
vv_status_t vv_w4a16_gather_dev(
    const void* x, const int32_t* perm, void* out, int M, int K,
    void* stream);

/**
 * @brief C[M,N] = A[M,K] . W^T (+ bias). Small M runs the GEMV kernel,
 *        larger M a fused tensor-core GEMM (sm_80+) that keeps the weights
 *        packed until they are in registers.
 *
 * `scratch` holds split-K partials (M*N*4 bytes per slice; the split is
 * reduced to what fits) and, on devices without the tensor-core path, the
 * dequantized weight (N*K*2 bytes).
 */
vv_status_t vv_w4a16_gemm_dev(
    const void* A, const void* packed, const void* sz, const void* bias,
    void* C, void* scratch, size_t scratch_bytes,
    int M, int N, int K, int group_size, void* stream);

/** @brief GPU-layout INT4 → dense FP16 [N, K]. */
vv_status_t vv_w4a16_dequant_dev(
    const void* packed, const void* sz, void* out_fp16,
    int N, int K, int group_size, void* stream);

/* ─── Dense linear ───────────────────────────────────────────────────────── */

/** @brief C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C. */
vv_status_t vv_gemm_fp16_dev(
    const void* A, const void* B, void* C,
    int M, int N, int K, float alpha, float beta, void* stream);

/** @brief C[M,P] = alpha * A[M,K] @ B[K,P] + beta * C. */
vv_status_t vv_gemm_fp16_nn_dev(
    const void* A, const void* B, void* C,
    int M, int K, int P, float alpha, float beta, void* stream);

/** @brief Retained for API symmetry; no handle to release. */
void vv_gemm_cleanup(void);

/* ─── Transformer ops ────────────────────────────────────────────────────── */

vv_status_t vv_embedding_dev(
    const void* table, const int32_t* ids, void* output,
    int seq_len, int hidden_size, void* stream);

vv_status_t vv_rmsnorm_dev(
    const void* input, const void* weight, void* output,
    int seq_len, int hidden_size, float eps, void* stream);

/**
 * @brief RoPE in place over [seq_len, n_heads, head_dim].
 * @param position_offset Position of the first row, when `d_position` is NULL.
 * @param d_position      Device int holding it instead. Decode uses this so
 *                        the launch can be captured once and replayed.
 */
vv_status_t vv_rope_dev(
    void* x, int seq_len, int n_heads, int head_dim,
    int position_offset, const int* d_position, float theta, void* stream);

vv_status_t vv_swiglu_dev(
    const void* gate, const void* up, void* output,
    int n_elements, void* stream);

/** @brief Fused SwiGLU where gate and up are adjacent halves of one buffer. */
vv_status_t vv_swiglu_fused_dev(
    const void* gate_up, void* output, int rows, int inter, void* stream);

vv_status_t vv_residual_add_dev(void* x, const void* y, int total, void* stream);
vv_status_t vv_bias_add_dev(void* output, const void* bias,
                            int M, int N, void* stream);

/* ─── Attention: one entry point, several backends ───────────────────────── */

/**
 * @brief One layer of a KV cache as the attention kernels see it.
 *
 * `k`/`v` hold [row][n_kv_heads][bytes_per_vec] in the cache's format. With
 * `page_table` NULL, row = position. Otherwise position t lives at row
 * page_table[t / VV_KV_PAGE_SIZE] * VV_KV_PAGE_SIZE + t % VV_KV_PAGE_SIZE of
 * a pool shared with other contexts, and only the flashinfer backend reads
 * it. The table is device memory, so a replayed graph follows new pages.
 */
typedef struct vv_kv_view {
    const void* k;
    const void* v;
    const void* k_meta;       /**< per-vector FP16 scales (TurboQuant) or NULL */
    const void* v_meta;
    const int*  page_table;   /**< device, or NULL for a contiguous cache     */
    int         format;       /**< vv_kv_format_t                              */
    int         n_kv_heads;
    int         head_dim;
} vv_kv_view_t;

/**
 * @brief The backend a context should run, `auto` resolved for this device.
 *
 * Checks what the current device can execute and what the backend supports
 * (head_dim, GQA group, KV format, paging) and steps down to the next one
 * that works, logging nothing. Never returns VV_ATTN_AUTO.
 */
int vv_attn_resolve(int requested, int kv_format, bool paged,
                    int n_q_heads, int n_kv_heads, int head_dim);

/**
 * @brief Scratch one context needs for any backend's attention.
 *
 * The maximum over backends, so a context sized once can switch kernels and
 * the workspace layout does not move.
 */
size_t vv_attn_scratch_bytes(int n_q_heads, int n_kv_heads, int head_dim);

/**
 * @brief The decode attention's launch shape for `backend` at `cache_len`.
 *
 * A captured decode step replays the grid it recorded, so the caller
 * re-captures whenever this changes. Only equality means anything.
 */
int vv_attn_decode_shape(int backend, int n_q_heads, int n_kv_heads,
                         int cache_len);

/**
 * @brief `q_len` query rows at absolute positions `q_offset..` against the
 *        first `kv_len` positions of a cache (which already holds this
 *        chunk's own K and V).
 *
 * Q and O are [q_len][n_q_heads][head_dim] FP16. For a TurboQuant cache Q
 * must already be rotated and O comes back rotated (vv_kv_rotate_dev).
 * @param scratch  vv_attn_scratch_bytes() of device memory owned by the
 *                 caller; the split-KV path of flashinfer uses it.
 */
vv_status_t vv_attn_prefill(int backend, const void* q, const vv_kv_view_t* kv,
                            void* out, int n_q_heads, int q_len, int q_offset,
                            int kv_len, bool causal, void* scratch,
                            void* stream);

/**
 * @brief One query row per head against the whole cache.
 * @param cache_len    Sizes the launch; with `d_cache_len` it need only be in
 *                     the same vv_attn_decode_shape() bucket as the truth.
 * @param d_cache_len  Device int holding the exact length, or NULL.
 */
vv_status_t vv_attn_decode(int backend, const void* q, const vv_kv_view_t* kv,
                           void* out, int n_q_heads, int cache_len,
                           const int* d_cache_len, void* scratch,
                           void* stream);

/**
 * @brief Quantize (or copy, for FP16) K/V vectors into a cache layer, at
 *        rows from `page_table` when it is set.
 *
 * Same contract as vv_kv_quant_store_dev, plus FP16 and paging.
 */
vv_status_t vv_kv_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, const int* page_table, void* stream);

/**
 * @brief `page_table[first + i] = pages[i]` for i < n, ordered on `stream`.
 *
 * `pages` is host memory and is consumed before this returns. Not for use
 * inside a graph capture: the ids are baked into the launch.
 */
vv_status_t vv_kv_page_map_dev(int* page_table, int first, int n,
                               const int* pages, void* stream);

/**
 * @brief Scratch bytes the split-K decode attention needs for one caller.
 *
 * Decode splits the cache across warps and merges their online-softmax states
 * in a second pass, so the partial (o, m, l) triples have to live somewhere
 * between the two kernels. That buffer belongs to the caller: several
 * inference contexts decode concurrently on their own streams, and a shared
 * one would have them reading each other's partials.
 */
size_t vv_gqa_decode_scratch_bytes(int n_q_heads, int head_dim);

/**
 * @brief Flash decode: one query row against a KV cache of `cache_len`.
 * @param cache_len     Sizes the launch; with `d_cache_len` set it need only
 *                      be the length the capture was made at.
 * @param d_cache_len   Device int holding the exact length, or NULL to use
 *                      `cache_len` itself.
 * @param scratch       vv_gqa_decode_scratch_bytes() of device memory, owned
 *                      by the caller and not touched by any other stream.
 */
vv_status_t vv_gqa_attention_decode_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    const int* d_cache_len, void* scratch, void* stream);

/**
 * @brief The decode attention's launch shape at this cache length.
 *
 * The split count moves in steps as the cache grows, and a captured graph
 * holds only while it does not move. The caller compares this between tokens
 * and re-captures when it changes; the value itself means nothing else.
 */
int vv_gqa_decode_shape(int cache_len);

/** @brief Flash prefill of `q_len` rows at `q_offset` against a KV cache. */
vv_status_t vv_gqa_attention_prefill_cached_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal, void* stream);

/** @brief Flash prefill over a standalone (non-cached) K/V block. */
vv_status_t vv_gqa_attention_prefill_dev(
    const void* q, const void* k, const void* v, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream);

/**
 * @brief Flash decode against a quantized KV cache (see kv_quant.h).
 * @param scratch  As for vv_gqa_attention_decode_dev.
 */
vv_status_t vv_gqa_attention_decode_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    const int* d_cache_len, int kv_format, void* scratch, void* stream);

/** @brief Flash prefill against a quantized KV cache (see kv_quant.h). */
vv_status_t vv_gqa_attention_prefill_q_dev(
    const void* q, const void* k_store, const void* v_store,
    const void* k_meta, const void* v_meta, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal,
    int kv_format, void* stream);

/**
 * @brief Quantize a run of K/V vectors into the packed cache.
 *
 * @param k_ref     Per-head reference key subtracted before quantizing K.
 *                  Softmax is invariant to a constant shift of every key, so
 *                  this costs nothing in accuracy and buys a great deal: in
 *                  Qwen2 the keys share a large common component (|k| = 273
 *                  against |k - mean| = 11), and quantizing the difference
 *                  instead is the whole reason sub-byte KV is usable here.
 * @param build_ref Compute k_ref from this run first (the first prefill
 *                  chunk of a session); afterwards the same reference must
 *                  be reused for every position in the cache.
 * @param d_pos     Device int holding the first position, or NULL to use
 *                  `pos`. Decode uses this so the launch can be captured.
 */
vv_status_t vv_kv_quant_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos, const int* d_pos,
    int n_positions, int kv_format, void* stream);

/** @brief Unpack a quantized KV store back to FP16 (tests, CPU offload). */
vv_status_t vv_kv_dequant_dev(
    const void* store, const void* meta, const void* ref, void* out_fp16,
    int n_kv_heads, int head_dim, int n_pos, int kv_format, void* stream);

/* ─── LM head ────────────────────────────────────────────────────────────── */

/** @brief logits[V] = W[V,K] @ x[K], FP32 accumulation, FP32 output. */
vv_status_t vv_lm_head_gemv_dev(
    const void* x, const void* W, void* logits, int V, int K, void* stream);

/** @brief Two-stage on-device argmax; only the token id crosses the bus. */
vv_status_t vv_argmax_dev(
    const void* logits, int V, void* scratch_v, void* scratch_i,
    void* out_token, void* out_value, void* stream);

/* ─── Conversions ──────────────────────────────────────────────────────────── */

vv_status_t vv_fp32_to_fp16_dev(const void* in_fp32, void* out_fp16,
                                int n, void* stream);
vv_status_t vv_fp16_to_fp32_dev(const void* in_fp16, void* out_fp32,
                                int n, void* stream);

/* ─── Batched, streaming Conv-VAE (vae_kernels.cu) ───────────────────────── */

/*
 * Activations are channel-first and packed along time: the items of one
 * launch sit side by side in one [C][ld] buffer. Only convolutions look back
 * in time, so only they need item boundaries, which they take from a
 * descriptor table passed by value — no upload, nothing to keep alive, and a
 * launch that could be captured into a graph as it stands.
 */

/** @brief Items per launch; bounded so the table fits in the kernel params. */
#define VV_VAE_MAX_ITEMS 32

/** @brief One item's view of one convolution layer. */
typedef struct vv_vae_conv_item {
    int64_t     in_off;    /**< first input column in the packed input      */
    int64_t     out_off;   /**< first output column in the packed output    */
    const void* in_ptr;    /**< the item's own input, overriding in_off     */
    const void* tail_in;   /**< [in_ch][cap] left context; NULL = zeros     */
    void*       tail_out;  /**< [in_ch][cap] receives the next context      */
    void*       out_ptr;   /**< transposed output: rows [frame][out_ld]     */
    int         in_len;    /**< input columns this chunk                    */
    int         out_len;   /**< output columns this chunk                   */
    int         have;      /**< context columns in front of the input       */
    int         keep;      /**< context columns to leave for the next chunk */
    int         out_ld;    /**< transposed output row stride, elements      */
    int         skip;      /**< leading output frames not written           */
} vv_vae_conv_item_t;

typedef struct vv_vae_conv_desc {
    int                n;     /**< items                                    */
    int                cap;   /**< context stride per channel, columns      */
    vv_vae_conv_item_t it[VV_VAE_MAX_ITEMS];
} vv_vae_conv_desc_t;

/**
 * @brief Causal conv (groups = 1) over every item: stem, downsample, head.
 * @param transposed  Write each item's output as rows [frame][out_ch] at its
 *                    out_ptr (the head) instead of packed channel-first.
 */
vv_status_t vv_vae_conv_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, const void* w, const void* b,
                            void* y, int64_t ld_out, int in_ch, int out_ch,
                            int k, int stride, bool transposed, void* stream);

/** @brief Save each item's next left context (the last `keep` columns). */
vv_status_t vv_vae_tail_dev(const vv_vae_conv_desc_t* d, const void* x,
                            int64_t ld_in, int in_ch, void* stream);

/**
 * @brief 1/rms per packed column into rinv[0..T), and per context column of
 * the items in `d` (may be NULL) into rinv[T + item * cap + j].
 */
vv_status_t vv_vae_rms_dev(const void* x, int64_t ld, int64_t T, int C,
                           float eps, float* rinv,
                           const vv_vae_conv_desc_t* d, void* stream);

/**
 * @brief xout = xin + gamma * (dwconv(rmsnorm(xin)) + bias), fused.
 * rinv comes from vv_vae_rms_dev over the same input and descriptor.
 */
vv_status_t vv_vae_mixer_dev(const vv_vae_conv_desc_t* d, const void* xin,
                             void* xout, int64_t ld, const float* rinv,
                             int64_t T_total, const void* norm_w,
                             const void* conv_w, const void* conv_b,
                             const void* gamma, int C, int k, void* stream);

/** @brief Epilogues of vv_vae_gemm_nn_dev. */
#define VV_VAE_EPI_BIAS_GELU  0  /**< C = gelu(A @ B + bias)                 */
#define VV_VAE_EPI_BIAS_RESID 1  /**< C += gamma * (A @ B + bias), in place  */
#define VV_VAE_EPI_BIAS       2  /**< C = A @ B + bias, rounded once         */

/**
 * @brief Lay the inputs of a strided conv out as GEMM columns.
 *
 * col[ic * k + kk][q] is what tap kk of input channel ic sees for packed
 * output column p0 + q: context from the item's state, its input, or the
 * zero padding past a final chunk — the same resolution vv_vae_conv_dev
 * makes. A GEMM of the [out][in * k] weight against it is the conv.
 */
vv_status_t vv_vae_im2col_dev(const vv_vae_conv_desc_t* d, const void* x,
                              int64_t ld_in, int in_ch, int k, int stride,
                              int64_t p0, int pc, void* col, int64_t ldcol,
                              void* stream);

/**
 * @brief C[M, P] (stride ldc) from A[M, K] @ B[K, P] (stride ldb).
 * @param rinv, norm_w  Both or neither: B is RMS-normalised while it is
 *                      staged, B[k][p] * rinv[p] * norm_w[k].
 */
vv_status_t vv_vae_gemm_nn_dev(int epilogue, const void* A, const void* B,
                               int64_t ldb, void* C, int64_t ldc,
                               int M, int K, int P, const void* bias,
                               const void* gamma, const float* rinv,
                               const void* norm_w, void* stream);

/** @brief Where the rows of a TN GEMM go (speech connectors). */
#define VV_VAE_ROWS_PLAIN        0  /**< C[r][c], stride ldc                  */
#define VV_VAE_ROWS_STORE        1  /**< segment rows, as computed            */
#define VV_VAE_ROWS_STORE_TRUNC  2  /**< segment rows, FP16 truncated          */
#define VV_VAE_ROWS_ACC_TRUNC    3  /**< segment rows += value, truncated      */

typedef struct vv_vae_row_seg {
    int   row0;   /**< first packed row of this segment */
    int   ld;     /**< destination row stride, elements */
    void* dst;
} vv_vae_row_seg_t;

typedef struct vv_vae_rows {
    int              mode;
    int              n;
    vv_vae_row_seg_t seg[VV_VAE_MAX_ITEMS];
} vv_vae_rows_t;

/**
 * @brief C[M, N] = A[M, K] (stride lda) @ B[N, K]^T + bias, with the rows
 * scattered per `rows` (NULL = plain). Truncation reproduces the host-side
 * FP32 sum and float_to_half the connectors' outputs used to go through.
 */
vv_status_t vv_vae_gemm_tn_dev(const void* A, int64_t lda, const void* B,
                               void* C, int64_t ldc, int M, int N, int K,
                               const void* bias, const vv_vae_rows_t* rows,
                               void* stream);

/** @brief FP32 -> FP16 with a 64-bit count. */
vv_status_t vv_vae_f32_to_f16_dev(const float* in, void* out, int64_t n,
                                  void* stream);

/** @brief Block the calling host thread until `ev` has completed. */
vv_status_t vv_dev_event_sync(void* ev);

#ifdef __cplusplus
}
#endif

#endif /* VV_DEVICE_H */
