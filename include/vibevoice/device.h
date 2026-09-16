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
vv_status_t vv_dev_memset(void* ptr, int value, size_t size);
vv_status_t vv_dev_stream_create(void** stream);
vv_status_t vv_dev_stream_destroy(void* stream);
vv_status_t vv_dev_stream_sync(void* stream);
vv_status_t vv_dev_set_device(int device_id);
vv_status_t vv_dev_get_device_info(int device_id, size_t* total_mem,
                                   size_t* free_mem, int* sm_count);

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

vv_status_t vv_rope_dev(
    void* x, int seq_len, int n_heads, int head_dim,
    int position_offset, float theta, void* stream);

vv_status_t vv_swiglu_dev(
    const void* gate, const void* up, void* output,
    int n_elements, void* stream);

/** @brief Fused SwiGLU where gate and up are adjacent halves of one buffer. */
vv_status_t vv_swiglu_fused_dev(
    const void* gate_up, void* output, int rows, int inter, void* stream);

vv_status_t vv_residual_add_dev(void* x, const void* y, int total, void* stream);
vv_status_t vv_bias_add_dev(void* output, const void* bias,
                            int M, int N, void* stream);

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
 * @param scratch  vv_gqa_decode_scratch_bytes() of device memory, owned by
 *                 the caller and not touched by any other stream.
 */
vv_status_t vv_gqa_attention_decode_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len,
    void* scratch, void* stream);

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
    int kv_format, void* scratch, void* stream);

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
 */
vv_status_t vv_kv_quant_store_dev(
    const void* k_fp16, const void* v_fp16,
    void* k_store, void* v_store, void* k_meta, void* v_meta,
    void* k_ref, bool build_ref,
    int n_kv_heads, int head_dim, int pos,
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

/* ─── Conv-VAE speech tokenizer ──────────────────────────────────────────── */

vv_status_t vv_conv1d_dev(
    const void* input_fp16, const void* weight_fp16, const void* bias_fp16,
    void* output_fp16, int in_channels, int in_length, int out_channels,
    int kernel_size, int stride, int groups, bool causal,
    int* out_length, void* stream);

/** @brief conv1d with the padding and output length given explicitly. */
vv_status_t vv_conv1d_raw_dev(
    const void* input_fp16, const void* weight_fp16, const void* bias_fp16,
    void* output_fp16, int in_channels, int in_length, int out_channels,
    int kernel_size, int stride, int groups, int pad_left, int out_length,
    void* stream);

vv_status_t vv_rmsnorm_channel_first_dev(
    const void* input, const void* weight, void* output,
    int channels, int length, float eps, void* stream);

vv_status_t vv_residual_add_scaled_dev(
    void* x, const void* y, const void* scale,
    int channels, int length, void* stream);

vv_status_t vv_channel_bias_add_dev(void* output, const void* bias,
                                    int channels, int length, void* stream);

vv_status_t vv_silu_dev(void* data, int total, void* stream);
vv_status_t vv_gelu_dev(void* data, int total, void* stream);

vv_status_t vv_fp32_to_fp16_dev(const void* in_fp32, void* out_fp16,
                                int n, void* stream);
vv_status_t vv_fp16_to_fp32_dev(const void* in_fp16, void* out_fp32,
                                int n, void* stream);

vv_status_t vv_gather_tile_dev(const void* src, void* dst, int channels,
                               int full_len, int tile_offset, int tile_len,
                               void* stream);
vv_status_t vv_scatter_tile_dev(const void* src, void* dst, int channels,
                                int full_len, int tile_offset, int tile_len,
                                void* stream);

#ifdef __cplusplus
}
#endif

#endif /* VV_DEVICE_H */
