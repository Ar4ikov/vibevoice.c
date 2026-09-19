/**
 * @file cpu_kernels.h
 * @brief CPU compute kernels — the path taken when there is no accelerator.
 *
 * Activations are FP32 (CPUs are fast at FP32 and it avoids a conversion in
 * every inner loop); weights stay in the form they were loaded in, NF4 or
 * INT4-group, and are dequantized inside the kernel. Nothing here
 * materialises a whole weight matrix: at 7B that would be 13 GB of FP32 per
 * forward pass, which is what made the old CPU path unusable.
 *
 * Every kernel is OpenMP-parallel over output rows or heads when OpenMP is
 * available. The quantized GEMMs have an AVX2 specialisation selected at
 * runtime, so one binary still runs on a CPU without it.
 */
#ifndef VV_CPU_KERNELS_H
#define VV_CPU_KERNELS_H

#include "vibevoice/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Name of the SIMD path in use ("AVX2", "NEON", "scalar"). */
const char* vv_cpu_simd_name(void);

/** @brief Threads the CPU kernels will use. */
int vv_cpu_threads(void);

/**
 * @brief Physical cores (one per SMT group) this process may run on, or 0
 *        when the topology cannot be read. Changes nothing: no thread count
 *        is set and nothing is pinned.
 */
int vv_cpu_physical_cores(void);

/* ─── Quantized linear ──────────────────────────────────────────────────── */

/**
 * @brief output[M,N] = input[M,K] @ dequant(W)[N,K]^T + bias.
 *
 * @param packed  NF4, 2 values per byte, high nibble first, row-major [N, K/2]
 * @param scales  FP16 absmax per 64 values, already de-nested by the loader
 * @param bias    FP16 [N], or NULL
 */
vv_status_t vv_nf4_gemm_cpu(const float* input, const uint8_t* packed,
                            const void* scales, const void* bias,
                            float* output, int M, int N, int K);

/**
 * @brief Same shape, for INT4 group-affine weights (AWQ / GPTQ).
 *
 * @param scales FP16 [N, K/group]
 * @param mins   FP16 [N, K/group], equal to -zero * scale
 */
vv_status_t vv_int4g_gemm_cpu(const float* input, const uint8_t* packed,
                              const void* scales, const void* mins,
                              const void* bias, float* output,
                              int M, int N, int K, int group);

/**
 * @brief Same shape, for per-channel INT8 weights (W8A16).
 *
 * @param q      int8 [N, K], row-major
 * @param scales FP32 [N]; w[n,k] = q[n,k] * scales[n]
 * @param bias   FP16 [N], or NULL
 */
vv_status_t vv_int8_gemm_cpu(const float* input, const int8_t* q,
                             const float* scales, const void* bias,
                             float* output, int M, int N, int K);

/** @brief output[M,N] = input[M,K] @ W[N,K]^T + bias, W and bias FP16. */
vv_status_t vv_gemm_f16w_cpu(const float* input, const void* w_fp16,
                             const void* bias_fp16, float* output,
                             int M, int N, int K);

/** @brief Plain FP32 GEMM: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C. */
vv_status_t vv_gemm_f32_cpu(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta);

/* ─── Transformer ops ───────────────────────────────────────────────────── */

/** @brief RMSNorm with an FP16 weight vector. */
vv_status_t vv_rmsnorm_cpu(const float* input, const void* weight_fp16,
                           float* output, int rows, int n, float eps);

/** @brief RoPE in place over [rows, n_heads, head_dim]. */
vv_status_t vv_rope_cpu(float* x, int rows, int n_heads, int head_dim,
                        int position_offset, float theta);

/**
 * @brief Causal GQA attention over a contiguous block.
 * q/o: [rows, n_q_heads, head_dim]   k/v: [kv_len, n_kv_heads, head_dim]
 */
vv_status_t vv_attention_prefill_cpu(
    const float* q, const float* k, const float* v, float* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int rows, int q_offset, int kv_len, bool causal);

/** @brief One query row against a cache of `cache_len` positions. */
vv_status_t vv_attention_decode_cpu(
    const float* q, const float* k_cache, const float* v_cache, float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len);

/** @brief out[i] = silu(gate[i]) * up[i]. */
vv_status_t vv_swiglu_cpu(const float* gate, const float* up,
                          float* output, int n);

/** @brief x[i] += y[i]. */
vv_status_t vv_residual_add_cpu(float* x, const float* y, int n);

/** @brief Gather rows of an FP16 embedding table into FP32. */
vv_status_t vv_embedding_cpu(const void* table_fp16, const int32_t* ids,
                             float* output, int rows, int hidden);

/**
 * @brief logits = W[V,K] @ x[K] with an FP16 weight, then argmax.
 *
 * Fused because the transcription only ever needs the winning token, and
 * 152k logits is 600 KB of traffic that nothing else reads.
 */
vv_status_t vv_lm_head_argmax_cpu(const float* x, const void* w_fp16,
                                  int V, int K, int32_t* token_id,
                                  float* out_value);

/** @brief Greedy argmax over an FP32 logit vector. */
vv_status_t vv_sample_greedy_cpu(const float* logits, int vocab_size,
                                 int32_t* token_id);

#ifdef __cplusplus
}
#endif

#endif /* VV_CPU_KERNELS_H */
