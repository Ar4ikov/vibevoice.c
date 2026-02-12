/**
 * @file cpu_kernels.h
 * @brief CPU compute kernels for vibevoice.c — CPU-only inference path.
 *
 * All operations are in FP32.  NF4 weights are dequantized on-the-fly.
 * When OpenBLAS is linked, SGEMM is accelerated; otherwise a naive
 * triple-loop GEMM is used.
 */
#ifndef VV_CPU_KERNELS_H
#define VV_CPU_KERNELS_H

#include "vibevoice/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── GEMM ──────────────────────────────────────────────────────────────── */

/**
 * @brief FP32 GEMM: C = alpha*A*B + beta*C  (row-major).
 * A[M,K]  B[K,N]  C[M,N]
 */
vv_status_t vv_gemm_f32_cpu(const float* A, const float* B, float* C,
                             int M, int N, int K,
                             float alpha, float beta);

/**
 * @brief NF4 dequant + FP32 GEMM.
 * output = input @ dequant(weight)^T
 * input:  [M, K] FP32
 * weight: [N, K/2] uint8 (NF4)
 * scales: [N*K / block_size] FP32 (or FP16 that we'll convert)
 * temp:   [N * K] FP32 preallocated
 */
vv_status_t vv_nf4_gemm_cpu(const float* input,
                              const uint8_t* weight_packed,
                              const float* weight_scales,
                              float* output, float* temp_weight,
                              int M, int N, int K, int block_size);

/* ─── Normalization ─────────────────────────────────────────────────────── */

/**
 * @brief RMSNorm (FP32).
 * out[i] = (x[i] / rms(x)) * weight[i]
 */
vv_status_t vv_rmsnorm_cpu(const float* input, const float* weight,
                            float* output, int seq_len, int hidden_size,
                            float eps);

/* ─── Positional encoding ───────────────────────────────────────────────── */

/**
 * @brief Apply RoPE in-place (FP32).
 */
vv_status_t vv_rope_cpu(float* x, int seq_len, int n_heads,
                         int head_dim, int position_offset, float theta);

/* ─── Attention ─────────────────────────────────────────────────────────── */

/**
 * @brief GQA attention prefill, causal, FP32.
 * q/o: [seq_len, n_q_heads, head_dim]  k/v: [seq_len, n_kv_heads, head_dim] (seq-major)
 */
vv_status_t vv_attention_prefill_cpu(
    const float* q, const float* k, const float* v, float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int seq_len, bool causal);

/**
 * @brief GQA attention decode (single query against KV-cache), FP32.
 */
vv_status_t vv_attention_decode_cpu(
    const float* q, const float* k_cache, const float* v_cache,
    float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len);

/* ─── Activations ───────────────────────────────────────────────────────── */

/** @brief SwiGLU: out[i] = silu(gate[i]) * up[i].  */
vv_status_t vv_swiglu_cpu(const float* gate, const float* up,
                           float* output, int n_elements);

/* ─── Element-wise ──────────────────────────────────────────────────────── */

/** @brief Residual add: x[i] += y[i].  */
vv_status_t vv_residual_add_cpu(float* x, const float* y, int n);

/* ─── Embedding ─────────────────────────────────────────────────────────── */

/**
 * @brief CPU embedding lookup (FP16 table → FP32 output).
 */
vv_status_t vv_embedding_cpu(const void* table_fp16, const int32_t* ids,
                              float* output, int seq_len, int hidden_size);

/**
 * @brief FP32 greedy argmax sampling.
 */
vv_status_t vv_sample_greedy_cpu(const float* logits, int vocab_size,
                                  int32_t* token_id);

#ifdef __cplusplus
}
#endif

#endif /* VV_CPU_KERNELS_H */
