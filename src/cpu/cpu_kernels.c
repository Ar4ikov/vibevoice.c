/**
 * @file cpu_kernels.c
 * @brief CPU compute kernels for vibevoice.c — FP32 reference + OpenBLAS.
 *
 * Provides all operations needed for CPU-only inference:
 *   GEMM, NF4-dequant+GEMM, RMSNorm, RoPE, GQA Attention,
 *   SwiGLU, embedding lookup, residual add, sampling.
 *
 * When VV_HAS_OPENBLAS is defined, SGEMM delegates to cblas_sgemm.
 */

#include "vibevoice/cpu_kernels.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>

#ifdef VV_HAS_OPENBLAS
#include <cblas.h>
#endif

/** NF4 lookup table (defined in nf4_table.c) */
extern const float VV_NF4_TABLE[16];

/* ──────────────────────────── GEMM ─────────────────────────────────────── */

vv_status_t vv_gemm_f32_cpu(const float* A, const float* B, float* C,
                             int M, int N, int K,
                             float alpha, float beta) {
    if (!A || !B || !C) return VV_ERR_NULL_PTR;

#ifdef VV_HAS_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
#else
    /* Naive triple loop with beta handling */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
#endif
    return VV_OK;
}

/* ──────────────────────────── NF4 GEMM ─────────────────────────────────── */

vv_status_t vv_nf4_gemm_cpu(const float* input,
                              const uint8_t* weight_packed,
                              const float* weight_scales,
                              float* output, float* temp_weight,
                              int M, int N, int K, int block_size) {
    if (!input || !weight_packed || !weight_scales || !output || !temp_weight)
        return VV_ERR_NULL_PTR;

    /* Step 1: Dequantize weight [N, K/2 packed] → [N, K] FP32 row-major */
    int total = N * K;
    int n_packed = (total + 1) / 2;
    int ei = 0;
    for (int i = 0; i < n_packed && ei < total; i++) {
        uint8_t byte = weight_packed[i];
        /* High nibble */
        {
            int block_idx = ei / block_size;
            temp_weight[ei] = VV_NF4_TABLE[(byte >> 4) & 0x0F]
                              * weight_scales[block_idx];
            ei++;
        }
        /* Low nibble */
        if (ei < total) {
            int block_idx = ei / block_size;
            temp_weight[ei] = VV_NF4_TABLE[byte & 0x0F]
                              * weight_scales[block_idx];
            ei++;
        }
    }

    /* Step 2: output[M,N] = input[M,K] @ temp_weight[N,K]^T
     * temp_weight is row-major [N, K], so we need A * B^T.
     * Naive: output[i][j] = sum_k input[i][k] * temp_weight[j][k]
     */
#ifdef VV_HAS_OPENBLAS
    /* A[M,K] * B^T[K,N] => C[M,N]   (B stored as [N,K]) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, input, K, temp_weight, K, 0.0f, output, N);
#else
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            const float* row_in = input + i * K;
            const float* row_w  = temp_weight + j * K;
            for (int p = 0; p < K; p++) {
                sum += row_in[p] * row_w[p];
            }
            output[i * N + j] = sum;
        }
    }
#endif
    return VV_OK;
}

/* ──────────────────────────── RMSNorm ──────────────────────────────────── */

vv_status_t vv_rmsnorm_cpu(const float* input, const float* weight,
                            float* output, int seq_len, int hidden_size,
                            float eps) {
    if (!input || !weight || !output) return VV_ERR_NULL_PTR;

    for (int t = 0; t < seq_len; t++) {
        const float* x = input + t * hidden_size;
        float* o = output + t * hidden_size;

        /* Compute RMS */
        float ss = 0.0f;
        for (int d = 0; d < hidden_size; d++) {
            ss += x[d] * x[d];
        }
        float rms = sqrtf(ss / (float)hidden_size + eps);
        float inv_rms = 1.0f / rms;

        for (int d = 0; d < hidden_size; d++) {
            o[d] = x[d] * inv_rms * weight[d];
        }
    }
    return VV_OK;
}

/* ──────────────────────────── RoPE ─────────────────────────────────────── */

/**
 * x layout: [seq_len, n_heads, head_dim]  (seq-major)
 * Qwen2 convention: half-split pairs (d, d + half_dim)
 */
vv_status_t vv_rope_cpu(float* x, int seq_len, int n_heads,
                         int head_dim, int position_offset, float theta) {
    if (!x) return VV_ERR_NULL_PTR;

    int half_dim = head_dim / 2;

    for (int t = 0; t < seq_len; t++) {
        int pos = position_offset + t;
        for (int h = 0; h < n_heads; h++) {
            /* seq-major: stride between positions = n_heads * head_dim */
            float* vec = x + t * n_heads * head_dim + h * head_dim;

            for (int d = 0; d < half_dim; d++) {
                float freq = 1.0f / powf(theta,
                    (float)(2 * d) / (float)head_dim);
                float angle = (float)pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);

                float x0 = vec[d];
                float x1 = vec[d + half_dim];
                vec[d]            = x0 * cos_a - x1 * sin_a;
                vec[d + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
    return VV_OK;
}

/* ──────────────────────────── Attention ─────────────────────────────────── */

/**
 * Q/K/V/O layout: [seq_len, n_heads, head_dim]  (seq-major)
 * KV cache layout: [cache_len, n_kv_heads, head_dim]  (seq-major)
 */

vv_status_t vv_attention_prefill_cpu(
    const float* q, const float* k, const float* v, float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int seq_len, bool causal)
{
    if (!q || !k || !v || !output) return VV_ERR_NULL_PTR;

    float scale = 1.0f / sqrtf((float)head_dim);
    int gqa_ratio = n_q_heads / n_kv_heads;
    int q_stride = n_q_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    for (int h = 0; h < n_q_heads; h++) {
        int kv_h = h / gqa_ratio;

        for (int i = 0; i < seq_len; i++) {
            int max_j = causal ? (i + 1) : seq_len;

            /* Q[i, h, :] — seq-major */
            const float* qi = q + i * q_stride + h * head_dim;

            /* Compute attention scores and find max */
            float max_score = -FLT_MAX;
            for (int j = 0; j < max_j; j++) {
                /* K[j, kv_h, :] — seq-major */
                const float* kj = k + j * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += qi[d] * kj[d];
                dot *= scale;
                if (dot > max_score) max_score = dot;
            }

            /* Softmax + weighted V */
            float sum_exp = 0.0f;
            /* O[i, h, :] — seq-major */
            float* oi = output + i * q_stride + h * head_dim;
            memset(oi, 0, (size_t)head_dim * sizeof(float));

            for (int j = 0; j < max_j; j++) {
                const float* kj = k + j * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += qi[d] * kj[d];
                dot *= scale;
                float e = expf(dot - max_score);
                sum_exp += e;

                const float* vj = v + j * kv_stride + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    oi[d] += e * vj[d];
            }

            float inv_sum = 1.0f / (sum_exp + 1e-8f);
            for (int d = 0; d < head_dim; d++)
                oi[d] *= inv_sum;
        }
    }
    return VV_OK;
}

vv_status_t vv_attention_decode_cpu(
    const float* q, const float* k_cache, const float* v_cache,
    float* output,
    int n_q_heads, int n_kv_heads, int head_dim, int cache_len)
{
    if (!q || !k_cache || !v_cache || !output) return VV_ERR_NULL_PTR;

    float scale = 1.0f / sqrtf((float)head_dim);
    int gqa_ratio = n_q_heads / n_kv_heads;
    /* Q is [1, n_q_heads, head_dim]; for seq_len=1, same as [n_q_heads, head_dim] */
    /* KV cache: [cache_len, n_kv_heads, head_dim] (seq-major) */
    int kv_stride = n_kv_heads * head_dim;

    for (int h = 0; h < n_q_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float* qi = q + h * head_dim;

        /* Compute scores, find max */
        float max_s = -FLT_MAX;
        for (int t = 0; t < cache_len; t++) {
            /* K_cache[t, kv_h, :] */
            const float* kt = k_cache + t * kv_stride + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += qi[d] * kt[d];
            dot *= scale;
            if (dot > max_s) max_s = dot;
        }

        /* Softmax + weighted V */
        float sum_exp = 0.0f;
        float* oi = output + h * head_dim;
        memset(oi, 0, (size_t)head_dim * sizeof(float));

        for (int t = 0; t < cache_len; t++) {
            const float* kt = k_cache + t * kv_stride + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += qi[d] * kt[d];
            dot *= scale;
            float e = expf(dot - max_s);
            sum_exp += e;

            const float* vt = v_cache + t * kv_stride + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                oi[d] += e * vt[d];
        }

        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int d = 0; d < head_dim; d++)
            oi[d] *= inv_sum;
    }
    return VV_OK;
}

/* ──────────────────────────── SwiGLU ───────────────────────────────────── */

vv_status_t vv_swiglu_cpu(const float* gate, const float* up,
                           float* output, int n_elements) {
    if (!gate || !up || !output) return VV_ERR_NULL_PTR;

    for (int i = 0; i < n_elements; i++) {
        /* SiLU(gate) * up = gate * sigmoid(gate) * up */
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        output[i] = silu * up[i];
    }
    return VV_OK;
}

/* ──────────────────────────── Residual add ──────────────────────────────── */

vv_status_t vv_residual_add_cpu(float* x, const float* y, int n) {
    if (!x || !y) return VV_ERR_NULL_PTR;
    for (int i = 0; i < n; i++) x[i] += y[i];
    return VV_OK;
}

/* ──────────────────────────── Embedding ─────────────────────────────────── */

vv_status_t vv_embedding_cpu(const void* table_fp16, const int32_t* ids,
                              float* output, int seq_len, int hidden_size) {
    if (!table_fp16 || !ids || !output) return VV_ERR_NULL_PTR;

    const uint16_t* table = (const uint16_t*)table_fp16;

    for (int t = 0; t < seq_len; t++) {
        int32_t id = ids[t];
        const uint16_t* row = table + (size_t)id * (size_t)hidden_size;
        float* out = output + (size_t)t * (size_t)hidden_size;

        for (int d = 0; d < hidden_size; d++) {
            /* FP16 → FP32 conversion */
            uint16_t h = row[d];
            uint32_t sign = ((uint32_t)h & 0x8000) << 16;
            uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
            uint32_t frac = (uint32_t)h & 0x03FF;

            if (expo == 0) {
                /* Zero / denormal → flush to signed zero */
                union { float f; uint32_t u; } u;
                u.u = sign;
                out[d] = u.f;
            } else if (expo == 0x1F) {
                /* Inf / NaN */
                union { float f; uint32_t u; } u;
                u.u = sign | 0x7F800000 | (frac << 13);
                out[d] = u.f;
            } else {
                union { float f; uint32_t u; } u;
                u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
                out[d] = u.f;
            }
        }
    }
    return VV_OK;
}

/* ──────────────────────────── Sampling ──────────────────────────────────── */

vv_status_t vv_sample_greedy_cpu(const float* logits, int vocab_size,
                                  int32_t* token_id) {
    if (!logits || !token_id) return VV_ERR_NULL_PTR;

    float best = logits[0];
    int32_t best_id = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best) {
            best = logits[i];
            best_id = i;
        }
    }
    *token_id = best_id;
    return VV_OK;
}
