/**
 * @file bitnet.h
 * @brief Integer kernels for VibeVoice-ASR-BitNet: ternary (W1.58A8) and
 *        int8 (W8A8) linear layers, on the CPU.
 *
 * The numerics follow microsoft/VibeASR.cpp, the reference runtime for this
 * checkpoint; docs/BITNET.md has the derivation and the source lines.
 *
 * Ternary weights ("I2_S" layout, identical to the GGUF bytes)
 * ------------------------------------------------------------
 * A weight [N][K] with K % 128 == 0 is stored row after row, K/4 bytes per
 * row. Each group of 128 consecutive k is one 32-byte block; byte j of a
 * block holds four 2-bit codes:
 *
 *     bits 7:6 -> k = j        bits 5:4 -> k = j + 32
 *     bits 3:2 -> k = j + 64   bits 1:0 -> k = j + 96
 *
 * and code c means the value (c - 1) * scale, c in {0, 1, 2}. There is one
 * FP32 scale per tensor. This layout lets one 32-byte load and four
 * shift-and-mask steps produce four vectors of 32 codes that line up with
 * four contiguous 32-byte runs of the activation row.
 *
 * Activations (per-token int8)
 * ----------------------------
 * Each input row is quantized on its own:
 *
 *     amax = max(1e-5, max_k |x_k|)       (in double)
 *     s    = (float)(127 / amax)
 *     q_k  = clamp(round_half_even(x_k * s), -128, 127)
 *
 * and the linear output is
 *
 *     y[m][n] = (float)(sum_k (c_nk - 1) * q_mk) / s_m * w_scale + bias[n]
 *
 * evaluated in exactly that order, which is what makes it bit-identical to
 * the reference: the integer sum is exact, and the two FP32 operations are
 * the same two ggml performs.
 *
 * Int8 weights (VAE, "I8_S")
 * --------------------------
 * Row-major int8 [N][K] with one FP32 scale per tensor (w = q * scale).
 */
#ifndef VV_BITNET_H
#define VV_BITNET_H

#include "vibevoice/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Elements per I2_S block. */
#define VV_TERNARY_BLOCK 128

/** @brief Feature set the integer kernels picked ("AVX512-VNNI", "AVX-VNNI",
 *         "AVX2", "NEON-dotprod", "NEON", "scalar"). */
const char* vv_bitnet_cpu_isa(void);

/**
 * @brief Force a kernel family for testing: 0 = auto, 1 = scalar, 2 = AVX2,
 *        3 = AVX-VNNI, 4 = AVX512-VNNI. A family the CPU lacks is refused
 *        (VV_ERR_UNSUPPORTED). Not thread-safe; call before any kernel.
 *        The environment variable VV_BITNET_ISA (same numbers) does the same.
 */
vv_status_t vv_bitnet_cpu_force_isa(int isa);

/* ─── Weight preparation ─────────────────────────────────────────────────── */

/**
 * @brief Ternarize an F32 latent weight exactly as VibeASR.cpp's converter
 *        and `quantize_i2_s` do together, and pack it.
 *
 *     s      = 1 / max(mean|W|, 1e-5)           (per tensor, FP32)
 *     W'     = round(W * s).clamp(-1, 1) / s    (torch semantics, FP32)
 *     scale  = max |W'|                         (the I2_S scale)
 *     code   = |W'| < 1e-6 ? 1 : (W' > 0 ? 2 : 0)
 *
 * mean|W| is accumulated in double; torch accumulates FP32 pairwise, which
 * can move the last bit of s (see docs/BITNET.md for the measured effect).
 *
 * @param codes  out, N*K/4 bytes, K % 128 == 0
 * @param scale  out, the per-tensor scale
 */
vv_status_t vv_ternarize_f32(const float* w, int64_t N, int64_t K,
                             uint8_t* codes, float* scale);

/**
 * @brief Pack ternary values t in {-1, 0, 1} ([N][K], K % 128 == 0) into
 *        the I2_S layout.
 */
vv_status_t vv_ternary_pack(const int8_t* t, int64_t N, int64_t K,
                            uint8_t* codes);

/** @brief Unpack one row of K codes into t in {-1, 0, 1}. */
void vv_ternary_unpack_row(const uint8_t* codes, int64_t K, int8_t* t);

/**
 * @brief Quantize an FP32 weight [N][K] to int8 with one scale per tensor,
 *        as `quantize_i8_s` does: q = clamp(roundf(w * 127/amax), -127, 127),
 *        scale = amax / 127.
 */
vv_status_t vv_i8s_quantize_f32(const float* w, int64_t N, int64_t K,
                                int8_t* q, float* scale);

/**
 * @brief Quantize an FP32 weight [N][K] to int8 with one scale per row
 *        (used for the tied LM head): scale_n = amax_n / 127,
 *        q = clamp(round_half_even(w / scale_n), -127, 127).
 */
vv_status_t vv_i8_rowquant_f32(const float* w, int64_t N, int64_t K,
                               int8_t* q, float* scales);

/* ─── Activations ────────────────────────────────────────────────────────── */

/**
 * @brief Per-token int8 quantization (see the file comment).
 *
 * @param x        [M][K] FP32
 * @param q        out [M][K] int8
 * @param scale    out [M], the multiplier s (so x ~= q / s)
 * @param sum      out [M], sum of q over the row; may be NULL
 */
vv_status_t vv_act_quant_i8_cpu(const float* x, int M, int K,
                                int8_t* q, float* scale, int32_t* sum);

/* ─── Ternary x int8 ─────────────────────────────────────────────────────── */

/**
 * @brief Exact integer product: acc[m][n] = sum_k (c_nk - 1) * q_mk.
 *
 * The primitive every ternary path reduces to, and what the tests compare
 * across ISAs bit for bit. K % 128 == 0.
 */
vv_status_t vv_ternary_gemm_i32_cpu(const int8_t* q, const uint8_t* codes,
                                    int32_t* acc, int M, int N, int K);

/**
 * @brief Ternary linear layer on pre-quantized activations.
 *
 * y[m][n] = (float)acc[m][n] / scale[m] * w_scale (+ bias[n]).
 * The activation is taken already quantized so one quantization can feed
 * q, k and v (or gate and up).
 *
 * @param bias FP32 [N] or NULL
 */
vv_status_t vv_ternary_linear_cpu(const int8_t* q, const float* scale,
                                  const uint8_t* codes, float w_scale,
                                  const float* bias, float* y,
                                  int M, int N, int K);

/* ─── Int8 x int8 ────────────────────────────────────────────────────────── */

/**
 * @brief Exact integer product: acc[m][n] = sum_k w_nk * a_mk, int8 both.
 *
 * Weights must stay in [-127, 127] (every quantizer here clamps them
 * there); activations may use the full [-128, 127]. That is what keeps the
 * AVX2 sign trick free of int16 saturation.
 */
vv_status_t vv_i8_gemm_i32_cpu(const int8_t* a, const int8_t* w,
                               int32_t* acc, int M, int N, int K);

/**
 * @brief The I8_S linear step of the reference VAE, before requantization:
 *        y[m][n] = (float)acc[m][n] * d + bias[n], d = w_scale / a_scale,
 *        and the running max |y| (what the next requantization divides by).
 *
 * @param a_scale  the activation's multiplier (127 / amax), as I8_S stores it
 * @param w_scale  the weight's scale (amax / 127), as I8_S stores it
 * @param bias     FP32 [N] or NULL
 * @param absmax   out, max |y| over the whole output; may be NULL
 */
vv_status_t vv_i8_linear_cpu(const int8_t* a, float a_scale,
                             const int8_t* w, float w_scale,
                             const float* bias, float* y, float* absmax,
                             int M, int N, int K);

/**
 * @brief Requantize a whole tensor with one scale, as the reference VAE does
 *        after every op: inv = 127 / absmax, q = rne(clamp(y * inv, lo, 127))
 *        with lo = 0 when @p relu, else -127.
 * @param out_scale out, inv (the I8_S activation multiplier)
 */
vv_status_t vv_i8s_requant_cpu(const float* y, int64_t n, float absmax,
                               int relu, int8_t* q, float* out_scale);

/* ─── Quantized LM head ──────────────────────────────────────────────────── */

/**
 * @brief Tied-head argmax with int8 row-quantized weights and a per-token
 *        int8 activation: logit_n = (float)(sum_k w_nk q_k) * w_scale[n] / s.
 *
 * Reads V*K bytes instead of the FP16 table's 2*V*K, which at 1.5B is the
 * single largest read of a decode step. Ties go to the lowest index.
 *
 * @param q        [K] int8 activation, @p scale its multiplier
 * @param w        [V][K] int8
 * @param w_scale  [V] FP32
 */
vv_status_t vv_i8_head_argmax_cpu(const int8_t* q, float scale,
                                  const int8_t* w, const float* w_scale,
                                  int V, int K, int32_t* token,
                                  float* value);

#ifdef __cplusplus
}
#endif

#endif /* VV_BITNET_H */
