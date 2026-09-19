/**
 * @file kv_quant.h
 * @brief KV-cache storage formats: FP16, emulated FP8, and TurboQuant.
 *
 * The KV cache is the only allocation that grows with the audio length, so on
 * a 12 GB card it is what decides how long a file fits. At 32k positions the
 * FP16 cache is 1.9 GB; the quantized formats take that down to 0.94 GB (FP8)
 * or 0.18 GB (1.5-bit).
 *
 * FP8 is emulated in software — encode/decode are a handful of shifts — so it
 * works on Ampere and Apple Silicon, neither of which has FP8 hardware. The
 * point of the format here is halving memory, not using an FP8 MAC.
 *
 * TurboQuant (Zandieh et al., 2025) is the sub-byte path. Each 128-dim vector
 * is first passed through a randomized Hadamard transform, which spreads the
 * outlier channels that make raw K hard to quantize and leaves coordinates
 * that are close to Gaussian; those are then quantized with the MSE-optimal
 * Lloyd-Max levels for a Gaussian, scaled by the vector's own RMS.
 *
 * The transform is orthogonal, so attention never has to invert it per key:
 *   dot(Hq, Hk) == dot(q, k)  and  sum_t p_t * (H v_t) == H * sum_t p_t * v_t
 * The pipeline therefore rotates Q once before attention and un-rotates the
 * output once after, and the inner loops work directly on stored values.
 */
#ifndef VV_KV_QUANT_H
#define VV_KV_QUANT_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vv_kv_format {
    VV_KV_FP16     = 0,  /**< 16 bits/value, exact baseline            */
    VV_KV_FP8_E4M3 = 1,  /**< 8 bits, 4-bit exponent — the safe FP8    */
    VV_KV_FP8_E5M2 = 2,  /**< 8 bits, 5-bit exponent — wider range     */
    VV_KV_TQ4      = 3,  /**< TurboQuant 4-bit  (+ RMS per vector)     */
    VV_KV_TQ3      = 4,  /**< TurboQuant 3-bit                         */
    VV_KV_TQ2      = 5,  /**< TurboQuant 2-bit                         */
    VV_KV_TQ1_5    = 6,  /**< TurboQuant 1.5-bit, 8-entry 2-D codebook */
    VV_KV_FP32     = 7,  /**< 32 bits/value; the CPU path's activations  */
    VV_KV_FORMAT_COUNT
} vv_kv_format_t;

/**
 * @brief Positions per page of a paged KV cache.
 *
 * Equal to the KV tile of the tensor-core attention kernels, so a tile never
 * straddles two pages and each one costs a single table lookup.
 */
#define VV_KV_PAGE_SIZE 64

/** @brief Parse a format name ("fp16", "fp8", "fp8-e5m2", "tq4"...). */
vv_kv_format_t vv_kv_format_parse(const char* name);

/** @brief Canonical name of a format, for logs and CLI output. */
const char* vv_kv_format_name(vv_kv_format_t fmt);

/** @brief Bits stored per value, x100 (3200, 1600, 800, 400, ... 150). */
int vv_kv_format_bits_x100(vv_kv_format_t fmt);

/** @brief Bytes one head's vector occupies. Always a multiple of 4. */
static inline int vv_kv_bytes_per_vec(vv_kv_format_t fmt, int head_dim) {
    switch (fmt) {
        case VV_KV_FP32:     return head_dim * 4;
        case VV_KV_FP16:     return head_dim * 2;
        case VV_KV_FP8_E4M3:
        case VV_KV_FP8_E5M2: return head_dim;
        case VV_KV_TQ4:      return head_dim / 2;
        case VV_KV_TQ3:      return head_dim * 3 / 8;
        case VV_KV_TQ2:      return head_dim / 4;
        case VV_KV_TQ1_5:    return head_dim * 3 / 16;
        default:             return head_dim * 2;
    }
}

/** @brief True if values are stored verbatim, with no quantization. */
static inline bool vv_kv_is_raw(vv_kv_format_t fmt) {
    return fmt == VV_KV_FP16 || fmt == VV_KV_FP32;
}

/** @brief True if the format stores a per-vector FP16 scale alongside. */
static inline bool vv_kv_has_meta(vv_kv_format_t fmt) {
    return fmt >= VV_KV_TQ4 && fmt <= VV_KV_TQ1_5;
}

/** @brief True if the format needs Q rotated and O un-rotated (TurboQuant). */
static inline bool vv_kv_rotates(vv_kv_format_t fmt) {
    return fmt >= VV_KV_TQ4 && fmt <= VV_KV_TQ1_5;
}

/* ─── Hadamard rotation, applied outside the attention kernels ───────────── */

/** @brief In-place randomized Hadamard transform of [n_heads, head_dim]. */
vv_status_t vv_kv_rotate_dev(void* x_fp16, int n_heads, int head_dim,
                             int rows, void* stream);

/** @brief In-place inverse transform, undoing vv_kv_rotate_dev. */
vv_status_t vv_kv_unrotate_dev(void* x_fp16, int n_heads, int head_dim,
                               int rows, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* VV_KV_QUANT_H */
