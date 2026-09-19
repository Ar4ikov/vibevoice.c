/**
 * @file quant.h
 * @brief Weight quantization formats the runtime can load.
 */
#ifndef VV_QUANT_H
#define VV_QUANT_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief How a linear layer's weights are stored. */
typedef enum vv_quant_kind {
    VV_QUANT_NONE  = 0,  /**< Dense FP16/BF16                               */
    VV_QUANT_NF4   = 1,  /**< bitsandbytes NF4, block 64, double-quantized  */
    VV_QUANT_INT4G = 2,  /**< INT4 group-affine (AWQ / GPTQ, repacked)      */
    /**
     * INT8, symmetric, one FP32 scale per output channel. Slots of
     * vv_weight_t:
     *   tensor        int8 [N][K]  (VV_DTYPE_I8)
     *   quant.scales  FP32 [N]     w = q * scale
     *   mins          unused
     * The kernels here run it against FP16/FP32 activations (W8A16). The
     * layout is the one the W8A8 work (#18) uses, so the same weights can
     * later run on int8 activations too.
     */
    VV_QUANT_INT8  = 3,
} vv_quant_kind_t;

/** @brief How an INT4G weight's packed codes and group data are arranged. */
typedef enum vv_int4g_layout {
    /** packed [N][K/2] (high nibble = even k), scales and mins FP16
     *  [N][K/G], zeros uint8 [N][K/G]. What the CPU kernels read. */
    VV_INT4G_ROWMAJOR = 0,
    /** packed [N][K/2] with nibbles permuted per 16-byte chunk, scales
     *  holding half2 {scale, zero} [N][K/G], no mins. What the GPU reads;
     *  see vv_int4g_to_gpu_layout. */
    VV_INT4G_GPU      = 1,
} vv_int4g_layout_t;

/**
 * @brief Repack AutoAWQ tensors into the runtime's INT4G layout.
 *
 * @param qweight    int32 [K][N/8], AWQ column packing
 * @param qzeros     int32 [K/G][N/8]
 * @param scales     FP16  [K/G][N]
 * @param zero_bias  0 for AWQ; 1 for GPTQ, whose qzeros store zero_point - 1
 * @param out_packed uint8 [N][K/2], high nibble = even k
 * @param out_scales FP16  [N][K/G]
 * @param out_mins   FP16  [N][K/G], equal to -zero * scale
 * @param out_zeros  uint8 [N][K/G], the exact integer zero point (with
 *                   zero_bias applied), or NULL
 *
 * Parallelised over output rows when built with OpenMP.
 */
vv_status_t vv_awq_repack(const uint32_t* qweight, const uint32_t* qzeros,
                          const uint16_t* scales, int K, int N, int group_size,
                          int zero_bias,
                          uint8_t* out_packed, uint16_t* out_scales,
                          uint16_t* out_mins, uint8_t* out_zeros);

/**
 * @brief Repack AutoGPTQ / GPTQModel tensors into the same INT4G layout.
 *
 * GPTQ packs along the *input* dimension, in plain order:
 *   qweight int32 [K/8][N]      bits 4j of word (r, n) hold k = 8r + j
 *   qzeros  int32 [K/G][N/8]    bits 4j of word (g, c) hold n = 8c + j
 *   scales  FP16  [K/G][N]
 *   g_idx   int32 [K]           group of each input channel, or NULL
 *
 * The runtime layout needs each group to be a contiguous run of k. Act-order
 * (desc_act) checkpoints scatter each group over the input channels; for
 * those the channels are sorted by group (stable, so plain k / G stays the
 * identity): output column j holds input channel out_perm[j], and the
 * product must run on x[out_perm[j]] (vv_w4a16_gather_dev). A g_idx whose
 * groups are not all exactly group_size channels, or an act-order g_idx
 * with out_perm NULL, is refused with VV_ERR_UNSUPPORTED.
 *
 * @param zero_bias 1 for the classic "gptq" format (stored zero - 1),
 *                  0 for "gptq_v2"
 * @param out_perm  int32 [K], receives the column order (the identity when
 *                  the groups are already contiguous), or NULL
 */
vv_status_t vv_gptq_repack(const uint32_t* qweight, const uint32_t* qzeros,
                           const uint16_t* scales, const int32_t* g_idx,
                           int K, int N, int group_size, int zero_bias,
                           uint8_t* out_packed, uint16_t* out_scales,
                           uint16_t* out_mins, uint8_t* out_zeros,
                           int32_t* out_perm);

/**
 * @brief Convert one row-major INT4G weight to the GPU layout in place.
 *
 * `packed` keeps its size; within each 16-byte chunk (32 weights of one
 * row) word t receives the k-pairs t, t+4, t+8, t+12, nibble slot s holding
 * element (s >> 2) of pair t + 4*(s & 3). `out_sz` receives half2
 * {scale, zero} per [N][K/G]. Needs K % 32 == 0 and exact integer zeros.
 */
vv_status_t vv_int4g_to_gpu_layout(uint8_t* packed, const uint16_t* scales,
                                   const uint8_t* zeros, int N, int K,
                                   int group_size, uint16_t* out_sz);

/**
 * @brief Quantize a dense FP32 weight into the INT4G layout (min/max per
 *        group). Used by tools that convert an existing checkpoint.
 */
vv_status_t vv_int4g_quantize(const float* w, int N, int K, int group_size,
                              uint8_t* out_packed, uint16_t* out_scales,
                              uint16_t* out_mins);

/**
 * @brief What the loader should do with the checkpoint's projections.
 *
 * AUTO keeps whatever the checkpoint has: packed NF4 and AWQ stay as they
 * are, and a dense BF16/F16/F32 checkpoint stays dense (FP16) unless asked.
 * The other values quantize dense projections while they are read from the
 * file, before placement, so every byte count downstream is the final one.
 * A checkpoint that is already quantized is only accepted in its own format.
 *
 * New formats go before VV_LOAD_QUANT_COUNT, with a name in
 * vv_load_quant_name(); the parser picks them up from there.
 */
typedef enum vv_load_quant {
    VV_LOAD_QUANT_AUTO = 0,  /**< keep the checkpoint's format            */
    VV_LOAD_QUANT_NONE,      /**< dense FP16                               */
    VV_LOAD_QUANT_NF4,       /**< NF4, blockwise absmax 64, FP16 scales    */
    VV_LOAD_QUANT_INT4,      /**< INT4G asymmetric, group 128              */
    VV_LOAD_QUANT_INT8,      /**< INT8 symmetric, per output channel       */
    VV_LOAD_QUANT_COUNT
} vv_load_quant_t;

/** @brief "auto" | "none" | "nf4" | "int4" | "int8" → value; COUNT when
 *         unknown. */
vv_load_quant_t vv_load_quant_parse(const char* name);

/** @brief Name for logs and --help; "?" when out of range. */
const char* vv_load_quant_name(vv_load_quant_t q);

/** @brief Block size of the NF4 layout the runtime reads (bitsandbytes). */
#define VV_NF4_BLOCK 64
/** @brief Group size load-time INT4G quantization uses. */
#define VV_INT4G_LOAD_GROUP 128

/**
 * @brief Quantize dense FP32 rows to NF4 in the bitsandbytes layout.
 *
 * Blocks of 64 consecutive elements of the row-major [N,K] matrix share one
 * absmax scale, stored as FP16 (no double quantization — the loader would
 * only undo it again). Each element takes the NF4 code nearest to
 * w / scale, where `scale` is the FP16 value the kernels will multiply by,
 * so what is rounded is exactly what is reconstructed.
 *
 * @param out_packed [N][K/2], high nibble = even k
 * @param out_scales [N*K/64] FP16
 *
 * Rows are independent: the result does not depend on the thread count.
 */
vv_status_t vv_nf4_quantize(const float* w, int N, int K,
                            uint8_t* out_packed, uint16_t* out_scales);

/**
 * @brief Per-output-channel symmetric int8 quantization of dense rows.
 *
 * scale[n] = max|w[n,:]| / 127 and q = round-half-even(w / scale), clamped
 * to [-127, 127]; -128 is never produced. A row of zeros gets scale 0 and
 * zero codes; NaN weights become 0.
 *
 * Rows are independent: the result does not depend on the thread count.
 */
vv_status_t vv_int8_quantize_rows(const float* w, int N, int K,
                                  int8_t* out_q, float* out_scale);

#ifdef __cplusplus
}
#endif

#endif /* VV_QUANT_H */
