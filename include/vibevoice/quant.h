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

/**
 * @brief Repack AutoAWQ / GPTQ tensors into the runtime's INT4G layout.
 *
 * @param qweight    int32 [K][N/8], AWQ column packing
 * @param qzeros     int32 [K/G][N/8]
 * @param scales     FP16  [K/G][N]
 * @param zero_bias  0 for AWQ; 1 for GPTQ, whose qzeros store zero_point - 1
 * @param out_packed uint8 [N][K/2], high nibble = even k
 * @param out_scales FP16  [N][K/G]
 * @param out_mins   FP16  [N][K/G], equal to -zero * scale
 *
 * Parallelised over output rows when built with OpenMP.
 */
vv_status_t vv_awq_repack(const uint32_t* qweight, const uint32_t* qzeros,
                          const uint16_t* scales, int K, int N, int group_size,
                          int zero_bias,
                          uint8_t* out_packed, uint16_t* out_scales,
                          uint16_t* out_mins);

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
