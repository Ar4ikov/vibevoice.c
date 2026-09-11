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

#ifdef __cplusplus
}
#endif

#endif /* VV_QUANT_H */
