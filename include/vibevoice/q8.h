/**
 * @file q8.h
 * @brief Layout of per-token int8 activations, shared by the CPU and GPU.
 */
#ifndef VV_Q8_H
#define VV_Q8_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Order of the int8 activations inside each run of 32 columns. */
typedef enum vv_q8_layout {
    VV_Q8_NATURAL = 0,   /**< column order; W8A8 and the W4A8 GEMM read it   */
    /**
     * The 16 even columns, then the 16 odd ones. A packed 4-bit weight splits
     * into exactly these two runs (high nibbles = even k, low = odd k), so a
     * W4A8 dot product needs no byte shuffle. Sums are unaffected. The CPU
     * W4A8 kernels and the GPU W4A8 GEMV (M <= 8) read it.
     */
    VV_Q8_NIBBLE  = 1,
} vv_q8_layout_t;

#ifdef __cplusplus
}
#endif

#endif /* VV_Q8_H */
