/**
 * @file q8.h
 * @brief Layout of per-token int8 activations, shared by the CPU and GPU.
 *
 * A dot product does not care in which order it visits k, only that the
 * activations and the weights agree on it. The int8 activations are written
 * in whatever order inside each run of 32 columns lets the kernel reading
 * them take the packed 4-bit weights as they are, with no shuffle in its
 * inner loop. The per-32 sums (`xsum`) are the same in every layout.
 */
#ifndef VV_Q8_H
#define VV_Q8_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Order of the int8 activations inside each run of 32 columns. */
typedef enum vv_q8_layout {
    /** Column order. W8A8 everywhere reads it. */
    VV_Q8_NATURAL = 0,
    /**
     * The 16 even columns, then the 16 odd ones: a row-major INT4G byte run
     * (high nibble = even k) splits into exactly these two halves. The CPU
     * W4A8 kernels read it.
     */
    VV_Q8_NIBBLE  = 1,
    /**
     * Matches the W4A16 GPU layout (vv_int4g_to_gpu_layout): 32-bit word t
     * of a 16-byte chunk holds nibble slots s = 0..7. Positions 8t..8t+3
     * are the even slots, 8t+4..8t+7 the odd ones, which is what
     * `w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F` give. The GPU W4A8 GEMV
     * (M <= 8) reads it.
     */
    VV_Q8_W4_GEMV = 2,
    /**
     * Also the W4A16 GPU layout, arranged for mma.m16n8k32: the low half of
     * word t (slots 0..3) at 4t..4t+3, the high half at 16+4t..16+4t+3, so
     * a lane's two B registers are the two halves of one word and its A
     * fragment is read with ldmatrix as usual. The GPU W4A8 GEMMs read it.
     */
    VV_Q8_W4_MMA  = 3,
    VV_Q8_LAYOUT_COUNT
} vv_q8_layout_t;

/**
 * @brief Where column j (0..31) of a 32-column run is stored.
 *
 * In the W4A16 GPU layout, k = j sits in word t = (j >> 1) & 3 at nibble
 * slot s = (j >> 3) + 4 * (j & 1).
 */
static inline int vv_q8_pos(int layout, int j) {
    const int e = j & 1, t = (j >> 1) & 3, s = (j >> 3) + 4 * e;
    switch (layout) {
    case VV_Q8_NIBBLE:  return e ? 16 + (j >> 1) : (j >> 1);
    case VV_Q8_W4_GEMV: return 8 * t + 4 * (s & 1) + (s >> 1);
    case VV_Q8_W4_MMA:  return 16 * e + 4 * t + (s & 3);
    default:            return j;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* VV_Q8_H */
