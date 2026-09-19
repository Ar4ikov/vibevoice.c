/**
 * @file smooth.h
 * @brief SmoothQuant: activation ranges from a calibration run, folded into
 *        the weights of a dense checkpoint at load.
 *
 * Per-token int8 activations lose the small channels of any row that also
 * carries a large one. SmoothQuant moves that range into the weights: for a
 * projection y = W x whose input comes out of an op that can absorb a
 * per-channel factor, x_j / s_j and W[:, j] * s_j leave y unchanged, and
 *
 *     s_j = max|x_j|^alpha / max_r |W[r, j]|^(1 - alpha)
 *
 * evens out the activation range at the cost of some of the weights'.
 * Four places in a Qwen2 layer can absorb it:
 *
 *   input_layernorm        -> q, k, v        gamma_j / s_j, columns * s_j
 *   post_attention_layernorm -> gate, up     gamma_j / s_j, columns * s_j
 *   v_proj                 -> o_proj         rows of v (and its bias) / s_c,
 *                                            o columns of every query head
 *                                            that reads KV channel c * s_c
 *   up_proj                -> down_proj      rows of up / s_r, columns * s_r
 *
 * The rows of v and up are quantized per row (or per row group), so scaling
 * them changes nothing but their scales. The factors come from
 * vv_smooth_stats_t (a calibration pass over real audio, see inference.h)
 * and are folded in by vv_model_load_ex before a dense checkpoint is
 * quantized; a checkpoint that is already quantized (compressed-tensors,
 * AWQ, GPTQ, NF4) carries whatever smoothing it was made with.
 */
#ifndef VV_SMOOTH_H
#define VV_SMOOTH_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Which of the four foldable places to smooth (a bit mask). */
typedef enum vv_smooth_map {
    VV_SMOOTH_QKV    = 1,  /**< input_layernorm -> q, k, v              */
    VV_SMOOTH_GATEUP = 2,  /**< post_attention_layernorm -> gate, up    */
    VV_SMOOTH_VO     = 4,  /**< v_proj -> o_proj                        */
    VV_SMOOTH_UPDOWN = 8,  /**< up_proj -> down_proj                    */
    VV_SMOOTH_ALL    = 15,
} vv_smooth_map_t;

/** @brief Where each input's statistics sit in a layer's row. */
typedef enum vv_smooth_slot {
    VV_SMOOTH_ATTN_IN = 0,  /**< input_layernorm output, [hidden]       */
    VV_SMOOTH_MLP_IN,       /**< post_attention_layernorm output, [hidden] */
    VV_SMOOTH_ATTN_OUT,     /**< attention output (o_proj input), [q_dim] */
    VV_SMOOTH_MLP_MID,      /**< SwiGLU output (down_proj input), [inter] */
    VV_SMOOTH_SLOTS
} vv_smooth_slot_t;

/**
 * @brief Per-channel activation absmax of every projection input, per layer.
 *
 * Row `l` is absmax + l * width: ATTN_IN, MLP_IN, ATTN_OUT, MLP_MID, one
 * after the other (vv_smooth_offset). All values are >= 0.
 */
typedef struct vv_smooth_stats {
    int    n_layers;
    int    hidden;      /**< hidden_size                                  */
    int    q_dim;       /**< num_attention_heads * head_dim               */
    int    inter;       /**< intermediate_size                            */
    int    width;       /**< 2 * hidden + q_dim + inter                   */
    long   n_tokens;    /**< positions the statistics were taken over     */
    float* absmax;      /**< [n_layers][width]                            */
} vv_smooth_stats_t;

/** @brief Allocate zeroed statistics for a model of these dimensions. */
vv_status_t vv_smooth_stats_alloc(int n_layers, int hidden, int q_dim,
                                  int inter, vv_smooth_stats_t** out);

/** @brief Free statistics (NULL is fine). */
void vv_smooth_stats_free(vv_smooth_stats_t* st);

/** @brief Offset of `slot` inside one layer's row. */
int vv_smooth_offset(const vv_smooth_stats_t* st, vv_smooth_slot_t slot);

/** @brief Length of `slot`. */
int vv_smooth_len(const vv_smooth_stats_t* st, vv_smooth_slot_t slot);

/**
 * @brief Write statistics to a small binary file ("VVSQ", dimensions,
 *        FP32 values, little-endian), so a calibration is paid for once.
 */
vv_status_t vv_smooth_stats_save(const vv_smooth_stats_t* st,
                                 const char* path);

/** @brief Read what vv_smooth_stats_save wrote. */
vv_status_t vv_smooth_stats_load(const char* path, vv_smooth_stats_t** out);

/**
 * @brief The SmoothQuant factor of one channel.
 *
 * amax^alpha / wmax^(1 - alpha), clamped to [1e-4, 1e4]; 1 when either
 * range is zero (a channel the calibration never saw, or a zero column),
 * which folds as a no-op.
 */
float vv_smooth_factor(float amax, float wmax, float alpha);

/** @brief Default migration strength when the caller passes 0. */
#define VV_SMOOTH_ALPHA_DEFAULT 0.5f

/** @brief Default maps when the caller passes 0. */
#define VV_SMOOTH_MAPS_DEFAULT VV_SMOOTH_ALL

#ifdef __cplusplus
}
#endif

#endif /* VV_SMOOTH_H */
