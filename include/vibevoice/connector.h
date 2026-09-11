/**
 * @file connector.h
 * @brief SpeechConnector MLP API.
 *
 * Architecture (from modeling_vibevoice.py SpeechConnector):
 *   fc1: Linear(vae_dim, hidden_size)    # e.g. 64→3584 or 128→3584
 *   norm: RMSNorm(hidden_size)
 *   fc2: Linear(hidden_size, hidden_size) # 3584→3584
 *   forward: x → fc1 → GELU → norm → fc2
 *
 * Both connectors operate in FP16 (NOT quantized).
 */
#ifndef VV_CONNECTOR_H
#define VV_CONNECTOR_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief SpeechConnector weights. */
typedef struct vv_connector {
    vv_tensor_t fc1_weight;     /**< [hidden_size, vae_dim] */
    vv_tensor_t fc1_bias;       /**< [hidden_size] or NULL */
    vv_tensor_t norm_weight;    /**< [hidden_size] RMSNorm */
    vv_tensor_t fc2_weight;     /**< [hidden_size, hidden_size] */
    vv_tensor_t fc2_bias;       /**< [hidden_size] or NULL */
    int         vae_dim;
    int         hidden_size;
    float       rms_eps;
} vv_connector_t;

/**
 * @brief Initialize connector from loaded weights.
 */
vv_status_t vv_connector_init(vv_connector_t* conn,
                               int vae_dim, int hidden_size);

/**
 * @brief Forward pass on CPU.
 *
 * @param conn       Connector with loaded weights
 * @param input      [n_frames, vae_dim]
 * @param n_frames   Number of frames
 * @param output     Output: [n_frames, hidden_size] (caller must vv_free)
 */
vv_status_t vv_connector_forward_cpu(const vv_connector_t* conn,
                                      const float* input, int n_frames,
                                      float** output);

/**
 * @brief Forward pass on GPU (FP16).
 *
 * @param conn       Connector with weights on GPU
 * @param input_gpu  [n_frames, vae_dim] FP16 on GPU
 * @param n_frames   Number of frames
 * @param output_gpu Output: [n_frames, hidden_size] FP16 on GPU
 * @param stream     CUDA stream
 */
vv_status_t vv_connector_forward_dev(const vv_connector_t* conn,
                                       const void* input_gpu, int n_frames,
                                       void* output_gpu, void* stream);

/**
 * @brief Forward pass with GPU auto-fallback.
 *
 * Tries GPU first, falls back to CPU if GPU unavailable.
 * Same interface as CPU version.
 */
vv_status_t vv_connector_forward_auto(const vv_connector_t* conn,
                                        const float* input, int n_frames,
                                        float** output);

/**
 * @brief Free connector resources.
 */
vv_status_t vv_connector_free(vv_connector_t* conn);

#ifdef __cplusplus
}
#endif

#endif /* VV_CONNECTOR_H */
