/**
 * @file connector.h
 * @brief SpeechConnector MLP API.
 *
 * Architecture (from modeling_vibevoice.py SpeechConnector):
 *   fc1: Linear(vae_dim, hidden_size)    # e.g. 64→3584 or 128→3584
 *   norm: RMSNorm(hidden_size)
 *   fc2: Linear(hidden_size, hidden_size) # 3584→3584
 *   forward: x → fc1 → norm → fc2         (no activation)
 *
 * Both connectors run in FP16 on the device (NOT quantized). The device
 * weights are uploaded once and shared by every request on that device; the
 * output goes straight into the prompt's hidden-state rows.
 */
#ifndef VV_CONNECTOR_H
#define VV_CONNECTOR_H

#include "vibevoice/types.h"
#include "vibevoice/device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vv_connector_cpu;

/** @brief SpeechConnector weights (FP32 on the host, borrowed). */
typedef struct vv_connector {
    vv_tensor_t fc1_weight;     /**< [hidden_size, vae_dim] */
    vv_tensor_t fc1_bias;       /**< [hidden_size] or NULL */
    vv_tensor_t norm_weight;    /**< [hidden_size] RMSNorm */
    vv_tensor_t fc2_weight;     /**< [hidden_size, hidden_size] */
    vv_tensor_t fc2_bias;       /**< [hidden_size] or NULL */
    int         vae_dim;
    int         hidden_size;
    float       rms_eps;
    /** FP16 copies for the packed CPU GEMM; built on first CPU forward. */
    struct vv_connector_cpu* cpu;
} vv_connector_t;

/** @brief The connector's FP16 weights on one device. Immutable. */
typedef struct vv_connector_dev {
    void*       blob;
    size_t      bytes;
    const void* fc1_w;
    const void* fc1_b;
    const void* norm_w;
    const void* fc2_w;
    const void* fc2_b;
    int         vae_dim;
    int         hidden_size;
    float       rms_eps;
} vv_connector_dev_t;

/**
 * @brief Initialize connector from loaded weights.
 */
vv_status_t vv_connector_init(vv_connector_t* conn,
                               int vae_dim, int hidden_size);

/**
 * @brief Forward pass on CPU, through the packed FP16 GEMM.
 *
 * @param conn       Connector with loaded weights
 * @param input      [n_frames, vae_dim]
 * @param n_frames   Number of frames
 * @param output     Output: [n_frames, hidden_size] (caller must vv_free)
 */
vv_status_t vv_connector_forward_cpu(vv_connector_t* conn,
                                      const float* input, int n_frames,
                                      float** output);

/** @brief Device bytes vv_connector_upload takes. */
size_t vv_connector_dev_bytes(const vv_connector_t* conn);

/**
 * @brief Upload the weights as FP16 to the current device. Init-time only:
 * allocates and synchronises.
 */
vv_status_t vv_connector_upload(const vv_connector_t* conn, void* stream,
                                vv_connector_dev_t** out);
void vv_connector_dev_free(vv_connector_dev_t* dev);

/** @brief Scratch vv_connector_forward_dev needs for `n_frames` rows. */
size_t vv_connector_scratch_bytes(const vv_connector_dev_t* dev, int n_frames);

/**
 * @brief fc1 → RMSNorm → fc2 on the device; enqueue only.
 *
 * @param input    [n_frames][vae_dim] FP16, rows `input_ld` apart
 * @param scratch  vv_connector_scratch_bytes() of device memory
 * @param rows     Where the output rows go and how they combine with what
 *                 is there (VV_VAE_ROWS_*): straight into the prompt's
 *                 hidden state, the second connector accumulating onto the
 *                 first in the GEMM epilogue.
 */
vv_status_t vv_connector_forward_dev(const vv_connector_dev_t* dev,
                                     const void* input, int input_ld,
                                     int n_frames, void* scratch,
                                     const vv_vae_rows_t* rows, void* stream);

/**
 * @brief Free connector resources (the CPU copies; the tensors are borrowed).
 */
vv_status_t vv_connector_free(vv_connector_t* conn);

#ifdef __cplusplus
}
#endif

#endif /* VV_CONNECTOR_H */
