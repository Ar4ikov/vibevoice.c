/**
 * @file bitnet_load.h
 * @brief Pieces of the asr-bitnet loader shared by the GGUF and the
 *        safetensors paths (internal).
 */
#ifndef VV_BITNET_LOAD_H
#define VV_BITNET_LOAD_H

#include "vibevoice/model.h"
#include "vibevoice/vae_i8.h"

/**
 * @brief One int8 encoder layer from I8_S codes.
 *
 * @param q      codes as stored: [out][in][kp] (a conv), [out][in] (kp = 1),
 *               or [C][kp] (depthwise, `in` = 1, `out` = C)
 * @param kp     the stored kernel length, which VibeASR.cpp's converter pads
 *               at the front with zeros (7 -> 8, 10 -> 16)
 * @param k      the model's kernel length; the kp - k leading taps must be
 *               zero and are dropped (they add nothing to any sum)
 * @param bias   FP32 [out], copied
 */
vv_status_t vv_i8vae_layer_from_q(vv_i8vae_t* v, const int8_t* q, float scale,
                                  int out, int in, int kp, int k, int stride,
                                  int depthwise, const float* bias,
                                  vv_i8_layer_t* L);

/** @brief A float vector copied into memory the encoder owns. */
float* vv_i8vae_copy_f32(vv_i8vae_t* v, const float* src, size_t n);

/**
 * @brief Allocate the block arrays of a tower from its stage depths.
 */
vv_status_t vv_i8vae_tower_alloc(vv_i8vae_t* v, vv_i8_tower_t* tw,
                                 int n_stages, const int* depth);

/** @brief Kernel length of layer `ds` (0 = stem) in the HF model. */
int vv_bitnet_vae_kernel(const vv_model_config_t* c, bool acoustic, int ds);

/** @brief Stride of downsample `ds` (0 = stem, stride 1). */
int vv_bitnet_vae_stride(const vv_model_config_t* c, bool acoustic, int ds);

/**
 * @brief Row-quantize an F16 or F32 head [V][K] to int8 + FP32 scales
 *        (vv_i8_rowquant_f32), into model->head_i8 / head_i8_scale.
 * @param f16  nonzero if `src` is F16
 */
vv_status_t vv_bitnet_head_i8(vv_model_t* m, const void* src, int f16,
                              int V, int K);

/**
 * @brief The int8 filter in front of the F16 head (head_i8, head_i8_scale,
 *        head_bound), built from model->lm_head. The CPU default: the
 *        argmax stays the F16 head's, the head read per token halves.
 */
vv_status_t vv_bitnet_head_filter_prepare(vv_model_t* m);

#endif /* VV_BITNET_LOAD_H */
