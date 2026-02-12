/**
 * @file trt.h
 * @brief TensorRT engine API for Conv-VAE tokenizer encoders.
 *
 * Strategy: TRT for Conv-VAE encoders (FP16, standard conv ops).
 * LLM decoder stays as custom CUDA (NF4 not native in TRT).
 */
#ifndef VV_TRT_H
#define VV_TRT_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VV_HAS_TRT

/** @brief Opaque TRT engine handle. */
typedef struct vv_trt_engine vv_trt_engine_t;

/**
 * @brief Load a serialized TRT engine (.plan file).
 */
vv_status_t vv_trt_engine_load(const char* plan_path,
                                vv_trt_engine_t** out);

/**
 * @brief Run inference on a TRT engine.
 *
 * @param engine   Loaded engine
 * @param inputs   Array of device pointers for input bindings
 * @param outputs  Array of device pointers for output bindings
 * @param stream   CUDA stream
 */
vv_status_t vv_trt_engine_infer(vv_trt_engine_t* engine,
                                 const void** inputs, void** outputs,
                                 void* stream);

/**
 * @brief Get binding dimensions for an I/O binding.
 *
 * @param engine      Loaded engine
 * @param binding_idx Binding index
 * @param dims        Output: dimensions array
 * @param n_dims      Output: number of dimensions
 */
vv_status_t vv_trt_engine_get_binding_dims(
    const vv_trt_engine_t* engine, int binding_idx,
    int64_t* dims, int* n_dims);

/**
 * @brief Set dynamic input shape.
 */
vv_status_t vv_trt_engine_set_input_shape(
    vv_trt_engine_t* engine, int binding_idx,
    const int64_t* dims, int n_dims);

/**
 * @brief Free TRT engine.
 */
vv_status_t vv_trt_engine_free(vv_trt_engine_t* engine);

#endif /* VV_HAS_TRT */

#ifdef __cplusplus
}
#endif

#endif /* VV_TRT_H */
