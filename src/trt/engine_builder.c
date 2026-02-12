/**
 * @file engine_builder.c
 * @brief TensorRT engine builder for Conv-VAE encoders.
 *
 * Builds TRT engines from ONNX models exported by tools/export_onnx.py.
 * Uses TRT C API (via NvInfer.h with extern "C" wrappers).
 *
 * NOTE: Actual TRT API calls require C++ linkage. This file provides
 * the C interface wrapper. The implementation details depend on TRT version.
 */

#include "vibevoice/trt.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

#ifdef VV_HAS_TRT

/**
 * @brief Build a TRT engine from an ONNX model.
 *
 * This is typically done offline via tools/build_trt_engine.py.
 * This C function provides a runtime build capability as fallback.
 *
 * @param onnx_path    Path to ONNX model
 * @param plan_path    Output path for serialized .plan
 * @param fp16         Enable FP16 mode
 * @param max_batch    Maximum batch size
 * @param max_seq_len  Maximum input sequence length
 */
vv_status_t vv_trt_build_engine(const char* onnx_path,
                                 const char* plan_path,
                                 bool fp16,
                                 int max_batch,
                                 int max_seq_len) {
    if (!onnx_path || !plan_path) return VV_ERR_NULL_PTR;

    VV_LOG_I("trt: building engine from '%s' -> '%s'", onnx_path, plan_path);
    VV_LOG_I("trt: FP16=%d, max_batch=%d, max_seq=%d",
             fp16, max_batch, max_seq_len);

    /*
     * TODO: Implement TRT engine building via C++ TRT API.
     *
     * Steps:
     * 1. Create IBuilder, INetworkDefinition, IBuilderConfig
     * 2. Create OnnxParser, parse ONNX file
     * 3. Set optimization profile for dynamic shapes
     * 4. Enable FP16 if supported
     * 5. Build serialized network
     * 6. Write .plan to disk
     *
     * This is best done via Python (tools/build_trt_engine.py) for
     * flexibility. The C path is a fallback.
     */

    VV_LOG_W("trt: runtime engine building not yet implemented. "
             "Use tools/build_trt_engine.py instead.");
    return VV_ERR_UNSUPPORTED;
}

#endif /* VV_HAS_TRT */
