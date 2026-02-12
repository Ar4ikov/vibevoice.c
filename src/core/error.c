/**
 * @file error.c
 * @brief Status code to human-readable string mapping.
 */

#include "vibevoice/vibevoice.h"

const char* vv_status_str(vv_status_t status) {
    switch (status) {
        case VV_OK:               return "OK";
        case VV_ERR_INVALID_ARG:  return "Invalid argument";
        case VV_ERR_NULL_PTR:     return "Null pointer";
        case VV_ERR_OUT_OF_MEMORY:return "Out of memory";
        case VV_ERR_NOT_FOUND:    return "Not found";
        case VV_ERR_IO:           return "I/O error";
        case VV_ERR_PARSE:        return "Parse error";
        case VV_ERR_UNSUPPORTED:  return "Unsupported";
        case VV_ERR_OVERFLOW:     return "Overflow";
        case VV_ERR_SHAPE_MISMATCH: return "Shape mismatch";

        case VV_ERR_CUDA:         return "CUDA error";
        case VV_ERR_CUDA_OOM:     return "CUDA out of memory";
        case VV_ERR_CUDA_LAUNCH:  return "CUDA kernel launch failed";

        case VV_ERR_TRT:          return "TensorRT error";
        case VV_ERR_TRT_BUILD:    return "TensorRT build failed";
        case VV_ERR_TRT_RUNTIME:  return "TensorRT runtime error";

        case VV_ERR_MODEL_FORMAT: return "Invalid model format";
        case VV_ERR_MODEL_VERSION:return "Unsupported model version";
        case VV_ERR_WEIGHT_MISSING: return "Missing weight tensor";

        case VV_ERR_AUDIO_FORMAT: return "Unsupported audio format";
        case VV_ERR_AUDIO_RESAMPLE: return "Audio resampling failed";

        default:                  return "Unknown error";
    }
}

vv_status_t vv_tensor_free(vv_tensor_t* t) {
    if (!t) return VV_ERR_NULL_PTR;
    if (t->data) {
        if (t->on_gpu) {
            /* GPU memory freed via CUDA, handled elsewhere. */
            t->data = NULL;
        } else {
            vv_free(t->data);
            t->data = NULL;
        }
    }
    t->size_bytes = 0;
    return VV_OK;
}
