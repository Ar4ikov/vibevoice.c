/**
 * @file engine_runtime.c
 * @brief TensorRT engine loading and execution.
 *
 * Loads serialized .plan files and runs inference.
 * Used for Conv-VAE tokenizer encoder acceleration.
 */

#include "vibevoice/trt.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VV_HAS_TRT

struct vv_trt_engine {
    void*   runtime;           /* nvinfer1::IRuntime* */
    void*   engine;            /* nvinfer1::ICudaEngine* */
    void*   context;           /* nvinfer1::IExecutionContext* */
    int     n_bindings;
    char    plan_path[512];
};

vv_status_t vv_trt_engine_load(const char* plan_path,
                                vv_trt_engine_t** out) {
    if (!plan_path || !out) return VV_ERR_NULL_PTR;

    /* Read .plan file */
    FILE* f = fopen(plan_path, "rb");
    if (!f) {
        VV_LOG_E("trt: cannot open '%s'", plan_path);
        return VV_ERR_IO;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return VV_ERR_IO;
    }

    void* plan_data = vv_alloc((size_t)size);
    if (!plan_data) {
        fclose(f);
        return VV_ERR_OUT_OF_MEMORY;
    }
    fread(plan_data, 1, (size_t)size, f);
    fclose(f);

    vv_trt_engine_t* eng = (vv_trt_engine_t*)vv_alloc(
        sizeof(vv_trt_engine_t));
    if (!eng) {
        vv_free(plan_data);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(eng, 0, sizeof(*eng));
    strncpy(eng->plan_path, plan_path, sizeof(eng->plan_path) - 1);

    /*
     * TODO: Deserialize TRT engine from plan_data.
     *
     * IRuntime* runtime = createInferRuntime(logger);
     * ICudaEngine* engine = runtime->deserializeCudaEngine(plan_data, size);
     * IExecutionContext* context = engine->createExecutionContext();
     *
     * eng->runtime = runtime;
     * eng->engine = engine;
     * eng->context = context;
     * eng->n_bindings = engine->getNbBindings();
     */

    vv_free(plan_data);

    VV_LOG_I("trt: loaded engine from '%s' (size=%ld bytes)", plan_path, size);
    VV_LOG_W("trt: engine deserialization not yet implemented");

    *out = eng;
    return VV_OK;
}

vv_status_t vv_trt_engine_infer(vv_trt_engine_t* engine,
                                 const void** inputs, void** outputs,
                                 void* stream) {
    if (!engine || !inputs || !outputs) return VV_ERR_NULL_PTR;

    /*
     * TODO: Execute TRT inference.
     *
     * context->setBindingDimensions(0, input_dims);
     * void* bindings[] = { input_ptr, output_ptr };
     * context->enqueueV2(bindings, stream, nullptr);
     */

    VV_LOG_W("trt: inference execution not yet implemented");
    return VV_ERR_UNSUPPORTED;
}

vv_status_t vv_trt_engine_get_binding_dims(
    const vv_trt_engine_t* engine, int binding_idx,
    int64_t* dims, int* n_dims) {
    if (!engine || !dims || !n_dims) return VV_ERR_NULL_PTR;
    /* TODO: implement */
    *n_dims = 0;
    return VV_ERR_UNSUPPORTED;
}

vv_status_t vv_trt_engine_set_input_shape(
    vv_trt_engine_t* engine, int binding_idx,
    const int64_t* dims, int n_dims) {
    if (!engine || !dims) return VV_ERR_NULL_PTR;
    /* TODO: implement */
    return VV_ERR_UNSUPPORTED;
}

vv_status_t vv_trt_engine_free(vv_trt_engine_t* engine) {
    if (!engine) return VV_ERR_NULL_PTR;

    /*
     * TODO: Destroy TRT objects.
     * context->destroy();
     * engine->destroy();
     * runtime->destroy();
     */

    vv_free(engine);
    return VV_OK;
}

#endif /* VV_HAS_TRT */
