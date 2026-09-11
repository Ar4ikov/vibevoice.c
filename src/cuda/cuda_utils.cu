/**
 * @file cuda_utils.cu
 * @brief CUDA utility functions: memory, streams, device management.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Allocate GPU memory.
 */
vv_status_t vv_dev_alloc(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaErrorMemoryAllocation) return VV_ERR_CUDA_OOM;
    if (err != cudaSuccess) return VV_ERR_CUDA;
    return VV_OK;
}

/**
 * @brief Allocate pinned (page-locked) host memory for fast transfers.
 */
vv_status_t vv_dev_alloc_pinned(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) return VV_ERR_CUDA;
    return VV_OK;
}

/**
 * @brief Free GPU memory.
 */
vv_status_t vv_dev_free(void* ptr) {
    if (!ptr) return VV_OK;
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Free pinned host memory.
 */
vv_status_t vv_dev_free_pinned(void* ptr) {
    if (!ptr) return VV_OK;
    cudaError_t err = cudaFreeHost(ptr);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Copy host → device.
 */
vv_status_t vv_dev_memcpy_h2d(void* dst, const void* src, size_t size,
                                void* stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                              (cudaStream_t)stream);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Copy device → host.
 */
vv_status_t vv_dev_memcpy_d2h(void* dst, const void* src, size_t size,
                                void* stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                              (cudaStream_t)stream);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Copy device → device.
 */
vv_status_t vv_dev_memcpy_d2d(void* dst, const void* src, size_t size,
                                void* stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                              (cudaStream_t)stream);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Create a CUDA stream.
 */
vv_status_t vv_dev_stream_create(void** stream) {
    if (!stream) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Destroy a CUDA stream.
 */
vv_status_t vv_dev_stream_destroy(void* stream) {
    if (!stream) return VV_OK;
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Synchronize a CUDA stream.
 */
vv_status_t vv_dev_stream_sync(void* stream) {
    cudaError_t err;
    if (stream) {
        err = cudaStreamSynchronize((cudaStream_t)stream);
    } else {
        err = cudaDeviceSynchronize();
    }
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Get device properties.
 */
vv_status_t vv_dev_get_device_info(int device_id, size_t* total_mem,
                                     size_t* free_mem, int* sm_count) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return VV_ERR_CUDA;

    if (total_mem) *total_mem = prop.totalGlobalMem;
    if (sm_count) *sm_count = prop.multiProcessorCount;

    if (free_mem) {
        cudaSetDevice(device_id);
        size_t f, t;
        cudaMemGetInfo(&f, &t);
        *free_mem = f;
    }
    return VV_OK;
}

/**
 * @brief Set current CUDA device.
 */
vv_status_t vv_dev_set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}


/**
 * @brief Fill device memory with a byte value (synchronous on the null stream).
 */
vv_status_t vv_dev_memset(void* ptr, int value, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    return (cudaMemset(ptr, value, size) == cudaSuccess)
           ? VV_OK : VV_ERR_CUDA;
}

} /* extern "C" */

/* ─── Events and pinned host memory (weight streaming) ──────────────────── */

vv_status_t vv_dev_event_create(void** ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    cudaEvent_t e;
    /* No timing data: the event exists only to order two streams. */
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess)
        return VV_ERR_CUDA;
    *ev = (void*)e;
    return VV_OK;
}

vv_status_t vv_dev_event_destroy(void* ev) {
    if (!ev) return VV_OK;
    return cudaEventDestroy((cudaEvent_t)ev) == cudaSuccess
           ? VV_OK : VV_ERR_CUDA;
}

vv_status_t vv_dev_event_record(void* ev, void* stream) {
    if (!ev) return VV_ERR_NULL_PTR;
    return cudaEventRecord((cudaEvent_t)ev, (cudaStream_t)stream) == cudaSuccess
           ? VV_OK : VV_ERR_CUDA;
}

vv_status_t vv_dev_stream_wait_event(void* stream, void* ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    return cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)ev, 0)
           == cudaSuccess ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Page-lock an existing host buffer so H2D can DMA straight out of it.
 *
 * Weight streaming reads the same pages every token; from pageable memory the
 * driver has to bounce each transfer through its own staging buffer, which
 * costs roughly half the achievable PCIe bandwidth.
 */
vv_status_t vv_dev_host_register(void* p, size_t n) {
    if (!p || n == 0) return VV_ERR_NULL_PTR;
    const cudaError_t e = cudaHostRegister(p, n, cudaHostRegisterDefault);
    if (e == cudaErrorHostMemoryAlreadyRegistered) { cudaGetLastError(); return VV_OK; }
    if (e != cudaSuccess) { cudaGetLastError(); return VV_ERR_CUDA; }
    return VV_OK;
}

vv_status_t vv_dev_host_unregister(void* p) {
    if (!p) return VV_OK;
    if (cudaHostUnregister(p) != cudaSuccess) { cudaGetLastError(); return VV_ERR_CUDA; }
    return VV_OK;
}
