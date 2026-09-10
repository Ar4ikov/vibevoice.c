/**
 * @file cuda_utils.cu
 * @brief CUDA utility functions: memory, streams, device management.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief Allocate GPU memory.
 */
vv_status_t vv_cuda_alloc(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaErrorMemoryAllocation) return VV_ERR_CUDA_OOM;
    if (err != cudaSuccess) return VV_ERR_CUDA;
    return VV_OK;
}

/**
 * @brief Allocate pinned (page-locked) host memory for fast transfers.
 */
vv_status_t vv_cuda_alloc_pinned(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) return VV_ERR_CUDA;
    return VV_OK;
}

/**
 * @brief Free GPU memory.
 */
vv_status_t vv_cuda_free(void* ptr) {
    if (!ptr) return VV_OK;
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Free pinned host memory.
 */
vv_status_t vv_cuda_free_pinned(void* ptr) {
    if (!ptr) return VV_OK;
    cudaError_t err = cudaFreeHost(ptr);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Copy host → device.
 */
vv_status_t vv_cuda_memcpy_h2d(void* dst, const void* src, size_t size,
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
vv_status_t vv_cuda_memcpy_d2h(void* dst, const void* src, size_t size,
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
vv_status_t vv_cuda_memcpy_d2d(void* dst, const void* src, size_t size,
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
vv_status_t vv_cuda_stream_create(void** stream) {
    if (!stream) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Destroy a CUDA stream.
 */
vv_status_t vv_cuda_stream_destroy(void* stream) {
    if (!stream) return VV_OK;
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Synchronize a CUDA stream.
 */
vv_status_t vv_cuda_stream_sync(void* stream) {
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
vv_status_t vv_cuda_get_device_info(int device_id, size_t* total_mem,
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
vv_status_t vv_cuda_set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}


/**
 * @brief Fill device memory with a byte value (synchronous on the null stream).
 */
vv_status_t vv_cuda_memset(void* ptr, int value, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    return (cudaMemset(ptr, value, size) == cudaSuccess)
           ? VV_OK : VV_ERR_CUDA;
}

} /* extern "C" */
