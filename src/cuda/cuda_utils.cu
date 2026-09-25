/**
 * @file cuda_utils.cu
 * @brief CUDA utility functions: memory, streams, device management.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Allocate GPU memory.
 */
vv_status_t vv_dev_alloc(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        /* Not sticky, but it stays the thread's last error, and the next
         * kernel launch's check would report it as its own failure: a
         * drafter that did not fit took the transcription down with it. */
        (void)cudaGetLastError();
        return err == cudaErrorMemoryAllocation ? VV_ERR_CUDA_OOM : VV_ERR_CUDA;
    }
    return VV_OK;
}

/**
 * @brief Allocate pinned (page-locked) host memory for fast transfers.
 */
vv_status_t vv_dev_alloc_pinned(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();        /* as in vv_dev_alloc */
        return VV_ERR_CUDA;
    }
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

vv_status_t vv_dev_memcpy2d_d2d(void* dst, size_t dpitch, const void* src,
                                size_t spitch, size_t width, size_t height,
                                void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (width == 0 || height == 0) return VV_OK;
    const cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width,
                                              height, cudaMemcpyDeviceToDevice,
                                              (cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/* ─── A position that lives on the device ────────────────────────────────── */

__global__ void pos_add_kernel(int* dst, const int* src, int delta) {
    *dst = *src + delta;
}

/**
 * @brief Copy one row into `dst_base + (*d_index) * bytes`.
 *
 * A cudaMemcpyAsync would do, except that its destination is computed on the
 * host and a captured graph would replay the same address for every token.
 */
__global__ void copy_at_kernel(uint4* dst_base, const uint4* src,
                               const int* d_index, int vecs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vecs) dst_base[(size_t)(*d_index) * vecs + i] = src[i];
}

__global__ void copy_at_tail_kernel(unsigned char* dst_base,
                                    const unsigned char* src,
                                    const int* d_index, int bytes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes) dst_base[(size_t)(*d_index) * bytes + i] = src[i];
}

/** @brief Rows into `dst_base + (*d_row) * row_vecs`, 16 bytes a thread. */
__global__ void copy_rows_at_kernel(uint4* dst_base, const uint4* src,
                                    const int* d_row, int row_vecs, int vecs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vecs) dst_base[(size_t)(*d_row) * row_vecs + i] = src[i];
}

__global__ void copy_rows_at_tail_kernel(unsigned char* dst_base,
                                         const unsigned char* src,
                                         const int* d_row, int row_bytes,
                                         int bytes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes) dst_base[(size_t)(*d_row) * row_bytes + i] = src[i];
}

vv_status_t vv_pos_add_dev(int* dst, const int* src, int delta, void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    pos_add_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(dst, src, delta);
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dev_memcpy_d2d_at(void* dst_base, const void* src, size_t bytes,
                                 const int* d_index, void* stream) {
    if (!dst_base || !src || !d_index) return VV_ERR_NULL_PTR;
    if (bytes == 0) return VV_OK;
    cudaStream_t st = (cudaStream_t)stream;

    /* One row of a KV cache is a multiple of 16 bytes in every format. */
    if ((bytes % sizeof(uint4)) == 0 &&
        ((uintptr_t)dst_base % sizeof(uint4)) == 0 &&
        ((uintptr_t)src % sizeof(uint4)) == 0) {
        const int vecs = (int)(bytes / sizeof(uint4));
        const int threads = 128;
        copy_at_kernel<<<(vecs + threads - 1) / threads, threads, 0, st>>>(
            (uint4*)dst_base, (const uint4*)src, d_index, vecs);
    } else {
        const int threads = 128;
        copy_at_tail_kernel<<<((int)bytes + threads - 1) / threads, threads,
                              0, st>>>(
            (unsigned char*)dst_base, (const unsigned char*)src, d_index,
            (int)bytes);
    }
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

vv_status_t vv_dev_memcpy_d2d_rows_at(void* dst_base, const void* src,
                                      size_t row_bytes, int n_rows,
                                      const int* d_row, void* stream) {
    if (!dst_base || !src || !d_row) return VV_ERR_NULL_PTR;
    if (row_bytes == 0 || n_rows <= 0) return VV_OK;
    const size_t bytes = row_bytes * (size_t)n_rows;
    if (bytes > (size_t)INT_MAX) return VV_ERR_INVALID_ARG;
    cudaStream_t st = (cudaStream_t)stream;
    const int threads = 128;
    if ((row_bytes % sizeof(uint4)) == 0 &&
        ((uintptr_t)dst_base % sizeof(uint4)) == 0 &&
        ((uintptr_t)src % sizeof(uint4)) == 0) {
        const int vecs = (int)(bytes / sizeof(uint4));
        copy_rows_at_kernel<<<(vecs + threads - 1) / threads, threads, 0, st>>>(
            (uint4*)dst_base, (const uint4*)src, d_row,
            (int)(row_bytes / sizeof(uint4)), vecs);
    } else {
        copy_rows_at_tail_kernel<<<((int)bytes + threads - 1) / threads,
                                   threads, 0, st>>>(
            (unsigned char*)dst_base, (const unsigned char*)src, d_row,
            (int)row_bytes, (int)bytes);
    }
    return cudaGetLastError() == cudaSuccess ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/* ─── Graph capture ──────────────────────────────────────────────────────── */

vv_status_t vv_dev_graph_begin(void* stream) {
    if (!stream) return VV_ERR_INVALID_ARG;   /* the default stream cannot */
    cudaError_t err = cudaStreamBeginCapture((cudaStream_t)stream,
                                             cudaStreamCaptureModeThreadLocal);
    if (err == cudaSuccess) return VV_OK;
    cudaGetLastError();
    return VV_ERR_CUDA;
}

vv_status_t vv_dev_graph_end(void* stream, void** graph_exec) {
    if (!stream || !graph_exec) return VV_ERR_NULL_PTR;
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &graph);
    if (err != cudaSuccess || !graph) {
        /*
         * A capture can be invalidated from outside: another slot's cudaFree
         * synchronises the whole device, and a synchronised stream stops being
         * capturable. Everything issued since begin is then discarded, which
         * the caller handles by issuing it again for real -- but only if the
         * error does not linger. Leaving it here made the next launch wrapper
         * see a failure that had nothing to do with it and abandon the token.
         */
        cudaGetLastError();
        if (graph) cudaGraphDestroy(graph);
        return VV_ERR_CUDA;
    }

    cudaGraphExec_t exec = NULL;
    err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    /* The template is not needed once instantiated. */
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) { cudaGetLastError(); return VV_ERR_CUDA; }

    *graph_exec = exec;
    return VV_OK;
}

vv_status_t vv_dev_graph_launch(void* graph_exec, void* stream) {
    if (!graph_exec) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaGraphLaunch((cudaGraphExec_t)graph_exec,
                                      (cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

void vv_dev_graph_destroy(void* graph_exec) {
    if (graph_exec) cudaGraphExecDestroy((cudaGraphExec_t)graph_exec);
}

/**
 * @brief Fill device memory, ordered on a stream.
 *
 * The unstreamed cudaMemset runs on the legacy default stream, which
 * synchronises with every other blocking stream in the process. On a server
 * that means one request resetting its cache stalls all the others and
 * invalidates any capture in flight.
 */
vv_status_t vv_dev_memset_async(void* ptr, int value, size_t size,
                                void* stream) {
    if (!ptr) return VV_ERR_NULL_PTR;
    cudaError_t err = cudaMemsetAsync(ptr, value, size, (cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief Create a CUDA stream, one priority step above the lowest.
 *
 * The lowest priority is kept for vv_dev_stream_create_background, so that
 * bulk work on those streams (the speech encoder) yields to everything
 * else at block boundaries. On a device without priorities both are the
 * same, which is what cudaStreamCreate gave before.
 */
static vv_status_t stream_create_prio(void** stream, bool background) {
    if (!stream) return VV_ERR_NULL_PTR;
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
        cudaGetLastError();
        least = greatest = 0;
    }
    /* Numerically lower is more urgent: greatest <= least. */
    const int prio = background ? least : (greatest < least ? least - 1 : least);
    cudaError_t err = cudaStreamCreateWithPriority((cudaStream_t*)stream,
                                                   cudaStreamDefault, prio);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

vv_status_t vv_dev_stream_create(void** stream) {
    return stream_create_prio(stream, false);
}

vv_status_t vv_dev_stream_create_background(void** stream) {
    return stream_create_prio(stream, true);
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
 * @brief Devices this process can use; 0 when the driver is missing.
 *
 * cudaGetDeviceCount fails rather than returning zero when there is no
 * driver at all, which is the case the CPU path exists for, so the error is
 * folded into the count.
 */
vv_status_t vv_dev_memcpy_peer(void* dst, int dst_device,
                               const void* src, int src_device,
                               size_t size, void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    const cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device, src,
                                                src_device, size,
                                                (cudaStream_t)stream);
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA;
}

vv_status_t vv_dev_enable_peer(int device, int peer) {
    if (device == peer) return VV_OK;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, device, peer) != cudaSuccess || !can)
        return VV_ERR_UNSUPPORTED;
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(device) != cudaSuccess) return VV_ERR_CUDA;
    const cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
    cudaSetDevice(prev);
    /* Already enabled is the state we wanted, not a failure. */
    if (err == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return VV_OK; }
    return (err == cudaSuccess) ? VV_OK : VV_ERR_UNSUPPORTED;
}

int vv_dev_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

/* A discrete card: host copies are what the CPU paths read. */
bool vv_dev_host_shares_memory(int device_id) {
    (void)device_id;
    return false;
}

vv_status_t vv_dev_get_device_name(int device_id, char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return VV_ERR_NULL_PTR;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        buf[0] = '\0';
        return VV_ERR_CUDA;
    }
    snprintf(buf, buf_size, "%s", prop.name);
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

vv_status_t vv_dev_event_sync(void* ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    return cudaEventSynchronize((cudaEvent_t)ev) == cudaSuccess
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
