/**
 * @file metal_internal.h
 * @brief What the op wrappers see of the Metal runtime (metal_runtime.m).
 *
 * The wrappers are plain C: they check arguments, fill a parameter struct
 * and hand the runtime a launch -- a kernel name, the buffers it binds, the
 * struct, and a grid in threadgroups, which is CUDA's <<<grid, block>>>
 * shape. Everything Objective-C stays in the runtime.
 *
 * Device pointers are the CPU addresses of shared-storage MTLBuffers, so the
 * callers' pointer arithmetic works as it does on CUDA; the runtime maps a
 * pointer back to (buffer, offset) when it binds it. A pointer that is not
 * inside any live allocation is a caller bug and fails the launch.
 */
#ifndef VV_METAL_INTERNAL_H
#define VV_METAL_INTERNAL_H

#include "vibevoice/types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Buffers one launch binds, at [[buffer(1)]] onwards. */
#define VV_MTL_MAX_BUFS 16

/** @brief Buffers a launch reaches only through addresses in its params
 *         (a VAE descriptor table: four per item, 32 items). */
#define VV_MTL_MAX_USES 136

/** @brief Largest parameter struct (setBytes allows 4 KB). */
#define VV_MTL_MAX_PARAMS 4096

/**
 * @brief One kernel launch.
 *
 * `params` is copied when the launch is encoded (or recorded), so it may
 * live on the caller's stack. It is bound at [[buffer(0)]] as `constant`.
 * `bufs[i]` is bound at [[buffer(i + 1)]]; NULL binds nothing, and the
 * kernel must not touch that slot. `uses` lists buffers the kernel reaches
 * through GPU addresses stored in `params` (vv_mtl_addr); they are made
 * resident for the dispatch.
 */
typedef struct vv_mtl_launch {
    const char*  kernel;                    /**< MSL function name          */
    const void*  bufs[VV_MTL_MAX_BUFS];
    int          nbufs;
    const void*  params;
    size_t       params_size;
    const void*  uses[VV_MTL_MAX_USES];
    int          nuses;
    uint32_t     grid[3];                   /**< threadgroups               */
    uint32_t     block[3];                  /**< threads per threadgroup    */
    uint32_t     tg_mem;                    /**< bytes at [[threadgroup(0)]] */
} vv_mtl_launch_t;

/** @brief Encode (or record, during a graph capture) one launch. */
vv_status_t vv_mtl_run(void* stream, const vv_mtl_launch_t* l);

/**
 * @brief The GPU address of a device pointer, for structs that carry
 *        pointers (descriptor tables); 0 for NULL or an unknown pointer.
 *        Every such buffer must also go in the launch's `uses`.
 */
uint64_t vv_mtl_addr(const void* p);

/** @brief Whether `p` lies inside a live device allocation. */
int vv_mtl_is_device_ptr(const void* p);

/** @brief GPU cores of the device (the CUDA path's "SM count"). */
int vv_mtl_core_count(void);

/** @brief Simple 1-D helper: `n` threads in blocks of `block`. */
static inline void vv_mtl_grid1(vv_mtl_launch_t* l, uint64_t n, uint32_t block) {
    l->block[0] = block; l->block[1] = 1; l->block[2] = 1;
    l->grid[0] = (uint32_t)((n + block - 1) / block);
    l->grid[1] = 1; l->grid[2] = 1;
}

#ifdef __cplusplus
}
#endif

#endif /* VV_METAL_INTERNAL_H */
