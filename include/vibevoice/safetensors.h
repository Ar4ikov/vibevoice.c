/**
 * @file safetensors.h
 * @brief Safetensors binary format parser.
 *
 * Safetensors format:
 *   [8 bytes] header_size (uint64, little-endian)
 *   [N bytes] header JSON: {"tensor_name": {"dtype":"XX","shape":[...],"data_offsets":[start,end]}}
 *   [M bytes] raw tensor data
 */
#ifndef VV_SAFETENSORS_H
#define VV_SAFETENSORS_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque handle for a loaded safetensors file. */
typedef struct vv_safetensors vv_safetensors_t;

/** @brief Tensor metadata from safetensors header. */
typedef struct vv_st_tensor_info {
    char        name[256];
    vv_dtype_t  dtype;
    int64_t     shape[VV_MAX_DIMS];
    int         ndim;
    size_t      data_offset;   /**< Offset into data section */
    size_t      data_size;     /**< Size in bytes */
} vv_st_tensor_info_t;

/**
 * @brief Open a safetensors file (memory-mapped).
 * @param path  File path
 * @param out   Output handle
 */
vv_status_t vv_safetensors_open(const char* path, vv_safetensors_t** out);

/**
 * @brief Get number of tensors in the file.
 */
int vv_safetensors_num_tensors(const vv_safetensors_t* st);

/**
 * @brief Get tensor info by index.
 */
vv_status_t vv_safetensors_get_info(const vv_safetensors_t* st, int index,
                                     vv_st_tensor_info_t* info);

/**
 * @brief Get tensor info by name.
 */
vv_status_t vv_safetensors_find(const vv_safetensors_t* st, const char* name,
                                 vv_st_tensor_info_t* info);

/**
 * @brief Get raw data pointer for a tensor (zero-copy from mmap).
 * Valid as long as the safetensors handle is open.
 */
vv_status_t vv_safetensors_get_data(const vv_safetensors_t* st,
                                     const vv_st_tensor_info_t* info,
                                     const void** data_ptr);

/**
 * @brief Close safetensors file and release resources.
 */
vv_status_t vv_safetensors_close(vv_safetensors_t* st);

#ifdef __cplusplus
}
#endif

#endif /* VV_SAFETENSORS_H */
