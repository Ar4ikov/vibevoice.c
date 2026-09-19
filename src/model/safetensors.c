/**
 * @file safetensors.c
 * @brief Safetensors format parser with memory-mapped file access.
 *
 * Format:
 *   [8 bytes]  header_size (uint64 LE)
 *   [N bytes]  header JSON
 *   [M bytes]  raw tensor data (contiguous)
 */

#include "vibevoice/safetensors.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ─── Internal structure ────────────────────────────────────────────────── */

struct vv_safetensors {
    /* Memory-mapped file data */
#ifdef _WIN32
    HANDLE      file_handle;
    HANDLE      mapping;
#else
    int         fd;
#endif
    const uint8_t* mapped_data;
    size_t         mapped_size;

    /* Parsed header */
    uint64_t         header_size;
    const uint8_t*   data_start;     /**< Start of tensor data section */

    /* Tensor metadata */
    vv_st_tensor_info_t* tensors;
    int                  num_tensors;
};

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static vv_dtype_t parse_dtype(const char* s) {
    if (!s) return VV_DTYPE_F32;
    if (strcmp(s, "F32") == 0)     return VV_DTYPE_F32;
    if (strcmp(s, "F16") == 0)     return VV_DTYPE_F16;
    if (strcmp(s, "BF16") == 0)    return VV_DTYPE_BF16;
    if (strcmp(s, "U8") == 0)      return VV_DTYPE_U8;
    if (strcmp(s, "I32") == 0)     return VV_DTYPE_I32;
    if (strcmp(s, "I64") == 0)     return VV_DTYPE_I64;
    if (strcmp(s, "F8_E4M3") == 0) return VV_DTYPE_F8_E4M3;
    if (strcmp(s, "BOOL") == 0)    return VV_DTYPE_BOOL;
    if (strcmp(s, "I8") == 0)      return VV_DTYPE_I8;
    if (strcmp(s, "I16") == 0)     return VV_DTYPE_I16;
    if (strcmp(s, "U16") == 0)     return VV_DTYPE_U16;
    if (strcmp(s, "U32") == 0)     return VV_DTYPE_U32;
    if (strcmp(s, "F8_E5M2") == 0) return VV_DTYPE_F8_E5M2;
    if (strcmp(s, "F64") == 0)     return VV_DTYPE_F64;
    /* Guessing F32 here once made an unknown tensor load as garbage. */
    VV_LOG_W("safetensors: unknown dtype '%s'", s);
    return VV_DTYPE_UNKNOWN;
}

static size_t mmap_file(const char* path, const uint8_t** out_data,
                         void** platform_handles) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return 0;

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(file, &fsize)) {
        CloseHandle(file);
        return 0;
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY,
                                         fsize.HighPart, fsize.LowPart, NULL);
    if (!mapping) {
        CloseHandle(file);
        return 0;
    }

    const uint8_t* data = (const uint8_t*)MapViewOfFile(
        mapping, FILE_MAP_READ, 0, 0, 0);
    if (!data) {
        CloseHandle(mapping);
        CloseHandle(file);
        return 0;
    }

    /* Store handles for cleanup */
    platform_handles[0] = file;
    platform_handles[1] = mapping;
    *out_data = data;
    return (size_t)fsize.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return 0;
    }

    const uint8_t* data = (const uint8_t*)mmap(
        NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return 0;
    }

    platform_handles[0] = (void*)(intptr_t)fd;
    *out_data = data;
    return (size_t)st.st_size;
#endif
}

/* ─── Public API ────────────────────────────────────────────────────────── */

vv_status_t vv_safetensors_open(const char* path, vv_safetensors_t** out) {
    if (!path || !out) return VV_ERR_NULL_PTR;

    vv_safetensors_t* st = (vv_safetensors_t*)vv_alloc(sizeof(vv_safetensors_t));
    if (!st) return VV_ERR_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));

    /* Memory-map the file */
    void* handles[2] = {0};
    st->mapped_size = mmap_file(path, &st->mapped_data, handles);
    if (st->mapped_size == 0 || !st->mapped_data) {
        VV_LOG_E("safetensors: failed to mmap '%s'", path);
        vv_free(st);
        return VV_ERR_IO;
    }

#ifdef _WIN32
    st->file_handle = handles[0];
    st->mapping = handles[1];
#else
    st->fd = (int)(intptr_t)handles[0];
#endif

    /* Read header size (first 8 bytes, uint64 LE) */
    if (st->mapped_size < 8) {
        VV_LOG_E("safetensors: file too small");
        vv_safetensors_close(st);
        return VV_ERR_MODEL_FORMAT;
    }
    memcpy(&st->header_size, st->mapped_data, 8);

    if (8 + st->header_size > st->mapped_size) {
        VV_LOG_E("safetensors: header_size (%llu) exceeds file size",
                 (unsigned long long)st->header_size);
        vv_safetensors_close(st);
        return VV_ERR_MODEL_FORMAT;
    }

    st->data_start = st->mapped_data + 8 + st->header_size;

    /* Parse header JSON */
    char* header_str = (char*)vv_alloc(st->header_size + 1);
    if (!header_str) {
        vv_safetensors_close(st);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memcpy(header_str, st->mapped_data + 8, st->header_size);
    header_str[st->header_size] = '\0';

    cJSON* root = cJSON_Parse(header_str);
    vv_free(header_str);
    if (!root) {
        VV_LOG_E("safetensors: JSON parse failed");
        vv_safetensors_close(st);
        return VV_ERR_PARSE;
    }

    /* Count tensors (skip __metadata__ key) */
    int count = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") != 0) {
            count++;
        }
    }

    st->tensors = (vv_st_tensor_info_t*)vv_alloc(
        (size_t)count * sizeof(vv_st_tensor_info_t));
    if (!st->tensors) {
        cJSON_Delete(root);
        vv_safetensors_close(st);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(st->tensors, 0, (size_t)count * sizeof(vv_st_tensor_info_t));
    st->num_tensors = count;

    int idx = 0;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") == 0) continue;
        if (idx >= count) break;

        vv_st_tensor_info_t* info = &st->tensors[idx];

        /* Name */
        size_t name_len = strlen(item->string);
        if (name_len >= sizeof(info->name)) name_len = sizeof(info->name) - 1;
        memcpy(info->name, item->string, name_len);
        info->name[name_len] = '\0';

        /* dtype */
        cJSON* dtype_val = cJSON_GetObjectItem(item, "dtype");
        info->dtype = parse_dtype(dtype_val ? dtype_val->valuestring : NULL);

        /* shape */
        cJSON* shape_arr = cJSON_GetObjectItem(item, "shape");
        if (shape_arr && cJSON_IsArray(shape_arr)) {
            info->ndim = cJSON_GetArraySize(shape_arr);
            if (info->ndim > VV_MAX_DIMS) info->ndim = VV_MAX_DIMS;
            for (int d = 0; d < info->ndim; d++) {
                cJSON* dim = cJSON_GetArrayItem(shape_arr, d);
                info->shape[d] = dim ? (int64_t)dim->valuedouble : 0;
            }
        }

        /* data_offsets */
        cJSON* offsets = cJSON_GetObjectItem(item, "data_offsets");
        if (offsets && cJSON_IsArray(offsets) &&
            cJSON_GetArraySize(offsets) >= 2) {
            size_t start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
            size_t end   = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
            info->data_offset = start;
            info->data_size = end - start;
        }

        idx++;
    }

    cJSON_Delete(root);
    *out = st;
    VV_LOG_I("safetensors: opened '%s' with %d tensors", path, st->num_tensors);
    return VV_OK;
}

int vv_safetensors_num_tensors(const vv_safetensors_t* st) {
    return st ? st->num_tensors : 0;
}

vv_status_t vv_safetensors_get_info(const vv_safetensors_t* st, int index,
                                     vv_st_tensor_info_t* info) {
    if (!st || !info) return VV_ERR_NULL_PTR;
    if (index < 0 || index >= st->num_tensors) return VV_ERR_INVALID_ARG;
    *info = st->tensors[index];
    return VV_OK;
}

vv_status_t vv_safetensors_find(const vv_safetensors_t* st, const char* name,
                                 vv_st_tensor_info_t* info) {
    if (!st || !name || !info) return VV_ERR_NULL_PTR;

    for (int i = 0; i < st->num_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            *info = st->tensors[i];
            return VV_OK;
        }
    }
    return VV_ERR_NOT_FOUND;
}

vv_status_t vv_safetensors_get_data(const vv_safetensors_t* st,
                                     const vv_st_tensor_info_t* info,
                                     const void** data_ptr) {
    if (!st || !info || !data_ptr) return VV_ERR_NULL_PTR;

    const uint8_t* base = st->data_start + info->data_offset;
    size_t data_section_size = st->mapped_size - 8 - st->header_size;

    if (info->data_offset + info->data_size > data_section_size) {
        VV_LOG_E("safetensors: tensor '%s' data out of bounds", info->name);
        return VV_ERR_MODEL_FORMAT;
    }

    *data_ptr = base;
    return VV_OK;
}

vv_status_t vv_safetensors_close(vv_safetensors_t* st) {
    if (!st) return VV_ERR_NULL_PTR;

    if (st->mapped_data) {
#ifdef _WIN32
        UnmapViewOfFile(st->mapped_data);
        if (st->mapping) CloseHandle(st->mapping);
        if (st->file_handle) CloseHandle(st->file_handle);
#else
        munmap((void*)st->mapped_data, st->mapped_size);
        if (st->fd >= 0) close(st->fd);
#endif
        st->mapped_data = NULL;
    }

    if (st->tensors) {
        vv_free(st->tensors);
        st->tensors = NULL;
    }

    vv_free(st);
    return VV_OK;
}
