/**
 * @file gguf.h
 * @brief Read-only GGUF (v2/v3) reader, memory-mapped.
 *
 * GGUF is the container ggml writes. It is here for one reason: the
 * ready-made VibeVoice-ASR-BitNet files (`vibeasr-lm-i2_s-embed-q6_k.gguf`,
 * `vibeasr-vae-encoder-i8_s.gguf`) are GGUF, produced by the ggml fork that
 * ships with microsoft/VibeASR.cpp, and two of their tensor types (I2_S = 36,
 * I8_S = 37) exist only in that fork. docs/BITNET.md has the byte layouts.
 *
 * File layout (all little-endian):
 *
 *     "GGUF" u32 version, u64 n_tensors, u64 n_kv
 *     n_kv    x { string key, u32 type, value }
 *     n_tensors x { string name, u32 n_dims, u64 ne[n_dims], u32 type,
 *                   u64 offset }                  offset: into the data block
 *     padding to general.alignment (default 32)
 *     data block
 *
 * `ne[0]` is the innermost (contiguous) dimension, i.e. the reverse of the
 * PyTorch shape: a Linear(in=K, out=N) weight has ne = {K, N}.
 *
 * The reader validates every length and offset against the file size before
 * it hands out a pointer, so a truncated or hostile file fails with
 * VV_ERR_PARSE instead of reading past the mapping.
 */
#ifndef VV_GGUF_H
#define VV_GGUF_H

#include "vibevoice/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ggml tensor type ids this reader knows the size of. */
typedef enum vv_ggml_type {
    VV_GGML_F32  = 0,
    VV_GGML_F16  = 1,
    VV_GGML_Q4_0 = 2,
    VV_GGML_Q4_1 = 3,
    VV_GGML_Q5_0 = 6,
    VV_GGML_Q5_1 = 7,
    VV_GGML_Q8_0 = 8,
    VV_GGML_Q8_1 = 9,
    VV_GGML_Q2_K = 10,
    VV_GGML_Q3_K = 11,
    VV_GGML_Q4_K = 12,
    VV_GGML_Q5_K = 13,
    VV_GGML_Q6_K = 14,
    VV_GGML_Q8_K = 15,
    VV_GGML_I8   = 24,
    VV_GGML_I16  = 25,
    VV_GGML_I32  = 26,
    VV_GGML_BF16 = 30,
    /** BitNet fork: ternary, 2 bits per weight, one FP32 scale per tensor. */
    VV_GGML_I2_S = 36,
    /** BitNet fork: int8, one FP32 scale per tensor. */
    VV_GGML_I8_S = 37,
} vv_ggml_type_t;

/** @brief GGUF metadata value types. */
typedef enum vv_gguf_vtype {
    VV_GGUF_U8 = 0, VV_GGUF_I8 = 1, VV_GGUF_U16 = 2, VV_GGUF_I16 = 3,
    VV_GGUF_U32 = 4, VV_GGUF_I32 = 5, VV_GGUF_F32 = 6, VV_GGUF_BOOL = 7,
    VV_GGUF_STRING = 8, VV_GGUF_ARRAY = 9, VV_GGUF_U64 = 10,
    VV_GGUF_I64 = 11, VV_GGUF_F64 = 12,
} vv_gguf_vtype_t;

/** @brief One tensor as described by the file. Pointers borrow the mapping. */
typedef struct vv_gguf_tensor {
    const char*    name;      /**< NUL-terminated copy owned by the handle */
    int            n_dims;
    int64_t        ne[4];     /**< ggml order: ne[0] contiguous; unused = 1 */
    uint32_t       type;      /**< vv_ggml_type_t (unknown ids are kept)    */
    uint64_t       offset;    /**< relative to the data block               */
    const uint8_t* data;      /**< into the mapping                         */
    size_t         nbytes;    /**< including the I2_S/I8_S scale trailer    */
} vv_gguf_tensor_t;

/** @brief One metadata entry. Scalars are widened; arrays are borrowed. */
typedef struct vv_gguf_kv {
    const char*    key;
    uint32_t       type;      /**< vv_gguf_vtype_t                          */
    uint32_t       arr_type;  /**< element type when type == ARRAY          */
    uint64_t       arr_n;     /**< element count when type == ARRAY         */
    const uint8_t* arr_data;  /**< first element, in the mapping            */
    union {
        uint64_t u;
        int64_t  i;
        double   f;
    } v;                      /**< scalar value (bools in .u)               */
    const char*    str;       /**< NUL-terminated copy for STRING           */
} vv_gguf_kv_t;

/** @brief Opaque handle. */
typedef struct vv_gguf vv_gguf_t;

/** @brief Map and parse a GGUF file. */
vv_status_t vv_gguf_open(const char* path, vv_gguf_t** out);

/**
 * @brief Parse a GGUF image already in memory. The buffer must outlive the
 *        handle; it is not copied. Used by the tests.
 */
vv_status_t vv_gguf_open_memory(const void* data, size_t size,
                                vv_gguf_t** out);

/** @brief Unmap and free. NULL is accepted. */
void vv_gguf_close(vv_gguf_t* g);

uint32_t vv_gguf_version(const vv_gguf_t* g);
uint32_t vv_gguf_alignment(const vv_gguf_t* g);
/** @brief Byte offset of the data block from the start of the file. */
uint64_t vv_gguf_data_offset(const vv_gguf_t* g);

int                    vv_gguf_n_tensors(const vv_gguf_t* g);
const vv_gguf_tensor_t* vv_gguf_tensor(const vv_gguf_t* g, int i);
/** @brief Tensor by exact name, or NULL. */
const vv_gguf_tensor_t* vv_gguf_find_tensor(const vv_gguf_t* g,
                                            const char* name);

int                 vv_gguf_n_kv(const vv_gguf_t* g);
const vv_gguf_kv_t* vv_gguf_kv(const vv_gguf_t* g, int i);
/** @brief Metadata entry by key, or NULL. */
const vv_gguf_kv_t* vv_gguf_find_kv(const vv_gguf_t* g, const char* key);

/** @brief Integer metadata (any integer or bool type), else @p def. */
int64_t     vv_gguf_get_int(const vv_gguf_t* g, const char* key, int64_t def);
/** @brief Float metadata (F32/F64 or integer), else @p def. */
double      vv_gguf_get_float(const vv_gguf_t* g, const char* key, double def);
/** @brief String metadata, else NULL. */
const char* vv_gguf_get_str(const vv_gguf_t* g, const char* key);
/**
 * @brief Element @p idx of a STRING array (e.g. tokenizer.ggml.tokens).
 * @param len receives the byte length; the string is NOT NUL-terminated.
 */
const char* vv_gguf_arr_str(const vv_gguf_kv_t* kv, uint64_t idx,
                            size_t* len);

/** @brief Name of a ggml type id ("f32", "q6_K", "i2_s", ...). */
const char* vv_ggml_type_name(uint32_t type);

/**
 * @brief Bytes a tensor of this type and shape occupies in the file,
 *        or 0 for a type this reader does not know.
 *
 * I2_S is `nelements / 4 + 32` and I8_S `nelements + 32`: the per-tensor
 * FP32 scale sits right after the codes, padded to 32 bytes.
 */
size_t vv_ggml_nbytes(uint32_t type, const int64_t ne[4]);

/** @brief Number of elements (product of ne). */
int64_t vv_gguf_nelements(const vv_gguf_tensor_t* t);

/* ─── Decoding into the runtime's layouts ───────────────────────────────── */

/**
 * @brief Zero-copy view of an I2_S tensor.
 *
 * The I2_S byte layout is also the runtime's ternary layout (see
 * vv_ternary_* in bitnet.h), so this is just the data pointer and the
 * trailing scale: w[n][k] = scale * (code(n, k) - 1).
 *
 * @param codes  receives the packed codes, `ne[0]*ne[1]/4` bytes
 * @param scale  receives the per-tensor scale (= max|w|)
 */
vv_status_t vv_gguf_i2s_view(const vv_gguf_tensor_t* t,
                             const uint8_t** codes, float* scale);

/**
 * @brief Zero-copy view of an I8_S tensor: w = q * scale, q in [-127, 127].
 * @param scale receives max|w| / 127 as stored by the quantizer
 */
vv_status_t vv_gguf_i8s_view(const vv_gguf_tensor_t* t,
                             const int8_t** q, float* scale);

/**
 * @brief Dequantize rows [row0, row0 + nrows) of a 2-D view of @p t
 *        (row length ne[0]) to FP32.
 *
 * Supports F32, F16, BF16, Q8_0, Q6_K, I8_S and I2_S. Q6_K follows ggml's
 * dequantize_row_q6_K operation for operation, so a gathered embedding row
 * is bit-identical to what ggml's get_rows produces.
 */
vv_status_t vv_gguf_dequant_rows_f32(const vv_gguf_tensor_t* t,
                                     int64_t row0, int64_t nrows, float* out);

/** @brief As vv_gguf_dequant_rows_f32, rounded to FP16 (round to nearest). */
vv_status_t vv_gguf_dequant_rows_f16(const vv_gguf_tensor_t* t,
                                     int64_t row0, int64_t nrows,
                                     uint16_t* out);

/**
 * @brief Dequantize one Q6_K row of @p k values (k % 256 == 0).
 *        Exposed for the embedding gather and for tests.
 */
void vv_q6k_dequant_row(const uint8_t* blocks, float* out, int64_t k);

#ifdef __cplusplus
}
#endif

#endif /* VV_GGUF_H */
