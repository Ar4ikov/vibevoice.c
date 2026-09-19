/**
 * @file attn_internal.h
 * @brief Per-backend attention entry points behind vv_attn_prefill/decode.
 *
 * Not part of the device op set: callers go through the dispatch in
 * device.h, which is the only place that knows which of these a backend is.
 */
#ifndef VV_CUDA_ATTN_INTERNAL_H
#define VV_CUDA_ATTN_INTERNAL_H

#include "vibevoice/types.h"
#include "vibevoice/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief fa1: the tiled scalar prefill (one warp per query row), FP16 KV. */
vv_status_t vv_attn_fa1_prefill_dev(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal, void* stream);

/**
 * @brief GQA-packed tensor-core prefill.
 * @param packed_only  true for fa2: always the 64-row kernel, never split KV.
 */
vv_status_t vv_attn_fi_prefill_dev(
    bool packed_only, const void* q, const vv_kv_view_t* kv, void* out,
    int n_q_heads, int q_len, int q_offset, int kv_len, bool causal,
    void* scratch, void* stream);

/** @brief Tensor-core GQA decode with split KV and a merge. */
vv_status_t vv_attn_fi_decode_dev(
    const void* q, const vv_kv_view_t* kv, void* out, int n_q_heads,
    int cache_len, const int* d_cache_len, void* scratch, void* stream);

/** @brief Whether vv_attn_gqa_decode_dev takes this head layout. */
bool vv_attn_gqa_decode_ok(int n_q_heads, int n_kv_heads, int head_dim);

/**
 * @brief fa2 decode: GQA-packed split-KV decode on CUDA cores, any format,
 *        contiguous or paged, bit-identical to the per-head fa1 kernels.
 */
vv_status_t vv_attn_gqa_decode_dev(
    const void* q, const vv_kv_view_t* kv, void* out, int n_q_heads,
    int cache_len, const int* d_cache_len, void* scratch, void* stream);

/** @brief Compute capability of the current device as major*10+minor. */
int vv_attn_device_sm(void);

/** @brief Multiprocessor count of the current device (cached per thread). */
int vv_attn_device_sm_count(void);

#ifdef __cplusplus
}
#endif

#endif /* VV_CUDA_ATTN_INTERNAL_H */
