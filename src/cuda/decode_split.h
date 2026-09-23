/**
 * @file decode_split.h
 * @brief Layout of the split-K decode scratch, shared by the raw and the
 *        quantized decode kernels.
 *
 * Both walk the KV cache in `n_parts` slices, each warp keeping its own
 * online-softmax state, and both merge those states in a second kernel. The
 * partials therefore have to survive between two launches on one stream.
 *
 * The buffer is the caller's — `vv_gqa_decode_scratch_bytes()` sizes it and
 * an inference context allocates one — so that two contexts decoding at the
 * same time on different streams cannot land on the same partials. They did
 * once, and the second request's transcript quietly drifted.
 */
#ifndef VV_CUDA_DECODE_SPLIT_H
#define VV_CUDA_DECODE_SPLIT_H

#include <stddef.h>

/* The split counts are needed on both sides: the host sizes the grid, and a
 * verified block's rows recompute theirs on the device. */
#ifdef __CUDACC__
#define VV_SPLIT_HD __host__ __device__
#else
#define VV_SPLIT_HD
#endif

/*
 * Decode attention is latency bound, not bandwidth bound: each warp walks its
 * slice one position at a time and every step depends on a warp reduction, so
 * nothing hides the K/V load latency. The only lever is more warps in flight,
 * hence a short slice and a high cap on the split count.
 */
#define VV_DECODE_WARPS         8
#define VV_DECODE_POS_PER_WARP  128
#define VV_DECODE_MAX_PARTS     256

/** @brief The three partial arrays carved out of one scratch allocation. */
typedef struct {
    float* o;   /**< [n_q_heads][VV_DECODE_MAX_PARTS][head_dim] */
    float* m;   /**< [n_q_heads][VV_DECODE_MAX_PARTS]           */
    float* l;   /**< [n_q_heads][VV_DECODE_MAX_PARTS]           */
} vv_decode_parts_t;

static inline size_t vv_decode_parts_bytes(int n_q_heads, int head_dim) {
    const size_t rows = (size_t)n_q_heads * VV_DECODE_MAX_PARTS;
    return rows * ((size_t)head_dim + 2) * sizeof(float);
}

static inline vv_decode_parts_t vv_decode_parts(void* scratch,
                                                int n_q_heads, int head_dim) {
    const size_t rows = (size_t)n_q_heads * VV_DECODE_MAX_PARTS;
    vv_decode_parts_t p;
    p.o = (float*)scratch;
    p.m = p.o + rows * (size_t)head_dim;
    p.l = p.m + rows;
    return p;
}

/**
 * @brief The partials of `rows` decodes at once (a verified block): every
 *        row's o, then every row's m, then every row's l, each row's the
 *        one-row layout. The kernels step to row r by r times that layout.
 */
static inline size_t vv_decode_parts_rows_bytes(int n_q_heads, int head_dim,
                                                int rows) {
    return (size_t)rows * vv_decode_parts_bytes(n_q_heads, head_dim);
}

static inline vv_decode_parts_t vv_decode_parts_rows(void* scratch,
                                                     int n_q_heads,
                                                     int head_dim, int rows) {
    const size_t per = (size_t)n_q_heads * VV_DECODE_MAX_PARTS;
    vv_decode_parts_t p;
    p.o = (float*)scratch;
    p.m = p.o + (size_t)rows * per * (size_t)head_dim;
    p.l = p.m + (size_t)rows * per;
    return p;
}

/**
 * @brief Split count for a cache of `cache_len`, rounded to whole blocks.
 *
 * The kernel gives each part `ceil(cache_len / n_parts)` positions, so this
 * only has to be a multiple of the warps per block and within the cap.
 */
static inline VV_SPLIT_HD int vv_decode_n_parts(int cache_len) {
    int n = (cache_len + VV_DECODE_POS_PER_WARP - 1) / VV_DECODE_POS_PER_WARP;
    if (n < 1) n = 1;
    if (n > VV_DECODE_MAX_PARTS) n = VV_DECODE_MAX_PARTS;
    return ((n + VV_DECODE_WARPS - 1) / VV_DECODE_WARPS) * VV_DECODE_WARPS;
}

/** @brief Split count of the flashinfer decode; steps every 1024
 *         positions, at most VV_DECODE_MAX_PARTS / n_kv_heads (>= 4). */
static inline VV_SPLIT_HD int vv_fi_decode_parts(int n_kv_heads,
                                                  int cache_len) {
    int b = (cache_len + 1023) / 1024;
    if (b < 1) b = 1;
    int cap = VV_DECODE_MAX_PARTS / (n_kv_heads > 0 ? n_kv_heads : 1);
    if (cap < 4) cap = 4;
    if (cap > VV_DECODE_MAX_PARTS) cap = VV_DECODE_MAX_PARTS;
    const int p = 4 * b;
    return p < cap ? p : cap;
}

#endif /* VV_CUDA_DECODE_SPLIT_H */
