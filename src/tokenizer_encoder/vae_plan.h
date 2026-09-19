/**
 * @file vae_plan.h
 * @brief The Conv-VAE encoder as a flat list of layers (internal).
 *
 * Both the GPU and the CPU encoder walk the same list, and both need the
 * same bookkeeping for a causal convolution over a chunk of a stream: how
 * many outputs it can produce and how much input it must keep for the next
 * chunk. That arithmetic lives here, once.
 */
#ifndef VV_VAE_PLAN_H
#define VV_VAE_PLAN_H

#include "vibevoice/tokenizer_encoder.h"

#define VAE_MAX_LAYERS 128
#define VAE_MAX_STAGES 8

typedef enum {
    VAE_L_STEM = 0,
    VAE_L_MIXER,
    VAE_L_DS,
    VAE_L_HEAD
} vae_layer_kind_t;

typedef struct {
    int    kind;       /**< vae_layer_kind_t                                */
    int    stage;
    int    block;
    int    in_ch;
    int    out_ch;
    int    k;
    int    stride;
    int    cap;        /**< most context columns it can carry: k - 1        */
    size_t tail_off;   /**< elements into one buffer of a state's tails     */
} vae_layer_t;

typedef struct {
    int         n_layers;
    vae_layer_t L[VAE_MAX_LAYERS];
    size_t      tail_elems;              /**< one buffer of a state       */
    int         n_stages;
    int         C[VAE_MAX_STAGES];       /**< channels inside each stage  */
    int         hidden[VAE_MAX_STAGES];  /**< widest FFN in each stage    */
    int64_t     div[VAE_MAX_STAGES];     /**< input samples per column    */
    int         max_cap;
    int         vae_dim;
} vae_plan_t;

/**
 * @brief Lay the encoder out as layers: stem, then per stage its mixers and
 * its downsample, then the head. Fails if a weight the walk needs is absent.
 */
vv_status_t vae_plan_build(const vv_conv_vae_encoder_t* enc, vae_plan_t* p);

/**
 * @brief One causal convolution over one chunk of a stream.
 *
 * `have` columns of context sit in front of `in_len` new ones. A non-final
 * chunk emits every output whose window is complete and keeps the rest,
 * total - out * stride columns, which is between k - stride and k - 1. A
 * final chunk pads the right edge with zeros up to ceil alignment, which is
 * what SConv1d's extra_padding does to a whole signal.
 */
static inline void vae_layer_step(const vae_layer_t* L, int64_t have,
                                  int64_t in_len, bool is_final,
                                  int64_t* out_len, int64_t* keep) {
    const int64_t total = have + in_len;
    const int64_t k = L->k, s = L->stride;
    int64_t ol;
    if (is_final) {
        const int64_t n = total - (k - s);
        ol = n > 0 ? (n + s - 1) / s : 0;
        *keep = 0;
    } else {
        ol = total >= k ? (total - k) / s + 1 : 0;
        *keep = total - ol * s;
    }
    *out_len = ol;
}

/** @brief Release the CPU path's FP16 weight copies (vae_cpu.c). */
void vv_vae_cpu_weights_free(struct vv_vae_cpu_weights* cw);

/** @brief Context a fresh stream starts with: the causal left padding. */
static inline int64_t vae_layer_fresh_have(const vae_layer_t* L) {
    const int h = L->k - L->stride;
    return h > 0 ? h : 0;
}

#endif /* VV_VAE_PLAN_H */
