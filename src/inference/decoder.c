/**
 * @file decoder.c
 * @brief Qwen2 transformer decoder: per-layer forward with optional
 *        layer-weight streaming and CPU-only fallback.
 *
 * Per layer (28 total):
 * 1. RMSNorm -> Q, K, V (NF4 dequant + GEMM)
 * 2. RoPE -> GQA Attention -> O proj -> Residual
 * 3. RMSNorm -> SwiGLU MLP (NF4 dequant + GEMM) -> Residual
 *
 * Layer pool integration:
 *   If a vv_layer_pool_t is provided and !all_resident, the prefill/step
 *   functions stage each layer's weights to GPU just before processing,
 *   then unstage (restore CPU pointers) afterward.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/cpu_kernels.h"

#include <string.h>
#include <stdio.h>
#include <time.h>


/* ─── CUDA kernel forward declarations ──────────────────────────────────── */

extern vv_status_t vv_rmsnorm_cuda(
    const void* input, const void* weight, void* output,
    int seq_len, int hidden_size, float eps, void* stream);

extern vv_status_t vv_rope_cuda(
    void* x, int seq_len, int n_heads, int head_dim,
    int position_offset, float theta, void* stream);

extern vv_status_t vv_gqa_attention_decode_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int cache_len, void* stream);

extern vv_status_t vv_gqa_attention_prefill_cached_cuda(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int q_len, int q_offset, int kv_len, bool causal, void* stream);

extern vv_status_t vv_gqa_attention_prefill_cuda(
    const void* q, const void* k, const void* v,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream);

extern vv_status_t vv_swiglu_cuda(
    const void* gate, const void* up, void* output,
    int n_elements, void* stream);

extern vv_status_t vv_nf4_gemv_cuda(
    const void* x, const uint8_t* packed, const void* scales,
    const void* bias, void* y, int N, int K, void* stream);

extern vv_status_t vv_nf4_gemm_cuda(
    const void* input_fp16,
    const uint8_t* weight_packed,
    const void* weight_scales_fp16,
    void* output_fp16,
    void* temp_weight_fp16,
    int M, int N, int K,
    int block_size,
    void* stream);

extern vv_status_t vv_gemm_fp16_cuda(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    void* stream);

extern vv_status_t vv_cuda_memcpy_d2d(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_memcpy_h2d(void* dst, const void* src,
                                        size_t size, void* stream);
extern vv_status_t vv_cuda_stream_sync(void* stream);
extern vv_status_t vv_cuda_alloc(void** ptr, size_t size);
extern vv_status_t vv_cuda_free(void* ptr);

extern vv_status_t vv_residual_add_cuda(void* x, const void* y, int total,
                                          void* stream);
extern vv_status_t vv_bias_add_cuda(void* output, const void* bias,
                                      int M, int N, void* stream);

/* ═══════════════════════════════════════════════════════════════════════════
 * Layer Pool — stage / unstage helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Calculate total GPU buffer size needed for one layer's tensors.
 */
static size_t layer_gpu_size(const vv_layer_weights_t* L) {
    size_t total = 0;
    total += L->input_layernorm.size_bytes;
    total += L->post_attn_layernorm.size_bytes;

    #define ADD_WEIGHT_SIZE(w) do {           \
        total += (w).tensor.size_bytes;       \
        if ((w).quant.scales.data)            \
            total += (w).quant.scales.size_bytes; \
        if ((w).bias.data)                    \
            total += (w).bias.size_bytes;     \
    } while(0)

    ADD_WEIGHT_SIZE(L->attn.q_proj);
    ADD_WEIGHT_SIZE(L->attn.k_proj);
    ADD_WEIGHT_SIZE(L->attn.v_proj);
    ADD_WEIGHT_SIZE(L->attn.o_proj);
    ADD_WEIGHT_SIZE(L->mlp.gate_proj);
    ADD_WEIGHT_SIZE(L->mlp.up_proj);
    ADD_WEIGHT_SIZE(L->mlp.down_proj);

    #undef ADD_WEIGHT_SIZE
    return total;
}

vv_status_t vv_layer_pool_create(vv_layer_pool_t** pool,
                                  const vv_model_t* model,
                                  bool all_resident) {
    if (!pool || !model) return VV_ERR_NULL_PTR;

    vv_layer_pool_t* p = (vv_layer_pool_t*)vv_alloc(sizeof(vv_layer_pool_t));
    if (!p) return VV_ERR_OUT_OF_MEMORY;
    memset(p, 0, sizeof(*p));
    p->all_resident = all_resident;
    p->loaded[0] = p->loaded[1] = -1;

    if (!all_resident && model->num_layers > 0) {
        /* Find max layer size */
        size_t max_sz = 0;
        for (int i = 0; i < model->num_layers; i++) {
            size_t sz = layer_gpu_size(&model->layers[i]);
            if (sz > max_sz) max_sz = sz;
        }
        /* Add 256-byte alignment padding per tensor (16 tensors * 256) */
        p->buf_size = max_sz + 16 * 256;

        for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
            vv_status_t st = vv_cuda_alloc(&p->gpu_buf[s], p->buf_size);
            if (st != VV_OK) {
                /* Fall back to single buffer */
                if (s == 1) {
                    VV_LOG_W("layer_pool: only 1 staging buffer (no double-buffering)");
                    break;
                }
                vv_free(p);
                return st;
            }
        }
        VV_LOG_I("layer_pool: %d staging buffer(s) of %.1f MB each",
                 p->gpu_buf[1] ? 2 : 1,
                 (double)p->buf_size / (1024.0 * 1024.0));
    }

    *pool = p;
    return VV_OK;
}

/**
 * @brief Stage layer weights to GPU: copy CPU→GPU, patch pointers.
 */
vv_status_t vv_layer_pool_stage(vv_layer_pool_t* pool,
                                 vv_model_t* model, int layer_idx,
                                 void* stream) {
    if (!pool || pool->all_resident) return VV_OK;
    if (layer_idx < 0 || layer_idx >= model->num_layers)
        return VV_ERR_INVALID_ARG;

    int slot = layer_idx % VV_LAYER_POOL_SLOTS;
    /* If only 1 buffer, always use slot 0 */
    if (!pool->gpu_buf[1]) slot = 0;

    /* Unstage whatever was previously in this slot */
    if (pool->loaded[slot] >= 0 && pool->loaded[slot] != layer_idx) {
        vv_layer_pool_unstage(pool, model, pool->loaded[slot]);
    }
    if (pool->loaded[slot] == layer_idx) return VV_OK; /* already staged */

    vv_layer_weights_t* L = &model->layers[layer_idx];
    uint8_t* buf = (uint8_t*)pool->gpu_buf[slot];
    size_t off = 0;
    int idx = 0;

    #define STAGE_TENSOR(t) do {                                        \
        pool->saved_ptrs[slot][idx] = (t).data;                        \
        if ((t).data && !(t).on_gpu && (t).size_bytes > 0) {           \
            /* Align to 256 bytes */                                    \
            off = (off + 255) & ~(size_t)255;                          \
            vv_cuda_memcpy_h2d(buf + off, (t).data,                    \
                               (t).size_bytes, stream);                \
            (t).data = buf + off;                                      \
            (t).on_gpu = true;                                         \
            off += (t).size_bytes;                                     \
        }                                                              \
        idx++;                                                         \
    } while(0)

    STAGE_TENSOR(L->input_layernorm);
    STAGE_TENSOR(L->post_attn_layernorm);
    STAGE_TENSOR(L->attn.q_proj.tensor);
    STAGE_TENSOR(L->attn.q_proj.quant.scales);
    STAGE_TENSOR(L->attn.q_proj.bias);
    STAGE_TENSOR(L->attn.k_proj.tensor);
    STAGE_TENSOR(L->attn.k_proj.quant.scales);
    STAGE_TENSOR(L->attn.k_proj.bias);
    STAGE_TENSOR(L->attn.v_proj.tensor);
    STAGE_TENSOR(L->attn.v_proj.quant.scales);
    STAGE_TENSOR(L->attn.v_proj.bias);
    STAGE_TENSOR(L->attn.o_proj.tensor);
    STAGE_TENSOR(L->attn.o_proj.quant.scales);
    STAGE_TENSOR(L->attn.o_proj.bias);
    STAGE_TENSOR(L->mlp.gate_proj.tensor);
    STAGE_TENSOR(L->mlp.gate_proj.quant.scales);
    STAGE_TENSOR(L->mlp.up_proj.tensor);
    STAGE_TENSOR(L->mlp.up_proj.quant.scales);
    STAGE_TENSOR(L->mlp.down_proj.tensor);
    STAGE_TENSOR(L->mlp.down_proj.quant.scales);

    #undef STAGE_TENSOR

    pool->loaded[slot] = layer_idx;
    return VV_OK;
}

/**
 * @brief Unstage layer: restore original CPU pointers.
 */
vv_status_t vv_layer_pool_unstage(vv_layer_pool_t* pool,
                                   vv_model_t* model, int layer_idx) {
    if (!pool || pool->all_resident) return VV_OK;
    if (layer_idx < 0 || layer_idx >= model->num_layers)
        return VV_ERR_INVALID_ARG;

    int slot = -1;
    for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
        if (pool->loaded[s] == layer_idx) { slot = s; break; }
    }
    if (slot < 0) return VV_OK; /* not staged */

    vv_layer_weights_t* L = &model->layers[layer_idx];
    int idx = 0;

    #define UNSTAGE_TENSOR(t) do {                                      \
        if (pool->saved_ptrs[slot][idx]) {                             \
            (t).data = pool->saved_ptrs[slot][idx];                    \
            (t).on_gpu = false;                                        \
        }                                                              \
        idx++;                                                         \
    } while(0)

    UNSTAGE_TENSOR(L->input_layernorm);
    UNSTAGE_TENSOR(L->post_attn_layernorm);
    UNSTAGE_TENSOR(L->attn.q_proj.tensor);
    UNSTAGE_TENSOR(L->attn.q_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.q_proj.bias);
    UNSTAGE_TENSOR(L->attn.k_proj.tensor);
    UNSTAGE_TENSOR(L->attn.k_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.k_proj.bias);
    UNSTAGE_TENSOR(L->attn.v_proj.tensor);
    UNSTAGE_TENSOR(L->attn.v_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.v_proj.bias);
    UNSTAGE_TENSOR(L->attn.o_proj.tensor);
    UNSTAGE_TENSOR(L->attn.o_proj.quant.scales);
    UNSTAGE_TENSOR(L->attn.o_proj.bias);
    UNSTAGE_TENSOR(L->mlp.gate_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.gate_proj.quant.scales);
    UNSTAGE_TENSOR(L->mlp.up_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.up_proj.quant.scales);
    UNSTAGE_TENSOR(L->mlp.down_proj.tensor);
    UNSTAGE_TENSOR(L->mlp.down_proj.quant.scales);

    #undef UNSTAGE_TENSOR

    pool->loaded[slot] = -1;
    return VV_OK;
}

vv_status_t vv_layer_pool_free(vv_layer_pool_t* pool) {
    if (!pool) return VV_OK;
    for (int s = 0; s < VV_LAYER_POOL_SLOTS; s++) {
        if (pool->gpu_buf[s]) vv_cuda_free(pool->gpu_buf[s]);
    }
    vv_free(pool);
    return VV_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU decoder — per-layer forward (unchanged core logic)
 * ═══════════════════════════════════════════════════════════════════════════ */



/**
 * @brief One quantised linear layer: y[M,N] = x[M,K] @ dequant(W)^T + bias.
 *
 * Single-token decode takes the fused path, which reads the 4-bit weights
 * straight into the MAC instead of materialising an FP16 copy first — the
 * scratch round-trip costs ~5x the memory traffic and dominates decode.
 */
static vv_status_t quant_linear(
    const vv_weight_t* w, const void* x, void* y,
    void* scratch, int M, int N, int K, void* stream)
{
    vv_status_t s;

    if (w->is_quantized) {
        if (M == 1) {
            s = vv_nf4_gemv_cuda(x, (const uint8_t*)w->tensor.data,
                                 w->quant.scales.data, w->bias.data,
                                 y, N, K, stream);
            if (s == VV_OK) return VV_OK;      /* bias already folded in */
            if (s != VV_ERR_UNSUPPORTED) return s;
        }
        s = vv_nf4_gemm_cuda(x, (const uint8_t*)w->tensor.data,
                             w->quant.scales.data, y, scratch,
                             M, N, K, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(x, w->tensor.data, y, M, N, K,
                              1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    if (w->bias.data)
        s = vv_bias_add_cuda(y, w->bias.data, M, N, stream);
    return s;
}

/**
 * @brief Dump a GPU FP16 buffer as FP32 to $VV_DUMP_DIR (debug builds only).
 */
static void dump_gpu_fp16(const char* name, const void* gpu, size_t n,
                          void* stream) {
    if (!vv_debug_dump_dir() || !gpu || n == 0) return;
    uint16_t* h = (uint16_t*)vv_alloc(n * 2);
    if (!h) return;
    vv_cuda_stream_sync(stream);
    vv_cuda_memcpy_d2h(h, gpu, n * 2, NULL);
    float* f = (float*)vv_alloc(n * sizeof(float));
    if (f) {
        for (size_t i = 0; i < n; i++) f[i] = vv_half_to_float(h[i]);
        vv_debug_dump(name, f, n * sizeof(float));
        vv_free(f);
    }
    vv_free(h);
}

static vv_status_t decoder_layer_impl(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    void* temp_workspace,
    size_t workspace_size,
    void* stream)
{
    if (!layer || !config || !hidden_states || !kv_cache ||
        !temp_workspace) {
        return VV_ERR_NULL_PTR;
    }

    int hs = config->hidden_size;
    int n_heads = config->num_attention_heads;
    int n_kv_heads = config->num_key_value_heads;
    int head_dim = config->head_dim;
    int inter_size = config->intermediate_size;
    vv_status_t s;

    /*
     * Workspace layout:
     * [0]: norm_out    [seq_len * hs] FP16
     * [1]: q           [seq_len * n_heads * head_dim] FP16
     * [2]: k           [seq_len * n_kv_heads * head_dim] FP16
     * [3]: v           [seq_len * n_kv_heads * head_dim] FP16
     * [4]: attn_out    [seq_len * hs] FP16
     * [5]: gate        [seq_len * inter_size] FP16
     * [6]: up          [seq_len * inter_size] FP16
     * [7]: mlp_out     [seq_len * hs] FP16
     * [8]: temp_weight [max(N*K)] FP16 for NF4 dequant
     */
    uint8_t* wp = (uint8_t*)temp_workspace;
    size_t offset = 0;

    void* norm_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* q_buf = wp + offset;
    offset += (size_t)seq_len * n_heads * head_dim * 2;

    void* k_buf = wp + offset;
    offset += (size_t)seq_len * n_kv_heads * head_dim * 2;

    void* v_buf = wp + offset;
    offset += (size_t)seq_len * n_kv_heads * head_dim * 2;

    void* attn_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* gate_buf = wp + offset;
    offset += (size_t)seq_len * inter_size * 2;

    void* up_buf = wp + offset;
    offset += (size_t)seq_len * inter_size * 2;

    void* mlp_out = wp + offset;
    offset += (size_t)seq_len * hs * 2;

    void* temp_weight = wp + offset;

    /* 1. Input LayerNorm */
    s = vv_rmsnorm_cuda(hidden_states, layer->input_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 2. Q, K, V projections (+ bias if present) */
    s = quant_linear(&layer->attn.q_proj, norm_out, q_buf, temp_weight,
                     seq_len, n_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.k_proj, norm_out, k_buf, temp_weight,
                     seq_len, n_kv_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->attn.v_proj, norm_out, v_buf, temp_weight,
                     seq_len, n_kv_heads * head_dim, hs, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1) {
        dump_gpu_fp16("c_l0_norm", norm_out, (size_t)seq_len * hs, stream);
        dump_gpu_fp16("c_l0_q_prerope", q_buf,
                      (size_t)seq_len * n_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_k_prerope", k_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_v", v_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
    }

    /* 3. RoPE */
    s = vv_rope_cuda(q_buf, seq_len, n_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;
    s = vv_rope_cuda(k_buf, seq_len, n_kv_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1) {
        dump_gpu_fp16("c_l0_q", q_buf,
                      (size_t)seq_len * n_heads * head_dim, stream);
        dump_gpu_fp16("c_l0_k", k_buf,
                      (size_t)seq_len * n_kv_heads * head_dim, stream);
    }

    /* 4. KV-cache append */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, stream);
    if (s != VV_OK) return s;

    /* 5. GQA attention over the cache (queries of this chunk see all of it) */
    {
        const void* k_cached;
        const void* v_cached;
        int cache_len;
        vv_kv_cache_get(kv_cache, layer_idx, &k_cached, &v_cached, &cache_len);
        /*
         * current_len only advances on the last layer, so derive the true
         * length from this call's position instead.
         */
        const int actual_cache_len = position_offset + seq_len;

        if (seq_len > 1) {
            s = vv_gqa_attention_prefill_cached_cuda(
                q_buf, k_cached, v_cached, attn_out,
                n_heads, n_kv_heads, head_dim,
                seq_len, position_offset, actual_cache_len, true, stream);
        } else {
            s = vv_gqa_attention_decode_cuda(
                q_buf, k_cached, v_cached, attn_out,
                n_heads, n_kv_heads, head_dim, actual_cache_len, stream);
        }
    }
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1)
        dump_gpu_fp16("c_l0_attn", attn_out, (size_t)seq_len * hs, stream);

    /* 6. O projection + bias + residual */
    s = quant_linear(&layer->attn.o_proj, attn_out, norm_out, temp_weight,
                     seq_len, hs, hs, stream);
    if (s != VV_OK) return s;
    s = vv_residual_add_cuda(hidden_states, norm_out,
                              seq_len * hs, stream);
    if (s != VV_OK) return s;

    if (layer_idx == 0 && seq_len > 1)
        dump_gpu_fp16("c_l0_postattn", hidden_states, (size_t)seq_len * hs, stream);

    /* 7. Post-attention LayerNorm */
    s = vv_rmsnorm_cuda(hidden_states, layer->post_attn_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* 8. MLP: gate + up (+ bias if present) */
    s = quant_linear(&layer->mlp.gate_proj, norm_out, gate_buf, temp_weight,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    s = quant_linear(&layer->mlp.up_proj, norm_out, up_buf, temp_weight,
                     seq_len, inter_size, hs, stream);
    if (s != VV_OK) return s;

    /* 9. SwiGLU */
    s = vv_swiglu_cuda(gate_buf, up_buf, gate_buf,
                        seq_len * inter_size, stream);
    if (s != VV_OK) return s;

    /* 10. Down projection + bias + residual */
    s = quant_linear(&layer->mlp.down_proj, gate_buf, mlp_out, temp_weight,
                     seq_len, hs, inter_size, stream);
    if (s != VV_OK) return s;
    s = vv_residual_add_cuda(hidden_states, mlp_out,
                              seq_len * hs, stream);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU decoder — public API (with layer pool support)
 * ═══════════════════════════════════════════════════════════════════════════ */

vv_status_t vv_decoder_layer_forward(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    void* temp_workspace,
    size_t workspace_size,
    void* stream)
{
    return decoder_layer_impl(layer, config, hidden_states, kv_cache,
                               layer_idx, position_offset, 1,
                               temp_workspace, workspace_size, stream);
}

vv_status_t vv_decoder_step(
    vv_model_t* model,
    void* hidden_state,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;

    int position = kv_cache->current_len;
    bool streaming = pool && !pool->all_resident;

    for (int i = 0; i < model->num_layers; i++) {
        /* Stage layer i if streaming */
        if (streaming) {
            vv_layer_pool_stage(pool, model, i, xfer_stream ? xfer_stream : compute_stream);
            if (xfer_stream) vv_cuda_stream_sync(xfer_stream);
        }

        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            hidden_state, kv_cache, i, position, 1,
            workspace, workspace_size, compute_stream);

        /* Unstage after processing */
        if (streaming) {
            vv_cuda_stream_sync(compute_stream);
            vv_layer_pool_unstage(pool, model, i);
        }

        if (s != VV_OK) {
            VV_LOG_E("decoder: layer %d failed: %s", i, vv_status_str(s));
            return s;
        }
    }

    return VV_OK;
}

vv_status_t vv_decoder_prefill(
    vv_model_t* model,
    void* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    vv_layer_pool_t* pool,
    void* workspace,
    size_t workspace_size,
    void* compute_stream,
    void* xfer_stream)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;

    VV_LOG_I("decoder: prefill %d tokens through %d layers%s",
             seq_len, model->num_layers,
             (pool && !pool->all_resident) ? " (streaming)" : "");

    bool streaming = pool && !pool->all_resident;

    const vv_llm_config_t* cfg = &model->config.llm;
    const int hs = cfg->hidden_size;

    /*
     * Chunked prefill. Activation scratch grows linearly with the number of
     * tokens in flight (the two 18944-wide MLP buffers dominate), so a
     * 30-minute prompt would need gigabytes if run in one shot. Chunking caps
     * that at a fixed budget; correctness is preserved because each chunk
     * attends to the whole KV cache written by the chunks before it.
     */
    size_t per_token = (size_t)(hs * 3
                       + cfg->num_attention_heads * cfg->head_dim
                       + 2 * cfg->num_key_value_heads * cfg->head_dim
                       + 2 * cfg->intermediate_size) * 2;
    size_t weight_scratch = (size_t)cfg->intermediate_size * hs * 2;
    int chunk = seq_len;
    if (workspace_size > weight_scratch + per_token) {
        size_t budget = (workspace_size - weight_scratch) / per_token;
        if (budget < 1) budget = 1;
        if (budget > 2048) budget = 2048;
        if ((int)budget < chunk) chunk = (int)budget;
    }
    if (chunk < 1) chunk = 1;
    if (chunk < seq_len)
        VV_LOG_I("decoder: prefill in %d chunks of %d tokens",
                 (seq_len + chunk - 1) / chunk, chunk);

    for (int start = 0; start < seq_len; start += chunk) {
      const int len = (start + chunk <= seq_len) ? chunk : (seq_len - start);
      void* chunk_hidden = (uint8_t*)hidden_states + (size_t)start * hs * 2;

      for (int i = 0; i < model->num_layers; i++) {

        /* Stage layer i */
        if (streaming) {
            vv_layer_pool_stage(pool, model, i, xfer_stream ? xfer_stream : compute_stream);
            if (xfer_stream) vv_cuda_stream_sync(xfer_stream);
        }

        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            chunk_hidden, kv_cache, i, start, len,
            workspace, workspace_size, compute_stream);

        /* Unstage */
        if (streaming) {
            vv_cuda_stream_sync(compute_stream);
            vv_layer_pool_unstage(pool, model, i);
        }

        if (s != VV_OK) {
            VV_LOG_E("decoder: prefill layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }

        if (start + len == seq_len && vv_debug_dump_dir() && start == 0) {
            size_t n = (size_t)seq_len * (size_t)model->config.llm.hidden_size;
            uint16_t* h = (uint16_t*)vv_alloc(n * 2);
            if (h) {
                vv_cuda_stream_sync(compute_stream);
                vv_cuda_memcpy_d2h(h, hidden_states, n * 2, NULL);
                float* f = (float*)vv_alloc(n * sizeof(float));
                if (f) {
                    for (size_t j = 0; j < n; j++) f[j] = vv_half_to_float(h[j]);
                    char nm[64];
                    snprintf(nm, sizeof(nm), "c_layer%02d", i + 1);
                    vv_debug_dump(nm, f, n * sizeof(float));
                    vv_free(f);
                }
                vv_free(h);
            }
        }
      }
    }

    return VV_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU decoder — per-layer forward (FP32, all on CPU)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** NF4 lookup table (for scale conversion) */
extern const float VV_NF4_TABLE[16];

/**
 * @brief Convert FP16 scale tensor to FP32 float array.
 * Caller must free returned pointer.
 */
static float* fp16_scales_to_fp32(const vv_tensor_t* scales) {
    if (!scales || !scales->data || scales->size_bytes == 0) return NULL;
    int n = (int)(scales->size_bytes / 2); /* FP16 = 2 bytes */
    float* out = (float*)vv_alloc((size_t)n * sizeof(float));
    if (!out) return NULL;
    const uint16_t* src = (const uint16_t*)scales->data;
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = ((uint32_t)h & 0x8000) << 16;
        uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
        uint32_t frac = (uint32_t)h & 0x03FF;
        union { float f; uint32_t u; } u;
        if (expo == 0)
            u.u = sign;
        else if (expo == 0x1F)
            u.u = sign | 0x7F800000 | (frac << 13);
        else
            u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
        out[i] = u.f;
    }
    return out;
}

/**
 * @brief Convert FP16 weight tensor to FP32.
 * Caller must free returned pointer.
 */
static float* fp16_to_fp32(const void* data, int n_elements) {
    if (!data || n_elements <= 0) return NULL;
    float* out = (float*)vv_alloc((size_t)n_elements * sizeof(float));
    if (!out) return NULL;
    const uint16_t* src = (const uint16_t*)data;
    for (int i = 0; i < n_elements; i++) {
        uint16_t h = src[i];
        uint32_t sign = ((uint32_t)h & 0x8000) << 16;
        uint32_t expo = ((uint32_t)h >> 10) & 0x1F;
        uint32_t frac = (uint32_t)h & 0x03FF;
        union { float f; uint32_t u; } u;
        if (expo == 0)
            u.u = sign;
        else if (expo == 0x1F)
            u.u = sign | 0x7F800000 | (frac << 13);
        else
            u.u = sign | ((expo - 15 + 127) << 23) | (frac << 13);
        out[i] = u.f;
    }
    return out;
}

static vv_status_t decoder_layer_cpu(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    float* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,
    float* workspace,
    size_t workspace_size)
{
    int hs = config->hidden_size;
    int n_heads = config->num_attention_heads;
    int n_kv_heads = config->num_key_value_heads;
    int head_dim = config->head_dim;
    int inter_size = config->intermediate_size;
    vv_status_t s;

    /* Workspace layout (FP32): each element 4 bytes */
    float* wp = workspace;
    size_t off = 0;

    float* norm_out = wp + off; off += seq_len * hs;
    float* q_buf    = wp + off; off += seq_len * n_heads * head_dim;
    float* k_buf    = wp + off; off += seq_len * n_kv_heads * head_dim;
    float* v_buf    = wp + off; off += seq_len * n_kv_heads * head_dim;
    float* attn_out = wp + off; off += seq_len * hs;
    float* gate_buf = wp + off; off += seq_len * inter_size;
    float* up_buf   = wp + off; off += seq_len * inter_size;
    float* mlp_out  = wp + off; off += seq_len * hs;
    float* temp_w   = wp + off;
    /* remaining workspace used for temp dequant weight */

    /* Convert FP16 layernorm weights to FP32 */
    float* ln1_w = fp16_to_fp32(layer->input_layernorm.data,
                                 hs);
    float* ln2_w = fp16_to_fp32(layer->post_attn_layernorm.data,
                                 hs);
    if (!ln1_w || !ln2_w) {
        if (ln1_w) vv_free(ln1_w);
        if (ln2_w) vv_free(ln2_w);
        return VV_ERR_OUT_OF_MEMORY;
    }

    /* 1. RMSNorm */
    s = vv_rmsnorm_cpu(hidden_states, ln1_w, norm_out,
                        seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) goto cleanup_norms;

    /* 2. Q, K, V projections (NF4 or FP16→FP32) + bias */
    #define CPU_PROJ(w, out_buf, M, N, K_dim) do {                        \
        if ((w).is_quantized) {                                           \
            float* sc = fp16_scales_to_fp32(&(w).quant.scales);           \
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }    \
            s = vv_nf4_gemm_cpu(norm_out,                                 \
                (const uint8_t*)(w).tensor.data, sc,                      \
                (out_buf), temp_w, (M), (N), (K_dim), 64);               \
            vv_free(sc);                                                  \
        } else {                                                          \
            float* wf = fp16_to_fp32((w).tensor.data, (N) * (K_dim));     \
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }    \
            /* output = input @ weight^T */                               \
            for (int _i = 0; _i < (M); _i++) {                           \
                for (int _j = 0; _j < (N); _j++) {                       \
                    float sum = 0.0f;                                     \
                    for (int _k = 0; _k < (K_dim); _k++)                  \
                        sum += norm_out[_i*(K_dim)+_k] * wf[_j*(K_dim)+_k]; \
                    (out_buf)[_i*(N)+_j] = sum;                           \
                }                                                         \
            }                                                             \
            vv_free(wf);                                                  \
        }                                                                 \
        if (s != VV_OK) goto cleanup_norms;                               \
        /* Add bias if present */                                         \
        if ((w).bias.data) {                                              \
            float* bf = fp16_to_fp32((w).bias.data, (N));                 \
            if (bf) {                                                     \
                for (int _i = 0; _i < (M); _i++)                         \
                    for (int _j = 0; _j < (N); _j++)                     \
                        (out_buf)[_i*(N)+_j] += bf[_j];                   \
                vv_free(bf);                                              \
            }                                                             \
        }                                                                 \
    } while(0)

    CPU_PROJ(layer->attn.q_proj, q_buf, seq_len, n_heads * head_dim, hs);
    CPU_PROJ(layer->attn.k_proj, k_buf, seq_len, n_kv_heads * head_dim, hs);
    CPU_PROJ(layer->attn.v_proj, v_buf, seq_len, n_kv_heads * head_dim, hs);

    /* 3. RoPE */
    s = vv_rope_cpu(q_buf, seq_len, n_heads, head_dim,
                     position_offset, config->rope_theta);
    if (s != VV_OK) goto cleanup_norms;
    s = vv_rope_cpu(k_buf, seq_len, n_kv_heads, head_dim,
                     position_offset, config->rope_theta);
    if (s != VV_OK) goto cleanup_norms;

    /* 4. KV cache append (CPU cache uses memcpy) */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, NULL);
    if (s != VV_OK) goto cleanup_norms;

    /* 5. GQA Attention */
    if (seq_len > 1) {
        s = vv_attention_prefill_cpu(q_buf, k_buf, v_buf, attn_out,
                                      n_heads, n_kv_heads, head_dim,
                                      seq_len, true);
    } else {
        const void* kc; const void* vc; int cl;
        vv_kv_cache_get(kv_cache, layer_idx, &kc, &vc, &cl);
        /* Include the just-appended token */
        int actual_cl = position_offset + seq_len;
        s = vv_attention_decode_cpu(q_buf, (const float*)kc,
                                     (const float*)vc, attn_out,
                                     n_heads, n_kv_heads, head_dim, actual_cl);
    }
    if (s != VV_OK) goto cleanup_norms;

    /* 6. O projection + bias + residual */
    {
        /* Reuse norm_out as temp */
        float* o_out = norm_out; /* overwrite ok, done with input norm */

        if (layer->attn.o_proj.is_quantized) {
            float* sc = fp16_scales_to_fp32(&layer->attn.o_proj.quant.scales);
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            s = vv_nf4_gemm_cpu(attn_out,
                (const uint8_t*)layer->attn.o_proj.tensor.data, sc,
                o_out, temp_w, seq_len, hs, hs, 64);
            vv_free(sc);
        } else {
            float* wf = fp16_to_fp32(layer->attn.o_proj.tensor.data, hs * hs);
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            for (int _i = 0; _i < seq_len; _i++)
                for (int _j = 0; _j < hs; _j++) {
                    float sum = 0.0f;
                    for (int _k = 0; _k < hs; _k++)
                        sum += attn_out[_i*hs+_k] * wf[_j*hs+_k];
                    o_out[_i*hs+_j] = sum;
                }
            vv_free(wf);
        }
        if (s != VV_OK) goto cleanup_norms;
        /* Add bias if present */
        if (layer->attn.o_proj.bias.data) {
            float* bf = fp16_to_fp32(layer->attn.o_proj.bias.data, hs);
            if (bf) {
                for (int _i = 0; _i < seq_len; _i++)
                    for (int _j = 0; _j < hs; _j++)
                        o_out[_i*hs+_j] += bf[_j];
                vv_free(bf);
            }
        }
        vv_residual_add_cpu(hidden_states, o_out, seq_len * hs);
    }

    /* 7. Post-attention LayerNorm */
    s = vv_rmsnorm_cpu(hidden_states, ln2_w, norm_out,
                        seq_len, hs, config->rms_norm_eps);
    if (s != VV_OK) goto cleanup_norms;

    /* 8. MLP: gate + up + SwiGLU + down + residual */
    CPU_PROJ(layer->mlp.gate_proj, gate_buf, seq_len, inter_size, hs);
    CPU_PROJ(layer->mlp.up_proj,   up_buf,   seq_len, inter_size, hs);

    #undef CPU_PROJ

    s = vv_swiglu_cpu(gate_buf, up_buf, gate_buf, seq_len * inter_size);
    if (s != VV_OK) goto cleanup_norms;

    /* Down proj + bias */
    {
        if (layer->mlp.down_proj.is_quantized) {
            float* sc = fp16_scales_to_fp32(&layer->mlp.down_proj.quant.scales);
            if (!sc) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            s = vv_nf4_gemm_cpu(gate_buf,
                (const uint8_t*)layer->mlp.down_proj.tensor.data, sc,
                mlp_out, temp_w, seq_len, hs, inter_size, 64);
            vv_free(sc);
        } else {
            float* wf = fp16_to_fp32(layer->mlp.down_proj.tensor.data,
                                      hs * inter_size);
            if (!wf) { s = VV_ERR_OUT_OF_MEMORY; goto cleanup_norms; }
            for (int _i = 0; _i < seq_len; _i++)
                for (int _j = 0; _j < hs; _j++) {
                    float sum = 0.0f;
                    for (int _k = 0; _k < inter_size; _k++)
                        sum += gate_buf[_i*inter_size+_k] * wf[_j*inter_size+_k];
                    mlp_out[_i*hs+_j] = sum;
                }
            vv_free(wf);
        }
        if (s != VV_OK) goto cleanup_norms;
        /* Add bias if present */
        if (layer->mlp.down_proj.bias.data) {
            float* bf = fp16_to_fp32(layer->mlp.down_proj.bias.data, hs);
            if (bf) {
                for (int _i = 0; _i < seq_len; _i++)
                    for (int _j = 0; _j < hs; _j++)
                        mlp_out[_i*hs+_j] += bf[_j];
                vv_free(bf);
            }
        }
        vv_residual_add_cpu(hidden_states, mlp_out, seq_len * hs);
    }

cleanup_norms:
    vv_free(ln1_w);
    vv_free(ln2_w);
    return s;
}

/* ─── CPU prefill / step ────────────────────────────────────────────────── */

vv_status_t vv_decoder_prefill_cpu(
    vv_model_t* model,
    float* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;

    VV_LOG_I("decoder: CPU prefill %d tokens through %d layers",
             seq_len, model->num_layers);

    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_cpu(
            &model->layers[i], &model->config.llm,
            hidden_states, kv_cache, i, 0, seq_len,
            workspace, workspace_size);
        if (s != VV_OK) {
            VV_LOG_E("decoder: CPU prefill layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}

vv_status_t vv_decoder_step_cpu(
    vv_model_t* model,
    float* hidden_state,
    vv_kv_cache_t* kv_cache,
    float* workspace,
    size_t workspace_size)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;

    int position = kv_cache->current_len;

    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_cpu(
            &model->layers[i], &model->config.llm,
            hidden_state, kv_cache, i, position, 1,
            workspace, workspace_size);
        if (s != VV_OK) {
            VV_LOG_E("decoder: CPU step layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }
    return VV_OK;
}
