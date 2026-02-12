/**
 * @file decoder.c
 * @brief Qwen2 transformer decoder: per-layer forward and autoregressive loop.
 *
 * Per layer (28 total):
 * 1. RMSNorm → Q, K, V (NF4 dequant + GEMM)
 * 2. RoPE → GQA Attention → O proj → Residual
 * 3. RMSNorm → SwiGLU MLP (NF4 dequant + GEMM) → Residual
 *
 * Final: model.norm (RMSNorm) → lm_head (FP16 Linear, 3584→152064)
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"

#include <string.h>

/* Forward declarations for CUDA kernel launchers */
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

extern vv_status_t vv_gqa_attention_prefill_cuda(
    const void* q, const void* k, const void* v,
    void* output,
    int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, bool causal, void* stream);

extern vv_status_t vv_swiglu_cuda(
    const void* gate, const void* up, void* output,
    int n_elements, void* stream);

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

extern vv_status_t vv_residual_add_cuda(void* x, const void* y, int total,
                                          void* stream);

/* ─── Internal per-layer forward (handles both prefill and decode) ──────── */

static vv_status_t decoder_layer_impl(
    const vv_layer_weights_t* layer,
    const vv_llm_config_t* config,
    void* hidden_states,
    vv_kv_cache_t* kv_cache,
    int layer_idx,
    int position_offset,
    int seq_len,                    /* 1 for decode, >1 for prefill */
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

    /* Temp buffer for dequantized weights - use remaining workspace */
    void* temp_weight = wp + offset;

    /* ── 1. Input LayerNorm ── */
    s = vv_rmsnorm_cuda(hidden_states,
                         layer->input_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* ── 2. Q, K, V projections (NF4 GEMM) ── */
    if (layer->attn.q_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(norm_out,
                              (const uint8_t*)layer->attn.q_proj.tensor.data,
                              layer->attn.q_proj.quant.scales.data,
                              q_buf, temp_weight,
                              seq_len, n_heads * head_dim, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(norm_out, layer->attn.q_proj.tensor.data,
                               q_buf, seq_len, n_heads * head_dim, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    if (layer->attn.k_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(norm_out,
                              (const uint8_t*)layer->attn.k_proj.tensor.data,
                              layer->attn.k_proj.quant.scales.data,
                              k_buf, temp_weight,
                              seq_len, n_kv_heads * head_dim, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(norm_out, layer->attn.k_proj.tensor.data,
                               k_buf, seq_len, n_kv_heads * head_dim, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    if (layer->attn.v_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(norm_out,
                              (const uint8_t*)layer->attn.v_proj.tensor.data,
                              layer->attn.v_proj.quant.scales.data,
                              v_buf, temp_weight,
                              seq_len, n_kv_heads * head_dim, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(norm_out, layer->attn.v_proj.tensor.data,
                               v_buf, seq_len, n_kv_heads * head_dim, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    /* ── 3. RoPE on Q and K ── */
    s = vv_rope_cuda(q_buf, seq_len, n_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;

    s = vv_rope_cuda(k_buf, seq_len, n_kv_heads, head_dim,
                      position_offset, config->rope_theta, stream);
    if (s != VV_OK) return s;

    /* ── 4. KV-cache append ── */
    s = vv_kv_cache_append(kv_cache, layer_idx, k_buf, v_buf,
                            seq_len, stream);
    if (s != VV_OK) return s;

    /* ── 5. GQA Attention ── */
    if (seq_len > 1) {
        /* Prefill: use full-sequence attention (causal mask) */
        s = vv_gqa_attention_prefill_cuda(
            q_buf, k_buf, v_buf, attn_out,
            n_heads, n_kv_heads, head_dim,
            seq_len, true, stream);
    } else {
        /* Decode: use cached K/V attention */
        const void* k_cached;
        const void* v_cached;
        int cache_len;
        vv_kv_cache_get(kv_cache, layer_idx,
                         &k_cached, &v_cached, &cache_len);
        s = vv_gqa_attention_decode_cuda(
            q_buf, k_cached, v_cached, attn_out,
            n_heads, n_kv_heads, head_dim, cache_len, stream);
    }
    if (s != VV_OK) return s;

    /* ── 6. O projection + residual ── */
    if (layer->attn.o_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(attn_out,
                              (const uint8_t*)layer->attn.o_proj.tensor.data,
                              layer->attn.o_proj.quant.scales.data,
                              norm_out, temp_weight,
                              seq_len, hs, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(attn_out, layer->attn.o_proj.tensor.data,
                               norm_out, seq_len, hs, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    /* Residual add: hidden_states += o_proj output (stored in norm_out) */
    s = vv_residual_add_cuda(hidden_states, norm_out,
                              seq_len * hs, stream);
    if (s != VV_OK) return s;

    /* ── 7. Post-attention LayerNorm ── */
    s = vv_rmsnorm_cuda(hidden_states,
                         layer->post_attn_layernorm.data,
                         norm_out, seq_len, hs,
                         config->rms_norm_eps, stream);
    if (s != VV_OK) return s;

    /* ── 8. MLP: gate, up projections ── */
    if (layer->mlp.gate_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(norm_out,
                              (const uint8_t*)layer->mlp.gate_proj.tensor.data,
                              layer->mlp.gate_proj.quant.scales.data,
                              gate_buf, temp_weight,
                              seq_len, inter_size, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(norm_out, layer->mlp.gate_proj.tensor.data,
                               gate_buf, seq_len, inter_size, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    if (layer->mlp.up_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(norm_out,
                              (const uint8_t*)layer->mlp.up_proj.tensor.data,
                              layer->mlp.up_proj.quant.scales.data,
                              up_buf, temp_weight,
                              seq_len, inter_size, hs, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(norm_out, layer->mlp.up_proj.tensor.data,
                               up_buf, seq_len, inter_size, hs,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    /* ── 9. SwiGLU activation ── */
    s = vv_swiglu_cuda(gate_buf, up_buf, gate_buf,
                        seq_len * inter_size, stream);
    if (s != VV_OK) return s;

    /* ── 10. Down projection + residual ── */
    if (layer->mlp.down_proj.is_quantized) {
        s = vv_nf4_gemm_cuda(gate_buf,
                              (const uint8_t*)layer->mlp.down_proj.tensor.data,
                              layer->mlp.down_proj.quant.scales.data,
                              mlp_out, temp_weight,
                              seq_len, hs, inter_size, 64, stream);
    } else {
        s = vv_gemm_fp16_cuda(gate_buf, layer->mlp.down_proj.tensor.data,
                               mlp_out, seq_len, hs, inter_size,
                               1.0f, 0.0f, stream);
    }
    if (s != VV_OK) return s;

    /* Residual add: hidden_states += mlp_out */
    s = vv_residual_add_cuda(hidden_states, mlp_out,
                              seq_len * hs, stream);
    if (s != VV_OK) return s;

    return VV_OK;
}

/* ─── Public per-layer forward (decode: single token) ──────────────────── */

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

/* ─── Full decode step (all layers) ─────────────────────────────────────── */

vv_status_t vv_decoder_step(
    const vv_model_t* model,
    void* hidden_state,
    vv_kv_cache_t* kv_cache,
    void* workspace,
    size_t workspace_size,
    void* stream)
{
    if (!model || !hidden_state || !kv_cache) return VV_ERR_NULL_PTR;

    int position = kv_cache->current_len;

    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            hidden_state, kv_cache, i, position, 1,
            workspace, workspace_size, stream);
        if (s != VV_OK) {
            VV_LOG_E("decoder: layer %d failed: %s", i, vv_status_str(s));
            return s;
        }
    }

    return VV_OK;
}

/* ─── Full prefill ──────────────────────────────────────────────────────── */

vv_status_t vv_decoder_prefill(
    const vv_model_t* model,
    void* hidden_states,
    int seq_len,
    vv_kv_cache_t* kv_cache,
    void* workspace,
    size_t workspace_size,
    void* stream)
{
    if (!model || !hidden_states || !kv_cache) return VV_ERR_NULL_PTR;

    VV_LOG_I("decoder: prefill %d tokens through %d layers",
             seq_len, model->num_layers);

    /* Prefill processes the full sequence through each layer,
     * using the prefill attention variant with causal mask. */
    for (int i = 0; i < model->num_layers; i++) {
        vv_status_t s = decoder_layer_impl(
            &model->layers[i], &model->config.llm,
            hidden_states, kv_cache, i, 0, seq_len,
            workspace, workspace_size, stream);
        if (s != VV_OK) {
            VV_LOG_E("decoder: prefill layer %d failed: %s",
                     i, vv_status_str(s));
            return s;
        }
    }

    return VV_OK;
}
