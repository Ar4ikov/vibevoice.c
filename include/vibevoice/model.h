/**
 * @file model.h
 * @brief Model loading API: config.json, safetensors weights, model assembly.
 */
#ifndef VV_MODEL_H
#define VV_MODEL_H

#include "vibevoice/types.h"
#include "vibevoice/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Model weight component types ─────────────────────────────────────── */

typedef enum vv_component_type {
    VV_COMP_ACOUSTIC_ENCODER,
    VV_COMP_SEMANTIC_ENCODER,
    VV_COMP_ACOUSTIC_CONNECTOR,
    VV_COMP_SEMANTIC_CONNECTOR,
    VV_COMP_LLM_LAYER,
    VV_COMP_LLM_EMBED,
    VV_COMP_LLM_NORM,
    VV_COMP_LM_HEAD,
} vv_component_type_t;

/** @brief A single loaded weight tensor (may be quantized). */
typedef struct vv_weight {
    char            name[256];
    vv_tensor_t     tensor;       /**< FP16/BF16 tensors */
    vv_quant_tensor_t quant;      /**< NF4 quantized tensors (if applicable) */
    bool            is_quantized; /**< true if NF4 quantized */
} vv_weight_t;

/** @brief Attention layer weights. */
typedef struct vv_attn_weights {
    vv_weight_t q_proj;
    vv_weight_t k_proj;
    vv_weight_t v_proj;
    vv_weight_t o_proj;
} vv_attn_weights_t;

/** @brief MLP layer weights. */
typedef struct vv_mlp_weights {
    vv_weight_t gate_proj;
    vv_weight_t up_proj;
    vv_weight_t down_proj;
} vv_mlp_weights_t;

/** @brief Single transformer layer weights. */
typedef struct vv_layer_weights {
    vv_tensor_t       input_layernorm;
    vv_tensor_t       post_attn_layernorm;
    vv_attn_weights_t attn;
    vv_mlp_weights_t  mlp;
} vv_layer_weights_t;

/** @brief Full model weights. */
typedef struct vv_model {
    vv_model_config_t   config;

    /* Tokenizer encoder weights (FP16) */
    int                  n_acoustic_weights;
    vv_weight_t*         acoustic_weights;
    int                  n_semantic_weights;
    vv_weight_t*         semantic_weights;

    /* Connector weights (FP16) */
    vv_weight_t          acoustic_connector_fc1;
    vv_weight_t          acoustic_connector_norm;
    vv_weight_t          acoustic_connector_fc2;
    vv_weight_t          semantic_connector_fc1;
    vv_weight_t          semantic_connector_norm;
    vv_weight_t          semantic_connector_fc2;

    /* LLM weights */
    vv_tensor_t          embed_tokens;
    vv_tensor_t          final_norm;
    vv_tensor_t          lm_head;
    int                  num_layers;
    vv_layer_weights_t*  layers;

    /* Safetensors handles (kept open for mmap) */
    vv_safetensors_t**   st_files;
    int                  n_st_files;
} vv_model_t;

/**
 * @brief Parse config.json and populate model config.
 */
vv_status_t vv_config_parse(const char* json_path, vv_model_config_t* config);

/**
 * @brief Load the full model from a directory containing safetensors + config.
 */
vv_status_t vv_model_load(const char* model_dir, vv_model_t** out);

/**
 * @brief Free all model weights and resources.
 */
vv_status_t vv_model_free(vv_model_t* model);

#ifdef __cplusplus
}
#endif

#endif /* VV_MODEL_H */
