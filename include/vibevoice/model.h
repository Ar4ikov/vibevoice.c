/**
 * @file model.h
 * @brief Model loading API: config.json, safetensors weights, model assembly.
 */
#ifndef VV_MODEL_H
#define VV_MODEL_H

#include "vibevoice/types.h"
#include "vibevoice/safetensors.h"
#include "vibevoice/quant.h"

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
    vv_tensor_t     tensor;       /**< Dense FP16/BF16, or the packed codes  */
    vv_tensor_t     bias;         /**< Optional bias (FP16), NULL if absent  */
    vv_quant_tensor_t quant;      /**< .scales holds per-group scales        */
    vv_tensor_t     mins;         /**< INT4G only: -zero * scale per group   */
    int             quant_kind;   /**< vv_quant_kind_t                       */
    int             group_size;   /**< INT4G only: weights per scale         */
    bool            is_quantized; /**< quant_kind != VV_QUANT_NONE           */
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
    /**
     * The LM head. With `lm_head_tied` it is the embedding table itself:
     * the same host buffer (and, on the GPU, the same device buffer), which
     * is counted, uploaded and freed once, as `embed_tokens`.
     */
    vv_tensor_t          lm_head;
    bool                 lm_head_tied;
    int                  num_layers;
    vv_layer_weights_t*  layers;

    /* Safetensors handles (kept open for mmap) */
    vv_safetensors_t**   st_files;
    int                  n_st_files;
} vv_model_t;

/* ─── Walking a layer's tensors ─────────────────────────────────────────── */

/**
 * @brief Tensor slots one transformer layer has: 2 norms, then for each of
 *        the 7 projections its weight, scales, mins and bias.
 *
 * Every slot is always listed, present or not, in a fixed order, so a caller
 * can index a saved array by slot position; an absent tensor has NULL data.
 * Whoever adds a per-weight tensor adds it to vv_layer_tensors() and bumps
 * this, and every upload, stage, pin, size and free follows.
 */
#define VV_LAYER_TENSOR_SLOTS (2 + 7 * 4)

/**
 * @brief Every tensor slot of `L`, in a fixed order.
 * @return VV_LAYER_TENSOR_SLOTS
 */
int vv_layer_tensors(vv_layer_weights_t* L,
                     vv_tensor_t* out[VV_LAYER_TENSOR_SLOTS]);

/** @brief The 7 projections of `L`, q k v o gate up down. */
int vv_layer_projections(vv_layer_weights_t* L, vv_weight_t* out[7]);

/** @brief Bytes of every present tensor of `L`. */
size_t vv_layer_bytes(const vv_layer_weights_t* L);

/**
 * @brief Parse config.json and populate model config.
 *
 * LLM dimensions a model cannot run without (hidden size, layers, heads,
 * intermediate size, vocabulary) are required: a config that lacks one is
 * refused rather than silently given the 7B value. Activations other than
 * SiLU, rope scaling and sliding windows are refused as unsupported.
 */
vv_status_t vv_config_parse(const char* json_path, vv_model_config_t* config);

/**
 * @brief Read preprocessor_config.json into `config->audio`.
 *
 * A missing file is not an error: the reference processor's defaults apply
 * (24 kHz, 3200 samples per frame, normalize to -25 dBFS). Re-derives
 * `config->family`, since chunk geometry is what marks a streaming model.
 */
vv_status_t vv_config_parse_preprocessor(const char* json_path,
                                         vv_model_config_t* config);

/**
 * @brief config.json plus preprocessor_config.json from a model directory,
 *        and the family they describe.
 */
vv_status_t vv_config_load(const char* model_dir, vv_model_config_t* config);

/** @brief "asr-7b" | "asr-bitnet" | "asr-streaming-7b". */
const char* vv_model_family_name(vv_model_family_t f);

/** @brief Options for vv_model_load_ex(). */
typedef struct vv_model_load_opts {
    int quant;   /**< vv_load_quant_t; VV_LOAD_QUANT_AUTO keeps the format */
} vv_model_load_opts_t;

static inline vv_model_load_opts_t vv_model_load_opts_default(void) {
    vv_model_load_opts_t o;
    o.quant = 0;
    return o;
}

/**
 * @brief Load the full model from a directory containing safetensors + config.
 */
vv_status_t vv_model_load(const char* model_dir, vv_model_t** out);

/**
 * @brief Load with options; NULL opts is vv_model_load().
 *
 * Projections are routed by what the file holds, not by their name: U8 with
 * an `.absmax` companion is NF4, I32 `qweight` is AWQ/GPTQ, and F16, BF16 or
 * F32 is dense — kept as FP16 or quantized while it is read, per
 * `opts->quant`. Every tensor the model needs is checked for presence and
 * shape, and the first one that is wrong fails the load with its name.
 */
vv_status_t vv_model_load_ex(const char* model_dir,
                             const vv_model_load_opts_t* opts,
                             vv_model_t** out);

/**
 * @brief Free all model weights and resources.
 */
vv_status_t vv_model_free(vv_model_t* model);

#ifdef __cplusplus
}
#endif

#endif /* VV_MODEL_H */
