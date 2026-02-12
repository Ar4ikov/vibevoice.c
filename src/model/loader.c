/**
 * @file loader.c
 * @brief Orchestrates loading all model components from safetensors + config.
 */

#include "vibevoice/model.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <string.h>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/**
 * @brief Check if a tensor name is the main NF4 packed weight (not its metadata).
 *
 * NF4 quantized projections end with _proj.weight and are stored as U8.
 * Their companion tensors (.absmax, .quant_state, .quant_map, .nested_*)
 * are metadata loaded separately.
 */
static bool is_nf4_weight(const char* name) {
    if (strstr(name, ".self_attn.") && strstr(name, "_proj.weight") &&
        !strstr(name, ".absmax") && !strstr(name, ".quant_state") &&
        !strstr(name, ".quant_map") && !strstr(name, ".nested_"))
        return true;
    if (strstr(name, ".mlp.") && strstr(name, "_proj.weight") &&
        !strstr(name, ".absmax") && !strstr(name, ".quant_state") &&
        !strstr(name, ".quant_map") && !strstr(name, ".nested_"))
        return true;
    return false;
}

/**
 * @brief Check if a tensor is NF4 metadata (scale, quant_state, quant_map).
 * These are loaded alongside their parent weight, not independently.
 */
static bool is_nf4_metadata(const char* name) {
    return strstr(name, ".absmax") != NULL ||
           strstr(name, ".quant_state") != NULL ||
           strstr(name, ".quant_map") != NULL ||
           strstr(name, ".nested_") != NULL ||
           strstr(name, "SCB") != NULL;
}

/**
 * @brief Build full path: dir/filename
 */
static void build_path(char* buf, size_t buf_size,
                        const char* dir, const char* filename) {
    size_t dlen = strlen(dir);
    if (dlen > 0 && (dir[dlen-1] == '/' || dir[dlen-1] == '\\')) {
        snprintf(buf, buf_size, "%s%s", dir, filename);
    } else {
        snprintf(buf, buf_size, "%s/%s", dir, filename);
    }
}

/**
 * @brief Parse model.safetensors.index.json to find safetensors file list.
 */
static vv_status_t find_safetensors_files(const char* model_dir,
                                           char*** out_files, int* out_count) {
    char path[512];
    build_path(path, sizeof(path), model_dir, "model.safetensors.index.json");

    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Try single-file model */
        build_path(path, sizeof(path), model_dir, "model.safetensors");
        f = fopen(path, "rb");
        if (f) {
            fclose(f);
            *out_files = (char**)vv_alloc(sizeof(char*));
            (*out_files)[0] = (char*)vv_alloc(strlen("model.safetensors") + 1);
            strcpy((*out_files)[0], "model.safetensors");
            *out_count = 1;
            return VV_OK;
        }
        VV_LOG_E("loader: no safetensors files found in '%s'", model_dir);
        return VV_ERR_NOT_FOUND;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)vv_alloc((size_t)size + 1);
    if (!json) { fclose(f); return VV_ERR_OUT_OF_MEMORY; }
    fread(json, 1, (size_t)size, f);
    fclose(f);
    json[size] = '\0';

    cJSON* root = cJSON_Parse(json);
    vv_free(json);
    if (!root) return VV_ERR_PARSE;

    /* Collect unique filenames from weight_map values */
    cJSON* weight_map = cJSON_GetObjectItem(root, "weight_map");
    if (!weight_map) {
        cJSON_Delete(root);
        return VV_ERR_PARSE;
    }

    /* First pass: count unique filenames */
    char unique_files[16][256];
    int n_unique = 0;

    cJSON* entry;
    cJSON_ArrayForEach(entry, weight_map) {
        if (!cJSON_IsString(entry)) continue;
        const char* fname = entry->valuestring;
        bool found = false;
        for (int i = 0; i < n_unique; i++) {
            if (strcmp(unique_files[i], fname) == 0) {
                found = true;
                break;
            }
        }
        if (!found && n_unique < 16) {
            strncpy(unique_files[n_unique], fname, 255);
            unique_files[n_unique][255] = '\0';
            n_unique++;
        }
    }

    cJSON_Delete(root);

    *out_files = (char**)vv_alloc((size_t)n_unique * sizeof(char*));
    for (int i = 0; i < n_unique; i++) {
        (*out_files)[i] = (char*)vv_alloc(strlen(unique_files[i]) + 1);
        strcpy((*out_files)[i], unique_files[i]);
    }
    *out_count = n_unique;

    VV_LOG_I("loader: found %d safetensors files", n_unique);
    return VV_OK;
}

/* ─── Load a single weight tensor ───────────────────────────────────────── */

static vv_status_t load_tensor_from_st(vv_safetensors_t* st,
                                        const char* name,
                                        vv_tensor_t* tensor) {
    vv_st_tensor_info_t info;
    vv_status_t s = vv_safetensors_find(st, name, &info);
    if (s != VV_OK) return s;

    const void* data;
    s = vv_safetensors_get_data(st, &info, &data);
    if (s != VV_OK) return s;

    /* Copy data to our managed memory */
    tensor->data = vv_alloc(info.data_size);
    if (!tensor->data) return VV_ERR_OUT_OF_MEMORY;
    memcpy(tensor->data, data, info.data_size);

    tensor->dtype = info.dtype;
    tensor->ndim = info.ndim;
    memcpy(tensor->shape, info.shape, sizeof(info.shape));
    tensor->size_bytes = info.data_size;
    tensor->on_gpu = false;

    return VV_OK;
}

/**
 * @brief Search all safetensors files for a tensor by name and load it.
 */
static vv_status_t load_tensor_any(vv_safetensors_t** st_files, int n_st,
                                    const char* name, vv_tensor_t* tensor) {
    for (int i = 0; i < n_st; i++) {
        vv_status_t s = load_tensor_from_st(st_files[i], name, tensor);
        if (s == VV_OK) return VV_OK;
    }
    return VV_ERR_NOT_FOUND;
}

/**
 * @brief Load NF4 absmax scales for a quantized weight.
 *
 * For a weight named "foo.weight", loads "foo.weight.absmax" as the
 * per-block FP16 scales needed for dequantization.
 */
static vv_status_t load_nf4_scales(vv_safetensors_t** st_files, int n_st,
                                    const char* weight_name,
                                    vv_weight_t* weight) {
    char absmax_name[512];
    snprintf(absmax_name, sizeof(absmax_name), "%s.absmax", weight_name);

    vv_status_t s = load_tensor_any(st_files, n_st, absmax_name,
                                     &weight->quant.scales);
    if (s == VV_OK) {
        weight->quant.block_size = 64;  /* bitsandbytes default */
        VV_LOG_D("loader: loaded NF4 scales for '%s'", weight_name);
    }
    return s;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

vv_status_t vv_model_load(const char* model_dir, vv_model_t** out) {
    if (!model_dir || !out) return VV_ERR_NULL_PTR;

    VV_LOG_I("loader: loading model from '%s'", model_dir);

    /* Allocate model struct */
    vv_model_t* model = (vv_model_t*)vv_alloc(sizeof(vv_model_t));
    if (!model) return VV_ERR_OUT_OF_MEMORY;
    memset(model, 0, sizeof(*model));

    /* Parse config.json */
    char config_path[512];
    build_path(config_path, sizeof(config_path), model_dir, "config.json");
    vv_status_t s = vv_config_parse(config_path, &model->config);
    if (s != VV_OK) {
        VV_LOG_E("loader: failed to parse config.json");
        vv_free(model);
        return s;
    }

    /* Find and open safetensors files */
    char** st_names = NULL;
    int n_st = 0;
    s = find_safetensors_files(model_dir, &st_names, &n_st);
    if (s != VV_OK) {
        vv_free(model);
        return s;
    }

    model->st_files = (vv_safetensors_t**)vv_alloc(
        (size_t)n_st * sizeof(vv_safetensors_t*));
    model->n_st_files = n_st;

    for (int i = 0; i < n_st; i++) {
        char full_path[512];
        build_path(full_path, sizeof(full_path), model_dir, st_names[i]);
        s = vv_safetensors_open(full_path, &model->st_files[i]);
        if (s != VV_OK) {
            VV_LOG_E("loader: failed to open '%s'", full_path);
            /* Clean up opened files */
            for (int j = 0; j < i; j++) {
                vv_safetensors_close(model->st_files[j]);
            }
            for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
            vv_free(st_names);
            vv_free(model->st_files);
            vv_free(model);
            return s;
        }
    }

    for (int j = 0; j < n_st; j++) vv_free(st_names[j]);
    vv_free(st_names);

    /* Allocate LLM layer weights */
    int n_layers = model->config.llm.num_hidden_layers;
    model->num_layers = n_layers;
    model->layers = (vv_layer_weights_t*)vv_alloc(
        (size_t)n_layers * sizeof(vv_layer_weights_t));
    if (!model->layers) {
        vv_model_free(model);
        return VV_ERR_OUT_OF_MEMORY;
    }
    memset(model->layers, 0, (size_t)n_layers * sizeof(vv_layer_weights_t));

    /*
     * Count tokenizer encoder tensors for pre-allocation.
     */
    int n_acoustic_enc = 0, n_semantic_enc = 0;
    for (int fi = 0; fi < n_st; fi++) {
        int nt = vv_safetensors_num_tensors(model->st_files[fi]);
        for (int ti = 0; ti < nt; ti++) {
            vv_st_tensor_info_t info;
            vv_safetensors_get_info(model->st_files[fi], ti, &info);
            if (strstr(info.name, "acoustic_tokenizer.encoder."))
                n_acoustic_enc++;
            if (strstr(info.name, "semantic_tokenizer.encoder."))
                n_semantic_enc++;
        }
    }

    if (n_acoustic_enc > 0) {
        model->acoustic_weights = (vv_weight_t*)vv_alloc(
            (size_t)n_acoustic_enc * sizeof(vv_weight_t));
        if (model->acoustic_weights)
            memset(model->acoustic_weights, 0,
                   (size_t)n_acoustic_enc * sizeof(vv_weight_t));
    }
    if (n_semantic_enc > 0) {
        model->semantic_weights = (vv_weight_t*)vv_alloc(
            (size_t)n_semantic_enc * sizeof(vv_weight_t));
        if (model->semantic_weights)
            memset(model->semantic_weights, 0,
                   (size_t)n_semantic_enc * sizeof(vv_weight_t));
    }
    int ai = 0, si = 0;  /* Indices for encoder weight arrays */

    /*
     * Load tensors from safetensors.
     * Iterate all tensors in all files and route to the right destination.
     */
    for (int fi = 0; fi < n_st; fi++) {
        vv_safetensors_t* st = model->st_files[fi];
        int nt = vv_safetensors_num_tensors(st);

        for (int ti = 0; ti < nt; ti++) {
            vv_st_tensor_info_t info;
            vv_safetensors_get_info(st, ti, &info);

            /* Skip NF4 metadata — loaded alongside their parent weight */
            if (is_nf4_metadata(info.name)) continue;

            /* ── LLM global weights ── */
            if (strstr(info.name, "embed_tokens.weight")) {
                load_tensor_from_st(st, info.name, &model->embed_tokens);
            }
            else if (strstr(info.name, "language_model.norm.weight") ||
                     (strcmp(info.name, "model.norm.weight") == 0)) {
                load_tensor_from_st(st, info.name, &model->final_norm);
            }
            else if (strcmp(info.name, "lm_head.weight") == 0) {
                load_tensor_from_st(st, info.name, &model->lm_head);
            }

            /* ── Connector weights (FP16) ── */
            else if (strstr(info.name, "acoustic_connector.fc1.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc1.tensor);
                strncpy(model->acoustic_connector_fc1.name, info.name, 255);
            }
            else if (strstr(info.name, "acoustic_connector.fc1.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc1.quant.packed);
                /* Reuse quant.packed as bias storage for non-quantized */
            }
            else if (strstr(info.name, "acoustic_connector.norm.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_norm.tensor);
            }
            else if (strstr(info.name, "acoustic_connector.fc2.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc2.tensor);
            }
            else if (strstr(info.name, "acoustic_connector.fc2.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->acoustic_connector_fc2.quant.packed);
            }
            else if (strstr(info.name, "semantic_connector.fc1.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc1.tensor);
                strncpy(model->semantic_connector_fc1.name, info.name, 255);
            }
            else if (strstr(info.name, "semantic_connector.fc1.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc1.quant.packed);
            }
            else if (strstr(info.name, "semantic_connector.norm.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_norm.tensor);
            }
            else if (strstr(info.name, "semantic_connector.fc2.weight")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc2.tensor);
            }
            else if (strstr(info.name, "semantic_connector.fc2.bias")) {
                load_tensor_from_st(st, info.name,
                    &model->semantic_connector_fc2.quant.packed);
            }

            /* ── Tokenizer encoder weights (FP16) ── */
            else if (strstr(info.name, "acoustic_tokenizer.encoder.") &&
                     model->acoustic_weights && ai < n_acoustic_enc) {
                load_tensor_from_st(st, info.name,
                                    &model->acoustic_weights[ai].tensor);
                strncpy(model->acoustic_weights[ai].name, info.name, 255);
                ai++;
            }
            else if (strstr(info.name, "semantic_tokenizer.encoder.") &&
                     model->semantic_weights && si < n_semantic_enc) {
                load_tensor_from_st(st, info.name,
                                    &model->semantic_weights[si].tensor);
                strncpy(model->semantic_weights[si].name, info.name, 255);
                si++;
            }

            /* ── LLM layer weights ── */
            else if (strstr(info.name, ".layers.")) {
                /* Parse layer index — works for both model.layers.N
                 * and model.language_model.layers.N */
                const char* lp = strstr(info.name, ".layers.");
                if (!lp) continue;
                lp += strlen(".layers.");
                int layer_idx = atoi(lp);
                if (layer_idx < 0 || layer_idx >= n_layers) continue;

                vv_layer_weights_t* layer = &model->layers[layer_idx];

                if (strstr(info.name, "input_layernorm.weight")) {
                    load_tensor_from_st(st, info.name,
                                        &layer->input_layernorm);
                }
                else if (strstr(info.name, "post_attention_layernorm.weight")) {
                    load_tensor_from_st(st, info.name,
                                        &layer->post_attn_layernorm);
                }
                else if (is_nf4_weight(info.name)) {
                    /* Determine which projection this is */
                    vv_weight_t* w = NULL;
                    if (strstr(info.name, "self_attn.q_proj.weight"))
                        w = &layer->attn.q_proj;
                    else if (strstr(info.name, "self_attn.k_proj.weight"))
                        w = &layer->attn.k_proj;
                    else if (strstr(info.name, "self_attn.v_proj.weight"))
                        w = &layer->attn.v_proj;
                    else if (strstr(info.name, "self_attn.o_proj.weight"))
                        w = &layer->attn.o_proj;
                    else if (strstr(info.name, "mlp.gate_proj.weight"))
                        w = &layer->mlp.gate_proj;
                    else if (strstr(info.name, "mlp.up_proj.weight"))
                        w = &layer->mlp.up_proj;
                    else if (strstr(info.name, "mlp.down_proj.weight"))
                        w = &layer->mlp.down_proj;

                    if (w) {
                        w->is_quantized = true;
                        load_tensor_from_st(st, info.name, &w->tensor);
                        strncpy(w->name, info.name, 255);
                        /* Load NF4 absmax scales */
                        load_nf4_scales(model->st_files, n_st,
                                        info.name, w);
                    }
                }
            }
        }
    }

    model->n_acoustic_weights = ai;
    model->n_semantic_weights = si;

    VV_LOG_I("loader: model loaded successfully (%d layers, embed=%s, norm=%s, "
             "connectors=%s, encoders=%d+%d)",
             n_layers,
             model->embed_tokens.data ? "yes" : "no",
             model->final_norm.data ? "yes" : "no",
             model->acoustic_connector_fc1.tensor.data ? "yes" : "no",
             ai, si);

    *out = model;
    return VV_OK;
}

vv_status_t vv_model_free(vv_model_t* model) {
    if (!model) return VV_ERR_NULL_PTR;

    /* Free layer weights (including NF4 scales) */
    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            vv_layer_weights_t* l = &model->layers[i];
            vv_tensor_free(&l->input_layernorm);
            vv_tensor_free(&l->post_attn_layernorm);
            vv_tensor_free(&l->attn.q_proj.tensor);
            vv_tensor_free(&l->attn.q_proj.quant.scales);
            vv_tensor_free(&l->attn.k_proj.tensor);
            vv_tensor_free(&l->attn.k_proj.quant.scales);
            vv_tensor_free(&l->attn.v_proj.tensor);
            vv_tensor_free(&l->attn.v_proj.quant.scales);
            vv_tensor_free(&l->attn.o_proj.tensor);
            vv_tensor_free(&l->attn.o_proj.quant.scales);
            vv_tensor_free(&l->mlp.gate_proj.tensor);
            vv_tensor_free(&l->mlp.gate_proj.quant.scales);
            vv_tensor_free(&l->mlp.up_proj.tensor);
            vv_tensor_free(&l->mlp.up_proj.quant.scales);
            vv_tensor_free(&l->mlp.down_proj.tensor);
            vv_tensor_free(&l->mlp.down_proj.quant.scales);
        }
        vv_free(model->layers);
    }

    /* Free LLM global weights */
    vv_tensor_free(&model->embed_tokens);
    vv_tensor_free(&model->final_norm);
    vv_tensor_free(&model->lm_head);

    /* Free connector weights */
    vv_tensor_free(&model->acoustic_connector_fc1.tensor);
    vv_tensor_free(&model->acoustic_connector_fc1.quant.packed); /* bias */
    vv_tensor_free(&model->acoustic_connector_norm.tensor);
    vv_tensor_free(&model->acoustic_connector_fc2.tensor);
    vv_tensor_free(&model->acoustic_connector_fc2.quant.packed); /* bias */
    vv_tensor_free(&model->semantic_connector_fc1.tensor);
    vv_tensor_free(&model->semantic_connector_fc1.quant.packed); /* bias */
    vv_tensor_free(&model->semantic_connector_norm.tensor);
    vv_tensor_free(&model->semantic_connector_fc2.tensor);
    vv_tensor_free(&model->semantic_connector_fc2.quant.packed); /* bias */

    /* Free tokenizer encoder weights */
    if (model->acoustic_weights) {
        for (int i = 0; i < model->n_acoustic_weights; i++)
            vv_tensor_free(&model->acoustic_weights[i].tensor);
        vv_free(model->acoustic_weights);
    }
    if (model->semantic_weights) {
        for (int i = 0; i < model->n_semantic_weights; i++)
            vv_tensor_free(&model->semantic_weights[i].tensor);
        vv_free(model->semantic_weights);
    }

    /* Close safetensors files */
    if (model->st_files) {
        for (int i = 0; i < model->n_st_files; i++) {
            if (model->st_files[i]) {
                vv_safetensors_close(model->st_files[i]);
            }
        }
        vv_free(model->st_files);
    }

    vv_free(model);
    return VV_OK;
}
