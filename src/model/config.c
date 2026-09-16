/**
 * @file config.c
 * @brief Parse VibeVoice-ASR config.json and populate vv_model_config_t.
 */

#include "vibevoice/model.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/vibevoice.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static char* read_file_to_string(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return NULL;
    }

    char* buf = (char*)vv_alloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[read] = '\0';
    return buf;
}

static int parse_int(cJSON* obj, const char* key, int def) {
    cJSON* v = cJSON_GetObjectItem(obj, key);
    return (v && cJSON_IsNumber(v)) ? (int)v->valuedouble : def;
}

static float parse_float(cJSON* obj, const char* key, float def) {
    cJSON* v = cJSON_GetObjectItem(obj, key);
    return (v && cJSON_IsNumber(v)) ? (float)v->valuedouble : def;
}

static bool parse_bool(cJSON* obj, const char* key, bool def) {
    cJSON* v = cJSON_GetObjectItem(obj, key);
    if (!v) return def;
    if (cJSON_IsBool(v)) return cJSON_IsTrue(v);
    return def;
}

static void parse_int_array(cJSON* obj, const char* key,
                             int* arr, int* n, int max_n) {
    *n = 0;
    cJSON* v = cJSON_GetObjectItem(obj, key);
    if (!v || !cJSON_IsArray(v)) return;
    int count = cJSON_GetArraySize(v);
    if (count > max_n) count = max_n;
    for (int i = 0; i < count; i++) {
        cJSON* item = cJSON_GetArrayItem(v, i);
        if (item && cJSON_IsNumber(item)) {
            arr[i] = (int)item->valuedouble;
        }
    }
    *n = count;
}

/** Parse depths string like "3-3-3-3-3-3-8" into int array. */
static void parse_depths_string(cJSON* obj, const char* key,
                                 int* arr, int* n, int max_n) {
    *n = 0;
    cJSON* v = cJSON_GetObjectItem(obj, key);
    if (!v) return;

    /* Could be an array or a dash-separated string */
    if (cJSON_IsArray(v)) {
        parse_int_array(obj, key, arr, n, max_n);
        return;
    }

    if (!cJSON_IsString(v) || !v->valuestring) return;

    const char* s = v->valuestring;
    int count = 0;
    while (*s && count < max_n) {
        arr[count] = (int)strtol(s, NULL, 10);
        count++;
        while (*s && *s != '-') s++;
        if (*s == '-') s++;
    }
    *n = count;
}

/* ─── Tokenizer config parsing ──────────────────────────────────────────── */

static void parse_acoustic_tokenizer_config(
    cJSON* obj, vv_acoustic_tokenizer_config_t* c) {
    c->channels          = parse_int(obj, "channels", 1);
    c->causal            = parse_bool(obj, "causal", true);
    c->vae_dim           = parse_int(obj, "vae_dim", 64);
    c->fix_std           = parse_float(obj, "fix_std", 0.5f);
    {
        cJSON* d = cJSON_GetObjectItem(obj, "std_dist_type");
        const char* name = (d && cJSON_IsString(d)) ? d->valuestring : NULL;
        vv_acoustic_sampling_t m = vv_acoustic_sampling_parse(name);
        if (m >= VV_ACOUSTIC_SAMPLING_COUNT) {
            VV_LOG_W("config: unknown std_dist_type '%s', treating as the mode",
                     name ? name : "");
            m = VV_ACOUSTIC_MODE;
        }
        c->std_dist_type = m;
    }
    c->encoder_n_filters = parse_int(obj, "encoder_n_filters", 32);
    c->layernorm_eps     = parse_float(obj, "layernorm_eps", 1e-5f);
    c->layer_scale_init_value = parse_float(obj, "layer_scale_init_value",
                                             1e-6f);
    parse_int_array(obj, "encoder_ratios", c->encoder_ratios,
                    &c->n_ratios, 8);
    parse_depths_string(obj, "encoder_depths", c->encoder_depths,
                        &c->n_depths, 8);
}

static void parse_semantic_tokenizer_config(
    cJSON* obj, vv_semantic_tokenizer_config_t* c) {
    c->channels          = parse_int(obj, "channels", 1);
    c->causal            = parse_bool(obj, "causal", true);
    c->vae_dim           = parse_int(obj, "vae_dim", 128);
    c->encoder_n_filters = parse_int(obj, "encoder_n_filters", 32);
    c->layernorm_eps     = parse_float(obj, "layernorm_eps", 1e-5f);
    c->layer_scale_init_value = parse_float(obj, "layer_scale_init_value",
                                             1e-6f);
    parse_int_array(obj, "encoder_ratios", c->encoder_ratios,
                    &c->n_ratios, 8);
    parse_depths_string(obj, "encoder_depths", c->encoder_depths,
                        &c->n_depths, 8);
}

/* ─── LLM config parsing ───────────────────────────────────────────────── */

static void parse_decoder_config(cJSON* obj, vv_llm_config_t* c) {
    c->hidden_size             = parse_int(obj, "hidden_size", 3584);
    c->num_hidden_layers       = parse_int(obj, "num_hidden_layers", 28);
    c->num_attention_heads     = parse_int(obj, "num_attention_heads", 28);
    c->num_key_value_heads     = parse_int(obj, "num_key_value_heads", 4);
    c->intermediate_size       = parse_int(obj, "intermediate_size", 18944);
    c->vocab_size              = parse_int(obj, "vocab_size", 152064);
    c->max_position_embeddings = parse_int(obj, "max_position_embeddings",
                                            131072);
    c->rope_theta              = parse_float(obj, "rope_theta", 1000000.0f);
    c->rms_norm_eps            = parse_float(obj, "rms_norm_eps", 1e-6f);

    /* head_dim derived or explicit */
    c->head_dim = parse_int(obj, "head_dim", 0);
    if (c->head_dim == 0 && c->num_attention_heads > 0) {
        c->head_dim = c->hidden_size / c->num_attention_heads;
    }
}

/* ─── Public API ────────────────────────────────────────────────────────── */

vv_status_t vv_config_parse(const char* json_path, vv_model_config_t* config) {
    if (!json_path || !config) return VV_ERR_NULL_PTR;
    memset(config, 0, sizeof(*config));

    char* json_str = read_file_to_string(json_path);
    if (!json_str) {
        VV_LOG_E("config: failed to read '%s'", json_path);
        return VV_ERR_IO;
    }

    cJSON* root = cJSON_Parse(json_str);
    vv_free(json_str);
    if (!root) {
        VV_LOG_E("config: JSON parse failed for '%s'", json_path);
        return VV_ERR_PARSE;
    }

    /* Acoustic tokenizer config */
    cJSON* ac_cfg = cJSON_GetObjectItem(root, "acoustic_tokenizer_config");
    if (ac_cfg) {
        parse_acoustic_tokenizer_config(ac_cfg, &config->acoustic);
    } else {
        /* Defaults for VibeVoice-ASR */
        config->acoustic.channels = 1;
        config->acoustic.causal = true;
        config->acoustic.vae_dim = 64;
        config->acoustic.fix_std = 0.5f;
        config->acoustic.std_dist_type = VV_ACOUSTIC_GAUSSIAN;
        config->acoustic.encoder_n_filters = 32;
        int default_ratios[] = {8, 5, 5, 4, 2, 2};
        memcpy(config->acoustic.encoder_ratios, default_ratios,
               sizeof(default_ratios));
        config->acoustic.n_ratios = 6;
        int default_depths[] = {3, 3, 3, 3, 3, 3, 8};
        memcpy(config->acoustic.encoder_depths, default_depths,
               sizeof(default_depths));
        config->acoustic.n_depths = 7;
    }

    /* Semantic tokenizer config */
    cJSON* sem_cfg = cJSON_GetObjectItem(root, "semantic_tokenizer_config");
    if (sem_cfg) {
        parse_semantic_tokenizer_config(sem_cfg, &config->semantic);
    } else {
        config->semantic.channels = 1;
        config->semantic.causal = true;
        config->semantic.vae_dim = 128;
        config->semantic.encoder_n_filters = 32;
        int default_ratios[] = {8, 5, 5, 4, 2, 2};
        memcpy(config->semantic.encoder_ratios, default_ratios,
               sizeof(default_ratios));
        config->semantic.n_ratios = 6;
        int default_depths[] = {3, 3, 3, 3, 3, 3, 8};
        memcpy(config->semantic.encoder_depths, default_depths,
               sizeof(default_depths));
        config->semantic.n_depths = 7;
    }

    /* Decoder config (Qwen2) - may be nested or at root level */
    cJSON* dec_cfg = cJSON_GetObjectItem(root, "decoder_config");
    if (dec_cfg) {
        parse_decoder_config(dec_cfg, &config->llm);
    } else {
        /* Try root level (some configs have flat layout) */
        parse_decoder_config(root, &config->llm);
    }

    config->acoustic_vae_dim = config->acoustic.vae_dim;
    config->semantic_vae_dim = config->semantic.vae_dim;

    cJSON_Delete(root);

    VV_LOG_I("config: loaded — LLM hidden=%d layers=%d heads=%d kv_heads=%d "
             "max_pos=%d, acoustic_vae=%d, semantic_vae=%d",
             config->llm.hidden_size, config->llm.num_hidden_layers,
             config->llm.num_attention_heads, config->llm.num_key_value_heads,
             config->llm.max_position_embeddings,
             config->acoustic_vae_dim, config->semantic_vae_dim);
    return VV_OK;
}
