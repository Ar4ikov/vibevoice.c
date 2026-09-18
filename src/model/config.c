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

/**
 * @brief A dimension the model cannot run without.
 *
 * Missing used to mean "the 7B value", which let a 1.5B config load as a 7B
 * one and fail much later on a shape nobody could explain. Now it is an
 * error with the key's name in it.
 */
static vv_status_t require_int(cJSON* obj, const char* key, int* out) {
    cJSON* v = cJSON_GetObjectItem(obj, key);
    if (!v || !cJSON_IsNumber(v) || v->valuedouble < 1.0) {
        VV_LOG_E("config: '%s' is missing or not a positive number", key);
        return VV_ERR_MODEL_FORMAT;
    }
    *out = (int)v->valuedouble;
    return VV_OK;
}

static vv_status_t parse_decoder_config(cJSON* obj, cJSON* root,
                                        vv_llm_config_t* c) {
    vv_status_t s;
    if ((s = require_int(obj, "hidden_size", &c->hidden_size)) != VV_OK ||
        (s = require_int(obj, "num_hidden_layers",
                         &c->num_hidden_layers)) != VV_OK ||
        (s = require_int(obj, "num_attention_heads",
                         &c->num_attention_heads)) != VV_OK ||
        (s = require_int(obj, "intermediate_size",
                         &c->intermediate_size)) != VV_OK ||
        (s = require_int(obj, "vocab_size", &c->vocab_size)) != VV_OK)
        return s;

    /* The rest have Qwen2Config's own defaults, which are not the 7B's. */
    c->num_key_value_heads     = parse_int(obj, "num_key_value_heads",
                                           c->num_attention_heads);
    c->max_position_embeddings = parse_int(obj, "max_position_embeddings",
                                           32768);
    c->rope_theta              = parse_float(obj, "rope_theta", 10000.0f);
    c->rms_norm_eps            = parse_float(obj, "rms_norm_eps", 1e-6f);
    c->attention_bias          = parse_bool(obj, "attention_bias", true);

    /* Tied embeddings: inside decoder_config (BitNet) or at the root. */
    {
        cJSON* t = cJSON_GetObjectItem(obj, "tie_word_embeddings");
        if (!t && root) t = cJSON_GetObjectItem(root, "tie_word_embeddings");
        c->tie_word_embeddings = (t && cJSON_IsBool(t)) ? cJSON_IsTrue(t)
                                                         : false;
    }

    /* head_dim derived or explicit */
    c->head_dim = parse_int(obj, "head_dim", 0);
    if (c->head_dim == 0) {
        if (c->hidden_size % c->num_attention_heads != 0) {
            VV_LOG_E("config: hidden_size %d is not a multiple of %d heads "
                     "and there is no head_dim", c->hidden_size,
                     c->num_attention_heads);
            return VV_ERR_MODEL_FORMAT;
        }
        c->head_dim = c->hidden_size / c->num_attention_heads;
    }
    if (c->num_key_value_heads <= 0 ||
        c->num_attention_heads % c->num_key_value_heads != 0) {
        VV_LOG_E("config: %d attention heads cannot share %d KV heads",
                 c->num_attention_heads, c->num_key_value_heads);
        return VV_ERR_MODEL_FORMAT;
    }

    /*
     * What the layer does is fixed in the kernels: SwiGLU with SiLU, plain
     * RoPE, full attention everywhere. A config asking for anything else
     * would run and produce nonsense, so it is refused here instead.
     */
    {
        cJSON* a = cJSON_GetObjectItem(obj, "hidden_act");
        if (a && (!cJSON_IsString(a) || strcmp(a->valuestring, "silu") != 0)) {
            VV_LOG_E("config: hidden_act '%s' is not supported (only silu)",
                     cJSON_IsString(a) ? a->valuestring : "?");
            return VV_ERR_UNSUPPORTED;
        }
        cJSON* rs = cJSON_GetObjectItem(obj, "rope_scaling");
        if (rs && !cJSON_IsNull(rs)) {
            VV_LOG_E("config: rope_scaling is not supported");
            return VV_ERR_UNSUPPORTED;
        }
        if (parse_bool(obj, "use_sliding_window", false)) {
            VV_LOG_E("config: sliding-window attention is not supported");
            return VV_ERR_UNSUPPORTED;
        }
    }
    return VV_OK;
}

/**
 * @brief The family config.json alone points to.
 *
 * The streaming model names itself; the batch 7B and BitNet share an
 * architecture string (`VibeVoiceForASRTraining`) and differ in size, so
 * the 1536-wide LM is what marks BitNet. Chunk geometry in
 * preprocessor_config.json promotes a model to streaming as well.
 */
static vv_model_family_t family_of(const vv_model_config_t* c) {
    if (strstr(c->architecture, "Streaming") || c->audio.chunk_frames > 0)
        return VV_FAMILY_ASR_STREAMING_7B;
    if (strstr(c->architecture, "BitNet") || c->llm.hidden_size == 1536)
        return VV_FAMILY_ASR_BITNET;
    return VV_FAMILY_ASR_7B;
}

const char* vv_model_family_name(vv_model_family_t f) {
    switch (f) {
        case VV_FAMILY_ASR_7B:           return "asr-7b";
        case VV_FAMILY_ASR_BITNET:       return "asr-bitnet";
        case VV_FAMILY_ASR_STREAMING_7B: return "asr-streaming-7b";
        default:                         return "?";
    }
}

/** @brief What VibeVoiceASRProcessor assumes without a preprocessor file. */
static void audio_defaults(vv_audio_config_t* a) {
    a->target_sample_rate = 24000;
    a->normalize_audio    = true;
    a->target_db_fs       = -25.0f;
    a->eps                = 1e-6f;
    a->compress_ratio     = 3200;
    a->chunk_frames       = 0;
    a->lookahead_frames   = 0;
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
    vv_status_t s = parse_decoder_config(dec_cfg ? dec_cfg : root, root,
                                         &config->llm);
    if (s != VV_OK) {
        VV_LOG_E("config: '%s' does not describe a usable decoder",
                 json_path);
        cJSON_Delete(root);
        return s;
    }

    {
        cJSON* arch = cJSON_GetObjectItem(root, "architectures");
        cJSON* a0 = (arch && cJSON_IsArray(arch))
                    ? cJSON_GetArrayItem(arch, 0) : NULL;
        if (a0 && cJSON_IsString(a0)) {
            strncpy(config->architecture, a0->valuestring,
                    sizeof(config->architecture) - 1);
            config->architecture[sizeof(config->architecture) - 1] = '\0';
        }
    }

    config->acoustic_vae_dim = config->acoustic.vae_dim;
    config->semantic_vae_dim = config->semantic.vae_dim;
    audio_defaults(&config->audio);
    /* The streaming checkpoint repeats these two in config.json. */
    config->audio.target_sample_rate =
        parse_int(root, "target_sample_rate", config->audio.target_sample_rate);
    config->audio.compress_ratio =
        parse_int(root, "speech_tok_compress_ratio", config->audio.compress_ratio);
    config->family = family_of(config);

    cJSON_Delete(root);

    VV_LOG_I("config: loaded — LLM hidden=%d layers=%d heads=%d kv_heads=%d "
             "max_pos=%d, acoustic_vae=%d, semantic_vae=%d, tied=%d, "
             "family=%s",
             config->llm.hidden_size, config->llm.num_hidden_layers,
             config->llm.num_attention_heads, config->llm.num_key_value_heads,
             config->llm.max_position_embeddings,
             config->acoustic_vae_dim, config->semantic_vae_dim,
             (int)config->llm.tie_word_embeddings,
             vv_model_family_name(config->family));
    return VV_OK;
}

vv_status_t vv_config_parse_preprocessor(const char* json_path,
                                         vv_model_config_t* config) {
    if (!json_path || !config) return VV_ERR_NULL_PTR;

    char* json_str = read_file_to_string(json_path);
    if (!json_str) {
        /* The BF16 and BitNet repositories ship none: defaults apply. */
        config->family = family_of(config);
        return VV_OK;
    }
    cJSON* root = cJSON_Parse(json_str);
    vv_free(json_str);
    if (!root) {
        VV_LOG_E("config: JSON parse failed for '%s'", json_path);
        return VV_ERR_PARSE;
    }

    vv_audio_config_t* a = &config->audio;
    a->target_sample_rate = parse_int(root, "target_sample_rate",
                                      a->target_sample_rate);
    a->compress_ratio     = parse_int(root, "speech_tok_compress_ratio",
                                      a->compress_ratio);
    a->normalize_audio    = parse_bool(root, "normalize_audio",
                                       a->normalize_audio);
    a->target_db_fs       = parse_float(root, "target_dB_FS", a->target_db_fs);
    a->eps                = parse_float(root, "eps", a->eps);
    a->chunk_frames       = parse_int(root, "chunk_frames", 0);
    a->lookahead_frames   = parse_int(root, "lookahead_frames", 0);
    cJSON_Delete(root);

    if (a->target_sample_rate != 24000 || a->compress_ratio != 3200) {
        VV_LOG_E("config: %d Hz at %d samples per frame is not supported "
                 "(the speech encoder is built for 24000 / 3200)",
                 a->target_sample_rate, a->compress_ratio);
        return VV_ERR_UNSUPPORTED;
    }
    if (a->chunk_frames < 0 || a->lookahead_frames < 0) {
        VV_LOG_E("config: negative chunk geometry in '%s'", json_path);
        return VV_ERR_MODEL_FORMAT;
    }

    config->family = family_of(config);
    return VV_OK;
}

vv_status_t vv_config_load(const char* model_dir, vv_model_config_t* config) {
    if (!model_dir || !config) return VV_ERR_NULL_PTR;

    char path[1024];
    const size_t n = strlen(model_dir);
    const char* sep = (n > 0 && (model_dir[n - 1] == '/' ||
                                 model_dir[n - 1] == '\\')) ? "" : "/";
    snprintf(path, sizeof(path), "%s%sconfig.json", model_dir, sep);
    vv_status_t s = vv_config_parse(path, config);
    if (s != VV_OK) return s;

    snprintf(path, sizeof(path), "%s%spreprocessor_config.json",
             model_dir, sep);
    s = vv_config_parse_preprocessor(path, config);
    if (s != VV_OK) return s;

    if (config->family == VV_FAMILY_ASR_STREAMING_7B &&
        config->audio.chunk_frames <= 0) {
        /* Upstream refuses a streaming checkpoint without its geometry. */
        VV_LOG_E("config: streaming model without chunk_frames in "
                 "preprocessor_config.json");
        return VV_ERR_MODEL_FORMAT;
    }
    VV_LOG_I("config: family %s (%s), normalize=%d, chunk %d + %d frames",
             vv_model_family_name(config->family),
             config->architecture[0] ? config->architecture : "no architecture",
             (int)config->audio.normalize_audio, config->audio.chunk_frames,
             config->audio.lookahead_frames);
    return VV_OK;
}
