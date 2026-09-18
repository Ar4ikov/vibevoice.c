/**
 * @file family.c
 * @brief Prompt, stop tokens and generation mode per model family.
 */

#include "vibevoice/family.h"
#include "vibevoice/model.h"
#include "vibevoice/text_tokenizer.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

/* Token strings (special_tokens.c). */
extern const char* VV_TOKEN_IM_START;
extern const char* VV_TOKEN_IM_END;
extern const char* VV_TOKEN_ENDOFTEXT;
extern const char* VV_TOKEN_SPEECH_START;
extern const char* VV_TOKEN_SPEECH_PAD;
extern const char* VV_TOKEN_SPEECH_END;
extern const char* VV_TOKEN_TEXT_CHUNK_END;

vv_status_t vv_family_init(vv_family_t* f, const vv_model_config_t* cfg,
                           const vv_tokenizer_t* tok) {
    if (!f || !cfg || !tok) return VV_ERR_NULL_PTR;
    memset(f, 0, sizeof(*f));

    f->id               = cfg->family;
    f->mode             = (cfg->family == VV_FAMILY_ASR_STREAMING_7B)
                          ? VV_GEN_CHUNKED : VV_GEN_ONE_SHOT;
    f->normalize_audio  = cfg->audio.normalize_audio;
    f->sample_rate      = cfg->audio.target_sample_rate;
    f->frame_samples    = cfg->audio.compress_ratio;
    f->chunk_frames     = (f->mode == VV_GEN_CHUNKED) ? cfg->audio.chunk_frames : 0;
    f->lookahead_frames = (f->mode == VV_GEN_CHUNKED)
                          ? cfg->audio.lookahead_frames : 0;

    vv_family_tokens_t* t = &f->tok;
    t->im_start       = vv_tokenizer_special_id(tok, VV_TOKEN_IM_START);
    t->im_end         = vv_tokenizer_special_id(tok, VV_TOKEN_IM_END);
    t->endoftext      = vv_tokenizer_special_id(tok, VV_TOKEN_ENDOFTEXT);
    t->speech_start   = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_START);
    t->speech_pad     = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_PAD);
    t->speech_end     = vv_tokenizer_special_id(tok, VV_TOKEN_SPEECH_END);
    t->text_chunk_end = vv_tokenizer_special_id(tok, VV_TOKEN_TEXT_CHUNK_END);

    if (t->speech_start < 0 || t->speech_pad < 0 || t->speech_end < 0) {
        VV_LOG_E("family: the tokenizer has no speech marker tokens");
        return VV_ERR_MODEL_FORMAT;
    }
    if (f->mode == VV_GEN_ONE_SHOT && (t->im_start < 0 || t->im_end < 0)) {
        VV_LOG_E("family: %s needs <|im_start|> and <|im_end|>",
                 vv_model_family_name(f->id));
        return VV_ERR_MODEL_FORMAT;
    }
    if (f->mode == VV_GEN_CHUNKED) {
        /* Without it no chunk ever ends; upstream refuses the same way. */
        if (t->text_chunk_end < 0 || t->endoftext < 0) {
            VV_LOG_E("family: %s needs <|text_chunk_end|> and <|endoftext|> "
                     "in tokenizer.json", vv_model_family_name(f->id));
            return VV_ERR_MODEL_FORMAT;
        }
        if (f->chunk_frames <= 0) {
            VV_LOG_E("family: %s without chunk_frames",
                     vv_model_family_name(f->id));
            return VV_ERR_MODEL_FORMAT;
        }
    }
    return VV_OK;
}

vv_stop_t vv_family_stop(const vv_family_t* f, int32_t id) {
    if (!f) return VV_STOP_END;
    const vv_family_tokens_t* t = &f->tok;
    if (f->mode == VV_GEN_CHUNKED) {
        /* streaming_generate stops a chunk on these two and nothing else. */
        if (id == t->text_chunk_end) return VV_STOP_YIELD;
        if (id == t->endoftext)      return VV_STOP_END;
        return VV_STOP_CONTINUE;
    }
    /* ChatML: the assistant turn ends with <|im_end|>; <|endoftext|> too. */
    if (t->im_end >= 0 && id == t->im_end)       return VV_STOP_END;
    if (t->endoftext >= 0 && id == t->endoftext) return VV_STOP_END;
    return VV_STOP_CONTINUE;
}

void vv_prompt_free(vv_prompt_t* p) {
    if (!p) return;
    if (p->ids) vv_free(p->ids);
    memset(p, 0, sizeof(*p));
}

/** @brief Log the ids with the audio run folded, as the 7B builder always did. */
static void log_prompt(const vv_prompt_t* p, int32_t pad, float dur) {
    VV_LOG_I("prompt: %d tokens (audio frames %d at offset %d, audio=%.2f sec)",
             p->n, p->n_audio, p->audio_offset, (double)dur);
    if (vv_log_get_level() < VV_LOG_DEBUG) return;
    char line[4096];
    int w = 0, shown = 0;
    for (int i = 0; i < p->n && w < (int)sizeof(line) - 16; i++) {
        if (i == p->audio_offset && p->n_audio > 0) {
            w += snprintf(line + w, sizeof(line) - (size_t)w,
                          "[%d x%d] ", pad, p->n_audio);
            i += p->n_audio - 1;
            continue;
        }
        w += snprintf(line + w, sizeof(line) - (size_t)w, "%d ", p->ids[i]);
        shown++;
    }
    VV_LOG_D("prompt ids (%d shown): %s", shown, line);
}

/**
 * @brief The batch models' ChatML turn.
 *
 * Exact training/inference format, reproduced from
 * vibevoice_asr_processor.py::_process_single_audio + the ASR chat template
 * ("<|im_start|>{role}\n{content}<|im_end|>\n" per message):
 *
 *   [system + "<|im_start|>user\n"] <|object_ref_start|> [pad × N]
 *   <|object_ref_end|> ["\nThis is a D seconds audio, ...<|im_end|>\n"]
 *
 * The processor does NOT append a generation prompt — the model emits
 * "<|im_start|>assistant\n" itself as its first tokens.
 */
static vv_status_t build_chatml(const vv_family_t* f, const vv_tokenizer_t* tok,
                                int n_audio_frames, float dur,
                                const char* context_info, vv_prompt_t* out) {
    static const char* SYSTEM_MSG =
        "<|im_start|>system\n"
        "You are a helpful assistant that transcribes audio input into text "
        "output in JSON format.<|im_end|>\n"
        "<|im_start|>user\n";
    static const char* KEYS = "Start time, End time, Speaker ID, Content";

    int32_t *pre = NULL, *post = NULL;
    int n_pre = 0, n_post = 0;
    vv_status_t s = vv_tokenizer_encode(tok, SYSTEM_MSG, &pre, &n_pre);
    if (s != VV_OK) return s;

    char buf[1024];
    int w;
    if (context_info && context_info[0])
        w = snprintf(buf, sizeof(buf),
                     "\nThis is a %.2f seconds audio, with extra info: %s\n\n"
                     "Please transcribe it with these keys: %s<|im_end|>\n",
                     (double)dur, context_info, KEYS);
    else
        w = snprintf(buf, sizeof(buf),
                     "\nThis is a %.2f seconds audio, please transcribe it "
                     "with these keys: %s<|im_end|>\n", (double)dur, KEYS);
    if (w < 0 || (size_t)w >= sizeof(buf)) {
        /* A truncated instruction loses its <|im_end|>; say so instead. */
        vv_free(pre);
        VV_LOG_E("prompt: extra info too long (%d bytes)", w);
        return VV_ERR_OVERFLOW;
    }
    s = vv_tokenizer_encode(tok, buf, &post, &n_post);
    if (s != VV_OK) { vv_free(pre); return s; }

    const int total = n_pre + 1 + n_audio_frames + 1 + n_post;
    int32_t* ids = (int32_t*)vv_alloc((size_t)total * sizeof(int32_t));
    if (!ids) { vv_free(pre); vv_free(post); return VV_ERR_OUT_OF_MEMORY; }

    int p = 0;
    for (int i = 0; i < n_pre; i++) ids[p++] = pre[i];
    ids[p++] = f->tok.speech_start;
    const int audio_off = p;
    for (int i = 0; i < n_audio_frames; i++) ids[p++] = f->tok.speech_pad;
    ids[p++] = f->tok.speech_end;
    for (int i = 0; i < n_post; i++) ids[p++] = post[i];
    vv_free(pre);
    vv_free(post);

    out->ids = ids;
    out->n = p;
    out->audio_offset = audio_off;
    out->n_audio = n_audio_frames;
    return VV_OK;
}

/**
 * @brief The streaming model's plain-text instruction (no chat template),
 *        verbatim from `streaming_generate` / `init_streaming_state`.
 */
static vv_status_t build_streaming(const vv_tokenizer_t* tok,
                                   const char* context_info,
                                   vv_prompt_t* out) {
    static const char* HEAD =
        "You are a helpful assistant that transcribes audio input into text "
        "output. Please transcribe the following audios streamingly with "
        "these keys: speaker, content";
    char buf[1024];
    int w;
    if (context_info && context_info[0])
        w = snprintf(buf, sizeof(buf), "%s and extra info: %s\n",
                     HEAD, context_info);
    else
        w = snprintf(buf, sizeof(buf), "%s\n", HEAD);
    if (w < 0 || (size_t)w >= sizeof(buf)) {
        VV_LOG_E("prompt: extra info too long (%d bytes)", w);
        return VV_ERR_OVERFLOW;
    }
    int32_t* ids = NULL;
    int n = 0;
    vv_status_t s = vv_tokenizer_encode(tok, buf, &ids, &n);
    if (s != VV_OK) return s;
    out->ids = ids;
    out->n = n;
    out->audio_offset = -1;
    out->n_audio = 0;
    return VV_OK;
}

vv_status_t vv_family_build_prompt(const vv_family_t* f,
                                   const vv_tokenizer_t* tok,
                                   int n_audio_frames, float duration_sec,
                                   const char* context_info,
                                   vv_prompt_t* out) {
    if (!f || !tok || !out) return VV_ERR_NULL_PTR;
    memset(out, 0, sizeof(*out));
    if (n_audio_frames < 0) return VV_ERR_INVALID_ARG;

    vv_status_t s;
    if (f->mode == VV_GEN_CHUNKED)
        s = build_streaming(tok, context_info, out);
    else
        s = build_chatml(f, tok, n_audio_frames, duration_sec, context_info,
                         out);
    if (s == VV_OK) log_prompt(out, f->tok.speech_pad, duration_sec);
    return s;
}

vv_status_t vv_family_chunk_prompt(const vv_family_t* f, int n_frames,
                                   bool lead_chunk_end, vv_prompt_t* out) {
    if (!f || !out) return VV_ERR_NULL_PTR;
    memset(out, 0, sizeof(*out));
    if (n_frames <= 0) return VV_ERR_INVALID_ARG;
    if (lead_chunk_end && f->tok.text_chunk_end < 0) return VV_ERR_UNSUPPORTED;

    const int total = (lead_chunk_end ? 1 : 0) + 1 + n_frames + 1;
    int32_t* ids = (int32_t*)vv_alloc((size_t)total * sizeof(int32_t));
    if (!ids) return VV_ERR_OUT_OF_MEMORY;
    int p = 0;
    if (lead_chunk_end) ids[p++] = f->tok.text_chunk_end;
    ids[p++] = f->tok.speech_start;
    out->audio_offset = p;
    for (int i = 0; i < n_frames; i++) ids[p++] = f->tok.speech_pad;
    ids[p++] = f->tok.speech_end;
    out->ids = ids;
    out->n = p;
    out->n_audio = n_frames;
    return VV_OK;
}

int vv_family_chunk_samples(const vv_family_t* f) {
    return f ? f->chunk_frames * f->frame_samples : 0;
}

int vv_family_window_samples(const vv_family_t* f) {
    return f ? (f->chunk_frames + f->lookahead_frames) * f->frame_samples : 0;
}
