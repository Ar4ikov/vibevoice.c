/**
 * @file special_tokens.c
 * @brief VibeVoice-ASR special token definitions and helpers.
 *
 * VibeVoice reuses Qwen2.5 extended tokens for speech:
 *   speech_start = <|object_ref_start|>  (151646)
 *   speech_pad   = <|box_start|>         (151648)
 *   speech_end   = <|object_ref_end|>    (151647)
 * ChatML framing:
 *   <|im_start|> (151644)  /  <|im_end|> (151645)
 * See: vllm_plugin/tools/generate_tokenizer_files.py
 */

#include "vibevoice/text_tokenizer.h"
#include "vibevoice/vibevoice.h"

#include <string.h>
#include <stdio.h>

/* ── ChatML framing tokens ─────────────────────────────────────────────── */
const char* VV_TOKEN_IM_START     = "<|im_start|>";
const char* VV_TOKEN_IM_END       = "<|im_end|>";
const char* VV_TOKEN_ENDOFTEXT    = "<|endoftext|>";

/* ── VibeVoice speech tokens (Qwen2.5 extended) ───────────────────────── */
const char* VV_TOKEN_SPEECH_START = "<|object_ref_start|>";  /* speech_start_id */
const char* VV_TOKEN_SPEECH_PAD   = "<|box_start|>";         /* speech_pad_id   */
const char* VV_TOKEN_SPEECH_END   = "<|object_ref_end|>";    /* speech_end_id   */

/* ── Streaming model: ends one chunk of text (151665) ─────────────────── */
const char* VV_TOKEN_TEXT_CHUNK_END = "<|text_chunk_end|>";

/* Legacy names (kept for link compat, pipeline.c no longer uses them) */
const char* VV_TOKEN_START_TRANSCRIPT = "<|object_ref_start|>";
const char* VV_TOKEN_END_TRANSCRIPT   = "<|im_end|>";
const char* VV_TOKEN_NOSPEECH         = "<|nospeech|>";
const char* VV_TOKEN_HOTWORDS         = "<|hotwords|>";

/**
 * @brief Check if a token ID is an end-of-generation token.
 *
 * In ChatML the assistant turn ends with <|im_end|>.
 * <|endoftext|> is also a valid stop signal.
 */
bool vv_is_end_token(const vv_tokenizer_t* tok, int32_t token_id) {
    int im_end = vv_tokenizer_special_id(tok, VV_TOKEN_IM_END);
    if (im_end >= 0 && token_id == im_end) return true;
    int eot = vv_tokenizer_special_id(tok, VV_TOKEN_ENDOFTEXT);
    if (eot >= 0 && token_id == eot) return true;
    return false;
}

/**
 * @brief Build the special token string for a speaker.
 *
 * @param speaker_id  Speaker number (1-based)
 * @param buf         Output buffer
 * @param buf_size    Buffer size
 */
void vv_speaker_token_str(int speaker_id, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "<|speaker_%d|>", speaker_id);
}

/**
 * @brief Build timestamp token string.
 *
 * @param time_sec  Time in seconds
 * @param buf       Output buffer
 * @param buf_size  Buffer size
 */
void vv_timestamp_token_str(float time_sec, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "<|timestamp_%.2f|>", time_sec);
}
