/**
 * @file special_tokens.c
 * @brief VibeVoice-ASR special token definitions and helpers.
 */

#include "vibevoice/text_tokenizer.h"
#include "vibevoice/vibevoice.h"

#include <string.h>
#include <stdio.h>

/* Well-known special token strings */
const char* VV_TOKEN_START_TRANSCRIPT = "<|startoftranscript|>";
const char* VV_TOKEN_END_TRANSCRIPT   = "<|endoftranscript|>";
const char* VV_TOKEN_NOSPEECH         = "<|nospeech|>";
const char* VV_TOKEN_HOTWORDS         = "<|hotwords|>";

/**
 * @brief Check if a token ID corresponds to the end-of-transcript token.
 */
bool vv_is_end_token(const vv_tokenizer_t* tok, int32_t token_id) {
    int end_id = vv_tokenizer_special_id(tok, VV_TOKEN_END_TRANSCRIPT);
    return (end_id >= 0 && token_id == end_id);
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
