/**
 * @file text_tokenizer.h
 * @brief BPE text tokenizer API for Qwen2.5 vocabulary (152064 tokens).
 *
 * Loads tokenizer.json from the model directory (HuggingFace format).
 * Supports encoding text → token IDs and decoding token IDs → text.
 * Special tokens for VibeVoice-ASR: <|object_ref_start|>, <|im_end|>, etc.
 */
#ifndef VV_TEXT_TOKENIZER_H
#define VV_TEXT_TOKENIZER_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque tokenizer handle. */
typedef struct vv_tokenizer vv_tokenizer_t;

/**
 * @brief Load tokenizer from a directory containing tokenizer.json.
 *
 * The tokenizer.json format (HuggingFace) contains:
 * - model.vocab: map of token → id
 * - model.merges: BPE merge rules
 * - added_tokens: special tokens with their IDs
 */
vv_status_t vv_tokenizer_load(const char* dir_path, vv_tokenizer_t** out);

/**
 * @brief Encode text to token IDs using BPE.
 *
 * @param tok      Tokenizer handle
 * @param text     Input text (UTF-8)
 * @param ids      Output: array of token IDs (caller must vv_free)
 * @param n_tokens Output: number of tokens
 */
vv_status_t vv_tokenizer_encode(const vv_tokenizer_t* tok, const char* text,
                                 int32_t** ids, int* n_tokens);

/**
 * @brief Decode token IDs back to text.
 *
 * @param tok      Tokenizer handle
 * @param ids      Token ID array
 * @param n_tokens Number of tokens
 * @param text     Output: decoded text string (caller must vv_free)
 */
vv_status_t vv_tokenizer_decode(const vv_tokenizer_t* tok,
                                 const int32_t* ids, int n_tokens,
                                 char** text);

/**
 * @brief Decode token IDs, optionally dropping HF "special" added tokens.
 *
 * @param skip_special  When true, tokens flagged `special` in tokenizer.json
 *                      (`<|im_end|>`, `<|endoftext|>`, the speech markers …)
 *                      are omitted from the output.
 */
vv_status_t vv_tokenizer_decode_ex(const vv_tokenizer_t* tok,
                                    const int32_t* ids, int n_tokens,
                                    bool skip_special, char** text);

/**
 * @brief Raw vocabulary string for a token id (NULL when unknown).
 *
 * The result is the byte-level-encoded form, not human-readable text; use
 * vv_tokenizer_decode() to get actual text.
 */
const char* vv_tokenizer_id_to_token(const vv_tokenizer_t* tok, int32_t id);

/**
 * @brief Get the token ID for a special token by name.
 *
 * @param tok   Tokenizer handle
 * @param name  Special token string (e.g., "<|im_end|>")
 * @return Token ID, or -1 if not found
 */
int vv_tokenizer_special_id(const vv_tokenizer_t* tok, const char* name);

/**
 * @brief Get vocabulary size.
 */
int vv_tokenizer_vocab_size(const vv_tokenizer_t* tok);

/**
 * @brief Free tokenizer.
 */
vv_status_t vv_tokenizer_free(vv_tokenizer_t* tok);

#ifdef __cplusplus
}
#endif

#endif /* VV_TEXT_TOKENIZER_H */
