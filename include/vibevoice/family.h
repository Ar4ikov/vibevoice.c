/**
 * @file family.h
 * @brief What differs between the VibeVoice ASR models at run time.
 *
 * The 7B batch model, its BitNet sibling and the streaming 7B share the
 * speech front end and the Qwen2 layer. What they do not share is the
 * prompt around the audio, which tokens stop generation, whether the audio
 * is loudness-normalized first, and whether text comes out once per clip or
 * once per chunk. A family bundles exactly that, resolved once per context
 * (token ids included) so nothing on the per-token path looks a string up.
 */
#ifndef VV_FAMILY_H
#define VV_FAMILY_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vv_tokenizer;

/** @brief What a generated token means for the generation loop. */
typedef enum vv_stop {
    VV_STOP_CONTINUE = 0,  /**< keep decoding                               */
    VV_STOP_YIELD,         /**< chunk finished; more audio may follow       */
    VV_STOP_END,           /**< transcript finished                         */
} vv_stop_t;

/** @brief How text is produced. */
typedef enum vv_gen_mode {
    VV_GEN_ONE_SHOT = 0,   /**< whole clip in one prompt, one answer        */
    VV_GEN_CHUNKED,        /**< a prompt, then audio chunk by chunk on top of
                                the same KV cache, text after each chunk    */
} vv_gen_mode_t;

/** @brief Special token ids a family uses; -1 where the vocabulary has none. */
typedef struct vv_family_tokens {
    int32_t im_start;
    int32_t im_end;
    int32_t endoftext;
    int32_t speech_start;     /**< <|object_ref_start|> */
    int32_t speech_pad;       /**< <|box_start|>        */
    int32_t speech_end;       /**< <|object_ref_end|>   */
    int32_t text_chunk_end;   /**< <|text_chunk_end|>, streaming only */
} vv_family_tokens_t;

/** @brief A family, resolved against one tokenizer. */
typedef struct vv_family {
    vv_model_family_t  id;
    vv_gen_mode_t      mode;
    bool               normalize_audio;  /**< -25 dBFS before encoding    */
    int                sample_rate;      /**< 24000                       */
    int                frame_samples;    /**< samples per frame, 3200     */
    int                chunk_frames;     /**< chunked only, else 0        */
    int                lookahead_frames; /**< chunked only, else 0        */
    vv_family_tokens_t tok;
} vv_family_t;

/**
 * @brief Resolve the family for a loaded config and its tokenizer.
 *
 * Fails when the tokenizer lacks a token the family cannot work without
 * (the speech markers everywhere, `<|text_chunk_end|>` for streaming).
 */
vv_status_t vv_family_init(vv_family_t* f, const vv_model_config_t* cfg,
                           const struct vv_tokenizer* tok);

/** @brief What `token_id` means to the generation loop. */
vv_stop_t vv_family_stop(const vv_family_t* f, int32_t token_id);

/** @brief A token sequence with the place where audio features go. */
typedef struct vv_prompt {
    int32_t* ids;           /**< vv_alloc'd; free with vv_prompt_free()    */
    int      n;
    int      audio_offset;  /**< first speech_pad position; -1 if none     */
    int      n_audio;       /**< speech_pad positions from audio_offset    */
} vv_prompt_t;

void vv_prompt_free(vv_prompt_t* p);

/**
 * @brief The prompt that starts a transcription.
 *
 * One-shot families: the whole ChatML turn with `n_audio_frames` pads,
 * exactly as vibevoice_asr_processor.py builds it — no generation prompt,
 * the model writes `<|im_start|>assistant\n` itself.
 *
 * Chunked families: the plain-text instruction `streaming_generate` prefills
 * before any audio; `n_audio_frames` and `duration_sec` are ignored and the
 * result has no audio positions. Chunks come from vv_family_chunk_prompt().
 *
 * @param context_info  hotwords / extra info, NULL or "" for none
 */
vv_status_t vv_family_build_prompt(const vv_family_t* f,
                                   const struct vv_tokenizer* tok,
                                   int n_audio_frames, float duration_sec,
                                   const char* context_info,
                                   vv_prompt_t* out);

/**
 * @brief One streaming chunk: `[<|text_chunk_end|>] speech_start pad×n
 *        speech_end`.
 *
 * The chunk-end token of the previous chunk is what `streaming_generate`
 * feeds after its last text token; folding it into the next chunk's prefill
 * is the same computation with one step fewer.
 */
vv_status_t vv_family_chunk_prompt(const vv_family_t* f, int n_frames,
                                   bool lead_chunk_end, vv_prompt_t* out);

/** @brief Samples of new audio per chunk (chunk_frames × frame_samples). */
int vv_family_chunk_samples(const vv_family_t* f);

/** @brief Samples one chunk's encoder window covers, lookahead included. */
int vv_family_window_samples(const vv_family_t* f);

#ifdef __cplusplus
}
#endif

#endif /* VV_FAMILY_H */
