/**
 * @file stream.h
 * @brief Chunked streaming transcription (VibeVoice-ASR-Streaming-7B).
 *
 * The streaming checkpoint transcribes while audio arrives. Its protocol
 * (docs/STREAMING.md has the full description, verified against upstream
 * `streaming_generate`) is:
 *
 *  1. Prefill a plain-text prompt, no chat template.
 *  2. Cut the 24 kHz PCM into windows of chunk + lookahead samples
 *     (22 + 4 frames of 3200 samples = 83200), one every chunk (70400)
 *     samples. The last window is zero-padded to full length. Each window is
 *     encoded on its own, with no encoder state carried between windows.
 *  3. Per window, prefill `[<|object_ref_start|>, 26 feature rows,
 *     <|object_ref_end|>]` on top of the running KV cache, greedy-decode
 *     until `<|text_chunk_end|>` or EOS (or a per-chunk token cap), then
 *     feed `<|text_chunk_end|>` and keep the cache for the next window.
 *
 * This header splits that into pieces that can be tested without weights:
 *
 *  - ::vv_stream_chunker_t turns pushed PCM of any size into windows;
 *  - vv_stream_chunk_layout() assembles the per-chunk prefill rows;
 *  - ::vv_stream_text_t turns token bytes into UTF-8-safe text deltas,
 *    holding back incomplete sequences and the control strings upstream
 *    strips;
 *  - ::vv_stream_t is the session, driving a ::vv_stream_backend_t that owns
 *    the model work (encode a window, prefill at an offset, decode a step).
 *
 * vv_stream_open() puts a session on a real ::vv_inference_ctx_t (GPU or
 * CPU): the prompt into a reset cache, each window through the device's
 * speech front end as a stateless job, the chunk prefilled at the cache's
 * current length and greedy decode on the captured step (src/inference/
 * stream_ctx.c). Tests drive the same session through
 * vv_stream_open_backend() with a scripted backend.
 */
#ifndef VV_STREAM_H
#define VV_STREAM_H

#include "vibevoice/types.h"
#include "vibevoice/text_tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vv_inference_ctx;

/* ─── Geometry ──────────────────────────────────────────────────────────── */

/**
 * @brief Window geometry, from the checkpoint's preprocessor_config.json.
 *
 * Streaming-7B: 24000 Hz, 3200 samples per frame, `chunk_frames` 22,
 * `lookahead_frames` 4. `normalize_audio` is false for this checkpoint: the
 * PCM goes to the encoder exactly as loaded (resampled, never rescaled).
 */
typedef struct {
    int  sample_rate;       /**< 24000 */
    int  frame_samples;     /**< speech_tok_compress_ratio, 3200 */
    int  chunk_frames;      /**< frames of new audio per chunk, 22 */
    int  lookahead_frames;  /**< frames of right context per window, 4 */
    bool normalize_audio;   /**< false for Streaming-7B */
} vv_stream_geom_t;

/** @brief Streaming-7B defaults (22 + 4 frames at 24 kHz). */
void vv_stream_geom_default(vv_stream_geom_t* g);

/** @brief VV_ERR_INVALID_ARG unless every count is positive and sane. */
vv_status_t vv_stream_geom_validate(const vv_stream_geom_t* g);

/** @brief Samples of new audio per chunk (the window stride), 70400. */
int64_t vv_stream_chunk_samples(const vv_stream_geom_t* g);

/** @brief Samples per window, chunk + lookahead, 83200. */
int64_t vv_stream_window_samples(const vv_stream_geom_t* g);

/** @brief Encoder frames per window, 26. */
int vv_stream_window_frames(const vv_stream_geom_t* g);

/** @brief Windows a clip of `total` samples produces: ceil(total / chunk). */
int64_t vv_stream_n_windows(const vv_stream_geom_t* g, int64_t total);

/* ─── Chunk scheduler ───────────────────────────────────────────────────── */

/** @brief Where one window sits in the stream. */
typedef struct {
    int64_t index;     /**< 0, 1, 2, ... */
    int64_t start;     /**< first sample, index * chunk_samples */
    int64_t n_valid;   /**< real samples in the window; the rest is zeros */
    bool    is_last;   /**< known only after vv_stream_chunker_finish() */
} vv_stream_window_t;

/** @brief Opaque PCM-to-window scheduler. */
typedef struct vv_stream_chunker vv_stream_chunker_t;

vv_status_t vv_stream_chunker_create(const vv_stream_geom_t* g,
                                     vv_stream_chunker_t** out);
void vv_stream_chunker_free(vv_stream_chunker_t* c);

/**
 * @brief Append 24 kHz mono PCM. Any size, including zero.
 *
 * Only the samples the next window still needs are kept: after a window is
 * popped its first `chunk_samples` are dropped and the lookahead stays.
 * VV_ERR_INVALID_ARG after finish.
 */
vv_status_t vv_stream_chunker_push(vv_stream_chunker_t* c,
                                   const float* pcm, size_t n);

/** @brief No more input: the tail becomes zero-padded windows. */
void vv_stream_chunker_finish(vv_stream_chunker_t* c);

/**
 * @brief Pop the next ready window.
 *
 * Before finish a window is ready once all of its samples have arrived.
 * After finish every window whose start is inside the audio is ready, and
 * the missing part is zeros, which is what upstream's `pad_last_chunk` does.
 *
 * @param out   window_samples floats, overwritten
 * @param info  where the window sits (may be NULL)
 * @return 1 when a window was written, 0 when none is ready
 */
int vv_stream_chunker_next(vv_stream_chunker_t* c, float* out,
                           vv_stream_window_t* info);

/** @brief Windows ready to pop right now. */
int64_t vv_stream_chunker_ready(const vv_stream_chunker_t* c);

/** @brief Samples pushed so far. */
int64_t vv_stream_chunker_total(const vv_stream_chunker_t* c);

/* ─── Prompt and per-chunk token layout ─────────────────────────────────── */

/** @brief Special token ids the protocol uses. -1 = unknown. */
typedef struct {
    int32_t speech_start;    /**< <|object_ref_start|>, 151646 */
    int32_t speech_end;      /**< <|object_ref_end|>,   151647 */
    int32_t text_chunk_end;  /**< <|text_chunk_end|>,   151665 */
    int32_t eos;             /**< <|endoftext|>,        151643 */
} vv_stream_ids_t;

/** @brief The Streaming-7B tokenizer's ids. */
void vv_stream_ids_default(vv_stream_ids_t* ids);

/** @brief Resolve the ids by name; VV_ERR_NOT_FOUND if one is missing. */
vv_status_t vv_stream_ids_from_tokenizer(const vv_tokenizer_t* tok,
                                         vv_stream_ids_t* ids);

/**
 * @brief The session prompt, byte for byte what upstream tokenizes.
 *
 * "You are a helpful assistant that transcribes audio input into text
 * output. Please transcribe the following audios streamingly with these
 * keys: speaker, content\n", or with hotwords "... content and extra info:
 * {context_info}\n". No chat template and no special tokens.
 *
 * @return the length the full prompt needs (like snprintf); the output is
 *         truncated and NUL-terminated when it does not fit.
 */
size_t vv_stream_prompt_text(const char* context_info, char* buf, size_t cap);

/** @brief Marks a feature row in a chunk layout. */
#define VV_STREAM_ROW_FEATURE (-2)

/**
 * @brief Rows of one chunk prefill.
 *
 * `[lead?] <|object_ref_start|> feat × n_frames <|object_ref_end|>`, where
 * `lead` (when >= 0) is the previous chunk's pending `<|text_chunk_end|>`
 * folded into this prefill instead of costing a separate forward.
 *
 * @param rows         token id per row, VV_STREAM_ROW_FEATURE for features
 * @param cap          capacity of `rows`
 * @param feat_offset  row index of the first feature row (may be NULL)
 * @return rows written, or -1 when `cap` is too small or an id is missing
 */
int vv_stream_chunk_layout(const vv_stream_ids_t* ids, int32_t lead,
                           int n_frames, int32_t* rows, int cap,
                           int* feat_offset);

/* ─── UTF-8-safe text deltas ────────────────────────────────────────────── */

/**
 * @brief Turns the bytes of generated tokens into text deltas.
 *
 * Byte-level BPE splits multi-byte characters across tokens, so a token's
 * bytes are not always valid UTF-8 on their own. The emitter releases only
 * complete characters, decodes invalid bytes to U+FFFD the way Python's
 * `bytes.decode("utf-8", "replace")` does (one per maximal invalid
 * subpart), strips the control strings upstream removes from each chunk's
 * text (`<|text_chunk_end|>`, `<|object_ref_start|>`, ...) and holds back a
 * tail that could still become one of them. Concatenating every delta of a
 * chunk plus its flush gives exactly upstream's chunk text.
 */
typedef struct vv_stream_text vv_stream_text_t;

vv_status_t vv_stream_text_create(vv_stream_text_t** out);
void vv_stream_text_free(vv_stream_text_t* t);

/**
 * @brief Add one token's bytes; return what can be shown now.
 *
 * @param out      NUL-terminated delta, owned by the emitter, valid until
 *                 the next call. Empty when everything is held back.
 */
vv_status_t vv_stream_text_push(vv_stream_text_t* t, const char* bytes,
                                size_t n, const char** out, size_t* out_len);

/**
 * @brief End of a chunk: release everything held back.
 *
 * An incomplete UTF-8 sequence becomes U+FFFD (upstream decodes each chunk
 * on its own), and a partial control string is shown as the text it is.
 */
vv_status_t vv_stream_text_flush(vv_stream_text_t* t, const char** out,
                                 size_t* out_len);

/**
 * @brief Bytes of one token for the text stream, special tokens dropped.
 *
 * `vv_tokenizer_decode_ex(skip_special=true)` of a single id, which yields
 * raw (possibly partial UTF-8) bytes for byte-level tokens and nothing for
 * tokens flagged special in tokenizer.json.
 *
 * @param buf  receives the bytes (not NUL-terminated)
 * @param n    bytes written; VV_ERR_OVERFLOW if `cap` is too small
 */
vv_status_t vv_stream_token_bytes(const vv_tokenizer_t* tok, int32_t id,
                                  char* buf, size_t cap, size_t* n);

/* ─── Session ───────────────────────────────────────────────────────────── */

/** @brief Why a chunk's decode stopped. */
typedef enum {
    VV_STREAM_STOP_CHUNK_END  = 0,  /**< <|text_chunk_end|> */
    VV_STREAM_STOP_EOS        = 1,  /**< <|endoftext|>; upstream treats it
                                         like a chunk end and goes on */
    VV_STREAM_STOP_MAX_TOKENS = 2,  /**< per-chunk cap reached */
} vv_stream_stop_t;

typedef enum {
    VV_STREAM_EVENT_DELTA = 0,  /**< new text for the current chunk */
    VV_STREAM_EVENT_CHUNK = 1,  /**< a chunk finished; text = whole chunk */
    VV_STREAM_EVENT_DONE  = 2,  /**< session finished; text = transcript */
    VV_STREAM_EVENT_ERROR = 3,  /**< the session failed; see status */
} vv_stream_event_type_t;

/** @brief What the session reports through ::vv_stream_event_fn. */
typedef struct {
    vv_stream_event_type_t type;
    int64_t          chunk_index;
    const char*      text;         /**< NUL-terminated, valid in the call */
    size_t           text_len;
    int              n_tokens;     /**< CHUNK: tokens generated */
    vv_stream_stop_t stop;         /**< CHUNK */
    double           audio_start;  /**< seconds: the chunk covers */
    double           audio_end;    /**< [start, start + chunk) of new audio */
    vv_status_t      status;       /**< ERROR */
    /* CHUNK: where the chunk's time went, wall clock on the calling thread. */
    double           prefill_ms;   /**< window ready -> first token: encode,
                                        prefill, head */
    double           encode_ms;    /**< of which the encode, when measured */
    double           decode_ms;    /**< the greedy steps after it */
    int64_t          kv_len;       /**< cache positions after the chunk */
} vv_stream_event_t;

typedef void (*vv_stream_event_fn)(void* user, const vv_stream_event_t* ev);

/** @brief Most windows one encode_ahead call takes. */
#define VV_STREAM_AHEAD_MAX 8

/** @brief Describes one chunk prefill to the backend. */
typedef struct {
    int64_t        index;
    int64_t        start_sample;
    int64_t        n_valid;     /**< real samples; the rest of the window is 0 */
    int            n_frames;    /**< feature rows the window must encode to */
    const int32_t* rows;        /**< vv_stream_chunk_layout() output */
    int            n_rows;
    int            feat_offset; /**< first feature row */
    /** Where the backend may report how long the window's encode took, in
        ms (0 when it does not measure it separately). Never NULL. */
    double*        encode_ms;
} vv_stream_chunk_t;

/**
 * @brief The model work a session needs.
 *
 * Every function returns VV_OK or an error that ends the session. Token
 * outputs are greedy argmax ids. The KV cache persists across calls; the
 * backend is the only thing that touches it.
 */
typedef struct {
    void* self;

    /** Tokenize plain text (no special-token handling); ids are vv_alloc'd. */
    vv_status_t (*encode_text)(void* self, const char* text,
                               int32_t** ids, int* n);
    /** Text-stream bytes of one token, empty for special tokens. */
    vv_status_t (*token_bytes)(void* self, int32_t id,
                               char* buf, size_t cap, size_t* n);
    /** Prefill the prompt into an empty cache. */
    vv_status_t (*prefill_prompt)(void* self, const int32_t* ids, int n);
    /**
     * Encode `window` (window_samples floats) into `chunk->n_frames` rows,
     * lay out `chunk->rows` with the features at `feat_offset`, prefill it
     * at the current cache length, and return the argmax of the last row.
     */
    vv_status_t (*prefill_chunk)(void* self, const vv_stream_chunk_t* chunk,
                                 const float* window, int32_t* first);
    /** Feed one token at the current length and return the next argmax. */
    vv_status_t (*decode_step)(void* self, int32_t token, int32_t* next);
    /**
     * Optional. Encode `n` (<= VV_STREAM_AHEAD_MAX) consecutive ready
     * windows, chunk indices `first`.., in one go, ahead of their prefills:
     * an uploaded file, or a live session that fell behind. `windows` holds
     * them back to back. The prefill_chunk calls that follow for those
     * chunks use the result instead of encoding again; the features must be
     * the same as encoding each window alone. `encode_ms` receives the
     * time taken (0 when not measured). NULL: every window is encoded in its
     * own prefill_chunk.
     */
    vv_status_t (*encode_ahead)(void* self, int64_t first, int n,
                                const float* windows, double* encode_ms);
    /**
     * Feed tokens whose logits nobody needs (the trailing
     * <|text_chunk_end|>). NULL: the session folds that token into the next
     * chunk's prefill as its first row instead.
     */
    vv_status_t (*append_tokens)(void* self, const int32_t* ids, int n);
    /** Positions in the cache now, and how many it can hold. */
    int64_t (*kv_len)(void* self);
    int64_t (*kv_capacity)(void* self);
    /** Called by vv_stream_close(). May be NULL. */
    void (*destroy)(void* self);
} vv_stream_backend_t;

/** @brief Session settings. */
typedef struct {
    vv_stream_geom_t   geom;
    vv_stream_ids_t    ids;
    const char*        context_info;   /**< hotwords, NULL for none */
    int                max_new_tokens; /**< per chunk; upstream default 256 */
    /**
     * Fold the pending <|text_chunk_end|> into the next chunk's prefill
     * (one forward fewer per chunk) instead of appending it at chunk end.
     * Mathematically the same cache; forced on when the backend has no
     * append_tokens.
     */
    bool               fold_chunk_end;
    /**
     * Admission against a KV page pool shared by several slots. The
     * session maps pages for about this many seconds of audio when it
     * opens, and vv_stream_open() fails with VV_ERR_KV_POOL_EXHAUSTED when
     * the pool cannot cover them -- so a server refuses a new session up
     * front rather than letting it squeeze the ones already running.
     * Capped at the KV window. 0: map as the session grows. No effect on a
     * cache of its own (one slot) or on the CPU.
     */
    double             kv_reserve_sec;
    /**
     * Never wait for pages. Past the reservation a session takes pages as
     * it goes; with this set, a pool that has none free ends the session
     * with VV_ERR_KV_POOL_EXHAUSTED (an ERROR event) instead of blocking
     * it until another slot finishes -- which for a live stream may be
     * never, and meanwhile nobody reads its socket.
     */
    bool               kv_no_wait;
    vv_stream_event_fn on_event;
    void*              user;
} vv_stream_params_t;

/** @brief Streaming-7B geometry and ids, 256 tokens per chunk, folding on. */
void vv_stream_params_default(vv_stream_params_t* p);

/** @brief Counters for a session. */
typedef struct {
    int64_t chunks;
    int64_t tokens;         /**< generated text tokens, stops excluded */
    int64_t kv_len;         /**< cache positions in use */
    int64_t kv_capacity;
    int64_t samples;        /**< PCM pushed */
    int64_t prompt_tokens;
    int64_t prefill_rows;   /**< chunk rows prefilled, all chunks */
    double  prompt_ms;      /**< prompt prefill */
    double  prefill_ms;     /**< sum of the chunks' prefill_ms */
    double  encode_ms;      /**< sum of the chunks' encode_ms */
    double  decode_ms;      /**< sum of the chunks' decode_ms */
    double  max_chunk_ms;   /**< slowest chunk, prefill + decode */
} vv_stream_stats_t;

typedef struct vv_stream vv_stream_t;

/**
 * @brief Open a session over an inference context.
 *
 * The context must hold a chunked model (family `asr-streaming-7b`); the
 * geometry and the special ids come from the model, overriding those in
 * `params`, and everything else (hotwords, token cap, callbacks) from
 * `params`. The context is the session's until vv_stream_close(): the KV
 * cache is reset here and grows with every chunk, and nothing else may
 * transcribe on it meanwhile. All calls must come from one thread at a
 * time; on a GPU context each call binds the context's device first.
 *
 * @return VV_ERR_UNSUPPORTED for a one-shot model, VV_ERR_OVERFLOW when the
 *         prompt alone does not fit the KV window.
 */
vv_status_t vv_stream_open(struct vv_inference_ctx* ctx,
                           const vv_stream_params_t* params,
                           vv_stream_t** out);

/**
 * @brief Fill `p` with the defaults for `ctx`'s model: its chunk geometry
 *        and special ids, 256 tokens per chunk.
 */
vv_status_t vv_stream_params_for(const struct vv_inference_ctx* ctx,
                                 vv_stream_params_t* p);

/**
 * @brief Ask a session to stop. Safe from another thread (the flag is an
 *        atomic) or from inside the event callback (e.g. when a write to a
 *        disconnected client fails).
 *
 * The session checks between tokens and between chunks; the call in
 * progress returns VV_ERR_CANCELLED and so does every later one. No ERROR
 * event is emitted for a cancel.
 */
void vv_stream_cancel(vv_stream_t* s);

/**
 * @brief Transcribe a whole clip through a session on `ctx`.
 *
 * What vv_inference_transcribe() does for a chunked model: every window,
 * then the tail. Segments are the speaker turns the model writes inline
 * (" 
 Speaker N:"), timed to the chunks they came from. `on_event` may be
 * NULL; with it the chunks can be shown as they are produced.
 */
vv_status_t vv_stream_transcribe(struct vv_inference_ctx* ctx,
                                 const float* pcm24k, int n_samples,
                                 const char* context_info,
                                 vv_stream_event_fn on_event, void* user,
                                 struct vv_transcription** result);

/**
 * @brief Open a session over any backend (copied; destroyed on close).
 *
 * Tokenizes and prefills the prompt before returning.
 */
vv_status_t vv_stream_open_backend(const vv_stream_backend_t* backend,
                                   const vv_stream_params_t* params,
                                   vv_stream_t** out);

/**
 * @brief Push 24 kHz mono PCM; runs every chunk that becomes ready.
 *
 * Events arrive through the callback on the calling thread before this
 * returns. After an error every call returns that error.
 */
vv_status_t vv_stream_push(vv_stream_t* s, const float* pcm, size_t n);

/** @brief End of audio: run the zero-padded tail windows, emit DONE. */
vv_status_t vv_stream_finish(vv_stream_t* s);

/**
 * @brief Build a transcription (speaker-turn segments) from chunk texts.
 *
 * `texts[i]` is chunk i's text; chunk i covers [i * chunk_sec,
 * (i + 1) * chunk_sec) clamped to `duration`. A turn starts at a newline
 * followed by "Speaker N:"; text before the first marker goes to
 * "Speaker 0".
 */
vv_status_t vv_stream_build_transcription(const char* const* texts,
                                          int n_chunks, double chunk_sec,
                                          double duration,
                                          struct vv_transcription** out);

/** @brief Full transcript so far (concatenated chunk texts). */
const char* vv_stream_transcript(const vv_stream_t* s);

void vv_stream_get_stats(const vv_stream_t* s, vv_stream_stats_t* out);

/** @brief Free the session and destroy its backend. NULL is fine. */
void vv_stream_close(vv_stream_t* s);

#ifdef __cplusplus
}
#endif

#endif /* VV_STREAM_H */
