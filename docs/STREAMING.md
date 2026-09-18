# Streaming transcription (VibeVoice-ASR-Streaming-7B)

[`microsoft/VibeVoice-ASR-Streaming-7B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)
writes text while the audio is still arriving: one piece of text per 2.93 s
chunk, with 0.53 s of lookahead. This page covers three things:

1. The protocol, exactly as upstream `streaming_generate` runs it. It was
   checked against upstream commit `1541f59` running on gpubox.
2. The reference dumps and the C-side diff (`tools/compare_ref.py
   dump-stream` / `cmp-stream`).
3. The C session API (`include/vibevoice/stream.h`) and the server protocol
   built on it.

Tracking issue: #20.

**Status.** Phase 1 is done: the reference, the protocol, and the model-free
C pieces with tests. Phase 2 is the runtime backend. It needs three things
that other branches are building:

- BF16 loading and the `asr-streaming-7b` family (#15);
- the stateless-window encoder (#16);
- prefill at `kv->current_len` with the split-KV small-q attention path (#17).

Until then `vv_stream_open()` returns `VV_ERR_UNSUPPORTED`.

---

## 1. Protocol

### The checkpoint

| | |
|---|---|
| LLM | Qwen2-7B: 28 layers, hidden size 3584, 28 query heads and 4 KV heads, vocabulary 152064. Same shapes as the batch model. |
| Speech encoders | Same architecture and config as the batch model. Both are `causal: true`, acoustic `vae_dim` is 64, semantic is 128. |
| Weights | BF16, 8 safetensors shards. The upstream class also carries a diffusion head (`diffusion_head_config`); ASR does not use it. |
| `preprocessor_config.json` | `target_sample_rate` 24000, `speech_tok_compress_ratio` 3200, **`normalize_audio` false**, `chunk_frames` 22, `lookahead_frames` 4 |
| Tokenizer | The checkpoint ships its own `tokenizer.json`: 151643 vocabulary entries and 26 added tokens. Our BPE loads it as-is. |

Special ids (all flagged `special` in `tokenizer.json`):

| token | id | role |
|---|---|---|
| `<\|endoftext\|>` | 151643 | EOS (`eos_token`). `<\|im_end\|>` plays no part. |
| `<\|object_ref_start\|>` | 151646 | speech start |
| `<\|object_ref_end\|>` | 151647 | speech end |
| `<\|text_chunk_end\|>` | 151665 | ends a chunk's text |

### Audio

The demo loads audio with `load_audio_use_ffmpeg(resample=True,
target_sr=24000)`, which gives mono float32 at 24 kHz. **Nothing normalises
it.** This differs from the batch model, which rescales the whole clip to
−25 dBFS. It also means live input needs no running-RMS workaround: raw PCM
goes straight in.

### Prompt

The prompt is plain text. There is no chat template and no special tokens,
and it is tokenized with `add_special_tokens=False`:

```
You are a helpful assistant that transcribes audio input into text output. Please transcribe the following audios streamingly with these keys: speaker, content\n
```

With hotwords, `" and extra info: {context_info}"` goes before the final
`\n`. An empty string counts as no hotwords, because upstream tests
`if context_info:`. Without hotwords the prompt is 31 tokens:

```
2610 525 264 10950 17847 429 1356 55136 7699 1946 1119 1467 2550 13 5209
1356 3114 279 2701 6136 3530 4269 11307 448 1493 6894 25 18601 11 2213 198
```

`vv_stream_prompt_text()` builds this text byte for byte. The C tokenizer
produces the same ids; `test_stream` checks this when `VV_TEST_STREAM_MODEL`
is set.

### Window geometry

```
frame        = 3200 samples (7.5 Hz)
chunk        = chunk_frames     × frame = 22 × 3200 = 70400 samples (2.933 s)
lookahead    = lookahead_frames × frame =  4 × 3200 = 12800 samples (0.533 s)
window       = chunk + lookahead        = 83200 samples = 26 frames

start = 0
while start < total:
    w = pcm[start : min(start + window, total)]
    if len(w) < window: w = zero-pad at the end to window   # pad_last_chunk
    feats = encode_speech(w)          # 26 × 3584
    start += chunk
```

- The number of windows is `ceil(total / 70400)`: 4 for jfk (11 s), 11 for
  test30, 41 for test120, and 103 for a 300 s excerpt.
- The demo computes the geometry in seconds and upstream converts back to
  samples with `int()`. With 22 and 4 frames the float math comes out exact
  (70400.0 and 12800.0). `tools/compare_ref.py stream_geometry()` replays the
  same expressions in case a future checkpoint picks counts where that
  truncation matters.
- **Each window is encoded on its own**: `encode_speech()` on an 83200-sample
  tensor, which is under 60 s and so takes the non-cached path, with fresh
  conv state. Nothing carries over between windows. Consecutive windows
  overlap by 12800 samples, so the lookahead audio is encoded twice.
- Features are `acoustic_connector(acoustic_latent) +
  semantic_connector(semantic_mean)`. Upstream draws the acoustic latent from
  its gaussian: `std_dist_type` is `gaussian` and `fix_std` is 0.5, so the
  draw is a per-batch scale `randn × 0.625` times `randn_like`. **The
  reference used here takes the mean**, the same as `tools/compare_ref.py
  dump` does for the batch model and as the C runtime does by default.
- `streaming_generate` also has an `encode_mode="encode_then_split"` that
  encodes the whole clip once and slices the features. It is not the
  default and the demo does not use it.

### Per chunk

The positions continue from the KV length. The prompt fills positions 0–30,
so chunk 0 starts at position 31.

1. Prefill `[<|object_ref_start|>, 26 feature rows, <|object_ref_end|>]`, 28
   rows, on top of the cache. The argmax of the last row is the first token.
2. Greedy loop, up to `max_new_tokens_per_chunk` (default 256):
   - if the token is `<|text_chunk_end|>` or `<|endoftext|>`, stop;
   - otherwise keep it and feed it (one decode step), and take the next
     argmax.

   When the cap is reached, the last kept token has still been fed.
3. **Always** feed `<|text_chunk_end|>` afterwards, whatever stopped the loop.
   **EOS does not end the session**: upstream handles it exactly like a
   chunk end and moves on to the next window.
4. Chunk text: `tokenizer.decode(chunk_tokens, skip_special_tokens=True)`
   (`errors="replace"`), then remove these literal strings:
   `<|text_chunk_end|>`, `<|object_ref_start|>`, `<|object_ref_end|>`,
   `<|box_start|>`, `<|speech_start|>`, `<|speech_end|>`, `<|speech_pad|>`.
   Each chunk is decoded **independently**, so a multi-byte character split
   across two chunks would decode as U+FFFD in both.
5. Transcript: `"".join(chunk_texts)`. Speaker turns show up inline as
   `" \n Speaker N:"`. For example, jfk gives
   `" \n Speaker 0:And so, my fellow Americans, "` +
   `"ask not what your "` + `"country can do for you, ask what "` +
   `"you can do for your country."`.

**Folding the chunk end.** Appending `<|text_chunk_end|>` and then
prefilling the next chunk puts the same tokens at the same positions as
prefilling `[<|text_chunk_end|>, <|object_ref_start|>, 26 feats,
<|object_ref_end|>]` in one go. The C session does the latter by default
(`fold_chunk_end`), which saves one forward per chunk. The results agree up
to the rounding difference between a batched row and a single-row GEMV.
After the last chunk the pending token is simply never fed.

### KV budget

Each chunk costs 28 rows, plus about 10 text tokens, plus 1 chunk end.
Measured with the reference (BF16, greedy, acoustic mean):

| clip | length | chunks | text tokens | tokens/chunk (max) | KV positions | positions/min |
|---|---|---|---|---|---|---|
| jfk | 11 s | 4 | 34 | 8.5 (13) | 181 | 987 |
| test30 | 30 s | 11 | 104 | 9.5 (17) | 454 | 908 |
| test120 | 120 s | 41 | 418 | 10.2 (17) | 1638 | 819 |
| long30m, first 300 s | 300 s | 103 | 1057 | 10.3 (18) | 4075 | 815 |

- Continuous speech settles at about **815 positions per minute**:
  20.45 chunks × 29 rows = 593, plus about 210 text tokens and the 31-token
  prompt.
- FP16 KV is 28 layers × 2 × 4 heads × 128 × 2 B = 56 KiB per position,
  which is about **45 MB per minute of audio**.
- A 32K window lasts about **40 minutes**, and 131072 positions (the model's
  maximum) about 2 h 40 min.
- With `--kv-cache tq4` the same 32K window costs 462 MB instead of 1.8 GB.
- No chunk came close to the 256-token cap; the longest had 18 tokens.

The session refuses a chunk that would not fit rather than writing past the
cache (`VV_ERR_OVERFLOW` plus an ERROR event). A rollover policy (re-prompt
into a fresh cache) is a phase-2 option.

### Reference speed (for scale, not a target)

Upstream in BF16 through transformers 4.51 with sdpa, on one 3090, takes
13.7 s for test120 (RTF 0.114) and 42.2 s for the 300 s excerpt
(RTF 0.141).

---

## 2. Reference tooling

```bash
# gpubox: reference venv, one 3090, about 17 GB of VRAM
gpu-run 1 /mnt/ssd0/vibevoice/.venv/bin/python tools/compare_ref.py dump-stream \
    --model /mnt/ssd0/vv3/models/vibevoice-asr-streaming-7b \
    --repo /mnt/ssd0/vibevoice/ref_vibevoice \
    --audio /mnt/ssd0/vibevoice/audio/jfk.wav --out ref/jfk \
    [--speech-dtype fp32|bf16] [--context-info "A,B"] \
    [--start S --duration D] [--check-upstream]

VV_DUMP_DIR=cdump vv_cli --model <streaming-7b> --audio jfk.wav   # phase 2
python tools/compare_ref.py cmp-stream --ref ref/jfk --c cdump [--all] [-v]
```

`dump-stream` runs the upstream loop step by step so it can save what each
chunk sees. It writes:

- `meta.json`: the geometry, the prompt text and ids, the special ids, and
  per chunk `start`, `end`, `pad`, the generated `ids`, `stop`
  (`text_chunk_end`, `eos` or `max_tokens`), `text`, and `kv_before` /
  `kv_after`. Also the transcript and timing.
- `audio24k.f32`: the exact PCM the reference used, raw float32. Feed the C
  side this and resampler differences are ruled out.
- `prompt_ids.npy`, `chunkNNNN_feats.npy` (26 × 3584 float32, after the
  connectors) and `chunkNNNN_logits.npy` (the first generated token's
  logits, float32 [152064]).
- `transcript.txt`.

Options:

- `--speech-dtype fp32` (the default) runs the encoders and connectors in
  FP32. That is the clean target the C encoder is held to, as with `dump`.
- `--speech-dtype bf16` runs exactly what upstream runs.
- `--check-upstream` also calls upstream `streaming_generate` itself, with
  the acoustic latent patched to the mean, and diffs the chunk texts.

The C side (phase 2, when `VV_DUMP_DIR` is set) writes:

| file | content |
|---|---|
| `stream_prompt_ids.bin` | int32 |
| `stream_cNNNN_ids.bin` | int32 `[count, ids...]`, the stop id excluded (so an empty chunk still leaves a file) |
| `stream_cNNNN_text.bin` | UTF-8 chunk text (absent when the text is empty) |
| `stream_cNNNN_feats.bin` | float32 `[26, hidden]` (backend) |
| `stream_cNNNN_logits.bin` | float32 `[vocab]`, the first token's logits (backend) |

The session already writes the first three. The last two belong to the
phase-2 backend.

`cmp-stream` reports:

- prompt ids: identical or not;
- per chunk: ids identical, text identical, and the first diverging token;
- for chunks that diverge (every chunk with `-v`): feature and logit
  max / rel / cosine, top-1 agreement and top-5 overlap.

It stops at the first diverging chunk, because every later chunk runs on a
different history. `--all` goes on anyway.

### Checks done on gpubox

The reference dumps live in `/mnt/ssd0/vv3/streaming-asr/ref/{jfk,test30,
test120,long5m}` (FP32 speech path), plus `jfk_bf16` and `test120_bf16`
(upstream precision, with `--check-upstream`).

- The replicated loop matches upstream `streaming_generate` on
  **jfk: 4/4 and test120: 41/41 chunk texts** (bf16).
- FP32 and BF16 speech paths give **identical token ids** on every chunk of
  jfk (4) and test120 (41). On jfk the feature cosine is 0.9992–0.9999 and
  the first-token logits cosine 0.9997–1.0000.
- The C text emitter reproduces all **159/159** reference chunk texts (jfk,
  test30, test120, long5m) from their token ids through the C tokenizer:
  `VV_TEST_STREAM_REF=ref/jfk:ref/test30:ref/test120:ref/long5m
  VV_TEST_STREAM_MODEL=<dir> build/test_stream`.
- Observation: on test120 and on long5m the model labels every turn
  `Speaker 0`.

---

## 3. C session API (`include/vibevoice/stream.h`)

The C side is split into pieces, and each one is tested in `tests/test_stream.c`
without weights:

| piece | what it does |
|---|---|
| `vv_stream_geom_t` | Geometry. `vv_stream_geom_default()` gives 22 + 4 frames at 24 kHz. |
| `vv_stream_chunker_t` | Takes pushed PCM of any size, including 0 and 1 sample, and yields full windows as soon as all of their samples are there. After `finish()` it yields the zero-padded tail windows. It keeps only what the next window still needs. Tested against upstream's loop for 11 lengths × 4 push patterns. |
| `vv_stream_chunk_layout()` | Builds the prefill rows `[lead?] <\|object_ref_start\|> feat×26 <\|object_ref_end\|>`. |
| `vv_stream_prompt_text()` | Builds the prompt, byte for byte. |
| `vv_stream_text_t` | Turns token bytes into text deltas: complete UTF-8 only, CPython `errors="replace"` semantics for invalid bytes, upstream's strip list, and a held-back tail that might still become a stripped string. The joined deltas of a chunk equal upstream's chunk text for any split. |
| `vv_stream_t` | The session. Takes a `vv_stream_backend_t` and works as below. |

The session:

- tokenizes and prefills the prompt;
- runs every ready window through `prefill_chunk` and then `decode_step`
  until a stop;
- emits `DELTA` events per token, one `CHUNK` event per chunk and `DONE` at
  the end;
- folds or appends the chunk end;
- refuses a chunk before the cache would overflow.

```c
vv_stream_params_t p;
vv_stream_params_default(&p);        /* 22+4 frames, ids, 256 tok/chunk */
p.context_info = "Azure,VibeVoice";
p.on_event = on_event; p.user = &state;
vv_stream_t* s;
vv_stream_open(ctx, &p, &s);         /* phase 2; tests use _open_backend */
while (have_audio) vv_stream_push(s, pcm24k, n);   /* events fire inline */
vv_stream_finish(s);                 /* tail windows, then DONE */
vv_stream_close(s);
```

### Phase-2 hooks (`TODO(phase 2)` in `src/inference/stream.c`)

The backend over a `vv_inference_ctx_t` implements the following.

- `encode_text` / `token_bytes`: `ctx->tokenizer` through
  `vv_tokenizer_encode` and `vv_stream_token_bytes`.
- `prefill_prompt`: reset the KV cache (primary and shards), embed, and
  `vv_decoder_prefill`.
- `prefill_chunk`, in order:
  - Encode the window through the stateless-window batched encoder (#16),
    acoustic and semantic, with the acoustic latent as the mean.
  - Run the connectors on the device, writing straight into rows
    `feat_offset..` of the chunk's hidden buffer. Token rows go through the
    embedding.
  - Prefill at `kv->current_len` (#15). With about 29 query rows against a
    growing cache this wants the split-KV small-q attention path (#17).
  - Final norm on the last row, LM head, argmax.
  - Call `vv_kv_cache_publish_len` on every shard.
  - When dumping, write `stream_cNNNN_feats` / `_logits`.
- `decode_step`: the existing graphed step. Keep the `graph_slot_t` array
  and decode buffers in the session instead of on `transcribe_gpu`'s stack.
  The captures are position-invariant, so they survive across chunks. The
  replay needs the `kv->current_len < max_seq_len` guard (pipeline map H1).
- `kv_len` / `kv_capacity`: `kv->current_len` / `kv->max_seq_len`.
- Quantised KV: `k_ref` is built from the first append, which is the
  prompt. Either accept that or delay it until the first chunk. Parity has
  to be measured.
- Family plumbing (#15):
  - geometry from `preprocessor_config.json` (`chunk_frames`,
    `lookahead_frames`);
  - ids from `vv_stream_ids_from_tokenizer`;
  - `vv_cli --audio` on a streaming checkpoint pushes the file through a
    session and prints chunk by chunk;
  - `mic` pushes capture blocks straight into a session instead of cutting
    VAD utterances.

The engine holds one slot for a session's lifetime
(`vv_engine_session_open`). Sessions count against `--slots` and the server
queue.

---

## 4. Server protocol

### `POST /v1/audio/transcriptions` with `stream=true`

This follows OpenAI's streaming transcription events. The request is the
existing multipart upload plus the form field `stream=true`. The `model`,
`prompt` (hotwords), `language` and `temperature` fields keep their current
meaning.

The response is `200`, `Content-Type: text/event-stream`, with no
Content-Length; the body ends when the connection closes:

```
event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":" \n Speaker 0:And so, my fellow Americans, ","x_vibevoice":{"chunk":0,"start":0.0,"end":2.933}}

event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":"ask not what your ","x_vibevoice":{"chunk":1,"start":2.933,"end":5.867}}

...

event: transcript.text.done
data: {"type":"transcript.text.done","text":" \n Speaker 0:And so, ... your country.","x_vibevoice":{"chunks":4,"duration":11.0,"rtf":0.05}}
```

- A delta goes out when a chunk finishes (one per `CHUNK` event), or per
  token behind `stream_granularity=token`.
- The emitter already guarantees that every delta is valid UTF-8 with no
  control strings.
- On failure the server sends `event: error` with the OpenAI error envelope
  and closes.
- A failed write means the client disconnected. It cancels the session: the
  decode loop checks a flag between tokens and the slot is released.

A file upload is already complete when the handler runs, so the whole file
is pushed at once. The chunks come out as fast as the GPU produces them;
the stream is paced by compute, not by real time.

### `GET /v1/audio/stream` (WebSocket) for live PCM

A normal RFC 6455 upgrade, with a bearer token in `Authorization` or
`?api_key=` for browsers.

Client to server:

- A text frame first:
  `{"type":"session.start","sample_rate":16000,"format":"pcm_s16le"|"pcm_f32le","hotwords":"A,B"}`.
  Any rate other than 24000 goes through the existing resampler in
  fixed-ratio streaming mode.
- Then binary frames of raw PCM, any size.
- `{"type":"session.finish"}` flushes the zero-padded tail. Closing the
  socket instead abandons the tail.

Server to client (text frames):

- `{"type":"transcript.text.delta","delta":"...","chunk":n,"start":s,"end":e}`
- `{"type":"transcript.text.done","text":"..."}`
- `{"type":"error","error":{...}}`

Then a close frame with 1000, or 1011 on an internal error.

- Latency is one window: 3.47 s of audio before the first chunk can run,
  plus about 30–60 ms of GPU time per chunk (a phase-2 estimate).
- The socket thread reads frames and pushes PCM into the session. The
  session runs in the same thread, because a chunk's compute (tens of ms)
  is far shorter than the 2.93 s between chunks. The kernel socket buffer
  absorbs the gap.
- If the session falls behind real time, which only happens when a GPU is
  shared across too many sessions, the chunker simply queues windows.
  Phase 2 may batch consecutive windows through the encoder.

### What `http.c` needs

| need | status |
|---|---|
| Streaming response: status + headers without Content-Length, then writes, delimited by the close | **done**: `vv_http_respond_begin`, `vv_http_write`, `vv_http_sse_send` (end of `http.c`, additive) |
| SSE event framing (multi-line data, `event:` field) | **done**: `vv_sse_format` (`src/server/ws.c`) |
| WebSocket upgrade: validate `Upgrade` / `Connection` / `Sec-WebSocket-Version: 13`, `Sec-WebSocket-Accept` = base64(SHA-1(key + GUID)), 101 | **done**: `vv_http_ws_accept`, `vv_ws_accept_key`, `vv_sha1` |
| WebSocket frames: incremental parser (masking, 16/64-bit lengths, fragmentation, control frames interleaved, size cap, protocol errors → close code), server frame header, send | **done**: `vv_ws_parser_t`, `vv_ws_frame_header`, `vv_http_ws_send` |
| Read after the request (frames) with a timeout | **done**: `vv_http_read`. Bytes that arrived with the upgrade request are in `req->body` and are fed first. |
| Disconnect detection | **done**: `vv_http_write` returns false, and `MSG_NOSIGNAL` means no SIGPIPE |
| Incremental request-body read (a chunked or long upload feeding a session while it arrives) | **phase 2**. `read_request` buffers the whole body, which the security branch (#14) is reworking right now. Add a handler flag that makes the reader stop after the headers and hand over the socket. |
| Keep-alive, chunked transfer encoding | Not needed. Every streaming response closes its connection. |
| Text-frame UTF-8 validation (close 1007) | Not done. Control messages are parsed as JSON, which rejects garbage anyway. |

`tests/test_ws.c` covers:

- SHA-1 (FIPS vectors including the padding edges), base64, the RFC 6455
  accept key and the section 5.7 frame examples;
- every protocol violation, byte-by-byte vs whole feeding, and SSE
  formatting;
- a loopback round trip through the real server (POSIX): an SSE response,
  and a WebSocket handshake with a frame in the same packet, echoed back
  and then closed.

---

## Open questions for phase 2

- **Encoder precision.** Upstream runs the encoder in BF16 and ours runs in
  FP16/FP32. The FP32 and BF16 references agree on every token id on jfk
  and test120. Check again on the long excerpt with `cmp-stream`.
- **Quantised weights.** `--quant int4|int8|nf4` at load time (#15) changes
  numerics. Measure the transcript diff and first-token logits cosine
  against the BF16 reference per chunk.
- **CPU path.** The same session and backend over `transcribe_cpu`'s
  stages. Each chunk is 29 prefill rows and about 10 decode steps, so at
  7.9 tok/s CPU decode a chunk takes about 1.3 s of decode for 2.93 s of
  audio. Real time on a 5900X looks plausible (estimate).
