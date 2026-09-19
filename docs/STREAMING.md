# Streaming transcription (VibeVoice-ASR-Streaming-7B)

[`microsoft/VibeVoice-ASR-Streaming-7B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)
writes text while the audio is still arriving: one piece of text per 2.93 s
chunk, with 0.53 s of lookahead. This page covers:

1. The protocol, exactly as upstream `streaming_generate` runs it. It was
   checked against upstream commit `1541f59` running on gpubox.
2. The reference dumps and the C-side diff (`tools/compare_ref.py
   dump-stream` / `cmp-stream`).
3. The C session API (`include/vibevoice/stream.h`), the backend over an
   inference context, and the CLI and server built on it.
4. The server protocol.
5. Results: parity with upstream, latency, sessions per GPU.

Tracking issue: #20.

**Status.** The model runs end to end on GPU and CPU, from BF16 as-is or
quantized at load (`--quant int4|int8|nf4`). With `--quant none` every
chunk's token ids equal upstream `streaming_generate` on jfk, test30 and
test120 (156/156 chunks). `vv_cli --audio` prints text chunk by chunk,
`mic` streams live, `serve` answers `stream=true` with SSE and takes live
PCM over a WebSocket. See [section 5](#5-results) for the numbers.

```bash
vv_cli --model ./VibeVoice-ASR-Streaming-7B --quant int4 --audio talk.wav
vv_cli mic --model ./VibeVoice-ASR-Streaming-7B --quant int4 [--timestamps]
vv_cli serve --model ./VibeVoice-ASR-Streaming-7B --quant int4 --slots 16
python tools/stream_client.py --url ws://127.0.0.1:8080/v1/audio/stream \
    --audio talk.wav                     # live PCM, paced to real time
```

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
into a fresh cache) is not implemented.

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

# the exact PCM the reference used, as a float32 WAV (see audio24k.f32)
VV_DUMP_DIR=cdump vv_cli --model <streaming-7b> --quant none --audio jfk.wav
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

The C side (when `VV_DUMP_DIR` is set) writes:

| file | content |
|---|---|
| `stream_prompt_ids.bin` | int32 |
| `stream_cNNNN_ids.bin` | int32 `[count, ids...]`, the stop id excluded (so an empty chunk still leaves a file) |
| `stream_cNNNN_text.bin` | UTF-8 chunk text (absent when the text is empty) |
| `stream_cNNNN_feats.bin` | float32 `[26, hidden]`, the chunk's feature rows (backend) |
| `stream_cNNNN_logits.bin` | float32 `[vocab]`, the first token's logits (GPU backend) |

The session writes the first three, the backend the last two.

`cmp-stream` reports:

- prompt ids: identical or not;
- per chunk: ids identical, text identical, and the first diverging token;
- for chunks that diverge (every chunk with `-v`): feature and logit
  max / rel / cosine, top-1 agreement and top-5 overlap.

It stops at the first diverging chunk, because every later chunk runs on a
different history. `--all` goes on anyway. When the C side wrote every
chunk it also compares the joined transcript: identical or not, word edits,
and a normalized WER (case and punctuation ignored).

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
vv_stream_open(ctx, &p, &s);         /* tests use _open_backend */
while (have_audio) vv_stream_push(s, pcm24k, n);   /* events fire inline */
vv_stream_finish(s);                 /* tail windows, then DONE */
vv_stream_close(s);
```

### The backend over an inference context (`src/inference/stream_ctx.c`)

`vv_stream_open(ctx, ...)` takes the geometry and the special ids from the
model's family and builds a backend over the context. The context belongs to
the session until it closes.

- **Prompt**: the caches of the primary and of every shard are reset, the
  31 prompt ids embedded and prefilled.
- **Chunk**: the token rows (`[<|text_chunk_end|>] <|object_ref_start|> ...
  <|object_ref_end|>`, features as `<|box_start|>` placeholders) are
  embedded into a buffer allocated at open. The window goes to the device's
  speech front end as a *stateless* job (no encoder state, as upstream
  encodes each window), and the connectors write the 26 feature rows straight
  into that buffer, after the embedding (an event, not a host sync). The 29
  rows are prefilled at `kv->current_len` through every shard, the last row
  goes through the final norm and the LM head, and its argmax is the first
  token. The device lengths are then published for decode.
- **Decode**: the same captured step as a one-shot transcription
  (`pipeline_internal.h` exports it, so there is one implementation). The
  captures live in the session and read the position from the device, so
  they survive every prefill in between; a session re-captures only when the
  attention launch shape moves, a few dozen times an hour. The token the
  argmax leaves on the device is embedded from there.
- **The chunk end** is folded into the next chunk's prefill as its first row
  (the same cache, one forward fewer).
- **Several windows ready at once** (a file, or a live session that fell
  behind): their encodes go to the front end together first, up to 8, which
  it packs into shared launches, and the chunks then copy their rows from
  there. Batched windows are bit-identical to single ones, so this changes
  timing only. `VV_STREAM_AHEAD=0` turns it off.
- **Paged KV**: a chunk reserves its rows plus the chunk end before it runs;
  decode steps take pages one at a time. Pages go back to the pool when the
  session closes. Two parameters make a shared pool safe for live sessions:
  `kv_reserve_sec` maps pages for that much audio when the session opens,
  all of them or none (`VV_ERR_KV_POOL_EXHAUSTED` from `vv_stream_open`, so
  a server refuses the session up front instead of letting it squeeze the
  ones already running), and `kv_no_wait` makes a pool that has run dry
  later end the session instead of blocking it until another slot finishes
  -- for a live stream that may be never, and meanwhile nobody reads its
  socket. Without them (the CLI, `vv_stream_transcribe`) a chunk waits for
  pages as a batch request does.
- **Buffers**: every device buffer, the prompt's rows included, is
  allocated at open, and a closed session parks them on its context for the
  next one (`ctx->stream_bufs`). Opening and closing sessions in `serve`
  does not `cudaMalloc` or `cudaFree`, either of which would synchronise
  the device under every other live session. Token text is decoded into a
  buffer, not allocated per token.
- **Refusal**: the session checks the KV window before every chunk and every
  step and ends with `VV_ERR_OVERFLOW` (an ERROR event, and the server tells
  the client why) instead of writing past it. A pool that cannot supply
  pages ends it with `VV_ERR_KV_POOL_EXHAUSTED`.
- **Cancel**: `vv_stream_cancel()` from any thread (the flag is an atomic),
  or from inside the event callback; the session stops at the next token.
- **Head**: the final norm, LM head and argmax of a step are the batch
  loop's own (`vv_pipeline_head_argmax`), not a copy.
- **CPU**: the same sequence over the host kernels (`--cpu`): the window
  through both CPU encoders and connectors, `vv_decoder_prefill_cpu` at the
  current length, the fused host head.
- `VV_STREAM_PROFILE=1` syncs after each encode to report its time
  separately.

`vv_inference_transcribe()` on a streaming model runs a whole clip through
a session (`vv_stream_transcribe()`), so `vv_cli`, `chat`, the engine and
the non-streaming server path all work. The segments are the speaker turns
the model writes inline (`" \n Speaker N:"`), timed to the chunks they came
from, and `full_text` joins them.

The engine holds one slot for a session's life (`vv_engine_stream_open`,
`_push`, `_finish`, `_close`), resampling any input rate on the fly with a
resampler that gives exactly what the whole-file resample gives. Sessions
count against `--slots` and the server queue. `vv_engine_stream_open_ex`
waits at most a given time for a slot and returns `VV_ERR_BUSY` when none
freed up.

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
- A client that disconnects cancels the session at the next token: a failed
  write, or between chunks (when nothing is written) a non-blocking look at
  the socket once per token (`vv_http_peer_gone`). The slot is released.
- The upload's whole duration of KV is reserved when the session opens; a
  shared pool that cannot cover it answers `503 server_overloaded` before
  any event, instead of stalling the stream halfway.

A file upload is already complete when the handler runs, so the whole file
is pushed at once. The chunks come out as fast as the GPU produces them;
the stream is paced by compute, not by real time.

### `GET /v1/audio/stream` (WebSocket) for live PCM

A normal RFC 6455 upgrade, with a bearer token in `Authorization` or
`?api_key=` for browsers.

Client to server:

- A text frame first:
  `{"type":"session.start","sample_rate":16000,"format":"pcm_s16le"|"pcm_f32le","hotwords":"A,B"}`.
  Any rate other than 24000 goes through a streaming resampler that
  reproduces the whole-file resample sample for sample. The server answers
  `{"type":"session.started",...}` once the session holds a slot.
  `sample_rate` outside 8000..192000 (checked on the number before any
  conversion) is an error and close 1008.
- Then binary frames of raw PCM, any size.
- `{"type":"session.finish"}` flushes the zero-padded tail. Closing the
  socket instead abandons the tail.

Server to client (text frames):

- `{"type":"transcript.text.delta","delta":"...","chunk":n,"start":s,"end":e}`
- `{"type":"transcript.text.done","text":"..."}`
- `{"type":"error","error":{...}}`

Then a close frame with 1000; 1008 for a protocol or policy violation
(bad message, idle, trickle); 1013 "try again later" when the KV pool
cannot take the session; 1011 on an internal error.

Admission. A live session holds a slot for as long as its audio lasts, so
it is not queued behind other holders the way a batch request is:

- The upgrade is answered only when a slot is free within
  `--stream-slot-wait` (default 5 s) and no request is queued ahead of it;
  otherwise `503 server_overloaded`, before the upgrade.
- `session.start` reserves `--stream-reserve` seconds of KV (default 180)
  in the shared page pool. A pool that cannot cover it refuses the session
  (error `server_overloaded`, close 1013). Past the reservation a session
  keeps taking pages while the pool has them, and ends with the same error
  when it runs dry, rather than blocking on its own socket thread.
- Idle time counts from the last audio or JSON message, not the last byte:
  pings and pongs are answered but do not keep a session. After
  `--stream-idle` (default 60 s) without either it is closed (1008).
- A trickle does not keep it either: past one idle window a session must
  have sent at least a tenth of the wall time it has held the slot for
  (a live microphone sends all of it), or it is closed (1008).

- Latency is one window: 3.47 s of audio before the first chunk can run,
  plus the chunk's compute (about 0.1 s with `--quant int4`, section 5).
- The socket thread reads frames and pushes PCM into the session. The
  session runs in the same thread, because a chunk's compute (tens of ms)
  is far shorter than the 2.93 s between chunks. The kernel socket buffer
  absorbs the gap.
- If the session falls behind real time, which only happens when a GPU is
  shared across too many sessions, the chunker queues windows and their
  encodes go out together.
- A client that closes the socket abandons its tail and releases its slot,
  at the next token (a look at the socket per token) rather than at the next
  chunk.

### What `http.c` needs

| need | status |
|---|---|
| Streaming response: status + headers without Content-Length, then writes, delimited by the close | **done**: `vv_http_respond_begin`, `vv_http_write`, `vv_http_sse_send` (end of `http.c`, additive) |
| SSE event framing (multi-line data, `event:` field) | **done**: `vv_sse_format` (`src/server/ws.c`) |
| WebSocket upgrade: validate `Upgrade` / `Connection` / `Sec-WebSocket-Version: 13`, `Sec-WebSocket-Accept` = base64(SHA-1(key + GUID)), 101 | **done**: `vv_http_ws_accept`, `vv_ws_accept_key`, `vv_sha1` |
| WebSocket frames: incremental parser (masking, 16/64-bit lengths, fragmentation, control frames interleaved, size cap, protocol errors → close code), server frame header, send | **done**: `vv_ws_parser_t`, `vv_ws_frame_header`, `vv_http_ws_send` |
| Read after the request (frames) with a timeout | **done**: `vv_http_read`. Bytes that arrived with the upgrade request are in `req->body` and are fed first. |
| Disconnect detection | **done**: `vv_http_write` returns false, and `MSG_NOSIGNAL` means no SIGPIPE; between writes `vv_http_peer_gone` (a zero-timeout `select` + `MSG_PEEK`) |
| Incremental request-body read (a chunked or long upload feeding a session while it arrives) | Not done. `read_request` buffers the whole body; live input goes over the WebSocket instead. |
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

`tests/test_stream_server.c` drives `serve`'s session logic on a real
Streaming-7B engine over loopback (POSIX, `VV_TEST_STREAM_E2E=1`): a
`session.start` fragmented around a ping, 4801-byte frames against
96000-byte ones (same transcript), audio before `session.start` (1008), a
ping-only holder of the only slot closed after the idle time while a second
client gets 503 after the slot wait, a trickle closed too, and a client
that drops mid-stream giving its slot back. `tests/test_queue.c` covers
`vv_queue_try_enter`.

---

## 5. Results

All on gpubox (RTX 3090, Ryzen 9 5900X), against the reference dumps in
section 2 (upstream BF16, greedy, acoustic mean). Timings are the best of
three runs unless marked otherwise.

### Parity with upstream `streaming_generate`

`tools/compare_ref.py cmp-stream --all` over a `VV_DUMP_DIR` dump. "WER"
is over lower-cased words with punctuation dropped.

| weights | jfk (4 chunks) | test30 (11) | test120 (41) | transcript |
|---|---|---|---|---|
| `--quant none` (FP16 dense), GPU | **4/4** ids identical | **11/11** | **41/41** | identical |
| `--quant none`, `--cpu` | **4/4** | – | – | identical |
| `--quant int8` | 4/4 | 11/11 | 41/41 | identical |
| `--quant int4` (W4A16) | 4/4 | 11/11 | 39/41, from chunk 16 | identical |
| `--quant int4`, `--cpu` | 4/4 | – | – | identical |
| `--quant int8`, `--cpu` | 4/4 | – | – | identical |
| `--quant nf4` | 4/4 | 8/11, from chunk 5 | 27/41, from chunk 5 | same words, punctuation differs (WER 0.00%) |

- On int4 test120 the word "not" moves across the boundary between chunks
  16 and 17 ("ask" | "not what" becomes "ask not" | "what"); the joined
  transcript is the same.
- nf4 writes "thick, peppered, flour-fattened" where the reference has
  "thick peppered flour fattened": 4 and 24 raw word edits on test30 and
  test120, none after normalisation.
- On the 300 s excerpt of long30m, `--quant none` is 101/103 chunks
  identical, with the same one-word boundary move in chunk 87, and the same
  transcript.
- KV formats with int4 weights, test120: `--kv-cache fp8` is 39/41 with the
  same transcript; `tq4` is 36/41 with 3 word edits (WER 0.71%).
- The table is under `auto`, which is flashinfer for this family (below).
  `--attn fa2` gives the same ids on every row checked: with `--quant none`
  156/156, and with int4 the output JSON is byte-identical on test120 and
  on the 32-minute long30m.
- `test_stream` with the model: the emitter rebuilds 159/159 reference chunk
  texts from the reference ids, and a live session (PCM pushed in small
  blocks) reproduces jfk and test30 chunk for chunk.

### One stream

`vv_cli --audio test120.wav` (120 s, 41 chunks). "Per chunk" is encode +
prefill + decode of one chunk; the encode is 22-29 ms of it with one window
per launch, less when ready windows share a launch.

| weights | per chunk, mean (p95) | RTF | decode | VRAM |
|---|---|---|---|---|
| `--quant none` (FP16) | 306 ms (379) | 0.106 | 55 tok/s | 18.6 GB |
| `--quant int4` | **92 ms** (135) | **0.032** | 155 tok/s | 9.8 GB |
| `--quant int4`, `--attn fa2` | 96 ms (137) | 0.033 | 148 tok/s | 9.8 GB |
| `--quant int8` | 259 ms | 0.090 | – | 12.5 GB |
| `--quant nf4` | 263 ms (313) | 0.091 | 97 tok/s | 9.8 GB |
| `--quant int4`, long30m (32 min, 654 chunks) | **114 ms** (191) | **0.042** | 134 tok/s | 9.8 GB |
| `--quant int4`, long30m, `--attn fa2` | 134 ms (214) | 0.048 | 120 tok/s | 9.8 GB |
| `--quant none`, 300 s excerpt | 342 ms (455) | 0.118 | 52 tok/s | 18.6 GB |

`auto` attention is flashinfer for this family (fa2 for the batch model).
A chunk prefills 29 rows on top of a cache that keeps growing; fa2 runs
them as one packed-rows kernel -- 7 query heads per KV head make 203 rows,
4 KV heads x 4 row tiles = 16 blocks on an 82-SM card, each walking the
whole cache -- while flashinfer splits the cache across the idle SMs, and
its decode is on tensor cores. The gap grows with the session: 4% on
test120, 15% over 32 minutes. The output JSON of both is byte-identical on
test120 and on long30m, and `--quant none` under flashinfer still matches
upstream 156/156 chunks. The batch model stays on fa2 because its
transcripts are pinned bit for bit to the old kernels; this family has no
such history.

The int8, nf4 and FP16 rows predate encoding ready windows ahead, which
took about 11 ms off each int4 chunk, and the flashinfer default; they
would move by about as much. Only int4 uses kernels built for a 29-row
prefill: the other formats run it through GEMMs sized for long prompts,
which is most of the gap.

CPU (`--cpu`, 12 cores, jfk, one run each): int4 2.8 s per chunk, RTF 1.09;
int8 4.2 s, RTF 1.58; FP16 4.4 s, RTF 1.68. It works, but does not keep up
with real time: a chunk prefill on the CPU runs at about 20 tok/s against
77 tok/s for a long batch prompt.

### Sessions per 3090

`vv_cli serve --quant int4 --slots 48`, `tools/stream_client.py --sessions N
--stagger 0.37`: N WebSocket clients each stream test120 at wall-clock speed.
Latency is from sending a window's last sample to receiving its text.

| sessions | latency p50 | p95 | max | keeps up | texts identical to one session |
|---|---|---|---|---|---|
| 1 | 116 ms | 138 ms | 149 ms | yes | yes |
| 16 | 163 ms | 205 ms | 214 ms | yes | yes |
| 24 | 213 ms | 264 ms | 279 ms | yes | yes |
| 32 | 297 ms | 1.1 s | 2.2 s | yes, just (all done 0.7 s after the audio) | yes |
| 40 | 11.3 s | 18.9 s | 21.5 s | no, 21 s behind by the end | yes |
| 48 | 17.8 s | 33.3 s | 42.3 s | no | yes |

So one 3090 carries about **24 live int4 streams** with sub-300 ms chunk
latency, and about 32 at the edge; past that the queue grows. Every session
produced the same text as a session alone, with no errors.

Memory per session is a 256 MB workspace plus KV at about 45 MB per minute
of audio (FP16) from the shared page pool. With 48 slots the server sizes
the pool to what the card has left: 1261 pages, 80704 positions, 4.4 GB,
enough for 49 two-minute streams at once. Before the budget charged
streaming slots at their real 256 MB workspace, 32 slots got a 458 MB pool
and live sessions past the fifth failed with "KV page pool exhausted".
