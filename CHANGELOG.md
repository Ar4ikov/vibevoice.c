# Changelog

Newest first. Versions follow [SemVer 2.0](https://semver.org): below 1.0 a
minor bump may break, a patch never does. The release number lives in
`VV_VERSION_STRING` in `include/vibevoice/vibevoice.h`; the Release workflow
bumps it, turns **Unreleased** into the release's section, tags `v<version>`
and publishes the image and binary -- see [docs/RELEASING.md](docs/RELEASING.md).
Write the entry for a change under Unreleased in the same pull request.

## Unreleased

- Speculative decoding with [DFlash 2](https://inco.ai/blog/dflash2/)
  drafters (`--draft <dir>`, also for `serve`, `chat` and streaming
  sessions): a small drafter proposes a block of tokens in one pass, the
  model checks the block in one pass and keeps what it agrees with. The
  check computes every row with a decode step's own arithmetic -- the
  tensor-core W4A16 GEMV below, the split-KV decode attention (fa1, fa2,
  flashinfer, every KV format, paged or not), the ternary BitNet
  projections, the LM head and the argmax, each tested bit for bit against
  one row at a time -- so a transcript with a drafter is byte-for-byte the
  transcript without one on the same attention backend (`--draft-check
  exact`, the default). With a drafter `--attn auto` is flashinfer, whose
  checked rows share their tiles where fa2 walks the cache once per row;
  the transcript is then that of `--attn flashinfer`, and `--attn fa2
  --draft` keeps fa2's. `--draft-check fast` checks with flashinfer's
  attention and the prefill's projections instead: the greedy transcript
  within rounding. NF4 and INT8 weights check exactly only one row at a
  time, which never pays, so there the drafter is declined unless the check
  is fast, which reads each weight once (the small-M kernel now takes 1..64
  rows). `--draft-block N` sets how many rows a pass checks (without it:
  half the drafter's block or all of it, whichever keeps more tokens per
  ms), `--draft-quant int4|f16` the drafter's weights (INT4 by default); a
  drafter may also ship its projections as INT4 (compressed-tensors,
  `tools/dflash/awq_drafter.py`). A cycle -- draft, check, accept, the
  drafter's context -- reads its positions on the device and runs as one
  captured graph. Blocks run only while they beat plain steps (measured).
  RTX 3090, AWQ 7B against its default decode: jfk 2.57x, test120 2.17x,
  a 32-minute file 1.73x, 20 held-out clips 2.70x; Streaming-1.5B: test120
  2.50x, 32 minutes 2.13x. Drafters for the AWQ checkpoints are on the Hub
  ([ASR-7B](https://huggingface.co/Ar4ikov/VibeVoice-ASR-DFlash2-Drafter),
  [Streaming-1.5B](https://huggingface.co/Ar4ikov/VibeVoice-ASR-Streaming-1.5B-DFlash2-Drafter),
  each also as `-AWQ-W4A16-ASYM`), and bundled with their models
  ([ASR-7B](https://huggingface.co/Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM-DFlash2),
  [Streaming-1.5B](https://huggingface.co/Ar4ikov/VibeVoice-ASR-Streaming-1.5B-AWQ-W4A16-ASYM-DFlash2)):
  a model directory's `drafter/` is used without `--draft` (`--draft none`
  turns it off). They are trained on the target's own
  transcripts (`tools/dflash`: corpus, `vv_dflash_data gen|trace`,
  `train.py`, `awq_drafter.py`, `check_drafter.py`); the design and the
  numbers are in [docs/DFLASH.md](docs/DFLASH.md). Metal and the CPU path
  decline (`VV_ERR_UNSUPPORTED`) and decode without the drafter
  ([#47](https://github.com/Ar4ikov/vibevoice.c/issues/47)).
- **W4A16 decoding changed its arithmetic** (AWQ, GPTQ, `--quant int4`).
  The step's GEMV is now a tensor-core kernel whose rows are independent
  of each other (`vv_w4a16_mv_dev`: the weights as the MMA's A operand, K cut
  into fixed slices summed in a fixed tree), which is what lets a drafted
  block be checked exactly at about one step's cost; it also runs prefills
  of up to 16 rows. Its sums are FP32 inside the MMA where the old GEMV
  summed FP16 chains of four, so its bits differ from 0.5.1's: on a 3090
  the AWQ 7B's transcripts of jfk, test30, test120 and a 32-minute file
  (9777 tokens) and the Streaming-1.5B's of jfk and test120 still came out
  character for character the same; plain decoding of the 7B is 1-2 %
  slower (a 1.5B layer's GEMVs take 10 % less). `VV_W4A16_MV=0` gives the
  old GEMV. Before sm_80 nothing changes.
- A checked block's flashinfer rows share their 16-row fragments (two or
  three rows for 7 heads), found on the device from the cache length: 8 rows
  at 24K positions cost 1.64 one-row decodes instead of 3.33, bit for bit
  the same.
- A failed `cudaMalloc` stayed the thread's last CUDA error, and the next
  kernel launch reported it as its own: a drafter that did not fit on the
  card ("decoding without it") took the transcription down with "CUDA
  kernel launch failed". Allocations now clear it.
- `vv_init_params_default()` now zero-fills the struct first: the per-device
  memory caps of an empty `gpus` set were left as stack garbage, so an API
  caller that did not go through the CLI's flag parser could get a random
  cap and land on the CPU-only placement with no error.

## [0.5.1](https://github.com/Ar4ikov/vibevoice.c/compare/v0.5.0...v0.5.1) — 2026-09-21

- A container that holds more `/dev/nvidiaN` nodes than
  `CUDA_VISIBLE_DEVICES` lists now says so in one warning, with the counts
  and where the knob is. GPUStack 2.2.2's automatic scheduler hands a custom
  backend one card whatever `--gpus all` asks, mounts every card anyway and
  narrows the variable to the one it picked, so a two-card deployment ran on
  one with nothing in the log to show for it. The GPUStack guide gets the
  API form of the GPU selector, the fact that changing it does not restart
  the running replica, and an eight-slots-on-two-cards recipe with what it
  holds ([#46](https://github.com/Ar4ikov/vibevoice.c/pull/46)).

## [0.5.0](https://github.com/Ar4ikov/vibevoice.c/compare/v0.4.1...v0.5.0) — 2026-09-21

- Teacher forcing is a build option, off by default
  (`-DVV_TEACHER_FORCING=ON`). `VV_TEACHER_TOKENS` points at a file whose
  contents replace the tokens the model produced, and `serve` hands those to
  a client: CodeQL called that an exposure of system data
  (`cpp/system-data-exposure`, alerts 4 and 5) and it was right about a
  release binary. A build without the flag says so when the variable is set.
  `VV_SAVE_TOKENS` is unchanged.
- The repository has its community files: `LICENSE` (MIT, as the README
  always said), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` and a
  pull request template.

- The mark ships as a PNG as well (`assets/brand/logo-256.png`), because a
  catalogue that shows a logo usually takes one format and it is not SVG.
  The GPUStack guide says how its backend list gets an icon: not from the
  imported YAML, which drops the field, but from the community catalog or
  from the column directly.

- **The language model answers text too**: `POST /v1/chat/completions`,
  OpenAI-shaped, so the same deployment can be pointed at by a chat client
  ([#43](https://github.com/Ar4ikov/vibevoice.c/issues/43)). Messages render
  to the ChatML the checkpoint was trained on, `stream: true` sends
  `chat.completion.chunk` deltas cut at whole UTF-8 characters, and
  `temperature` / `top_p` / `top_k` / `seed` / `stop` / `max_completion_tokens`
  do what they say -- greedy by default, and the same seed gives the same
  answer. `usage` counts the prompt and the answer. A generation takes a
  slot exactly as a transcription does, so the queue and `--slots` bound
  both, and the two share the KV cache of the slot. 3090, AWQ: 135 tok/s,
  the same decode path as a transcript; the CPU path answers too, greedily.
  What comes back is this checkpoint talking -- it was fine-tuned to
  transcribe, and it shows.
- Transcriptions carry OpenAI's `usage` (`{"type":"duration","seconds":N}`)
  in `json` and `verbose_json`. Without it GPUStack recorded every request
  with zero usage and logged an error per call.

- A container is no longer told the host's GPU numbers
  ([#41](https://github.com/Ar4ikov/vibevoice.c/issues/41)). GPUStack 2.2.2
  mounts the cards it assigned by UUID and still sets
  `CUDA_VISIBLE_DEVICES` to their host indexes, so a replica given host GPU 1
  alone saw `/dev/nvidia1` as device 0, was asked for device 1, found none
  and ran the whole model on the CPU. The list is now read against the nodes
  the container actually holds and translated before CUDA starts, or dropped
  with a warning when it names nothing present; a host, where the two
  numberings agree, is untouched, and `VV_KEEP_CUDA_VISIBLE_DEVICES=1` turns
  it off. An out-of-range `--gpus` id now also says that the ids count the
  cards this process was given, not the host's.

- GPUStack backend ([#25](https://github.com/Ar4ikov/vibevoice.c/issues/25)):
  the run command passes `--gpus all`, so a replica uses every GPU GPUStack
  assigned to it instead of only the first. GPUStack spreads a replica
  over several GPUs on its own when no single card has room for its 8.3 GiB
  estimate. The UI offers every `serve` option added since the manifest was
  written (`--split-mode`, `--gpu-memory`, `--gpus`, `--quant`, `--attn`,
  `--kv-paged`, `--calib`, the BitNet and streaming ones). The deployment
  guide gains a table of models: which repositories GPUStack can pull
  as-is, since the dense and NF4 ones ship no `tokenizer.json`, their
  parameters, and GPUStack's VRAM estimate against what each actually
  holds (2-3x too high for Streaming-7B and BitNet). It also covers
  replicas vs layer split per GPU, why `{{gpu_ids}}` must not be used (it
  holds host indexes), what `--gpu-memory` counts, and CPU-only replicas.

- `--cpu` now wins over `--gpus` / `--gpu-memory` ([#33](https://github.com/Ar4ikov/vibevoice.c/issues/33)):
  the device flags are no longer evaluated under it. `--gpus all --cpu`
  on a machine with no visible GPU used to exit with `gpus: 'all' asked
  for, but no device is visible`, which is what a CPU-only GPUStack
  replica ran into. `serve`, `chat`, `mic` and the file command share one
  `vv_gpu_set_from_flags()` instead of three copies.
- `serve` no longer warns `raise --slots to use them all` when a layer
  split already uses every device. Under `--split-mode auto` fewer slots
  than devices means a layer split, so the warning now fires only for an
  explicit `--split-mode replica`, where devices do sit idle.

- **A Metal backend: the whole pipeline on the Apple Silicon GPU**
  ([#37](https://github.com/Ar4ikov/vibevoice.c/issues/37)).
  `src/metal/` is the second implementation of `include/vibevoice/device.h`,
  and everything above the seam is the code that already ran on CUDA: every
  op, every weight format (NF4, AWQ/GPTQ INT4, INT8, W8A8/W4A8, BitNet
  ternary), every KV format (FP16, FP8 x2, TurboQuant x4), all three
  attention backends, graph capture and the paged cache. **Xcode is not
  required** -- the shaders are embedded as MSL source and compiled by the
  Metal framework on first use. `scripts/build-macos.sh` picks it up on
  Apple Silicon; `--cpu`, `VV_METAL_DISABLE=1` or `-DVV_ENABLE_METAL=OFF`
  leave it out. M4, 8 GPU cores, NF4, 30 s file: encode 1.9 s,
  prefill 77-99 tok/s, decode 15-17 tok/s, RTF 0.56 against 13.6 on the CPU
  path of the same machine. Three ops decline rather than pretend: skinny linear,
  the int8 MMA path (Apple GPUs have no integer matrix unit) and multi-GPU.
  See [docs/METAL.md](docs/METAL.md).
- **Unified memory is budgeted as memory, not as a GPU limit.** Where the
  device shares the machine's RAM (`vv_dev_host_shares_memory`), an uploaded
  tensor's host copy is released -- the embedding table and the head are
  pointed at their device copies and the speech encoders' FP32 host weights
  are freed once the front end has its FP16 ones (4.7 GB of the 7B) -- and
  `vv_gpu_budget_reclaim` counts those bytes as spendable, because placing
  them is a memcpy rather than a second copy across a bus. What the device
  reports as free is also bounded by what the OS can still give, so the KV
  window is trimmed to what exists instead of to what Metal permits. A
  discrete card is unaffected: it reclaims nothing and decides as before.
- **INT4 weights are not unrolled to run a GEMM on Metal.**
  `vv_w4a16_gemm_dev` dequantized a whole projection into scratch first --
  136 MB for one 7B gate or up, and more scratch than its callers hand it,
  which returned `VV_ERR_OVERFLOW` on the widest shapes. The tile GEMM now
  dequantizes one 16-byte chunk per row into threadgroup memory as it goes,
  which is exactly one K step of 32. Bit for bit what the old path produced;
  `VV_W4_FUSED=0` restores it.
- **The Neural Engine was measured, and is not used**
  ([docs/ANE.md](docs/ANE.md)). It is reachable only through Core ML, with
  the weights baked into a compiled model, and only FP16 ones run fast:
  14.9 TFLOP/s on a prefill matmul, but int4 weights are not placed on it at
  all, decode reads at 62 GB/s against the GPU kernels' 94, and the speech
  encoder's own graph runs three times slower there than on the GPU.
  Metal 4's tensor operations -- M5's door to the GPU's neural accelerators,
  and already offered on M4 -- land in the same band as the simdgroup-matrix
  kernels here, because on this generation they are the same units.
  `tools/ane_probe/` and `tools/metal4_probe/` reproduce both.

## [0.4.1](https://github.com/Ar4ikov/vibevoice.c/compare/v0.4.0...v0.4.1) — 2026-09-19

- **Streaming-7B chunks prefill on kernels sized for them in every format**
  ([#35](https://github.com/Ar4ikov/vibevoice.c/issues/35)).
  A linear layer of 9..64 rows -- a 29-row chunk, a short prompt, the tail of
  a chunked one -- on dense FP16, `--quant int8` or NF4 weights now runs on
  tensor-core kernels built for it (`src/cuda/gemm_skinny.cu`) instead of the
  128-row tile GEMM: 64 output columns per block, q/k/v and gate/up in one
  launch each so every SM has blocks, and the weight read once in the form
  it is stored, INT8 and NF4 dequantized in registers rather than written
  out as an FP16 copy first. The output is bit-identical to the old path
  (the same `mma.m16n8k16` sequence in the same k order, no split-K, the
  same dequant and bias arithmetic), so no transcript moves wherever a
  prefill chunk boundary falls: 37 of 37 outputs byte-identical to 0.4.0
  (streaming none/int8/nf4, one-shot NF4 and AWQ on jfk, test30, test120
  and the 32-minute file, and two NF4 prompts that do land a chunk in 9..64
  rows). 3090, test120, chunk latency: FP16 281 → 210 ms, int8 227 → 130 ms,
  nf4 205 → 110 ms; one 7B layer at 29 rows 5.3-8.4x faster.
  `VV_SKINNY=0` restores the old path; `tests/test_skinny.c` holds the
  kernels to it byte for byte and to the CPU kernels.

## [0.4.0](https://github.com/Ar4ikov/vibevoice.c/compare/v0.3.0...v0.4.0) — 2026-09-19

- **`microsoft/VibeVoice-ASR-Streaming-7B` runs** ([#20](https://github.com/Ar4ikov/vibevoice.c/issues/20)):
  text chunk by chunk while audio arrives, on GPU and CPU, from BF16 as-is or
  `--quant int4|int8|nf4`. `vv_cli --audio` prints each chunk as it is
  produced and then the transcript with speaker-turn segments; `mic` pushes
  capture blocks into one live session (no VAD); `serve` answers
  `stream=true` with SSE `transcript.text.delta` / `transcript.text.done`
  and takes live PCM of any rate over a WebSocket at `/v1/audio/stream`.
  A session holds one slot, and a client that leaves releases it at the
  next token. The session (`include/vibevoice/stream.h`, backend in
  `src/inference/stream_ctx.c`) prefills the prompt into a reset cache,
  encodes each 26-frame window through the device front end as a stateless
  job with the connectors writing straight into the chunk's rows, prefills
  the chunk at `kv->current_len` with the previous `<|text_chunk_end|>`
  folded in, and decodes on the captured step, whose graphs stay alive
  across chunks; several ready windows are encoded in shared launches. With
  `--quant none` every chunk's token ids equal upstream `streaming_generate`
  on jfk, test30 and test120 (156/156); on a 300 s excerpt 101/103 chunks,
  one word moving across a chunk boundary, same transcript. int4 and int8
  give the same transcripts, nf4 the same words with different punctuation.
  3090, test120: 92 ms per chunk and RTF 0.032 with int4 (9.8 GB), 306 ms
  and 0.106 in FP16 (18.6 GB). `serve` carries about 24 live int4 streams per 3090 under 300 ms p95
  chunk latency; streaming slots are budgeted at their real 256 MB
  workspace so the shared KV pool gets the rest of the card. A session refuses a chunk that
  would outgrow its KV window instead of writing past it. Also: a streaming
  resampler equal to the whole-file one sample for sample,
  `VV_ERR_CANCELLED`, `tools/stream_client.py` (live WebSocket client and
  sessions-per-GPU bench), and `cmp-stream` reports the joined transcript's
  word edits. The protocol, reference tooling and numbers are in
  `docs/STREAMING.md`.
- W8A8 and W4A8: activations quantized to int8 per token, int8 × int8 on
  the s8 tensor cores for prefill and `dp4a` GEMVs for decode, with no FP16
  weight copy anywhere. `--quant w8a8|w4a8` quantizes a BF16 checkpoint at
  load; `--quant w4a8` also runs an AWQ checkpoint exactly (not the NF4
  one: its codes are not integers);
  compressed-tensors checkpoints (llm-compressor `int-quantized` /
  `pack-quantized`, dynamic int8 activations) load directly. SmoothQuant
  folds in at load from a built-in calibration pass (`--calib <audio>`,
  `--calib-stats <file>`, also on `serve`). On a 3090 the 32-minute prompt
  prefills at 7152 tok/s (W8A8) and 5100 (W4A8) against 3451 for `int4`;
  words are the same as dense BF16 on jfk, test30, test120 and the
  32-minute file. `VV_SAVE_TOKENS` / `VV_TEACHER_TOKENS` measure
  teacher-forced top-1 agreement. Default behaviour is unchanged.
- VibeVoice-ASR-BitNet runs end to end on the CPU and the GPU (`vv_cli`,
  `serve`, `chat`), from VibeASR.cpp's GGUF pair or the F32 safetensors
  (ternarized and int8-quantized at load). The LM uses exact ternary x
  int8 kernels (AVX2 / AVX-VNNI / AVX512-VNNI / NEON on the CPU, dp4a GEMV
  and an int8 tensor-core GEMM on the GPU) in the reference's operation
  order; the CPU picks the F16 head's argmax through an exact int8 filter.
  `--vae int8` is VibeASR.cpp's fully int8 speech encoder bit for bit (the
  CPU default), `--vae float` the same weights with GELU through the shared
  GPU front end (the GPU default). With the int8 encoder the transcripts of
  jfk, test30 and test120 equal VibeASR.cpp's `--greedy` output on both
  backends. 5900X, test30: RTF 0.16 at 12 threads against the reference's
  0.26; RTX 3090: RTF 0.012 (test30), 0.005 (120 s), 0.007 (32 min), 335
  tok/s decode, 4.2 GB VRAM. `docs/BITNET.md` has the formats, numerics and
  measurements; `tools/bitnet_ref.py` is a numpy fake-quant reference.
- `serve --cpu` ran every request's OpenMP team on one core (RTF 13.7 on
  jfk.wav, now 0.23): the connection threads inherited the accepting
  thread's core-0 pin and now drop it (`vv_cpu_thread_unbind`). The main
  thread stays pinned: leaving it unpinned cost the 7B CPU path a third
  (prefill 73 -> 47 tok/s, decode 8.0 -> 5.3).
- An asr-bitnet run that placement sends to the CPU (a small
  `--gpu-memory`, a busy card) is loaded again the way `--cpu` loads it;
  before, it read the GPU's F16 norms as FP32 and decoded garbage until the
  KV window was full.
- The front end's counters (`vv_frontend_stats`) have their own lock; a
  reader no longer waits behind the service thread for a whole long file.

## [0.3.0](https://github.com/Ar4ikov/vibevoice.c/compare/v0.2.0...v0.3.0) — 2026-09-19

- Unquantized checkpoints load: `microsoft/VibeVoice-ASR` in BF16 runs as-is
  (dense FP16 on the GPU, 56 tok/s, 18.7 GB) or quantized while it is read
  with `--quant nf4|int4|int8` (`vv_cli`, `serve`, `chat`; `auto` keeps
  whatever the checkpoint has). `int8` is per-channel symmetric with its own
  GEMV (97 tok/s, 12.5 GB); `int4` is the fastest (149 tok/s, 9.7 GB). Projections are routed by what the file holds rather
  than by name, every tensor is checked against the config's shape, and a
  missing or misshaped one fails the load with its name instead of a
  "prefill layer 0 failed" later on.
- Model load runs on the physical cores (unless `OMP_NUM_THREADS` is set)
  instead of one thread per SMT sibling: the NF4 checkpoint loads in 4.0 s
  instead of 8.7 s, AWQ in 5.5 s instead of 10.2 s (3090 + 5900X, best of 3).
- `tie_word_embeddings` is honoured (at the root or in `decoder_config`): the
  head is the embedding buffer, uploaded and budgeted once.
- Model families: `asr-7b`, `asr-bitnet` and `asr-streaming-7b` are told
  apart from `config.json` / `preprocessor_config.json`, each with its own
  prompt, stop tokens, normalization and chunk geometry (`family.h`).
- The speech encoder's BF16 weights widen to FP32 exactly; they were
  truncated to FP16 first, which zeroed 1.2 M of them. F32 LM tensors are
  rounded to FP16 instead of reaching FP16 kernels unconverted.
- Prefill appends at the cache's current length on GPU and CPU, and CPU
  prefill runs in chunks, so CPU-only transcription is no longer capped at
  about five minutes of audio by a 512 MB workspace.
- Config dimensions are required: a config without, say,
  `intermediate_size` is refused instead of silently becoming the 7B.
  Non-SiLU activations, rope scaling and sliding windows are refused too.
- **AWQ / GPTQ fast path** ([#19](https://github.com/Ar4ikov/vibevoice.c/issues/19)):
  INT4 group-affine weights get a GPU layout at load (nibbles permuted per
  16-byte run, scale and exact integer zero point as one `half2` per group)
  and two kernels in `src/cuda/w4a16.cu` that share it. The decode GEMV
  uses 128-bit loads, `LOP3`/`0x6400` conversion and HFMA2, with as many
  lanes per row as it takes to fill the card, and takes the projections
  that share an input in one launch: on a 3090 gate+up 861 GB/s, down 822,
  q+k+v 742 (from 516 as three launches), o 683, against 727 / 669 / 597 /
  208 before. Every M > 1 runs a Marlin-style tensor-core GEMM that keeps
  the weights packed until they are in registers, instead of expanding each
  weight into an FP16 copy first: 14–25× faster at 9–16 rows, 3–9× at 64,
  1.0–1.8× for a 2048-token chunk (0.97× on q/o, see the PR). End to end on
  the AWQ checkpoint: decode 133 → 149.1 tok/s at 0.2K context, 116 → 126.2
  at 1.5K, 77 → 82.4 over 14K → 24K; prefill 1010 → 2394 tok/s on jfk,
  2810 → 3466 on test120; the 32-minute file goes from RTF 0.078 to
  0.074. Dequantization is now the reference's `(q − z)·s` with the
  checkpoint's integer zero point; transcripts on jfk / test30 / test120 are
  unchanged, and on the 32-minute file the text and speakers are too while
  9 of 168 segment boundaries move by at most 20 ms. The loader reads
  real AutoGPTQ tensors (`qweight [K/8, N]`, told apart from AWQ by shape),
  honours GPTQModel's `gptq_v2` zero points, runs act-order (`desc_act`)
  checkpoints by sorting their input channels by group at load and
  gathering the activations through the same order, checks every tensor
  size against the header, and fails the load on a tensor it cannot
  repack. The GPU layout costs 0.3–1.2 s at load for the 7B (in place, no
  second copy). `VV_INT4G_LEGACY=1` restores the old path. `--quant int4`
  on a dense checkpoint now quantizes AWQ's way, with an exact integer zero
  point per group (the range widened to hold 0), and so runs on the same
  kernels: `microsoft/VibeVoice-ASR` with `--quant int4` decodes at 149
  instead of 133 tok/s, RTF 0.053 / 0.052 / 0.052 on 11 s / 30 s / 120 s
  instead of 0.064 / 0.060 / 0.057, same words.
- **Attention backends** ([#17](https://github.com/Ar4ikov/vibevoice.c/issues/17)):
  `--attn auto|fa1|fa2|flashinfer` (and `VV_ATTN=`; `VV_ATTN_MMA=0` still
  means the scalar kernels) on `vv_cli`, `serve`, `chat` and `mic`, behind
  one dispatch point. `auto` runs **fa2**: the FA2 prefill with the 7 heads
  of a GQA group packed into one block and a decode that runs a group
  together — bit-identical to the kernels before, so no transcript moves,
  and 32-minute decode goes from 75.0 to 93.5 tok/s (RTF 0.080 -> 0.067).
  **flashinfer** puts decode on tensor cores too, splits the cache for
  few-row prefill, and runs fp8/tq caches through the same kernels
  (107.5 tok/s, RTF 0.060; one-hundredth-second timestamp moves, no word
  changes on the parity set). A `tq4` cache on the 32-minute file: 33.4 ->
  73.7 tok/s with fa2, 111.2 with flashinfer. Turing gets the tensor-core paths through
  `mma.m16n8k8`.
- **Paged KV cache**: the slots of a device share one pool of 64-position
  pages instead of a full window each (`--kv-paged auto|on|off`, on by
  default with several slots whose kernels read pages). The pool honours
  the VRAM budget, which clones' slabs never did. A request reserves its
  expected pages before it starts and waits while others hold them; a step
  that finds the pool dry waits for pages too, and only when no other
  holder could ever free any does one request fail, with the new
  `VV_ERR_KV_POOL_EXHAUSTED` (503 from `serve`). A pool never truncates a
  transcript and reports success.
- A decode step that fails now fails the request; before, the tokens so far
  were post-processed and returned as a complete transcript.
- **Speech front end** ([#16](https://github.com/Ar4ikov/vibevoice.c/issues/16)):
  the Conv-VAE encoders are split into weights (FP16, one copy per device,
  shared by every slot -- each `--slots` clone used to upload its own
  1.4 GB), per-stream state (the convolution tails, ~1.4 MB) and one arena
  sized from (jobs, samples) per launch and counted by the placement budget
  in place of the flat 256 MB guess. An encode is kernel launches on the
  caller's stream and nothing else: the ~110 `cudaMalloc`/`cudaFree` pairs
  per segment, the per-stage syncs and the host round trips are gone. One
  launch takes any mix of jobs packed along time -- segments of a file,
  several requests, chunks of live streams, stateless windows (from zero
  state, as Streaming-7B's 26-frame windows need) -- and both encoders run
  concurrently on two streams. The mixer is one fused RMSNorm + depthwise
  conv + gamma-residual kernel, the FFN norm, bias, GELU and residual ride
  in the GEMMs' operand load and epilogues, and the downsample and head
  convs are an FP32 GEMM over im2col columns that sums in the direct
  kernel's order; nothing re-associates a sum, so latents, prompt
  embeddings and transcripts stay bit-identical to before (jfk, test30,
  test120, the 32-minute file, `--acoustic-sampling gaussian`). Streaming
  is exact for any chunk length, not only multiples of 3200 samples. The
  connectors run on the device and write the prompt's rows in place. With
  `serve --slots N` a per-device worker plans one launch at a time from
  whatever has arrived, so concurrent requests share launches and a short
  request is not stuck behind a long file's encode; the encoder runs at
  background stream priority so other slots' decode steps go first. The
  CPU encoder is time-tiled with carried state (bounded memory), OpenMP,
  through the packed GEMM, and no longer runs on the GPU under `--cpu`.
  Batched, chunked and windowed encodes equal one-by-one bit for bit
  (`tests/test_vae_stream.c`). 3090, against master with the attention
  backends and the W4A16 path: encoder 197 -> 52 ms on 11 s, 1347 -> 350 ms
  on 120 s (RTF 0.054 -> 0.046), 21.9 -> 5.4 s on the 32-minute file (RTF
  0.067 -> 0.058; AWQ 0.060 -> 0.051); eight 30 s files through
  `serve --slots 4` 13.1 -> 10.5 s and 20.5 -> 16.9 GB of VRAM; jfk sent
  0.3 s into a 32-minute request 2.4 -> 2.3 s; CPU encoder on 11 s 71 ->
  2.5 s.
- The TensorRT stubs (`src/trt`, `trt.h`, the CMake option, the ONNX/TRT
  export tools) are removed; they never ran, and `VV_ENABLE_TRT` defaulted
  to ON, linking `libnvinfer` into dev builds. `--trt-acoustic` and
  `--trt-semantic` still parse for this release and log a deprecation
  warning.

## [0.2.0](https://github.com/Ar4ikov/vibevoice.c/releases/tag/v0.2.0) — 2026-09-18

- **Security** ([#14](https://github.com/Ar4ikov/vibevoice.c/issues/14)):
  ffmpeg and the recorder are started from an argument vector
  (`posix_spawnp`, `CreateProcessW` with only the pipe and `NUL`
  inherited) instead of through `popen()` and a shell, so a quote, `$(...)`,
  backtick, `;`, `&` or `%VAR%` in a path or device name is passed, not run
  (CodeQL `cpp/command-line-injection`); ffmpeg gets its input as
  `file:<path>`. Upload temp files are created exclusively (`mkstemp`,
  `GetTempFileName`) -- two concurrent uploads could share a name and each
  other's audio through an unsynchronised counter. The HTTP reader parses
  only within the request head, honours `Content-Length` in any
  capitalisation and only as a header of its own, refuses
  `Transfer-Encoding`, and times out slow clients; the API key is compared
  in constant time; error and `/health` JSON is escaped. A decode step that
  would pass the end of the KV window now stops the transcript with a
  warning -- a replayed CUDA graph used to write past the cache. SRT/VTT
  segments longer than 1 KB are no longer truncated. `tools/requirements.txt`
  moves past `torch.load` RCE (CVE-2025-32434) and CVE-2024-34062. Closing
  `mic` no longer frees the recorder's stream under the capture thread.

- One model can be split across devices: layers 0..k on the first, the rest
  on the next, each owning the KV cache for its own layers and only the
  hidden state crossing between them. `--split-mode auto|replica|layer`
  chooses; `auto` replicates while there is a slot per device and shards
  otherwise. On two 3090s with a 6 GiB budget each, a 30 s file goes from
  7 of 28 layers resident and RTF 0.595 to all 28 and RTF 0.059.
- `--gpus 0,1` / `--gpus all` selects devices, `--gpu-memory` caps what each
  one may hold (`80%`, `18GiB`, `8192M`, a byte count, or one value per
  device). The cap covers the CUDA context, the speech encoder and its
  scratch as well as the transformer, and it steers the placement decision
  instead of being checked after the fact.
- `serve` puts one replica per selected device and spreads its slots over
  them: eight 30 s clips through four slots take 13.2 s on one 3090 and
  7.6 s on two.
- When the budget is tight the KV window is now halved down to 8192 tokens
  before layers start streaming, instead of only being trimmed in the case
  where everything already fitted.
- CPU prefill is a packed GEMM: k-major panels, a 6x16 AVX2 register tile and
  cache blocking sized off L1 and L2. 283 → 991 GFLOP/s on a 5900X, prefill
  22 → 77 tok/s, and a 30 s file goes from RTF 1.125 to 0.809 with a
  character-identical transcript.
- CPU worker threads are pinned one per physical core. Without it the OS
  regularly stacks two of them on one core's hyperthreads, which cost up to
  40% at random. `VV_CPU_BIND=0` or any OpenMP affinity variable opts out.
- The physical-core count now respects a cpuset or taskset instead of
  counting every core on the host.
- Every build has a SemVer version, compiled in: `0.2.0` for the v0.2.0
  tag, `0.2.1-dev.5+g1a2b3c4` five commits past it (a prerelease of the next
  version, so it sorts between the two). One rule in `cmake/Version.cmake`,
  used by local builds, CI and the image alike.
- Every published image has an immutable SemVer tag to pin: `0.2.0` for a
  release, `0.2.1-dev.5` for each push to master, plus `sha-…`. Moving tags:
  `0.2` and (from 1.0) `1` follow the newest release of their line, `latest`
  the newest release only -- no longer master, which is `master` and `edge`.
  Pull requests build as `pr-…`, not pushed, and are checked like the rest.
- Releases are one click: the Release workflow infers the next version from
  Conventional Commits (or takes `patch|minor|major|X.Y.Z-rc.N`), bumps the
  header, moves this section, tags, and publishes the image, a GitHub release
  and a `linux-x86_64` binary tarball with its checksum.
- `vv_cli --version`, `GET /health` and a `vibevoice_build_info` metric all
  report the version and the commit; CI checks that each image reports
  exactly the version it is tagged with, before it is announced.
- OCI labels on the image: title, description, source, licence, version,
  revision.
- Identity: hand-drawn mark, wordmark and lockup under `assets/brand/`,
  documented in [docs/BRAND.md](docs/BRAND.md).

## 0.1.0 — never tagged

No image or tag was ever published for 0.1.0; 0.2.0 is the first release.

The runtime as first described in the README: transcription with
diarization and timestamps, an OpenAI-compatible server with slots and a
bounded FIFO queue, live microphone capture, NF4/AWQ/GPTQ weights, FP8 and
TurboQuant KV caches, layer streaming for small cards, CUDA-graph decode and
tensor-core prefill. Output matches the PyTorch reference character for
character.
