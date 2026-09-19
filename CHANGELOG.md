# Changelog

Newest first. Versions follow [SemVer 2.0](https://semver.org): below 1.0 a
minor bump may break, a patch never does. The release number lives in
`VV_VERSION_STRING` in `include/vibevoice/vibevoice.h`; the Release workflow
bumps it, turns **Unreleased** into the release's section, tags `v<version>`
and publishes the image and binary -- see [docs/RELEASING.md](docs/RELEASING.md).
Write the entry for a change under Unreleased in the same pull request.

## Unreleased

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
  (`tests/test_vae_stream.c`). 3090: encoder 195 -> 50 ms on 11 s, 1352 ->
  352 ms on 120 s (RTF 0.064 -> 0.055), 21.1 -> 5.4 s on the 32-minute file
  (RTF 0.084 -> 0.071); eight 30 s files through `serve --slots 4` 18.2 ->
  14.8 s with 1.3 GB less VRAM per extra slot; jfk sent 0.3 s into a
  32-minute request 2.4 -> 2.3 s; CPU encoder on 11 s 70 -> 2.4 s.
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
