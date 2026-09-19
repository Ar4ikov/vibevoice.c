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
  GEMV (97 tok/s, 12.5 GB); `int4` is the fastest (133 tok/s, 9.7 GB). Projections are routed by what the file holds rather
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
