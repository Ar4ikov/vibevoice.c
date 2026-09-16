# Changelog

Newest first. The version is `VV_VERSION_STRING` in
`include/vibevoice/vibevoice.h`; tagging `v<version>` publishes the matching
container images — see [docs/RELEASING.md](docs/RELEASING.md).

## Unreleased

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
- Container images are versioned: `0.2.1`, `0.2`, `master`, `sha-…` and
  `pr-…` alongside `latest`, which still points at the newest published
  build. The workflow refuses a `v*` tag that disagrees with the header, and
  verifies after publishing that the image reports its own tag.
- `vv_cli --version`, `GET /health` and a `vibevoice_build_info` metric all
  report the release version and the build ref.
- OCI labels on the image: title, description, source, licence, version,
  revision.
- Identity: hand-drawn mark, wordmark and lockup under `assets/brand/`,
  documented in [docs/BRAND.md](docs/BRAND.md).

## 0.1.0

The runtime as first described in the README: transcription with
diarization and timestamps, an OpenAI-compatible server with slots and a
bounded FIFO queue, live microphone capture, NF4/AWQ/GPTQ weights, FP8 and
TurboQuant KV caches, layer streaming for small cards, CUDA-graph decode and
tensor-core prefill. Output matches the PyTorch reference character for
character.
