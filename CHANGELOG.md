# Changelog

Newest first. The version is `VV_VERSION_STRING` in
`include/vibevoice/vibevoice.h`; tagging `v<version>` publishes the matching
container images — see [docs/RELEASING.md](docs/RELEASING.md).

## Unreleased

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
