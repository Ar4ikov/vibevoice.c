# Changelog

Newest first. Versions follow [SemVer 2.0](https://semver.org): below 1.0 a
minor bump may break, a patch never does. The release number lives in
`VV_VERSION_STRING` in `include/vibevoice/vibevoice.h`; the Release workflow
bumps it, turns **Unreleased** into the release's section, tags `v<version>`
and publishes the image and binary -- see [docs/RELEASING.md](docs/RELEASING.md).
Write the entry for a change under Unreleased in the same pull request.

## Unreleased

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
