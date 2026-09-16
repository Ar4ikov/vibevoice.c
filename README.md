# vibevoice.c

A runtime for Microsoft's **VibeVoice-ASR** written in C and CUDA, with no
Python, PyTorch, ONNX Runtime or cuBLAS anywhere in the inference path. One
binary transcribes audio with speaker diarization and timestamps, serves an
OpenAI-compatible HTTP endpoint, or listens to a microphone.

Output is **identical, character for character, to the PyTorch reference**
running the same checkpoint — verified on 11 s, 120 s and 32-minute inputs by
diffing every intermediate tensor against a `transformers` + `bitsandbytes`
dump (`tools/compare_ref.py`).

```
$ vv_cli --model ./model_hf --audio meeting.wav --output transcript.json
[  0.00 -  11.11] Speaker 0  And so, my fellow Americans, ask not what your
                             country can do for you, ask what you can do for
                             your country.
[ 11.73 -  22.29] Speaker 1  He hoped there would be stew for dinner, turnips
                             and carrots and bruised potatoes...
RTF 0.065  (120 s of audio in 7.8 s)
```

---

## Why this exists

VibeVoice-ASR is a Qwen2-7B decoder fed by two Conv-VAE speech tokenizers.
Running it normally means a Python stack, a PyTorch install, and 4-bit
weights unpacked through bitsandbytes. This is the same model with the same
outputs, as a single file you can drop on a machine and run.

**No CUDA toolkit on the target.** The CUDA runtime is linked statically and
cuBLAS is not used at all — both GEMM shapes are hand-written WMMA kernels,
which measured slightly *faster* than cuBLAS here. The release binary is
9.6 MB and links against nothing but `libc` and `libm`:

```
$ ldd vv_cli
    linux-vdso.so.1
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

It carries cubins for Turing, Ampere, Ada and Hopper plus PTX for anything
newer, and the driver is loaded lazily — so the same binary also starts on a
machine with no NVIDIA driver at all and falls back to the CPU kernels.

---

## Performance

RTX 3090, CUDA 12.4, Ryzen 9 5900X. Defaults unless noted.

| | |
|---|---|
| Model load | **9.5 s** |
| Speech encoding | **243 ms** per 11 s of audio |
| Prefill | **3056 tok/s** on a 14449-token prompt |
| Decode | **123 tok/s** at 0.2K context, 100 at 1.5K, 57 at 24K |
| RTF | **0.065** on a 120 s file, **0.081** on 32 minutes |
| VRAM | 9.8 GB (3.2 GB weights + 1.8 GB KV at a 32K window) |

`transformers` + `bitsandbytes` on the same GPU and checkpoint: 27.6 tok/s.

A 32-minute recording transcribes in 155 s: 14449 prompt tokens, 9522
generated, 168 segments, flat memory throughout.

### CPU only

No GPU, or `--cpu`. Runtime-selected AVX2 on x86 and NEON on ARM, threads
defaulted to physical cores.

| | |
|---|---|
| Decode | **7.9 tok/s** (12 cores, AVX2) |
| RTF | **1.24** on an 11 s file |

Output is identical to the GPU path. Hyperthreads are deliberately not used:
these kernels are bandwidth bound and a second thread per core takes decode
from 7.9 to 4.3 tok/s. `OMP_NUM_THREADS` still overrides.

---

## What it does

### Transcription

```bash
vv_cli --model ./model_hf --audio recording.wav --output transcript.json
vv_cli --model ./model_hf --audio meeting.m4a --hotwords "Kubernetes,Grafana"
```

WAV is parsed directly; anything else goes through `ffmpeg` if it is on PATH.
Audio longer than 60 s is encoded in streaming segments with convolution
caches, so the result is bit-identical to processing the whole signal at once
and memory stays flat.

Output is deterministic: the same file gives the same transcript every time.
`--acoustic-sampling gaussian --seed N` swaps that for the reference's draw of
the acoustic latent, seeded so it stays repeatable — see below.

### HTTP server

```bash
vv_cli serve --model ./model_hf --port 8080 --slots 2 --kv-cache tq4
```

OpenAI-compatible, which is also the contract GPUStack's speech-to-text
backend expects:

| endpoint | |
|---|---|
| `POST /v1/audio/transcriptions` | multipart `file`, plus optional `model`, `prompt` (used as hotwords), `temperature`, `response_format` |
| `POST /v1/audio/translations` | same handler; this model outputs English |
| `GET /v1/models`, `GET /v1/models/{id}` | the loaded model |
| `GET /health`, `/healthz`, `/v1/health` | liveness, slot count, completed requests |
| `GET /metrics` | Prometheus counters |

`response_format` accepts `json`, `verbose_json` (segments with timings and
speaker), `text`, `srt` and `vtt`.

```bash
curl http://localhost:8080/v1/audio/transcriptions \
     -F file=@meeting.mp3 -F response_format=verbose_json
```

`--slots N` runs N requests concurrently against **one** copy of the weights;
each slot costs only its own KV cache and workspace. Two 30-second files
finish in 4.4 s together against 5.9 s back to back. Not 2×, because decode
is bandwidth-bound on the weights and interleaves — everything either side of
it overlaps. With `--kv-cache tq4 --max-seq-len 8192` a slot is 116 MB.

### Live microphone

```bash
vv_cli devices                                          # what can record
vv_cli mic --model ./model_hf --timestamps
vv_cli mic --model ./model_hf --device 1                # index, name, or part of one
vv_cli mic --model ./model_hf --from-file meeting.wav   # replayed in real time
```

Captures continuously and transcribes each utterance as it ends, segmenting
on an energy VAD with hysteresis, pre-roll and hangover. Capture is ALSA
(loaded with `dlopen`, so the binary still runs without it), falling back to
an external recorder on a pipe — `arecord` on Linux, ffmpeg's `dshow` on
Windows, `avfoundation` on macOS.

`vv_cli devices` lists what the active backend can open, and `--device` takes
an index from that list, a device name, or any fragment of one. dshow has no
device called "default", so on Windows the first device is resolved by name
before ffmpeg is started; what actually gets passed is the alternative name
(`@device_cm_{...}`), which stays unambiguous when two endpoints share a
label — a combined headset, or two identical ones on the same machine.

`--from-file` replays a WAV at wall-clock speed through the identical path,
which is how the streaming path is tested without a sound card.

### Chat

```bash
vv_cli chat --model ./model_hf
> meeting.wav
> rec 5
> hotwords Kubernetes,Grafana
```

The model loads once and stays warm. Ten seconds of startup per file is
absurd for short clips; this pays it once.

---

## Quantization

### Weights

| format | how |
|---|---|
| **NF4** (bitsandbytes, double-quantized) | loaded directly |
| **AWQ** (AutoAWQ) | loaded directly |
| **GPTQ** | loaded directly (its zero points are off by one; handled) |

AWQ and GPTQ store weights K-major with the eight columns of a word
interleaved, which is the wrong orientation for a GEMV — walking `k` for one
output row would stride by `N/8` words. They are repacked once at load into
the row-major layout the NF4 path uses, with the zero point folded into a
per-group `min` so dequantization in the kernel is a single FMA. The repack
is OpenMP-parallel and costs ~1.4 s for the 7B.

On the same audio, AWQ gives an identical transcript and 2% faster decode
(102.3 vs 100.0 tok/s) for 3% more VRAM — NF4's own scales are
double-quantized, so it is already at 4.13 bits against AWQ's 4.16.

[`Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM`](https://huggingface.co/Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM)
is a real activation-aware build: `llm-compressor`'s `AWQModifier` over 256
calibration samples (64 audio, 192 text), group 128, asymmetric, exported into
the AWQ GEMM container this runtime reads. 7.0 GB against the BF16 original's
17.3 GB. On nine held-out LibriSpeech clips it scores 14 word errors out of
135 against the BF16 baseline's 16, and the C runtime reproduces the PyTorch
AWQ answer on all nine — a smoke test, not a benchmark.

`tools/convert_awq.py` re-quantizes an NF4 checkpoint into the same container
if you would rather not download a second set of weights. It does not do the
activation-aware scale search, so it is the weaker of the two; the runtime
treats both identically.

### KV cache

The KV cache is the only allocation that grows with audio length, so it is
what decides how long a recording fits. `--kv-cache` picks the format:

| `--kv-cache` | at 32K positions | 120 s transcript | decode, 0.2K ctx |
|---|---|---|---|
| `fp16` (default) | 1792 MB | reference | 105 tok/s |
| `fp8` (E4M3) | 896 MB | **identical** | 100 |
| `fp8-e5m2` | 896 MB | **identical** | 98 |
| `tq4` | 462 MB | **identical** | 90 |
| `tq3` | 350 MB | 94% similar | 87 |
| `tq2` | 238 MB | degraded | 89 |
| `tq1.5` | 182 MB | degraded | 87 |

FP8 is emulated in software — encode and decode are a handful of shifts — so
it works on Ampere and would work on Apple Silicon, neither of which has FP8
hardware. The point is halving memory, not using an FP8 MAC.

`tq*` is **TurboQuant** (Zandieh et al., 2025): each 128-dim vector goes
through a randomized Hadamard transform, which spreads the outlier channels
that make raw K hard to quantize, and the result is quantized with the
MSE-optimal Lloyd-Max levels for a Gaussian. The transform is orthogonal, so
attention never inverts it per key — Q is rotated once before the kernel and
the output un-rotated after, and the kernels read stored values directly.

The part that took measuring to find: **quantizing K as it comes destroys the
output at every rate**, including FP8 (attention cosine 0.86). Qwen2's keys
share a large common component — ‖k‖ is 273 against ‖k − mean‖ of 11 — so the
quantizer spends its whole range representing a constant. Softmax is
invariant to a constant shift of every key, so the cache stores K relative to
a per-layer reference vector built from the first prefill chunk. That single
change takes FP8 from 0.856 to 0.9995 and TQ4 from 0.673 to 0.995.

At long context the sub-byte formats cost throughput: on the 32-minute file,
`fp16` decodes at 56.7 tok/s and `tq4` at 30.4, because unpacking bits with
warp shuffles is work the FP16 path does not do. `tq4` wins prefill for the
same reason it loses decode — there is a quarter as much cache to read.

---

## Fitting on a smaller card

`--gpu-layers N` keeps N transformer layers resident and streams the rest
from host RAM per token; `-1` (the default) fits as many as the VRAM budget
allows. The copy for layer *i+1* overlaps the compute of layer *i* through
CUDA events, and the host copies are page-locked so the driver DMAs straight
out of them.

Measured on a PCIe 4.0 x16 link, transcript identical at every setting:

| `--gpu-layers` | decode |
|---|---|
| 28 (all resident) | 124 tok/s |
| 25 | 75.2 |
| 20 | 23.5 |
| 14 | 12.9 |
| 0 (everything streamed) | 6.6 |

Streaming is bandwidth-bound and nothing can change that: with every layer
streamed, 3.3 GB moves per token at 21.8 GB/s, which is the link saturated.
Halving the link halves the number — on a x4 slot the same 20/28 setting
gives 6.1 tok/s instead of 23.5. Use it to make a model fit, not to make it
fast.

Other knobs for tight memory: `--max-seq-len` bounds the KV window,
`--vram-budget 0.0-1.0` caps the fraction of free VRAM used, `--cpu` forces
the CPU path.

---

## Building

### Linux (CUDA)

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVV_ENABLE_TRT=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

For the shipping binary — all architectures, static runtimes, no tests:

```bash
scripts/build-release.sh
```

### Windows

```powershell
scripts\build-release.ps1
```

or `build.ps1` / `build.bat` for a development build. Requires MSVC 2022 and
a CUDA toolkit to *build*; the resulting `.exe` needs neither on the machine
that runs it.

### macOS

```bash
brew install libomp        # optional, but without it the kernels are serial
scripts/build-macos.sh
```

This is the CPU build: NEON on Apple Silicon, AVX2 on Intel Macs, threads on
the performance cores. **There is no Metal backend.** See "Not done" below.

### Tests

```bash
VV_TEST_MODEL=./model_hf ctest --test-dir build --output-on-failure
```

Eleven suites. The ones that need weights report SKIP without
`VV_TEST_MODEL`. `test_cpu_kernels` checks every CPU kernel against a scalar
reference, which is what makes the SIMD paths verifiable per architecture —
it passes natively on AVX2 and under `qemu-aarch64` on NEON, both to 4e-7
relative.

---

## Getting the model

| what | where |
|---|---|
| NF4 weights (bitsandbytes) | [`scerz/VibeVoice-ASR-4bit`](https://huggingface.co/scerz/VibeVoice-ASR-4bit) |
| AWQ weights (W4A16, asymmetric) | [`Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM`](https://huggingface.co/Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM) |
| `tokenizer.json` | [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) or Qwen2.5-7B |

Either checkpoint works; point `--model` at the directory. The NF4 repo does
not ship tokenizer files, so drop `tokenizer.json` next to the safetensors.
The AWQ repo already carries its own.

---

## How it works

```
audio ──> resample 24 kHz ──> normalize -25 dBFS
              │
              ├──> acoustic Conv-VAE ──> connector ──┐
              │    (vae_dim 64)                      ├──> +  ──> Qwen2-7B ──> JSON
              └──> semantic Conv-VAE ──> connector ──┘            28 layers
                   (vae_dim 128)
```

No mel spectrogram, no FFT. Raw 24 kHz PCM goes straight into two Conv-VAE
tokenizers, each compressing 3200× (ratios 8·5·5·4·2·2) down to 7.5 Hz. The
two latent streams are projected to the LLM's 3584 dims and added.

### The acoustic latent is a distribution

The acoustic tokenizer does not return a latent, it returns the mean of a
Gaussian. Which draw the reference takes is in the checkpoint
(`std_dist_type`), and for this one it is `gaussian`: a single scale
`s ~ N(0, fix_std/0.8)` for the whole clip, then `mean + s·N(0,1)` per element.
`s` is drawn per run, is as often negative as positive, and has RMS 0.625.

The default here is the mode — the mean itself, which is reproducible and is
the most likely latent rather than one draw from around it. `--acoustic-sampling
gaussian --seed N` takes the reference's draw instead, and `fix` takes the
simpler `mean + fix_std·N(0,1)`. The seed pins it: same seed, same latent, same
transcript. It cannot be bit-identical to PyTorch, because a different
generator gives different numbers from the same seed; what it answers is
whether the transcript survives noise of that size.

On a clean 30 s clip, eight draws with scales from −0.47 to +0.85 all gave a
transcript identical to the mode. On an 11 s clip of slurred, noisy Russian,
four of eight draws changed a word — `ложи` against `ложе`, the ending of
`Долбоёб...` — which is where the audio is genuinely ambiguous and the noise
decides the coin flip. That is the honest reason the mean was never a problem.

The kernels that matter:

- **`gemm.cu`** — both GEMM shapes on tensor cores via WMMA, 128×128×32 block
  tiles over 8 warps, double-buffered shared memory. Replaces cuBLAS.
- **`nf4_gemv.cu`** / **`awq_gemv.cu`** — dequantization fused into the MAC.
  The generic path (expand to FP16 scratch, then GEMM) moves 5× the bytes;
  single-token decode is purely bandwidth bound, so the weights stay 4-bit
  all the way into the multiply.
- **`attention.cu`**, **`attention_mma.cuh`** — FlashAttention-2 prefill with
  both matmuls on tensor cores, and a split-KV flash decode where each warp
  owns a slice of the cache and a second kernel merges the partial softmax
  states. The prefill kernel issues `mma.sync` as PTX rather than through the
  WMMA API, because the online softmax has to rescale the output accumulator
  per row and only `mma.sync` documents which row each accumulator register
  holds. That layout pays twice: the A operand of the next matmul is laid out
  exactly like the accumulator of the previous one, so the softmax
  probabilities feed P·V from the registers they were computed in.
- **`kv_quant.cu`** — the FP8 and TurboQuant stores, plus attention kernels
  that read them. Sub-byte codes are laid out so lane L owns dims 4L..4L+3,
  putting its bits in a contiguous run the warp fetches with one coalesced
  load and a shuffle.
- **`lm_head.cu`** — the 152k-row head fused with a two-stage on-device
  argmax, so 4 bytes cross the bus per token instead of 600 KB.

Four bugs found by diffing against the reference, each of which alone turned
the output into repetitive noise, are written up in `CLAUDE.md` §0 — byte-level
BPE, the causal `SConv1d` padding formula, exact GELU versus SiLU, and warps
of one block reaching different numbers of `__syncthreads()` in flash
attention.

### Prefill attention

Both matmuls in attention are O(S²) and the first version of the kernel ran
them as scalar FP32 FMAs, one warp per query row. On a 3090, causal
self-attention with 28 heads at head_dim 128:

| sequence | scalar | tensor cores |
|---|---|---|
| 1024 | 4.58 ms (1.6 TFLOP/s) | **0.18 ms (42.1 TFLOP/s)** |
| 4096 | 72.1 ms (1.7 TFLOP/s) | **2.10 ms (57.3 TFLOP/s)** |
| 8192 | 288.2 ms (1.7 TFLOP/s) | **8.03 ms (59.9 TFLOP/s)** |

59.9 TFLOP/s is 84% of the card's 71 TFLOP/s ceiling for FP16 multiply with
FP32 accumulate. End to end on the 32-minute file, prefill goes from 30.7 s to
4.7 s (471 → 3056 tok/s) and the whole transcription from 182 s to 155 s.

The text is identical. Four of the 336 timestamps the model emits move by
10 ms, which is FP16 rounding inside the P·V product landing on the other side
of a digit, and is inside the ±100 ms the timestamps are held to.

`VV_ATTN_MMA=0` forces the scalar kernel, which is also what `ctest` runs the
attention suite a second time under: both have to agree with an FP64 reference
across fourteen shapes, on and off every tile boundary, with and without the
chunked-prefill offset.

---

## Not done

- **Metal.** macOS runs on the CPU path. A Metal backend was not written
  because it can be neither compiled nor run from the machine this was
  developed on, and a few thousand lines of untested shaders would be a
  claim that cannot be stood behind. The seam exists:
  `include/vibevoice/device.h` declares the op set, a build links exactly one
  implementation of it, and `src/device/device_none.c` shows the shape.
- **CUDA Graphs.** Decode issues ~500 kernel launches per token. Capturing
  them would take a chunk out of the per-token floor.
- **A packed CPU micro-kernel.** CPU prefill runs at ~20% of peak FMA.
  Tiling the M and K loops was tried and measured slower; beating it needs a
  proper packed micro-kernel.
- **`src/trt/`** is stubs. The CUDA encoder does 11 s of audio in 243 ms, so
  TensorRT may never be worth it.
---

## Layout

```
include/vibevoice/   public headers; device.h is the accelerator op set
src/core/            allocator, logger, clock, threading shim
src/audio/           WAV, resample, normalize, ffmpeg bridge, mic, VAD
src/text_tokenizer/  byte-level BPE
src/tokenizer_encoder/  Conv-VAE speech tokenizers
src/model/           safetensors, config, weight loading and repacking
src/quant/           NF4 tables, AWQ repack, KV format registry
src/cpu/             CPU kernels (AVX2 / NEON / scalar)
src/cuda/            CUDA kernels
src/device/          the no-accelerator stub
src/inference/       pipeline, decoder, KV cache, sampling, post-processing
src/engine/          slot pool over one set of weights
src/server/          HTTP and the OpenAI-compatible API
cli/                 transcribe, serve, chat, mic
tools/               reference diffing, AWQ conversion, ONNX/TRT export
```

## Licence

MIT, matching the model. Third-party: cJSON (MIT). CUDA and TensorRT are
covered by the NVIDIA EULA.
