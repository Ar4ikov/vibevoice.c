# VibeVoice-ASR-BitNet: formats, numerics and the reference runtime

[microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)
is the edge build of VibeVoice-ASR: the same Conv-VAE front end, a
Qwen2-1.5B-shaped language model whose seven projections are ternary
(BitNet b1.58), and an int8 speech encoder. Its reference runtime is
[microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp), a ggml
fork. This page records exactly what that runtime computes, read from its
source at commit `c433400` with the ggml fork at `XsquirrelC/llama.cpp@a2fdadc`,
so this runtime can match it and then beat it. Everything marked *verified*
was checked against the real files by `tests/test_gguf.c` or
`tools/bitnet_ref.py`.

## Model shape

| | |
|---|---|
| LM | Qwen2: hidden 1536, 28 layers, 12 query heads / 2 KV heads (GQA 6), head_dim 128, intermediate 8960, vocab 151936, RoPE θ = 1e6, RMSNorm ε = 1e-6, q/k/v biases, SwiGLU |
| Ternary | q, k, v, o, gate, up, down of every layer (196 tensors); biases, norms stay FP32 |
| Embedding | tied (`tie_word_embeddings: true`); the GGUF stores it twice, Q6_K for the lookup and F16 for the head |
| Encoders | acoustic (vae_dim 64) and semantic (128), same Conv-VAE as the 7B model; the connectors project to 1536 |
| Checkpoint | F32 *latent* weights in safetensors (11.3 GB, 1177 tensors): ternarization happens in the converter |

The latent F32 projections are not what the model runs: BitNet trains through
the ternary forward, so running a layer on the latent weights gives cosine
0.63 (layer 0), 0.27 (layer 14) and −0.08 (layer 27) against the ternary
layer (`bitnet_ref.py lm-layer`). The F32
safetensors are only an input to ternarization.

## Files and type ids

| File | Size | Contents |
|---|---|---|
| `vibeasr-lm-i2_s-embed-q6_k.gguf` | 993 MB | 196 × `I2_S` (type **36**), `token_embd.weight` `Q6_K` (14), `output.weight` `F16` (1), 141 F32 norms and biases; `general.architecture = qwen2`, `general.file_type = 40` |
| `vibeasr-vae-encoder-i8_s.gguf` | 703 MB | 176 × `I8_S` (type **37**, every conv and linear weight of both encoders and both connectors), 386 F32 biases, norms, layer scales; `general.architecture = vibeasr-vae`, `general.file_type = 41` |

GGUF v3, alignment 32. `ne[0]` is the contiguous dimension (a Linear(in=K,
out=N) has `ne = {K, N}`). `src/model/gguf.c` reads both (verified: 339 and
562 tensors, every extent inside the file). The fork's type ids 36/37 collide
with nothing in upstream ggml of that era but are **not** upstream types.

### I2_S (ternary)

Byte layout, x86 build of the fork (`ggml-lm-mad.cpp: quantize_i2_s`, the
`ACT_PARALLEL` branch — `lm-config.h` defines it):

* the tensor is flattened row-major and cut into blocks of **128** elements;
  each block is **32 bytes**;
* byte `j` of a block holds the codes of elements `j`, `j+32`, `j+64`, `j+96`
  in bits 7:6, 5:4, 3:2, 1:0;
* code `0 → −1`, `1 → 0`, `2 → +1` (3 never occurs);
* after the `n/4` code bytes comes **one FP32 scale for the whole tensor**,
  and the tensor is padded by 32 bytes (`ggml_nbytes = n/4 + 32`).

Value = `(code − 1) · scale`. The ARM build of the fork uses the same byte
layout (its `quantize_i2_s` NEON branch is identical; the NEON dot product
simply walks it 16 bytes at a time with `I2S_Y_BASE`), so the GGUF is
portable. This runtime uses the I2_S layout as its own ternary layout, so the
GGUF bytes are used as-is.

### Ternarization (converter formula)

`utils/convert_lm_to_gguf.py: quant_weight_fp16`, then `quantize_i2_s`:

```
s     = 1 / max(mean|W|, 1e-5)              per tensor, torch FP32
W'    = round(W · s).clamp(−1, 1) / s       torch.round = half to even
scale = max|W'|                             (= fl(1/s) for any non-zero tensor)
code  = |W'| < 1e-6 ? 1 : (W' > 0 ? 2 : 0)
```

**Scale granularity: one per tensor.** `vv_ternarize_f32` implements this;
against the shipped GGUF it reproduces 140,378,108 of 140,378,112 codes over
21 tensors (layers 0, 13, 27) and the scale to within 1 ulp (11 of 21 equal).
The 4 codes and the ulps come from `mean|W|`: torch reduces in FP32, this
runtime in FP64; the flipped codes sit exactly on the rounding boundary.
`bitnet_ref.py check-gguf` measures the same with numpy (3 codes of 93.6 M
for layers 0 and 27 with an FP64 mean, 5 with an FP32 pairwise mean). The
GGUF is the source of truth; the F32 route is for when only safetensors are
available.

### Activations of ternary layers (W1.58A8)

`ggml-quants.c: quantize_row_i8_s`, applied to the F32 input of **every**
I2_S matmul (the RMSNorm output for q/k/v and gate/up, the attention output
for o, the SwiGLU output for down):

```
amax = max(1e-5, max_k |x_k|)       in double
s    = (float)(127 / amax)          per token (per row)
q_k  = clamp(nearest_int(x_k · s), −128, 127)    x·s in FP32, round half to even
sum  = Σ q_k
```

`nearest_int` is the 1.5·2²³ trick, i.e. round-half-to-even. The dot product
(`ggml_vec_dot_i2_i8_s`, AVX2 `maddubs` on the unsigned codes 0..2) returns
`Σ code·q`, and `ggml_compute_forward_mul_mat` finishes with

```
y = ((float)Σ code·q − (float)sum) / s · scale        then + bias (ggml_add)
```

Both integers are below 2²⁴, so this equals `(float)(Σ (code−1)·q) / s ·
scale` exactly. `vv_ternary_linear_cpu` and `vv_ternary_gemm_dev` evaluate
that same expression in that order and are bit-identical to it (tested
against a transcription of the ggml code). Two reference quirks, both
harmless: the AVX2 kernel carries int16 partial sums over 32 blocks (4096
weights), which can only overflow for adversarial inputs; this runtime widens
every 16 blocks and is exact for any input.

What stays floating point: RMSNorm (sum of squares in double, then
`1.0f/sqrtf(mean + eps)`), RoPE (NEOX style), attention, SwiGLU, residuals,
all in FP32 — except that llama.cpp's KV cache is **F16** (`type_k = type_v =
F16`, flash attention off), so K and V are rounded to FP16 and the `KQ`
product converts Q to FP16 as well. `bitnet_ref.py lm-layer --f16-attn`
models this.

Measured effect of the int8 activations alone (layer output, 64 random
embedding rows, W1.58A8 vs the same ternary weights with float activations):
cosine 0.99933 / 0.99924 / 0.99868 after layers 0 / 14 / 27 run in
isolation. Against the latent F32 weights the same layers give 0.63 / 0.27 /
−0.08: the latent weights are not a usable model on their own.

### Embedding and head

* `token_embd.weight` is **Q6_K**: blocks of 256, `ql[128] qh[64] scales[16]
  (int8) d (fp16)`, `y = d · scale · (q − 32)`. `get_rows` dequantizes the
  looked-up rows to F32. Relative RMS error against the F32 table: 1.97 %.
  `vv_q6k_dequant_row` follows ggml's operation order, so gathered rows are
  bit-identical.
* `output.weight` is a separate **F16** copy of the same table (verified: row
  151648 equals the F32 embedding rounded to F16). The head is an ordinary
  F16 `mul_mat`: the activation is converted to F16, products accumulate in
  FP32. It is 467 MB — the largest read of every decode step.
* This runtime keeps an **int8 row-quantized copy** of the head (233 MB) on
  the CPU, as a filter in front of the F16 one: every row's int8 logit comes
  with a rigorous bound on its distance from the F16 logit (see "The head"
  below), only rows that could still win are scored in F16, and the token
  is the full F16 scan's, ties included. `--head int8` uses the int8 rows
  alone (approximate); `--head f16` scans the whole F16 table.

### I8_S (int8, the encoder)

Weights (`quantize_i8_s`): **one FP32 scale per tensor**,
`q = clamp(roundf(w · 127/amax), −127, 127)` (roundf: ties away from zero),
stored `scale = amax/127` after the data, +32 bytes padding. Verified: all
176 I8_S tensors of the GGUF are reproduced bit for bit from the F32
safetensors (`bitnet_ref.py check-gguf`, and three of them in
`test_gguf`).

Conv kernels are zero-padded **at the front** to a SIMD-friendly length by
`convert_vae_to_gguf.py`: the depthwise mixers and the stem 7 → 8, the stride-4
downsample 8 (unchanged), stride-5 downsamples 10 → 16, stride-8 16, head
7 → 8. With the padding taken from the padded length (`k − stride`) the
convolution is unchanged.

Activations: the int8 encoder is a **fully int8 pipeline with one scale per
activation tensor**. Every op (`ggml_mul_mat_add`, `ggml_rms_norm_scaled`,
`ggml_add_scaled`, `im2col_asym`) produces int8 plus a single multiplier
`127 / max|y|` taken over the **whole tensor — all channels and all time
steps**. Consequences:

* the output for one frame depends on the loudest value anywhere in the
  file, so this encoder is not streamable as written;
* requantization: `q = cvtps(clamp(y · 127/amax, −127, 127))` (round half to
  even; the scalar tails of `rms_norm_scaled`/`add_scaled` use `roundf`,
  ties away, and `mul_mat_add` tails `rintf` — only exact .5 ties differ);
* `add_scaled` splits the flat tensor into `-t` equal ranges; each range's
  scalar head (up to the next row start) and tail (its last `len % 8`
  elements) compute `a / s_a · γ + b / s_b` with divisions and requantize
  the tail with `roundf`, while the 8-wide body multiplies by `1/s_a`,
  `1/s_b`. **The encoder's output therefore depends on the thread count**;
  this runtime repeats the partition, so equal thread counts give equal
  features;
* the build is `-march=native` with GCC's default `-ffp-contract=fast`, and
  GCC contracts the intrinsic `_mm256_add_ps(_mm256_mul_ps(…))` pairs as
  well as plain C: in `libggml.so` the `mul_mat_add` epilogue is
  `fma((float)acc, w_scale/x_scale, bias)` (vector body and scalar tail),
  the `add_scaled` body `fma(b, 1/s_b, (a · 1/s_a) · γ)` and its scalar
  parts `fma(a / s_a, γ, b / s_b)`. Without the fused form the stem's
  output scale already differs in the last bit, and the error compounds to
  a feature cosine of 0.98;
* the input audio itself is quantized: `s = 127/max(|a|, 1e-5)`,
  `q = clamp(roundf(a·s), −128, 127)`, i.e. **8-bit audio**.

Per block (`src/vae.cpp: ConvNeXtBlock::forward`, I8_S branch):

```
h = rms_norm_scaled(x, norm.weight)      eps 1e-5, rms from int8 sum of squares
h = depthwise conv, k 8 (7 + front pad), left pad 7, + bias
x = h / s_h · gamma + x / s_x            (add_scaled, then requantized)
h = rms_norm_scaled(x, ffn_norm.weight)
h = ReLU(linear1(h) + b1)                <-- ReLU, not GELU
h = linear2(h) + b2
x = h / s_h · ffn_gamma + x / s_x
```

Linear/conv: `y = fma((float)Σ w·a, w_scale / a_scale, bias)`, then the
global-absmax requantization. `src/tokenizer_encoder/vae_i8.c` implements
the whole encoder this way; `tests/test_bitnet_vae.c` checks it against an
op-by-op transcription at several thread counts and, with the model,
against a hash of VibeASR.cpp's features.

Two deliberate departures from the HF model in the reference's int8 path:

* **ReLU instead of the exact GELU** in every encoder FFN (the F32/F16 GGUF
  path of the same runtime uses GELU; HF uses `ACT2FN["gelu"]`);
* the connector RMSNorm uses **ε = 1e-5** (hard-coded in `ggml_nn_rms_norm`)
  where HF's `SpeechConnector` uses 1e-6, and the GGUF metadata says
  `vae.rms_norm_eps = 1e-6`.

How far the reference's int8 encoder is from the F32 one (`bitnet_ref.py vae`,
first 4 s of jfk.wav, 30 frames, cosine against the F32 safetensors with
GELU):

| tower | int8 + ReLU (reference) latent / connector | int8 + GELU latent / connector |
|---|---|---|
| acoustic | 0.876 / 0.926 | 0.934 / 0.933 |
| semantic | 0.953 / 0.970 | 0.964 / 0.976 |

So the int8 pipeline already costs more than the ReLU swap does; matching the
reference's transcripts needs its exact int8 encoder, while an FP16 encoder on
the I8_S weights is closer to the model as trained. This runtime offers both
and picks per backend; see "Which encoder" below.

The connectors (`fc1 → RMSNorm → fc2`) live in the VAE GGUF and are int8 too;
the encoder emits 1536-wide features directly and the acoustic and semantic
outputs are added.

### Acoustic latent

No sampling: the acoustic encoder output goes through its connector as the
mean. (`fix_std = 0.5` in the config is unused by the reference.)

### Frame count

ggml's convolutions have no right padding, so each strided conv yields
`floor` rather than HF's `ceil` frames: 11.0 s (264,000 samples) gives 82
frames, not 83. The prompt still reserves `ceil(n/3200)` = 83 `speech_pad`
tokens; `prefill_segmented` feeds only the 82 frames it has and continues the
positions from there, so the last pad token is never fed (the prompt is 129
positions, reported as "130 tokens"). Decoding then starts at
`cur_pos = n_prompt_tokens` = 130, not at 129: every generated token is
rotated one position further than the cache row it lands in (llama.cpp
finds the first free cell, so rows stay contiguous and only RoPE sees the
gap). This runtime builds the prompt with the 82 pads it feeds and carries
the gap in `vv_kv_cache_t.rope_gap` on the CPU, so its decode positions are
the reference's (the GPU path keeps positions contiguous).

## Prompt and decoding

`utils/prompt_builder.h`, assembled from token ids (the special ids are
hard-coded: 151644 `<|im_start|>`, 151645 `<|im_end|>`, 151646/7/8 speech
start/end/pad):

```
<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n
<|im_start|>user\n<|speech_start|><|speech_pad|>×N<|speech_end|>\nThis is a {dur:.2f} seconds audio, please transcribe it.<|im_end|>\n
```

The default `--prompt-format text` asks for **plain text** ("please
transcribe it."), not the 7B model's JSON keys; `json` restores
"...with these keys: Start, End, Speaker, Content". No generation prompt; the
model writes `<|im_start|>assistant\n` itself and the reference strips it.
Stops on 151645 or 151643.

Decoding defaults to **sampling** (top-k 40, top-p 0.9, temperature 0.7,
seed 42); `--greedy` gives argmax. All numbers below are `--greedy`, which
is what this runtime does.

The model directory's `tokenizer.json` is Qwen2's plain one (three added
tokens, no `<|object_ref_start|>` & co.), so the `asr-bitnet` family falls
back to the same hard-coded 151646/7/8 when the tokenizer lacks them, and
builds exactly the prompt above (the tokenizer splits the text segments the
way the reference's per-segment `tokenize` calls do; the ids are equal).

Audio: `dr_wav`, linear-interpolation resampling to 24 kHz, RMS
normalization to −25 dBFS (`scalar = 10^(−25/20) / (rms + 1e-6)` with the RMS
taken as a float). The int8 encoder quantizes the whole clip with one scale,
so a gain that differs in the last bit already moves int8 samples; the
`asr-bitnet` family therefore prepares audio exactly this way
(`vv_audio_prepare_vibeasr`) instead of with the windowed-sinc resampler
and double-precision gain the other families use.

## Ground truth on gpubox

VibeASR.cpp built on gpubox (GCC 11.4, `-march=native` → AVX2 + FMA on the
Ryzen 9 5900X, OpenMP), CPU only — the reference has no GPU backend for
I2_S/I8_S (`n_gpu_layers = 0`, the VAE uses the CPU backend).

Transcripts (`--greedy -t 12`):

| file | reference transcript |
|---|---|
| jfk.wav (11 s) | And so my fellow american, ask not what your country can do for you. Ask what you can do for your country. |
| test30.wav (30 s) | And so, my fellow american, ask not what your country can do for you. Ask what you can do for your country. He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour, fat and sauce. And so my fellow american, ask not what your country can do for you. |
| test120.wav (120 s, 2 speakers) | And so, my fellow american, ask not what your country can do for you. Ask what you can do for your country. He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour, fat and sauce.<br>And so my fellow american, ask not. |

(test120 stops after the first ~40 s of content: the reference's own
behaviour with greedy decoding on this file.)

RTF on test30.wav by thread count (`--greedy --warmup`, best of 3, through
`cpu-bench`):

| threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| RTF | 1.144 | 0.624 | 0.448 | 0.369 | 0.307 | 0.389* | 0.264 |
| VAE (A+S), s | 27.2 | 14.6 | 10.2 | 8.1 | 6.5 | 7.5* | 5.3 |
| prefill, s | 3.46 | 1.82 | 1.24 | 0.97 | 0.69 | 0.61 | 0.62 |
| decode 83 tok, s | 3.72 | 2.30 | 2.00 | 1.86 | 1.96 | 1.98 | 1.99 |

\* the box was under load (load average > 100) during the 8-thread runs.
test120 at 12 threads: RTF 0.217 (VAE 22 s of 26 s). jfk at 12 threads,
no warm-up: RTF 0.280.

Where the reference's time goes at 12 threads on 30 s: **~67 % in the int8
VAE** (5.3 s), 8 % prefill, 25 % decode (24 ms per token, of which the F16
head alone is an estimated ~9 ms at 50 GB/s). Decode stops scaling at 3 threads — it is
memory-bound on 467 MB (head) + 328 MB (ternary layers) per token.

## This runtime's kernels (phase 1)

All exact in int32 and bit-identical across ISAs and to the GPU
(`tests/test_bitnet.c`, run natively, with `VV_BITNET_MMA=0`, and under
`qemu-aarch64` with and without `+dotprod`). Checked on gpubox (GCC 11,
CUDA 12.4, RTX 3090, sm_86; the `.cu` also compiles for sm_75) and on
Windows (MSVC 17, CUDA 13.4, RTX 3070). The AVX-VNNI and AVX512-VNNI paths
compile everywhere but have not run on hardware that has them: the 5900X is
Zen 3 and neither box has VNNI, so the test exercises them only where CPUID
reports them.

| | CPU | GPU |
|---|---|---|
| ternary × int8 | AVX2 `maddubs` (widened every 16 blocks), AVX-VNNI / AVX512-VNNI `vpdpbusd` (runtime CPUID), NEON `sdot` or `smull`+`sadalp`, scalar | dp4a GEMV (M ≤ 8, codes unpacked with one shift + mask per 16 weights); `mma.m16n8k32.s8` GEMM with the 2-bit codes kept in shared memory and unpacked into B fragments in registers (sm_80+); GEMV loop on sm_75 |
| int8 × int8 | sign trick on the activation side (exact for a = −128), VNNI, NEON | same GEMM with an int8 B tile |
| activation quant | per token, as ggml | same, FP32 or FP16 input |
| head | int8 rows × int8 activation, fused argmax, ties → lowest id | dp4a + two-stage argmax |

Throughput, Ryzen 9 5900X (AVX2, 12 threads pinned one per core; the
single-matrix numbers are cache-warm, whole-model decode will be
DRAM-bound):

| shape (N×K) | M=1 µs | GB/s | M=512 ms | GOP/s |
|---|---|---|---|---|
| q/o 1536×1536 | 7.0 | 84.6 | 1.82 | 1327 |
| k/v 256×1536 | 2.6 | 38.1 | 0.36 | 1114 |
| gate/up 8960×1536 | 33.3 | 103 | 10.25 | 1375 |
| down 1536×8960 | 23.4 | 147 | 8.33 | 1692 |
| int8 head 151936×1536 | 4.51 ms | 51.8 | | |

One layer at M=1 is 109 µs (3.1 ms for 28 layers cache-warm; the full 328 MB
from DRAM at ~45 GB/s is ~7 ms), plus 4.5 ms for the int8 head: ~12 ms per
token against the reference's 24. On one thread the ternary GEMV streams
15–24 GB/s and prefill runs 174–190 GOP/s.

RTX 3090:

| shape (N×K) | M=1 µs | GB/s | M=2048 ms | TOP/s |
|---|---|---|---|---|
| q/o 1536×1536 | 4.2 | 139 | 0.259 | 149 |
| k/v 256×1536 | 3.8 | 26 | 0.058 | 111 |
| gate/up 8960×1536 | 8.4 | 409 | 1.427 | 158 |
| down 1536×8960 | 10.7 | 321 | 1.118 | 202 |
| int8 head 151936×1536 | 0.270 ms | 864 | | |
| int8 GEMM 2048×8192×2048 | | | 0.703 | 98 |

The M=1 layer is 44 µs (1.22 ms per token for 28 layers), launch-bound on
the small shapes; graphs and fusion (phase 2) are the lever there.

`vv_bench_bitnet [cpu|gpu] [M]` reproduces these.

## Tools

* `tests/test_gguf.c` — synthetic files, every truncation and corruption
  refused, and with `VV_BITNET_MODEL=<dir>` the cross-checks above.
* `tools/bitnet_ref.py check-gguf | lm-layer | vae` — the numpy fake-quant
  reference: the ternarization and I8_S checks against the GGUF, one LM layer
  in W1.58A8 (reference numerics) vs W1.58A16 vs latent F32 with `--dump`
  for tensor-by-tensor comparison, and the encoder in F32 / int8+ReLU /
  int8+GELU.

## Phase 2 (integration) notes

* Loader: family `asr-bitnet` (issue #15 adds families and F32 → FP16 for
  the dense tensors). GGUF pair via `vv_gguf_open`; I2_S tensors map
  zero-copy onto the ternary layout; `blk.N.attn_q ↔
  model.language_model.layers.N.self_attn.q_proj` etc.
* Quantize each input once and feed q/k/v (and gate/up) from the same int8
  buffer. The quantizer here (`vv_act_quant_i8_{cpu,dev}`) and the one of
  #18 (`vv_quant_act_q8_cpu`, `vv_act_quant_dev`, branch `w8a8-w4a8`) should
  be unified. They agree on per-token absmax and round half to even, and
  differ in details that change bits: #18 multiplies by `127.0f/amax`
  computed in FP32, returns `amax/127` as the scale and clamps to ±127; ggml
  (and this branch) computes `(float)(127.0/amax)` in double with a 1e-5
  floor, divides by that multiplier in the epilogue, clamps to [−128, 127]
  and returns Σq (the ternary kernels need it for the code−1 offset).
  #18 additionally emits per-32 sums and a nibble-interleaved layout. A
  shared quantizer needs a mode flag for the scale formula; the BitNet mode
  must keep ggml's to match VibeASR.cpp. The symbol names do not collide.
* The encoder: decide between reproducing the int8+ReLU pipeline (matches
  the reference, not streamable) and running the I8_S weights dequantized
  with GELU (matches the model, streamable); `bitnet_ref.py vae` measures
  both against F32.
* The prompt is the text format, not JSON; the postprocessor must accept
  plain text.
