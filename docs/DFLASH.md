# DFlash 2 speculative decoding

A drafter proposes the next few tokens and the target model checks all of
them in one pass, keeping the longest prefix it agrees with plus one token of
its own. The drafter here is a [DFlash 2](https://inco.ai/blog/dflash2/) block
drafter trained on the target's own transcripts; the check is **exact**: every
row of a checked block is computed with the arithmetic of a one-token decode
step, so a transcript with `--draft` is byte-for-byte the transcript without
it on the same attention kernels (`tests/test_spec.c`, and the identical runs
in the PR). With a drafter `--attn auto` means flashinfer, which checks rows
together (below); `--attn fa2 --draft` keeps the batch model's default.

```
vv_cli --model ./model_hf --audio talk.wav --draft ./drafter
vv_cli serve --model ./model_hf --draft ./drafter --slots 2
```

## The drafter

DFlash 2 is a handful of Qwen3-style layers (q/k norms, no biases) that never
see text as a causal sequence. A draft is **one forward pass** over a block of
B rows, `[anchor, MASK × (B-1)]` at positions p..p+B-1, where the anchor is
the last token the target produced (not yet fed to it). Rows attend to each
other in both directions and to a context the drafter keeps of its own:

* **KV injection.** For every position the target has processed, the outputs
  of five of its layers (`target_layer_ids`, 1/7/13/19/25 of 28) are
  concatenated, projected by `fc` (5H → H) and normalized (`hidden_norm`);
  every drafter layer turns that into its own keys and values for that
  position. The drafter therefore reads the target's view of the audio, the
  prompt and everything generated so far, and its own context never runs
  through the drafter's layers.
* **Two-tap dynamic convolution** around the attention and the MLP of every
  layer: each row adds the row before it in the block, weighted per channel
  group by a kernel predicted from the row itself (`base_kernel` +
  `kernel_projection`, groups of 16 channels).
* **Candidate selector.** The target's LM head scores each MASK row; the top
  16 candidates are re-ranked by a pairwise term between the token chosen for
  the previous row and each candidate (`predecessor_codebook` ×
  `hidden_projection(h)` · `successor_codebook`, rank 256), walked greedily
  left to right (`vv_dflash_walk_dev`).

The embedding and the LM head are the target's, frozen. The checkpoint is a
`model.safetensors` + `config.json` pair with the names of the published
DFlash 2 drafters (`layers.N.self_attn.q_proj.weight`, …,
`candidate_selector.*`, `fc.weight`, `hidden_norm.weight`, `norm.weight`),
plus `draft_vocab` (below).

### On the device

* **FP16 with an FP32 residual.** A drafter trained in BF16 has activations
  FP16 cannot hold: the target's tapped features reach 1.3e4 (Qwen2's massive
  activations), a SwiGLU output 2e4 feeding a 9472-wide down projection, and
  the 7B drafter's own residual 3e6 in a few channels. The residual stream is
  FP32, and three exact rescalings keep every FP16 intermediate in range:
  `fc` is stored ÷64 with the RMSNorm's ε ÷64², `up` ÷32 and `down` ×32/1024,
  `o` ÷1024, the two sublayer outputs added ×1024 (`vv_fp16` in config.json
  overrides the factors). At ÷64 the 7B drafter's down projection overflowed
  to inf on some blocks; ÷1024 drafts the same (Streaming-1.5B, test120: 2.71
  against 2.68 tokens per block). Whatever the drafter computes, the ids it
  proposes exist: the top-k counts a NaN as −inf and the vocabulary map
  clamps, so a broken drafter costs drafts, never a request.
* **INT4 by default** (`--draft-quant int4`; `f16` keeps the trained
  weights): the projections are quantized at load into INT4 groups of 128
  (round to nearest, exact zero point) and run on the W4A16 kernels — 625 MB
  instead of 1679 MB for the 7B drafter, a draft pass that reads a quarter
  of the bytes, and the same drafts kept on test120 (1.39 against 1.40
  tokens per cycle). A draft is only a proposal, so quantizing it costs
  acceptance, never correctness. For the same reason the drafter's block
  attends on flashinfer's tensor cores whatever backend the target uses
  (0.21 ms a cycle less than the fa2 path on the 7B). Below sm_80 (or with
  `VV_W4A16_MMA=0`) there is no tensor-core W4A16 GEMM, and the INT4
  drafter expands each weight into a scratch before its GEMM (122 MB for
  the 7B's drafter, said at load); `--draft-quant f16` reads them as they
  are.
* **Stored INT4.** A drafter can also ship its projections quantized, the
  way compressed-tensors stores a pack-quantized W4A16 Linear
  (`weight_packed`, `weight_scale`, `weight_zero_point`, groups of 128):
  those codes go into the GPU layout as they are, and the log says
  `(stored quantized)`. `tools/dflash/awq_drafter.py` writes such a
  checkpoint with AWQ — activation-aware scales where they fold into the
  drafter without changing it (the MLP input, up → down, v → o per KV
  channel; not the attention input, whose k/v projections also read the
  context from the shared `hidden_norm`), then AWQ's clip search on every
  projection. On the Streaming-1.5B drafter's eval traces the accepted
  length is 3.094 in BF16, 3.080 with the runtime's round-to-nearest INT4
  and 3.090 with AWQ; the checkpoint is 234 MB instead of 454.
* **Draft vocabulary.** The drafter scores only the `draft_vocab` ids — the
  N most frequent in the transcripts it was trained on (32768 at most; the
  whole training corpus uses ~26 K distinct ids and 99 % of occurrences fall
  in ~22 K), plus every id that stopped a chunk or a transcript. Those are
  not in a transcript's tokens, and leaving them out cost more than it
  looks: a streaming chunk ends on `<|text_chunk_end|>` every ~10 tokens,
  and a drafter that cannot propose it loses every block that reaches a
  chunk's end (the first Streaming-1.5B drafter never cut a block at a
  stop). The rows of the target's head for these ids are gathered once at
  load, so the draft pass reads a fifth of the head. The target still
  checks against the full vocabulary.

## Checking a block exactly

The target processes the block `[anchor, d1 … d(Bv-1)]` at positions p… and
takes its own greedy token after every row. Kept: d1…dk while each equals the
target's token before it, then the target's own next token — k + 1 tokens for
one pass. If those tokens are to be the ones plain decoding would produce, the
logits of row i must be bit-for-bit the logits of the decode step that would
have produced them. So every op of the check computes each row with the
decode step's own arithmetic:

| Op | Decode step, and each row of a checked block |
|---|---|
| W4A16 projections | `vv_w4a16_mv_dev`: tensor cores (mma.m16n8k16) with the weights as the A operand, 16 output rows to a warp tile, and the rows of x as the columns of B. K is cut into 16 slices of 128-k blocks, each slice adds its blocks in order and the slices meet in a fixed halving tree, so a row's bits depend on its weight and its x alone — not on how many rows share the launch, which projections are fused into it, or the launch shape the kernel picks for the card. The step runs it with one row, the check with Bv. |
| Attention | split-KV decode — fa2's scalar `gqa_decode_kernel` or flashinfer's `att_split` — where row r sees p + r + 1 positions, cut with that length's own split. fa2 gives every row a grid layer of its own; flashinfer packs rows that cut the cache alike two or three to a 16-row fragment (7 heads a row), so a K/V tile is read once for all of them. |
| LM head | `lm_head_gemv_kernel`: a warp per vocab row, lane l takes k = 8l + 256i; `lm_head_rows_kernel` does the same per row and streams the head past L1 (`ld.global.nc.L1::no_allocate`) so the x rows stay there |
| Argmax | 256 partial blocks, lowest index on ties, per row |

Norms, RoPE, SwiGLU and the residual adds are row-wise already.
`tests/test_w4a16.c` checks the tensor-core GEMV's rows for M = 2..16 against
M = 1, four other launch shapes and every fused projection alone, bit for
bit, on the 7B and 1.5B shapes; `tests/test_spec.c` does the same for the
attention (fa1/fa2/flashinfer, fp16, fp8 and tq4 caches, paged and slab, up
to 24K positions, the length on the host and on the device), the head and
the argmax. Other weight formats check with the kernels a decode step runs:
dense FP16 and ternary take up to 8 rows at once, NF4 and INT8 one row at a
time (exact, slower), W8A8 any count (integer sums). W4A8 checks at most 8
rows: past what its GEMV takes, the activations go to the GEMM, which adds
the weight groups up in another order, so a larger `--draft-block` is cut to
8 (`vv_decoder_verify_rows_max`).

**The step changed to make the check cheap.** Until this release the W4A16
step was an FP16-chain GEMV (four HFMA2 chains per 32-weight chunk, flushed
to FP32), and the only exact check of M rows was M times its FP16 work: at 8
rows 2.6 steps on a 3090, and the check ate most of what a block saved. The
tensor-core GEMV reads the weights once for all rows and does the arithmetic
on the tensor cores, and because its rows are independent it can be the
step's kernel as well. Its sums are FP32 inside the mma, so plain W4A16
decoding (AWQ, GPTQ, `--quant int4`) has other bits than 0.5.1's —
`VV_W4A16_MV=0` gives the old GEMV, and the old multi-row check with it.
(Where the tensor-core GEMV declines — before sm_80, K % 128, N % 16 — both
sides keep the old pair.) Time of M rows over the old one-row GEMV, 3090,
7B layer (q/k/v, o, gate/up, down; `VV_W4A16_BENCH=2 tests/test_w4a16`):

| rows | old GEMV rows | tensor-core GEMV |
|---|---|---|
| 1 | 1.00 | 0.98 |
| 4 | 1.71 | 1.11 |
| 8 | 2.64 | 1.13 |
| 16 | 5.2 | 1.78 |

(The 1.5B layer: 0.90 for one row, 1.09 for eight.)

The attention is where exactness still costs, and only on a long cache. A
decode's walk over its slice is latency-bound — one dependent warp reduction
per position — so fa2's rows run as separate grid layers, and a block of 8 at
24K positions costs ~6.6 steps' attention (3070); a variant that walked each
slice once for all rows, same slices and same arithmetic, was no faster and
was dropped. flashinfer's rows share fragments: 1.64 steps' attention for 8
rows at 24K, 0.98 for 4. So with a drafter `--attn auto` is flashinfer (the
streaming models attend with it anyway): on the AWQ 7B and the 32-minute
file, exact blocks made 1.06× the plain decode on fa2 and 1.47× on
flashinfer. The transcript is then exactly that of `--attn flashinfer`
without a drafter, which on the AWQ 7B has the default's words on test120
and the 32-minute file, with one timestamp 10 ms apart on the latter.
`--attn fa2 --draft` keeps the default's transcript bit for bit, at fa2's
cost for the rows.

So the runtime checks fewer rows than it drafts when that pays: the drafter
always drafts its whole block (it was trained on whole blocks), and
`--draft-block n` checks the anchor and the first n-1 drafts only. Without
it the width is measured: half the block or all of it, whichever keeps more
tokens per millisecond. A full block of n tokens tells what a half one would
have kept (min(n, B/2)), each width's time is its own running average, the
narrow width is timed once and, while it wins, every 32nd block runs full so
what the longer runs would keep stays measured. Each width has its own
captured cycle. (3070, Streaming-1.5B, test120: 646 tok/s chosen, 637 at 8
rows, 505 at 4.)

`--draft-check fast` runs the same cycle with flashinfer's rows whatever the
cache's backend, and the prefill's projections instead of each row's decode
arithmetic: the same kernel for W4A16, a GEMM instead of a GEMV per row for
NF4 and INT8. Its logits then differ from the decode step's wherever those
kernels round differently (a fa2 target's attention), so a near-tie can go
the other way than in plain decoding. What comes out is still the model's
greedy transcript, computed with another rounding — the trade the usual
speculative decoders (vLLM, SGLang) make — but not byte for byte the one
without `--draft`. With W4A16 weights and flashinfer attention, fast and
exact are the same computation. `--draft-check exact` (the default) keeps
the promise.

## One cycle

```
cache and drafter context hold [0, p); anchor = token at p, not fed
  draft:  B rows over the drafter context            -> d1..d(B-1)
  check:  Bv rows [anchor, d1..] through the target  -> post[0..Bv-1]
          (writes K/V at p..p+Bv-1, taps for those rows)
  keep:   k = longest prefix with d(i+1) == post[i]; out = post[0..k]
  context: the Bv rows' taps into the drafter's context at p..p+Bv-1
  both caches end at p + k + 1; post[k] is the next anchor
```

The rows past p + k + 1 that the check and the context update wrote are
simply overwritten later. When a kept token ends generation (EOS,
`<|im_end|>`, a streaming chunk's `<|text_chunk_end|>`, or the token cap) the
tokens after it were fed too; one-shot transcription stops there anyway, and
a streaming session cuts the cache back (`decode_block` / `truncate` in
`vv_stream_backend_t`), so the next chunk's prefill sees exactly what plain
decoding would have left. `tests/test_stream.c` drives a session through
blocks of 2..16 with stops and caps landing inside blocks and checks tokens,
text, stops, the fed sequence and the cache length against one step at a
time.

**A cycle is one graph launch.** Everything that moves from one cycle to the
next — the anchor and both caches' lengths — the cycle reads from a small
device block it fills from pinned memory first, and the tokens it keeps and
their count go back to pinned memory last. Nothing else in it depends on p:
the rows' RoPE and K/V writes take a device position
(`vv_kv_cache_append_at`, `vv_dev_memcpy_d2d_rows_at`), the attention a
device length (`vv_attn_decode_rows`, and `vv_attn_block_dev` for the draft
pass), the rows the drafter's context gets are all Bv of them, kept or not.
So the cycle is captured once and replayed until one of its attention
launches changes shape (every 1024 positions) — one launch instead of ~600.
What that is worth depends on the driver: on a Windows (WDDM) 3070 the
Streaming-1.5B cycle launched kernel by kernel made 1.79× plain decoding and
replayed makes 2.76×; on a Linux 3090, 4 % (`VV_SPEC_GRAPH=0` launches every
kernel).

The drafter's context follows the target through the target's **taps**
(`vv_taps_t`, inference.h): a prefill (the prompt, a streaming chunk), a
checked block, or a plain step copies the tapped layers' output rows aside,
and the drafter turns them into its keys and values. A plain step taps into
row 0 of a one-row buffer, which keeps it capturable as a graph of its own.

## Blocks only while they pay

How many drafts are kept depends on the audio. On speech like the drafter's
corpus a block brings 2–3 tokens; on a language it barely saw, about 1, at
the price of roughly two steps. So every decode position asks
`vv_spec_want_block()`: the first three positions of a sequence are plain
steps (the step's cost, measured; the first one captures the tapped step's
graph and is not counted), then blocks, while their running tokens per
millisecond beat a step's; when they fall below, blocks pause for a run of
plain steps (16, doubling up to 512 while blocks keep losing) and are tried
again. While blocks pay, one plain step every 64 blocks measures a step
again, at the length the blocks are running at, and a step time 1.5× the
estimate (a new capture, a hiccup) is set aside unless three in a row say
the cost has moved. The tokens are the same either way; `spec: …, N plain
steps, M pauses` in the log says what happened. A block that finds no
pages in a shared pool (a live session does not wait for them) is plain
steps for a while too: a step needs one page at most.

On the 43 held-out clips with the Streaming-1.5B drafter this is what keeps
the worst clips (Mandarin and Russian, which the drafter saw little of) near
plain speed — 0.93–0.95× — instead of paying for blocks that keep one token.

On a card shared with other busy processes blocks lose more to time slicing
than steps do (before cycles were captured, the same clip measured 0.23×
with a training run on the card and 0.95× without). The numbers below are
from an otherwise idle card.

## Training

Self-distillation: the drafter learns to predict what *this* target writes,
timestamps and speaker ids included.

1. **Transcripts.** `vv_dflash_data gen` runs the target greedily over a
   corpus (`tools/dflash/build_corpus.py`: LibriSpeech, FLEURS ×8 languages,
   SOVA, AMI, earnings calls, VoxConverse; 2912 training clips, ~140 h, and
   43 held-out) and writes the token ids, per chunk for streaming models.
2. **Traces.** `vv_dflash_data trace` replays each transcript through the
   runtime — the one-shot prompt and its transcript in one prefill, or a
   streaming session with every chunk's tokens forced — and writes, per cache
   position, the token, its role (context / generated / label) and the five
   tapped layers' FP16 outputs (`VVDT` files; ~87 MB for a 3-minute clip on
   the 7B). With `--ring N` it keeps N files ahead of a trainer that deletes
   what it read.
3. **Training.** `tools/dflash/train.py` (PyTorch, flex attention): anchors
   are generated positions; a block's rows 1..B-1 are scored against the
   next B-1 tokens with weights e^(−k/4), cross-entropy of the (subset) head
   plus the selector's cross-entropy over the top-16 candidates; AdamW 6e-4,
   β (0.9, 0.95), 4 % warm-up, cosine to 10 %, clip 1.0, BF16 autocast.
   The cosine runs over the steps the traces actually make (`--steps` is
   cut to what `--epochs` passes give): a schedule cut off by the data
   leaves the last checkpoint at a high learning rate, as the first 7B
   drafter found out (step 359 of 680, lr at 55 % of peak).
4. **Check.** `tools/dflash/check_drafter.py` runs the trained model on a
   trace and compares its drafts with what the runtime drafted for the same
   positions (`VV_SPEC_LOG`); the C drafter agrees with PyTorch on 97–100 %
   of draft positions (the rest are BF16/FP16 near-ties).
   `tools/dflash/eval_types.py` splits depth-1 accuracy by the kind of token
   (JSON structure, digits, words), which is where a drafter's weakness shows.

The whole pipeline for one target, as it was run for the drafters in this
repository's release (paths are examples; `vv_dflash_data` is built with the
CLI):

```bash
# clips: WAV + manifest (audio only; the datasets' own text is never used)
python tools/dflash/build_corpus.py --src datasets/ --out clips/ --hours 140

# the target's own transcripts (resumable; --shard i/n splits across cards)
vv_dflash_data gen --model ./model_hf --list clips/train.tsv --out gen.jsonl
vv_dflash_data gen --model ./model_hf --list clips/eval.tsv  --out eval.jsonl

# traces: every position's token, role and five tapped layers (FP16)
vv_dflash_data trace --model ./model_hf --gen eval.jsonl --dir traces/eval \
    --layers 1,7,13,19,25
vv_dflash_data trace --model ./model_hf --gen gen.jsonl --dir traces/train \
    --layers 1,7,13,19,25          # or --ring 16 --epochs E next to a trainer

# the drafter: --out keeps the best one on the held-out traces
python tools/dflash/train.py --target ./model_hf --train-dir traces/train \
    --eval-dir traces/eval --out drafter --layers 1,7,13,19,25 \
    --epochs 5 --steps 700 --vocab-from gen.jsonl

vv_cli --model ./model_hf --audio talk.wav --draft drafter

# optional: the same drafter stored as INT4 with AWQ scales
python tools/dflash/awq_drafter.py --target ./model_hf --draft drafter \
    --calib traces/train --eval traces/eval --out drafter-awq
```

A 7B target's traces take ~87 MB per three-minute clip, so a full corpus
is ~250 GB; `--ring` keeps only N of them on disk and replays the
transcripts again each epoch instead.

## Results

RTX 3090 (Linux, CUDA 12.4, the card otherwise idle), greedy decoding,
`--draft-check exact`, decode tokens per second; plain is the same binary
without `--draft`. Every drafted transcript below was compared with its plain
one byte for byte, and was the same.

**VibeVoice-ASR 7B, AWQ W4A16** (`Ar4ikov/VibeVoice-ASR-DFlash2-Drafter`,
commit a9cfb15, the check width measured as it runs):

| | plain, fa2 (the default) | plain, flashinfer | drafted (flashinfer) | tokens per block |
|---|---|---|---|---|
| jfk, 11 s | 149.6 | 152.2 | 384.1 (**2.57×**) | 3.92 |
| test30 | 149.2 | 152.4 | 325.5 (**2.18×**) | 2.85 |
| test120, 2 speakers | 143.1 | 150.6 | 310.0 (**2.17×**) | 2.74 |
| 32-minute file | 106.1 | 122.7 | 183.5 (**1.73×**) | 2.36 |
| 20 held-out clips | 138.2 | 147.3 | 373.3 (**2.70×**) | 3.56 |

(× against the default. The clips' fa2 figure is from 7d7d43e, whose plain
decode is the same code.) The drafter stored as INT4
(`-AWQ-W4A16-ASYM`, 0.55 GB) keeps 3.747 tokens per block on the held-out
traces against the BF16 one's 3.741 and runs the same: clips 375.0, test120
306.0, the 32-minute file 185.8.

**VibeVoice-ASR-Streaming-1.5B, AWQ W4A16** (`Ar4ikov/VibeVoice-ASR-Streaming-1.5B-DFlash2-Drafter`,
7d7d43e, 8 rows; flashinfer is this model's default either way):

| | plain | drafted | tokens per block |
|---|---|---|---|
| test120 | 415.6 | 1040.2 (**2.50×**) | 3.99 |
| 32-minute file | 341.4 | 728.6 (**2.13×**) | 4.06 |
| 20 held-out clips | 409.2 | 869.4 (**2.12×**) | 3.52 |

**Other weights of the 7B, the same drafter**, test30, 7d7d43e, fa2, 8 rows:
BF16 56.5 → 90.6 (1.60×), W8A8 98.6 → 237.8 (2.41×), W4A8 146.3 → 232.4
(1.59×). NF4 and INT8 check exactly only one row at a time and lost (NF4,
test120: 123.9 → 112.3), so the drafter is declined there in exact mode;
`--draft-check fast` reads each weight once through the small-M kernel
(b72be15): NF4 test30 127.0 → 213.9 (1.68×), test120 122.3 → 149.8 (1.22×);
INT8 test30 98.2 → 176.9 (1.80×), test120 95.6 → 174.7 (1.83×) -- the
transcripts happened to come out the same, which fast does not promise.

**BitNet**: 0.99× on the 20 clips -- its drafter does not learn (held-out
depth-1 accuracy 0.24 against 0.80 for the others) and the controller keeps
the blocks paused. Not published.

Where the numbers come from: the 7B's check costs 1.3 steps at 8 rows (the
projections 1.13, then the head, the rows' attention and the drafter's own
pass), so the speedup is about tokens per block / 1.3. The drafters are
data-starved -- on the 7B, 5.4 tokens per block on the training traces
against 3.7 held out -- which is where the next ones come from.

## Not supported

* The CPU path and Metal: the drafter and the multi-row check exist for CUDA
  only; `--draft` there logs a warning and decodes without it (the device
  seam declines the drafter's kernels with `VV_ERR_UNSUPPORTED`, asked once
  before anything loads).
* A model split across devices (`--split-mode layer`): the taps would have
  to cross devices.
* Sampling (`temperature > 0`): the check is greedy.
* Replaying fa1's cycles: its rows run the one-row kernel row after row on
  the host's lengths, so with `--attn fa1` (or before sm_75, where nothing
  else resolves) every cycle launches its kernels. Exact all the same.
