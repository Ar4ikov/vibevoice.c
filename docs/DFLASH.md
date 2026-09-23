# DFlash 2 speculative decoding

A drafter proposes the next few tokens and the target model checks all of
them in one pass, keeping the longest prefix it agrees with plus one token of
its own. The drafter here is a [DFlash 2](https://inco.ai/blog/dflash2/) block
drafter trained on the target's own transcripts; the check is **exact**: every
row of a checked block is computed with the arithmetic of a one-token decode
step, so a transcript with `--draft` is byte-for-byte the transcript without
it (`tests/test_spec.c`, and the `IDENTICAL` runs in the PR).

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
  activations), a SwiGLU output 2e4 feeding a 9472-wide down projection. The
  residual stream is FP32, and three exact rescalings keep every FP16
  intermediate in range: `fc` is stored ÷64 with the RMSNorm's ε ÷64², `up`
  ÷32 and `down` ×32/64, `o` ÷64, the two sublayer outputs added ×64
  (`vv_fp16` in config.json overrides the factors).
* **INT4 by default** (`--draft-quant int4`; `f16` keeps the trained
  weights): the projections are quantized at load into INT4 groups of 128
  (round to nearest, exact zero point) and run on the W4A16 kernels — 625 MB
  instead of 1679 MB for the 7B drafter, a draft pass that reads a quarter
  of the bytes, and the same drafts kept on test120 (1.39 against 1.40
  tokens per cycle). A draft is only a proposal, so quantizing it costs
  acceptance, never correctness. For the same reason the drafter's block
  attends on flashinfer's tensor cores whatever backend the target uses
  (0.21 ms a cycle less than the fa2 path on the 7B).
* **Draft vocabulary.** The drafter scores only the `draft_vocab` ids — the
  N most frequent in the transcripts it was trained on (32768 at most; the
  whole training corpus uses ~26 K distinct ids and 99 % of occurrences fall
  in ~22 K). Those rows of the target's head are gathered once at load, so
  the draft pass reads a fifth of the head. The target still checks against
  the full vocabulary.

## Checking a block exactly

The target processes the block `[anchor, d1 … d(Bv-1)]` at positions p… and
takes its own greedy token after every row. Kept: d1…dk while each equals the
target's token before it, then the target's own next token — k + 1 tokens for
one pass. If those tokens are to be the ones plain decoding would produce, the
logits of row i must be bit-for-bit the logits of the decode step that would
have produced them. The prefill kernels do not give that (different reduction
orders, tensor cores), so the check has its own kernels, each a multi-row copy
of the decode step's:

| Op | Decode step | Checked block |
|---|---|---|
| W4A16 projections | `w4a16_gemv_kernel`: LPR lanes per row, chunk c on lane c mod LPR, four HFMA2 chains per 32-weight chunk, FP32 flush, xor butterfly | `w4a16_gemv_rows_kernel`: the same lane→chunk map and chains for up to 8 x rows; x staged in shared memory, each weight chunk loaded and dequantized once for all rows |
| Attention | split-KV decode (`gqa_decode_kernel` / flashinfer `att_split`), split count from the cache length | the same kernels with a row index: row r sees cache length p + r + 1 and picks its own split count |
| LM head | `lm_head_gemv_kernel`: a warp per vocab row, lane l takes k = 8l + 256i | `lm_head_rows_kernel`: the same per row; the head streams past L1 (`ld.global.nc.L1::no_allocate`) so the x rows stay in L1 |
| Argmax | 256 partial blocks, lowest index on ties | the same per row |

Norms, RoPE, SwiGLU and the residual adds are row-wise already. With these, a
checked row and its decode step compute the same bits; `tests/test_spec.c`
compares every kernel against the one-row version bit for bit (7B and 1.5B
shapes, M 1..16, fa1/fa2/flashinfer, fp16 and tq4 caches, paged and slab).

What that costs is compute. The one-token GEMV is memory-bound with its FP16
pipe mostly idle; the same arithmetic for M rows is M times the FP16 work
(HFMA2 chains and half→float flushes cannot be shared between rows), and from
about 3 rows on it is the FP16 pipe, not the weights, that sets the time.
3090, 7B AWQ shapes, time of M rows over one row (`VV_SPEC_BENCH=1
tests/test_spec`):

| rows | q/k/v | o | gate/up | down | a layer |
|---|---|---|---|---|---|
| 2 | 1.18 | 1.23 | 1.05 | 1.12 | **1.09** |
| 3 | 1.20 | 1.38 | 1.10 | 1.34 | **1.20** |
| 4 | 1.41 | 1.63 | 1.23 | 1.54 | **1.36** |
| 8 | 2.28 | 2.61 | 2.19 | 2.68 | **2.4** |

Two kernels, one arithmetic: x in shared memory a tile at a time with each
lane group carrying 4 weight rows (`w4a16_gemv_rows_kernel`), and for 3–4
rows of a 3584-wide x a persistent one that loads all of x once per block
and walks every projection's rows in turn (`w4a16_gemv_rows_persist_kernel`,
3–8 % faster there, slower elsewhere). The head is 1.08× one row for 8 rows
(1.32 against 1.23 ms) once it streams past L1 — read through L1 it evicted
the x rows and took 1.75×. Decode attention is latency-bound, not
bandwidth-bound: rows run as separate grid layers (8 rows at 1K positions
2.05× one row; 4 rows at 24K 3.1×), and a variant that reads each cached
position once for all rows, with the same slices, was slower at every size
(fewer warps in flight) and was dropped.

So the runtime checks fewer rows than it drafts when that pays: the drafter
always drafts its whole block (it was trained on whole blocks), and
`--draft-block n` checks the anchor and the first n-1 drafts only.

`VV_SPEC_EXACT=0` checks with the prefill kernels instead (tensor-core
GEMMs): measurements only, since a near-tie can then flip.

## One cycle

```
cache and drafter context hold [0, p); anchor = token at p, not fed
  draft:  B rows over the drafter context            -> d1..d(B-1)
  check:  Bv rows [anchor, d1..] through the target  -> post[0..Bv-1]
          (writes K/V at p..p+Bv-1, taps for those rows)
  keep:   k = longest prefix with d(i+1) == post[i]; out = post[0..k]
  both caches end at p + k + 1; post[k] is the next anchor
```

The rows past p + k + 1 that the check wrote are simply overwritten later.
When a kept token ends generation (EOS, `<|im_end|>`, a streaming chunk's
`<|text_chunk_end|>`, or the token cap) the tokens after it were fed too;
one-shot transcription stops there anyway, and a streaming session cuts the
cache back (`decode_block` / `truncate` in `vv_stream_backend_t`), so the
next chunk's prefill sees exactly what plain decoding would have left.
`tests/test_stream.c` drives a session through blocks of 2..16 with stops
and caps landing inside blocks and checks tokens, text, stops, the fed
sequence and the cache length against one step at a time.

The drafter's context follows the target through the target's **taps**
(`vv_taps_t`, inference.h): a prefill (the prompt, a streaming chunk) or a
checked block copies the tapped layers' output rows aside, and the drafter
turns them into its keys and values. A plain decode step does not tap, so a
session that has to take one (the cache too full for a block) stops drafting
for the rest of that session.

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
4. **Check.** `tools/dflash/check_drafter.py` runs the trained model on a
   trace and compares its drafts with what the runtime drafted for the same
   positions (`VV_SPEC_LOG`); the C drafter agrees with PyTorch on 97–100 %
   of draft positions (the rest are BF16/FP16 near-ties).

## Results

TBD.

## Not supported

* The CPU path and Metal: the drafter and the multi-row check exist for CUDA
  only; `--draft` there logs a warning and decodes without it
  (`VV_ERR_UNSUPPORTED` from the device seam).
* A model split across devices (`--split-mode layer`): the taps would have
  to cross devices.
* Sampling (`temperature > 0`): the check is greedy.
