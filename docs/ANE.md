# The Neural Engine, and what actually runs fast on Apple Silicon

Every M-series chip has three engines that can do this work: the GPU, the
CPU's matrix coprocessor (AMX, SME on M4), and the Apple Neural Engine. This
runtime uses the GPU. That is a measured decision, not an oversight, and the
measurements are here so the next person does not have to repeat them.

Machine: MacBook Air, M4, 8 GPU cores, 16-core Neural Engine (38 TOPS
claimed), 16 GB unified memory, macOS 26.4, Core ML via
`MLComputeUnits` with the model compiled by `MLModel compileModelAtURL:`.
Each row is the mean of 10-50 runs after a warm-up.

---

## 1. How the Neural Engine is reached at all

There is no public API that runs a kernel on the ANE. The only door is
**Core ML**: a graph and its weights are compiled into an `.mlmodelc`, and
the runtime decides per operation whether it goes to the ANE, the GPU or the
CPU. `MLComputePlan` reports that decision, which is how the placements
below were read rather than guessed.

Three consequences follow immediately, before any timing:

- **The weights live inside the model.** They are baked into the compiled
  artifact, not loaded from a checkpoint at run time. A 7B in FP16 is 15 GB
  of `.mlmodelc`.
- **The graph is fixed at compile time.** Shapes are enumerated or ranged;
  a growing KV cache is `MLState`, not a pointer the caller owns.
- **Compiling is slow.** 8 to 65 seconds for the models below, against
  400 ms to compile every Metal kernel this project has.

## 2. What the ANE is good at

A plain FP16 matmul of a 7B MLP projection, 256 rows through 3584x18944:

| | time | rate | ops on ANE |
|---|---|---|---|
| ANE | **2.33 ms** | **14.9 TFLOP/s** | 3 of 3 |
| Core ML on the GPU | 14.2 ms | 2.44 TFLOP/s | 0 |
| Core ML on the CPU (AMX) | 11.9 ms | 2.93 TFLOP/s | 0 |
| this runtime's Metal tile GEMM | — | 2.11 TFLOP/s | — |

That is 78% of the chip's claimed peak, and **seven times** what the GPU
kernels in `src/metal/` reach on the same shape. The ANE is real and it is
fast.

A conv stack shaped like an image model (512 channels, 1024 positions,
depthwise mixer plus a 4x FFN, eight blocks) behaves the same way: 2.95 ms
on the ANE (11.7 TFLOP/s) against 33.9 ms through Core ML's GPU path.

## 3. Why this runtime still cannot use it

**Four bits are not on the menu.** Quantize that same matmul to int4 with
per-block scales -- what this project stores and what makes a 7B fit in
3.3 GB -- and Core ML stops using the ANE at all:

| | time | rate | ops on ANE |
|---|---|---|---|
| int4 blockwise, `MLComputeUnitsAll` | 15.7 ms | 2.21 TFLOP/s | **0 of 3** |
| int4 blockwise, CPU + ANE only | 56.9 ms | 0.61 TFLOP/s | 0 of 3 |

The 14.9 TFLOP/s is an FP16-weight number. Keeping the 7B in FP16 to reach
it means 15 GB of weights on a 16 GB machine, which is the opposite of what
this runtime is for.

**And the decode shape does not want an accelerator.** One token through the
same projection is 136 MB of weights read and 3.6 MFLOP of arithmetic:

| | time | effective bandwidth |
|---|---|---|
| ANE | 2.18 ms | 62 GB/s |
| this runtime's NF4 GEMV on the GPU | **0.385 ms** | 94 GB/s |

Decode is memory bound. The ANE reads at 62 GB/s of the machine's 96, and it
has to read FP16 rather than the four bits the GPU kernel reads. It is 5.7x
slower for the step that dominates a transcript.

## 4. The speech encoder: the ANE is the wrong shape

This is where an accelerator looked most promising -- the Conv-VAE encoder is
a fixed graph of 1-D convolutions in FP16, no cache, no token loop, and it is
the slowest part of the Metal backend. So the real graph was built as a Core
ML model (shapes read from the checkpoint: stem 1->32, seven stages of
3-3-3-3-3-3-8 blocks, strides 2,2,4,5,5,8 for 3200x, head to 64):

| audio | ANE | Core ML GPU | Core ML CPU | ops on ANE |
|---|---|---|---|---|
| 1 s | 29.1 ms | **10.2 ms** | 9.6 ms | 178 of 178 |
| 4 s | 116.5 ms | **39.8 ms** | 60.1 ms | 148 of 178 |

The ANE takes the whole graph and runs it **three times slower than the
GPU**. The shapes are why: at the front of this encoder a tensor is 32
channels by 720,000 positions, and the ANE is built for the opposite --
hundreds of channels over a small spatial extent. The channel-axis RMSNorm
is also not its kind of reduction; at 4 s, 30 of the 178 operations were
pushed off it.

Compiling that model took **27 to 65 seconds**, once per shape, and the
audio length is a shape. A 30-second input did not compile at all:
`ANECCompile() FAILED`, the sequence being past what the compiler accepts.

## 5. What the measurements did find

Core ML's own GPU path runs that same encoder graph in **10.2 ms per second
of audio**. This runtime's Metal encoder does about 36 ms per second of
audio per encoder. Nothing about the ANE explains that gap -- it is FP16
against the FP32 convolutions `src/metal/kernels/vae.metal` runs to keep
batching and chunking bit-exact (see `docs/METAL.md`). So the 3.6x is ours
to take, on the GPU, and it is tracked as its own piece of work rather than
as a Neural Engine one.

The CPU numbers say the same thing from the other side: Core ML on the CPU
reaches 2.93 TFLOP/s on the prefill matmul, because Accelerate uses the AMX
block. The CPU kernels in `src/cpu/` are NEON and do not. That is a second
piece of headroom that needs no new hardware, only `cblas` -- at the cost of
the summation order the CPU path is currently bit-exact under.

## 6. Reproducing this

`tools/ane_probe/` holds what produced the tables: `gen.py` and `gen_enc.py`
emit the Core ML models (coremltools, run once, never at run time), and
`probe.m` compiles each one, prints the `MLComputePlan` device placement, and
times `MLComputeUnitsAll`, `CPUAndNeuralEngine`, `CPUAndGPU` and `CPUOnly`.

```bash
cd tools/ane_probe && pip install coremltools && python3 gen.py
clang -fobjc-arc -O2 -o probe probe.m -framework Foundation -framework CoreML
./probe mm_prefill.mlpackage 30 $(python3 -c "print(2*256*3584*18944)")
```

Nothing in `src/` depends on any of it.
