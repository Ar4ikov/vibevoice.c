# Metal backend

The Apple Silicon half of `include/vibevoice/device.h`. A build links either
`src/cuda/` or `src/metal/`, never both, and the pipeline above the seam is
the same code on either.

This is what the backend is, what it costs, and where its numbers differ
from the CUDA path's.

---

## 1. Building

```bash
scripts/build-macos.sh            # Metal by default on Apple Silicon
cmake -B build -DVV_ENABLE_METAL=OFF   # CPU only
```

`vv_cli --version` ends in `[metal]` when the backend is compiled in.
`--cpu`, or `VV_METAL_DISABLE=1`, runs the CPU path instead; so does a Mac
whose GPU is not Apple7 or newer (an Intel Mac), which the runtime reports
as "no device" the way a machine without an NVIDIA driver does.

**Xcode is not required.** The kernels are Metal Shading Language, embedded
in the binary as text by `cmake/EmbedMetal.cmake` and compiled by the Metal
framework on first use (about 400 ms on an M4, cached by the system
afterwards). The offline `metal` compiler ships only with Xcode; the
Command Line Tools' clang and the macOS SDK are enough to build this.

---

## 2. What the runtime does differently

### Memory is the machine's memory

Every allocation is a shared-storage `MTLBuffer`, and the pointer handed out
is its CPU address, so all the pointer arithmetic the pipeline does on
"device pointers" works unchanged. A sorted table maps a pointer back to
(buffer, offset) when a kernel binds it.

Two consequences the rest of the runtime is built around:

- **A host copy of an uploaded tensor is the same RAM twice.**
  `vv_dev_host_shares_memory()` says so, and the pipeline then points the
  embedding table and the head at their device copies (2.1 GB for the 7B)
  and frees the speech encoders' FP32 host weights once the front end has
  uploaded its FP16 ones (2.6 GB). Without that, the first end-to-end run
  peaked at 14.9 GB on a 16 GB Mac and swapped: a 1 GB upload took 20 s and
  prefill fell to 3 tok/s.
- **What Metal allows is not what the machine has.**
  `recommendedMaxWorkingSetSize` is 11.8 GiB on a 16 GB Mac that is also
  running macOS, and placement that believes it takes a 32K KV window and
  gets the compressor: 66 MB of free pages, 3.4 GB of swap, and a 16-minute
  file decoding at 3.1 tok/s. `vv_dev_get_device_info` reports the smaller of
  what Metal allows and what the OS can still give (free, inactive, purgeable
  and speculative pages, less a margin).
  That on its own makes placement worse, because the weights it is deciding
  about are already in RAM: a Mac that has just loaded 8 GB of model sees
  nothing free and streams layers it would have to hold anyway. So
  `vv_gpu_budget_reclaim` adds back what placing them frees — uploading a
  weight here is a memcpy, and the host copy goes. A discrete card reclaims
  nothing and decides as before.
- **Untouched pages cost nothing.** Allocations of 64 KB and up are
  anonymous mappings wrapped with `newBufferWithBytesNoCopy`, so their pages
  are zero and are committed on first use; a zero fill of a buffer nothing
  has written is skipped. A 32K-position KV cache is 1.8 GB that a 30-second
  file never touches.

### Streams, events, graphs

A stream is a command queue plus the command buffer being filled. Launches
encode into it; a sync, an event record or a device-to-host copy commits it.
The NULL stream keeps CUDA's legacy default-stream semantics: it waits for
every other stream and completes before the call returns.

Graph capture records the dispatches issued to a stream -- pipeline,
buffers resolved to (buffer, offset), parameter bytes, grid -- and a replay
encodes them again in one go. There is no Metal object for a captured
graph, but the point of the capture holds: a decode step issues ~450
kernels and the pipeline's own bookkeeping is skipped on a replay. Anything
that needs the host mid-capture (a sync, an event, a pageable copy) breaks
the capture, and the caller falls back to launching each kernel, exactly as
it does on CUDA.

**Two limits the CUDA path never meets**, both because of the system's GPU
watchdog, which counts from submission and not from when a buffer starts
running. The speech encoder on a 16-minute file queued a minute of work in a
few seconds and every buffer behind it was killed with
`kIOGPUCommandBufferCallbackErrorTimeout`.

- A command buffer is committed every 128 launches **or every 64K
  threadgroups**, whichever comes first.
- A stream may have **192K threadgroups submitted and not yet finished**;
  committing past that waits for its oldest buffer. Finished buffers are
  retired by reading their status, which never blocks.

The second bound is on work, not on buffers, and the difference is the whole
decode loop: a step is ~450 tiny launches across four buffers, so a cap of
two *buffers* made it wait for the GPU four times a token — 6.5 tok/s against
14.9, RTF 1.29 against 0.61 on a 30-second file.

`VV_METAL_TRACE=<ms>` logs every command buffer that runs longer than that,
with its launches, its threadgroups and the last kernel in it; it is how both
numbers were chosen.

### Numerics

The library is compiled with safe math: IEEE results, precise
transcendentals, and `a*b+c` fused only inside one expression -- which is
exactly what Apple clang does to the CPU kernels, so the two agree where
the tests compare them bit for bit. Kernels that must not fuse at all (the
BitNet epilogues, whose CPU file is built with `-ffp-contract=off`) say
`#pragma METAL fp contract(off)` in the function body.

Where a CUDA kernel's reduction order is part of a test's expectations, the
port keeps it: the GEMVs sum their lanes' partial sums in the same xor
butterfly, the Conv-VAE convolutions fuse each product into the sum in
(input channel, tap) order, and the activation quantizers produce the
FP16 value the unfused op would have written and round it the same way.
`vv_erf` is FreeBSD's `s_erff` (as musl carries it), because MSL has no
`erf` and the exact GELU needs one.

---

## 3. The kernels

| file | what it holds |
|---|---|
| `runtime.metal` | copies, fills, the device-side index copy and position add |
| `transformer.metal` | embedding, RMSNorm, RoPE, SwiGLU, residual/bias adds, conversions |
| `linear.metal` | NF4 GEMV and dequant, dense FP16 GEMM (simdgroup matrices), LM head, argmax |
| `quant.metal` | INT8 (W8A16) and the legacy row-major INT4 path |
| `w4a16.metal` | the INT4 GPU layout: GEMV (up to three projections per launch), act-order gather, dequant |
| `int8act.metal` | the fused int8 activation quantizers, calibration absmax, W8A8 / W4A8 |
| `attention.metal` | KV formats (FP16, FP8 ×2, TurboQuant ×4), store/dequant/rotate, split-KV decode, scalar and simdgroup-matrix prefill |
| `vae.metal` | the batched streaming Conv-VAE and the connectors' GEMM |
| `bitnet.metal` | ternary and int8 GEMV/GEMM, requant, the int8 head's argmax |

Three things have no equivalent on this hardware and say so:

- **`vv_skinny_linear_dev` declines.** 9..64 rows take the dequant + tile
  GEMM path, which is what the op's contract allows.
- **`VV_I8_PATH_MMA` declines.** Apple GPUs have no integer matrix unit;
  `auto` takes the tiled int32 kernel.
- **Multi-GPU is one device.** `vv_dev_device_count()` is 1 or 0,
  `vv_dev_enable_peer` refuses, and a peer copy is a device-to-device copy.

The three attention backends map onto two kernel families: decode is one
split-KV kernel for all of them (the port of CUDA's fa2, whose arithmetic
its fa1 kernels share), and prefill is the scalar kernel for `fa1`, the
simdgroup-matrix one for `fa2` and `flashinfer`. `fa2` feeds P V an FP16
pair (the value and what rounding dropped) to keep ~22 bits of each
probability; `flashinfer` rounds P once and issues half the matrix
multiplies, as its CUDA counterpart does.

---

## 4. Numbers (MacBook Air, M4, 8 GPU cores, 16 GB)

What the hardware gives, measured with the probes in this repo's history:

| | measured | note |
|---|---|---|
| memory read | 96 GB/s | `dispatchThreads` over 1 GB |
| FP16 simdgroup matrix | 2.7 TFLOP/s | 1 simdgroup/group; 2.3 with 4 |
| FP32 simdgroup matrix | 0.5 TFLOP/s | why P V runs in FP16 |
| empty dispatch | 2.1 µs | serial dispatch, 1 threadgroup |

Against that, the kernels:

| kernel | shape | Metal | of the ceiling |
|---|---|---|---|
| NF4 GEMV | 18944×3584 | 385 µs, 94 GB/s | 98 % |
| NF4 GEMV | 3584×3584 | 89 µs, 76 GB/s | 79 % |
| LM head GEMV | 152064×3584 | 11.2 ms, 97 GB/s | 100 % |
| FP16 GEMM | 512×18944×3584 | 32.9 ms, 2.11 TFLOP/s | 92 % |
| FP16 GEMM | 143×18944×3584 | 12.2 ms, 1.60 TFLOP/s | 70 % (M tail) |
| NF4 dequant | 18944×3584 | 1.97 ms, 86 GB/s | 90 % |

---

## 5. What is left

- **The Conv-VAE encoder** is the largest remaining gap: ~2 s for 30 s of
  audio against 0.1 s on a 3090. Its convolutions are FP32 SIMT because
  their summation order is what makes batching and chunking bit-exact.
- **BitNet's ternary GEMV** runs at 8-17 GB/s of its 96: the byte-to-vector
  conversions MSL emits for the unpacking dominate (a float FMA variant of
  the same loop measures 6.5 GB/s against 63 GB/s for the loads alone). The
  magic-constant trick the W4A16 kernel uses for nibbles would apply here
  too.
- **`vv_skinny_linear_dev`** could be a real kernel: the tile GEMM it falls
  back to pads 9..64 rows to 64.
- **Model load** is dominated by reading and converting the checkpoint on
  the CPU; the GPU copies are memcpys of the same RAM.
