# Metal 4 tensor operations, against this runtime's own GEMM

`mpp::tensor_ops::matmul2d` is how Apple exposes the matrix units in Metal 4,
and on M5 it is the door to the GPU's neural accelerators. This probe runs it
next to `vv_gemm_tn` from `src/metal/kernels/linear.metal` -- the same
C = A · Bᵀ, the same buffers, the same command queue -- and checks both
against a CPU reference before reporting a number.

```bash
clang -fobjc-arc -O2 -o run run.m -framework Foundation -framework Metal
./run 512 18944 3584        # M N K; a 7B MLP projection at a prefill width
```

`gemm.metal` holds the runtime's kernel (copied, not included, so the probe
stands alone) and the tensor-ops one. Three tile shapes are tried, 64x32,
64x64 and 32x32.

Two things it takes care of, both of which produced nonsense first:

- The tensor's extents are **(columns, rows)**, and `slice(x, y)` shifts
  columns then rows -- so a row-major M×K matrix is
  `tensor(A, dextents<int32_t,2>(K, M))` and its M offset is the *second*
  slice argument. The transposed-right form (`matmul2d_descriptor(..., false,
  true, false)`) then takes B as (K, N), which is exactly how this runtime
  stores a weight.
- The destination cannot be `half` here, and the operands cannot be
  `device const` -- the implementation matches value types literally and
  falls through to `static_assert(... "Unsupported type")`.

What it measures on an M4 is in `docs/ANE.md`.
