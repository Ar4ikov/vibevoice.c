#!/usr/bin/env bash
#
# Build the shipping binary: one file, no CUDA toolkit on the target machine,
# every GPU from Turing to Blackwell.
#
#   scripts/build-release.sh [build-dir]
#
# What makes it self-contained:
#   - the CUDA runtime is linked statically, so there is no libcudart to find;
#   - cuBLAS is not used at all (src/cuda/gemm.cu is hand-written WMMA), which
#     is what keeps the binary at a couple of MB instead of shipping 600 MB of
#     math libraries;
#   - the driver (libcuda.so.1 / nvcuda.dll) is dlopened by the static runtime,
#     so the binary also starts on a machine with no NVIDIA driver and falls
#     back to the CPU kernels.
#
# Cubins are emitted for every architecture the toolkit supports, plus PTX for
# the newest, which the driver JITs for anything newer still.

set -euo pipefail

BUILD_DIR="${1:-build-release}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not on PATH; set PATH=/usr/local/cuda-12.x/bin:\$PATH" >&2
    exit 1
fi

CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')"
# sm_100 and sm_120 only exist from CUDA 12.8 on; asking earlier nvcc for them
# is a hard error, so the list is trimmed to what the toolkit knows.
ARCHS="75-real;80-real;86-real;89-real;90-real;90-virtual"
if [ "${CUDA_VER:-0}" -ge 128 ]; then
    ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
fi
echo "CUDA $CUDA_VER, architectures: $ARCHS"

# Link the GCC runtimes statically as well, so the only shared objects left
# are libc and libm. libgomp.so is normally present, but "normally" is not
# the same as "always" on a stripped container image.
GOMP_A="$(gcc -print-file-name=libgomp.a 2>/dev/null || true)"
OMP_ARGS=()
if [ -f "$GOMP_A" ]; then
    OMP_ARGS=(
        "-DOpenMP_C_FLAGS=-fopenmp"
        "-DOpenMP_C_LIB_NAMES=gomp"
        "-DOpenMP_gomp_LIBRARY=$GOMP_A"
    )
fi

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVV_ENABLE_TRT=OFF \
    -DVV_BUILD_TESTS=OFF \
    -DVV_BUILD_BENCH=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    "${OMP_ARGS[@]}"

cmake --build "$BUILD_DIR" -j "$(nproc)"
strip "$BUILD_DIR/vv_cli" 2>/dev/null || true

echo
echo "binary: $BUILD_DIR/vv_cli  ($(du -h "$BUILD_DIR/vv_cli" | cut -f1))"
echo "shared library dependencies:"
ldd "$BUILD_DIR/vv_cli" | sed 's/^/  /'
