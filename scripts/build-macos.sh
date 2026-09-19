#!/usr/bin/env bash
#
# Build on macOS. On Apple Silicon this includes the Metal backend, which
# needs only the Command Line Tools -- the shaders are compiled from source
# at runtime, so no Xcode and no offline `metal` compiler. On an Intel Mac,
# and with -DVV_ENABLE_METAL=OFF, it is the CPU build: AVX2 there, NEON on
# Apple Silicon, threads defaulted to the performance cores.
#
#   scripts/build-macos.sh [build-dir]
#
# OpenMP is optional but worth having -- without it the CPU kernels run on
# one core. Apple's clang does not ship it:
#
#   brew install libomp
#
# ffmpeg is also optional; it is what lets the server accept anything that
# is not a WAV, and what the microphone falls back to.

set -euo pipefail

BUILD_DIR="${1:-build}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXTRA=()
if BREW_PREFIX="$(brew --prefix libomp 2>/dev/null)"; then
    echo "libomp: $BREW_PREFIX"
    EXTRA+=(
        "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I$BREW_PREFIX/include"
        "-DOpenMP_C_LIB_NAMES=omp"
        "-DOpenMP_omp_LIBRARY=$BREW_PREFIX/lib/libomp.dylib"
    )
else
    echo "libomp not found; building single-threaded (brew install libomp)"
fi

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVV_ENABLE_CUDA=OFF \
    "${EXTRA[@]}"

cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.ncpu)"

echo
echo "binary: $BUILD_DIR/vv_cli"
"$BUILD_DIR/vv_cli" --help | head -3 || true
