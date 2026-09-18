# Skill: Build System (CMake + MSVC + CUDA)

## Target
- Windows 11 x64, MSVC 2022 (v143), C11 + CUDA C++17
- CMake 3.28+, CUDA 12.2+

## CMake Targets
- `vibevoice_core` — static lib (C): core, audio, tokenizer, model, quant
- `vibevoice_cuda` — static lib (CUDA): all .cu kernels
- `vibevoice` — shared lib (.dll): links all above + CUDA
- `vv_cli` — executable: CLI tool
- `vv_tests` — executable: CTest tests
- `vv_bench` — executable: benchmarks

## CUDA Architectures
- sm_80 (A100, RTX 3090)
- sm_86 (RTX 3060/3070/3080)
- sm_89 (RTX 4090)

## Environment Variables
- CUDA_PATH
