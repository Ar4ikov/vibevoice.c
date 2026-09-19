cmake_minimum_required(VERSION 3.22)

# ─── Version ─────────────────────────────────────────────────────────────────
#
# The release number lives in the public header and nowhere else; the full
# SemVer of *this* build (0.2.0 on a release tag, 0.2.1-dev.5+g1a2b3c4 five
# commits later) is derived from it and from git by cmake/Version.cmake -- the
# same file CI runs to name the image, so the two cannot disagree. Pin it with
# -DVV_VERSION_FULL=... -DVV_REVISION=... where there is no .git to ask.
set(VV_VERSION_FULL "" CACHE STRING "Override the derived SemVer of this build")
set(VV_REVISION     "" CACHE STRING "Override the derived source revision")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.cmake")
vv_header_version("${CMAKE_CURRENT_SOURCE_DIR}" VV_VERSION)

# ─── Bootstrap project (C/CXX first, CUDA probed afterward) ──────────────────
project(vibevoice VERSION ${VV_VERSION} LANGUAGES C CXX)
vv_compute_version("${CMAKE_CURRENT_SOURCE_DIR}")

option(VV_ENABLE_CUDA  "Enable CUDA GPU support"       ON)
option(VV_ENABLE_BLAS  "Enable OpenBLAS for CPU GEMM"   OFF)
option(VV_BUILD_TESTS  "Build test executables"          ON)
option(VV_BUILD_BENCH  "Build benchmark executable"      ON)
option(VV_BUILD_CLI    "Build CLI executable"            ON)
option(VV_ENABLE_NVTX  "Enable NVTX profiling markers"   OFF)

# ─── Probe for CUDA after C/CXX toolchain is ready ───────────────────────────
set(VV_HAS_CUDA OFF)
if(VV_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(VV_HAS_CUDA ON)
        message(STATUS "CUDA enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(WARNING "CUDA requested but not found — building CPU-only")
    endif()
else()
    message(STATUS "CUDA disabled — building CPU-only")
endif()

# ─── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(VV_HAS_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
    endif()
endif()

# ─── Find CUDA ────────────────────────────────────────────────────────────────
if(VV_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ─── OpenMP (weight repacking and the CPU kernels) ───────────────────────────
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_FLAGS}")
endif()

# ─── Find OpenBLAS (optional) ─────────────────────────────────────────────────
set(VV_HAS_BLAS OFF)
if(VV_ENABLE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        set(VV_HAS_BLAS ON)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    else()
        find_library(OPENBLAS_LIB openblas)
        if(OPENBLAS_LIB)
            set(VV_HAS_BLAS ON)
            set(BLAS_LIBRARIES ${OPENBLAS_LIB})
            message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
        else()
            message(WARNING "OpenBLAS not found — using naive CPU GEMM")
        endif()
    endif()
endif()

# ─── Compiler flags ───────────────────────────────────────────────────────────
#
# A call with no prototype in sight is an error, not a warning. C happily
# invents `int f()` and promotes float arguments to double, which on the
# Windows ABI quietly corrupts every float parameter -- that is how the GPU
# speech encoder came to run with eps = garbage and transcribe speech as
# "[Noise]" on MSVC while Linux was fine.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/we4013>)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /D_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /D_DEBUG")
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-Werror=implicit-function-declaration>)
endif()

if(VV_HAS_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
endif()

# ─── Include directories ─────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/cjson
)

# ─── Third-party: cJSON ──────────────────────────────────────────────────────
add_library(cjson STATIC
    third_party/cjson/cJSON.c
)
target_include_directories(cjson PUBLIC third_party/cjson)

# Internal headers (threading shim, HTTP plumbing) live next to their users.
include_directories(${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/src/server)

# ─── vibevoice_core (pure C, no GPU) ─────────────────────────────────────────
add_library(vibevoice_core STATIC
    src/core/alloc.c
    src/core/logger.c
    src/core/error.c
    src/core/clock.c
    src/core/version.c
    src/core/gpu_select.c
    src/core/vv_spawn.c
    src/core/attn_backend.c
    src/audio/wav.c
    src/audio/resample.c
    src/audio/normalize.c
    src/audio/decode.c
    src/audio/mic.c
    src/audio/mic_devices.c
    src/audio/vad.c
    src/engine/engine.c
    src/server/http.c
    src/server/api.c
    src/server/ws.c
    src/text_tokenizer/bpe.c
    src/text_tokenizer/vocab.c
    src/text_tokenizer/special_tokens.c
    src/tokenizer_encoder/conv_vae.c
    src/tokenizer_encoder/vae_gpu.c
    src/tokenizer_encoder/vae_cpu.c
    src/tokenizer_encoder/vae_i8.c
    src/tokenizer_encoder/latent_sample.c
    src/connector/speech_connector.c
    src/model/safetensors.c
    src/model/config.c
    src/model/loader.c
    src/model/gguf.c
    src/model/loader_gguf.c
    src/quant/nf4_table.c
    src/quant/kv_format.c
    src/quant/awq_repack.c
    src/quant/load_quant.c
    src/quant/smooth.c
    src/quant/dequant_cpu.c
    src/cpu/cpu_kernels.c
    src/cpu/cpu_int8.c
    src/cpu/bitnet_cpu.c
    src/inference/pipeline.c
    src/inference/calib.c
    src/inference/frontend.c
    src/inference/decoder.c
    src/inference/kv_cache.c
    src/inference/sampling.c
    src/inference/postprocess.c
    src/inference/family.c
    src/inference/stream.c
    src/inference/stream_ctx.c
    src/inference/bitnet_lm.c
)
# The BitNet reference paths repeat VibeASR.cpp's float expressions operation
# for operation; a fused multiply-add the compiler invents would move bits.
if(NOT MSVC)
    set_source_files_properties(src/tokenizer_encoder/vae_i8.c
        src/inference/bitnet_lm.c src/cpu/bitnet_cpu.c src/model/gguf.c
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
# With no accelerator backend the device op set still has to link. The stub
# implements memory and streams on the host and refuses every compute op,
# which is what steers the pipeline to the CPU kernels.
if(NOT VV_HAS_CUDA AND NOT VV_HAS_METAL)
    target_sources(vibevoice_core PRIVATE src/device/device_none.c)
endif()

target_link_libraries(vibevoice_core PUBLIC cjson)

# What `vv_cli --version`, /health and vibevoice_build_info report. Stamped
# on every build rather than at configure time, so a commit is reflected
# without re-running cmake; the header is only rewritten when the text
# changes, so an unchanged version recompiles nothing, and only version.c
# includes it, so a changed one recompiles one file.
set(VV_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_target(vv_build_info
    COMMAND "${CMAKE_COMMAND}"
            "-DVV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DVV_OUT=${VV_GENERATED_DIR}/vv_build_info.h"
            "-DVV_VERSION_FULL=$CACHE{VV_VERSION_FULL}"
            "-DVV_REVISION=$CACHE{VV_REVISION}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.cmake"
    BYPRODUCTS "${VV_GENERATED_DIR}/vv_build_info.h"
    COMMENT "Stamping the build version"
    VERBATIM)
add_dependencies(vibevoice_core vv_build_info)
set_source_files_properties(src/core/version.c PROPERTIES
    INCLUDE_DIRECTORIES "${VV_GENERATED_DIR}")
if(OpenMP_C_FOUND)
    target_link_libraries(vibevoice_core PUBLIC OpenMP::OpenMP_C)
endif()

# Propagate VV_HAS_CUDA define to C code
if(VV_HAS_CUDA)
    target_compile_definitions(vibevoice_core PUBLIC VV_HAS_CUDA=1)
endif()

# OpenBLAS
if(VV_HAS_BLAS)
    target_compile_definitions(vibevoice_core PUBLIC VV_HAS_OPENBLAS=1)
    target_link_libraries(vibevoice_core PUBLIC ${BLAS_LIBRARIES})
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(vibevoice_core PUBLIC m dl)
endif()
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(vibevoice_core PUBLIC Threads::Threads)
endif()
if(WIN32)
    target_link_libraries(vibevoice_core PUBLIC ws2_32)
endif()

# ─── vibevoice_cuda (CUDA kernels) ───────────────────────────────────────────
if(VV_HAS_CUDA)
    add_library(vibevoice_cuda STATIC
        src/cuda/dequant_nf4.cu
        src/cuda/nf4_gemv.cu
        src/cuda/kv_quant.cu
        src/cuda/awq_gemv.cu
        src/cuda/int8_gemv.cu
        src/cuda/w4a16.cu
        src/cuda/act_quant.cu
        src/cuda/gemv_i8.cu
        src/cuda/gemm_i8.cu
        src/cuda/lm_head.cu
        src/cuda/conv_vae.cu
        src/cuda/vae_kernels.cu
        src/cuda/rmsnorm.cu
        src/cuda/rope.cu
        src/cuda/attention.cu
        src/cuda/attention_fi.cu
        src/cuda/attention_decode.cu
        src/cuda/swiglu.cu
        src/cuda/gemm.cu
        src/cuda/gemm_skinny.cu
        src/cuda/embedding.cu
        src/cuda/cuda_utils.cu
        src/cuda/bitnet.cu
    )
    target_link_libraries(vibevoice_cuda PUBLIC CUDA::cudart_static)
    if(VV_ENABLE_NVTX)
        target_compile_definitions(vibevoice_cuda PUBLIC VV_ENABLE_NVTX)
        target_link_libraries(vibevoice_cuda PUBLIC CUDA::nvToolsExt)
    endif()
endif()

# ─── vibevoice shared library ─────────────────────────────────────────────────
add_library(vibevoice SHARED
    src/core/alloc.c  # re-add for DLL export; in production use object libs
)
target_link_libraries(vibevoice PUBLIC vibevoice_core)
if(VV_HAS_CUDA)
    target_link_libraries(vibevoice PUBLIC vibevoice_cuda)
endif()

# ─── CLI executable ───────────────────────────────────────────────────────────
if(VV_BUILD_CLI)
    add_executable(vv_cli cli/main.c cli/cmd_serve.c cli/cmd_chat.c)
    target_link_libraries(vv_cli PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_cli PRIVATE vibevoice_cuda)
    endif()
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(VV_BUILD_TESTS)
    enable_testing()

    function(vv_add_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE vibevoice_core)
        if(VV_HAS_CUDA)
            target_link_libraries(${name} PRIVATE vibevoice_cuda)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    vv_add_test(test_audio tests/test_audio.c)
    vv_add_test(test_queue tests/test_queue.c)
    vv_add_test(test_gpu_select tests/test_gpu_select.c)
    vv_add_test(test_safetensors tests/test_safetensors.c)
    vv_add_test(test_nf4 tests/test_nf4.c)
    vv_add_test(test_conv_vae tests/test_conv_vae.c)
    vv_add_test(test_vae_stream tests/test_vae_stream.c)
    vv_add_test(test_tokenizer tests/test_tokenizer.c)
    vv_add_test(test_kv_quant tests/test_kv_quant.c)
    vv_add_test(test_kv_shard tests/test_kv_shard.c)
    vv_add_test(test_concurrent_decode tests/test_concurrent_decode.c)
    vv_add_test(test_kv_pool tests/test_kv_pool.c)
    vv_add_test(test_attention tests/test_attention.c)
    vv_add_test(test_decode_position tests/test_decode_position.c)
    vv_add_test(test_stream tests/test_stream.c)
    vv_add_test(test_ws tests/test_ws.c)
    vv_add_test(test_stream_server tests/test_stream_server.c)
    # The same suite again with the tensor-core prefill kernel switched off:
    # the scalar fallback has to agree with the reference too, and the two
    # timings side by side are what the speedup is measured from.
    add_test(NAME test_attention_scalar COMMAND test_attention)
    set_tests_properties(test_attention_scalar PROPERTIES
                         ENVIRONMENT "VV_ATTN_MMA=0")
    # INT4 group-affine (AWQ / GPTQ) linear layers, and the same suite
    # through the dequant + dense GEMM fallback that Turing takes.
    vv_add_test(test_w4a16 tests/test_w4a16.c)
    add_test(NAME test_w4a16_fallback COMMAND test_w4a16)
    set_tests_properties(test_w4a16_fallback PROPERTIES
                         ENVIRONMENT "VV_W4A16_MMA=0")
    # The same path through the loader on a real AWQ / GPTQ checkpoint
    # (VV_TEST_MODEL; skipped without one).
    vv_add_test(test_w4a16_model tests/test_w4a16_model.c)
    # Dense, INT8 and NF4 linear layers at 9..64 rows: the small-M kernels
    # byte for byte against the dequant + tile GEMM path, and both against
    # the CPU kernels and an FP64 reference.
    vv_add_test(test_skinny tests/test_skinny.c)
    # Every attention backend against the FP64 reference, once per backend.
    add_executable(test_attn_backends tests/test_attn_backends.c)
    target_link_libraries(test_attn_backends PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(test_attn_backends PRIVATE vibevoice_cuda)
    endif()
    foreach(backend fa1 fa2 flashinfer)
        add_test(NAME test_attn_${backend} COMMAND test_attn_backends)
        set_tests_properties(test_attn_${backend} PROPERTIES
                             ENVIRONMENT "VV_ATTN=${backend}")
        add_test(NAME test_concurrent_decode_${backend}
                 COMMAND test_concurrent_decode)
        set_tests_properties(test_concurrent_decode_${backend} PROPERTIES
                             ENVIRONMENT "VV_ATTN=${backend}")
    endforeach()
    vv_add_test(test_cpu_kernels tests/test_cpu_kernels.c)

    # Arguments to helper programs, uploads and HTTP heads stay data (#14).
    # spawn_child echoes its argv so the test can check it byte for byte.
    add_executable(spawn_child tests/spawn_child.c)
    vv_add_test(test_untrusted_input tests/test_untrusted_input.c)
    add_dependencies(test_untrusted_input spawn_child)
    target_compile_definitions(test_untrusted_input PRIVATE
        VV_SPAWN_CHILD="$<TARGET_FILE:spawn_child>")
    vv_add_test(test_model_load tests/test_model_load.c)
    target_compile_definitions(test_model_load PRIVATE
                               VV_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
    vv_add_test(test_prefill_append tests/test_prefill_append.c)
    # W8A8 / W4A8: quantizers bit-exact, integer sums exact, every GPU path
    # (dp4a GEMV, tensor-core GEMM, SIMT fallback) forced in turn.
    vv_add_test(test_int8_linear tests/test_int8_linear.c)
    vv_add_test(test_e2e tests/test_e2e.c)
    vv_add_test(test_gguf tests/test_gguf.c)
    vv_add_test(test_bitnet tests/test_bitnet.c)
    # Its expected values repeat the kernels' "same two operations in the
    # same order"; a compiler that fuses them into one FMA (clang on arm64
    # does by default) makes the test disagree with a correct library.
    if(NOT MSVC)
        set_source_files_properties(tests/test_bitnet.c PROPERTIES
            COMPILE_OPTIONS "-ffp-contract=off")
    endif()
    # Again with the tensor-core GEMM off: the dp4a fallback that pre-Ampere
    # GPUs take has to agree with the CPU bit for bit as well.
    add_test(NAME test_bitnet_dp4a COMMAND test_bitnet)
    set_tests_properties(test_bitnet_dp4a PROPERTIES
                         ENVIRONMENT "VV_BITNET_MMA=0")
    vv_add_test(test_bitnet_vae tests/test_bitnet_vae.c)
    vv_add_test(test_bitnet_placement tests/test_bitnet_placement.c)
    # Again with every op recomputing its FP32 results in the second pass
    # instead of keeping them: the other half of each two-pass op.
    add_test(NAME test_bitnet_vae_recompute COMMAND test_bitnet_vae)
    set_tests_properties(test_bitnet_vae_recompute PROPERTIES
                         ENVIRONMENT "VV_I8VAE_KEEP_K=1000000;VV_I8VAE_KEEP_DW=0")
endif()

# ─── Benchmark ────────────────────────────────────────────────────────────────
if(VV_BUILD_BENCH)
    add_executable(vv_bench bench/benchmark.c)
    target_link_libraries(vv_bench PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_bench PRIVATE vibevoice_cuda)
    endif()
    # Speech front end throughput, alone and under concurrency.
    add_executable(vv_bench_encoder bench/bench_encoder.c)
    target_link_libraries(vv_bench_encoder PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_bench_encoder PRIVATE vibevoice_cuda)
    endif()
    add_executable(vv_bench_bitnet bench/bench_bitnet.c)
    target_link_libraries(vv_bench_bitnet PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_bench_bitnet PRIVATE vibevoice_cuda)
    endif()
endif()

# ─── Install ──────────────────────────────────────────────────────────────────
set(VV_INSTALL_TARGETS vibevoice)
if(VV_BUILD_CLI)
    list(APPEND VV_INSTALL_TARGETS vv_cli)
endif()
install(TARGETS ${VV_INSTALL_TARGETS}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/vibevoice DESTINATION include)

# ─── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "  vibevoice.c ${VV_VERSION_FULL} build configuration:")
message(STATUS "    Revision:  ${VV_REVISION}")
message(STATUS "    CUDA:      ${VV_HAS_CUDA}")
message(STATUS "    OpenBLAS:  ${VV_HAS_BLAS}")
message(STATUS "    CLI:       ${VV_BUILD_CLI}")
message(STATUS "    Tests:     ${VV_BUILD_TESTS}")
message(STATUS "")
