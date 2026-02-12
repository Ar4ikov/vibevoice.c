cmake_minimum_required(VERSION 3.28)
project(vibevoice C CXX CUDA)

# ─── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Ampere + Ada if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
endif()

# ─── Options ──────────────────────────────────────────────────────────────────
option(VV_BUILD_TESTS "Build test executables" ON)
option(VV_BUILD_BENCH "Build benchmark executable" ON)
option(VV_BUILD_CLI   "Build CLI executable" ON)
option(VV_ENABLE_TRT  "Enable TensorRT support" ON)
option(VV_ENABLE_NVTX "Enable NVTX profiling markers" OFF)

# ─── Find CUDA ────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ─── Find TensorRT (optional) ────────────────────────────────────────────────
if(VV_ENABLE_TRT)
    if(DEFINED ENV{TENSORRT_PATH})
        set(TENSORRT_ROOT "$ENV{TENSORRT_PATH}")
    elseif(DEFINED TENSORRT_PATH)
        set(TENSORRT_ROOT "${TENSORRT_PATH}")
    endif()

    if(TENSORRT_ROOT)
        find_library(NVINFER_LIB nvinfer
            PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH)
        find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
            PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH)
        find_library(NVONNXPARSER_LIB nvonnxparser
            PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
    else()
        find_library(NVINFER_LIB nvinfer)
        find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)
        find_library(NVONNXPARSER_LIB nvonnxparser)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
    endif()

    if(NVINFER_LIB AND TENSORRT_INCLUDE_DIR)
        set(VV_HAS_TRT ON)
        message(STATUS "TensorRT found: ${NVINFER_LIB}")
    else()
        set(VV_HAS_TRT OFF)
        message(WARNING "TensorRT not found, TRT support disabled")
    endif()
endif()

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
    # Release flags set via CMAKE_C_FLAGS_RELEASE etc.
    set(CMAKE_C_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /D_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /D_DEBUG")
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

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

# ─── vibevoice_core (pure C, no GPU) ─────────────────────────────────────────
add_library(vibevoice_core STATIC
    src/core/alloc.c
    src/core/logger.c
    src/core/error.c
    src/audio/wav.c
    src/audio/resample.c
    src/audio/normalize.c
    src/text_tokenizer/bpe.c
    src/text_tokenizer/vocab.c
    src/text_tokenizer/special_tokens.c
    src/tokenizer_encoder/conv_vae.c
    src/connector/speech_connector.c
    src/model/safetensors.c
    src/model/config.c
    src/model/loader.c
    src/quant/nf4_table.c
    src/quant/dequant_cpu.c
    src/inference/pipeline.c
    src/inference/decoder.c
    src/inference/kv_cache.c
    src/inference/sampling.c
    src/inference/postprocess.c
)
target_link_libraries(vibevoice_core PUBLIC cjson)

# ─── vibevoice_cuda (CUDA kernels) ───────────────────────────────────────────
add_library(vibevoice_cuda STATIC
    src/cuda/dequant_nf4.cu
    src/cuda/conv1d.cu
    src/cuda/conv_vae.cu
    src/cuda/rmsnorm.cu
    src/cuda/rope.cu
    src/cuda/attention.cu
    src/cuda/swiglu.cu
    src/cuda/gemm.cu
    src/cuda/embedding.cu
    src/cuda/cuda_utils.cu
)
target_link_libraries(vibevoice_cuda PUBLIC
    CUDA::cudart
    CUDA::cublas
)
if(VV_ENABLE_NVTX)
    target_compile_definitions(vibevoice_cuda PUBLIC VV_ENABLE_NVTX)
    target_link_libraries(vibevoice_cuda PUBLIC CUDA::nvToolsExt)
endif()

# ─── vibevoice_trt (TensorRT wrapper, optional) ──────────────────────────────
if(VV_HAS_TRT)
    add_library(vibevoice_trt STATIC
        src/trt/engine_builder.c
        src/trt/engine_runtime.c
        src/trt/trt_utils.c
    )
    target_include_directories(vibevoice_trt PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(vibevoice_trt PUBLIC
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
    )
    if(NVONNXPARSER_LIB)
        target_link_libraries(vibevoice_trt PUBLIC ${NVONNXPARSER_LIB})
    endif()
    target_compile_definitions(vibevoice_trt PUBLIC VV_HAS_TRT=1)
endif()

# ─── vibevoice shared library ─────────────────────────────────────────────────
add_library(vibevoice SHARED
    src/core/alloc.c  # re-add for DLL export; in production use object libs
)
target_link_libraries(vibevoice PUBLIC
    vibevoice_core
    vibevoice_cuda
)
if(VV_HAS_TRT)
    target_link_libraries(vibevoice PUBLIC vibevoice_trt)
endif()

# ─── CLI executable ───────────────────────────────────────────────────────────
if(VV_BUILD_CLI)
    add_executable(vv_cli cli/main.c)
    target_link_libraries(vv_cli PRIVATE vibevoice_core vibevoice_cuda)
    if(VV_HAS_TRT)
        target_link_libraries(vv_cli PRIVATE vibevoice_trt)
    endif()
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(VV_BUILD_TESTS)
    enable_testing()

    function(vv_add_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE vibevoice_core vibevoice_cuda)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    vv_add_test(test_audio tests/test_audio.c)
    vv_add_test(test_safetensors tests/test_safetensors.c)
    vv_add_test(test_nf4 tests/test_nf4.c)
    vv_add_test(test_conv_vae tests/test_conv_vae.c)
    vv_add_test(test_e2e tests/test_e2e.c)
endif()

# ─── Benchmark ────────────────────────────────────────────────────────────────
if(VV_BUILD_BENCH)
    add_executable(vv_bench bench/benchmark.c)
    target_link_libraries(vv_bench PRIVATE vibevoice_core)
endif()

# ─── Install ──────────────────────────────────────────────────────────────────
install(TARGETS vibevoice vv_cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/vibevoice DESTINATION include)
