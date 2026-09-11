cmake_minimum_required(VERSION 3.22)

# ─── Bootstrap project (C/CXX first, CUDA probed afterward) ──────────────────
project(vibevoice C CXX)

option(VV_ENABLE_CUDA  "Enable CUDA GPU support"       ON)
option(VV_ENABLE_BLAS  "Enable OpenBLAS for CPU GEMM"   OFF)
option(VV_BUILD_TESTS  "Build test executables"          ON)
option(VV_BUILD_BENCH  "Build benchmark executable"      ON)
option(VV_BUILD_CLI    "Build CLI executable"            ON)
option(VV_ENABLE_TRT   "Enable TensorRT support"         ON)
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

# ─── Find TensorRT (optional) ────────────────────────────────────────────────
set(VV_HAS_TRT OFF)
if(VV_HAS_CUDA AND VV_ENABLE_TRT)
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
        message(WARNING "TensorRT not found, TRT support disabled")
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
    src/text_tokenizer/bpe.c
    src/text_tokenizer/vocab.c
    src/text_tokenizer/special_tokens.c
    src/tokenizer_encoder/conv_vae.c
    src/connector/speech_connector.c
    src/model/safetensors.c
    src/model/config.c
    src/model/loader.c
    src/quant/nf4_table.c
    src/quant/kv_format.c
    src/quant/awq_repack.c
    src/quant/dequant_cpu.c
    src/cpu/cpu_kernels.c
    src/inference/pipeline.c
    src/inference/decoder.c
    src/inference/kv_cache.c
    src/inference/sampling.c
    src/inference/postprocess.c
)
# With no accelerator backend the device op set still has to link. The stub
# implements memory and streams on the host and refuses every compute op,
# which is what steers the pipeline to the CPU kernels.
if(NOT VV_HAS_CUDA AND NOT VV_HAS_METAL)
    target_sources(vibevoice_core PRIVATE src/device/device_none.c)
endif()

target_link_libraries(vibevoice_core PUBLIC cjson)
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
        src/cuda/lm_head.cu
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
    target_link_libraries(vibevoice_cuda PUBLIC CUDA::cudart_static)
    if(VV_ENABLE_NVTX)
        target_compile_definitions(vibevoice_cuda PUBLIC VV_ENABLE_NVTX)
        target_link_libraries(vibevoice_cuda PUBLIC CUDA::nvToolsExt)
    endif()
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
target_link_libraries(vibevoice PUBLIC vibevoice_core)
if(VV_HAS_CUDA)
    target_link_libraries(vibevoice PUBLIC vibevoice_cuda)
endif()
if(VV_HAS_TRT)
    target_link_libraries(vibevoice PUBLIC vibevoice_trt)
endif()

# ─── CLI executable ───────────────────────────────────────────────────────────
if(VV_BUILD_CLI)
    add_executable(vv_cli cli/main.c cli/cmd_serve.c cli/cmd_chat.c)
    target_link_libraries(vv_cli PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_cli PRIVATE vibevoice_cuda)
    endif()
    if(VV_HAS_TRT)
        target_link_libraries(vv_cli PRIVATE vibevoice_trt)
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
    vv_add_test(test_safetensors tests/test_safetensors.c)
    vv_add_test(test_nf4 tests/test_nf4.c)
    vv_add_test(test_conv_vae tests/test_conv_vae.c)
    vv_add_test(test_tokenizer tests/test_tokenizer.c)
    vv_add_test(test_kv_quant tests/test_kv_quant.c)
    vv_add_test(test_cpu_kernels tests/test_cpu_kernels.c)
    vv_add_test(test_e2e tests/test_e2e.c)
endif()

# ─── Benchmark ────────────────────────────────────────────────────────────────
if(VV_BUILD_BENCH)
    add_executable(vv_bench bench/benchmark.c)
    target_link_libraries(vv_bench PRIVATE vibevoice_core)
    if(VV_HAS_CUDA)
        target_link_libraries(vv_bench PRIVATE vibevoice_cuda)
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
message(STATUS "  vibevoice.c build configuration:")
message(STATUS "    CUDA:      ${VV_HAS_CUDA}")
message(STATUS "    OpenBLAS:  ${VV_HAS_BLAS}")
message(STATUS "    TensorRT:  ${VV_HAS_TRT}")
message(STATUS "    CLI:       ${VV_BUILD_CLI}")
message(STATUS "    Tests:     ${VV_BUILD_TESTS}")
message(STATUS "")
