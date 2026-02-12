/**
 * @file gemm.cu
 * @brief GEMM wrappers: cuBLAS FP16 + NF4 dequant-then-GEMM.
 *
 * Two main operations:
 * 1. Standard FP16 GEMM via cuBLAS (for non-quantized layers)
 * 2. NF4 dequant → FP16 GEMM (for quantized LLM layers)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>

/* Forward declare the dequant kernel launch */
extern "C" {
    #include "vibevoice/types.h"

    vv_status_t vv_dequant_nf4_cuda(
        const uint8_t* packed, const void* scales_fp16,
        void* output_fp16, int n_elements, int block_size, void* stream);
}

/**
 * @brief cuBLAS handle management.
 * One handle per GPU, created on first use.
 */
static cublasHandle_t s_cublas_handle = NULL;

static cublasHandle_t get_cublas_handle(void) {
    if (!s_cublas_handle) {
        cublasCreate(&s_cublas_handle);
        cublasSetMathMode(s_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    }
    return s_cublas_handle;
}

extern "C" {

/**
 * @brief FP16 GEMM via cuBLAS.
 *
 * C = alpha * A @ B + beta * C
 * A: [M, K] FP16
 * B: [K, N] FP16
 * C: [M, N] FP16
 *
 * Uses Tensor Cores on Ampere+ (FP16 with FP32 accumulation).
 */
vv_status_t vv_gemm_fp16_cuda(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    void* stream)
{
    if (!A || !B || !C) return VV_ERR_NULL_PTR;

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return VV_ERR_CUDA;

    cublasSetStream(handle, (cudaStream_t)stream);

    __half h_alpha = __float2half(alpha);
    __half h_beta  = __float2half(beta);

    /*
     * cuBLAS uses column-major. For row-major A[M,K] @ B[K,N] = C[M,N]:
     * Compute C^T = B^T @ A^T using cuBLAS (column-major).
     * cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
     *              &alpha, B, N, A, K, &beta, C, N)
     */
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &h_alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &h_beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? VV_OK : VV_ERR_CUDA;
}

/**
 * @brief NF4 dequant + FP16 GEMM.
 *
 * Dequantizes NF4 weight matrix to FP16, then performs GEMM.
 * output = input @ dequant(weight)^T
 *
 * input:  [M, K] FP16
 * weight: [N, K/2] uint8 (NF4 packed)
 * scales: [N*K/block_size] FP16
 * output: [M, N] FP16
 * temp:   [N, K] FP16 (pre-allocated temporary for dequantized weight)
 */
vv_status_t vv_nf4_gemm_cuda(
    const void* input_fp16,
    const uint8_t* weight_packed,
    const void* weight_scales_fp16,
    void* output_fp16,
    void* temp_weight_fp16,
    int M, int N, int K,
    int block_size,
    void* stream)
{
    if (!input_fp16 || !weight_packed || !weight_scales_fp16 ||
        !output_fp16 || !temp_weight_fp16) {
        return VV_ERR_NULL_PTR;
    }

    /* Step 1: Dequantize weight from NF4 to FP16 */
    int total_elements = N * K;
    vv_status_t s = vv_dequant_nf4_cuda(
        weight_packed, weight_scales_fp16,
        temp_weight_fp16, total_elements, block_size, stream);
    if (s != VV_OK) return s;

    /* Step 2: FP16 GEMM: output = input @ weight^T */
    /* input[M,K] @ temp_weight[N,K]^T = output[M,N] */
    s = vv_gemm_fp16_cuda(
        input_fp16, temp_weight_fp16, output_fp16,
        M, N, K,
        1.0f, 0.0f, stream);

    return s;
}

/**
 * @brief Cleanup cuBLAS handle (call at shutdown).
 */
void vv_gemm_cleanup(void) {
    if (s_cublas_handle) {
        cublasDestroy(s_cublas_handle);
        s_cublas_handle = NULL;
    }
}

} /* extern "C" */
