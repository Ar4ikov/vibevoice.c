/**
 * @file gemm.cu
 * @brief GEMM wrappers: cuBLAS FP16 + NF4 dequant-then-GEMM.
 *
 * Two main operations:
 * 1. Standard FP16 GEMM via cuBLAS (for non-quantized layers)
 * 2. NF4 dequant → FP16 GEMM (for quantized LLM layers)
 *
 * IMPORTANT: All linear-layer callers pass weight in [N, K] layout
 * (i.e., [out_features, in_features], the standard PyTorch convention).
 * The GEMM functions compute:  C = A @ B^T  where B is [N, K].
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
 * @brief FP16 GEMM via cuBLAS:  C = alpha * A @ B^T + beta * C
 *
 * A: [M, K] FP16 (row-major)
 * B: [N, K] FP16 (row-major) — weight in [out_features, in_features] layout
 * C: [M, N] FP16 (row-major)
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

    /*
     * We want (row-major): C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C
     *
     * Row-major → column-major duality:
     *   A[M,K] rm = A^T[K,M] cm  (pointer A, ld=K)
     *   B[N,K] rm = B^T[K,N] cm  (pointer B, ld=K)
     *   C[M,N] rm = C^T[N,M] cm  (pointer C, ld=N)
     *
     * Transpose identity:  C^T = (A @ B^T)^T = B @ A^T
     *
     * In cuBLAS column-major terms:
     *   C_cm[N,M] = B_cm[N,K] @ A_cm[K,M]
     *
     * B_cm[N,K] = OP_T(B_stored_cm[K,N]) — transpose the stored B^T
     * A_cm[K,M] = OP_N(A_stored_cm[K,M]) — use A^T as-is
     *
     * cuBLAS call: gemm(OP_T, OP_N, N, M, K, B_ptr:ld=K, A_ptr:ld=K, C_ptr:ld=N)
     *
     * Use CUBLAS_COMPUTE_32F for FP32 accumulation (critical for K=3584).
     */
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, K,    /* B[N,K] rm → B^T[K,N] cm, ld=K; transposed to B[N,K] */
        A, CUDA_R_16F, K,    /* A[M,K] rm → A^T[K,M] cm, ld=K                       */
        &beta,
        C, CUDA_R_16F, N,    /* C[M,N] rm → C^T[N,M] cm, ld=N                       */
        CUBLAS_COMPUTE_32F,
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
 * weight: [N, K/2] uint8 (NF4 packed, row-major [out, in/2])
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

    /* Step 1: Dequantize weight from NF4 to FP16 → temp_weight[N, K] */
    int total_elements = N * K;
    vv_status_t s = vv_dequant_nf4_cuda(
        weight_packed, weight_scales_fp16,
        temp_weight_fp16, total_elements, block_size, stream);
    if (s != VV_OK) return s;

    /* Step 2: FP16 GEMM: output = input @ temp_weight^T
     * input[M,K], temp_weight[N,K] → output[M,N]
     */
    s = vv_gemm_fp16_cuda(
        input_fp16, temp_weight_fp16, output_fp16,
        M, N, K,
        1.0f, 0.0f, stream);

    return s;
}

/**
 * @brief FP16 GEMM via cuBLAS:  C = alpha * A @ B + beta * C   (no transpose)
 *
 * A: [M, K] FP16 (row-major)
 * B: [K, P] FP16 (row-major)
 * C: [M, P] FP16 (row-major)
 *
 * This is useful for Conv-VAE FFN: output[out_ch, len] = weight[out_ch, in_ch] @ input[in_ch, len]
 *   → M=out_ch, K=in_ch, P=len
 */
vv_status_t vv_gemm_fp16_nn_cuda(
    const void* A, const void* B, void* C,
    int M, int K, int P,
    float alpha, float beta,
    void* stream)
{
    if (!A || !B || !C) return VV_ERR_NULL_PTR;

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return VV_ERR_CUDA;

    cublasSetStream(handle, (cudaStream_t)stream);

    /*
     * Row-major C[M,P] = A[M,K] @ B[K,P]
     *
     * Transpose identity:  C^T = B^T @ A^T
     *
     * In cuBLAS column-major terms:
     *   A_ptr stores A[M,K] rm = A^T[K,M] cm  (ld=K)
     *   B_ptr stores B[K,P] rm = B^T[P,K] cm  (ld=P)
     *   C_ptr stores C[M,P] rm = C^T[P,M] cm  (ld=P)
     *
     *   C^T = B^T @ A^T  →  gemm(OP_N, OP_N, P, M, K, B:ld=P, A:ld=K, C:ld=P)
     */
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        P, M, K,
        &alpha,
        B, CUDA_R_16F, P,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, P,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? VV_OK : VV_ERR_CUDA;
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
