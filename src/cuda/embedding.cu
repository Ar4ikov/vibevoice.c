/**
 * @file embedding.cu
 * @brief Embedding lookup CUDA kernel.
 *
 * Qwen2: vocab_size=152064, hidden_size=3584
 * Embedding table stored in FP16 on GPU.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/**
 * @brief Embedding lookup kernel.
 *
 * table: [vocab_size, hidden_size] FP16
 * ids:   [seq_len] int32
 * output:[seq_len, hidden_size] FP16
 *
 * Grid: (seq_len, 1)
 * Block: (256, 1)
 */
__global__ void embedding_kernel(
    const half* __restrict__ table,
    const int32_t* __restrict__ ids,
    half* __restrict__ output,
    int seq_len, int hidden_size)
{
    int pos = blockIdx.x;
    int tid = threadIdx.x;

    if (pos >= seq_len) return;

    int token_id = ids[pos];
    const half* row = table + token_id * hidden_size;
    half* out_row = output + pos * hidden_size;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = row[i];
    }
}

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Embedding lookup.
 *
 * @param table      [vocab_size, hidden_size] FP16 on GPU
 * @param ids        [seq_len] int32 on GPU
 * @param output     [seq_len, hidden_size] FP16 on GPU
 * @param seq_len    Number of tokens
 * @param hidden_size Embedding dimension
 * @param stream     CUDA stream
 */
vv_status_t vv_embedding_dev(
    const void* table, const int32_t* ids, void* output,
    int seq_len, int hidden_size, void* stream)
{
    if (!table || !ids || !output) return VV_ERR_NULL_PTR;

    dim3 grid(seq_len);
    dim3 block(256);

    embedding_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const half*)table, ids, (half*)output,
        seq_len, hidden_size);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
