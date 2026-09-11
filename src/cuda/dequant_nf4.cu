/**
 * @file dequant_nf4.cu
 * @brief CUDA kernel for NF4 dequantization.
 *
 * Strategy:
 * - NF4 lookup table in constant memory (16 floats)
 * - One warp (32 threads) processes one block of 64 elements
 *   (each thread unpacks 2 elements from 1 packed byte)
 * - Output in FP16 (half2 vectorized stores)
 * - Near memory-bandwidth limited throughput
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/* NF4 lookup table — exact bitsandbytes values (scipy.stats.norm quantiles) */
__constant__ float c_nf4_table[16] = {
    -1.0f,              -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f,  0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};

/**
 * @brief Dequantize NF4 packed uint8 to FP16 (block-scaled, blocksize 64).
 *
 * One thread handles 4 packed bytes = 8 values, so a 256-thread block covers
 * 2048 elements. The earlier one-warp-per-64-element mapping launched a block
 * for every 32 bytes of input — over a million blocks for a single MLP
 * weight — and spent most of its time on scheduling rather than bandwidth.
 *
 * Eight consecutive values starting at a multiple of 8 always live in the
 * same 64-element scale block, so one scale lookup covers the whole group.
 *
 * @param packed     Packed uint8 data (2 NF4 values per byte)
 * @param scales     Per-block FP16 scales [n_elements / 64]
 * @param output     Output FP16 data [n_elements]
 * @param n_elements Total number of elements (multiple of 8)
 */
__global__ void vv_dequant_nf4_kernel(
    const uint8_t* __restrict__ packed,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int            n_elements)
{
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_nf4_table[threadIdx.x];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e0  = idx * 8;
    if (e0 >= n_elements) return;

    const uint32_t bits = ((const uint32_t*)packed)[idx];
    const float sc = __half2float(scales[e0 >> 6]);

    half2 out[4];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const uint32_t byte = (bits >> (b * 8)) & 0xFFu;
        out[b] = __floats2half2_rn(lut[byte >> 4] * sc, lut[byte & 0xF] * sc);
    }
    *(float4*)(output + e0) = *(const float4*)out;
}

/**
 * @brief Dequantize NF4 with double quantization (FP8 scales).
 *
 * scales_fp8[i] + scale_offset → actual FP16 scale
 * Then proceed as above.
 */
__global__ void vv_dequant_nf4_double_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales_fp8,     /* quantized absmax */
    const float*   __restrict__ scale_offsets,   /* per-superblock offset */
    const half*    __restrict__ scale_scales,    /* scale of scale */
    half*          __restrict__ output,
    int            n_elements,
    int            superblock_size)              /* how many blocks per offset */
{
    int quant_block = blockIdx.x;
    int lane = threadIdx.x;

    int base_elem = quant_block * 64;
    int byte_idx  = quant_block * 32 + lane;
    int out_idx   = base_elem + lane * 2;

    if (out_idx >= n_elements) return;

    /* Reconstruct scale from double quantization */
    int super_idx = quant_block / superblock_size;
    float s_scale = __half2float(scale_scales[super_idx]);
    float s_offset = scale_offsets[super_idx];

    /* Dequantize FP8 scale: scale = fp8_val * s_scale + s_offset */
    float fp8_raw = (float)scales_fp8[quant_block];
    float scale = fp8_raw * s_scale + s_offset;

    uint8_t byte_val = packed[byte_idx];
    uint8_t hi = (byte_val >> 4) & 0x0F;
    uint8_t lo = byte_val & 0x0F;

    float val_first  = c_nf4_table[hi] * scale;   /* HIGH nibble = first element  */
    float val_second = c_nf4_table[lo] * scale;   /* LOW nibble  = second element */

    half2 out_val = __floats2half2_rn(val_first, val_second);

    if (out_idx + 1 < n_elements) {
        ((half2*)output)[quant_block * 32 + lane] = out_val;
    } else {
        output[out_idx] = __float2half(val_first);
    }
}

/* ─── C API wrappers ────────────────────────────────────────────────────── */

extern "C" {

#include "vibevoice/types.h"

#include "vibevoice/device.h"

/**
 * @brief Launch NF4 dequantization kernel (simple, FP16 scales).
 */
vv_status_t vv_dequant_nf4_dev(
    const uint8_t* packed,
    const void*    scales_fp16,
    void*          output_fp16,
    int            n_elements,
    int            block_size,
    void*          stream)
{
    if (!packed || !scales_fp16 || !output_fp16) return VV_ERR_NULL_PTR;
    if (n_elements <= 0 || block_size <= 0) return VV_ERR_INVALID_ARG;

    if (block_size != 64 || (n_elements & 7) != 0) return VV_ERR_UNSUPPORTED;

    const int threads = 256;
    const int groups = n_elements / 8;
    dim3 grid((groups + threads - 1) / threads);
    dim3 block(threads);

    cudaStream_t s = (cudaStream_t)stream;

    vv_dequant_nf4_kernel<<<grid, block, 0, s>>>(
        packed,
        (const half*)scales_fp16,
        (half*)output_fp16,
        n_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return VV_ERR_CUDA_LAUNCH;
    }

    return VV_OK;
}

/**
 * @brief Launch NF4 dequantization kernel with double quantization.
 */
vv_status_t vv_dequant_nf4_double_dev(
    const uint8_t* packed,
    const uint8_t* scales_fp8,
    const float*   scale_offsets,
    const void*    scale_scales_fp16,
    void*          output_fp16,
    int            n_elements,
    int            block_size,
    int            superblock_size,
    void*          stream)
{
    if (!packed || !scales_fp8 || !output_fp16) return VV_ERR_NULL_PTR;

    int n_quant_blocks = (n_elements + block_size - 1) / block_size;

    dim3 grid(n_quant_blocks);
    dim3 block(32);

    cudaStream_t s = (cudaStream_t)stream;

    vv_dequant_nf4_double_kernel<<<grid, block, 0, s>>>(
        packed,
        scales_fp8,
        scale_offsets,
        (const half*)scale_scales_fp16,
        (half*)output_fp16,
        n_elements,
        superblock_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return VV_ERR_CUDA_LAUNCH;
    }

    return VV_OK;
}

} /* extern "C" */
