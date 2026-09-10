/**
 * @file conv1d.cu
 * @brief CUDA kernels for 1D convolution operations used by Conv-VAE encoder.
 *
 * Supports:
 * - Standard 1D convolution (pointwise)
 * - Depthwise 1D convolution
 * - Strided convolution for downsampling
 * - Causal padding
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/**
 * @brief 1D convolution kernel (FP16).
 *
 * input:  [in_channels, in_length]
 * weight: [out_channels, in_channels/groups, kernel_size]
 * bias:   [out_channels] or NULL
 * output: [out_channels, out_length]
 */
__global__ void conv1d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int in_channels, int in_length,
    int out_channels, int kernel_size,
    int stride, int groups,
    int pad_left,
    int out_length)
{
    int oc = blockIdx.y;
    int t  = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= out_channels || t >= out_length) return;

    int ch_per_group_in  = in_channels / groups;
    int ch_per_group_out = out_channels / groups;
    int g = oc / ch_per_group_out;

    float sum = 0.0f;

    for (int ic = 0; ic < ch_per_group_in; ic++) {
        int abs_ic = g * ch_per_group_in + ic;
        for (int k = 0; k < kernel_size; k++) {
            int in_t = t * stride + k - pad_left;
            if (in_t >= 0 && in_t < in_length) {
                float w = __half2float(
                    weight[oc * ch_per_group_in * kernel_size +
                           ic * kernel_size + k]);
                float x = __half2float(
                    input[abs_ic * in_length + in_t]);
                sum += w * x;
            }
        }
    }

    if (bias) {
        sum += __half2float(bias[oc]);
    }

    output[oc * out_length + t] = __float2half(sum);
}

/**
 * @brief Depthwise 1D convolution kernel (FP16).
 *
 * Each output channel uses only the corresponding input channel.
 * weight: [channels, 1, kernel_size]
 */
__global__ void depthwise_conv1d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int channels, int in_length,
    int kernel_size, int stride,
    int pad_left, int out_length)
{
    int c = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= channels || t >= out_length) return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int in_t = t * stride + k - pad_left;
        if (in_t >= 0 && in_t < in_length) {
            float w = __half2float(weight[c * kernel_size + k]);
            float x = __half2float(input[c * in_length + in_t]);
            sum += w * x;
        }
    }

    if (bias) {
        sum += __half2float(bias[c]);
    }

    output[c * out_length + t] = __float2half(sum);
}

extern "C" {

#include "vibevoice/types.h"

/**
 * @brief 1D convolution with caller-supplied padding and output length.
 *
 * Used by the streaming path, where the left context comes from the previous
 * chunk instead of zero padding and the right padding is expressed by asking
 * for more output positions (out-of-range reads yield zero).
 */
vv_status_t vv_conv1d_raw_cuda(
    const void* input_fp16,
    const void* weight_fp16,
    const void* bias_fp16,
    void* output_fp16,
    int in_channels, int in_length,
    int out_channels, int kernel_size,
    int stride, int groups,
    int pad_left, int out_length,
    void* stream)
{
    if (!input_fp16 || !weight_fp16 || !output_fp16) return VV_ERR_NULL_PTR;
    if (out_length <= 0) return VV_OK;

    dim3 block(256);
    dim3 grid((out_length + 255) / 256, out_channels);

    if (groups == in_channels && groups == out_channels) {
        depthwise_conv1d_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16, (const half*)weight_fp16,
            bias_fp16 ? (const half*)bias_fp16 : NULL,
            (half*)output_fp16, in_channels, in_length,
            kernel_size, stride, pad_left, out_length);
    } else {
        conv1d_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16, (const half*)weight_fp16,
            bias_fp16 ? (const half*)bias_fp16 : NULL,
            (half*)output_fp16, in_channels, in_length,
            out_channels, kernel_size, stride, groups,
            pad_left, out_length);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

/**
 * @brief Launch 1D convolution (FP16).
 */
vv_status_t vv_conv1d_cuda(
    const void* input_fp16,
    const void* weight_fp16,
    const void* bias_fp16,
    void* output_fp16,
    int in_channels, int in_length,
    int out_channels, int kernel_size,
    int stride, int groups, bool causal,
    int* out_length,
    void* stream)
{
    /*
     * Causal SConv1d (modular_vibevoice_tokenizer.py::SConv1d):
     *   padding_total = (k - 1) * dilation - (stride - 1)   [dilation == 1]
     *   extra_padding = enough zeros on the right for ceil() alignment
     * which together give out_len = ceil(in_length / stride) and a left pad
     * of (k - stride). Using (k - 1) here shifts every strided layer.
     */
    int pad_left;
    if (causal) {
        pad_left = kernel_size - stride;
        if (pad_left < 0) pad_left = 0;
        *out_length = (in_length + stride - 1) / stride;
    } else {
        pad_left = (kernel_size - 1) / 2;
        *out_length = (in_length + (kernel_size - 1) - kernel_size) / stride + 1;
    }
    if (*out_length < 1) *out_length = 1;

    dim3 block(256);
    dim3 grid((*out_length + 255) / 256, out_channels);

    if (groups == in_channels && groups == out_channels) {
        /* Depthwise */
        depthwise_conv1d_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16,
            (const half*)weight_fp16,
            bias_fp16 ? (const half*)bias_fp16 : NULL,
            (half*)output_fp16,
            in_channels, in_length,
            kernel_size, stride,
            pad_left, *out_length);
    } else {
        conv1d_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const half*)input_fp16,
            (const half*)weight_fp16,
            bias_fp16 ? (const half*)bias_fp16 : NULL,
            (half*)output_fp16,
            in_channels, in_length,
            out_channels, kernel_size,
            stride, groups,
            pad_left, *out_length);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? VV_OK : VV_ERR_CUDA_LAUNCH;
}

} /* extern "C" */
