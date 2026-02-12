# Skill: NF4 Dequantization

## Purpose
Dequantize NF4 (4-bit NormalFloat) packed weights to FP16 for GEMM.
CPU reference + CUDA kernel.

## ONLY Qwen2 LLM Linear layers are NF4
Tokenizer encoders, connectors, embeddings, norms are all BF16/FP16.

## NF4 Lookup Table
```c
static const float NF4_TABLE[16] = {
    -1.0f, -0.6961928f, -0.5250730f, -0.3949338f,
    -0.2844871f, -0.1848489f, -0.0911179f,  0.0f,
     0.0796009f,  0.1609302f,  0.2461123f,  0.3379930f,
     0.4407233f,  0.5626170f,  0.7229568f,  1.0f
};
```

## Key API
```c
vv_status_t vv_dequant_nf4_cpu(const uint8_t* packed, const float* scales,
                                float* output, int n_elements, int block_size);
vv_status_t vv_dequant_nf4_cuda(const uint8_t* packed, const void* scales,
                                 void* output_fp16, int n_elements,
                                 int block_size, cudaStream_t stream);
```

## CUDA Kernel Strategy
- One warp (32 threads) per quantization block (64 elements)
- NF4 lookup table in constant memory
- Vectorized half2 stores
- Near memory-bandwidth-limited throughput
