# Skill: Custom CUDA Kernels

## Purpose
Optimized CUDA kernels for all VibeVoice-ASR operations.
All public functions: extern "C", accept cudaStream_t, return vv_status_t.

## Kernels Needed

### For LLM (Qwen2-7B):
1. **RMSNorm** — Welford algorithm, FP32 accum, FP16 output, eps=1e-6
2. **RoPE** — theta=1e6, precomputed sin/cos, in-place half2
3. **GQA Flash Attention** — 28 Q heads, 4 KV heads, head_dim=128
4. **SwiGLU** — fused silu(gate) * up
5. **GEMM** — cuBLAS FP16 + NF4 dequant-GEMM wrapper
6. **Embedding** — lookup table [152064, 3584]
7. **NF4 Dequant** — see nf4-quantization skill

### For Conv-VAE Tokenizer Encoders:
8. **1D Causal Conv** — depthwise + pointwise, variable strides
9. **1D Strided Conv** — for downsample stages (strides 8,5,5,4,2,2)
10. **Layer Scale** — element-wise multiply by learnable scalar (1e-6 init)

### For Connectors:
11. **GELU** — activation for SpeechConnector MLP

## Target Performance (RTX 3080)
- Attention: < 2ms for seq=4096, 28 heads, dim=128
- GEMM: > 80% peak Ampere FP16 TFLOPs
- All kernels: match CPU reference within FP16 tolerance
