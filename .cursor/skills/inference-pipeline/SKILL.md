# Skill: Inference Pipeline

## Purpose
End-to-end: WAV file → structured JSON transcription.

## Pipeline Steps
1. Load audio → resample 24kHz → normalize -25 dBFS
2. Acoustic encoder (Conv-VAE FP16) → gaussian sample → acoustic connector
3. Semantic encoder (Conv-VAE FP16) → (mean only) → semantic connector
4. Combine: acoustic_out + semantic_out → [B, T, 3584]
5. Build prompt: embed text tokens, insert combined features at audio mask
6. LLM prefill: 28 Qwen2 layers (NF4), fill KV-cache
7. Autoregressive decode until <|endoftranscript|>
8. Parse tokens → Rich Transcription JSON

## Streaming (audio > 60s)
Process 60s chunks through tokenizer encoders with streaming cache,
concatenate means, then run LLM once on full sequence.

## Key API
```c
vv_status_t vv_inference_init(const char* model_dir, int gpu_id,
                               vv_inference_ctx_t** ctx);
vv_status_t vv_inference_transcribe(vv_inference_ctx_t* ctx,
                                     const float* audio, int n_samples,
                                     const vv_inference_params_t* params,
                                     vv_transcription_t** result);
vv_status_t vv_inference_free(vv_inference_ctx_t* ctx);
```

## KV-Cache
- 28 layers × 2 (K+V) × 4 KV heads × 128 head_dim
- FP16 default, FP8 option for 12GB GPUs
- Supports up to 131072 positions (128K context)
