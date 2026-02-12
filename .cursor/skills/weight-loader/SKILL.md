# Skill: Weight Loading (safetensors + config)

## Purpose
Load VibeVoice-ASR-4bit model from HuggingFace safetensors format.
Detect NF4 quantized vs FP16/BF16 tensors and load appropriately.

## Model Files
- model-00001-of-00002.safetensors (4.97 GB)
- model-00002-of-00002.safetensors (2.69 GB)
- config.json, model.safetensors.index.json

## Weight Categories
- `model.acoustic_tokenizer.encoder.*` — BF16, Conv-VAE
- `model.semantic_tokenizer.encoder.*` — BF16, Conv-VAE
- `model.acoustic_connector.*` / `model.semantic_connector.*` — BF16
- `model.layers.N.self_attn.{q,k,v,o}_proj.*` — NF4 quantized
- `model.layers.N.mlp.{gate,up,down}_proj.*` — NF4 quantized
- `model.layers.N.*_layernorm.weight` — BF16
- `model.embed_tokens.weight`, `model.norm.weight` — BF16
- `lm_head.weight` — BF16

## Key API
```c
vv_status_t vv_safetensors_open(const char* path, vv_safetensors_t** out);
vv_status_t vv_safetensors_get_tensor(vv_safetensors_t* st, const char* name,
                                       vv_tensor_t* out);
vv_status_t vv_model_load(const char* model_dir, vv_model_t** out);
```
