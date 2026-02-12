# Skill: Model Conversion Tools (Python, offline only)

## Scripts (in tools/)
1. `convert_weights.py` — HF safetensors → .vvmodel (mmap-friendly binary)
2. `export_onnx.py` — Conv-VAE tokenizer encoders → ONNX
3. `build_trt_engine.py` — ONNX → TensorRT .plan
4. `validate_weights.py` — compare C runtime vs Python output

## Python Dependencies (tools/requirements.txt)
torch>=2.2, transformers>=4.40, safetensors>=0.4,
bitsandbytes>=0.43, onnx>=1.15, tensorrt>=10.0

## NOTE: these are NOT runtime dependencies
The C runtime loads safetensors directly or .vvmodel format.
