# Skill: Model Conversion Tools (Python, offline only)

## Scripts (in tools/)
1. `convert_weights.py` — HF safetensors → .vvmodel (mmap-friendly binary)
2. `validate_weights.py` — compare C runtime vs Python output

## Python Dependencies (tools/requirements.txt)
torch>=2.2, transformers>=4.40, safetensors>=0.4,
bitsandbytes>=0.43

## NOTE: these are NOT runtime dependencies
The C runtime loads safetensors directly or .vvmodel format.
