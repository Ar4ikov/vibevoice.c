#!/usr/bin/env python3
"""
Convert HuggingFace safetensors weights to .vvmodel format.

Usage:
    python convert_weights.py --input ./model_hf --output ./model.vvmodel

The C runtime can load safetensors directly, so this tool is optional.
It creates an optimized, mmap-friendly binary format for faster loading.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice-ASR weights")
    parser.add_argument("--input", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="Output .vvmodel path")
    parser.add_argument("--verify", action="store_true", help="Verify after conversion")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        sys.exit(1)

    print(f"Input: {input_dir}")
    print(f"Output: {args.output}")

    # Check for required files
    config_path = input_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    # Find safetensors files
    st_files = sorted(input_dir.glob("model*.safetensors"))
    if not st_files:
        print("Error: no safetensors files found")
        sys.exit(1)

    print(f"Found {len(st_files)} safetensors file(s)")
    for f in st_files:
        print(f"  {f.name} ({f.stat().st_size / 1e9:.2f} GB)")

    # TODO: Implement actual conversion
    # For now, the C runtime loads safetensors directly
    print("\nNote: The C runtime can load safetensors directly.")
    print("The .vvmodel format will be implemented in a future version.")
    print("For now, point the C runtime at the HF model directory.")


if __name__ == "__main__":
    main()
