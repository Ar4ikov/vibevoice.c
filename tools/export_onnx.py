#!/usr/bin/env python3
"""
Export Conv-VAE tokenizer encoders to ONNX for TensorRT optimization.

Usage:
    python export_onnx.py --input ./model_hf --output ./encoder_acoustic.onnx --encoder acoustic
    python export_onnx.py --input ./model_hf --output ./encoder_semantic.onnx --encoder semantic
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Export Conv-VAE to ONNX")
    parser.add_argument("--input", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--encoder", required=True, choices=["acoustic", "semantic"])
    args = parser.parse_args()

    print(f"Exporting {args.encoder} encoder from {args.input} to {args.output}")

    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        print("Error: torch and transformers are required. Install via:")
        print("  pip install -r tools/requirements.txt")
        sys.exit(1)

    # TODO: Load model and export specific encoder to ONNX
    # model = AutoModel.from_pretrained(args.input, trust_remote_code=True)
    # encoder = model.acoustic_tokenizer.encoder if args.encoder == "acoustic" ...
    # torch.onnx.export(encoder, dummy_input, args.output, ...)

    print("Note: ONNX export not yet implemented. Coming in future version.")


if __name__ == "__main__":
    main()
