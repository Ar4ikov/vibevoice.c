#!/usr/bin/env python3
"""
Validate C runtime output against Python reference.

Usage:
    python validate_weights.py --model ./model_hf --audio test.wav
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Validate C vs Python output")
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--audio", required=True, help="Test WAV file")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Validating model: {args.model}")
    print(f"Audio: {args.audio}")
    print(f"Tolerance: {args.tolerance}")

    # TODO: Run Python inference, run C inference, compare outputs
    print("Note: Validation not yet implemented. Coming in future version.")


if __name__ == "__main__":
    main()
