#!/usr/bin/env python3
"""
Convert PyTorch checkpoint with adaptive normalization to standard format.

This script handles the key mismatch between:
- Saved format: layer.input_layernorm.dense.weight
- Expected format: layer.input_layernorm.weight
"""

import argparse
import safetensors.torch
import torch
from pathlib import Path


def convert_checkpoint(input_path: str, output_path: str):
    """Convert checkpoint format."""
    print(f"Loading checkpoint from: {input_path}")

    # Load the checkpoint
    state_dict = {}
    with safetensors.safe_open(input_path, framework='pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} keys")

    # Convert keys: remove .dense from layernorm keys
    converted_dict = {}
    converted_count = 0

    for key, value in state_dict.items():
        # Check if this is a layernorm.dense key
        if '.input_layernorm.dense.' in key or '.post_attention_layernorm.dense.' in key:
            # Remove .dense from the key
            new_key = key.replace('.dense.', '.')
            converted_dict[new_key] = value
            converted_count += 1
            if converted_count <= 5:
                print(f"  Converted: {key} -> {new_key}")
        else:
            converted_dict[key] = value

    print(f"Converted {converted_count} keys")
    print(f"Total keys in output: {len(converted_dict)}")

    # Save the converted checkpoint
    print(f"Saving converted checkpoint to: {output_path}")
    safetensors.torch.save_file(converted_dict, output_path)
    print("✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint format")
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
