#!/usr/bin/env python3
"""
Diagnose LoRA checkpoint to check if pretrained weights were loaded correctly
"""

import torch
import safetensors.torch as st
import numpy as np

print("="*80)
print("LoRA Checkpoint Diagnosis")
print("="*80)

# Load base model
base_path = '/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors'
print(f"\n1. Loading base model from: {base_path}")
base_tensors = st.load_file(base_path)
print(f"   Total keys: {len(base_tensors)}")

# Load step 0 checkpoint
ckpt_path = '/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/checkpoints/pi05_droid_lora/pickplace_lora_save_0/0/model.safetensors'
print(f"\n2. Loading step 0 checkpoint from: {ckpt_path}")
ckpt_tensors = st.load_file(ckpt_path)
print(f"   Total keys: {len(ckpt_tensors)}")

# Find matching keys (ignoring structure differences)
print(f"\n3. Comparing weights...")

# Try to find a key that exists in both (without structure changes)
test_key = 'action_in_proj.weight'

if test_key in base_tensors and test_key in ckpt_tensors:
    base_weight = base_tensors[test_key]
    ckpt_weight = ckpt_tensors[test_key]

    print(f"\n   Test key: {test_key}")
    print(f"   Base shape: {base_weight.shape}")
    print(f"   Ckpt shape: {ckpt_weight.shape}")

    if base_weight.shape == ckpt_weight.shape:
        # Compare values
        diff = torch.abs(base_weight - ckpt_weight).mean().item()
        max_diff = torch.abs(base_weight - ckpt_weight).max().item()

        print(f"\n   Mean absolute difference: {diff}")
        print(f"   Max absolute difference: {max_diff}")

        if diff < 1e-6:
            print(f"   ✓ Weights match! (diff < 1e-6)")
            print(f"   → Pretrained weights were loaded correctly")
        elif diff < 0.01:
            print(f"   ~ Weights similar (diff < 0.01)")
            print(f"   → Weights might be from same model but slightly modified")
        else:
            print(f"   ✗ Weights differ significantly (diff >= 0.01)")
            print(f"   → Pretrained weights were NOT loaded correctly!")
            print(f"   → Model is likely randomly initialized!")

        # Show sample values
        print(f"\n   Sample values:")
        print(f"   Base:  {base_weight.flatten()[:5].tolist()}")
        print(f"   Ckpt:  {ckpt_weight.flatten()[:5].tolist()}")
    else:
        print(f"   ✗ Shape mismatch!")
else:
    print(f"   ✗ Key '{test_key}' not found in both models")

# Check layernorm structure issue
print(f"\n4. Checking layernorm structure...")
base_ln_keys = [k for k in base_tensors.keys() if 'layernorm' in k.lower()]
ckpt_ln_keys = [k for k in ckpt_tensors.keys() if 'layernorm' in k.lower()]

print(f"   Base layernorm keys: {len(base_ln_keys)}")
print(f"   Sample: {base_ln_keys[:3]}")
print(f"\n   Ckpt layernorm keys: {len(ckpt_ln_keys)}")
print(f"   Sample: {ckpt_ln_keys[:3]}")

if base_ln_keys and ckpt_ln_keys:
    base_has_dense = any('dense' in k for k in base_ln_keys)
    ckpt_has_dense = any('dense' in k for k in ckpt_ln_keys)

    print(f"\n   Base uses 'dense' structure: {base_has_dense}")
    print(f"   Ckpt uses 'dense' structure: {ckpt_has_dense}")

    if base_has_dense != ckpt_has_dense:
        print(f"\n   ✗ STRUCTURE MISMATCH!")
        print(f"   → Checkpoint cannot load pretrained weights correctly")
        print(f"   → This is why the model performs poorly!")

print("\n" + "="*80)
print("Diagnosis Complete")
print("="*80)
