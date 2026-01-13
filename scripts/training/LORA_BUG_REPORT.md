# LoRA Training Bug - Complete Diagnosis & Solutions

## Problem Summary

**Your LoRA checkpoint at step 0 is randomly initialized, not pretrained.**

This is why it performs poorly even without training - the pretrained weights never loaded correctly.

---

## Diagnosis Results

```bash
python scripts/training/diagnose_lora_checkpoint.py
```

**Results:**
- ✗ Weight difference: 0.169 (should be <1e-6)
- ✗ Structure mismatch: `layernorm.weight` vs `layernorm.dense.weight/bias`
- ✗ Pretrained weights NOT loaded
- ✗ Model is randomly initialized

**Root Cause:**
Your training script uses `paligemma_variant="gemma_2b_lora"`, which changes the model structure (adds dense layers to layernorm). This prevents the pretrained weights from loading correctly.

---

## Solution Options

### Option 1: Full Finetuning (Recommended if you have GPU memory)

**Pros:**
- Simple, no LoRA complexity
- Guaranteed to load pretrained weights correctly
- Better final performance

**Cons:**
- Requires more GPU memory (~40GB)
- Slower training

**How to use:**
```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World

# Use the pi05_droid_finetune config (already exists in OpenPI)
python openpi/src/openpi/scripts/train_pytorch.py pi05_droid_finetune \
    --exp-name pickplace_full_finetune \
    --data.repo-id local/synthetic_pickplace_0002 \
    --num-train-steps 5000 \
    --batch-size 16
```

### Option 2: Fixed LoRA Training (if GPU memory is limited)

I've created a fixed version but I'm **not 100% sure** it will work with your OpenPI version.

The fix changes:
- `paligemma_variant="gemma_2b_lora"` → `"gemma_2b"`
- `action_expert_variant="gemma_300m_lora"` → `"gemma_300m"`

This keeps the structure compatible while still allowing LoRA.

**To try:**
```bash
python scripts/training/finetune_pi05_droid_lora_fixed.py \
    --repo-id local/synthetic_pickplace_0002 \
    --exp-name pickplace_lora_fixed \
    --num-train-steps 5000 \
    --batch-size 32
```

**Warning:** This may not work if your OpenPI version doesn't support adding LoRA to non-lora variants.

### Option 3: Check OpenPI Documentation

The safest approach is to check OpenPI's official documentation for the correct way to do LoRA finetuning that preserves pretrained weights.

---

## Verification

After training with the fixed version, verify pretrained weights loaded:

```bash
python scripts/training/diagnose_lora_checkpoint.py
```

Expected output:
```
✓ Weights match! (diff < 1e-6)
→ Pretrained weights were loaded correctly
```

---

## Summary

1. **Current checkpoint is broken**: Randomly initialized, not pretrained
2. **Cause**: LoRA config changes model structure
3. **Best fix**: Use full finetuning (Option 1) if GPU memory allows
4. **Alternative**: Try fixed LoRA (Option 2) or check OpenPI docs (Option 3)

The synthetic data generation bug I fixed earlier is **separate** from this issue. That fix will help your training data quality, but you also need to fix the checkpoint loading issue.
