#!/bin/bash
# Run Pi0.5-DROID LoRA finetuning on GPU cuda:1 using JAX

set -e

echo "=========================================="
echo "Pi0.5-DROID LoRA Finetuning (JAX) on GPU cuda:1"
echo "=========================================="
echo ""

# Set CUDA device to cuda:1
export CUDA_VISIBLE_DEVICES=1

# Paths
REPO_ID="local/synthetic_pickplace_0002"
EXP_NAME="pickplace_lora_jax"
PYTHON_EXE="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/bin/python"

cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World

# Run LoRA finetuning with JAX
$PYTHON_EXE scripts/training/finetune_pi05_droid_lora_jax.py \
    --repo-id "$REPO_ID" \
    --exp-name "$EXP_NAME" \
    --num-train-steps 5000 \
    --batch-size 32 \
    --lora-rank 16

echo ""
echo "=========================================="
echo "✓ LoRA finetuning complete!"
echo "=========================================="
echo ""
echo "Model saved to: openpi/checkpoints/$EXP_NAME/"
echo "Format: JAX (native, no conversion needed)"
echo ""
