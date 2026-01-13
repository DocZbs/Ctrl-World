#!/bin/bash
set -e

echo "=========================================="
echo "Pi0.5-DROID Finetuning Fix Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Reconvert synthetic data with correct 32-dim actions"
echo "2. Compute normalization statistics"
echo "3. Retrain the model"
echo ""

# Paths
SYNTHETIC_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/synthetic_data/pickplace_0002"
REPO_ID="local/synthetic_pickplace_0002"
EXP_NAME="pickplace_0002_finetune_fixed"
PYTHON_EXE="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/bin/python"

echo "Step 1: Reconverting synthetic data with 32-dim actions..."
$PYTHON_EXE scripts/finetune_pi05_synthetic.py convert \
    --synthetic-dir "$SYNTHETIC_DIR" \
    --output-repo-id "$REPO_ID"

echo ""
echo "Step 2: Computing normalization statistics..."
$PYTHON_EXE scripts/compute_synthetic_norm_stats.py

echo ""
echo "Step 3: Training pi05 model..."
$PYTHON_EXE scripts/finetune_pi05_synthetic.py train \
    --repo-id "$REPO_ID" \
    --exp-name "$EXP_NAME" \
    --num-train-steps 5000 \
    --batch-size 16

echo ""
echo "=========================================="
echo "✓ All done!"
echo "=========================================="
echo ""
echo "Model saved to: checkpoints/$EXP_NAME/"
echo ""
