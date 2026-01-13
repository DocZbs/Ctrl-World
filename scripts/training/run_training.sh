#!/bin/bash
# Clear Python cache before running training
find /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/lib/python3.11/site-packages/transformers -name "*.pyc" -delete 2>/dev/null
find /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/lib/python3.11/site-packages/transformers -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Run training
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World
python scripts/finetune_pi05_synthetic.py train \
  --repo-id local/synthetic_pickplace_0002 \
  --exp-name pickplace_0002_finetune \
  --num-gpus 1 \
  --num-train-steps 20000 \
  --batch-size 32
