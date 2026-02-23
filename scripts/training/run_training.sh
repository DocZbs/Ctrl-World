#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World"
cd "${ROOT_DIR}"

# Optional: clear transformers cache pyc
find "${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages/transformers" -name "*.pyc" -delete 2>/dev/null || true
find "${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages/transformers" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true


CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009"
DATASET_DIR="${ROOT_DIR}/droid_data"

# ----------------------
# Expert 1: pick_place_into on GPU 0
# ----------------------
nohup bash -lc '
GPU=0 \
EXP_NAME=pi05_droid_expert1_pick_place_into_chunk000_009 \
DROID_HF_INDEX_JSON=experiments/task_indices_expert5_chunk000_009/pick_place_into.json \
DROID_HF_SCENE=pick_place_into \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert1_pick_place_into_chunk000_009.launch.log 2>&1 &
PID1=$!

# ----------------------
# Expert 2: pick_place_onto on GPU 1
# ----------------------
nohup bash -lc '
GPU=1 \
EXP_NAME=pi05_droid_expert2_pick_place_onto_chunk000_009 \
DROID_HF_INDEX_JSON=experiments/task_indices_expert5_chunk000_009/pick_place_onto.json \
DROID_HF_SCENE=pick_place_onto \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert2_pick_place_onto_chunk000_009.launch.log 2>&1 &
PID2=$!

echo "Launched expert1 (GPU0), PID=${PID1}"
echo "Launched expert2 (GPU1), PID=${PID2}"
echo ""
echo "Check processes:"
echo "  ps -ef | rg 'expert1_pick_place_into|expert2_pick_place_onto|openpi/scripts/train.py'"
echo "Check GPUs:"
echo "  nvidia-smi"
echo "Follow logs:"
echo "  tail -f experiments/pi05_droid_expert1_pick_place_into_chunk000_009.launch.log"
echo "  tail -f experiments/pi05_droid_expert2_pick_place_onto_chunk000_009.launch.log"
