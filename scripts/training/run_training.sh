#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World"
cd "${ROOT_DIR}"

# Optional: clear transformers cache pyc
find "${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages/transformers" -name "*.pyc" -delete 2>/dev/null || true
find "${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages/transformers" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true


CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009"
DATASET_DIR="${ROOT_DIR}/droid_data"

if [[ "${RUN_BASE_PAIR:-1}" == "1" ]]; then

# ----------------------
# Expert 1: pick_place on GPU 0
# ----------------------
nohup bash -lc '
GPU=0 \
EXP_NAME=pi05_droid_expert1_pick_place_chunk000_009 \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=pick_place \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert1_pick_place_chunk000_009.launch.log 2>&1 &
PID1=$!

# ----------------------
# Expert 2: reorientation on GPU 1
# ----------------------
nohup bash -lc '
GPU=1 \
EXP_NAME=pi05_droid_expert2_reorientation_chunk000_009 \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=reorientation \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert2_reorientation_chunk000_009.launch.log 2>&1 &
PID2=$!

echo "Launched expert1 (GPU0), PID=${PID1}"
echo "Launched expert2 (GPU1), PID=${PID2}"
echo ""
echo "Check processes:"
echo "  ps -ef | rg 'expert1_pick_place|expert2_reorientation|openpi/scripts/train.py'"
echo "Check GPUs:"
echo "  nvidia-smi"
echo "Follow logs:"
echo "  tail -f experiments/pi05_droid_expert1_pick_place_chunk000_009.launch.log"
echo "  tail -f experiments/pi05_droid_expert2_reorientation_chunk000_009.launch.log"

fi

# ----------------------
# Tool-use + Reorientation (low-memory) launch pair
# ----------------------
nohup bash -lc '
GPU=0 \
EXP_NAME=pi05_droid_expert_tool_use_chunk000_009 \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=tool_use \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
RESUME=1 \
PREFLIGHT=0 \
BATCH_SIZE=8 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert_tool_use_chunk000_009.launch.log 2>&1 &
PID_TOOL=$!

nohup bash -lc '
GPU=1 \
EXP_NAME=pi05_droid_expert_reorientation_chunk000_009 \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=reorientation \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert_reorientation_chunk000_009.launch.log 2>&1 &
PID_REOR=$!

echo "Launched tool_use (GPU0), PID=${PID_TOOL}"
echo "Launched reorientation (GPU1), PID=${PID_REOR}"
echo "tail -f experiments/pi05_droid_expert_tool_use_chunk000_009.launch.log"
echo "tail -f experiments/pi05_droid_expert_reorientation_chunk000_009.launch.log"


# ----------------------
# Expert: deformable_object_manipulation on GPU 0
# ----------------------
nohup bash -lc '
GPU=0 \
EXP_NAME=pi05_droid_expert_articulation_manipulation_chunk000_009 \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=articulation_manipulation \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
BATCH_SIZE=8 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert_articulation_manipulation_chunk000_009.launch.log 2>&1 &
PID_ART=$!

# ----------------------
# Expert: deformable_object_manipulation on GPU 1
# ----------------------
nohup bash -lc '
GPU=1 \
EXP_NAME=pi05_droid_expert_deformable_object_manipulation_chunk000_009 \
DATASET_DIR=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
DROID_HF_INDEX_JSON=experiments/droid_instruction_task_index_chunk000_009.json \
DROID_HF_SCENE=deformable_object_manipulation \
DROID_HF_CHUNKS="chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009" \
BATCH_SIZE=8 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash scripts/training/run_pi05_finetune_chunk000.sh
' > experiments/pi05_droid_expert_deformable_object_manipulation_chunk000_009.launch.log 2>&1 &
PID_DEF=$!

echo "Launched articulation_manipulation (GPU0), PID=${PID_ART}"
echo "Launched deformable_object_manipulation (GPU1), PID=${PID_DEF}"
echo "tail -f experiments/pi05_droid_expert_articulation_manipulation_chunk000_009.launch.log"
echo "tail -f experiments/pi05_droid_expert_deformable_object_manipulation_chunk000_009.launch.log"
