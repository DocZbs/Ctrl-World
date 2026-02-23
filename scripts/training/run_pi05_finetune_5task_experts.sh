#!/usr/bin/env bash
set -euo pipefail

# Train 5 task-specialist pi05-DROID experts on DROID chunk-000..009.
# Strategy: keep pi05 finetune defaults, only change the episode subset by instruction category.
#
# Categories:
#  - pick_place
#  - reorientation
#  - articulation_manipulation
#  - tool_use
#  - deformable_object_manipulation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Conda env ----
CONDA_BASE="${CONDA_BASE:-/mnt/nvme-fast/zbs/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ctrl-world}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# ---- Dataset scope ----
DROID_ROOT="${DROID_ROOT:-${ROOT_DIR}/droid_data}"
CHUNKS="${CHUNKS:-chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009}"
INDEX_JSON="${INDEX_JSON:-${ROOT_DIR}/experiments/droid_instruction_task_index_chunk000_009.json}"
FORCE_REINDEX="${FORCE_REINDEX:-0}"
MAX_EPISODES_PER_CATEGORY="${MAX_EPISODES_PER_CATEGORY:-0}"   # 0 means no cap
ACTION_HORIZON="${ACTION_HORIZON:-15}"

# ---- Training knobs (kept aligned with pi05 base script defaults) ----
GPU="${GPU:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"
EXP_PREFIX="${EXP_PREFIX:-pi05_droid_task_expert}"

# keep original settings
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-1}"
SHUFFLE="${SHUFFLE:-0}"
REQUIRE_JOINT_VEL="${REQUIRE_JOINT_VEL:-1}"
PREFLIGHT="${PREFLIGHT:-0}"

RUNNER_SCRIPT="${ROOT_DIR}/scripts/training/run_pi05_finetune_chunk000.sh"
INDEX_SCRIPT="${ROOT_DIR}/scripts/training/index_droid_instruction_tasks.py"

if [[ ! -x "${RUNNER_SCRIPT}" ]]; then
  echo "Error: missing runner script: ${RUNNER_SCRIPT}"
  exit 1
fi
if [[ ! -x "${INDEX_SCRIPT}" ]]; then
  echo "Error: missing index script: ${INDEX_SCRIPT}"
  exit 1
fi

mkdir -p "$(dirname "${INDEX_JSON}")"

if [[ "${FORCE_REINDEX}" == "1" || ! -f "${INDEX_JSON}" ]]; then
  echo "[1/3] Building task index from instructions..."
  INDEX_ARGS=(
    --dataset-path "${DROID_ROOT}"
    --chunks ${CHUNKS}
    --action-horizon "${ACTION_HORIZON}"
    --out "${INDEX_JSON}"
  )
  if [[ "${MAX_EPISODES_PER_CATEGORY}" != "0" ]]; then
    INDEX_ARGS+=(--max-episodes-per-category "${MAX_EPISODES_PER_CATEGORY}")
  fi
  "${ROOT_DIR}/openpi/.venv/bin/python" "${INDEX_SCRIPT}" "${INDEX_ARGS[@]}"
else
  echo "[1/3] Reusing existing task index: ${INDEX_JSON}"
fi

echo "[2/3] Task index summary"
"${ROOT_DIR}/openpi/.venv/bin/python" - <<PY
import json
from pathlib import Path
p=Path(r"${INDEX_JSON}")
d=json.loads(p.read_text(encoding="utf-8"))
print(f"index={p}")
for k,v in d.get("scenes",{}).items():
    print(f"  {k:30s} episodes={v.get('num_episodes',0):5d} samples≈{v.get('total_samples',0):8d}")
PY

echo "[3/3] Start 5 specialist fine-tunes (sequential)"
CATEGORIES=(
  pick_place
  reorientation
  articulation_manipulation
  tool_use
  deformable_object_manipulation
)

for CAT in "${CATEGORIES[@]}"; do
  EXP_NAME="${EXP_PREFIX}_${CAT}_chunk000_009"
  LOG_PATH="${ROOT_DIR}/experiments/${EXP_NAME}.log"

  echo ""
  echo "============================================================"
  echo "Training specialist: ${CAT}"
  echo "Exp name:            ${EXP_NAME}"
  echo "Log:                 ${LOG_PATH}"
  echo "============================================================"

  (
    export DROID_HF_INDEX_JSON="${INDEX_JSON}"
    export DROID_HF_SCENE="${CAT}"
    export DROID_HF_CHUNKS="${CHUNKS}"

    GPU="${GPU}" \
    EXP_NAME="${EXP_NAME}" \
    DATASET_DIR="${DROID_ROOT}" \
    CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    MODEL_ACTION_HORIZON="${ACTION_HORIZON}" \
    RESUME="${RESUME}" \
    OVERWRITE="${OVERWRITE}" \
    SHUFFLE="${SHUFFLE}" \
    REQUIRE_JOINT_VEL="${REQUIRE_JOINT_VEL}" \
    PREFLIGHT="${PREFLIGHT}" \
    bash "${RUNNER_SCRIPT}"
  ) 2>&1 | tee "${LOG_PATH}"
done

echo "All 5 specialist training runs finished."
