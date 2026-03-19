#!/usr/bin/env bash
set -euo pipefail

# Train the selected 5 pi05 experts (instruction-based subsets).
# Keep pi05 finetune defaults; only swap index files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_SCRIPT="${ROOT_DIR}/scripts/training/run_pi05_finetune_chunk000.sh"

if [[ ! -x "${RUNNER_SCRIPT}" ]]; then
  echo "Error: missing ${RUNNER_SCRIPT}"
  exit 1
fi

INDEX_DIR="${INDEX_DIR:-${ROOT_DIR}/experiments/task_indices_expert5_chunk000_009}"
DROID_ROOT="${DROID_ROOT:-${ROOT_DIR}/droid_data}"
CHUNKS="${CHUNKS:-chunk-000 chunk-001 chunk-002 chunk-003 chunk-004 chunk-005 chunk-006 chunk-007 chunk-008 chunk-009}"

GPU="${GPU:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
ACTION_HORIZON="${ACTION_HORIZON:-15}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"
EXP_PREFIX="${EXP_PREFIX:-pi05_droid_expert5}"

RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-1}"
SHUFFLE="${SHUFFLE:-0}"
REQUIRE_JOINT_VEL="${REQUIRE_JOINT_VEL:-1}"
PREFLIGHT="${PREFLIGHT:-0}"

CATEGORIES=(
  pick_place
  reorientation
  articulation_manipulation
  tool_use
  deformable_object_manipulation
)

echo "Using index dir: ${INDEX_DIR}"
echo "Using data root: ${DROID_ROOT}"

for CAT in "${CATEGORIES[@]}"; do
  INDEX_JSON="${INDEX_DIR}/${CAT}.json"
  if [[ ! -f "${INDEX_JSON}" ]]; then
    echo "Error: missing index file ${INDEX_JSON}"
    exit 1
  fi

done

for CAT in "${CATEGORIES[@]}"; do
  EXP_NAME="${EXP_PREFIX}_${CAT}_chunk000_009"
  LOG_PATH="${ROOT_DIR}/experiments/${EXP_NAME}.log"

  echo ""
  echo "============================================================"
  echo "Training specialist: ${CAT}"
  echo "Index:              ${INDEX_DIR}/${CAT}.json"
  echo "Exp:                ${EXP_NAME}"
  echo "Log:                ${LOG_PATH}"
  echo "============================================================"

  (
    export DROID_HF_INDEX_JSON="${INDEX_DIR}/${CAT}.json"
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

echo "All 5 selected expert trainings finished."
