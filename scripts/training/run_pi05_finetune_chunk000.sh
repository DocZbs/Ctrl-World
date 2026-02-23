#!/usr/bin/env bash
set -euo pipefail

# Full fine-tuning pi-0.5 (pi05) on HF DROID dump chunk-000 (parquet+mp4) WITHOUT LeRobot packaging.
#
# This is the pi05 counterpart to `scripts/training/run_pi0_finetune_chunk000.sh`.
#
# Defaults assume you run from the Ctrl-World repo and have:
#   - droid_data/data/chunk-000/*.parquet
#   - droid_data/videos/chunk-000/observation.images.*/*.mp4
#   - openpi assets cached under $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid/{assets,params}
#
# Action semantics:
# - We require `action.joint_velocity` (7) + `action.gripper_position` (1) in parquet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Conda env (as requested) ----
CONDA_BASE="${CONDA_BASE:-/mnt/nvme-fast/zbs/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ctrl-world}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "Warning: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh (skipping conda activate)"
fi

# ---- Paths / knobs ----
GPU="${GPU:-1}"
EXP_NAME="${EXP_NAME:-pi05_chunk000_finetune_debug}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/droid_data/data/chunk-000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"

TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

# pi05-droid uses action_horizon=15 (keep preflight aligned with training config).
MODEL_ACTION_HORIZON="${MODEL_ACTION_HORIZON:-15}"

# Resume/overwrite behavior:
# - Default is RESUME=0, OVERWRITE=1 (start fresh in the same exp dir).
# - For continuing an existing run, set RESUME=1 (will load the latest checkpoint and continue).
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-1}"
EXTRA_FLAGS=()
if [[ "${RESUME}" == "1" ]]; then
  OVERWRITE="0"
  EXTRA_FLAGS+=(--resume)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_FLAGS+=(--overwrite)
fi

# Force the correct action space: hard-fail if joint_velocity columns are missing.
REQUIRE_JOINT_VEL="${REQUIRE_JOINT_VEL:-1}"
if [[ "${REQUIRE_JOINT_VEL}" == "1" ]]; then
  export DROID_REQUIRE_JOINT_VELOCITY=1
fi

# Lightweight preflight check (recommended).
PREFLIGHT="${PREFLIGHT:-1}"

# IMPORTANT: our HF-DROID direct reader is optimized for sequential access.
# Keep SHUFFLE=0 unless you are okay with slower decoding.
SHUFFLE="${SHUFFLE:-0}"

# Avoid network; rely on local cache for gs://openpi-assets/*
unset HF_ENDPOINT || true
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/nvme-fast/huggingface/hub}"

# JAX memory tuning
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# Video backend hint for other parts of the stack
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"

# Use OpenPI venv python (matches other scripts in this repo)
PYTHON_EXE="${PYTHON_EXE:-${ROOT_DIR}/openpi/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "Error: OpenPI venv python not found/executable at ${PYTHON_EXE}"
  echo "Fix: set PYTHON_EXE=/path/to/openpi/.venv/bin/python"
  exit 1
fi

# Ensure OpenPI venv packages take precedence
export PYTHONPATH="${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

TRAIN_SCRIPT="${ROOT_DIR}/openpi/scripts/train.py"

if [[ "${SHUFFLE}" == "1" ]]; then
  SHUFFLE_FLAG="--shuffle"
else
  SHUFFLE_FLAG="--no-shuffle"
fi

echo "=========================================="
echo "Pi-0.5 (pi05) Full Fine-tune on DROID chunk-000"
echo "=========================================="
echo "Conda env:           ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "OPENPI_DATA_HOME:    ${OPENPI_DATA_HOME}"
echo "Dataset dir:         ${DATASET_DIR}"
echo "Exp name:            ${EXP_NAME}"
echo "Steps:               ${TRAIN_STEPS}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Num workers:         ${NUM_WORKERS}"
echo "Shuffle:             ${SHUFFLE}"
echo "Action horizon:      ${MODEL_ACTION_HORIZON}"
echo "Require joint vel:   ${REQUIRE_JOINT_VEL}"
echo "Preflight:           ${PREFLIGHT}"
echo "Resume:              ${RESUME}"
echo "Overwrite:           ${OVERWRITE}"
echo "Checkpoint dir:      ${CHECKPOINT_BASE_DIR}"
echo "Python:              ${PYTHON_EXE}"
echo ""

if [[ "${PREFLIGHT}" == "1" ]]; then
  echo "Running DROID HF preflight..."
  "${PYTHON_EXE}" "${ROOT_DIR}/scripts/training/preflight_droid_hf_chunk.py" \
    --dataset-path "${DATASET_DIR}" \
    --num-episodes 3 \
    --action-horizon "${MODEL_ACTION_HORIZON}" \
    $( [[ "${REQUIRE_JOINT_VEL}" == "1" ]] && echo "--require-joint-velocity" )
  echo ""
fi

"${PYTHON_EXE}" "${TRAIN_SCRIPT}" pi05_droid_finetune \
  --exp-name "${EXP_NAME}" \
  --data.repo-id "${DATASET_DIR}" \
  --num-train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  ${SHUFFLE_FLAG} \
  "${EXTRA_FLAGS[@]}"
