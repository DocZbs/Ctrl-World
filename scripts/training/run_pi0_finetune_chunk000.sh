#!/usr/bin/env bash
set -euo pipefail

# Full fine-tuning pi-0 on HF DROID dump chunk-000 (parquet+mp4) WITHOUT LeRobot packaging.
#
# Defaults assume you run from the Ctrl-World repo and have:
#   - droid_data/data/chunk-000/*.parquet
#   - droid_data/videos/chunk-000/observation.images.*/*.mp4
#   - openpi assets cached under $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_droid/{assets,params}
#
# Note on action semantics:
# - HF DROID parquet includes both `action.joint_velocity` and `action.joint_position` (and `action` == joint_position+gripper_position).
# - The direct HF reader prefers `action.joint_velocity` + `action.gripper_position` to match pi0-DROID / ctrl-world usage.
# - If those columns are unavailable, it falls back to `action` and converts joint-position targets to deltas.

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
EXP_NAME="${EXP_NAME:-pi0_chunk000_finetune_replay}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/droid_data/data/chunk-000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"

TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

# Extra args passed through to `openpi/scripts/train.py` (tyro flags).
# Example:
#   EXTRA_TRAIN_ARGS="--lr-schedule.peak-lr 1e-5"
EXTRA_TRAIN_ARGS_STR="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS_STR}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS=(${EXTRA_TRAIN_ARGS_STR})
fi

# Resume/overwrite behavior:
# - Default is OVERWRITE=1 (start fresh in the same exp dir).
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

# Force the correct action space:
# - If set, training will hard-fail if `action.joint_velocity` cannot be read and the loader would fall back to `action`.
REQUIRE_JOINT_VEL="${REQUIRE_JOINT_VEL:-1}"
if [[ "${REQUIRE_JOINT_VEL}" == "1" ]]; then
  export DROID_REQUIRE_JOINT_VELOCITY=1
fi

# Lightweight preflight check (recommended).
PREFLIGHT="${PREFLIGHT:-0}"

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
echo "Pi-0 Full Fine-tune on DROID chunk-000"
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
echo "Require joint vel:   ${REQUIRE_JOINT_VEL}"
echo "Preflight:           ${PREFLIGHT}"
echo "Resume:              ${RESUME}"
echo "Overwrite:           ${OVERWRITE}"
echo "Checkpoint dir:      ${CHECKPOINT_BASE_DIR}"
echo "Python:              ${PYTHON_EXE}"
echo "Extra train args:    ${EXTRA_TRAIN_ARGS_STR}"
echo ""

if [[ "${PREFLIGHT}" == "1" ]]; then
  echo "Running DROID HF preflight..."
  "${PYTHON_EXE}" "${ROOT_DIR}/scripts/training/preflight_droid_hf_chunk.py" \
    --dataset-path "${DATASET_DIR}" \
    --num-episodes 3 \
    --action-horizon 10 \
    $( [[ "${REQUIRE_JOINT_VEL}" == "1" ]] && echo "--require-joint-velocity" )
  echo ""
fi

"${PYTHON_EXE}" "${TRAIN_SCRIPT}" pi0_droid_finetune \
  --exp-name "${EXP_NAME}" \
  --data.repo-id "${DATASET_DIR}" \
  --num-train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  ${SHUFFLE_FLAG} \
  "${EXTRA_FLAGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}"
