#!/usr/bin/env bash
set -euo pipefail

# Full fine-tuning pi-0-FAST (pi0fast) on HF DROID dump chunk-000 (parquet+mp4) WITHOUT LeRobot packaging.
#
# Defaults assume you run from the Ctrl-World repo and have:
#   - droid_data/data/chunk-000/*.parquet
#   - droid_data/videos/chunk-000/observation.images.*/*.mp4
#   - openpi assets cached under $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_fast_droid/{assets,params}
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
GPU="${GPU:-0}"
EXP_NAME="${EXP_NAME:-pi0_fast_chunk000_finetune_debug}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/droid_data/data/chunk-000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"

TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

# Match pretrained pi0-fast-droid horizon.
MODEL_ACTION_HORIZON="${MODEL_ACTION_HORIZON:-10}"

# Extra args passed through to `openpi/scripts/train.py` (tyro flags).
# Example:
#   EXTRA_TRAIN_ARGS="--model.max-token-len 512 --lr-schedule.peak-lr 1e-5"
EXTRA_TRAIN_ARGS_STR="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS_STR}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS=(${EXTRA_TRAIN_ARGS_STR})
fi

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

# HuggingFace/Transformers cache (used by PI0-FAST tokenizer).
# Keep this aligned with OPENPI_DATA_HOME so we can run fully offline after pre-downloading.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${OPENPI_DATA_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HOME="${HF_HOME:-$(dirname "${HF_HUB_CACHE}")}"

# PI0-FAST requires a tokenizer repo hosted on HuggingFace ("physical-intelligence/fast").
# If we're offline and it's not in cache, fail early with a clear message.
FAST_TOKENIZER_CACHE_DIR="${HF_HUB_CACHE}/models--physical-intelligence--fast"
ALT_HF_HUB_CACHE_DIR="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi0_fast_droid"
ALT_FAST_TOKENIZER_CACHE_DIR="${ALT_HF_HUB_CACHE_DIR}/models--physical-intelligence--fast"
if [[ ! -d "${FAST_TOKENIZER_CACHE_DIR}" && -d "${ALT_FAST_TOKENIZER_CACHE_DIR}" ]]; then
  echo "Info: found cached PI0-FAST tokenizer under ${ALT_HF_HUB_CACHE_DIR}; using it for HF/Transformers cache."
  export HF_HUB_CACHE="${ALT_HF_HUB_CACHE_DIR}"
  export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
  export HF_HOME="${HF_HOME:-$(dirname "${HF_HUB_CACHE}")}"
  FAST_TOKENIZER_CACHE_DIR="${HF_HUB_CACHE}/models--physical-intelligence--fast"
fi
if [[ "${HF_HUB_OFFLINE}" =~ ^(1|true|yes)$ && ! -d "${FAST_TOKENIZER_CACHE_DIR}" ]]; then
  cat >&2 <<EOF
Error: missing cached HuggingFace repo for PI0-FAST tokenizer: physical-intelligence/fast

Searched for:
  - ${FAST_TOKENIZER_CACHE_DIR}
  - ${ALT_FAST_TOKENIZER_CACHE_DIR}

Fix (run once with network access; choose ONE cache dir):
  # Option A (recommended, shared cache):
  HF_HUB_OFFLINE=0 HF_HUB_CACHE=${OPENPI_DATA_HOME} python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("physical-intelligence/fast", cache_dir="${OPENPI_DATA_HOME}")
PY

  # Option B (store alongside pi0_fast_droid checkpoint assets):
  HF_HUB_OFFLINE=0 HF_HUB_CACHE=${ALT_HF_HUB_CACHE_DIR} python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("physical-intelligence/fast", cache_dir="${ALT_HF_HUB_CACHE_DIR}")
PY

Then re-run this script with HF_HUB_OFFLINE=1 and (if needed) HF_HUB_CACHE=<chosen cache dir>.
EOF
  exit 1
fi

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
echo "Pi-0-FAST (pi0fast) Fine-tune on DROID chunk-000"
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
echo "Extra train args:    ${EXTRA_TRAIN_ARGS_STR}"
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

"${PYTHON_EXE}" "${TRAIN_SCRIPT}" pi0_fast_droid_finetune \
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
