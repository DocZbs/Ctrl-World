#!/usr/bin/env bash
set -euo pipefail

# Run Ctrl-World `scripts/inference/rollout_interact_pi.py` with a Pi policy checkpoint.
#
# If TASK_TYPE is set, runs only that task_type.
# Otherwise runs the full droid_new_setup suite covering 0001..0013:
#   pickplace, towel_fold, wipe_table, tissue, close_laptop, stack
#
# Common overrides via env vars:
#   POLICY_TYPE=pi05|pi0|pi0fast
#   PI_CKPT=/path/to/openpi_ckpt_dir
#   WM_CKPT=/path/to/checkpoint-*.pt
#   CUDA_VISIBLE_DEVICES=0
#   HF_HUB_OFFLINE=1
#   OPENPI_DATA_HOME=/mnt/nvme-fast/huggingface/hub
#
# Examples:
#   # Run pi05 base on all 13 new_setup scenarios
#   bash run_rollout_interact_pi05.sh
#
#   # Run pi0 base on all 13 new_setup scenarios
#   POLICY_TYPE=pi0 PI_CKPT=/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi0_droid \
#     bash run_rollout_interact_pi05.sh
#
#   # Run only pickplace
#   TASK_TYPE=pickplace bash run_rollout_interact_pi05.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Optional: auto-activate conda env if available.
CONDA_BASE="${CONDA_BASE:-/mnt/nvme-fast/zbs/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ctrl-world}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
fi

# Make repo-local modules importable (e.g., `models/`).
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/openpi:${PYTHONPATH:-}"

# Offline/cache
unset HF_ENDPOINT || true
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/nvme-fast/huggingface/hub}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# GPU selection for OpenPI JAX policy.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

POLICY_TYPE="${POLICY_TYPE:-pi05}"  # pi05 | pi0 | pi0fast

# Default checkpoints by policy type (can always override with PI_CKPT env var).
case "${POLICY_TYPE}" in
  pi0)     PI_CKPT_DEFAULT="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi0_droid" ;;
  pi0fast) PI_CKPT_DEFAULT="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi0_fast_droid" ;;
  pi05|*)  PI_CKPT_DEFAULT="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid" ;;
esac

PI_CKPT="${PI_CKPT:-${PI_CKPT_DEFAULT}}"
WM_CKPT="${WM_CKPT:-/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/mnt/nvme-fast/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}"

[[ -e "${WM_CKPT}" ]] || { echo "Error: WM_CKPT not found: ${WM_CKPT}" >&2; exit 1; }

if [[ -n "${TASK_TYPE:-}" ]]; then
  TASKS=("${TASK_TYPE}")
else
  TASKS=(pickplace towel_fold wipe_table tissue close_laptop stack)
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "POLICY_TYPE=${POLICY_TYPE}"
echo "PI_CKPT=${PI_CKPT}"
echo "WM_CKPT=${WM_CKPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for t in "${TASKS[@]}"; do
  echo "============================================================"
  echo "Running rollout_interact_pi.py (task_type=${t})"
  echo "============================================================"

  python -u scripts/inference/rollout_interact_pi.py \
    --task_type "${t}" \
    --policy_type "${POLICY_TYPE}" \
    --pi_ckpt "${PI_CKPT}" \
    --ckpt_path "${WM_CKPT}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    "$@"
done
