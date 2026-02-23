#!/usr/bin/env bash
set -euo pipefail

# Disable HuggingFace mirror and use local cache to avoid proxy issues
unset HF_ENDPOINT
export HF_HUB_OFFLINE=1  # Force offline mode to use local cached models
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Video writer:
# - Default OpenCV uses `mp4v` (MPEG-4 Part 2), which many browsers won't play.
# - "auto" prefers PyAV/libx264 (H.264) and falls back to OpenCV.
export CTRLWORLD_VIDEO_WRITER="${CTRLWORLD_VIDEO_WRITER:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${CONFIG:-${ROOT_DIR}/omni_ctrl/configs/pi0_droid.yaml}"
OUT_BASE="${OUT_BASE:-${ROOT_DIR}/experiments/pi0_batch}"

# Optional: override policy checkpoint (base or finetuned).
# Examples:
#   export PI05_CKPT=/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid
export PI_CKPT=/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/checkpoints/pi0_droid_finetune/pi0_chunk000_finetune_debug/60000
# PI_CKPT="${PI_CKPT:-}"

extra_args=()
if [[ -n "${PI_CKPT}" ]]; then
  extra_args+=(--override-policy-checkpoint "pi0=${PI_CKPT}")
fi

python -u "${ROOT_DIR}/scripts/inference/run_all_droid_new_setup.py" \
  --out-base "${OUT_BASE}" \
  --config "${CONFIG}" \
  "${extra_args[@]}"
