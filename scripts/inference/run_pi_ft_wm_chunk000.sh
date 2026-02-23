#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "Error: $*" >&2; exit 1; }
cd "${ROOT_DIR}"

POLICY_TYPE="${POLICY_TYPE:-pi0fast}" # pi0 | pi05 | pi0fast
# Normalize common aliases.
case "${POLICY_TYPE}" in
  pi0-fast|pi0_fast) POLICY_TYPE="pi0fast" ;;
  pi05-fast|pi05_fast|pi05fast)
    die "Unknown POLICY_TYPE=${POLICY_TYPE}. Did you mean POLICY_TYPE=pi0fast or POLICY_TYPE=pi05?"
    ;;
esac
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/droid_data}"
WM_CKPT="${WM_CKPT:-/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt}"

PI_CKPT="${PI_CKPT:-}"
if [[ -z "${PI_CKPT}" ]]; then
  case "${POLICY_TYPE}" in
    pi05) PI_CKPT="${PI05_FT_CKPT:-${ROOT_DIR}/checkpoints/pi05_droid_finetune}" ;;
    pi0)  PI_CKPT="${PI0_FT_CKPT:-${ROOT_DIR}/checkpoints/pi0_droid_finetune}" ;;
    pi0fast) PI_CKPT="${PI0_FAST_FT_CKPT:-${ROOT_DIR}/checkpoints/pi0_fast_droid_finetune}" ;;
    *)    die "Set PI_CKPT for POLICY_TYPE=${POLICY_TYPE} (supported: pi05|pi0|pi0fast)" ;;
  esac
fi

# Optional: pin a specific numeric step (e.g. PI_STEP=99999).
# If unset/empty, we'll auto-resolve the latest checkpoint under PI_CKPT.
PI_STEP="${PI_STEP:-}"
if [[ -n "${PI_STEP}" && ! ( -d "${PI_CKPT}/assets" && -d "${PI_CKPT}/params" ) ]]; then
  if [[ -d "${PI_CKPT}/${PI_STEP}" ]]; then
    PI_CKPT="${PI_CKPT}/${PI_STEP}"
  else
    # If PI_CKPT looks like an exp dir (contains numeric step dirs), report available steps.
    steps_here="$(ls -1 "${PI_CKPT}" 2>/dev/null | awk '/^[0-9]+$/{print}' | sort -n | tail -n 20 | xargs || true)"
    if [[ -n "${steps_here}" ]]; then
      die "PI_STEP=${PI_STEP} not found under PI_CKPT=${PI_CKPT} (available steps: ${steps_here}; set PI_CKPT to the exp dir or a step dir)"
    fi

    # Otherwise, treat PI_CKPT as a base dir with experiment subdirs; try newest exp then the requested step.
    exp="$(ls -1t "${PI_CKPT}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
    [[ -n "${exp}" ]] || die "No experiment dirs found under PI_CKPT=${PI_CKPT} (set PI_CKPT to an exp dir or a step dir)"
    if [[ -d "${exp}/${PI_STEP}" ]]; then
      PI_CKPT="${exp}/${PI_STEP}"
    else
      steps="$(ls -1 "${exp}" 2>/dev/null | awk '/^[0-9]+$/{print}' | sort -n | tail -n 20 | xargs || true)"
      die "PI_STEP=${PI_STEP} not found under ${exp} (available steps: ${steps:-none}; set PI_CKPT to the exp dir or a step dir)"
    fi
  fi
fi

resolve_ckpt() {
  local p="$1" step exp
  [[ -d "${p}/assets" && -d "${p}/params" ]] && { echo "${p}"; return; }
  [[ -d "${p}" ]] || { echo "${p}"; return; }
  step="$(ls -1 "${p}" 2>/dev/null | awk '/^[0-9]+$/{print}' | sort -n | tail -n 1 || true)"
  [[ -n "${step}" && -d "${p}/${step}/assets" && -d "${p}/${step}/params" ]] && { echo "${p}/${step}"; return; }
  exp="$(ls -1t "${p}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
  [[ -n "${exp}" ]] && resolve_ckpt "${exp}" || echo "${p}"
}
PI_CKPT="$(resolve_ckpt "${PI_CKPT}")"

if [[ -z "${PI_STEP}" ]]; then
  base="$(basename "${PI_CKPT}")"
  if [[ "${base}" =~ ^[0-9]+$ ]]; then
    PI_STEP="${base}"
  else
    PI_STEP="latest"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/synthetic_data/${POLICY_TYPE}_ft_chunk000_interact_${PI_STEP}}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
WM_DEVICE="${WM_DEVICE:-cuda:1}"
START_EPISODE="${START_EPISODE:-0}"
END_EPISODE="${END_EPISODE:-1000}"
MAX_GEN_STEPS="${MAX_GEN_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-1}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/nvme-fast/huggingface/hub}"
# Make HF/Transformers look in the same cache dir (important for pi0-fast tokenizer offline).
export HF_HUB_CACHE="${HF_HUB_CACHE:-${OPENPI_DATA_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HOME="${HF_HOME:-$(dirname "${HF_HUB_CACHE}")}"
unset HF_ENDPOINT || true

# If running offline, prefer passing local snapshot dirs for SVD/CLIP to avoid Hub metadata lookups.
# This also avoids requiring a single HF_HUB_CACHE to contain *all* repos.
if [[ -z "${SVD_MODEL_PATH:-}" ]]; then
  svd_snaps="${OPENPI_DATA_HOME}/models--stabilityai--stable-video-diffusion-img2vid/snapshots"
  if [[ -d "${svd_snaps}" ]]; then
    svd_latest="$(ls -1td "${svd_snaps}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
    [[ -n "${svd_latest}" ]] && SVD_MODEL_PATH="${svd_latest}"
  fi
fi
if [[ -z "${CLIP_MODEL_PATH:-}" ]]; then
  clip_snaps="${OPENPI_DATA_HOME}/models--openai--clip-vit-base-patch32/snapshots"
  if [[ -d "${clip_snaps}" ]]; then
    clip_latest="$(ls -1td "${clip_snaps}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
    [[ -n "${clip_latest}" ]] && CLIP_MODEL_PATH="${clip_latest}"
  fi
fi

[[ -e "${WM_CKPT}" ]] || die "WM_CKPT not found: ${WM_CKPT}"
[[ -d "${PI_CKPT}/assets" && -d "${PI_CKPT}/params" ]] || die "PI_CKPT missing assets/params: ${PI_CKPT}"

echo "policy=${POLICY_TYPE} pi_ckpt=${PI_CKPT} wm=$(basename "${WM_CKPT}") out=${OUTPUT_DIR}"

python -u scripts/inference/generate_synthetic_droid_batch.py \
  --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" \
  --policy-type "${POLICY_TYPE}" --pi-ckpt "${PI_CKPT}" --wm-ckpt "${WM_CKPT}" \
  --start-episode "${START_EPISODE}" --end-episode "${END_EPISODE}" \
  --wm-device "${WM_DEVICE}" --policy-device "${POLICY_DEVICE}" \
  --max-gen-steps "${MAX_GEN_STEPS}" --save-every "${SAVE_EVERY}" \
  ${SVD_MODEL_PATH:+--svd-model-path "${SVD_MODEL_PATH}"} \
  ${CLIP_MODEL_PATH:+--clip-model-path "${CLIP_MODEL_PATH}"} \
  ${DATA_STAT_PATH:+--data-stat-path "${DATA_STAT_PATH}"} \
  ${ACTION_ADAPTER_PATH:+--action-adapter-path "${ACTION_ADAPTER_PATH}"}
