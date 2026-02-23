#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "Error: $*" >&2; exit 1; }
cd "${ROOT_DIR}"

# Allow passing PI_CKPT as first positional arg:
#   bash scripts/inference/run_pi_ft_wm_simple.sh /path/to/ckpt_step_dir [--dry-run]
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  export PI_CKPT="${PI_CKPT:-$1}"
  shift
fi

# Optional flag passthrough (kept minimal on purpose).
DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="1"
  shift
fi
if [[ "${1:-}" != "" ]]; then
  die "Unknown argument: $1 (supported: [ckpt_path] [--dry-run])"
fi

# ---- Required ----
POLICY_TYPE="${POLICY_TYPE:-pi0fast}"          # pi0 | pi05 | pi0fast
PI_CKPT="${PI_CKPT:-}"                         # must point to a step dir containing assets/ and params/

# ---- Optional ----
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/droid_data}"
WM_CKPT="${WM_CKPT:-/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/synthetic_data/${POLICY_TYPE}_interact_30000}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
WM_DEVICE="${WM_DEVICE:-cuda:1}"
START_EPISODE="${START_EPISODE:-0}"
END_EPISODE="${END_EPISODE:-1000}"
MAX_GEN_STEPS="${MAX_GEN_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-1}"

# If the policy isn't on CUDA, force JAX to avoid trying to init the CUDA backend.
if [[ "${POLICY_DEVICE}" != cuda:* ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

# ---- Offline HF setup (defaults chosen for your machine) ----
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Cache root that contains SVD/CLIP (models--stabilityai--..., models--openai--...)
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/nvme-fast/huggingface/hub}"

# Use one consistent cache root by default.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${OPENPI_DATA_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HOME="${HF_HOME:-${OPENPI_DATA_HOME}}"
unset HF_ENDPOINT || true

# Make sure pi0-fast tokenizer cache is visible under HF_HUB_CACHE.
# (Some setups store it under openpi-assets/checkpoints/pi0_fast_droid/; we symlink it into the main cache.)
if [[ "${POLICY_TYPE}" == "pi0fast" ]]; then
  tok_dst="${HF_HUB_CACHE}/models--physical-intelligence--fast"
  tok_src_1="${OPENPI_DATA_HOME}/models--physical-intelligence--fast"
  tok_src_2="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi0_fast_droid/models--physical-intelligence--fast"
  if [[ ! -d "${tok_dst}" ]]; then
    if [[ -d "${tok_src_1}" ]]; then
      ln -s "${tok_src_1}" "${tok_dst}" 2>/dev/null || true
    elif [[ -d "${tok_src_2}" ]]; then
      ln -s "${tok_src_2}" "${tok_dst}" 2>/dev/null || true
    fi
  fi
fi

# Prefer explicit local snapshot dirs for diffusion/text models (avoid Hub metadata in offline mode).
if [[ -z "${SVD_MODEL_PATH:-}" ]]; then
  svd_snaps="${OPENPI_DATA_HOME}/models--stabilityai--stable-video-diffusion-img2vid/snapshots"
  [[ -d "${svd_snaps}" ]] || die "Missing cached SVD model under ${svd_snaps}"
  SVD_MODEL_PATH="$(ls -1td "${svd_snaps}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
  [[ -n "${SVD_MODEL_PATH}" ]] || die "No SVD snapshots found under ${svd_snaps}"
fi
if [[ -z "${CLIP_MODEL_PATH:-}" ]]; then
  clip_snaps="${OPENPI_DATA_HOME}/models--openai--clip-vit-base-patch32/snapshots"
  [[ -d "${clip_snaps}" ]] || die "Missing cached CLIP model under ${clip_snaps}"
  CLIP_MODEL_PATH="$(ls -1td "${clip_snaps}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true)"
  [[ -n "${CLIP_MODEL_PATH}" ]] || die "No CLIP snapshots found under ${clip_snaps}"
fi

[[ "${POLICY_TYPE}" == "pi0" || "${POLICY_TYPE}" == "pi05" || "${POLICY_TYPE}" == "pi0fast" ]] || die "Bad POLICY_TYPE=${POLICY_TYPE} (use pi0|pi05|pi0fast)"
[[ -n "${PI_CKPT}" ]] || die "Set PI_CKPT to a step dir (must contain assets/ and params/)"
[[ -d "${PI_CKPT}/assets" && -d "${PI_CKPT}/params" ]] || die "PI_CKPT must contain assets/ and params/: ${PI_CKPT}"
[[ -e "${WM_CKPT}" ]] || die "WM_CKPT not found: ${WM_CKPT}"

echo "policy=${POLICY_TYPE}"
echo "pi_ckpt=${PI_CKPT}"
echo "wm_ckpt=${WM_CKPT}"
echo "out=${OUTPUT_DIR}"
echo "hf_cache=${HF_HUB_CACHE}"
echo "svd=${SVD_MODEL_PATH}"
echo "clip=${CLIP_MODEL_PATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN_ARG=""
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_RUN_ARG="--dry-run"
fi

"${PYTHON_BIN}" -u scripts/inference/generate_synthetic_droid_batch.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --policy-type "${POLICY_TYPE}" \
  --pi-ckpt "${PI_CKPT}" \
  --wm-ckpt "${WM_CKPT}" \
  --start-episode "${START_EPISODE}" \
  --end-episode "${END_EPISODE}" \
  --wm-device "${WM_DEVICE}" \
  --policy-device "${POLICY_DEVICE}" \
  --max-gen-steps "${MAX_GEN_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --svd-model-path "${SVD_MODEL_PATH}" \
  --clip-model-path "${CLIP_MODEL_PATH}" \
  ${DRY_RUN_ARG} \
  ${DATA_STAT_PATH:+--data-stat-path "${DATA_STAT_PATH}"} \
  ${ACTION_ADAPTER_PATH:+--action-adapter-path "${ACTION_ADAPTER_PATH}"}
