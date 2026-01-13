#!/usr/bin/env bash
# Run batch tests with Pi0.5 policy

# Disable HuggingFace mirror and use local cache to avoid proxy issues
unset HF_ENDPOINT
export HF_HUB_OFFLINE=1  # Force offline mode to use local cached models
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ROOT_DIR}/scripts/run_all_droid_new_setup.py" \
    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
    --ann-dir dataset_example/droid_new_setup/annotation/val \
    --droid-root dataset_example/droid_new_setup \
    --out-base experiments/omni_ctrl_pi05_batch_myadapter \
    --iterations 1 \
    --skip-existing \
    "$@"
