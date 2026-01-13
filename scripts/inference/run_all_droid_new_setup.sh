#!/usr/bin/env bash
set -euo pipefail

# Disable HuggingFace mirror and use local cache to avoid proxy issues
unset HF_ENDPOINT
export HF_HUB_OFFLINE=1  # Force offline mode to use local cached models
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Use OpenPI venv Python for correct transformers version
PYTHON_EXE="${ROOT_DIR}/openpi/.venv/bin/python"

# Ensure OpenPI venv packages take precedence
export PYTHONPATH="${ROOT_DIR}/openpi/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

"${PYTHON_EXE}" "${ROOT_DIR}/scripts/run_all_droid_new_setup.py" "$@" --skip-existing \
    --out-base "${ROOT_DIR}/experiments/droid_new_setup_our_wm40k_pi05droid" \