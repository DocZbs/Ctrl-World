#!/usr/bin/env bash
# Run batch tests with Octo policy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use unified batch runner
bash "${SCRIPT_DIR}/run_batch.sh" \
    omni_ctrl/configs/omni_ctrl_octo.yaml \
    "$@"
