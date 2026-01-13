#!/usr/bin/env bash
# Run batch tests with Pi0-FAST DROID policy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use unified batch runner
bash "${SCRIPT_DIR}/run_batch.sh" \
    omni_ctrl/configs/omni_ctrl_pi0_droid.yaml \
    "$@"
