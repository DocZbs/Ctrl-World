#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper so you can run from scripts/inference/.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/run_rollout_interact_pi05.sh" "$@"

