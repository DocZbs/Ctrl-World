#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_BASE="${CONDA_BASE:-/mnt/nvme-fast/zbs/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ctrl-world}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# Global quick-run overrides: num_iterations=20, rollout.max_steps=120
for cfg in \
  evow/configs/droid_data_expert_articulation_manipulation.yaml \
  evow/configs/droid_data_expert_deformable_object_manipulation.yaml \
  evow/configs/droid_data_expert_tool_use.yaml \
  evow/configs/droid_data_expert_reorientation.yaml \
  evow/configs/droid_data_expert2_on_sequential_compositional.yaml \
  evow/configs/droid_data_expert2_pick_place_onto_iter100_seed777_self_instruct.yaml; do
  sed -i -E 's/^num_iterations:.*/num_iterations: 20/' "$cfg"
  sed -i -E 's/^  max_steps:.*/  max_steps: 120/' "$cfg"
done

# articulation_manipulation reroll with pi0_droid checkpoint (GPU1)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert_articulation_manipulation.yaml --iterations 20

# deformable_object_manipulation reroll with pi0_droid checkpoint (GPU1)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert_deformable_object_manipulation.yaml --iterations 20

# tool_use reroll with pi0_droid checkpoint (GPU1)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert_tool_use.yaml --iterations 20

# reorientation reroll with pi0_droid checkpoint (GPU1)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert_reorientation.yaml --iterations 20

XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert2_on_sequential_compositional.yaml --iterations 20

# pick_place_onto self-instruct (deterministic top1 retrieval)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
CTRLWORLD_PI0_FIRST_INFER_TIMEOUT_SEC=1200 \
python evow/run_evow.py --config evow/configs/droid_data_expert2_pick_place_onto_iter100_seed777_self_instruct.yaml --iterations 20
