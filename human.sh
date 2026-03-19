#!/usr/bin/env bash
set -euo pipefail

python scripts/utils/human_annotation_server.py \
  --root /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/Exp_Groot/gr00t-base-tooluse \
  --output /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/Exp_Groot/gr00t-base-tooluse/human_annotations.json \
  --port 18764

# 备用：evow_droid_expert2_pick_place_onto_iter100_seed777
# 使用时可把上面的命令注释掉，再取消下面这段注释运行。
python scripts/utils/human_annotation_server.py \
  --root /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/evow_droid_expert_articulation_manipulation_pi0_droid\
  --output /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/evow_droid_expert_articulation_manipulation_pi0_droid/human_annotations.json \
  --port 18769
