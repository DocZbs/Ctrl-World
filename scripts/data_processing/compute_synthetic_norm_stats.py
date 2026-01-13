#!/usr/bin/env python3
"""Compute normalization statistics for synthetic pickplace data"""

import sys
sys.path.insert(0, '/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi')

import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import openpi.shared.normalize as normalize
import tqdm

# Load the synthetic dataset
dataset = LeRobotDataset("local/synthetic_pickplace_0002")

print(f"Dataset size: {len(dataset)}")
print(f"Features: {dataset.features}")

# Compute statistics
state_stats = normalize.RunningStats()
action_stats = normalize.RunningStats()

for i in tqdm.tqdm(range(len(dataset)), desc="Computing stats"):
    sample = dataset[i]

    # State: joint_position + gripper_position
    joint_pos = np.array(sample['joint_position'])  # (7,)
    gripper_pos = np.array([sample['gripper_position']])  # scalar -> (1,)
    state = np.concatenate([joint_pos, gripper_pos])  # (8,)

    # Pad state to 32 dimensions to match pi05_droid
    state_32d = np.pad(state, (0, 24), mode='constant', constant_values=0)
    state_stats.update(state_32d)

    # Actions (should already be 32-dim if dataset was converted correctly)
    actions = np.array(sample['actions'])
    if actions.shape[0] == 8:
        # Pad to 32 if needed
        actions = np.pad(actions, (0, 24), mode='constant', constant_values=0)
    action_stats.update(actions)

# Get statistics
norm_stats = {
    'state': state_stats.get_statistics(),
    'actions': action_stats.get_statistics(),
}

# Save to checkpoint directory
output_path = Path('/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/checkpoints/pi05_droid_finetune/pickplace_0002_finetune/assets/droid')
output_path.mkdir(parents=True, exist_ok=True)

print(f"\nWriting stats to: {output_path}")
normalize.save(output_path, norm_stats)

print("\n=== Computed Statistics ===")
print("Actions mean:", norm_stats['actions'].mean[:8])
print("Actions std:", norm_stats['actions'].std[:8])
print("\nState mean:", norm_stats['state'].mean[:8])
print("State std:", norm_stats['state'].std[:8])
print("\n✓ Norm stats computed and saved!")
