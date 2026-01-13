#!/usr/bin/env python3
"""
Comprehensive analysis of synthetic data and training issues
"""

import sys
sys.path.insert(0, '/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi')

import json
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

print("="*80)
print("SYNTHETIC DATA QUALITY ANALYSIS")
print("="*80)

# 1. Load and analyze raw synthetic data
print("\n1. RAW SYNTHETIC DATA ANALYSIS")
print("-"*80)

synthetic_dir = Path('/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/synthetic_data/pickplace_0002')
anno_files = sorted((synthetic_dir / "annotation" / "synthetic").glob("*.json"))

print(f"Total episodes: {len(anno_files)}")

# Analyze a few episodes
success_count = 0
episode_lengths = []
joint_velocity_stats = []
gripper_stats = []

for i, anno_file in enumerate(anno_files[:10]):  # Check first 10
    with open(anno_file) as f:
        anno = json.load(f)

    success_count += anno['success']
    joints = np.array(anno['joints'])
    episode_lengths.append(len(joints))

    # Compute velocities
    velocities = np.diff(joints[:, :7], axis=0)
    joint_velocity_stats.append({
        'mean': np.mean(np.abs(velocities)),
        'max': np.max(np.abs(velocities)),
        'std': np.std(velocities)
    })

    gripper_stats.append({
        'min': joints[:, 7].min(),
        'max': joints[:, 7].max(),
        'mean': joints[:, 7].mean()
    })

print(f"Success rate: {success_count}/{len(anno_files[:10])}")
print(f"Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
print(f"\nJoint velocity stats (first 10 episodes):")
print(f"  Mean abs velocity: {np.mean([s['mean'] for s in joint_velocity_stats]):.4f}")
print(f"  Max abs velocity: {np.mean([s['max'] for s in joint_velocity_stats]):.4f}")
print(f"  Std velocity: {np.mean([s['std'] for s in joint_velocity_stats]):.4f}")
print(f"\nGripper stats:")
print(f"  Range: [{np.mean([s['min'] for s in gripper_stats]):.3f}, {np.mean([s['max'] for s in gripper_stats]):.3f}]")
print(f"  Mean: {np.mean([s['mean'] for s in gripper_stats]):.3f}")

# 2. Load and analyze converted LeRobot dataset
print("\n2. CONVERTED LEROBOT DATASET ANALYSIS")
print("-"*80)

dataset = LeRobotDataset('local/synthetic_pickplace_0002')
print(f"Dataset size: {len(dataset)} frames")
print(f"Number of episodes: {dataset.num_episodes}")

# Sample actions from dataset
actions_sample = []
for i in range(0, min(1000, len(dataset)), 10):
    actions_sample.append(dataset[i]['actions'].numpy())
actions_sample = np.array(actions_sample)

print(f"\nDataset action statistics (first 1000 frames, sampled every 10):")
print(f"  Shape: {actions_sample.shape}")
print(f"  First 8 dims mean: {actions_sample[:, :8].mean(axis=0)}")
print(f"  First 8 dims std: {actions_sample[:, :8].std(axis=0)}")
print(f"  Dims 8-32 (should be zero): all_zero={np.all(actions_sample[:, 8:] == 0)}")

# 3. Compare with DROID norm stats
print("\n3. COMPARISON WITH DROID NORM STATS")
print("-"*80)

droid_norm_stats = json.load(open('/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json'))
droid_actions_mean = np.array(droid_norm_stats['norm_stats']['actions']['mean'][:8])
droid_actions_std = np.array(droid_norm_stats['norm_stats']['actions']['std'][:8])

synthetic_actions_mean = actions_sample[:, :8].mean(axis=0)
synthetic_actions_std = actions_sample[:, :8].std(axis=0)

print("\nAction statistics comparison:")
print(f"{'Dimension':<12} {'DROID Mean':<15} {'Synthetic Mean':<15} {'Ratio':<10}")
print("-"*60)
for i in range(8):
    ratio = synthetic_actions_mean[i] / (droid_actions_mean[i] + 1e-8)
    print(f"Dim {i:<8} {droid_actions_mean[i]:<15.4f} {synthetic_actions_mean[i]:<15.4f} {ratio:<10.2f}")

print(f"\n{'Dimension':<12} {'DROID Std':<15} {'Synthetic Std':<15} {'Ratio':<10}")
print("-"*60)
for i in range(8):
    ratio = synthetic_actions_std[i] / (droid_actions_std[i] + 1e-8)
    print(f"Dim {i:<8} {droid_actions_std[i]:<15.4f} {synthetic_actions_std[i]:<15.4f} {ratio:<10.2f}")

# 4. Identify potential issues
print("\n4. POTENTIAL ISSUES IDENTIFIED")
print("-"*80)

issues = []

# Check if velocities are too small
if np.mean([s['mean'] for s in joint_velocity_stats]) < 0.05:
    issues.append("⚠️  Joint velocities are very small (< 0.05 rad/s)")

# Check if gripper range is limited
gripper_range = np.mean([s['max'] - s['min'] for s in gripper_stats])
if gripper_range < 0.5:
    issues.append("⚠️  Gripper range is limited (< 0.5)")

# Check if all episodes failed
if success_count == 0:
    issues.append("⚠️  All episodes marked as failed (success=0)")

# Check if episode lengths are too short
if np.mean(episode_lengths) < 100:
    issues.append("⚠️  Episodes are very short (< 100 frames)")

# Check action distribution mismatch
std_ratio = synthetic_actions_std / (droid_actions_std + 1e-8)
if np.mean(std_ratio[:7]) < 0.5:
    issues.append("⚠️  Action std is much smaller than DROID (< 50%)")

if len(issues) == 0:
    print("✓ No major issues detected")
else:
    for issue in issues:
        print(issue)

# 5. Recommendations
print("\n5. RECOMMENDATIONS")
print("-"*80)

if np.mean(std_ratio[:7]) < 0.5:
    print("1. CRITICAL: Action distribution mismatch")
    print("   - Your synthetic data has much smaller action magnitudes than real DROID data")
    print("   - This causes domain shift - the model can't generalize")
    print("   - Solutions:")
    print("     a) Regenerate synthetic data with larger, more dynamic movements")
    print("     b) Scale actions during training to match DROID distribution")
    print("     c) Use your own norm stats instead of DROID's")

if success_count == 0:
    print("\n2. All episodes marked as failed")
    print("   - This might affect training if the model uses success signals")
    print("   - Solution: Manually label successful trajectories")

if np.mean(episode_lengths) < 100:
    print("\n3. Episodes are very short")
    print("   - Short episodes provide less training signal")
    print("   - Solution: Generate longer trajectories")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
