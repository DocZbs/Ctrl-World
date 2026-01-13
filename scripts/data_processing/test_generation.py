#!/usr/bin/env python3
"""
Quick test to verify synthetic trajectory generation produces correct format.
Runs without VLM and minimal steps for speed.
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from generate_synthetic_trajectories import (
    SyntheticDataGenerator,
    load_initial_observation
)


def test_generation(annotation_path: str, dataset_root: str):
    """Test generation with minimal configuration."""

    print("=" * 80)
    print("TESTING SYNTHETIC TRAJECTORY GENERATION")
    print("=" * 80)

    # Load initial observation
    print(f"\n1. Loading initial observation from {annotation_path}")
    initial_obs = load_initial_observation(annotation_path, dataset_root)
    print(f"   Task: {initial_obs['instruction']}")
    print(f"   Initial state shape: {initial_obs['state'].shape}")
    print(f"   Number of camera views: {len(initial_obs['images'])}")

    # Create generator (without VLM for speed)
    print(f"\n2. Creating generator (VLM disabled for testing)")
    generator = SyntheticDataGenerator(
        wm_ckpt='/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt',
        svd_model_path='/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e',
        clip_model_path='/mnt/nvme-fast/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        pi_ckpt='/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid',
        action_adapter_path='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/train_adapter/checkpoints/adapter_finetuned_best.pth',
        wm_device='cuda:0',
        policy_device='cuda:1',
        use_vlm=False,  # Disable VLM for quick test
    )

    # Generate one trajectory with minimal steps
    print(f"\n3. Generating test trajectory (max 5 steps)")
    trajectory = generator.generate_trajectory(
        initial_observation=initial_obs,
        instruction=initial_obs['instruction'],
        max_steps=5,  # Only 5 steps for quick test
        min_steps=10,  # High value prevents VLM checks
    )

    # Check alignment
    print(f"\n4. Checking data alignment")
    print(f"   Number of steps executed: {trajectory.metadata['num_steps']}")
    print(f"   Video length (frames): {len(trajectory.images_view0)}")
    print(f"   Number of states: {len(trajectory.cartesian_poses)}")
    print(f"   Number of joints: {len(trajectory.joint_positions)}")

    # Verify alignment
    video_length = len(trajectory.images_view0)
    num_states = len(trajectory.cartesian_poses)
    num_joints = len(trajectory.joint_positions)

    errors = []

    if video_length != num_states:
        errors.append(f"Video length ({video_length}) != states ({num_states})")

    if video_length != num_joints:
        errors.append(f"Video length ({video_length}) != joints ({num_joints})")

    if num_states != num_joints:
        errors.append(f"States ({num_states}) != joints ({num_joints})")

    # Check dimensions
    if len(trajectory.cartesian_poses) > 0:
        state_dim = len(trajectory.cartesian_poses[0])
        if state_dim != 7:
            errors.append(f"State dimension should be 7, got {state_dim}")

    if len(trajectory.joint_positions) > 0:
        joint_dim = len(trajectory.joint_positions[0])
        if joint_dim != 8:
            errors.append(f"Joint dimension should be 8, got {joint_dim}")

    print("\n" + "=" * 80)
    if errors:
        print("❌ TEST FAILED - ALIGNMENT ERRORS:")
        for error in errors:
            print(f"   - {error}")
        print("=" * 80)
        return False
    else:
        print("✅ TEST PASSED - PERFECT ALIGNMENT!")
        print(f"   ✓ video_length ({video_length}) == states ({num_states}) == joints ({num_joints})")
        print(f"   ✓ State dimension: 7 (xyz + euler + gripper)")
        print(f"   ✓ Joint dimension: 8 (7 joints + gripper)")
        print("=" * 80)

        # Show sample data
        print("\nSample data (first 2 frames):")
        print(f"\n  Frame 0:")
        print(f"    State:  {trajectory.cartesian_poses[0]}")
        print(f"    Joints: {trajectory.joint_positions[0]}")
        print(f"\n  Frame 1:")
        print(f"    State:  {trajectory.cartesian_poses[1]}")
        print(f"    Joints: {trajectory.joint_positions[1]}")

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str,
                       default='dataset_example/droid_new_setup/annotation/val/0002.json')
    parser.add_argument('--dataset-root', type=str,
                       default='dataset_example/droid_new_setup')
    args = parser.parse_args()

    try:
        success = test_generation(args.annotation_file, args.dataset_root)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
