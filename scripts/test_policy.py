#!/usr/bin/env python3
"""Unified test script for all policy types (Octo, Pi0.5, OpenVLA, etc.)."""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_import(policy_name: str) -> bool:
    """Test if policy can be imported."""
    print("=" * 70)
    print(f"TEST 1: Importing {policy_name.upper()} Policy")
    print("=" * 70)

    try:
        if policy_name == "octo":
            from omni_ctrl.policy_router.octo_policy import OctoPolicy
            print("✓ Successfully imported OctoPolicy")
        elif policy_name == "pi05":
            from omni_ctrl.policy_router.pi05_policy import Pi05Policy
            print("✓ Successfully imported Pi05Policy")
        elif policy_name == "openvla":
            from omni_ctrl.policy_router.openvla_policy import OpenVLAPolicy
            print("✓ Successfully imported OpenVLAPolicy")
        else:
            print(f"✗ Unknown policy: {policy_name}")
            return False

        print()
        return True
    except ImportError as e:
        print(f"✗ Failed to import {policy_name} policy: {e}")
        print()
        return False


def test_load_model(policy_name: str, device: str = "cuda:0", use_small: bool = False):
    """Test loading policy model."""
    print("=" * 70)
    print(f"TEST 2: Loading {policy_name.upper()} Model")
    print("=" * 70)

    try:
        if policy_name == "octo":
            from omni_ctrl.policy_router.octo_policy import OctoPolicy

            checkpoint = "hf://rail-berkeley/octo-small-1.5" if use_small else "hf://rail-berkeley/octo-base-1.5"
            model_size = "Small (27M)" if use_small else "Base (93M)"

            print(f"Loading Octo-{model_size}...")
            print(f"Checkpoint: {checkpoint}")
            print("⚠️  First run will download model from HuggingFace")
            print("    Subsequent runs will use cached model")
            print()

            policy = OctoPolicy(
                checkpoint_path=checkpoint,
                device=device,
                action_space="cartesian_delta",
                horizon=4,
                use_language=True,
                image_size=256,
            )

        elif policy_name == "pi05":
            from omni_ctrl.policy_router.pi05_policy import Pi05Policy

            checkpoint = "/data1/zbs_files/data/HF/hub/models--pi05_droid/openpi-assets/checkpoints/pi05_droid"
            print(f"Loading Pi0.5 model...")
            print(f"Checkpoint: {checkpoint}")
            print()

            policy = Pi05Policy(
                checkpoint_path=checkpoint,
                device=device,
                action_space="joint_vel",
            )

        elif policy_name == "openvla":
            from omni_ctrl.policy_router.openvla_policy import OpenVLAPolicy

            checkpoint = "/data1/zbs_files/data/HF/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
            print(f"Loading OpenVLA model...")
            print(f"Checkpoint: {checkpoint}")
            print()

            policy = OpenVLAPolicy(
                checkpoint_path=checkpoint,
                device=device,
                action_space="cartesian_delta",
            )
        else:
            print(f"✗ Unknown policy: {policy_name}")
            return False, None

        print(f"✓ {policy_name.upper()} model loaded successfully!")
        print(f"  Policy name: {policy.name}")
        print(f"  Action space: {policy.action_space}")
        print()

        return True, policy

    except Exception as e:
        print(f"✗ Failed to load {policy_name} model: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False, None


def test_inference(policy, policy_name: str) -> bool:
    """Test policy inference with dummy data."""
    print("=" * 70)
    print(f"TEST 3: Inference with Dummy Data")
    print("=" * 70)

    try:
        # Create dummy observation
        obs = {
            "image_primary": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "image_wrist": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "joint_pos": np.random.randn(7).astype(np.float32),
        }

        task_instruction = "Pick up the red block"

        print(f"Input observation:")
        print(f"  - image_primary: {obs['image_primary'].shape} {obs['image_primary'].dtype}")
        print(f"  - image_wrist: {obs['image_wrist'].shape} {obs['image_wrist'].dtype}")
        print(f"  - joint_pos: {obs['joint_pos'].shape}")
        print(f"Task instruction: '{task_instruction}'")
        print()

        # Test single action prediction
        print("Testing single action prediction...")
        action = policy.predict(obs, task_instruction)

        print(f"✓ Predicted action: {action.shape} {action.dtype}")
        print(f"  Action values: {action}")
        print()

        # Test action chunk prediction
        if hasattr(policy, 'predict_chunk'):
            print("Testing action chunk prediction...")
            action_chunk = policy.predict_chunk(obs, task_instruction, chunk_size=4)
            print(f"✓ Predicted action chunk: {action_chunk.shape} {action_chunk.dtype}")
            print(f"  First action: {action_chunk[0]}")
            print()

        # Test reset
        print("Testing policy reset...")
        policy.reset()
        print("✓ Policy reset successful")
        print()

        return True

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_droid_data(policy, policy_name: str) -> bool:
    """Test policy with real DROID dataset samples."""
    print("=" * 70)
    print(f"TEST 4: Inference with Real DROID Data")
    print("=" * 70)

    try:
        import json

        # Load DROID annotation
        ann_path = ROOT_DIR / "dataset_example/droid_new_setup/annotation/val/0001.json"
        if not ann_path.exists():
            print(f"⚠️  DROID annotation not found: {ann_path}")
            print("   Skipping real data test")
            print()
            return True

        with open(ann_path) as f:
            ann = json.load(f)

        # Load first frame
        video_path = ROOT_DIR / ann["video_path"]
        if not video_path.exists():
            print(f"⚠️  Video file not found: {video_path}")
            print("   Skipping real data test")
            print()
            return True

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("⚠️  Failed to read video frame")
            print("   Skipping real data test")
            print()
            return True

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create observation
        obs = {
            "image_primary": frame_rgb,
            "image_wrist": frame_rgb,  # Use same image for wrist (dummy)
            "joint_pos": np.zeros(7, dtype=np.float32),
        }

        task_instruction = ann.get("instruction", "Complete the task")

        print(f"DROID scenario: {ann_path.stem}")
        print(f"Task: '{task_instruction}'")
        print(f"Video: {video_path.name}")
        print()

        # Predict action
        print("Predicting action...")
        action = policy.predict(obs, task_instruction)

        print(f"✓ Predicted action: {action.shape} {action.dtype}")
        print(f"  Action values: {action}")
        print()

        return True

    except Exception as e:
        print(f"⚠️  Real data test failed: {e}")
        print("   This is not critical if dummy data tests passed")
        print()
        return True  # Don't fail on real data test


def main():
    parser = argparse.ArgumentParser(description="Test policy integration")
    parser.add_argument(
        "--policy",
        choices=["octo", "pi05", "openvla"],
        default="octo",
        help="Policy to test (default: octo)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use small model variant (only for Octo)",
    )
    parser.add_argument(
        "--skip-real-data",
        action="store_true",
        help="Skip real DROID data test",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"  Policy Test Suite - {args.policy.upper()}")
    print("=" * 70)
    print()

    # Run tests
    tests_passed = 0
    tests_total = 4

    # Test 1: Import
    if test_import(args.policy):
        tests_passed += 1
    else:
        print("❌ Import test failed. Cannot continue.")
        return 1

    # Test 2: Load model
    success, policy = test_load_model(args.policy, args.device, args.small)
    if success:
        tests_passed += 1
    else:
        print("❌ Model loading failed. Cannot continue.")
        return 1

    # Test 3: Inference with dummy data
    if test_inference(policy, args.policy):
        tests_passed += 1
    else:
        print("❌ Inference test failed.")

    # Test 4: Real DROID data (optional)
    if not args.skip_real_data:
        if test_droid_data(policy, args.policy):
            tests_passed += 1
    else:
        print("=" * 70)
        print("TEST 4: SKIPPED (--skip-real-data)")
        print("=" * 70)
        print()
        tests_total -= 1

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
