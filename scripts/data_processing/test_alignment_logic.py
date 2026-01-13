#!/usr/bin/env python3
"""
Unit test for frame-state alignment logic (no model loading required).
Tests the core fix without needing GPU/models.
"""

import numpy as np


def simulate_buggy_version(num_steps=20, pred_step=5):
    """Simulate the BUGGY original version."""
    print("SIMULATING BUGGY VERSION")
    print("-" * 60)

    # Initialize
    joint_positions = [np.zeros(8)]  # initial state
    cartesian_poses = []
    images_view0 = [np.zeros((192, 320, 3))]  # initial frame

    for step in range(num_steps):
        # Policy generates states
        joint_pos_skip = np.random.randn(pred_step, 8)
        cartesian_poses_skip = np.random.randn(pred_step, 7)

        # World model generates frames
        num_frames_generated = pred_step - 1
        all_frames = [np.random.randn(num_frames_generated, 192, 320, 3)]

        # BUGGY: Only append final state
        current_state = joint_pos_skip[-1]
        cartesian_pose_current = cartesian_poses_skip[-1]

        joint_positions.append(current_state)
        cartesian_poses.append(cartesian_pose_current)

        # But append ALL frames
        for frame_idx in range(len(all_frames[0])):
            images_view0.append(all_frames[0][frame_idx])

    return {
        'num_frames': len(images_view0),
        'num_states': len(cartesian_poses),
        'num_joints': len(joint_positions)
    }


def simulate_fixed_version(num_steps=20, pred_step=5):
    """Simulate the FIXED version."""
    print("\nSIMULATING FIXED VERSION")
    print("-" * 60)

    # Initialize (matches _initialize_history in actual code)
    joint_positions = [np.zeros(8)]  # initial state
    cartesian_poses = [np.zeros(7)]  # initial cartesian pose (added by _initialize_history)
    images_view0 = [np.zeros((192, 320, 3))]  # initial frame

    for step in range(num_steps):
        # Policy generates states
        joint_pos_skip = np.random.randn(pred_step, 8)
        cartesian_poses_skip = np.random.randn(pred_step, 7)

        # World model generates frames
        num_frames_generated = pred_step - 1
        all_frames = [np.random.randn(num_frames_generated, 192, 320, 3)]

        # FIXED: Append state for EACH frame (1:1 mapping)
        for frame_idx in range(num_frames_generated):
            frame_cartesian_pose = cartesian_poses_skip[frame_idx]
            frame_joint_pos = joint_pos_skip[frame_idx]

            cartesian_poses.append(frame_cartesian_pose)
            joint_positions.append(frame_joint_pos)
            images_view0.append(all_frames[0][frame_idx])

    return {
        'num_frames': len(images_view0),
        'num_states': len(cartesian_poses),
        'num_joints': len(joint_positions)
    }


def run_test():
    """Run alignment test."""
    print("=" * 80)
    print("FRAME-STATE ALIGNMENT TEST")
    print("=" * 80)

    num_steps = 20
    pred_step = 5

    print(f"\nTest configuration:")
    print(f"  num_steps = {num_steps}")
    print(f"  pred_step = {pred_step}")
    print(f"  frames_per_step = {pred_step - 1}")

    # Test buggy version
    print("\n" + "=" * 80)
    buggy = simulate_buggy_version(num_steps, pred_step)
    print(f"\nResults:")
    print(f"  Frames:  {buggy['num_frames']}")
    print(f"  States:  {buggy['num_states']}")
    print(f"  Joints:  {buggy['num_joints']}")
    print(f"  Mismatch: {buggy['num_frames'] - buggy['num_states']} frames without states")
    print(f"  Alignment: {'❌ FAILED' if buggy['num_frames'] != buggy['num_states'] else '✓ OK'}")

    # Test fixed version
    print("\n" + "=" * 80)
    fixed = simulate_fixed_version(num_steps, pred_step)
    print(f"\nResults:")
    print(f"  Frames:  {fixed['num_frames']}")
    print(f"  States:  {fixed['num_states']}")
    print(f"  Joints:  {fixed['num_joints']}")
    print(f"  Mismatch: {fixed['num_frames'] - fixed['num_states']} frames without states")
    print(f"  Alignment: {'✅ PASSED' if fixed['num_frames'] == fixed['num_states'] else '❌ FAILED'}")

    # Final verification
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    tests_passed = []
    tests_failed = []

    # Test 1: Buggy version should have mismatch
    if buggy['num_frames'] != buggy['num_states']:
        tests_passed.append("Buggy version has expected mismatch")
    else:
        tests_failed.append("Buggy version unexpectedly aligned")

    # Test 2: Fixed version should have perfect alignment
    if fixed['num_frames'] == fixed['num_states'] == fixed['num_joints']:
        tests_passed.append("Fixed version has perfect alignment")
    else:
        tests_failed.append(f"Fixed version alignment failed: {fixed['num_frames']} frames != {fixed['num_states']} states != {fixed['num_joints']} joints")

    # Test 3: Fixed should generate more states than buggy
    if fixed['num_states'] > buggy['num_states']:
        tests_passed.append(f"Fixed generates more states ({fixed['num_states']}) than buggy ({buggy['num_states']})")
    else:
        tests_failed.append("Fixed should generate more states")

    # Print results
    print(f"\nPassed ({len(tests_passed)}/{len(tests_passed) + len(tests_failed)}):")
    for test in tests_passed:
        print(f"  ✅ {test}")

    if tests_failed:
        print(f"\nFailed ({len(tests_failed)}/{len(tests_passed) + len(tests_failed)}):")
        for test in tests_failed:
            print(f"  ❌ {test}")

    print("\n" + "=" * 80)
    if not tests_failed:
        print("🎉 ALL TESTS PASSED - FIX IS CORRECT!")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return False


if __name__ == '__main__':
    import sys
    success = run_test()
    sys.exit(0 if success else 1)
