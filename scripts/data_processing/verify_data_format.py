#!/usr/bin/env python3
"""
Verify that synthetic data format matches droid_new_setup format exactly.
"""

import json
import sys
from pathlib import Path


def verify_annotation(annotation_path: Path, verbose: bool = True):
    """Verify annotation file format."""
    with open(annotation_path) as f:
        data = json.load(f)

    required_keys = ['texts', 'episode_id', 'success', 'video_length', 'videos', 'states', 'joints']
    missing_keys = [k for k in required_keys if k not in data]

    if missing_keys:
        print(f"ERROR: Missing keys: {missing_keys}")
        return False

    video_length = data['video_length']
    num_states = len(data['states'])
    num_joints = len(data['joints'])
    num_videos = len(data['videos'])

    if verbose:
        print(f"\nAnnotation: {annotation_path.name}")
        print(f"  Task: {data['texts'][0]}")
        print(f"  Episode ID: {data['episode_id']}")
        print(f"  Success: {data['success']}")
        print(f"  Video length: {video_length}")
        print(f"  Number of states: {num_states}")
        print(f"  Number of joints: {num_joints}")
        print(f"  Number of videos: {num_videos}")

    # Critical checks
    errors = []

    if num_videos != 3:
        errors.append(f"Expected 3 videos, got {num_videos}")

    if num_states != video_length:
        errors.append(f"States length ({num_states}) != video_length ({video_length})")

    if num_joints != video_length:
        errors.append(f"Joints length ({num_joints}) != video_length ({video_length})")

    # Check state format (should be 7 values: xyz, euler, gripper)
    if num_states > 0:
        state_dim = len(data['states'][0])
        if state_dim != 7:
            errors.append(f"State dimension should be 7, got {state_dim}")

    # Check joint format (should be 8 values: 7 joints + 1 gripper)
    if num_joints > 0:
        joint_dim = len(data['joints'][0])
        if joint_dim != 8:
            errors.append(f"Joint dimension should be 8, got {joint_dim}")

    if errors:
        print(f"\n  ERRORS:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        if verbose:
            print(f"  ✓ Format correct")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_data_format.py <annotation_dir>")
        sys.exit(1)

    annotation_dir = Path(sys.argv[1])

    if not annotation_dir.exists():
        print(f"ERROR: Directory not found: {annotation_dir}")
        sys.exit(1)

    annotation_files = sorted(annotation_dir.glob("*.json"))

    if not annotation_files:
        print(f"ERROR: No JSON files found in {annotation_dir}")
        sys.exit(1)

    print(f"Verifying {len(annotation_files)} annotation files...")
    print("=" * 80)

    all_valid = True
    for ann_file in annotation_files:
        valid = verify_annotation(ann_file, verbose=True)
        if not valid:
            all_valid = False

    print("=" * 80)
    if all_valid:
        print("✓ All annotations are valid!")
        return 0
    else:
        print("✗ Some annotations have errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
