#!/usr/bin/env python3
"""
将合成轨迹转换为LeRobot格式

用法:
    python scripts/convert_synthetic_to_lerobot.py \
        --input-dir synthetic_data/pickplace_success \
        --output-dir lerobot_synthetic/pickplace \
        --repo-id "local://lerobot_synthetic/pickplace"
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image


def convert_to_lerobot_format(
    synthetic_dir: Path,
    output_dir: Path,
    repo_id: str = None,
):
    """
    将合成轨迹转换为LeRobot格式

    LeRobot格式要求:
    - episodes/: 每个episode一个目录
    - episode_xxx/: 包含images和data
    - meta.json: 数据集元信息
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all trajectory directories
    traj_dirs = sorted([d for d in synthetic_dir.iterdir() if d.is_dir()])

    print(f"Converting {len(traj_dirs)} trajectories to LeRobot format...")

    episodes_data = []
    total_frames = 0

    for episode_idx, traj_dir in enumerate(tqdm(traj_dirs)):
        # Load trajectory metadata
        metadata_file = traj_dir / 'metadata.json'
        if not metadata_file.exists():
            print(f"Warning: No metadata for {traj_dir.name}, skipping")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Create episode directory
        episode_dir = output_dir / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(exist_ok=True)

        # Load images
        images_dir = traj_dir / 'images'
        if not images_dir.exists():
            print(f"Warning: No images for {traj_dir.name}, skipping")
            continue

        image_files = sorted(images_dir.glob('*.png'))

        # Process each frame
        states = metadata['states']
        actions = metadata['actions']
        num_frames = len(image_files)

        episode_data = {
            'episode_index': episode_idx,
            'task': metadata['task_instruction'],
            'length': num_frames,
            'frames': [],
        }

        for frame_idx, img_file in enumerate(image_files):
            # Copy image
            img_output = episode_dir / f"frame_{frame_idx:06d}.png"
            Image.open(img_file).save(img_output)

            # Frame data
            frame_data = {
                'frame_index': frame_idx,
                'timestamp': frame_idx / 10.0,  # Assuming 10 FPS
                'observation': {
                    'image': str(img_output.relative_to(output_dir)),
                    'state': states[frame_idx] if frame_idx < len(states) else states[-1],
                },
            }

            # Add action if available
            if frame_idx < len(actions):
                frame_data['action'] = actions[frame_idx]
            else:
                # Last frame has no action
                frame_data['action'] = [0.0] * len(actions[0])

            episode_data['frames'].append(frame_data)

        episodes_data.append(episode_data)
        total_frames += num_frames

    # Save episode index
    episodes_file = output_dir / 'episodes.json'
    with open(episodes_file, 'w') as f:
        json.dump(episodes_data, f, indent=2)

    # Create meta.json
    meta = {
        'repo_id': repo_id or f"local://{output_dir}",
        'num_episodes': len(episodes_data),
        'total_frames': total_frames,
        'fps': 10,
        'robot_type': 'franka',
        'action_space': 'joint_velocity',
        'observation_space': {
            'image': {'shape': [224, 224, 3], 'dtype': 'uint8'},
            'state': {'shape': [7], 'dtype': 'float32'},
        },
        'action_dim': len(episodes_data[0]['frames'][0]['action']) if episodes_data else 7,
    }

    meta_file = output_dir / 'meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 80)
    print("Conversion complete!")
    print(f"Episodes: {len(episodes_data)}")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")
    print(f"Meta file: {meta_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Convert synthetic trajectories to LeRobot format')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing successful synthetic trajectories')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for LeRobot dataset')
    parser.add_argument('--repo-id', type=str,
                       help='LeRobot repo ID (e.g., local://path or username/dataset)')

    args = parser.parse_args()

    convert_to_lerobot_format(
        synthetic_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        repo_id=args.repo_id,
    )


if __name__ == '__main__':
    main()
