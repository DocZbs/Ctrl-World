#!/usr/bin/env python3
"""
Convert droid_new_setup format synthetic data to LeRobot format and push to local cache
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2

def convert_to_lerobot(input_dir: Path, output_name: str):
    """Convert droid_new_setup format to LeRobot format"""

    input_dir = Path(input_dir)
    annotation_dir = input_dir / "annotation" / "synthetic"

    # Get all annotation files
    anno_files = sorted(annotation_dir.glob("*.json"))
    print(f"Found {len(anno_files)} episodes")

    if len(anno_files) == 0:
        raise ValueError(f"No annotation files found in {annotation_dir}")

    # Prepare output directory
    from datasets import Dataset, Features, Value, Image as HFImage, Sequence

    # Collect all data
    all_data = {
        'observation.images.top': [],
        'observation.images.wrist': [],
        'observation.images.wrist2': [],
        'observation.state': [],
        'action': [],
        'episode_index': [],
        'frame_index': [],
        'timestamp': [],
        'next.done': [],
    }

    episode_data_index = {
        'from': [],
        'to': [],
    }

    global_frame_idx = 0

    for episode_idx, anno_file in enumerate(anno_files):
        print(f"Processing episode {episode_idx}: {anno_file.name}")

        # Load annotation
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        episode_id = anno_file.stem
        num_frames = len(anno['states'])

        episode_start_idx = global_frame_idx

        # Load videos
        video_paths = [input_dir / vid['video_path'] for vid in anno['videos']]
        video_captures = [cv2.VideoCapture(str(vp)) for vp in video_paths]

        # Process each frame in episode
        for frame_idx in range(num_frames):
            # Read frames from videos
            frames = []
            for cap in video_captures:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx} from video")
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

            img_top, img_wrist, img_wrist2 = frames

            # Get state and action
            state = np.array(anno['states'][frame_idx], dtype=np.float32)  # (7,) cartesian pose
            joints = np.array(anno['joints'][frame_idx], dtype=np.float32)  # (8,) joint positions with gripper

            # Action is the next state (for last frame, repeat current)
            if frame_idx < num_frames - 1:
                action = np.array(anno['joints'][frame_idx + 1], dtype=np.float32)
            else:
                action = joints.copy()

            # Add to dataset
            all_data['observation.images.top'].append(img_top)
            all_data['observation.images.wrist'].append(img_wrist)
            all_data['observation.images.wrist2'].append(img_wrist2)
            all_data['observation.state'].append(state.tolist())
            all_data['action'].append(action.tolist())
            all_data['episode_index'].append(int(episode_idx))
            all_data['frame_index'].append(int(frame_idx))
            all_data['timestamp'].append(float(frame_idx / 10.0))  # Assuming 10 FPS
            all_data['next.done'].append(bool(frame_idx == num_frames - 1))

            global_frame_idx += 1

        # Close video captures
        for cap in video_captures:
            cap.release()

        episode_end_idx = global_frame_idx
        episode_data_index['from'].append(episode_start_idx)
        episode_data_index['to'].append(episode_end_idx)

    print(f"\nTotal frames: {global_frame_idx}")
    print(f"Total episodes: {len(episode_data_index['from'])}")

    # Create HuggingFace Dataset
    features = Features({
        'observation.images.top': HFImage(),
        'observation.images.wrist': HFImage(),
        'observation.images.wrist2': HFImage(),
        'observation.state': Sequence(Value('float32'), length=7),
        'action': Sequence(Value('float32'), length=8),
        'episode_index': Value('int64'),
        'frame_index': Value('int64'),
        'timestamp': Value('float32'),
        'next.done': Value('bool'),
    })

    hf_dataset = Dataset.from_dict(all_data, features=features)

    # Save to LeRobot format
    output_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    hf_dataset.save_to_disk(str(output_dir / "train"))

    # Save episode data index
    episode_data_index_path = output_dir / "meta_data" / "episode_data_index.safetensors"
    episode_data_index_path.parent.mkdir(parents=True, exist_ok=True)

    import safetensors.torch
    safetensors.torch.save_file(
        {
            'from': torch.tensor(episode_data_index['from'], dtype=torch.int64),
            'to': torch.tensor(episode_data_index['to'], dtype=torch.int64),
        },
        str(episode_data_index_path)
    )

    # Create info.json
    info = {
        'codebase_version': '2.0',
        'robot_type': 'droid',
        'total_episodes': len(episode_data_index['from']),
        'total_frames': global_frame_idx,
        'fps': 10,
        'splits': {
            'train': f'0:{len(episode_data_index["from"])}'
        },
        'data_path': str(output_dir / "train"),
        'video_path': str(output_dir / "videos" / "chunk-000"),
    }

    with open(output_dir / "meta_data" / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Create dataset_info.json
    dataset_info = {
        '_data_files': [{'split': 'train'}],
        '_fingerprint': 'synthetic',
        'features': hf_dataset.features.to_dict(),
    }

    with open(output_dir / "train" / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"  - Total episodes: {len(episode_data_index['from'])}")
    print(f"  - Total frames: {global_frame_idx}")
    print(f"  - Format: LeRobot v2.0")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Convert synthetic data to LeRobot format')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory with droid_new_setup format')
    parser.add_argument('--output-name', type=str, required=True,
                       help='Output dataset name (e.g., synthetic_pickplace_0002)')

    args = parser.parse_args()

    print("="*80)
    print("Convert Synthetic Data to LeRobot Format")
    print("="*80)
    print()

    convert_to_lerobot(args.input_dir, args.output_name)

    print()
    print("="*80)
    print("✓ Conversion complete!")
    print("="*80)
    print()
    print(f"You can now train with: --data.repo-id local/{args.output_name}")


if __name__ == '__main__':
    main()
