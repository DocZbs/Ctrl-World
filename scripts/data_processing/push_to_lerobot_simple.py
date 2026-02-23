#!/usr/bin/env python3
"""
Convert synthetic data to LeRobot format - simple version
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from datasets import Dataset, Features, Value, Image as HFImage, Sequence, Array2D, Array3D
import safetensors.torch


def convert_to_lerobot(input_dir: Path, output_name: str):
    """Convert droid_new_setup format to LeRobot format"""

    input_dir = Path(input_dir)
    annotation_dir = input_dir / "annotation" / "synthetic"

    # Get all annotation files
    anno_files = sorted(annotation_dir.glob("*.json"))
    print(f"Found {len(anno_files)} episodes")

    if len(anno_files) == 0:
        raise ValueError(f"No annotation files found in {annotation_dir}")

    # Collect all data as tensors
    all_frames = {
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

    episode_data_index = {'from': [], 'to': []}
    global_frame_idx = 0

    for episode_idx, anno_file in enumerate(anno_files):
        print(f"Processing episode {episode_idx}: {anno_file.name}")

        # Load annotation
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        num_frames = len(anno['states'])
        episode_start_idx = global_frame_idx

        # Load videos
        video_paths = [input_dir / vid['video_path'] for vid in anno['videos']]
        video_captures = [cv2.VideoCapture(str(vp)) for vp in video_paths]

        # Process each frame
        for frame_idx in range(num_frames):
            # Read frames from videos
            frames = []
            for cap in video_captures:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            img_top, img_wrist, img_wrist2 = frames

            # Get state and action
            state = np.array(anno['states'][frame_idx], dtype=np.float32)
            joints = np.array(anno['joints'][frame_idx], dtype=np.float32)

            # Action is next state
            if frame_idx < num_frames - 1:
                action = np.array(anno['joints'][frame_idx + 1], dtype=np.float32)
            else:
                action = joints.copy()

            # Append as tensors
            all_frames['observation.images.top'].append(torch.from_numpy(img_top))
            all_frames['observation.images.wrist'].append(torch.from_numpy(img_wrist))
            all_frames['observation.images.wrist2'].append(torch.from_numpy(img_wrist2))
            all_frames['observation.state'].append(torch.from_numpy(state))
            all_frames['action'].append(torch.from_numpy(action))
            all_frames['episode_index'].append(torch.tensor(episode_idx, dtype=torch.int64))
            all_frames['frame_index'].append(torch.tensor(frame_idx, dtype=torch.int64))
            all_frames['timestamp'].append(torch.tensor(frame_idx / 10.0, dtype=torch.float32))
            all_frames['next.done'].append(torch.tensor(frame_idx == num_frames - 1, dtype=torch.bool))

            global_frame_idx += 1

        # Close video captures
        for cap in video_captures:
            cap.release()

        episode_end_idx = global_frame_idx
        episode_data_index['from'].append(episode_start_idx)
        episode_data_index['to'].append(episode_end_idx)

    print(f"\nTotal frames: {global_frame_idx}")
    print(f"Total episodes: {len(episode_data_index['from'])}")

    # Stack all tensors
    print("\nStacking tensors...")
    for key in all_frames:
        all_frames[key] = torch.stack(all_frames[key])
        print(f"  {key}: {all_frames[key].shape}")

    # Save to project directory for easier access
    # Also create symlink in cache for compatibility
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "lerobot_datasets" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink in cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / output_name
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    if cache_dir.exists() or cache_dir.is_symlink():
        if cache_dir.is_symlink():
            cache_dir.unlink()
        else:
            import shutil
            shutil.rmtree(cache_dir)

    cache_dir.symlink_to(output_dir, target_is_directory=True)
    print(f"✓ Created symlink: {cache_dir} -> {output_dir}")

    # Save using HuggingFace datasets (will save as parquet automatically)
    print(f"\nCreating HuggingFace dataset...")
    from datasets import Dataset, Features, Value, Image as HFImage, Array2D, Array3D, Sequence
    from PIL import Image as PILImage

    # Convert tensors to correct format
    hf_data = {}
    print("Converting tensors to HuggingFace format...")
    from tqdm import tqdm

    for key, tensor in all_frames.items():
        if 'images' in key:
            print(f"  Converting {key} ({len(tensor)} images)...")
            hf_data[key] = [PILImage.fromarray(img.numpy().astype(np.uint8))
                           for img in tqdm(tensor, desc=f"    {key}", ncols=80)]
        elif key in ['observation.state', 'action']:
            hf_data[key] = [t.tolist() for t in tensor]
        else:
            hf_data[key] = [t.item() for t in tensor]

    # Define features
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

    dataset = Dataset.from_dict(hf_data, features=features)

    print(f"Saving dataset to {output_dir}...")
    # Save as parquet (OpenPI patch expects parquet files)
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(data_dir / "train.parquet"))

    # Save episode index
    meta_dir = output_dir / "meta_data"
    meta_dir.mkdir(exist_ok=True)

    safetensors.torch.save_file(
        {
            'from': torch.tensor(episode_data_index['from'], dtype=torch.int64),
            'to': torch.tensor(episode_data_index['to'], dtype=torch.int64),
        },
        str(meta_dir / "episode_data_index.safetensors")
    )

    # Save metadata
    info = {
        'codebase_version': '2.0',
        'robot_type': 'droid',
        'total_episodes': len(episode_data_index['from']),
        'total_frames': global_frame_idx,
        'fps': 10,
    }

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"  - Total episodes: {len(episode_data_index['from'])}")
    print(f"  - Total frames: {global_frame_idx}")
    print(f"  - Format: LeRobot v2.0 (safetensors)")

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-name', type=str, required=True)
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
