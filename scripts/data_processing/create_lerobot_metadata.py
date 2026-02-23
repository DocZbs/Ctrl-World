#!/usr/bin/env python3
"""
Create complete LeRobot metadata files for synthetic dataset
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def create_metadata(dataset_dir: Path):
    """Create all required LeRobot metadata files"""

    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta_data"
    meta_dir.mkdir(exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    # Read first file to check if it's concatenated or per-episode
    df_first = pd.read_parquet(parquet_files[0])
    is_concatenated = len(df_first['episode_index'].unique()) > 1

    if is_concatenated:
        print("Detected concatenated format (all episodes in one file)")
        df_all = df_first
        episodes = []
        tasks = []
        episodes_stats = []

        for ep_idx in tqdm(sorted(df_all['episode_index'].unique()), desc="Processing episodes"):
            ep_df = df_all[df_all['episode_index'] == ep_idx]
            task = ep_df['language_instruction'].iloc[0] if 'language_instruction' in ep_df else ""

            episodes.append({
                "episode_index": int(ep_idx),
                "episode_chunk": 0,
                "tasks": [task],
                "length": len(ep_df)
            })

            if task and task not in [t['task'] for t in tasks]:
                tasks.append({
                    "task_index": len(tasks),
                    "task": task
                })

            # Compute stats
            obs_state = np.stack(ep_df['observation.state'].values)
            actions = np.stack(ep_df['actions'].values)
            num_frames = len(ep_df)

            episodes_stats.append({
                "episode_index": int(ep_idx),
                "stats": {
                    "observation.state": {
                        "mean": obs_state.mean(axis=0).tolist(),
                        "std": obs_state.std(axis=0).tolist(),
                        "min": obs_state.min(axis=0).tolist(),
                        "max": obs_state.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    "actions": {
                        "mean": actions.mean(axis=0).tolist(),
                        "std": actions.std(axis=0).tolist(),
                        "min": actions.min(axis=0).tolist(),
                        "max": actions.max(axis=0).tolist(),
                        "count": [num_frames],
                    }
                }
            })

        total_frames = len(df_all)

    else:
        print("Detected per-episode format (one file per episode)")
        episodes = []
        tasks = []
        episodes_stats = []
        total_frames = 0

        for ep_idx, pf in enumerate(tqdm(parquet_files, desc="Processing episodes")):
            df = pd.read_parquet(pf)
            task = df['language_instruction'].iloc[0] if 'language_instruction' in df else ""

            episodes.append({
                "episode_index": ep_idx,
                "episode_chunk": 0,
                "tasks": [task],
                "length": len(df)
            })

            if task and task not in [t['task'] for t in tasks]:
                tasks.append({
                    "task_index": len(tasks),
                    "task": task
                })

            # Compute stats
            obs_state = np.stack(df['observation.state'].values)
            actions = np.stack(df['actions'].values)
            num_frames = len(df)

            episodes_stats.append({
                "episode_index": ep_idx,
                "stats": {
                    "observation.state": {
                        "mean": obs_state.mean(axis=0).tolist(),
                        "std": obs_state.std(axis=0).tolist(),
                        "min": obs_state.min(axis=0).tolist(),
                        "max": obs_state.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    "actions": {
                        "mean": actions.mean(axis=0).tolist(),
                        "std": actions.std(axis=0).tolist(),
                        "min": actions.min(axis=0).tolist(),
                        "max": actions.max(axis=0).tolist(),
                        "count": [num_frames],
                    }
                }
            })

            total_frames += len(df)

    # Save episodes.jsonl
    print("\nSaving metadata files...")
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + '\n')
    print(f"✓ Saved episodes.jsonl ({len(episodes)} episodes)")

    # Save tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    print(f"✓ Saved tasks.jsonl ({len(tasks)} tasks)")

    # Save episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", 'w') as f:
        for stat in episodes_stats:
            f.write(json.dumps(stat) + '\n')
    print(f"✓ Saved episodes_stats.jsonl")

    # Update info.json with more details (LeRobot v2.1 format)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "droid",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * 3,
        "total_chunks": 1,
        "chunks_size": len(episodes),
        "fps": 10,
        "splits": {
            "train": f"0:{len(episodes)}"
        },
        "data_path": "data/chunk-000/train.parquet" if is_concatenated else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channels"], "info": {"video.fps": 10.0, "video.height": 256, "video.width": 256, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}},
            "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channels"], "info": {"video.fps": 10.0, "video.height": 256, "video.width": 256, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}},
            "observation.images.wrist2": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channels"], "info": {"video.fps": 10.0, "video.height": 256, "video.width": 256, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}},
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
            "actions": {"dtype": "float32", "shape": [8], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        }
    }

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ Updated info.json")

    print(f"\n{'='*80}")
    print(f"✓ Metadata creation complete!")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")
    print(f"Tasks: {len(tasks)}")
    print(f"\nDataset is now ready for training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory (contains data/ and meta_data/)')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    create_metadata(dataset_dir)


if __name__ == '__main__':
    main()
