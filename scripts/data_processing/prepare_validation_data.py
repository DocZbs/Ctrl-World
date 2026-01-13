#!/usr/bin/env python3
"""Prepare validation data from chunk-000 episodes with non-empty tasks."""

import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data",
                       help="Data root directory")
    parser.add_argument("--chunk-id", type=int, default=0,
                       help="Chunk ID to process")
    parser.add_argument("--output-dir", type=str, default="data/validation",
                       help="Output directory for validation annotations")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    episodes_file = data_root / "meta" / "episodes.jsonl"
    output_dir = Path(args.output_dir)

    # Create output directories
    ann_dir = output_dir / "annotation" / "val"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Calculate episode range for this chunk
    start_id = args.chunk_id * 1000
    end_id = start_id + 999

    print(f"Processing chunk-{args.chunk_id:03d}")
    print(f"Episode range: {start_id} - {end_id}")
    print(f"Reading episodes from: {episodes_file}")

    # Read episodes and filter non-empty tasks
    valid_episodes = []
    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            ep_id = ep["episode_index"]

            # Check if in range
            if not (start_id <= ep_id <= end_id):
                continue

            # Check if has non-empty task
            tasks = ep.get("tasks", [])
            if tasks and tasks[0] and tasks[0].strip():
                valid_episodes.append({
                    "episode_id": ep_id,
                    "task": tasks[0],
                    "length": ep.get("length", 0)
                })

    print(f"\nFound {len(valid_episodes)} episodes with non-empty tasks")

    # Create annotation files for each valid episode
    chunk_str = f"chunk-{args.chunk_id:03d}"

    for ep in valid_episodes:
        ep_id = ep["episode_id"]
        ep_str = f"episode_{ep_id:06d}"
        task = ep["task"]
        length = ep["length"]

        # Check if video files exist
        video_base = data_root / "videos" / chunk_str
        cam_folders = [
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left"
        ]

        video_paths = [video_base / cam / f"{ep_str}.mp4" for cam in cam_folders]

        # Check existence
        if not all(vp.exists() for vp in video_paths):
            print(f"  Warning: Videos not found for {ep_str}, skipping...")
            continue

        # Check if latent files exist
        latent_base = data_root / "latents" / chunk_str / ep_str
        latents_exist = all(
            (latent_base / f"{i}.pt").exists()
            for i in range(3)
        )

        if not latents_exist:
            print(f"  Warning: Latents not found for {ep_str}, skipping...")
            continue

        # Create annotation file
        annotation = {
            "texts": [task],
            "episode_id": ep_id,
            "success": 0,  # Unknown
            "video_length": length,
            "state_length": length,
            "raw_length": length,
            "videos": [
                {"video_path": f"videos/{chunk_str}/{cam}/{ep_str}.mp4"}
                for cam in cam_folders
            ],
            "latent_videos": [
                {"latent_video_path": f"latents/{chunk_str}/{ep_str}/{i}.pt"}
                for i in range(3)
            ]
        }

        # Save annotation
        ann_path = ann_dir / f"{ep_id}.json"
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        print(f"  ✓ {ep_id}: {task[:50]}...")

    print(f"\nCreated {len(list(ann_dir.glob('*.json')))} annotation files in {ann_dir}")
    print(f"\nTo run validation, use:")
    print(f"  python scripts/run_all_droid_new_setup.py \\")
    print(f"    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \\")
    print(f"    --ann-dir {ann_dir} \\")
    print(f"    --droid-root {args.data_root} \\")
    print(f"    --out-base experiments/validation_chunk_{args.chunk_id:03d}")


if __name__ == "__main__":
    main()
