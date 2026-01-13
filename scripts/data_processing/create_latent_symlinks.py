#!/usr/bin/env python3
"""Create symlinks to map chunk latents to expected directory structure."""

import argparse
from pathlib import Path


def create_symlinks_for_chunk(data_root, chunk_id, split="val"):
    """Create symlinks from latent_videos/{split}/{ep_id} to latents/chunk-{chunk_id}/episode_{ep_id}."""

    data_root = Path(data_root)
    chunk_str = f"chunk-{chunk_id:03d}"

    # Source: data/latents/chunk-000/
    source_base = data_root / "latents" / chunk_str

    # Target: data/latent_videos/val/
    target_base = data_root / "latent_videos" / split
    target_base.mkdir(parents=True, exist_ok=True)

    if not source_base.exists():
        print(f"Error: Source directory not found: {source_base}")
        return

    # Find all episode directories in the chunk
    episode_dirs = sorted(source_base.glob("episode_*"))

    print(f"Creating symlinks for {len(episode_dirs)} episodes from {chunk_str}...")

    for ep_dir in episode_dirs:
        # Extract episode ID from "episode_000001" -> 1
        ep_str = ep_dir.name
        ep_id = int(ep_str.replace("episode_", ""))

        # Target symlink: data/latent_videos/val/1 -> ../../latents/chunk-000/episode_000001
        target_link = target_base / str(ep_id)

        # Calculate relative path from target to source
        relative_source = Path("../../latents") / chunk_str / ep_str

        # Create symlink (skip if already exists)
        if target_link.exists() or target_link.is_symlink():
            # print(f"  Skipping {ep_id} (already exists)")
            continue

        target_link.symlink_to(relative_source)
        if ep_id <= 10 or ep_id % 100 == 0:
            print(f"  ✓ {ep_id} -> {relative_source}")

    print(f"Done! Created symlinks in {target_base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    create_symlinks_for_chunk(args.data_root, args.chunk_id, args.split)


if __name__ == "__main__":
    main()
