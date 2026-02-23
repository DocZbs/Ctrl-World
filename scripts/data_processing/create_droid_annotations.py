#!/usr/bin/env python3
"""Create annotation structure for DROID data from episodes.jsonl."""

import json
import argparse
from pathlib import Path


def create_annotations(droid_path: str):
    droid_path = Path(droid_path)
    episodes_file = droid_path / "meta" / "episodes.jsonl"

    if not episodes_file.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")

    annotation_dir = droid_path / "annotation" / "train"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading episodes from {episodes_file}")
    print(f"Creating annotations in {annotation_dir}")

    count = 0
    with open(episodes_file, 'r') as f:
        for line in f:
            episode = json.loads(line.strip())
            episode_idx = episode['episode_index']
            tasks = episode.get('tasks', [''])

            instruction = tasks[0] if tasks else ''

            annotation = {
                'episode_index': episode_idx,
                'language_instruction': instruction,
                'texts': [instruction],
                'length': episode.get('length', 0)
            }

            output_file = annotation_dir / f"{episode_idx}.json"
            with open(output_file, 'w') as out:
                json.dump(annotation, out, indent=2)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} episodes...")

    print(f"Created {count} annotation files in {annotation_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create DROID annotations from episodes.jsonl")
    parser.add_argument(
        '--droid-path',
        type=str,
        default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data',
        help='Path to DROID data directory'
    )
    args = parser.parse_args()

    create_annotations(args.droid_path)


if __name__ == '__main__':
    main()
