#!/usr/bin/env python3
"""
轨迹标注和筛选工具

用于人工标注合成轨迹的成功/失败，并筛选出成功的轨迹用于微调。

用法:
    # 交互式标注
    python scripts/label_trajectories.py \
        --input-dir synthetic_data/pickplace \
        --output-file synthetic_data/pickplace/labels.json

    # 筛选成功的轨迹
    python scripts/label_trajectories.py \
        --input-dir synthetic_data/pickplace \
        --labels-file synthetic_data/pickplace/labels.json \
        --filter-success \
        --output-dir synthetic_data/pickplace_success
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import shutil
from tqdm import tqdm
import cv2


class TrajectoryLabeler:
    """轨迹标注工具"""

    def __init__(self, trajectories_dir: Path):
        self.trajectories_dir = trajectories_dir
        self.trajectory_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
        print(f"Found {len(self.trajectory_dirs)} trajectories")

    def load_trajectory_metadata(self, traj_dir: Path) -> Dict:
        """加载轨迹元数据"""
        metadata_file = traj_dir / 'metadata.json'
        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            return json.load(f)

    def show_trajectory_video(self, traj_dir: Path):
        """显示轨迹视频"""
        video_path = traj_dir / 'video.mp4'
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))

        print("\nPlaying video... (Press 'q' to quit, 's' to mark success, 'f' to mark failure)")

        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow('Trajectory Video', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                return 'success'
            elif key == ord('f'):
                cap.release()
                cv2.destroyAllWindows()
                return 'failure'

        cap.release()
        cv2.destroyAllWindows()
        return None

    def interactive_labeling(self) -> Dict[str, bool]:
        """交互式标注所有轨迹"""
        labels = {}

        print("\n" + "=" * 80)
        print("Interactive Trajectory Labeling")
        print("=" * 80)
        print("\nControls:")
        print("  's' - Mark as SUCCESS")
        print("  'f' - Mark as FAILURE")
        print("  'q' - Quit current video")
        print("  'n' - Skip to next")
        print("=" * 80)

        for i, traj_dir in enumerate(self.trajectory_dirs):
            traj_id = traj_dir.name
            metadata = self.load_trajectory_metadata(traj_dir)

            if metadata is None:
                print(f"\nSkipping {traj_id} (no metadata)")
                continue

            print(f"\n[{i+1}/{len(self.trajectory_dirs)}] Trajectory: {traj_id}")
            print(f"Task: {metadata.get('task_instruction', 'N/A')}")
            print(f"Steps: {metadata.get('metadata', {}).get('num_steps', 'N/A')}")

            # Show video and get label
            result = self.show_trajectory_video(traj_dir)

            if result == 'success':
                labels[traj_id] = True
                print(f"✓ Marked as SUCCESS")
            elif result == 'failure':
                labels[traj_id] = False
                print(f"✗ Marked as FAILURE")
            else:
                print(f"⊘ Skipped")

        return labels

    def save_labels(self, labels: Dict[str, bool], output_file: Path):
        """保存标注结果"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(labels, f, indent=2)

        success_count = sum(labels.values())
        total_count = len(labels)

        print(f"\n✓ Labels saved to {output_file}")
        print(f"Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    def load_labels(self, labels_file: Path) -> Dict[str, bool]:
        """加载标注结果"""
        with open(labels_file) as f:
            return json.load(f)

    def filter_successful_trajectories(
        self,
        labels: Dict[str, bool],
        output_dir: Path,
    ):
        """筛选并复制成功的轨迹"""
        output_dir.mkdir(parents=True, exist_ok=True)

        successful_trajs = [traj_id for traj_id, success in labels.items() if success]

        print(f"\nFiltering {len(successful_trajs)} successful trajectories...")

        for traj_id in tqdm(successful_trajs):
            src_dir = self.trajectories_dir / traj_id
            dst_dir = output_dir / traj_id

            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

                # Update metadata to mark as successful
                metadata_file = dst_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    metadata['success'] = True
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

        print(f"✓ Copied {len(successful_trajs)} successful trajectories to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Label and filter trajectories')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing synthetic trajectories')
    parser.add_argument('--labels-file', type=str,
                       help='Path to labels JSON file (for loading or saving)')
    parser.add_argument('--output-file', type=str,
                       help='Output file for labels (interactive mode)')
    parser.add_argument('--filter-success', action='store_true',
                       help='Filter and copy successful trajectories')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for successful trajectories')
    parser.add_argument('--auto-label', action='store_true',
                       help='Automatically label based on existing metadata')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    labeler = TrajectoryLabeler(input_dir)

    # Mode 1: Interactive labeling
    if not args.filter_success and not args.labels_file:
        labels = labeler.interactive_labeling()

        if args.output_file:
            labeler.save_labels(labels, Path(args.output_file))

    # Mode 2: Auto-label from metadata
    elif args.auto_label:
        print("Auto-labeling from metadata...")
        labels = {}
        for traj_dir in labeler.trajectory_dirs:
            metadata = labeler.load_trajectory_metadata(traj_dir)
            if metadata:
                labels[traj_dir.name] = metadata.get('success', False)

        if args.output_file:
            labeler.save_labels(labels, Path(args.output_file))

    # Mode 3: Filter successful trajectories
    elif args.filter_success:
        if not args.labels_file:
            print("Error: --labels-file required for filtering")
            return

        labels = labeler.load_labels(Path(args.labels_file))

        if not args.output_dir:
            print("Error: --output-dir required for filtering")
            return

        labeler.filter_successful_trajectories(labels, Path(args.output_dir))


if __name__ == '__main__':
    main()
