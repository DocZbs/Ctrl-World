#!/usr/bin/env python3
"""
Fine-tune pi05 on synthetic trajectories

完整流程:
1. 转换合成数据到 LeRobot 格式
2. 微调 pi05 模型

用法:
    # Step 1: 转换数据
    python scripts/finetune_pi05_synthetic.py convert \
        --synthetic-dir synthetic_data/pickplace \
        --output-repo-id username/synthetic_pickplace

    # Step 2: 微调模型
    python scripts/finetune_pi05_synthetic.py train \
        --repo-id username/synthetic_pickplace \
        --exp-name pickplace_finetune \
        --num-gpus 2

    # 或者一步完成
    python scripts/finetune_pi05_synthetic.py all \
        --synthetic-dir synthetic_data/pickplace \
        --output-repo-id username/synthetic_pickplace \
        --exp-name pickplace_finetune
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image using PIL"""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def load_video_frames(video_path: Path) -> list:
    """Load all frames from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def compute_joint_velocity_actions(joint_positions: list) -> np.ndarray:
    """
    Compute joint velocity actions from joint positions

    pi05-droid 使用 joint velocity actions:
    - 输入: joint positions (N, 8) - 7 joints + 1 gripper
    - 输出: actions (N-1, 32) - 7 joint velocities + 1 gripper position + 24 zeros (padding)

    Args:
        joint_positions: List of joint positions (N, 8)

    Returns:
        Joint velocity actions (N-1, 32) with padding
    """
    joint_positions = np.array(joint_positions)

    # Compute joint velocities (difference between consecutive positions)
    joint_velocities = np.diff(joint_positions[:, :7], axis=0)  # (N-1, 7)

    # Gripper actions are positions (not velocities)
    gripper_positions = joint_positions[1:, 7:]  # (N-1, 1)

    # Concatenate: [joint_velocity (7), gripper_position (1)]
    actions_8d = np.concatenate([joint_velocities, gripper_positions], axis=1)  # (N-1, 8)

    # Pad to 32 dimensions to match pi05_droid action_dim
    # pi05_droid expects 32-dim actions, but only first 8 are used
    actions_32d = np.pad(actions_8d, ((0, 0), (0, 24)), mode='constant', constant_values=0)  # (N-1, 32)

    return actions_32d.astype(np.float32)


def convert_to_lerobot(
    synthetic_dir: Path,
    output_repo_id: str,
    fps: int = 10,
    push_to_hub: bool = False,
):
    """
    Convert synthetic trajectories to LeRobot format

    Args:
        synthetic_dir: Directory containing synthetic data
        output_repo_id: Repository ID for the output dataset
        fps: Frames per second
        push_to_hub: Whether to push to Hugging Face Hub
    """
    print(f"\n{'='*80}")
    print("Step 1: Converting synthetic data to LeRobot format")
    print(f"{'='*80}\n")

    synthetic_dir = Path(synthetic_dir)

    # Clean up existing dataset
    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset with DROID format
    print(f"Creating LeRobot dataset: {output_repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="panda",
        fps=fps,
        features={
            # Match DROID naming conventions for pi05
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (32,),  # 32D actions (8 actual + 24 padding) to match pi05_droid
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Find all annotation files
    annotation_dir = synthetic_dir / "annotation" / "synthetic"
    videos_dir = synthetic_dir / "videos" / "synthetic"

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    annotation_files = sorted(annotation_dir.glob("*.json"))
    print(f"Found {len(annotation_files)} episodes")

    # Convert each episode
    for anno_file in tqdm(annotation_files, desc="Converting episodes"):
        with open(anno_file) as f:
            anno = json.load(f)

        episode_id = anno["episode_id"]
        task_instruction = anno["texts"][0]

        # Load videos (3 camera views)
        video_dir = videos_dir / episode_id
        frames_view0 = load_video_frames(video_dir / "0.mp4")  # exterior_image_1_left
        frames_view1 = load_video_frames(video_dir / "1.mp4")  # exterior_image_2_left
        frames_view2 = load_video_frames(video_dir / "2.mp4")  # wrist_image_left

        # Get joint positions and compute actions
        joint_positions = np.array(anno["joints"], dtype=np.float32)  # (N, 8)
        actions = compute_joint_velocity_actions(joint_positions)  # (N-1, 8)

        # Verify video frame counts match
        num_frames = len(frames_view0)
        assert len(frames_view1) == num_frames
        assert len(frames_view2) == num_frames

        # Subsample video frames to match joint data frequency
        # The joint data is sampled at a lower rate than video frames
        num_joint_samples = len(joint_positions)
        frame_indices = np.linspace(0, num_frames - 1, num_joint_samples, dtype=int)

        frames_view0_sampled = [frames_view0[i] for i in frame_indices]
        frames_view1_sampled = [frames_view1[i] for i in frame_indices]
        frames_view2_sampled = [frames_view2[i] for i in frame_indices]

        # Now lengths should match
        assert len(frames_view0_sampled) == num_joint_samples
        assert len(actions) == num_joint_samples - 1

        # Add frames to dataset
        for i in range(num_joint_samples - 1):
            dataset.add_frame(
                {
                    "exterior_image_1_left": resize_image(frames_view0_sampled[i], (320, 180)),
                    "exterior_image_2_left": resize_image(frames_view1_sampled[i], (320, 180)),
                    "wrist_image_left": resize_image(frames_view2_sampled[i], (320, 180)),
                    "joint_position": joint_positions[i, :7],
                    "gripper_position": joint_positions[i, 7:8],
                    "actions": actions[i],
                    "task": task_instruction,
                }
            )

        dataset.save_episode()

    # Optionally push to Hub
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {output_repo_id}")
        dataset.push_to_hub()

    print(f"\n✓ Dataset conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Episodes: {len(annotation_files)}")
    print(f"  Format: LeRobot (compatible with pi05)")

    return output_path


def train_pi05(
    repo_id: str,
    exp_name: str,
    num_gpus: int = 1,
    num_train_steps: int = 20000,
    batch_size: int = 32,
):
    """
    Train pi05 on the converted dataset

    Args:
        repo_id: LeRobot dataset repository ID
        exp_name: Experiment name
        num_gpus: Number of GPUs to use
        num_train_steps: Number of training steps
        batch_size: Batch size
    """
    print(f"\n{'='*80}")
    print("Step 2: Fine-tuning pi05 model")
    print(f"{'='*80}\n")

    # Use the existing pi05_droid_finetune config as base
    # We'll override the dataset repo_id via command line
    print(f"Using pi05_droid_finetune config as base")
    print(f"  Dataset: {repo_id}")
    print(f"  Experiment: {exp_name}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Training steps: {num_train_steps}")
    print(f"  Batch size: {batch_size}")

    # Build training command using pi05_droid_finetune config
    # Use the OpenPI venv Python to ensure correct transformers version
    python_exe = "/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/bin/python"

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={num_gpus}",
            "openpi/scripts/train_pytorch.py",
            "pi05_droid_finetune",
            "--exp-name", exp_name,
            "--data.repo-id", repo_id,
            "--num-train-steps", str(num_train_steps),
            "--batch-size", str(batch_size),
            "--pytorch-training-precision", "float32",
        ]
    else:
        cmd = [
            python_exe,
            "openpi/scripts/train_pytorch.py",
            "pi05_droid_finetune",
            "--exp-name", exp_name,
            "--data.repo-id", repo_id,
            "--num-train-steps", str(num_train_steps),
            "--batch-size", str(batch_size),
            "--pytorch-training-precision", "float32",
        ]

    print(f"\nRunning training command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run training
    subprocess.run(cmd, check=True)

    print(f"\n✓ Training complete!")
    print(f"  Checkpoints saved to: openpi/checkpoints/{exp_name}/")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pi05 on synthetic data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert synthetic data to LeRobot format")
    convert_parser.add_argument("--synthetic-dir", type=str, required=True,
                               help="Directory containing synthetic data")
    convert_parser.add_argument("--output-repo-id", type=str, required=True,
                               help="Output repository ID (e.g., username/synthetic_pickplace)")
    convert_parser.add_argument("--fps", type=int, default=10,
                               help="Frames per second (default: 10)")
    convert_parser.add_argument("--push-to-hub", action="store_true",
                               help="Push dataset to Hugging Face Hub")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train pi05 on converted dataset")
    train_parser.add_argument("--repo-id", type=str, required=True,
                             help="LeRobot dataset repository ID")
    train_parser.add_argument("--exp-name", type=str, required=True,
                             help="Experiment name")
    train_parser.add_argument("--num-gpus", type=int, default=1,
                             help="Number of GPUs (default: 1)")
    train_parser.add_argument("--num-train-steps", type=int, default=20000,
                             help="Number of training steps (default: 20000)")
    train_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size (default: 32)")

    # All command (convert + train)
    all_parser = subparsers.add_parser("all", help="Convert and train in one step")
    all_parser.add_argument("--synthetic-dir", type=str, required=True,
                           help="Directory containing synthetic data")
    all_parser.add_argument("--output-repo-id", type=str, required=True,
                           help="Output repository ID")
    all_parser.add_argument("--exp-name", type=str, required=True,
                           help="Experiment name")
    all_parser.add_argument("--fps", type=int, default=10,
                           help="Frames per second (default: 10)")
    all_parser.add_argument("--num-gpus", type=int, default=1,
                           help="Number of GPUs (default: 1)")
    all_parser.add_argument("--num-train-steps", type=int, default=20000,
                           help="Number of training steps (default: 20000)")
    all_parser.add_argument("--batch-size", type=int, default=32,
                           help="Batch size (default: 32)")
    all_parser.add_argument("--push-to-hub", action="store_true",
                           help="Push dataset to Hugging Face Hub")

    args = parser.parse_args()

    if args.command == "convert":
        convert_to_lerobot(
            Path(args.synthetic_dir),
            args.output_repo_id,
            args.fps,
            args.push_to_hub,
        )

    elif args.command == "train":
        train_pi05(
            args.repo_id,
            args.exp_name,
            args.num_gpus,
            args.num_train_steps,
            args.batch_size,
        )

    elif args.command == "all":
        # Convert
        convert_to_lerobot(
            Path(args.synthetic_dir),
            args.output_repo_id,
            args.fps,
            args.push_to_hub,
        )

        # Train
        train_pi05(
            args.output_repo_id,
            args.exp_name,
            args.num_gpus,
            args.num_train_steps,
            args.batch_size,
        )

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*80}")
    print("All done! 🎉")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
