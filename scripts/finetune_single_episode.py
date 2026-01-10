#!/usr/bin/env python3
"""
在单个DROID场景（episode）上微调Action Adapter

用法:
    python scripts/finetune_single_episode.py --episode-id 0
    python scripts/finetune_single_episode.py --episode-id 123 --epochs 20
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'action_adapter'))
from train2 import Dynamics


class SingleEpisodeDataset(Dataset):
    """Dataset for single episode training."""

    def __init__(
        self,
        data_root: str,
        episode_id: int,
        num_frames: int = 15,
        action_dim: int = 7,
        sample_interval: int = 4,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.episode_id = episode_id
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.sample_interval = sample_interval

        chunk_id = episode_id // 1000
        self.parquet_file = self.data_root / 'data' / f'chunk-{chunk_id:03d}' / f'episode_{episode_id:06d}.parquet'

        if not self.parquet_file.exists():
            raise FileNotFoundError(f"Episode file not found: {self.parquet_file}")

        self.df = pd.read_parquet(self.parquet_file)
        self.samples = self._create_samples()

        print(f"Loaded episode {episode_id} with {len(self.df)} frames")
        print(f"Created {len(self.samples)} training samples")

    def _create_samples(self):
        """Create sample indices from the episode."""
        samples = []
        episode_length = len(self.df)

        for start_frame in range(0, episode_length - self.num_frames - 1, self.sample_interval):
            samples.append({'start_frame': start_frame})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start_frame = sample['start_frame']

        try:
            frame_indices = list(range(start_frame, start_frame + self.num_frames + 1))
            frame_indices = [min(idx, len(self.df) - 1) for idx in frame_indices]

            joints = []
            joint_vels = []
            for idx in frame_indices:
                joint_pos = self.df.iloc[idx]['observation.state.joint_position'][:7]
                joint_vel = self.df.iloc[idx]['action.joint_velocity'][:7]
                joints.append(joint_pos)
                joint_vels.append(joint_vel)

            joints = np.array(joints, dtype=np.float32)
            joint_vels = np.array(joint_vels, dtype=np.float32)

            current_joint = joints[0:1]
            joint_vels = joint_vels[:-1]
            joints_delta = joints[1:] - joints[0]

            return {
                'joints': current_joint,
                'joint_vels': joint_vels,
                'joints_delta': joints_delta,
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


def finetune_single_episode(
    data_root: str,
    episode_id: int,
    pretrained_path: str,
    output_path: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_frames: int = 15,
    device: str = 'cuda',
):
    """Fine-tune on a single episode."""

    print("=" * 80)
    print(f"单场景微调: Episode {episode_id}")
    print("=" * 80)
    print(f"数据路径:     {data_root}")
    print(f"预训练模型:   {pretrained_path}")
    print(f"输出路径:     {output_path}")
    print(f"训练轮数:     {num_epochs}")
    print(f"批大小:       {batch_size}")
    print(f"学习率:       {learning_rate}")
    print(f"设备:         {device}")
    print("=" * 80)
    print()

    dataset = SingleEpisodeDataset(
        data_root=data_root,
        episode_id=episode_id,
        num_frames=num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"加载预训练模型: {pretrained_path}")
    model = Dynamics(
        action_dim=7,
        action_num=num_frames,
        hidden_size=512,
    ).to(device)

    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("✓ 预训练模型加载成功")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"开始训练 {num_epochs} 轮...")
    print("-" * 80)

    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            joint = batch['joints'].to(device)
            joint_vel = batch['joint_vels'].to(device)
            joint_delta = batch['joints_delta'].to(device)

            loss = model(joint, joint_vel, joint_delta, training=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ 保存最佳模型 (loss={best_loss:.6f})")

    print()
    print("=" * 80)
    print("训练完成!")
    print(f"最佳loss:     {best_loss:.6f}")
    print(f"最终loss:     {loss_history[-1]:.6f}")
    print(f"模型保存至:   {output_path}")
    print("=" * 80)

    # Save training history
    history_path = Path(output_path).parent / f"episode_{episode_id:06d}_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'episode_id': episode_id,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_loss': best_loss,
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
        }, f, indent=2)
    print(f"训练历史保存至: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune on single episode')
    parser.add_argument('--episode-id', type=int, required=True,
                       help='Episode ID to train on (e.g., 0, 123, 999)')
    parser.add_argument('--data-root', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data',
                       help='DROID data root directory')
    parser.add_argument('--pretrained-path', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/models/action_adapter/model2_15_9.pth',
                       help='Path to pretrained model')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/train_adapter/single_episode',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-frames', type=int, default=15,
                       help='Number of frames to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"adapter_episode_{args.episode_id:06d}.pth"

    finetune_single_episode(
        data_root=args.data_root,
        episode_id=args.episode_id,
        pretrained_path=args.pretrained_path,
        output_path=str(output_path),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_frames=args.num_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
