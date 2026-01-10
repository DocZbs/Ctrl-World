#!/usr/bin/env python3
"""
Train Action Adapter (Dynamics Model)

This script trains a dynamics model that predicts joint position changes
from current joint positions and joint velocities.

Usage:
    python train_adapter.py --data_path dataset_example/droid_subset \
                           --meta_path dataset_meta_info/droid_subset \
                           --output_dir models/action_adapter/checkpoints \
                           --num_epochs 10
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import datetime

from train2 import Dynamics


class DROIDActionDataset(Dataset):
    """Dataset for training action adapter on DROID data."""

    def __init__(
        self,
        data_root: str,
        meta_info_path: str,
        mode: str = 'train',
        num_frames: int = 15,
        action_dim: int = 7,
    ):
        """
        Args:
            data_root: Root directory of DROID dataset
            meta_info_path: Path to dataset metadata (train_sample.json, val_sample.json, stat.json)
            mode: 'train' or 'val'
            num_frames: Number of frames to predict
            action_dim: Action dimension (7 for joint velocities)
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.meta_info_path = Path(meta_info_path)
        self.mode = mode
        self.num_frames = num_frames
        self.action_dim = action_dim

        # Load samples
        sample_file = self.meta_info_path / f'{mode}_sample.json'
        with open(sample_file, 'r') as f:
            self.samples = json.load(f)

        # Load statistics for normalization
        stat_file = self.meta_info_path / 'stat.json'
        with open(stat_file, 'r') as f:
            stats = json.load(f)

        # Extract joint velocity and delta statistics
        self.joint_vel_min = np.array(stats.get('joint_vel_01', [-1.0] * 7))
        self.joint_vel_max = np.array(stats.get('joint_vel_99', [1.0] * 7))
        self.joint_delta_min = np.array(stats.get('joint_delta_01', [-0.5] * 7))
        self.joint_delta_max = np.array(stats.get('joint_delta_99', [0.5] * 7))

        print(f"Loaded {len(self.samples)} samples for {mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - joints: (1, 7) current joint position
                - joint_vels: (num_frames, 7) joint velocities
                - joints_delta: (num_frames, 7) joint position deltas
        """
        sample = self.samples[idx]

        try:
            # Load annotation
            ann_file = self.data_root / sample['ann_file']
            with open(ann_file, 'r') as f:
                label = json.load(f)

            # Get episode ID and chunk
            episode_id = int(sample['episode_id'])
            chunk_id = episode_id // 1000

            # Load parquet file
            parquet_file = self.data_root / 'data' / f'chunk-{chunk_id:03d}' / f'episode_{episode_id:06d}.parquet'
            df = pd.read_parquet(parquet_file)

            # Get frame indices
            start_idx = int(sample['frame_ids'][0])
            frame_indices = list(range(start_idx, start_idx + self.num_frames + 1))
            frame_indices = [min(idx, len(df) - 1) for idx in frame_indices]

            # Extract joint positions and velocities
            joints = []
            joint_vels = []
            for idx in frame_indices:
                joint_pos = df.iloc[idx]['observation.state.joint_position'][:7]
                joint_vel = df.iloc[idx]['action.joint_velocity'][:7]
                joints.append(joint_pos)
                joint_vels.append(joint_vel)

            joints = np.array(joints, dtype=np.float32)
            joint_vels = np.array(joint_vels, dtype=np.float32)

            # Compute deltas
            current_joint = joints[0:1]  # (1, 7)
            joint_vels = joint_vels[:-1]  # (num_frames, 7)
            joints_delta = joints[1:] - joints[0]  # (num_frames, 7)

            return {
                'joints': current_joint,
                'joint_vels': joint_vels,
                'joints_delta': joints_delta,
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a random valid sample
            return self.__getitem__(np.random.randint(0, len(self)))


def train_adapter(
    data_root: str,
    meta_info_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    num_workers: int = 8,
    num_frames: int = 15,
    device: str = 'cuda',
):
    """Train action adapter model."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("Creating datasets...")
    train_dataset = DROIDActionDataset(
        data_root=data_root,
        meta_info_path=meta_info_path,
        mode='train',
        num_frames=num_frames,
    )

    val_dataset = DROIDActionDataset(
        data_root=data_root,
        meta_info_path=meta_info_path,
        mode='val',
        num_frames=num_frames,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = Dynamics(
        action_dim=7,
        action_num=num_frames,
        hidden_size=512,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("-" * 70)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            joint = batch['joints'].to(device)
            joint_vel = batch['joint_vels'].to(device)
            joint_delta = batch['joints_delta'].to(device)

            # Forward pass
            loss = model(joint, joint_vel, joint_delta, training=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                joint = batch['joints'].to(device)
                joint_vel = batch['joint_vels'].to(device)
                joint_delta = batch['joints_delta'].to(device)

                loss = model(joint, joint_vel, joint_delta, training=True)

                val_loss += loss.item()
                val_steps += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_steps

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f'adapter_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output_dir / 'adapter_best.pth'
            torch.save(model.state_dict(), best_path)
            print(f"  Best model updated: {best_path}")

        print("-" * 70)

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Action Adapter')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to DROID dataset (e.g., dataset_example/droid_subset)')
    parser.add_argument('--meta_path', type=str, required=True,
                       help='Path to metadata (e.g., dataset_meta_info/droid_subset)')
    parser.add_argument('--output_dir', type=str, default='models/action_adapter/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--num_frames', type=int, default=15,
                       help='Number of frames to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    train_adapter(
        data_root=args.data_path,
        meta_info_path=args.meta_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
