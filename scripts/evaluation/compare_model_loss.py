#!/usr/bin/env python3
"""
Compare loss between original and fine-tuned models on chunk-001 data.

This script evaluates both the pre-trained and fine-tuned action adapter models
on chunk-001 episodes to measure the performance improvement from fine-tuning.
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
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'action_adapter'))
from train2 import Dynamics


class DROIDChunk1Dataset(Dataset):
    """Dataset for evaluating on chunk-001 data (episodes 1000-1999)."""

    def __init__(
        self,
        data_root: str,
        chunk_id: int = 1,
        num_frames: int = 15,
        action_dim: int = 7,
        sample_interval: int = 4,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.chunk_id = chunk_id
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.sample_interval = sample_interval

        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} samples from chunk-{chunk_id:03d}")

    def _create_samples(self) -> List[Dict]:
        """Create sample indices for all episodes in the chunk."""
        samples = []

        chunk_dir = self.data_root / 'data' / f'chunk-{self.chunk_id:03d}'
        if not chunk_dir.exists():
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

        start_episode = self.chunk_id * 1000
        end_episode = (self.chunk_id + 1) * 1000

        for episode_id in range(start_episode, end_episode):
            parquet_file = chunk_dir / f'episode_{episode_id:06d}.parquet'

            if not parquet_file.exists():
                continue

            df = pd.read_parquet(parquet_file)
            episode_length = len(df)

            for start_frame in range(0, episode_length - self.num_frames - 1, self.sample_interval):
                samples.append({
                    'episode_id': episode_id,
                    'start_frame': start_frame,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        episode_id = sample['episode_id']
        start_frame = sample['start_frame']

        try:
            chunk_dir = self.data_root / 'data' / f'chunk-{self.chunk_id:03d}'
            parquet_file = chunk_dir / f'episode_{episode_id:06d}.parquet'
            df = pd.read_parquet(parquet_file)

            frame_indices = list(range(start_frame, start_frame + self.num_frames + 1))
            frame_indices = [min(idx, len(df) - 1) for idx in frame_indices]

            joints = []
            joint_vels = []
            for idx in frame_indices:
                joint_pos = df.iloc[idx]['observation.state.joint_position'][:7]
                joint_vel = df.iloc[idx]['action.joint_velocity'][:7]
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
                'episode_id': episode_id,
                'start_frame': start_frame,
            }

        except Exception as e:
            print(f"Error loading sample {idx} (episode {episode_id}): {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_name: str,
) -> Dict[str, float]:
    """Evaluate model and return loss statistics."""

    model.eval()
    losses = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {model_name}")
        for batch in pbar:
            joint = batch['joints'].to(device)
            joint_vel = batch['joint_vels'].to(device)
            joint_delta = batch['joints_delta'].to(device)

            loss = model(joint, joint_vel, joint_delta, training=True)
            losses.append(loss.item())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    losses = np.array(losses)

    return {
        'mean': float(np.mean(losses)),
        'std': float(np.std(losses)),
        'min': float(np.min(losses)),
        'max': float(np.max(losses)),
        'median': float(np.median(losses)),
        'q25': float(np.percentile(losses, 25)),
        'q75': float(np.percentile(losses, 75)),
    }


def compare_models(
    data_root: str,
    original_model_path: str,
    finetuned_model_path: str,
    chunk_id: int = 1,
    batch_size: int = 128,
    num_workers: int = 8,
    num_frames: int = 15,
    device: str = 'cuda',
):
    """Compare original and fine-tuned models on chunk data."""

    print("=" * 80)
    print("Model Comparison on Chunk-001 Data")
    print("=" * 80)
    print(f"Original model:    {original_model_path}")
    print(f"Fine-tuned model:  {finetuned_model_path}")
    print(f"Test chunk:        chunk-{chunk_id:03d}")
    print(f"Device:            {device}")
    print("=" * 80)

    dataset = DROIDChunk1Dataset(
        data_root=data_root,
        chunk_id=chunk_id,
        num_frames=num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nLoaded {len(dataset)} samples from chunk-{chunk_id:03d}")
    print(f"Batch size: {batch_size}, Number of batches: {len(dataloader)}")
    print()

    original_model = Dynamics(
        action_dim=7,
        action_num=num_frames,
        hidden_size=512,
    ).to(device)

    finetuned_model = Dynamics(
        action_dim=7,
        action_num=num_frames,
        hidden_size=512,
    ).to(device)

    print("Loading original model...")
    original_ckpt = torch.load(original_model_path, map_location=device)
    original_model.load_state_dict(original_ckpt)
    print("✓ Original model loaded")

    print("Loading fine-tuned model...")
    finetuned_ckpt = torch.load(finetuned_model_path, map_location=device)
    finetuned_model.load_state_dict(finetuned_ckpt)
    print("✓ Fine-tuned model loaded")
    print()

    print("Evaluating original model...")
    original_stats = evaluate_model(original_model, dataloader, device, "Original")
    print()

    print("Evaluating fine-tuned model...")
    finetuned_stats = evaluate_model(finetuned_model, dataloader, device, "Fine-tuned")
    print()

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Metric':<15} {'Original':<15} {'Fine-tuned':<15} {'Difference':<15} {'Change %':<15}")
    print("-" * 80)

    metrics = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    for metric in metrics:
        orig_val = original_stats[metric]
        ft_val = finetuned_stats[metric]
        diff = ft_val - orig_val
        if orig_val != 0:
            pct_change = (diff / orig_val) * 100
        else:
            pct_change = 0.0

        print(f"{metric:<15} {orig_val:<15.6f} {ft_val:<15.6f} {diff:<15.6f} {pct_change:<15.2f}%")

    print()
    print("=" * 80)

    improvement = original_stats['mean'] - finetuned_stats['mean']
    improvement_pct = (improvement / original_stats['mean']) * 100 if original_stats['mean'] != 0 else 0

    print(f"\n{'IMPROVEMENT ANALYSIS':^80}")
    print("=" * 80)
    print(f"Mean Loss Improvement:     {improvement:.6f}")
    print(f"Relative Improvement:      {improvement_pct:.2f}%")

    if improvement > 0:
        print(f"\n✓ Fine-tuned model performs BETTER (lower loss)")
    elif improvement < 0:
        print(f"\n✗ Fine-tuned model performs WORSE (higher loss)")
    else:
        print(f"\n= Models perform EQUALLY")

    print("=" * 80)

    results = {
        'original': original_stats,
        'finetuned': finetuned_stats,
        'improvement': {
            'absolute': improvement,
            'relative_pct': improvement_pct,
        }
    }

    output_file = Path('model_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare model performance on chunk-001')
    parser.add_argument('--data_root', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data',
                       help='Path to DROID dataset root')
    parser.add_argument('--original_model', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/models/action_adapter/model2_15_9.pth',
                       help='Path to original pre-trained model')
    parser.add_argument('--finetuned_model', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/train_adapter/checkpoints/adapter_finetuned_best.pth',
                       help='Path to fine-tuned model')
    parser.add_argument('--chunk_id', type=int, default=1,
                       help='Chunk ID to test on (default: 1 for episodes 1000-1999)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--num_frames', type=int, default=15,
                       help='Number of frames to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    compare_models(
        data_root=args.data_root,
        original_model_path=args.original_model,
        finetuned_model_path=args.finetuned_model,
        chunk_id=args.chunk_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
