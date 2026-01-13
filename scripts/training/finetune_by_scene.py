#!/usr/bin/env python3
"""
按场景（building/location）微调Action Adapter

用法:
    # 列出所有可用场景
    python scripts/finetune_by_scene.py --list-scenes

    # 在特定场景上微调
    python scripts/finetune_by_scene.py --scene "Gates"
    python scripts/finetune_by_scene.py --scene "BAIR" --epochs 20
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
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'action_adapter'))
from train2 import Dynamics


class SceneDataset(Dataset):
    """Dataset for training on specific scene."""

    def __init__(
        self,
        data_root: str,
        scene_name: str,
        chunk_ids: list = [0],
        num_frames: int = 15,
        action_dim: int = 7,
        sample_interval: int = 4,
        split_ratio: float = 0.9,
        mode: str = 'train',
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.scene_name = scene_name
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.sample_interval = sample_interval
        self.mode = mode

        # Find all episodes for this scene
        self.episodes = self._find_scene_episodes(chunk_ids)

        if not self.episodes:
            raise ValueError(f"No episodes found for scene: {scene_name}")

        # Split train/val
        num_train = int(len(self.episodes) * split_ratio)
        if mode == 'train':
            self.episodes = self.episodes[:num_train]
        else:
            self.episodes = self.episodes[num_train:]

        # Create samples
        self.samples = self._create_samples()

        print(f"Scene '{scene_name}' {mode} set:")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Samples:  {len(self.samples)}")

    def _find_scene_episodes(self, chunk_ids):
        """Find all episodes belonging to this scene."""
        episodes = []

        for chunk_id in chunk_ids:
            chunk_dir = self.data_root / 'data' / f'chunk-{chunk_id:03d}'
            if not chunk_dir.exists():
                continue

            start_ep = chunk_id * 1000
            end_ep = (chunk_id + 1) * 1000

            for ep_id in range(start_ep, end_ep):
                parquet_file = chunk_dir / f'episode_{ep_id:06d}.parquet'
                if not parquet_file.exists():
                    continue

                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 0:
                        building = df.iloc[0].get('building', '')
                        if building == self.scene_name:
                            episodes.append({
                                'episode_id': ep_id,
                                'parquet_file': parquet_file,
                                'length': len(df)
                            })
                except Exception as e:
                    continue

        return episodes

    def _create_samples(self):
        """Create training samples from all episodes."""
        samples = []

        for ep_info in self.episodes:
            ep_id = ep_info['episode_id']
            ep_length = ep_info['length']

            for start_frame in range(0, ep_length - self.num_frames - 1, self.sample_interval):
                samples.append({
                    'episode_id': ep_id,
                    'parquet_file': ep_info['parquet_file'],
                    'start_frame': start_frame,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        parquet_file = sample['parquet_file']
        start_frame = sample['start_frame']

        try:
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
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


def list_available_scenes(data_root: str, chunk_ids: list = [0]):
    """List all available scenes in the dataset."""
    data_root = Path(data_root)
    scene_counter = Counter()

    print("扫描数据集中的场景...")

    for chunk_id in chunk_ids:
        chunk_dir = data_root / 'data' / f'chunk-{chunk_id:03d}'
        if not chunk_dir.exists():
            continue

        start_ep = chunk_id * 1000
        end_ep = (chunk_id + 1) * 1000

        for ep_id in range(start_ep, end_ep):
            parquet_file = chunk_dir / f'episode_{ep_id:06d}.parquet'
            if not parquet_file.exists():
                continue

            try:
                df = pd.read_parquet(parquet_file)
                if len(df) > 0:
                    building = df.iloc[0].get('building', 'unknown')
                    scene_counter[building] += 1
            except:
                continue

    print()
    print("=" * 80)
    print(f"可用场景列表 (chunk {chunk_ids})")
    print("=" * 80)
    print()

    for scene, count in scene_counter.most_common():
        print(f"  {scene:30s} : {count:4d} episodes")

    print()
    print(f"总计: {len(scene_counter)} 个不同场景")
    print("=" * 80)


def finetune_by_scene(
    data_root: str,
    scene_name: str,
    pretrained_path: str,
    output_path: str,
    chunk_ids: list = [0],
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-5,
    num_frames: int = 15,
    device: str = 'cuda',
):
    """Fine-tune on a specific scene."""

    print("=" * 80)
    print(f"场景微调: {scene_name}")
    print("=" * 80)
    print(f"数据路径:     {data_root}")
    print(f"Chunks:       {chunk_ids}")
    print(f"预训练模型:   {pretrained_path}")
    print(f"输出路径:     {output_path}")
    print(f"训练轮数:     {num_epochs}")
    print(f"批大小:       {batch_size}")
    print(f"学习率:       {learning_rate}")
    print(f"设备:         {device}")
    print("=" * 80)
    print()

    # Create datasets
    train_dataset = SceneDataset(
        data_root=data_root,
        scene_name=scene_name,
        chunk_ids=chunk_ids,
        num_frames=num_frames,
        mode='train',
    )

    val_dataset = SceneDataset(
        data_root=data_root,
        scene_name=scene_name,
        chunk_ids=chunk_ids,
        num_frames=num_frames,
        mode='val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Load model
    print(f"加载预训练模型...")
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

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

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

            loss = model(joint, joint_vel, joint_delta, training=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = train_loss / train_steps
        train_loss_history.append(avg_train_loss)

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

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_val_loss = val_loss / val_steps
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")

        # Save checkpoint
        ckpt_path = Path(output_path).parent / f"{Path(output_path).stem}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ 保存最佳模型 (val_loss={best_val_loss:.6f})")

        print("-" * 80)

    print()
    print("=" * 80)
    print("训练完成!")
    print(f"最佳验证loss: {best_val_loss:.6f}")
    print(f"模型保存至:   {output_path}")
    print("=" * 80)

    # Save training history
    history_path = Path(output_path).parent / f"{Path(output_path).stem}_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'scene_name': scene_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }, f, indent=2)
    print(f"训练历史保存至: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune by scene')
    parser.add_argument('--list-scenes', action='store_true',
                       help='List all available scenes')
    parser.add_argument('--scene', type=str,
                       help='Scene name to train on (e.g., "Gates", "BAIR")')
    parser.add_argument('--data-root', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data',
                       help='DROID data root directory')
    parser.add_argument('--chunks', type=int, nargs='+', default=[0],
                       help='Chunk IDs to use (e.g., 0 1 2)')
    parser.add_argument('--pretrained-path', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/models/action_adapter/model2_15_9.pth',
                       help='Path to pretrained model')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/train_adapter/by_scene',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--num-frames', type=int, default=15,
                       help='Number of frames to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    if args.list_scenes:
        list_available_scenes(args.data_root, args.chunks)
        return

    if not args.scene:
        print("错误: 请指定场景名称 (--scene) 或使用 --list-scenes 查看可用场景")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize scene name for filename
    scene_filename = args.scene.replace(' ', '_').replace('/', '_')
    output_path = output_dir / f"adapter_scene_{scene_filename}_best.pth"

    finetune_by_scene(
        data_root=args.data_root,
        scene_name=args.scene,
        pretrained_path=args.pretrained_path,
        output_path=str(output_path),
        chunk_ids=args.chunks,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_frames=args.num_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
