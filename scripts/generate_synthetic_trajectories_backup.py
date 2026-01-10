#!/usr/bin/env python3
"""
生成合成轨迹用于π0.5微调

实现论文中的 policy-in-the-loop imagination rollout:
1. 从初始观测开始，策略输出action chunk
2. 世界模型根据action生成下一帧
3. 循环直到完成整条轨迹
4. 保存成功的轨迹用于微调

用法:
    python scripts/generate_synthetic_trajectories.py \
        --task-type pickplace \
        --num-rollouts 400 \
        --output-dir synthetic_data/pickplace
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import mediapy
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from openpi.training import config as config_pi
from openpi.policies import policy_config


@dataclass
class SyntheticTrajectory:
    """合成轨迹数据结构"""
    trajectory_id: str
    task_instruction: str
    initial_state: List[float]
    states: List[List[float]]  # Cartesian states
    actions: List[List[float]]  # Joint velocity actions
    images: List[np.ndarray]  # Generated images
    success: bool = False
    metadata: Dict[str, Any] = None


class SyntheticDataGenerator:
    """合成数据生成器"""

    def __init__(
        self,
        wm_ckpt: str,
        svd_model_path: str,
        clip_model_path: str,
        pi_ckpt: str,
        policy_type: str = "pi05",
        action_adapter_path: str = None,
        data_stat_path: str = "dataset_meta_info/droid/stat.json",
        device: str = "cuda:0",
    ):
        self.device = device

        # Load world model
        print("Loading world model...")
        args = type('Args', (), {
            'val_model_path': wm_ckpt,
            'svd_model_path': svd_model_path,
            'clip_model_path': clip_model_path,
            'dtype': torch.float16,
            'data_stat_path': data_stat_path,
            'action_adapter': action_adapter_path,
        })()

        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(wm_ckpt))
        self.model.to(device).to(torch.float16)
        self.model.eval()

        # Load normalization stats
        with open(data_stat_path, 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

        # Load action adapter
        if action_adapter_path:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(device)
            self.dynamics_model.load_state_dict(torch.load(action_adapter_path, map_location=device))
        else:
            self.dynamics_model = None

        # Load policy
        print(f"Loading {policy_type} policy...")
        if 'pi05' in policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in policy_type:
            config = config_pi.get_config("pi0fast_droid")
        elif 'pi0' in policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        self.policy = policy_config.create_trained_policy(config, pi_ckpt)
        print("✓ Models loaded successfully")

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1]"""
        return 2 * (state - self.state_p01) / (self.state_p99 - self.state_p01 + 1e-8) - 1

    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state from [-1, 1]"""
        return (normalized_state + 1) / 2 * (self.state_p99 - self.state_p01 + 1e-8) + self.state_p01

    def generate_trajectory(
        self,
        initial_observation: Dict[str, Any],
        instruction: str,
        max_steps: int = 100,
        action_horizon: int = 16,
    ) -> SyntheticTrajectory:
        """
        生成一条合成轨迹

        Args:
            initial_observation: 初始观测（包含images和state）
            instruction: 任务指令
            max_steps: 最大步数
            action_horizon: 动作预测长度

        Returns:
            SyntheticTrajectory对象
        """
        trajectory_id = f"syn_{np.random.randint(1000000):06d}"

        states = [initial_observation['state']]
        actions = []
        images = [initial_observation['image']]

        current_obs = initial_observation
        current_state = initial_observation['state']

        for step in range(max_steps):
            # 1. Policy predicts action chunk
            policy_input = {
                'image': current_obs['image'],
                'instruction': instruction,
                'state': current_state,
            }

            # Get action from policy
            action_chunk = self.policy.predict(policy_input)  # Shape: (action_horizon, action_dim)

            # 2. Use action adapter if available
            if self.dynamics_model is not None:
                # Convert joint velocity to joint position delta
                current_joint = torch.tensor(current_state[:7], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                joint_vel = torch.tensor(action_chunk[:15, :7], dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    joint_delta = self.dynamics_model(current_joint, joint_vel, training=False)

                # Update state
                new_joint = current_joint.squeeze() + joint_delta.squeeze()[-1]
                current_state[:7] = new_joint.cpu().numpy()

            # 3. World model generates next frame
            # Normalize state for world model
            normalized_state = self.normalize_state(current_state)

            # Generate next observation using world model
            with torch.no_grad():
                next_image = self.model.generate_next_frame(
                    current_image=current_obs['image'],
                    action=action_chunk[0],  # Use first action in chunk
                    state=normalized_state,
                )

            # 4. Update trajectory
            states.append(current_state.copy())
            actions.append(action_chunk[0].tolist())
            images.append(next_image)

            # 5. Update current observation
            current_obs = {
                'image': next_image,
                'state': current_state,
            }

            # 6. Check termination (simplified)
            # In practice, you might want more sophisticated termination conditions
            if step >= max_steps - 1:
                break

        return SyntheticTrajectory(
            trajectory_id=trajectory_id,
            task_instruction=instruction,
            initial_state=initial_observation['state'].tolist(),
            states=[s.tolist() for s in states],
            actions=actions,
            images=images,
            success=False,  # Will be labeled manually
            metadata={
                'num_steps': len(actions),
                'max_steps': max_steps,
            }
        )

    def generate_diverse_trajectories(
        self,
        base_observation: Dict[str, Any],
        base_instruction: str,
        num_rollouts: int = 400,
        instruction_variants: List[str] = None,
        state_perturbation_std: float = 0.05,
    ) -> List[SyntheticTrajectory]:
        """
        生成多样化的轨迹

        实现论文中的两种多样性策略:
        1. Instruction rephrase: 改写指令
        2. Reset init state: 扰动初始状态

        Args:
            base_observation: 基础初始观测
            base_instruction: 基础任务指令
            num_rollouts: 生成轨迹数量
            instruction_variants: 指令变体列表
            state_perturbation_std: 状态扰动标准差

        Returns:
            轨迹列表
        """
        trajectories = []

        if instruction_variants is None:
            instruction_variants = [base_instruction]

        print(f"Generating {num_rollouts} diverse trajectories...")

        for i in tqdm(range(num_rollouts)):
            # 1. Select instruction variant
            instruction = instruction_variants[i % len(instruction_variants)]

            # 2. Perturb initial state
            perturbed_state = base_observation['state'].copy()
            # Add Gaussian noise to joint positions
            perturbed_state[:7] += np.random.normal(0, state_perturbation_std, 7)

            perturbed_obs = {
                'image': base_observation['image'],
                'state': perturbed_state,
            }

            # 3. Generate trajectory
            try:
                trajectory = self.generate_trajectory(
                    initial_observation=perturbed_obs,
                    instruction=instruction,
                )
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                continue

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        output_dir: Path,
    ):
        """保存轨迹到磁盘"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for traj in tqdm(trajectories, desc="Saving trajectories"):
            traj_dir = output_dir / traj.trajectory_id
            traj_dir.mkdir(exist_ok=True)

            # Save metadata
            metadata = {
                'trajectory_id': traj.trajectory_id,
                'task_instruction': traj.task_instruction,
                'initial_state': traj.initial_state,
                'states': traj.states,
                'actions': traj.actions,
                'success': traj.success,
                'metadata': traj.metadata,
            }

            with open(traj_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save images as video
            if traj.images:
                video_path = traj_dir / 'video.mp4'
                mediapy.write_video(str(video_path), traj.images, fps=10)

            # Save images individually
            images_dir = traj_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(traj.images):
                mediapy.write_image(str(images_dir / f'{i:04d}.png'), img)

        print(f"✓ Saved {len(trajectories)} trajectories to {output_dir}")


def load_initial_observation(annotation_path: str, dataset_root: str) -> Dict[str, Any]:
    """从annotation文件加载初始观测"""
    with open(annotation_path) as f:
        anno = json.load(f)

    # Load first frame from video
    video_path = Path(dataset_root) / anno['videos'][0]['video_path']
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read video: {video_path}")

    # Get initial state
    initial_state = np.array(anno['states'][0], dtype=np.float32)

    return {
        'image': frame,
        'state': initial_state,
        'instruction': anno['texts'][0],
    }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories')
    parser.add_argument('--annotation-file', type=str, required=True,
                       help='Path to annotation JSON file')
    parser.add_argument('--dataset-root', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/dataset_example/droid_new_setup',
                       help='Dataset root directory')
    parser.add_argument('--wm-ckpt', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt',
                       help='World model checkpoint')
    parser.add_argument('--svd-model-path', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e',
                       help='SVD model path')
    parser.add_argument('--clip-model-path', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                       help='CLIP model path')
    parser.add_argument('--pi-ckpt', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid',
                       help='Policy checkpoint')
    parser.add_argument('--action-adapter', type=str,
                       default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/train_adapter/checkpoints/adapter_finetuned_best.pth',
                       help='Action adapter checkpoint')
    parser.add_argument('--num-rollouts', type=int, default=400,
                       help='Number of rollouts to generate')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for synthetic data')
    parser.add_argument('--instruction-variants', type=str, nargs='+',
                       help='Instruction variants for diversity')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')

    args = parser.parse_args()

    # Load initial observation
    print(f"Loading initial observation from {args.annotation_file}")
    initial_obs = load_initial_observation(args.annotation_file, args.dataset_root)
    print(f"Task: {initial_obs['instruction']}")

    # Create generator
    generator = SyntheticDataGenerator(
        wm_ckpt=args.wm_ckpt,
        svd_model_path=args.svd_model_path,
        clip_model_path=args.clip_model_path,
        pi_ckpt=args.pi_ckpt,
        action_adapter_path=args.action_adapter,
        device=args.device,
    )

    # Generate trajectories
    trajectories = generator.generate_diverse_trajectories(
        base_observation=initial_obs,
        base_instruction=initial_obs['instruction'],
        num_rollouts=args.num_rollouts,
        instruction_variants=args.instruction_variants,
    )

    # Save trajectories
    output_dir = Path(args.output_dir)
    generator.save_trajectories(trajectories, output_dir)

    print()
    print("=" * 80)
    print("Synthetic data generation complete!")
    print(f"Generated: {len(trajectories)} trajectories")
    print(f"Saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Review and label trajectories (mark successful ones)")
    print("2. Convert successful trajectories to LeRobot format")
    print("3. Fine-tune π0.5 on the synthetic data")
    print("=" * 80)


if __name__ == '__main__':
    main()
