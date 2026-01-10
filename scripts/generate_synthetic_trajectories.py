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
from openpi_client import image_tools
import einops

sys.path.append(str(Path(__file__).parent.parent))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
from openpi.training import config as config_pi
from openpi.policies import policy_config
from scipy.spatial.transform import Rotation as R


@dataclass
class SyntheticTrajectory:
    """合成轨迹数据结构 - 用于VLA训练"""
    trajectory_id: str
    task_instruction: str
    initial_state: List[float]  # Initial joint positions (7 joints + 1 gripper)
    joint_positions: List[List[float]]  # Joint positions at each step
    joint_velocities: List[List[float]]  # Joint velocity actions from policy
    cartesian_poses: List[List[float]]  # Cartesian poses (xyz + euler + gripper) from FK
    images_view0: List[np.ndarray]  # Generated images from world model - view 0
    images_view1: List[np.ndarray]  # Generated images from world model - view 1
    images_view2: List[np.ndarray]  # Generated images from world model - view 2
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
        wm_device: str = "cuda:0",
        policy_device: str = "cuda:1",
    ):
        self.wm_device = wm_device
        self.policy_device = policy_device
        self.policy_type = policy_type

        # Load world model
        print("Loading world model...")
        args = type('Args', (), {
            'val_model_path': wm_ckpt,
            'svd_model_path': svd_model_path,
            'clip_model_path': clip_model_path,
            'dtype': torch.float16,
            'data_stat_path': data_stat_path,
            'action_adapter': action_adapter_path,
            # Required by CrtlWorld
            'action_dim': 7,
            'num_history': 6,
            'num_frames': 5,
            'text_cond': True,
            'frame_level_cond': True,
            'his_cond_zero': False,
            # Required for world model generation
            'width': 320,
            'height': 192,
            'num_inference_steps': 25,
            'decode_chunk_size': 8,
            'guidance_scale': 2.5,
            'fps': 7,
            'motion_bucket_id': 127,
            # Policy parameters
            'policy_skip_step': 2,
            'pred_step': 5,
        })()

        self.args = args
        self.dtype = torch.float16

        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(wm_ckpt, map_location=self.wm_device))
        self.model.to(self.wm_device).to(torch.float16)
        self.model.eval()

        # Load normalization stats
        with open(data_stat_path, 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

        # Load action adapter
        if action_adapter_path:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.wm_device)
            self.dynamics_model.load_state_dict(torch.load(action_adapter_path, map_location=self.wm_device))

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

        # Set JAX device for policy
        import os
        if 'cuda:' in self.policy_device:
            policy_gpu_id = self.policy_device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = policy_gpu_id
            os.environ['OPENPI_JAX_DEVICE'] = '0'  # JAX will see it as device 0
        
        self.policy = policy_config.create_trained_policy(config, pi_ckpt)
        print(f"✓ Models loaded successfully")
        print(f"  - World Model on: {self.wm_device}")
        print(f"  - Policy on: {self.policy_device}")

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1]"""
        return self.normalize_bound(state, self.state_p01, self.state_p99)

    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state from [-1, 1]"""
        return (normalized_state + 1) / 2 * (self.state_p99 - self.state_p01 + 1e-8) + self.state_p01

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode image to latent using VAE - matching rollout_interact_pi.py"""
        # image: (H, W, 3) numpy array in [0, 255]
        # Model expects 3 camera views, each encoded separately then concatenated
        # For synthetic data, we replicate the single view 3 times
        import cv2

        # Resize single view to (192, 320) if needed
        if image.shape[:2] != (192, 320):
            image = cv2.resize(image, (320, 192))  # cv2.resize takes (width, height)

        # Convert to torch tensor and normalize to [-1, 1]
        image_tensor = torch.from_numpy(image).to(self.dtype).to(self.wm_device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 192, 320)
        image_tensor = image_tensor / 255.0 * 2 - 1

        # Encode with VAE
        vae = self.model.pipeline.vae
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = latent.mul_(vae.config.scaling_factor)
            # latent shape: (1, 4, 24, 40) for single view

        # Replicate for 3 views and concatenate along height dimension
        # This matches rollout_interact_pi.py: torch.cat([v[0] for v in video_latents], dim=1)
        latent = latent.squeeze(0)  # (4, 24, 40)
        latent_3views = torch.cat([latent, latent, latent], dim=1)  # (4, 72, 40)
        latent_3views = latent_3views.unsqueeze(0)  # (1, 4, 72, 40)

        return latent_3views  # (1, 4, 72, 40)

    def decode_latent(self, latent: torch.Tensor) -> np.ndarray:
        """Decode latent to image using VAE"""
        # latent: (1, 4, H//8, W//8)
        vae = self.model.pipeline.vae
        with torch.no_grad():
            latent = latent / vae.config.scaling_factor
            image = vae.decode(latent, num_frames=1).sample

        # Convert to numpy array in [0, 255]
        image = ((image / 2.0 + 0.5).clamp(0, 1) * 255)
        image = image.detach().to(torch.float32).cpu().numpy()
        image = image.transpose(0, 2, 3, 1).astype(np.uint8)  # (1, H, W, 3)

        return image[0]  # (H, W, 3)

    def generate_next_frame(
        self,
        current_latent: torch.Tensor,
        action_sequence: np.ndarray,
        instruction: str,
        history: torch.Tensor = None,
    ) -> tuple:
        """
        Generate next frame using world model - matching rollout_interact_pi.py

        Args:
            current_latent: Current frame latent (1, 4, 72, 40)
            action_sequence: Cartesian pose sequence (num_frames+num_history, 7)
                            Format: [x, y, z, roll, pitch, yaw, gripper]
            instruction: Task instruction text
            history: History latents (1, num_history, 4, 72, 40) or None

        Returns:
            Tuple of (list of 3 frames, list of 3 view latents)
        """
        args = self.args

        # Use current latent directly (already encoded)
        image_cond = current_latent  # (1, 4, 72, 40)

        # Prepare action condition - exactly like rollout_interact_pi.py
        action_cond = self.normalize_bound(action_sequence, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.wm_device).to(self.dtype)

        # Verify shapes
        print(f"DEBUG: image_cond.shape = {image_cond.shape}")
        print(f"DEBUG: action_cond.shape = {action_cond.shape}")
        print(f"DEBUG: action_sequence.shape before normalize = {action_sequence.shape}")
        print(f"DEBUG: history shape = {history.shape if history is not None else None}")
        assert image_cond.shape[1:] == (4, 72, 40), f"Image shape mismatch: {image_cond.shape}"
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim), f"Action shape mismatch: {action_cond.shape}"

        # Generate using world model - exactly like rollout_interact_pi.py
        with torch.no_grad():
            bsz = action_cond.shape[0]
            print(f"DEBUG: bsz = {bsz}")
            print(f"DEBUG: instruction type={type(instruction)}, value={instruction}")
            if instruction is not None:
                text_token = self.model.action_encoder(action_cond, instruction, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            print(f"DEBUG: text_token.shape = {text_token.shape}")
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=history,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )

        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1)
        # latents shape: (3, num_frames, 4, 24, 40) - 3 camera views separated

        # Decode predicted video - decode all 3 views
        decoded_video = []
        bsz, frame_num = latents.shape[:2]  # bsz=3 (3 views), frame_num=5
        x = latents.flatten(0,1)  # (15, 4, 24, 40)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])  # (3, 5, 3, 192, 320)
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
        # videos shape: (3, 5, 192, 320, 3) - 3 views, 5 frames each

        # Return all frames and last frame separately
        pred_step = self.args.pred_step

        # All frames for saving (first pred_step-1 frames)
        all_frames = []
        for view_idx in range(3):
            all_frames.append(videos[view_idx, :pred_step-1])  # (4, 192, 320, 3) - first 4 frames

        # Last frame for next observation
        last_frames = []
        for view_idx in range(3):
            last_frames.append(videos[view_idx, pred_step-1])  # (192, 320, 3) - last frame

        # Prepare predicted_latents as list of 3 views for history update
        predicted_latents = []
        for view_idx in range(3):
            predicted_latents.append(latents[view_idx])  # (num_frames, 4, 24, 40)

        return all_frames, last_frames, predicted_latents  # list of 3 video clips, list of 3 frames, list of 3 view latents

    def forward_policy(self, current_obs, current_state, instruction):
        """
        Forward pass through policy to get action predictions

        Args:
            current_obs: Current observation with 3 camera views
            current_state: Current joint state (8,) - 7 joints + 1 gripper
            instruction: Task instruction text

        Returns:
            tuple: (joint_pos_skip, cartesian_poses_skip)
                - joint_pos_skip: (pred_step, 8) joint positions with gripper
                - cartesian_poses_skip: (pred_step, 7) cartesian poses
        """
        # Prepare policy input
        image1 = current_obs['images'][1]
        image2 = current_obs['images'][2]

        # Convert numpy to torch tensor
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)

        # Resize images
        image1 = torch.nn.functional.interpolate(
            image1.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image2 = torch.nn.functional.interpolate(
            image2.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8)

        # Convert back to numpy
        image1 = image1.numpy()
        image2 = image2.numpy()

        policy_input = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": current_state[:7],
            "observation/gripper_position": current_state[-1:],
            "prompt": instruction,
        }

        # Get action from policy
        action_chunk = self.policy.infer(policy_input)["actions"]

        # Action adapter: convert joint velocities to cartesian poses
        current_joint = current_state[None, :][:, :7]  # (1, 7)
        current_gripper = current_state[None, :][:, 7:]  # (1, 1)

        # Select action indices based on policy type
        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        # Policy output joint velocity and gripper position
        joint_vel = action_chunk[:, :7][idx]  # (15, 7)
        gripper_pos = action_chunk[:, 7:][idx]  # (15, 1)
        gripper_max = 0.75
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        # Calculate future joint positions using dynamics model
        with torch.no_grad():
            joint_future = self.dynamics_model(current_joint, joint_vel, None, training=False)

        # Concatenate current joint with future joints
        joint_pos_all = np.concatenate([current_joint, joint_future], axis=0)[:15]  # (15, 7)
        gripper_pos_all = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)

        # Convert joint positions to cartesian poses using forward kinematics
        cartesian_poses = []
        for i in range(joint_pos_all.shape[0]):
            fk_result = get_fk_solution(joint_pos_all[i, :7])
            xyz = fk_result[:3, 3]
            rotation_matrix = fk_result[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            cartesian_pose = np.concatenate([xyz, euler, gripper_pos_all[i]], axis=0)
            cartesian_poses.append(cartesian_pose)
        cartesian_poses = np.array(cartesian_poses)  # (15, 7)

        # Apply skip sampling
        skip = self.args.policy_skip_step
        pred_step = self.args.pred_step
        cartesian_poses_skip = cartesian_poses[::skip][:pred_step]  # (5, 7)
        joint_pos_skip = joint_pos_all[::skip][:pred_step]  # (5, 7)

        # Add gripper to joint positions
        joint_pos_skip = np.concatenate([joint_pos_skip, cartesian_poses_skip[:, -1:]], axis=-1)  # (5, 8)

        return joint_pos_skip, cartesian_poses_skip

    def forward_wm(self, his_cond, his_eef, cartesian_poses_skip, instruction, history_idx):
        """
        Forward pass through world model to generate next frame

        Args:
            his_cond: History latents list
            his_eef: History cartesian poses list
            cartesian_poses_skip: (pred_step, 7) cartesian poses from policy
            instruction: Task instruction text
            history_idx: History indices to select

        Returns:
            tuple: (all_frames, last_frames, predicted_latents)
                - all_frames: list of 3 video clips (pred_step-1, 192, 320, 3) for saving
                - last_frames: list of 3 frames (192, 320, 3) for next observation
                - predicted_latents: list of 3 view latents (num_frames, 4, 24, 40)
        """
        # Prepare action sequence for world model
        history_actions = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (6, 7)

        # Get future actions from skip-sampled cartesian_poses
        if len(cartesian_poses_skip) >= self.args.num_frames:
            future_actions = cartesian_poses_skip[:self.args.num_frames]  # (5, 7)
        else:
            padding = np.repeat(
                cartesian_poses_skip[-1:],
                self.args.num_frames - len(cartesian_poses_skip),
                axis=0
            )
            future_actions = np.concatenate([cartesian_poses_skip, padding], axis=0)

        # Combine history + future
        action_sequence = np.concatenate([history_actions, future_actions], axis=0)  # (11, 7)

        # Prepare history latents
        his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, 6, 4, 72, 40)

        # Get current latent
        current_latent = his_cond[-1]  # (1, 4, 72, 40)

        # Generate next frame using world model
        all_frames, last_frames, predicted_latents = self.generate_next_frame(
            current_latent=current_latent,
            action_sequence=action_sequence,
            instruction=instruction,
            history=his_latent if self.dynamics_model is not None else None,
        )

        return all_frames, last_frames, predicted_latents

    def generate_trajectory(
        self,
        initial_observation: Dict[str, Any],
        instruction: str,
        max_steps: int = 20,
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

        # Initialize trajectory storage
        joint_positions = [initial_observation['state'].copy()]
        joint_velocities = []
        cartesian_poses_list = []

        # Store 3 views separately
        initial_frames = initial_observation['images']
        images_view0 = [initial_frames[0]]
        images_view1 = [initial_frames[1]]
        images_view2 = [initial_frames[2]]

        # Initialize history buffers
        his_cond, his_eef = self._initialize_history(initial_observation, cartesian_poses_list)

        # Current state
        current_obs = initial_observation
        current_state = initial_observation['state'].copy()
        history_idx = [0, 0, -12, -9, -6, -3]

        # Main rollout loop
        for step in range(max_steps):
            print(f"Step {step}/{max_steps}")

            # Policy forward pass
            joint_pos_skip, cartesian_poses_skip = self.forward_policy(
                current_obs, current_state, instruction
            )

            print(f"  Policy output - first pose: {cartesian_poses_skip[0][:3]}")  # xyz
            print(f"  Policy output - last pose: {cartesian_poses_skip[-1][:3]}")  # xyz

            # Update state with last predicted position
            current_state[:7] = joint_pos_skip[-1, :7]
            current_state[7:] = joint_pos_skip[-1, 7:]
            cartesian_pose_current = cartesian_poses_skip[-1]

            # World model forward pass
            all_frames, last_frames, predicted_latents = self.forward_wm(
                his_cond, his_eef, cartesian_poses_skip, instruction, history_idx
            )

            # Update trajectory
            joint_positions.append(current_state.copy())
            joint_velocities.append(cartesian_pose_current.tolist())  # Save cartesian pose as action
            cartesian_poses_list.append(cartesian_pose_current.tolist())

            # Save all frames (pred_step-1 frames) for each view
            for frame_idx in range(len(all_frames[0])):  # Iterate over frames (4 frames)
                images_view0.append(all_frames[0][frame_idx])
                images_view1.append(all_frames[1][frame_idx])
                images_view2.append(all_frames[2][frame_idx])

            # Update history buffers
            pred_step = self.args.pred_step
            his_eef.append(cartesian_pose_current[None, :])
            new_latent = torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0)
            his_cond.append(new_latent)

            # Update current observation with last frame
            current_obs = {'images': last_frames, 'state': current_state}

            # Check termination
            if step >= max_steps - 1:
                break

        return SyntheticTrajectory(
            trajectory_id=trajectory_id,
            task_instruction=instruction,
            initial_state=initial_observation['state'].tolist(),
            joint_positions=[s.tolist() if isinstance(s, np.ndarray) else s for s in joint_positions],
            joint_velocities=joint_velocities,
            cartesian_poses=cartesian_poses_list,
            images_view0=images_view0,
            images_view1=images_view1,
            images_view2=images_view2,
            success=False,
            metadata={
                'num_steps': len(joint_velocities),
                'max_steps': max_steps,
            }
        )

    def _initialize_history(self, initial_observation, cartesian_poses_list):
        """Initialize history buffers for latents and cartesian poses"""
        his_cond = []
        his_eef = []

        # Encode initial 3 views
        initial_frames = initial_observation['images']
        initial_latents_list = []
        for frame in initial_frames:
            if frame.shape[:2] != (192, 320):
                import cv2
                frame = cv2.resize(frame, (320, 192))

            frame_tensor = torch.from_numpy(frame).to(self.dtype).to(self.wm_device)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
            frame_tensor = frame_tensor / 255.0 * 2 - 1

            vae = self.model.pipeline.vae
            with torch.no_grad():
                latent = vae.encode(frame_tensor).latent_dist.sample()
                latent = latent.mul_(vae.config.scaling_factor)
            initial_latents_list.append(latent.squeeze(0))

        # Concatenate 3 views along height dimension
        initial_latent = torch.cat(initial_latents_list, dim=1).unsqueeze(0)

        # Fill history with initial latent
        for _ in range(self.args.num_history * 4):
            his_cond.append(initial_latent)

        # Compute initial cartesian pose
        initial_joint = initial_observation['state'][:7]
        initial_gripper = initial_observation['state'][-1:]
        fk_result = get_fk_solution(initial_joint)
        xyz = fk_result[:3, 3]
        rotation_matrix = fk_result[:3, :3]
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('xyz')
        initial_cartesian = np.concatenate([xyz, euler, initial_gripper], axis=0)
        cartesian_poses_list.append(initial_cartesian.tolist())

        # Fill history with initial cartesian pose
        for _ in range(self.args.num_history * 4):
            his_eef.append(initial_cartesian[None, :])

        return his_cond, his_eef


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
                'images': base_observation['images'],
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
                import traceback
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                continue

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        output_dir: Path,
    ):
        """保存轨迹到磁盘 - 格式兼容droid_new_setup数据集"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure like droid_new_setup
        annotation_dir = output_dir / 'annotation' / 'synthetic'
        videos_dir = output_dir / 'videos' / 'synthetic'
        annotation_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        for idx, traj in enumerate(tqdm(trajectories, desc="Saving trajectories")):
            episode_id = f"{idx:04d}"

            # Save 3 videos (3 camera views) like droid_new_setup
            video_subdir = videos_dir / episode_id
            video_subdir.mkdir(exist_ok=True)

            # Save each view as a separate video
            if traj.images_view0:
                mediapy.write_video(str(video_subdir / '0.mp4'), traj.images_view0, fps=10)
            if traj.images_view1:
                mediapy.write_video(str(video_subdir / '1.mp4'), traj.images_view1, fps=10)
            if traj.images_view2:
                mediapy.write_video(str(video_subdir / '2.mp4'), traj.images_view2, fps=10)

            # Create annotation in droid_new_setup format
            annotation = {
                'texts': [traj.task_instruction],
                'episode_id': episode_id,
                'success': 1 if traj.success else 0,
                'video_length': len(traj.images_view0),
                'videos': [
                    {'video_path': f'videos/synthetic/{episode_id}/0.mp4'},
                    {'video_path': f'videos/synthetic/{episode_id}/1.mp4'},
                    {'video_path': f'videos/synthetic/{episode_id}/2.mp4'},
                ],
                'states': traj.cartesian_poses,  # Cartesian poses (xyz + euler + gripper)
                'joints': traj.joint_positions,  # Joint positions (7 joints + 1 gripper)
                # Note: We don't save joint_velocities in the annotation format
                # They can be computed from joint positions if needed
            }

            # Save annotation
            annotation_path = annotation_dir / f'{episode_id}.json'
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)

            # Also save detailed trajectory data for debugging/analysis
            debug_dir = output_dir / 'debug' / episode_id
            debug_dir.mkdir(parents=True, exist_ok=True)

            debug_data = {
                'trajectory_id': traj.trajectory_id,
                'episode_id': episode_id,
                'task_instruction': traj.task_instruction,
                'joint_velocities': traj.joint_velocities,  # Policy actions
                'metadata': traj.metadata,
            }

            with open(debug_dir / 'debug_info.json', 'w') as f:
                json.dump(debug_data, f, indent=2)

        print(f"✓ Saved {len(trajectories)} trajectories to {output_dir}")
        print(f"  Format: Compatible with droid_new_setup dataset")
        print(f"  - Annotations: {annotation_dir}/*.json")
        print(f"  - Videos: {videos_dir}/*/0.mp4")
        print(f"  - Debug info: {output_dir}/debug/*/debug_info.json")


def load_initial_observation(annotation_path: str, dataset_root: str) -> Dict[str, Any]:
    """从annotation文件加载初始观测 - 加载3个摄像头视角"""
    with open(annotation_path) as f:
        anno = json.load(f)

    # Load first frame from all 3 videos (3 camera views)
    import cv2
    frames = []
    for video_info in anno['videos']:
        video_path = Path(dataset_root) / video_info['video_path']
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        # Convert BGR to RGB (OpenCV reads in BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Get initial joint state (not cartesian state!)
    # Policy expects joint positions, not cartesian poses
    initial_joints = np.array(anno['joints'][0], dtype=np.float32)

    return {
        'images': frames,  # All 3 views [view0, view1, view2]
        'state': initial_joints,  # Joint positions (7 joints + 1 gripper = 8)
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
    parser.add_argument('--num-rollouts', type=int, default=5,
                       help='Number of rollouts to generate')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for synthetic data')
    parser.add_argument('--instruction-variants', type=str, nargs='+',
                       help='Instruction variants for diversity')
    parser.add_argument('--wm-device', type=str, default='cuda:0',
                       help='Device for world model')
    parser.add_argument('--policy-device', type=str, default='cuda:1',
                       help='Device for policy (VLA)')

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
        wm_device=args.wm_device,
        policy_device=args.policy_device,
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
