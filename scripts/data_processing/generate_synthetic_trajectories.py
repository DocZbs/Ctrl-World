#!/usr/bin/env python3
"""
Generate synthetic trajectories for pi0.5 finetuning with VLM-based success detection

Implements policy-in-the-loop imagination rollout with automatic success labeling:
1. Policy outputs action chunk from initial observation
2. World model generates next frames based on actions
3. VLM checks if task is successful
4. Stop episode once success is detected
5. Save trajectories in droid_new_setup format for VLA finetuning

Usage:
    python scripts/data_processing/generate_synthetic_trajectories.py \
        --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json \
        --dataset-root dataset_example/droid_new_setup \
        --num-rollouts 5 \
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
from typing import List, Dict, Any, Optional
from openpi_client import image_tools
import einops

# Add project root to path (scripts/data_processing -> scripts -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
from openpi.training import config as config_pi
from openpi.policies import policy_config
from scipy.spatial.transform import Rotation as R


@dataclass
class SyntheticTrajectory:
    """Synthetic trajectory data structure - compatible with droid_new_setup format"""
    trajectory_id: str
    task_instruction: str
    initial_state: List[float]
    joint_positions: List[List[float]]  # Joint positions (7 joints + 1 gripper) at each step
    cartesian_poses: List[List[float]]  # Cartesian poses (xyz + euler + gripper) at each step
    images_view0: List[np.ndarray]  # Generated images - camera view 0
    images_view1: List[np.ndarray]  # Generated images - camera view 1
    images_view2: List[np.ndarray]  # Generated images - camera view 2
    success: bool = False
    metadata: Dict[str, Any] = None


class VLMSuccessDetector:
    """Vision-Language Model for detecting task success"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize VLM for success detection

        Args:
            model_name: OpenAI model name (gpt-4o, gpt-4o-mini, etc.)
        """
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model_name = model_name
            print(f"✓ VLM Success Detector initialized with {model_name}")
        except ImportError:
            print("Warning: OpenAI library not found. Install with: pip install openai")
            self.client = None

    def check_success(
        self,
        image: np.ndarray,
        task_instruction: str,
        previous_success: bool = False,
    ) -> tuple[bool, float, str]:
        """Check if the task has been successfully completed

        Args:
            image: Current frame (H, W, 3) RGB image
            task_instruction: Task description
            previous_success: Whether success was detected in previous frame

        Returns:
            Tuple of (is_success, confidence, reasoning)
        """
        if self.client is None:
            return False, 0.0, "VLM not available"

        import base64
        from io import BytesIO
        from PIL import Image

        # Convert numpy image to base64
        pil_image = Image.fromarray(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Create prompt
        prompt = f"""You are a robot task success detector. Analyze this image carefully and determine if the following task has been SUCCESSFULLY COMPLETED:

Task: "{task_instruction}"

Consider the task successful ONLY if ALL of the following are true:
1. The described action has been FULLY completed (not in progress)
2. The object is in the final desired state/location as specified in the task
3. The robot gripper has RELEASED the object and moved away (if it was a manipulation task)
4. The scene is stable - no objects are in mid-motion or being held

Be STRICT and CONSERVATIVE in your assessment. If you have any doubt, mark as NOT successful.

Previous status: {"SUCCESS detected in previous frame" if previous_success else "Not yet successful"}

Respond in JSON format:
{{
    "success": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why task is/isn't complete"
}}

Example successful states:
- "pick up the blue block" → Block is being HELD by gripper (gripper closed, block lifted)
- "place the blue block in plate" → Block is IN the plate, gripper is OPEN and AWAY from object

Example NOT successful states:
- Gripper approaching but hasn't grasped yet
- Object in motion or being carried
- Object near target but not placed correctly
- Gripper still holding or touching the object when it should be released
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("success", False),
                    result.get("confidence", 0.0),
                    result.get("reasoning", "")
                )
            else:
                return False, 0.0, "Failed to parse VLM response"

        except Exception as e:
            print(f"Warning: VLM check failed: {e}")
            return False, 0.0, f"Error: {str(e)}"


class SyntheticDataGenerator:
    """Synthetic data generator with VLM-based success detection"""

    def __init__(
        self,
        wm_ckpt: str,
        svd_model_path: str,
        clip_model_path: str,
        pi_ckpt: str,
        policy_type: str = "pi05",
        action_adapter_path: str = None,
        data_stat_path: str = "dataset_meta_info/droid/stat.json",
        wm_device: str = "cuda:1",
        policy_device: str = "cuda:1",
        use_vlm: bool = True,
        vlm_model: str = "gpt-4o-mini",
        vlm_check_interval: int = 3,
    ):
        self.wm_device = wm_device
        self.policy_device = policy_device
        self.policy_type = policy_type
        self.vlm_check_interval = vlm_check_interval

        # Load world model
        print("Loading world model...")
        args = type('Args', (), {
            'val_model_path': wm_ckpt,
            'svd_model_path': svd_model_path,
            'clip_model_path': clip_model_path,
            'dtype': torch.float16,
            'data_stat_path': data_stat_path,
            'action_adapter': action_adapter_path,
            'action_dim': 7,
            'num_history': 6,
            'num_frames': 5,
            'text_cond': True,
            'frame_level_cond': True,
            'his_cond_zero': False,
            'width': 320,
            'height': 192,
            'num_inference_steps': 25,
            'decode_chunk_size': 8,
            'guidance_scale': 2.5,
            'fps': 7,
            'motion_bucket_id': 127,
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
            print(f"✓ Action adapter loaded")

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
            os.environ['OPENPI_JAX_DEVICE'] = '0'

        self.policy = policy_config.create_trained_policy(config, pi_ckpt)

        # Initialize VLM for success detection
        self.vlm_detector = VLMSuccessDetector(model_name=vlm_model) if use_vlm else None

        print(f"✓ Models loaded successfully")
        print(f"  - World Model on: {self.wm_device}")
        print(f"  - Policy on: {self.policy_device}")
        print(f"  - VLM Success Detection: {'Enabled' if use_vlm else 'Disabled'}")

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
        return self.normalize_bound(state, self.state_p01, self.state_p99)

    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        return (normalized_state + 1) / 2 * (self.state_p99 - self.state_p01 + 1e-8) + self.state_p01

    def generate_next_frame(
        self,
        current_latent: torch.Tensor,
        action_sequence: np.ndarray,
        instruction: str,
        history: torch.Tensor = None,
    ) -> tuple:
        """Generate next frame using world model

        Args:
            current_latent: Current frame latent (1, 4, 72, 40)
            action_sequence: Cartesian pose sequence (num_frames+num_history, 7)
            instruction: Task instruction text
            history: History latents (1, num_history, 4, 72, 40) or None

        Returns:
            Tuple of (all_frames, last_frames, predicted_latents)
        """
        args = self.args

        image_cond = current_latent

        # Prepare action condition
        action_cond = self.normalize_bound(action_sequence, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.wm_device).to(self.dtype)

        # Generate using world model
        with torch.no_grad():
            if instruction is not None:
                text_token = self.model.action_encoder(action_cond, instruction, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)

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

        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # Decode predicted video
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        pred_step = self.args.pred_step

        # All frames for saving (first pred_step-1 frames)
        all_frames = []
        for view_idx in range(3):
            all_frames.append(videos[view_idx, :pred_step-1])

        # Last frame for next observation
        last_frames = []
        for view_idx in range(3):
            last_frames.append(videos[view_idx, pred_step-1])

        # Predicted latents for history update
        predicted_latents = []
        for view_idx in range(3):
            predicted_latents.append(latents[view_idx])

        return all_frames, last_frames, predicted_latents

    def forward_policy(self, current_obs, current_state, instruction):
        """Forward pass through policy to get action predictions

        Args:
            current_obs: Current observation with 3 camera views
            current_state: Current joint state (8,) - 7 joints + 1 gripper
            instruction: Task instruction text

        Returns:
            tuple: (joint_pos_skip, cartesian_poses_skip, joint_velocities)
                - joint_pos_skip: (pred_step, 8) joint positions with gripper
                - cartesian_poses_skip: (pred_step, 7) cartesian poses
                - joint_velocities: (15, 7) raw joint velocity actions from policy
        """
        # Prepare policy input
        image1 = current_obs['images'][1]
        image2 = current_obs['images'][2]

        # Convert and resize images
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)

        image1 = torch.nn.functional.interpolate(
            image1.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image2 = torch.nn.functional.interpolate(
            image2.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8)

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
        current_joint = current_state[None, :][:, :7]
        current_gripper = current_state[None, :][:, 7:]

        # Select action indices based on policy type
        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        # Policy output: joint velocity and gripper position
        joint_vel = action_chunk[:, :7][idx]  # (15, 7)
        gripper_pos = action_chunk[:, 7:][idx]  # (15, 1)
        gripper_max = 0.75
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        # Calculate future joint positions using dynamics model
        with torch.no_grad():
            joint_future = self.dynamics_model(current_joint, joint_vel, None, training=False)

        # Concatenate current joint with future joints
        joint_pos_all = np.concatenate([current_joint, joint_future], axis=0)[:15]
        gripper_pos_all = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]

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
        cartesian_poses = np.array(cartesian_poses)

        # Apply skip sampling
        skip = self.args.policy_skip_step
        pred_step = self.args.pred_step
        cartesian_poses_skip = cartesian_poses[::skip][:pred_step]
        joint_pos_skip = joint_pos_all[::skip][:pred_step]

        # Add gripper to joint positions
        joint_pos_skip = np.concatenate([joint_pos_skip, cartesian_poses_skip[:, -1:]], axis=-1)

        return joint_pos_skip, cartesian_poses_skip, joint_vel

    def forward_wm(self, his_cond, his_eef, cartesian_poses_skip, instruction, history_idx):
        """Forward pass through world model to generate next frame"""
        # Prepare action sequence for world model
        history_actions = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)

        # Get future actions
        if len(cartesian_poses_skip) >= self.args.num_frames:
            future_actions = cartesian_poses_skip[:self.args.num_frames]
        else:
            padding = np.repeat(
                cartesian_poses_skip[-1:],
                self.args.num_frames - len(cartesian_poses_skip),
                axis=0
            )
            future_actions = np.concatenate([cartesian_poses_skip, padding], axis=0)

        # Combine history + future
        action_sequence = np.concatenate([history_actions, future_actions], axis=0)

        # Prepare history latents
        his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)

        # Get current latent
        current_latent = his_cond[-1]

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
        max_steps: int = 100,
        min_steps: int = 10,
        success_hold_steps: int = 3,
    ) -> SyntheticTrajectory:
        """Generate a synthetic trajectory with VLM-based success detection

        Args:
            initial_observation: Initial observation (images and state)
            instruction: Task instruction
            max_steps: Maximum steps before stopping
            min_steps: Minimum steps before checking success
            success_hold_steps: Number of consecutive success detections required

        Returns:
            SyntheticTrajectory object
        """
        trajectory_id = f"syn_{np.random.randint(1000000):06d}"

        # Initialize trajectory storage
        joint_positions = [initial_observation['state'].copy()]
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

        # Success detection
        success_detected = False
        success_count = 0
        success_step = None

        # Main rollout loop
        for step in range(max_steps):
            # Policy forward pass
            joint_pos_skip, cartesian_poses_skip, joint_vel = self.forward_policy(
                current_obs, current_state, instruction
            )

            # World model forward pass
            all_frames, last_frames, predicted_latents = self.forward_wm(
                his_cond, his_eef, cartesian_poses_skip, instruction, history_idx
            )

            # CRITICAL FIX: Save states for each generated frame, not just final state
            # The world model generates pred_step-1 frames with corresponding states
            # Each generated frame corresponds to one future state (direct 1:1 mapping)
            pred_step = self.args.pred_step
            num_frames_generated = len(all_frames[0])

            # Direct 1:1 mapping: frame i corresponds to state i
            for frame_idx in range(num_frames_generated):
                # Get corresponding state (direct mapping, no interpolation)
                frame_cartesian_pose = cartesian_poses_skip[frame_idx]
                frame_joint_pos = joint_pos_skip[frame_idx]

                # Append state and frame together
                cartesian_poses_list.append(frame_cartesian_pose.tolist())
                joint_positions.append(frame_joint_pos.tolist())

                images_view0.append(all_frames[0][frame_idx])
                images_view1.append(all_frames[1][frame_idx])
                images_view2.append(all_frames[2][frame_idx])

            # Update state with last predicted position
            current_state[:7] = joint_pos_skip[-1, :7]
            current_state[7:] = joint_pos_skip[-1, 7:]
            cartesian_pose_current = cartesian_poses_skip[-1]

            # Update history buffers
            his_eef.append(cartesian_pose_current[None, :])
            new_latent = torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0)
            his_cond.append(new_latent)

            # Update current observation
            current_obs = {'images': last_frames, 'state': current_state}

            # Check success with VLM (use wrist camera view - view 2)
            if (step >= min_steps and
                self.vlm_detector is not None and
                step % self.vlm_check_interval == 0):

                check_image = last_frames[2]
                is_success, confidence, reasoning = self.vlm_detector.check_success(
                    check_image, instruction, success_detected
                )

                print(f"  Step {step} - VLM Check: {is_success} (conf: {confidence:.2f}) - {reasoning}")

                if is_success and confidence > 0.8:
                    success_count += 1
                    if success_count >= success_hold_steps:
                        success_detected = True
                        success_step = step
                        print(f"✓ Success detected at step {step}! Stopping trajectory.")
                        break
                else:
                    success_count = 0

            # Check termination
            if step >= max_steps - 1:
                print(f"  Reached max steps ({max_steps}), stopping.")
                break

        return SyntheticTrajectory(
            trajectory_id=trajectory_id,
            task_instruction=instruction,
            initial_state=initial_observation['state'].tolist(),
            joint_positions=[s.tolist() if isinstance(s, np.ndarray) else s for s in joint_positions],
            cartesian_poses=cartesian_poses_list,
            images_view0=images_view0,
            images_view1=images_view1,
            images_view2=images_view2,
            success=success_detected,
            metadata={
                'num_steps': step + 1,
                'max_steps': max_steps,
                'success_step': success_step,
                'video_length': len(images_view0),
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
        output_dir: Path = None,
        save_every: int = 1,
        max_steps: int = 100,
        min_steps: int = 10,
    ) -> List[SyntheticTrajectory]:
        """Generate diverse trajectories with incremental saving

        Args:
            base_observation: Base initial observation
            base_instruction: Base task instruction
            num_rollouts: Number of trajectories to generate
            instruction_variants: Instruction variants for diversity
            state_perturbation_std: State perturbation standard deviation
            output_dir: Output directory for incremental saving
            save_every: Save every N trajectories

        Returns:
            List of trajectories
        """
        trajectories = []

        if instruction_variants is None:
            instruction_variants = [base_instruction]

        print(f"Generating {num_rollouts} diverse trajectories...")
        if output_dir:
            print(f"Saving incrementally to {output_dir}")

        success_count = 0

        for i in tqdm(range(num_rollouts), desc="Generating trajectories"):
            # Select instruction variant
            instruction = instruction_variants[i % len(instruction_variants)]

            # Perturb initial state
            perturbed_state = base_observation['state'].copy()
            perturbed_state[:7] += np.random.normal(0, state_perturbation_std, 7)

            perturbed_obs = {
                'images': base_observation['images'],
                'state': perturbed_state,
            }

            # Generate trajectory
            try:
                print(f"\nTrajectory {i+1}/{num_rollouts}")
                trajectory = self.generate_trajectory(
                    initial_observation=perturbed_obs,
                    instruction=instruction,
                    max_steps=max_steps,
                    min_steps=min_steps,
                )
                trajectories.append(trajectory)

                if trajectory.success:
                    success_count += 1
                    print(f"✓ Success! ({success_count}/{i+1} = {success_count/(i+1)*100:.1f}%)")
                else:
                    print(f"✗ Failed (Success rate: {success_count}/{i+1} = {success_count/(i+1)*100:.1f}%)")

                # Incremental saving
                if output_dir and (len(trajectories) % save_every == 0):
                    self.save_trajectories(trajectories[-save_every:], output_dir, start_idx=i-save_every+1)

            except Exception as e:
                import traceback
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                traceback.print_exc()
                continue

        print(f"\n✓ Generated {len(trajectories)} trajectories")
        print(f"  Success rate: {success_count}/{len(trajectories)} = {success_count/len(trajectories)*100:.1f}%")
        return trajectories

    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        output_dir: Path,
        start_idx: int = 0,
    ):
        """Save trajectories in droid_new_setup format

        Args:
            trajectories: List of trajectories
            output_dir: Output directory
            start_idx: Starting episode index
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure like droid_new_setup
        annotation_dir = output_dir / 'annotation' / 'synthetic'
        videos_dir = output_dir / 'videos' / 'synthetic'
        annotation_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        for idx, traj in enumerate(tqdm(trajectories, desc="Saving trajectories")):
            episode_id = f"{start_idx + idx:04d}"

            # Save 3 videos (3 camera views)
            video_subdir = videos_dir / episode_id
            video_subdir.mkdir(exist_ok=True)

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
                'states': traj.cartesian_poses,
                'joints': traj.joint_positions,
            }

            # Save annotation
            annotation_path = annotation_dir / f'{episode_id}.json'
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)

        print(f"✓ Saved {len(trajectories)} trajectories to {output_dir}")
        print(f"  Format: Compatible with droid_new_setup for Pi05 DROID finetuning")
        print(f"  - Annotations: {annotation_dir}/*.json")
        print(f"  - Videos: {videos_dir}/*/{{0,1,2}}.mp4")


def load_initial_observation(annotation_path: str, dataset_root: str) -> Dict[str, Any]:
    """Load initial observation from annotation file"""
    with open(annotation_path) as f:
        anno = json.load(f)

    # Load first frame from all 3 videos
    import cv2
    frames = []
    for video_info in anno['videos']:
        video_path = Path(dataset_root) / video_info['video_path']
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Get initial joint state
    initial_joints = np.array(anno['joints'][0], dtype=np.float32)

    return {
        'images': frames,
        'state': initial_joints,
        'instruction': anno['texts'][0],
    }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories with VLM success detection')
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
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save every N trajectories (default: 1)')
    parser.add_argument('--wm-device', type=str, default='cuda:0',
                       help='Device for world model')
    parser.add_argument('--policy-device', type=str, default='cuda:1',
                       help='Device for policy')
    parser.add_argument('--use-vlm', action='store_true', default=True,
                       help='Use VLM for success detection (default: True)')
    parser.add_argument('--no-vlm', action='store_false', dest='use_vlm',
                       help='Disable VLM success detection')
    parser.add_argument('--vlm-model', type=str, default='gpt-4o-mini',
                       help='VLM model name (default: gpt-4o-mini)')
    parser.add_argument('--vlm-check-interval', type=int, default=3,
                       help='Check success every N steps (default: 3)')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per trajectory (default: 100)')
    parser.add_argument('--min-steps', type=int, default=10,
                       help='Minimum steps before success check (default: 10)')

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
        use_vlm=args.use_vlm,
        vlm_model=args.vlm_model,
        vlm_check_interval=args.vlm_check_interval,
    )

    # Output directory
    output_dir = Path(args.output_dir)

    # Generate trajectories with incremental saving
    trajectories = generator.generate_diverse_trajectories(
        base_observation=initial_obs,
        base_instruction=initial_obs['instruction'],
        num_rollouts=args.num_rollouts,
        instruction_variants=args.instruction_variants,
        output_dir=output_dir,
        save_every=args.save_every,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
    )

    # Final save for any remaining trajectories
    remaining = len(trajectories) % args.save_every
    if remaining > 0:
        generator.save_trajectories(
            trajectories[-remaining:],
            output_dir,
            start_idx=len(trajectories) - remaining
        )

    print()
    print("=" * 80)
    print("Synthetic data generation complete!")
    print(f"Generated: {len(trajectories)} trajectories")
    print(f"Success rate: {sum(t.success for t in trajectories)}/{len(trajectories)}")
    print(f"Saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Review generated trajectories and success labels")
    print("2. Filter successful trajectories for training")
    print("3. Convert to LeRobot format if needed")
    print("4. Fine-tune Pi0.5 DROID on the synthetic data")
    print("=" * 80)


if __name__ == '__main__':
    main()
