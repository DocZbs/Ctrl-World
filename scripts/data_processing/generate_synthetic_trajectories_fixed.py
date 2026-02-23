#!/usr/bin/env python3
"""
Generate synthetic trajectories for pi0.5 finetuning with VLM-based success detection

FIXED VERSION that generates data in the exact droid_new_setup format:
- JSON annotations with states, joints, texts, videos, latent_videos
- MP4 videos in videos/{split}/{episode_id}/{view_id}.mp4
- Latent videos in latent_videos/{split}/{episode_id}/{view_id}.pt
- Success label from VLM

Usage:
    python scripts/data_processing/generate_synthetic_trajectories_fixed.py \
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
from dataclasses import dataclass
from typing import List, Dict, Any
from openpi_client import image_tools
import einops

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
from openpi.training import config as config_pi
from openpi.policies import policy_config
from scipy.spatial.transform import Rotation as R


class VLMSuccessDetector:
    """Vision-Language Model for detecting task success"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
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
    ) -> tuple:
        """Check if the task has been successfully completed"""
        if self.client is None:
            return False, 0.0, "VLM not available"

        import base64
        from io import BytesIO
        from PIL import Image

        pil_image = Image.fromarray(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

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

        with open(data_stat_path, 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

        if action_adapter_path:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.wm_device)
            self.dynamics_model.load_state_dict(torch.load(action_adapter_path, map_location=self.wm_device))
            print(f"✓ Action adapter loaded")

        print(f"Loading {policy_type} policy...")
        if 'pi05' in policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in policy_type:
            config = config_pi.get_config("pi0fast_droid")
        elif 'pi0' in policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        import os
        if 'cuda:' in self.policy_device:
            policy_gpu_id = self.policy_device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = policy_gpu_id
            os.environ['OPENPI_JAX_DEVICE'] = '0'

        self.policy = policy_config.create_trained_policy(config, pi_ckpt)

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
        """Generate next frame using world model"""
        args = self.args

        image_cond = current_latent

        action_cond = self.normalize_bound(action_sequence, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.wm_device).to(self.dtype)

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

        all_frames = []
        for view_idx in range(3):
            all_frames.append(videos[view_idx, :pred_step-1])

        last_frames = []
        for view_idx in range(3):
            last_frames.append(videos[view_idx, pred_step-1])

        predicted_latents = []
        for view_idx in range(3):
            predicted_latents.append(latents[view_idx])

        return all_frames, last_frames, predicted_latents

    def forward_policy(self, current_obs, current_state, instruction):
        """Forward pass through policy to get action predictions"""
        image1 = current_obs['images'][1]
        image2 = current_obs['images'][2]

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

        action_chunk = self.policy.infer(policy_input)["actions"]

        current_joint = current_state[None, :][:, :7]
        current_gripper = current_state[None, :][:, 7:]

        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        joint_vel = action_chunk[:, :7][idx]
        gripper_pos = action_chunk[:, 7:][idx]
        gripper_max = 0.75
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        with torch.no_grad():
            joint_future = self.dynamics_model(current_joint, joint_vel, None, training=False)

        joint_pos_all = np.concatenate([current_joint, joint_future], axis=0)[:15]
        gripper_pos_all = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]

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

        skip = self.args.policy_skip_step
        pred_step = self.args.pred_step
        cartesian_poses_skip = cartesian_poses[::skip][:pred_step]
        joint_pos_skip = joint_pos_all[::skip][:pred_step]

        joint_pos_skip = np.concatenate([joint_pos_skip, cartesian_poses_skip[:, -1:]], axis=-1)

        return joint_pos_skip, cartesian_poses_skip, joint_vel

    def forward_wm(self, his_cond, his_eef, cartesian_poses_skip, instruction, history_idx):
        """Forward pass through world model to generate next frame"""
        history_actions = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)

        if len(cartesian_poses_skip) >= self.args.num_frames:
            future_actions = cartesian_poses_skip[:self.args.num_frames]
        else:
            padding = np.repeat(
                cartesian_poses_skip[-1:],
                self.args.num_frames - len(cartesian_poses_skip),
                axis=0
            )
            future_actions = np.concatenate([cartesian_poses_skip, padding], axis=0)

        action_sequence = np.concatenate([history_actions, future_actions], axis=0)

        his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)

        current_latent = his_cond[-1]

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
    ) -> Dict[str, Any]:
        """Generate a synthetic trajectory with VLM-based success detection

        Returns trajectory in droid_new_setup format
        """
        joint_positions = [initial_observation['state'].tolist()]
        cartesian_poses_list = []

        initial_frames = initial_observation['images']
        images_view0 = [initial_frames[0]]
        images_view1 = [initial_frames[1]]
        images_view2 = [initial_frames[2]]

        latent_view0 = []
        latent_view1 = []
        latent_view2 = []

        his_cond, his_eef, initial_latents_per_view = self._initialize_history(initial_observation, cartesian_poses_list)

        # Add initial frame latents to latent lists
        latent_view0.append(initial_latents_per_view[0])
        latent_view1.append(initial_latents_per_view[1])
        latent_view2.append(initial_latents_per_view[2])

        current_obs = initial_observation
        current_state = initial_observation['state'].copy()
        history_idx = [0, 0, -12, -9, -6, -3]

        success_detected = False
        success_count = 0
        success_step = None

        for step in range(max_steps):
            joint_pos_skip, cartesian_poses_skip, joint_vel = self.forward_policy(
                current_obs, current_state, instruction
            )

            all_frames, last_frames, predicted_latents = self.forward_wm(
                his_cond, his_eef, cartesian_poses_skip, instruction, history_idx
            )

            pred_step = self.args.pred_step
            num_frames_generated = len(all_frames[0])

            for frame_idx in range(num_frames_generated):
                frame_cartesian_pose = cartesian_poses_skip[frame_idx]
                frame_joint_pos = joint_pos_skip[frame_idx]

                cartesian_poses_list.append(frame_cartesian_pose.tolist())
                joint_positions.append(frame_joint_pos.tolist())

                images_view0.append(all_frames[0][frame_idx])
                images_view1.append(all_frames[1][frame_idx])
                images_view2.append(all_frames[2][frame_idx])

            current_state[:7] = joint_pos_skip[-1, :7]
            current_state[7:] = joint_pos_skip[-1, 7:]
            cartesian_pose_current = cartesian_poses_skip[-1]

            his_eef.append(cartesian_pose_current[None, :])
            new_latent = torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0)
            his_cond.append(new_latent)

            for view_idx in range(3):
                latent_frames = predicted_latents[view_idx][:pred_step-1]
                if view_idx == 0:
                    latent_view0.extend([latent_frames[i].cpu() for i in range(len(latent_frames))])
                elif view_idx == 1:
                    latent_view1.extend([latent_frames[i].cpu() for i in range(len(latent_frames))])
                elif view_idx == 2:
                    latent_view2.extend([latent_frames[i].cpu() for i in range(len(latent_frames))])

            current_obs = {'images': last_frames, 'state': current_state}

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

            if step >= max_steps - 1:
                print(f"  Reached max steps ({max_steps}), stopping.")
                break

        return {
            'texts': [instruction],
            'success': 1 if success_detected else 0,
            'video_length': len(images_view0),
            'states': cartesian_poses_list,
            'joints': joint_positions,
            'images_view0': images_view0,
            'images_view1': images_view1,
            'images_view2': images_view2,
            'latent_view0': latent_view0,
            'latent_view1': latent_view1,
            'latent_view2': latent_view2,
            'metadata': {
                'num_steps': step + 1,
                'max_steps': max_steps,
                'success_step': success_step,
            }
        }

    def _initialize_history(self, initial_observation, cartesian_poses_list):
        """Initialize history buffers for latents and cartesian poses"""
        his_cond = []
        his_eef = []

        initial_frames = initial_observation['images']
        initial_latents_list = []
        initial_latents_per_view = []
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
            initial_latents_per_view.append(latent.squeeze(0).cpu())

        initial_latent = torch.cat(initial_latents_list, dim=1).unsqueeze(0)

        for _ in range(self.args.num_history * 4):
            his_cond.append(initial_latent)

        initial_joint = initial_observation['state'][:7]
        initial_gripper = initial_observation['state'][-1:]
        fk_result = get_fk_solution(initial_joint)
        xyz = fk_result[:3, 3]
        rotation_matrix = fk_result[:3, :3]
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('xyz')
        initial_cartesian = np.concatenate([xyz, euler, initial_gripper], axis=0)
        cartesian_poses_list.append(initial_cartesian.tolist())

        for _ in range(self.args.num_history * 4):
            his_eef.append(initial_cartesian[None, :])

        return his_cond, his_eef, initial_latents_per_view

    def save_trajectory_droid_format(
        self,
        trajectory: Dict[str, Any],
        output_dir: Path,
        episode_id: str,
        split: str = "train",
    ):
        """Save trajectory in droid_new_setup format

        Format matches dataset_example/droid_new_setup/:
        - annotation/{split}/{episode_id}.json
        - videos/{split}/{episode_id}/{view_id}.mp4
        - latent_videos/{split}/{episode_id}/{view_id}.pt
        """
        output_dir = Path(output_dir)

        annotation_dir = output_dir / "annotation" / split
        videos_dir = output_dir / "videos" / split / episode_id
        latent_videos_dir = output_dir / "latent_videos" / split / episode_id

        annotation_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        latent_videos_dir.mkdir(parents=True, exist_ok=True)

        for view_idx, (images, latents) in enumerate([
            (trajectory['images_view0'], trajectory['latent_view0']),
            (trajectory['images_view1'], trajectory['latent_view1']),
            (trajectory['images_view2'], trajectory['latent_view2']),
        ]):
            video_path = videos_dir / f"{view_idx}.mp4"
            mediapy.write_video(str(video_path), images, fps=10)

            latent_path = latent_videos_dir / f"{view_idx}.pt"
            latent_tensor = torch.stack(latents, dim=0)
            torch.save(latent_tensor, latent_path)

        annotation = {
            "texts": trajectory['texts'],
            "episode_id": episode_id,
            "success": trajectory['success'],
            "video_length": trajectory['video_length'],
            "videos": [
                {"video_path": f"videos/{split}/{episode_id}/0.mp4"},
                {"video_path": f"videos/{split}/{episode_id}/1.mp4"},
                {"video_path": f"videos/{split}/{episode_id}/2.mp4"},
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{split}/{episode_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{split}/{episode_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{split}/{episode_id}/2.pt"},
            ],
            "states": trajectory['states'],
            "joints": trajectory['joints'],
        }

        annotation_path = annotation_dir / f"{episode_id}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        print(f"✓ Saved episode {episode_id} in droid_new_setup format")

    def generate_diverse_trajectories(
        self,
        base_observation: Dict[str, Any],
        base_instruction: str,
        num_rollouts: int = 400,
        instruction_variants: List[str] = None,
        state_perturbation_std: float = 0.05,
        output_dir: Path = None,
        split: str = "train",
        max_steps: int = 100,
        min_steps: int = 10,
    ):
        """Generate diverse trajectories with incremental saving"""
        if instruction_variants is None:
            instruction_variants = [base_instruction]

        print(f"Generating {num_rollouts} diverse trajectories...")
        if output_dir:
            print(f"Saving to {output_dir} in droid_new_setup format")

        success_count = 0

        for i in tqdm(range(num_rollouts), desc="Generating trajectories"):
            instruction = instruction_variants[i % len(instruction_variants)]

            perturbed_state = base_observation['state'].copy()
            perturbed_state[:7] += np.random.normal(0, state_perturbation_std, 7)

            perturbed_obs = {
                'images': base_observation['images'],
                'state': perturbed_state,
            }

            try:
                print(f"\nTrajectory {i+1}/{num_rollouts}")
                trajectory = self.generate_trajectory(
                    initial_observation=perturbed_obs,
                    instruction=instruction,
                    max_steps=max_steps,
                    min_steps=min_steps,
                )

                if trajectory['success']:
                    success_count += 1
                    print(f"✓ Success! ({success_count}/{i+1} = {success_count/(i+1)*100:.1f}%)")
                else:
                    print(f"✗ Failed (Success rate: {success_count}/{i+1} = {success_count/(i+1)*100:.1f}%)")

                if output_dir:
                    episode_id = f"{i:04d}"
                    self.save_trajectory_droid_format(trajectory, output_dir, episode_id, split)

            except Exception as e:
                import traceback
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                traceback.print_exc()
                continue

        print(f"\n✓ Generated {num_rollouts} trajectories")
        print(f"  Success rate: {success_count}/{num_rollouts} = {success_count/num_rollouts*100:.1f}%")


def load_initial_observation(annotation_path: str, dataset_root: str) -> Dict[str, Any]:
    """Load initial observation from annotation file"""
    with open(annotation_path) as f:
        anno = json.load(f)

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
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split (train/val)')
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

    print(f"Loading initial observation from {args.annotation_file}")
    initial_obs = load_initial_observation(args.annotation_file, args.dataset_root)
    print(f"Task: {initial_obs['instruction']}")

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

    output_dir = Path(args.output_dir)

    generator.generate_diverse_trajectories(
        base_observation=initial_obs,
        base_instruction=initial_obs['instruction'],
        num_rollouts=args.num_rollouts,
        instruction_variants=args.instruction_variants,
        output_dir=output_dir,
        split=args.split,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
    )

    print()
    print("=" * 80)
    print("Synthetic data generation complete!")
    print(f"Saved to: {output_dir}")
    print(f"Format: droid_new_setup (JSON annotations + videos + latent videos)")
    print()
    print("Next steps:")
    print("1. Review generated trajectories and success labels")
    print("2. Use this data directly for pi05_droid finetuning")
    print("=" * 80)


if __name__ == '__main__':
    main()
