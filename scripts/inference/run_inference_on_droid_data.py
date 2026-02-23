#!/usr/bin/env python3
"""
对droid_data中的1000个episode进行推理并生成DROID格式的合成数据

Usage:
    python scripts/inference/run_inference_on_droid_data.py \
        --data-dir /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data \
        --output-dir inference_results \
        --start-episode 0 \
        --end-episode 1000 \
        --device cuda:0 \
        --generate-data \
        --wm-ckpt /path/to/world_model_checkpoint
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys
import mediapy
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openpi.training import config as config_pi
from openpi.policies import policy_config
from openpi_client import image_tools
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution
from scipy.spatial.transform import Rotation as R


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB uint8 frame, got shape={frame.shape}")
    return frame


def _normalize_action_value(action, action_dim: int) -> list:
    """Normalize action to a python list[float] with fixed length.

    PyArrow (via pandas.to_parquet) requires list columns to be consistently list-typed.
    """
    if action is None:
        return [0.0] * action_dim

    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    if isinstance(action, np.ndarray):
        action = action.reshape(-1).tolist()
    elif isinstance(action, (tuple, list)):
        action = list(action)
    else:
        # Scalar fallback
        try:
            scalar = float(action)
            return [scalar] + [0.0] * max(0, action_dim - 1)
        except Exception:
            return [0.0] * action_dim

    # If nested (e.g., [[...]]), take the first row.
    if len(action) == 1 and isinstance(action[0], (tuple, list, np.ndarray)):
        inner = action[0]
        if isinstance(inner, np.ndarray):
            action = inner.reshape(-1).tolist()
        else:
            action = list(inner)

    out = []
    for x in action:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)

    if len(out) < action_dim:
        out.extend([0.0] * (action_dim - len(out)))
    elif len(out) > action_dim:
        out = out[:action_dim]
    return out


def _write_video_pyav(video_file: Path, frames: list, fps: int) -> Tuple[bool, str]:
    try:
        import av  # type: ignore
    except Exception as e:
        return False, f"pyav import failed: {e}"

    if not frames:
        return False, "no frames to write"

    video_file.parent.mkdir(parents=True, exist_ok=True)
    first = _as_uint8_rgb(np.asarray(frames[0]))
    height, width = first.shape[:2]

    last_err = ""
    for codec_name in ("libx264", "h264", "mpeg4"):
        try:
            if video_file.exists():
                video_file.unlink()

            container = av.open(str(video_file), mode="w")
            stream = container.add_stream(codec_name, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for frame in frames:
                frame_np = _as_uint8_rgb(np.asarray(frame))
                if frame_np.shape[:2] != (height, width):
                    frame_np = cv2.resize(frame_np, (width, height), interpolation=cv2.INTER_AREA)
                av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            if not video_file.exists() or video_file.stat().st_size == 0:
                return False, f"pyav wrote empty file with codec={codec_name}"
            return True, f"pyav(codec={codec_name})"
        except Exception as e:
            last_err = f"{codec_name}: {e}"
            try:
                container.close()
            except Exception:
                pass
            try:
                if video_file.exists():
                    video_file.unlink()
            except Exception:
                pass
            continue

    return False, f"pyav failed: {last_err}"


def _write_video_opencv(video_file: Path, frames: list, fps: int) -> Tuple[bool, str]:
    if not frames:
        return False, "no frames to write"

    video_file.parent.mkdir(parents=True, exist_ok=True)
    first = _as_uint8_rgb(np.asarray(frames[0]))
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        return False, "cv2.VideoWriter could not be opened (mp4v)"

    try:
        for frame in frames:
            frame_np = _as_uint8_rgb(np.asarray(frame))
            if frame_np.shape[:2] != (height, width):
                frame_np = cv2.resize(frame_np, (width, height), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    if not video_file.exists() or video_file.stat().st_size == 0:
        return False, "opencv wrote empty file"
    return True, "opencv(mp4v)"


def _write_video_best_effort(video_file: Path, frames: list, fps: int) -> None:
    backends = []
    ok, msg = _write_video_pyav(video_file, frames, fps)
    backends.append(msg)
    if ok:
        return

    ok, msg = _write_video_opencv(video_file, frames, fps)
    backends.append(msg)
    if ok:
        return

    # Last resort: mediapy (requires stacking all frames)
    try:
        frames_np = np.stack([_as_uint8_rgb(np.asarray(f)) for f in frames], axis=0)
        mediapy.write_video(str(video_file), frames_np, fps=fps)
        if not video_file.exists() or video_file.stat().st_size == 0:
            raise RuntimeError("mediapy wrote empty file")
        return
    except Exception as e:
        backends.append(f"mediapy failed: {e}")

    raise RuntimeError("All video writers failed: " + " | ".join(backends))


def _maybe_save_png_fallback(video_dir: Path, frames: list, episode_idx: int) -> None:
    if os.environ.get("CTRLWORLD_SAVE_PNG_FALLBACK", "0") != "1":
        return
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return

    out_dir = video_dir / "frames" / f"episode_{episode_idx:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_np = _as_uint8_rgb(np.asarray(frame))
        Image.fromarray(frame_np).save(out_dir / f"frame_{i:06d}.png")


class DroidDataInference:
    """对droid_data格式的数据进行推理和生成"""

    def __init__(
        self,
        pi_ckpt: str,
        policy_type: str = "pi05",
        device: str = "cuda:0",
        generate_data: bool = False,
        wm_ckpt: str = None,
        wm_device: str = None,  # Separate device for world model
        svd_model_path: str = "stabilityai/stable-video-diffusion-img2vid",
        clip_model_path: str = "openai/clip-vit-base-patch32",
        data_stat_path: str = None,
        action_adapter_path: str = None,
    ):
        self.policy_type = policy_type
        self.device = device  # Policy device
        self.wm_device = wm_device if wm_device is not None else device  # World model device
        self.generate_data = generate_data

        print(f"Loading {policy_type} policy...")
        if 'pi05' in policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in policy_type:
            config = config_pi.get_config("pi0_fast_droid")
        elif 'pi0' in policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # Set JAX device
        import os
        if 'cuda:' in self.device:
            policy_gpu_id = self.device.split(':')[1]
            # Only set CUDA_VISIBLE_DEVICES if not already set by parent script
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = policy_gpu_id
            os.environ['OPENPI_JAX_DEVICE'] = policy_gpu_id

        self.policy = policy_config.create_trained_policy(config, pi_ckpt)
        print(f"✓ Policy loaded on {self.device}")

        # Load world model if data generation is enabled
        if self.generate_data:
            if not wm_ckpt:
                raise ValueError("World model checkpoint (--wm-ckpt) is required when --generate-data is enabled")

            print(f"Loading world model on {self.wm_device}...")
            self._load_world_model(
                wm_ckpt=wm_ckpt,
                svd_model_path=svd_model_path,
                clip_model_path=clip_model_path,
                data_stat_path=data_stat_path,
                action_adapter_path=action_adapter_path,
            )
            print(f"✓ World model loaded on {self.wm_device}")

    def _load_world_model(
        self,
        wm_ckpt: str,
        svd_model_path: str,
        clip_model_path: str,
        data_stat_path: str,
        action_adapter_path: str,
    ):
        """Load world model for data generation"""
        wm_dtype = torch.float16 if "cuda:" in str(self.wm_device) else torch.float32
        args = type('Args', (), {
            'val_model_path': wm_ckpt,
            'svd_model_path': svd_model_path,
            'clip_model_path': clip_model_path,
            'dtype': wm_dtype,
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
            'history_idx': [0, 0, -12, -9, -6, -3],
            'gripper_max': 0.75,
        })()

        self.wm_args = args
        self.dtype = wm_dtype

        self.wm_model = CrtlWorld(args)

        # Load checkpoint directly to target device
        checkpoint = torch.load(wm_ckpt, map_location=self.wm_device)
        self.wm_model.load_state_dict(checkpoint)
        self.wm_model.to(self.wm_device).to(wm_dtype)
        self.wm_model.eval()

        # Load normalization stats
        if data_stat_path:
            with open(data_stat_path, 'r') as f:
                data_stat = json.load(f)
                self.state_p01 = np.array(data_stat['state_01'])[None, :]
                self.state_p99 = np.array(data_stat['state_99'])[None, :]
        else:
            # Use default stats if not provided
            self.state_p01 = None
            self.state_p99 = None

        # Load action adapter
        if action_adapter_path:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.wm_device)
            adapter_checkpoint = torch.load(action_adapter_path, map_location=self.wm_device)
            self.dynamics_model.load_state_dict(adapter_checkpoint)
            print(f"✓ Action adapter loaded")

    def _encode_stacked_rgb_to_latent(self, stacked_rgb: np.ndarray) -> torch.Tensor:
        """Encode stacked RGB (H*3, W, 3) uint8 to a single latent (1, 4, 72, 40).

        This follows rollout_interact_pi.py more strictly by:
        - encoding each camera view separately with the VAE (sample + scaling_factor),
        - then concatenating the 3 view latents along the height dimension.
        """
        if stacked_rgb.dtype != np.uint8:
            stacked_rgb = stacked_rgb.astype(np.uint8)

        pipeline = self.wm_model.pipeline
        vae = pipeline.vae

        target_h = int(self.wm_args.height)
        target_w = int(self.wm_args.width)

        if stacked_rgb.ndim != 3 or stacked_rgb.shape[2] != 3:
            raise ValueError(f"Expected stacked RGB shape (H*3, W, 3), got {stacked_rgb.shape}")
        if stacked_rgb.shape[0] % 3 != 0:
            raise ValueError(f"Stacked RGB height must be divisible by 3, got H={stacked_rgb.shape[0]}")

        h = stacked_rgb.shape[0] // 3
        views = [stacked_rgb[:h], stacked_rgb[h:2 * h], stacked_rgb[2 * h:]]

        def _encode_one(view_rgb: np.ndarray) -> torch.Tensor:
            # (H, W, 3) uint8 -> (1, 3, H, W) in [-1, 1]
            x = torch.from_numpy(view_rgb).permute(2, 0, 1).unsqueeze(0).to(self.wm_device)
            x = x.to(dtype=torch.float32) / 255.0
            if x.shape[-2:] != (target_h, target_w):
                x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            x = x * 2.0 - 1.0
            x = x.to(dtype=vae.dtype)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
            return latent

        latents = [_encode_one(v) for v in views]  # each (1, 4, 24, 40)
        combined = torch.cat(latents, dim=2)  # (1, 4, 72, 40)
        return combined.to(dtype=self.dtype)

    def _forward_policy_to_cartesian(self, exterior_1_rgb: np.ndarray, wrist_rgb: np.ndarray, joints8: np.ndarray, text: str):
        """Match rollout_interact_pi.py forward_policy output (joint_pos_skip, state_fk_skip)."""
        # Resize for policy
        image1 = torch.from_numpy(exterior_1_rgb)
        image2 = torch.from_numpy(wrist_rgb)

        image1 = torch.nn.functional.interpolate(
            image1.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        image2 = torch.nn.functional.interpolate(
            image2.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": joints8[:7],
            "observation/gripper_position": joints8[-1:],
            "prompt": text,
        }
        action_chunk = self.policy.infer(example)["actions"]  # (15, 8)

        # action adapter + FK (copied structure from rollout_interact_pi.py)
        current_joint = joints8[None, :][:, :7]
        current_gripper = joints8[None, :][:, 7:]
        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        joint_vel = action_chunk[:, :7][idx]  # (15, 7)
        gripper_pos = action_chunk[:, 7:][idx]  # (15, 1)
        gripper_pos = np.clip(gripper_pos, 0, float(self.wm_args.gripper_max))

        if hasattr(self, "dynamics_model") and self.dynamics_model is not None:
            with torch.no_grad():
                joint_future = self.dynamics_model(current_joint, joint_vel, None, training=False)
            joint_pos = np.concatenate([current_joint, joint_future], axis=0)[:15]  # (15, 7)
        else:
            if not getattr(self, "_warned_no_action_adapter", False):
                print(
                    "Warning: action adapter not loaded; joint positions will not be updated from joint velocities "
                    "(rollouts may look static). Set --action-adapter-path / ACTION_ADAPTER_PATH."
                )
                self._warned_no_action_adapter = True
            joint_pos = np.tile(current_joint, (15, 1))

        joint_pos = np.array(joint_pos)
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)

        state_fk = []
        for i in range(joint_pos.shape[0]):
            current_state_fk = get_fk_solution(joint_pos[i, :7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk)  # (15, 7)

        skip = int(self.wm_args.policy_skip_step)
        pred_step = int(self.wm_args.pred_step)
        state_fk_skip = state_fk[::skip][:pred_step]  # (5, 7)
        joint_pos_skip = joint_pos[::skip][:pred_step]  # (5, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)  # (5, 8) add gripper

        return action_chunk, joint_pos_skip, state_fk_skip

    def _forward_world_model(
        self,
        action_cond: np.ndarray,
        current_latent: torch.Tensor,
        his_cond: torch.Tensor,
        instruction: str,
    ):
        """Match rollout_interact_pi.py forward_wm decode path.

        Args:
            action_cond: (6+pred_step, 7) cartesian action condition (history+future)
            current_latent: (1, 4, 72, 40) latent
            his_cond: (1, 6, 4, 72, 40) latent history condition
        Returns:
            next_latent: (1, 4, 72, 40)
            camera_views: np.uint8 array (3, pred_step, H, W, 3)
            latents_split: torch.Tensor (3, pred_step, 4, 24, 40)
        """
        import einops

        args = self.wm_args
        pipeline = self.wm_model.pipeline

        # Normalize & encode action/text
        action_norm = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_tensor = torch.from_numpy(action_norm).to(dtype=self.dtype, device=self.wm_device).unsqueeze(0)  # (1, 11, 7)

        with torch.no_grad():
            if instruction is not None:
                text_token = self.wm_model.action_encoder(
                    action_tensor, instruction, self.wm_model.tokenizer, self.wm_model.text_encoder
                )
            else:
                text_token = self.wm_model.action_encoder(action_tensor)

            _, video_pred = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=current_latent.to(self.wm_device),  # latent input branch
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond.to(self.wm_device),
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

        # Split into 3 views: (1, 5, 4, 72, 40) -> (3, 5, 4, 24, 40)
        latents_split = einops.rearrange(video_pred, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        pred_step = int(args.pred_step)
        next_latent = torch.cat([latents_split[v, pred_step - 1] for v in range(3)], dim=1).unsqueeze(0)  # (1, 4, 72, 40)

        # Decode (chunked) exactly like omni_ctrl/rollout/world_model_wrapper.py
        bsz, frame_num = latents_split.shape[:2]  # (3, 5)
        x = latents_split.flatten(0, 1)  # (15, 4, 24, 40)
        decoded_chunks = []
        decode_kwargs = {}
        decode_chunk_size = int(args.decode_chunk_size)
        for i in range(0, x.shape[0], decode_chunk_size):
            chunk = x[i:i + decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_chunks.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_chunks, dim=0)  # (15, 3, H, W)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])  # (3, 5, 3, H, W)
        videos = (videos / 2.0 + 0.5).clamp(0, 1)
        if videos.dtype == torch.bfloat16:
            videos = videos.float()
        videos_np = videos.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # (3, 5, H, W, 3)
        videos_np = (videos_np * 255).astype(np.uint8)

        return next_latent, videos_np, latents_split

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Normalize data to [-1, 1] range"""
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using loaded statistics"""
        if self.state_p01 is None or self.state_p99 is None:
            return state
        return self.normalize_bound(state, self.state_p01, self.state_p99)

    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state to original scale"""
        if self.state_p01 is None or self.state_p99 is None:
            return normalized_state
        return (normalized_state + 1) / 2 * (self.state_p99 - self.state_p01 + 1e-8) + self.state_p01

    def _convert_actions_to_cartesian(self, action_chunk: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert action chunk to cartesian poses

        Args:
            action_chunk: (15, 8) joint velocities + gripper
            current_state: (8,) current joint positions + gripper

        Returns:
            cartesian_poses: (15, 7) cartesian poses (xyz + euler + gripper)
        """
        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        joint_vel = action_chunk[:, :7][idx]
        gripper_pos = action_chunk[:, 7:][idx]
        gripper_max = 0.75
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        current_joint = current_state[None, :][:, :7]
        current_gripper = current_state[None, :][:, 7:]

        if hasattr(self, 'dynamics_model') and self.dynamics_model is not None:
            with torch.no_grad():
                joint_future = self.dynamics_model(current_joint, joint_vel, None, training=False)

            joint_pos_all = np.concatenate([current_joint, joint_future], axis=0)[:15]
            gripper_pos_all = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]
        else:
            joint_pos_all = np.tile(current_joint, (15, 1))
            gripper_pos_all = gripper_pos

        cartesian_poses = []
        for i in range(joint_pos_all.shape[0]):
            fk_result = get_fk_solution(joint_pos_all[i, :7])
            xyz = fk_result[:3, 3]
            rotation_matrix = fk_result[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            cartesian_pose = np.concatenate([xyz, euler, gripper_pos_all[i]], axis=0)
            cartesian_poses.append(cartesian_pose)

        return np.array(cartesian_poses)

    def generate_next_frames(
        self,
        current_image: np.ndarray,
        action_sequence: np.ndarray,
        instruction: str,
        history: torch.Tensor = None,
    ) -> tuple:
        """Generate next frames using world model

        Args:
            current_image: Current RGB image (H, W, 3) - stacked 3 views
            action_sequence: Cartesian pose sequence (num_frames+num_history, 7)
            instruction: Task instruction text
            history: History latents (1, num_history, 4, 72, 40) or None

        Returns:
            Tuple of (all_frames, predicted_latents)
                - all_frames: not used (None)
                - predicted_latents: list of 3 tensors, each (num_frames, 4, h, w)
        """
        args = self.wm_args
        import einops

        # Convert image to tensor for pipeline
        image_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.wm_device).to(torch.float32) / 255.0

        # Prepare action condition
        # Align action length with the frames that will be fed into the UNet
        # If history is provided, UNet sees (num_frames + current_history_len)
        his_len = 0 if history is None else int(history.shape[1])
        target_len = args.num_frames + his_len
        if action_sequence.shape[0] != target_len:
            if action_sequence.shape[0] > target_len:
                action_sequence = action_sequence[:target_len]
            else:
                pad = np.repeat(action_sequence[-1:], target_len - action_sequence.shape[0], axis=0)
                action_sequence = np.concatenate([action_sequence, pad], axis=0)

        action_cond = self.normalize_bound(action_sequence, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.wm_device).to(self.dtype)

        # Generate using world model
        with torch.no_grad():
            if instruction is not None:
                text_token = self.wm_model.action_encoder(action_cond, instruction, self.wm_model.tokenizer, self.wm_model.text_encoder)
            else:
                text_token = self.wm_model.action_encoder(action_cond)

            pipeline = self.wm_model.pipeline

            frames, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_tensor,
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

        # Rearrange latents to separate 3 views
        # latents shape: (b, f, c, h*3, w) -> (3, f, c, h, w)
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # Return as list of 3 views
        predicted_latents = [latents[i] for i in range(3)]

        return None, predicted_latents

    def load_episode_videos(self, data_dir: Path, episode_idx: int):
        """加载一个episode的初始帧（3个摄像头）"""
        videos_dir = data_dir / "videos" / "chunk-000"

        video_paths = {
            'exterior_1': videos_dir / "observation.images.exterior_1_left" / f"episode_{episode_idx:06d}.mp4",
            'exterior_2': videos_dir / "observation.images.exterior_2_left" / f"episode_{episode_idx:06d}.mp4",
            'wrist': videos_dir / "observation.images.wrist_left" / f"episode_{episode_idx:06d}.mp4",
        }

        # 检查文件是否存在
        for name, path in video_paths.items():
            if not path.exists():
                return None, f"Missing {name} video: {path}"

        # 只读取第一帧
        frames = {
            'exterior_1': [],
            'exterior_2': [],
            'wrist': [],
        }

        for name, path in video_paths.items():
            try:
                # Prefer PyAV (more reliable for AV1 mp4 than OpenCV/mediapy in some environments)
                import av
                container = av.open(str(path))
                stream = container.streams.video[0]
                first_frame = None
                for frame in container.decode(stream):
                    first_frame = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                    break
                container.close()
                if first_frame is None:
                    raise RuntimeError("No frames decoded")
                frames[name].append(first_frame)
                continue
            except Exception as e:
                # Fallback to mediapy/decord (legacy)
                try:
                    import mediapy
                    video = mediapy.read_video(str(path))
                    first_frame = video[0]  # (H, W, 3)
                    if first_frame.dtype != np.uint8:
                        first_frame = (first_frame * 255).astype(np.uint8)
                    frames[name].append(first_frame)
                    continue
                except Exception as e2:
                    try:
                        from decord import VideoReader, cpu
                        vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
                        try:
                            first_frame = vr.get_batch([0]).asnumpy()[0]
                        except Exception:
                            first_frame = vr.get_batch([0]).numpy()[0]
                        if first_frame.dtype != np.uint8:
                            first_frame = (first_frame * 255).astype(np.uint8)
                        frames[name].append(first_frame)
                        continue
                    except Exception as e3:
                        return None, (
                            f"Failed to read {name} first frame: "
                            f"pyav error: {e}, mediapy error: {e2}, decord error: {e3}"
                        )

        return frames, None

    def load_episode_parquet(self, data_dir: Path, episode_idx: int):
        """加载episode的parquet数据（包含状态和动作）"""
        parquet_path = data_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"

        if not parquet_path.exists():
            return None, f"Parquet file not found: {parquet_path}"

        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return df, None
        except Exception as e:
            return None, f"Failed to read parquet: {e}"

    def infer_episode(self, frames: dict, initial_state: np.ndarray, instruction: str):
        """对一个episode进行推理"""
        num_frames = len(frames['exterior_1'])

        predictions = []

        for frame_idx in range(num_frames):
            # 获取当前帧
            image1 = frames['exterior_1'][frame_idx]
            image2 = frames['wrist'][frame_idx]

            # Resize
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

            image1 = torch.nn.functional.interpolate(
                image1.permute(2, 0, 1).unsqueeze(0).float(),
                size=(180, 320), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

            image2 = torch.nn.functional.interpolate(
                image2.permute(2, 0, 1).unsqueeze(0).float(),
                size=(180, 320), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

            # 准备policy输入
            policy_input = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
                "observation/joint_position": initial_state[:7],
                "observation/gripper_position": initial_state[-1:],
                "prompt": instruction,
            }

            # 推理
            action_chunk = self.policy.infer(policy_input)["actions"]

            predictions.append({
                'frame_idx': frame_idx,
                'action': action_chunk.tolist(),
            })

        return predictions

    def infer_and_generate_episode(
        self,
        frames: dict,
        initial_state: np.ndarray,
        instruction: str,
        max_steps: int = 50,
    ):
        """对一个episode进行推理并使用世界模型生成新的轨迹

        Args:
            frames: Initial frames from the episode
            initial_state: Initial robot state (joint positions + gripper)
            instruction: Task instruction
            max_steps: Maximum number of steps to generate

        Returns:
            Dictionary containing predictions and generated data
        """
        if not self.generate_data:
            return self.infer_episode(frames, initial_state, instruction)

        print(f"Generating episode with {max_steps} max steps...")

        # Initialize generated trajectory
        generated_frames = {
            'exterior_1': [frames['exterior_1'][0]],
            'exterior_2': [frames['exterior_2'][0]],
            'wrist': [frames['wrist'][0]],
        }
        generated_states = [initial_state.astype(np.float32)]
        # One action per saved frame (aligned with generated_states). Start with a dummy action for t=0.
        generated_actions = [np.zeros((8,), dtype=np.float32)]

        # Initialize current RGB observations
        current_obs = {
            "exterior_1": frames["exterior_1"][0],
            "exterior_2": frames["exterior_2"][0],
            "wrist": frames["wrist"][0],
        }

        # Build initial latent from stacked RGB (matches official WM latent path)
        stacked_rgb = np.concatenate([current_obs["exterior_1"], current_obs["exterior_2"], current_obs["wrist"]], axis=0)
        first_latent = self._encode_stacked_rgb_to_latent(stacked_rgb)  # (1, 4, 72, 40)

        # Initial FK pose (xyz + euler + gripper)
        fk0 = get_fk_solution(initial_state[:7])
        xyz0 = fk0[:3, 3]
        r0 = R.from_matrix(fk0[:3, :3])
        euler0 = r0.as_euler("xyz")
        eef0 = np.concatenate([xyz0, euler0, initial_state[-1:]], axis=0).astype(np.float32)  # (7,)

        # History buffers: follow rollout_interact_pi.py (num_history*4 = 24)
        his_len = int(self.wm_args.num_history) * 4
        his_cond = [first_latent] * his_len  # each (1, 4, 72, 40)
        his_joint = [initial_state.astype(np.float32)[None, :]] * his_len  # each (1, 8)
        his_eef = [eef0[None, :]] * his_len  # each (1, 7)

        print(f"Starting generation loop for {max_steps} steps (pred_step={self.wm_args.pred_step})...")

        pred_step = int(self.wm_args.pred_step)
        history_idx = list(self.wm_args.history_idx)
        log_steps = os.environ.get("CTRLWORLD_STEP_LOG", "0") == "1"

        for step_idx in range(max_steps):
            try:
                if log_steps:
                    print(f"[episode] step {step_idx}/{max_steps} begin", flush=True)
                current_joint = his_joint[-1][0]  # (8,)
                if log_steps:
                    t0 = time.time()
                    print(f"[episode] step {step_idx}: policy.infer ...", flush=True)
                action_chunk, joint_pos_skip, cartesian_pose = self._forward_policy_to_cartesian(
                    current_obs["exterior_1"],
                    current_obs["wrist"],
                    current_joint,
                    text=instruction,
                )
                if log_steps:
                    print(f"[episode] step {step_idx}: policy.infer done ({time.time() - t0:.2f}s)", flush=True)
                # We will expand action_chunk into per-frame actions below.

                # Build WM action_cond and history latent exactly like rollout_interact_pi.py
                hist_actions = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (6, 7)
                action_cond = np.concatenate([hist_actions, cartesian_pose], axis=0)  # (11, 7)
                his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, 6, 4, 72, 40)
                current_latent = his_cond[-1]  # (1, 4, 72, 40)

                if log_steps:
                    t0 = time.time()
                    print(f"[episode] step {step_idx}: world_model ...", flush=True)
                next_latent, camera_views, latents_split = self._forward_world_model(
                    action_cond=action_cond,
                    current_latent=current_latent,
                    his_cond=his_latent,
                    instruction=instruction if self.wm_args.text_cond else None,
                )
                if log_steps:
                    print(f"[episode] step {step_idx}: world_model done ({time.time() - t0:.2f}s)", flush=True)

                # Append frames/states/actions for saving: follow rollout_interact_pi.py (save pred_step-1 frames)
                # We align states with the predicted joint positions at 1..pred_step-1 (exclude current and the final overlap frame).
                skip = int(self.wm_args.policy_skip_step)
                for f in range(pred_step - 1):
                    generated_frames["exterior_1"].append(camera_views[0, f])
                    generated_frames["exterior_2"].append(camera_views[1, f])
                    generated_frames["wrist"].append(camera_views[2, f])
                    # State/action for this generated frame
                    generated_states.append(joint_pos_skip[f + 1].astype(np.float32))
                    action_idx = min(f * skip, action_chunk.shape[0] - 1)
                    generated_actions.append(action_chunk[action_idx].astype(np.float32))

                # Update current obs to the last predicted frame (not appended, used as next-step input)
                current_obs = {
                    "exterior_1": camera_views[0, pred_step - 1],
                    "exterior_2": camera_views[1, pred_step - 1],
                    "wrist": camera_views[2, pred_step - 1],
                }

                # Push to history buffers (same as rollout_interact_pi.py)
                his_joint.append(joint_pos_skip[pred_step - 1][None, :].astype(np.float32))
                his_eef.append(cartesian_pose[pred_step - 1][None, :].astype(np.float32))
                his_cond.append(next_latent.detach())

                if (step_idx + 1) % 10 == 0:
                    print(f"  Step {step_idx + 1}/{max_steps} completed")

            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Generated {len(generated_frames['exterior_1'])} frames total")

        result = {
            'predictions': [{'frame_idx': i, 'action': act.tolist()} for i, act in enumerate(generated_actions)],
            'generated_frames': generated_frames,
            'generated_states': generated_states,
            'num_generated_frames': len(generated_frames['exterior_1']),
        }

        return result

    def run_inference(
        self,
        data_dir: Path,
        output_dir: Path,
        start_episode: int = 0,
        end_episode: int = 1000,
        save_every: int = 10,
        max_gen_steps: int = 50,
    ):
        """对多个episode进行推理"""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        errors = []

        print(f"Running inference on episodes {start_episode} to {end_episode-1}")
        print(f"Output directory: {output_dir}")
        print()

        progress_path = output_dir / "progress.json"
        _write_json_atomic(
            progress_path,
            {
                "stage": "start",
                "start_episode": start_episode,
                "end_episode": end_episode,
                "ts": time.time(),
            },
        )

        for episode_idx in tqdm(range(start_episode, end_episode), desc="Inference"):
            try:
                _write_json_atomic(
                    progress_path,
                    {
                        "episode_idx": episode_idx,
                        "stage": "load_videos",
                        "ts": time.time(),
                    },
                )
                # 加载视频
                frames, error = self.load_episode_videos(data_dir, episode_idx)
                if error:
                    errors.append({
                        'episode_idx': episode_idx,
                        'error': error,
                        'stage': 'load_videos'
                    })
                    _write_json_atomic(output_dir / "errors.json", errors)
                    continue

                _write_json_atomic(
                    progress_path,
                    {
                        "episode_idx": episode_idx,
                        "stage": "load_parquet",
                        "ts": time.time(),
                    },
                )
                # 加载parquet数据
                df, error = self.load_episode_parquet(data_dir, episode_idx)
                if error:
                    errors.append({
                        'episode_idx': episode_idx,
                        'error': error,
                        'stage': 'load_parquet'
                    })
                    _write_json_atomic(output_dir / "errors.json", errors)
                    continue

                # 获取初始状态和指令
                initial_joint = df.iloc[0]['observation.state.joint_position']
                initial_gripper = df.iloc[0]['observation.state.gripper_position']

                if isinstance(initial_joint, list):
                    initial_joint = np.array(initial_joint, dtype=np.float32)
                if isinstance(initial_gripper, (int, float)):
                    initial_gripper = np.array([initial_gripper], dtype=np.float32)
                elif isinstance(initial_gripper, list):
                    initial_gripper = np.array(initial_gripper, dtype=np.float32)

                initial_state = np.concatenate([initial_joint[:7], initial_gripper])

                instruction = df.iloc[0].get('language_instruction', 'No instruction')

                # 推理或生成数据
                if self.generate_data:
                    _write_json_atomic(
                        progress_path,
                        {
                            "episode_idx": episode_idx,
                            "stage": "infer_and_generate_episode",
                            "ts": time.time(),
                        },
                    )
                    result_data = self.infer_and_generate_episode(
                        frames,
                        initial_state,
                        instruction,
                        max_steps=max_gen_steps,
                    )

                    result = {
                        'episode_idx': episode_idx,
                        'instruction': instruction,
                        'num_frames': result_data.get('num_generated_frames', 0),
                        'predictions': result_data['predictions'],
                    }

                    # 保存生成的视频和数据
                    if 'generated_frames' in result_data:
                        _write_json_atomic(
                            progress_path,
                            {
                                "episode_idx": episode_idx,
                                "stage": "save_generated_episode",
                                "ts": time.time(),
                            },
                        )
                        self._save_generated_episode(
                            result_data,
                            output_dir,
                            episode_idx,
                            instruction
                        )
                    else:
                        print(f"Warning: Episode {episode_idx} - no generated_frames in result_data")
                else:
                    _write_json_atomic(
                        progress_path,
                        {
                            "episode_idx": episode_idx,
                            "stage": "infer_episode",
                            "ts": time.time(),
                        },
                    )
                    predictions = self.infer_episode(frames, initial_state, instruction)
                    result = {
                        'episode_idx': episode_idx,
                        'instruction': instruction,
                        'num_frames': len(frames['exterior_1']),
                        'predictions': predictions,
                    }

                results.append(result)

                # 定期保存推理结果
                if len(results) % save_every == 0:
                    self._save_results(results[-save_every:], output_dir, episode_idx)

                # Flush live stats/errors so you can see progress even if killed later.
                stats_live = {
                    'total_episodes': end_episode - start_episode,
                    'successful': len(results),
                    'failed': len(errors),
                    'success_rate': len(results) / (end_episode - start_episode) if (end_episode - start_episode) > 0 else 0.0,
                    'last_episode_idx': episode_idx,
                }
                _write_json_atomic(output_dir / "stats.json", stats_live)
                if errors:
                    _write_json_atomic(output_dir / "errors.json", errors)

                _write_json_atomic(
                    progress_path,
                    {
                        "episode_idx": episode_idx,
                        "stage": "done",
                        "ts": time.time(),
                    },
                )

            except Exception as e:
                import traceback
                errors.append({
                    'episode_idx': episode_idx,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'stage': 'inference'
                })
                _write_json_atomic(output_dir / "errors.json", errors)
                continue

        # 保存剩余结果
        remaining = len(results) % save_every
        if remaining > 0:
            self._save_results(results[-remaining:], output_dir, end_episode-1)

        # 保存错误日志
        if errors:
            error_path = output_dir / "errors.json"
            with open(error_path, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"\n✗ {len(errors)} episodes failed, errors saved to {error_path}")

        # 保存统计
        stats = {
            'total_episodes': end_episode - start_episode,
            'successful': len(results),
            'failed': len(errors),
            'success_rate': len(results) / (end_episode - start_episode),
        }

        stats_path = output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Inference complete!")
        print(f"  Successful: {len(results)}/{end_episode - start_episode}")
        print(f"  Failed: {len(errors)}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Results saved to: {output_dir}")

        return results, errors

    def _save_generated_episode(
        self,
        result_data: dict,
        output_dir: Path,
        episode_idx: int,
        instruction: str,
    ):
        """保存生成的episode为DROID格式

        Args:
            result_data: Generated episode data
            output_dir: Output directory
            episode_idx: Episode index
            instruction: Task instruction
        """
        print(f"Saving episode {episode_idx}...")

        # Create subdirectories for generated data
        generated_dir = output_dir / "generated_episodes"
        video_dir = generated_dir / "videos" / "chunk-000"
        data_dir = generated_dir / "data" / "chunk-000"

        for d in [video_dir, data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        generated_frames = result_data['generated_frames']
        generated_states = result_data['generated_states']

        required_views = {"exterior_1", "exterior_2", "wrist"}
        missing_views = required_views - set(generated_frames.keys())
        if missing_views:
            raise KeyError(f"Missing generated_frames views: {sorted(missing_views)}")

        print(f"  Generated {len(generated_frames['exterior_1'])} frames")

        # Marker file: helps debugging if saving gets stuck.
        marker_path = generated_dir / f"episode_{episode_idx:06d}.saving.json"
        _write_json_atomic(
            marker_path,
            {
                "episode_idx": episode_idx,
                "stage": "start",
                "ts": time.time(),
            },
        )

        # Create all view dirs up-front (so you can see expected structure immediately).
        view_dir_map = {
            "exterior_1": video_dir / "observation.images.exterior_1_left",
            "exterior_2": video_dir / "observation.images.exterior_2_left",
            "wrist": video_dir / "observation.images.wrist_left",
        }
        for p in view_dir_map.values():
            p.mkdir(parents=True, exist_ok=True)

        # Save parquet (best-effort). If it fails, still proceed to videos.
        parquet_data = {
            # Keep both the original DROID-style columns and our legacy split columns for compatibility.
            # DROID-style columns (preferred for downstream training/analysis):
            "observation.state": [],
            "action.joint_velocity": [],
            "action.gripper_position": [],
            # Legacy columns (kept for backward compatibility with earlier generations):
            'observation.state.joint_position': [],
            'observation.state.gripper_position': [],
            'action': [],
            'language_instruction': [],
        }

        predictions = result_data.get('predictions', []) or []
        inferred_action_dim = 8
        for pred in predictions:
            a = pred.get("action") if isinstance(pred, dict) else None
            if a is None:
                continue
            if torch.is_tensor(a):
                inferred_action_dim = int(a.numel())
                break
            if isinstance(a, np.ndarray):
                inferred_action_dim = int(a.reshape(-1).shape[0])
                break
            if isinstance(a, (list, tuple)) and len(a) > 0:
                # If nested, use inner length.
                if len(a) == 1 and isinstance(a[0], (list, tuple, np.ndarray)):
                    inferred_action_dim = int(len(a[0]))
                else:
                    inferred_action_dim = int(len(a))
                break

        for i, state in enumerate(generated_states):
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.shape[0] < 8:
                state = np.pad(state, (0, 8 - state.shape[0]))
            elif state.shape[0] > 8:
                state = state[:8]

            parquet_data["observation.state"].append(state.tolist())
            parquet_data['observation.state.joint_position'].append(state[:7].tolist())
            parquet_data['observation.state.gripper_position'].append(state[7:8].tolist())

            action_raw = None
            if i < len(predictions):
                pred = predictions[i]
                if isinstance(pred, dict):
                    action_raw = pred.get("action")

            action_list = _normalize_action_value(action_raw, max(8, inferred_action_dim))
            # Also provide a fixed 8-dim action vector (joint_vel[7] + gripper_pos[1]) for DROID compatibility.
            action8 = _normalize_action_value(action_raw, 8)

            parquet_data['action'].append(action_list)
            parquet_data["action.joint_velocity"].append(action8[:7])
            parquet_data["action.gripper_position"].append(float(action8[7]))

            parquet_data['language_instruction'].append(instruction)

        df = pd.DataFrame(parquet_data)
        parquet_file = data_dir / f"episode_{episode_idx:06d}.parquet"
        try:
            df.to_parquet(parquet_file)
            print(f"  Saved parquet: {parquet_file}")
        except Exception as e:
            _write_json_atomic(
                marker_path,
                {
                    "episode_idx": episode_idx,
                    "stage": "parquet_failed",
                    "parquet_file": str(parquet_file),
                    "error": str(e),
                    "ts": time.time(),
                },
            )
            print(f"  Warning: failed to save parquet {parquet_file}: {e}")

        # Save videos for each camera view
        for view_name in ['exterior_1', 'exterior_2', 'wrist']:
            frames = generated_frames[view_name]
            fps = 10

            if len(frames) == 0:
                raise RuntimeError(f"Episode {episode_idx}: generated_frames[{view_name}] is empty")

            # Ensure at least 1s duration to avoid 0s videos in some players
            min_frames = max(1, int(round(fps * 1.0)))
            if len(frames) > 0 and len(frames) < min_frames:
                pad_count = min_frames - len(frames)
                frames = frames + [frames[-1]] * pad_count

            video_path = view_dir_map[view_name]

            video_file = video_path / f"episode_{episode_idx:06d}.mp4"

            try:
                _write_json_atomic(
                    marker_path,
                    {
                        "episode_idx": episode_idx,
                        "stage": f"write_video:{view_name}",
                        "video_file": str(video_file),
                        "num_frames": len(frames),
                        "ts": time.time(),
                    },
                )
                _write_video_best_effort(video_file, frames, fps=fps)
                print(f"  Saved {view_name} video: {video_file}")
            except Exception as e:
                # Optional: if video writing fails, at least dump frames for debugging.
                _maybe_save_png_fallback(video_path, frames, episode_idx)
                raise RuntimeError(f"Failed to save {view_name} video to {video_file}: {e}") from e

        _write_json_atomic(
            marker_path,
            {
                "episode_idx": episode_idx,
                "stage": "done",
                "ts": time.time(),
            },
        )

    def _save_results(self, results: list, output_dir: Path, last_episode_idx: int):
        """保存推理结果"""
        output_path = output_dir / f"inference_results_up_to_{last_episode_idx:06d}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run inference on droid_data episodes')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data',
        help='Path to droid_data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference_results',
        help='Output directory for inference results'
    )
    parser.add_argument(
        '--pi-ckpt',
        type=str,
        default='/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid',
        help='Policy checkpoint path'
    )
    parser.add_argument(
        '--policy-type',
        type=str,
        default='pi05',
        help='Policy type (pi05, pi0fast, pi0)'
    )
    parser.add_argument(
        '--start-episode',
        type=int,
        default=0,
        help='Start episode index'
    )
    parser.add_argument(
        '--end-episode',
        type=int,
        default=1000,
        help='End episode index (exclusive)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save results every N episodes'
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Enable data generation using world model'
    )
    parser.add_argument(
        '--wm-ckpt',
        type=str,
        default=None,
        help='World model checkpoint path (required if --generate-data is set)'
    )
    parser.add_argument(
        '--svd-model-path',
        type=str,
        default='stabilityai/stable-video-diffusion-img2vid',
        help='SVD model path'
    )
    parser.add_argument(
        '--clip-model-path',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model path'
    )
    parser.add_argument(
        '--data-stat-path',
        type=str,
        default=None,
        help='Data statistics path for normalization'
    )
    parser.add_argument(
        '--action-adapter-path',
        type=str,
        default=None,
        help='Action adapter checkpoint path'
    )
    parser.add_argument(
        '--max-gen-steps',
        type=int,
        default=50,
        help='Maximum steps to generate per episode'
    )

    args = parser.parse_args()

    # 创建推理器
    inferencer = DroidDataInference(
        pi_ckpt=args.pi_ckpt,
        policy_type=args.policy_type,
        device=args.device,
        generate_data=args.generate_data,
        wm_ckpt=args.wm_ckpt,
        svd_model_path=args.svd_model_path,
        clip_model_path=args.clip_model_path,
        data_stat_path=args.data_stat_path,
        action_adapter_path=args.action_adapter_path,
    )

    # 运行推理
    results, errors = inferencer.run_inference(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        save_every=args.save_every,
        max_gen_steps=args.max_gen_steps,
    )

    print("\nInference complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
