"""World model wrapper for Omni-Ctrl framework.

This module wraps the existing Ctrl-World model to provide
a clean interface for the Omni-Ctrl orchestrator.
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

# Add Ctrl-World to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import wm_args
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


class WorldModelWrapper:
    """Wrapper around existing Ctrl-World model.

    Reuses the trained world model without any modifications.
    Provides a simplified interface for the orchestrator.
    """

    def __init__(self, config):
        """Initialize world model wrapper.

        Args:
            config: RolloutConfig instance
        """
        print("Initializing World Model Wrapper...")

        # Convert config to wm_args format
        args = wm_args()
        args.ckpt_path = config.wm_ckpt
        args.svd_model_path = config.svd_model_path
        args.clip_model_path = config.clip_model_path
        args.data_stat_path = config.data_stat_path
        args.num_frames = config.pred_step
        args.num_history = 6
        args.num_inference_steps = 50  # Reduced from 50 to save memory
        args.guidance_scale = 2.0
        args.seed = 42
        args.device = config.device
        args.dtype = torch.bfloat16  # Use bfloat16 for inference

        # Load world model
        print(f"Loading Ctrl-World model from {config.wm_ckpt}...")
        self.model = CrtlWorld(args)
        # Load checkpoint to CPU first to avoid OOM
        checkpoint = torch.load(config.wm_ckpt, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        del checkpoint  # Free memory
        torch.cuda.empty_cache()
        # Move model to GPU
        self.model.to(config.device)
        self.model.eval()
        self.args = args
        self.device = config.device
        # Get dtype from the loaded model
        self.dtype = next(self.model.parameters()).dtype
        print(f"World model loaded (dtype: {self.dtype})")

        # Load normalization stats
        print(f"Loading normalization stats from {config.data_stat_path}...")
        with open(args.data_stat_path) as f:
            self.stat = json.load(f)

        print("World Model Wrapper initialized successfully")

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using dataset statistics."""
        low = np.array(self.stat["state_01"])
        high = np.array(self.stat["state_99"])
        eps = 1e-8
        action_norm = 2 * (action - low) / (high - low + eps) - 1
        return np.clip(action_norm, -1.0, 1.0)

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to original scale.

        Args:
            action_norm: Normalized action (7,)

        Returns:
            Denormalized action (7,)
        """
        action = action_norm.copy()
        low = np.array(self.stat["state_01"])
        high = np.array(self.stat["state_99"])
        action = (action_norm + 1) / 2 * (high - low) + low
        return action

    def step(
        self,
        action_seq: np.ndarray,
        current_latent: torch.Tensor,
        history_latents: List[torch.Tensor],
        history_actions: List[np.ndarray],
        text: str,
    ) -> Tuple[torch.Tensor, list, np.ndarray]:
        """Execute one step in world model.

        Following the same pipeline as rollout_interact_pi.py forward_wm method (lines 151-226).

        Args:
            action_seq: Action sequence (pred_step, 7) in cartesian space
            current_latent: Current latent state (4, 72, 40) or (1, 4, 72, 40)
            history_latents: List of history latents, each with shape (4, 72, 40) or (1, 4, 72, 40)
            history_actions: List of history actions, each with shape (7,)
            text: Task instruction text

        Returns:
            Tuple of:
                - next_latent: Predicted next latent (1, 4, 72, 40) - with batch dim
                - next_frames: List of decoded frames
                - camera_views: Camera views array (3, pred_step, H, W, 3)
        """
        # Normalize all latents to have batch dimension (1, 4, 72, 40)
        if current_latent.ndim == 3:
            current_latent = current_latent.unsqueeze(0)

        normalized_history = []
        for lat in history_latents:
            if lat.ndim == 3:
                normalized_history.append(lat.unsqueeze(0))
            else:
                normalized_history.append(lat)
        history_latents = normalized_history
        # Following rollout_interact_pi.py: use history_idx = [0,0,-12,-9,-6,-3] (line 368)
        history_idx = [0, 0, -12, -9, -6, -3]

        # Ensure we have enough history (should be 24)
        if len(history_actions) < 24:
            while len(history_actions) < 24:
                history_actions.insert(0, history_actions[0])
        if len(history_latents) < 24:
            while len(history_latents) < 24:
                history_latents.insert(0, history_latents[0])

        # Build action condition: select history actions using history_idx - same as rollout_interact_pi.py line 369
        hist_actions = np.array([history_actions[idx] for idx in history_idx])  # (6, 7)

        # Normalize all actions - same as rollout_interact_pi.py line 157
        action_norm = np.array([self.normalize_action(a) for a in action_seq])  # (pred_step, 7)
        hist_norm = np.array([self.normalize_action(a) for a in hist_actions])  # (6, 7)

        # Concatenate history + future - same as rollout_interact_pi.py line 370
        action_cond = np.concatenate([hist_norm, action_norm], axis=0)  # (6+pred_step, 7)
        action_tensor = torch.from_numpy(action_cond).to(dtype=self.dtype, device=self.device).unsqueeze(0)  # (1, 6+pred_step, 7)

        # Prepare history latents: select using history_idx - same as rollout_interact_pi.py line 371
        # IMPORTANT: Each element in history_latents is (1, 4, 72, 40), so we use torch.cat not torch.stack
        his_cond = torch.cat([history_latents[idx].to(self.device) for idx in history_idx], dim=0).unsqueeze(0)  # (1, 6, 4, 72, 40)

        # World model forward - same as rollout_interact_pi.py lines 164-189
        with torch.no_grad():
            # Encode action and text into text tokens - same as rollout_interact_pi.py lines 166-169
            if text is not None:
                text_token = self.model.action_encoder(
                    action_tensor,
                    text,
                    self.model.tokenizer,
                    self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_tensor)

            # Call pipeline - same as rollout_interact_pi.py lines 170-189
            pipeline = self.model.pipeline
            _, video_pred = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=current_latent.to(self.device),  # current_latent is already (1, 4, 72, 40)
                text=text_token,
                width=self.args.width,
                height=int(self.args.height * 3),
                num_frames=self.args.num_frames,
                history=his_cond,  # his_cond is already (1, 6, 4, 72, 40)
                num_inference_steps=self.args.num_inference_steps,
                decode_chunk_size=self.args.decode_chunk_size,
                max_guidance_scale=self.args.guidance_scale,
                fps=self.args.fps,
                motion_bucket_id=self.args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )

        # video_pred shape: (1, num_frames, 4, 72, 40) - before rearrange

        # CRITICAL: Rearrange before decoding - same as rollout_interact_pi.py line 190
        # The latent has 3 views concatenated in height dim: (1, 5, 4, 72, 40)
        # Need to split into 3 separate views: (3, 5, 4, 24, 40)
        import einops
        latents_split = einops.rearrange(video_pred, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
        # Now: (3, 5, 4, 24, 40) - 3 views, 5 frames each

        # Build next_latent for history update - same as rollout_interact_pi.py line 380
        # Concatenate the last frame from all 3 views
        pred_step = self.args.num_frames
        next_latent = torch.cat([latents_split[v, pred_step-1] for v in range(3)], dim=1).unsqueeze(0)  # (1, 4, 72, 40)

        # Decode following rollout_interact_pi.py pattern (lines 208-220)
        decoded_video = []
        bsz, frame_num = latents_split.shape[:2]  # bsz=3, frame_num=5
        x = latents_split.flatten(0, 1)  # (15, 4, 24, 40) - flatten batch and frames
        decode_kwargs = {}
        decode_chunk_size = self.args.decode_chunk_size

        for i in range(0, x.shape[0], decode_chunk_size):
            chunk = x[i:i+decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)

        videos = torch.cat(decoded_video, dim=0)  # (15, 3, H, W)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])  # (3, 5, 3, H, W)
        videos = ((videos / 2.0 + 0.5).clamp(0, 1))  # Normalize to [0, 1]

        # Convert to numpy: (3, 5, 3, H, W) -> (3, 5, H, W, 3) - same as rollout_interact_pi.py line 208
        videos_np = videos.cpu().numpy().transpose(0, 1, 3, 4, 2)  # (3 views, 5 frames, H, W, 3)
        videos_np = (videos_np * 255).astype(np.uint8)  # Convert to uint8

        # Concatenate 3 views horizontally (for video saving) - same as rollout_interact_pi.py line 224
        # videos_np shape: (3, 5, H, W, 3)
        # Concatenate along width axis to get: (5, H, W*3, 3)
        videos_concat = np.concatenate([videos_np[i] for i in range(bsz)], axis=-2)  # (5, H, W*3, 3)

        # Convert concatenated to list of frames (for video saving)
        next_frames = [videos_concat[i] for i in range(frame_num)]

        # Also return separate camera views (for policy input)
        # videos_np shape: (3, 5, H, W, 3) - 3 cameras, 5 frames each
        camera_views = videos_np  # Keep as (3, 5, H, W, 3)

        return next_latent, next_frames, camera_views

    def decode_latent(self, latent: torch.Tensor) -> np.ndarray:
        """Decode a single latent to RGB frame.

        Args:
            latent: Latent tensor (4, H, W)

        Returns:
            RGB frame (H, W, 3) in [0, 1] range
        """
        with torch.no_grad():
            frame = self.model.pipeline.decode_latents(
                latent.unsqueeze(0).unsqueeze(0).to(self.device),
                num_frames=1
            )
        # Normalize from [-1, 1] to [0, 1] (VAE output range)
        # Extract all 3 color channels: [batch=0, all_channels, frame=0, :, :]
        frame_normalized = (frame[0, :, 0] / 2.0 + 0.5).clamp(0, 1)
        frame = frame_normalized.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        return frame

    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Encode RGB frame to latent.

        Args:
            frame: RGB frame (H, W, 3) in [0, 1] range

        Returns:
            Latent tensor (4, H, W)
        """
        # Convert to torch tensor
        frame_tensor = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            # Use VAE encoder directly
            latent = self.model.vae.encode(frame_tensor).latent_dist.sample()
            latent = latent * self.model.vae.config.scaling_factor

        return latent[0]

    def get_stats(self):
        """Get world model statistics.

        Returns:
            Dictionary with model info
        """
        return {
            "device": self.device,
            "num_frames": self.args.num_frames,
            "num_history": self.args.num_history,
            "guidance_scale": self.args.guidance_scale,
        }
