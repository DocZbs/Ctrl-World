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
        args.num_inference_steps = 50
        args.guidance_scale = 2.0
        args.seed = 42
        args.device = config.device

        # Load world model
        print(f"Loading Ctrl-World model from {config.wm_ckpt}...")
        self.model = CrtlWorld(args)
        self.args = args
        self.device = config.device

        # Load normalization stats
        print(f"Loading normalization stats from {config.data_stat_path}...")
        with open(args.data_stat_path) as f:
            self.stat = json.load(f)

        print("World Model Wrapper initialized successfully")

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using dataset statistics.

        Args:
            action: Unnormalized cartesian action (7,)

        Returns:
            Normalized action (7,)
        """
        action_norm = action.copy()
        for i in range(7):
            low = self.stat["cartesian_position"][f"{i}_1"]
            high = self.stat["cartesian_position"][f"{i}_99"]
            action_norm[i] = (action[i] - low) / (high - low) * 2 - 1
        return action_norm

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to original scale.

        Args:
            action_norm: Normalized action (7,)

        Returns:
            Denormalized action (7,)
        """
        action = action_norm.copy()
        for i in range(7):
            low = self.stat["cartesian_position"][f"{i}_1"]
            high = self.stat["cartesian_position"][f"{i}_99"]
            action[i] = (action_norm[i] + 1) / 2 * (high - low) + low
        return action

    def step(
        self,
        action_seq: np.ndarray,
        current_latent: torch.Tensor,
        history_latents: List[torch.Tensor],
        text: str,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Execute one step in world model.

        Args:
            action_seq: Action sequence (pred_step, 7) in cartesian space
            current_latent: Current latent state (4, H, W)
            history_latents: List of history latents, length 6
            text: Task instruction text

        Returns:
            Tuple of:
                - next_latent: Predicted next latent (4, H, W)
                - next_frame: Decoded next frame (H, W, 3)
        """
        # Normalize actions
        action_norm = np.array([self.normalize_action(a) for a in action_seq])
        action_tensor = (
            torch.from_numpy(action_norm).float().unsqueeze(0).to(self.device)
        )

        # Prepare history - take last 6 latents
        if len(history_latents) < 6:
            # Pad with first latent if not enough history
            while len(history_latents) < 6:
                history_latents.insert(0, history_latents[0])

        his_cond = torch.stack(history_latents[-6:]).to(self.device)

        # World model forward
        with torch.no_grad():
            video_pred = self.model(
                action_cond=action_tensor,
                video_latent_cond=current_latent.unsqueeze(0).to(self.device),
                his_cond=his_cond.unsqueeze(0),
                text=text,
            )

        # Decode last frame to RGB
        frame_decoded = self.model.pipeline.decode_latents(
            video_pred[:, -1:],
            num_frames=1
        )

        # Return next latent and decoded frame
        next_latent = video_pred[0, -1]
        next_frame = frame_decoded[0, 0].cpu().numpy()  # (H, W, 3)

        # Ensure frame is in [0, 1] range
        next_frame = np.clip(next_frame, 0, 1)

        return next_latent, next_frame

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
        frame = frame[0, 0].cpu().numpy()
        return np.clip(frame, 0, 1)

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
