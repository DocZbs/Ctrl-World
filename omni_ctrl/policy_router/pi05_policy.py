"""Pi0.5 policy wrapper."""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict

from .base_policy import BasePolicy


class Pi05Policy(BasePolicy):
    """Wrapper for Pi0.5 VLA model.

    Reuses existing Pi0.5 integration from rollout_interact_pi.py.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        """Initialize Pi0.5 policy.

        Args:
            checkpoint_path: Path to Pi0.5 checkpoint
            device: Device to run model on
        """
        # Add openpi to path
        openpi_path = Path(__file__).parent.parent.parent / "openpi"
        if openpi_path.exists():
            sys.path.insert(0, str(openpi_path))
        else:
            print(f"Warning: OpenPI directory not found at {openpi_path}")
            print("Pi0.5 policy may not work correctly without openpi module")

        try:
            from openpi import build_pi_policy, PI_POLICY_CONFIGS

            config = PI_POLICY_CONFIGS.get("pi0.5")
            if config is None:
                raise ValueError("Pi0.5 config not found in PI_POLICY_CONFIGS")

            self.policy = build_pi_policy(checkpoint_path, config)
            self.device = device
            self.action_buffer = []

            print(f"Pi0.5 policy loaded from {checkpoint_path}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import openpi: {e}. "
                "Please ensure openpi is installed and available."
            )

    @property
    def name(self) -> str:
        """Return policy name."""
        return "pi05"

    @property
    def action_space(self) -> str:
        """Return action space type."""
        return "joint_vel"

    def predict(self, obs: Dict, task_instruction: str) -> np.ndarray:
        """Predict joint velocity action.

        Pi0.5 generates 15-step action chunks, so we buffer them
        and return one at a time.

        Args:
            obs: Observation dict with:
                - image_primary: (H, W, 3) RGB image
                - image_wrist: (H, W, 3) RGB image
                - joint_pos: (7,) joint positions
            task_instruction: Natural language instruction

        Returns:
            Joint velocity action (8,) - 7 joint velocities + 1 gripper
        """
        # If buffer has actions, return next
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)

        # Policy forward (generates 15-step action chunk)
        try:
            with torch.no_grad():
                action_chunk = self.policy(
                    image_primary=obs["image_primary"],
                    image_wrist=obs["image_wrist"],
                    proprio=obs["joint_pos"],
                    instruction=task_instruction,
                )

            # Convert to numpy if needed
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()

            # Store in buffer
            self.action_buffer = list(action_chunk)

            if len(self.action_buffer) == 0:
                # Fallback: return zero action
                return np.zeros(8, dtype=np.float32)

            return self.action_buffer.pop(0)

        except Exception as e:
            print(f"Error in Pi0.5 forward pass: {e}")
            # Return safe zero action
            return np.zeros(8, dtype=np.float32)

    def reset(self):
        """Clear action buffer."""
        self.action_buffer = []
