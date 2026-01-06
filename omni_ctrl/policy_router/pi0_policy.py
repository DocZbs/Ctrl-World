"""Pi0 DROID policy wrapper."""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R

from .base_policy import BasePolicy


class Pi0Policy(BasePolicy):
    """Wrapper for Pi0 VLA model (DROID variant)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_adapter_path: Optional[str] = None,
        policy_type: str = "pi0",
        pred_step: int = 5,
        policy_skip_step: int = 3,
        gripper_max: float = 0.7,
    ):
        """Initialize Pi0 policy.

        Args:
            checkpoint_path: Path to Pi0 checkpoint
            device: Device to run model on
            action_adapter_path: Optional path to action adapter (dynamics model)
            policy_type: Policy type string (used for adapter indexing)
            pred_step: Number of prediction steps
            policy_skip_step: Skip step for policy output
            gripper_max: Maximum gripper position
        """
        openpi_path = Path(__file__).parent.parent.parent / "openpi"
        if openpi_path.exists():
            sys.path.insert(0, str(openpi_path))
        else:
            print(f"Warning: OpenPI directory not found at {openpi_path}")
            print("Pi0 policy may not work correctly without openpi module")

        try:
            from openpi.training import config as config_pi
            from openpi.policies import policy_config

            # Get Pi0 DROID config
            config = config_pi.get_config("pi0_droid")

            # Create trained policy with specified device
            self.policy = policy_config.create_trained_policy(
                config,
                checkpoint_path,
                pytorch_device=device
            )
            self.device = device
            self.action_buffer = []
            self.action_chunk_size = 10
            self.policy_type = policy_type
            self.pred_step = pred_step
            self.policy_skip_step = policy_skip_step
            self.gripper_max = gripper_max

            # Load action adapter (dynamics model) if provided
            self.dynamics_model = None
            if action_adapter_path is not None:
                from models.action_adapter.train2 import Dynamics
                from models.utils import get_fk_solution
                self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(device)
                self.dynamics_model.load_state_dict(torch.load(action_adapter_path, map_location=device))
                self.get_fk_solution = get_fk_solution
                print(f"✓ Action adapter loaded from {action_adapter_path}")

            print(f"✓ Pi0 policy loaded from {checkpoint_path}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import openpi: {e}. "
                "Please ensure openpi is installed and available."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pi0 policy: {e}")

    @property
    def name(self) -> str:
        """Return policy name."""
        return "pi0"

    @property
    def action_space(self) -> str:
        """Return action space type."""
        return "joint_vel"

    def predict(self, obs: Dict, task_instruction: str) -> np.ndarray:
        """Predict joint velocity action."""
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)

        action_chunk = self.predict_chunk(obs, task_instruction, chunk_size=self.action_chunk_size)
        if action_chunk is None or len(action_chunk) == 0:
            return np.zeros(8, dtype=np.float32)

        self.action_buffer = list(action_chunk)
        return self.action_buffer.pop(0)

    def predict_chunk(self, obs: Dict, task_instruction: str, chunk_size: int = 10) -> np.ndarray:
        """Predict a full action chunk from Pi0."""
        try:
            from openpi_client import image_tools

            image_primary = obs["image_primary"]
            image_wrist = obs["image_wrist"]

            example = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(image_primary, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(image_wrist, 224, 224),
                "observation/joint_position": obs["joint_pos"][:7],
                "observation/gripper_position": obs["joint_pos"][-1:],
                "prompt": task_instruction,
            }

            with torch.no_grad():
                action_chunk = self.policy.infer(example)["actions"]

            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()

            if action_chunk.shape[0] != chunk_size:
                if action_chunk.shape[0] < chunk_size:
                    action_chunk = np.tile(action_chunk, (chunk_size // action_chunk.shape[0] + 1, 1))
                action_chunk = action_chunk[:chunk_size]

            return action_chunk

        except Exception as e:
            print(f"Error in Pi0 forward pass: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((chunk_size, 8), dtype=np.float32)

    def predict_with_adapter(self, obs: Dict, task_instruction: str) -> tuple:
        """Predict action with action adapter and FK conversion."""
        if self.dynamics_model is None:
            raise RuntimeError("Action adapter not loaded. Please provide action_adapter_path during initialization.")

        action_chunk = self.predict_chunk(obs, task_instruction)

        joints = obs["joint_pos"]
        current_joint = joints[None, :][:, :7]
        current_gripper = joints[None, :][:, 7:]

        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        joint_vel = action_chunk[:, :7]
        gripper_pos = action_chunk[:, 7:]
        joint_vel = joint_vel[idx]
        gripper_pos = gripper_pos[idx]
        gripper_pos = np.clip(gripper_pos, 0, self.gripper_max)

        joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)

        state_fk = []
        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]
        for i in range(joint_pos.shape[0]):
            current_state_fk = self.get_fk_solution(joint_pos[i, :7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler("xyz")
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk)

        skip = self.policy_skip_step
        valid_num = int(skip * (self.pred_step - 1))
        policy_in_out = {
            "joint_pos": joint_pos[:valid_num],
            "joint_vel": joint_vel[:valid_num],
            "state_fk": state_fk[:valid_num],
        }
        state_fk_skip = state_fk[::skip][:self.pred_step]
        joint_pos_skip = joint_pos[::skip][:self.pred_step]
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)

        return policy_in_out, joint_pos_skip, state_fk_skip

    def reset(self):
        """Clear action buffer."""
        self.action_buffer = []
