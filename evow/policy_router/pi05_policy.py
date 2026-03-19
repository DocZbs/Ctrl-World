"""Pi0.5 policy wrapper."""

from pathlib import Path
import numpy as np
import torch
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R

from .base_policy import BasePolicy
from ..utils.runtime import OPENPI_ROOT, ensure_openpi_client_on_path, ensure_openpi_on_path, ensure_project_root_on_path


class Pi05Policy(BasePolicy):
    """Wrapper for Pi0.5 VLA model.

    Reuses existing Pi0.5 integration from rollout_interact_pi.py.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_adapter_path: Optional[str] = None,
        policy_type: str = "pi05",
        pred_step: int = 5,
        policy_skip_step: int = 3,
        gripper_max: float = 0.7,
    ):
        """Initialize Pi0.5 policy.

        Args:
            checkpoint_path: Path to Pi0.5 checkpoint
            device: Device to run model on
            action_adapter_path: Optional path to action adapter (dynamics model)
            policy_type: Policy type, one of "pi05", "pi0fast", "pi0"
            pred_step: Number of prediction steps
            policy_skip_step: Skip step for policy output
            gripper_max: Maximum gripper position
        """
        ensure_project_root_on_path()
        openpi_path = ensure_openpi_on_path()
        ensure_openpi_client_on_path()
        if openpi_path is not None:
            openpi_path = Path(openpi_path)
        else:
            print(f"Warning: OpenPI directory not found at {OPENPI_ROOT}")
            print("Pi0.5 policy may not work correctly without openpi module")

        try:
            from openpi.training import config as config_pi
            from openpi.policies import policy_config
            import openpi.models.pi0_config as pi0_config
            from openpi.training.config import TrainConfig, LeRobotDROIDDataConfig, DataConfig, AssetsConfig
            import openpi.training.weight_loaders as weight_loaders

            # Register LoRA config if needed
            def register_lora_config_if_needed():
                config_names = [c.name for c in config_pi._CONFIGS]
                if "pi05_droid_lora" not in config_names:
                    model_config = pi0_config.Pi0Config(
                        pi05=True,
                        action_dim=32,
                        action_horizon=16,
                        paligemma_variant="gemma_2b_lora",
                        action_expert_variant="gemma_300m_lora",
                    )
                    freeze_filter = pi0_config.Pi0Config(
                        pi05=True,
                        action_dim=32,
                        action_horizon=16,
                        paligemma_variant="gemma_2b_lora",
                        action_expert_variant="gemma_300m_lora",
                    ).get_freeze_filter()

                    lora_config = TrainConfig(
                        name="pi05_droid_lora",
                        model=model_config,
                        data=LeRobotDROIDDataConfig(
                            repo_id="local/synthetic_pickplace_0002",
                            base_config=DataConfig(prompt_from_task=True),
                            assets=AssetsConfig(
                                assets_dir="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/assets",
                                asset_id="droid",
                            ),
                        ),
                        weight_loader=weight_loaders.CheckpointWeightLoader(
                            "/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/params"
                        ),
                        pytorch_weight_path="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch",
                        freeze_filter=freeze_filter,
                        num_train_steps=5000,
                        batch_size=32,
                        ema_decay=None,
                    )
                    config_pi._CONFIGS.append(lora_config)
                    print(f"Registered pi05_droid_lora config for inference")

            # Get Pi0.5 DROID config
            checkpoint_path_obj = Path(checkpoint_path)
            if "lora" in str(checkpoint_path_obj).lower():
                register_lora_config_if_needed()
                config = config_pi.get_config("pi05_droid_lora")
                print(f"Using pi05_droid_lora config for LoRA checkpoint")
            elif "finetune" in str(checkpoint_path_obj):
                # For inference, finetuned checkpoints are compatible with pi05_droid config.
                # This avoids extra remote asset dependencies in pi05_droid_finetune config.
                config = config_pi.get_config("pi05_droid")
                print(f"Using pi05_droid config for finetuned checkpoint")
            else:
                config = config_pi.get_config("pi05_droid")

            # Create trained policy with specified device
            self.policy = policy_config.create_trained_policy(
                config,
                checkpoint_path,
                pytorch_device=device
            )
            self.device = device
            self.action_buffer = []
            # ctrl-world's dynamics adapter expects 15-step chunks (see rollout_interact_pi.py).
            self.action_chunk_size = 15
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

            print(f"✓ Pi0.5 policy loaded from {checkpoint_path}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import openpi: {e}. "
                "Please ensure openpi is installed and available."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pi0.5 policy: {e}")

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
                - gripper_pos: (1,) gripper position
            task_instruction: Natural language instruction

        Returns:
            Joint velocity action (8,) - 7 joint velocities + 1 gripper
        """
        # If buffer has actions, return next
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)

        action_chunk = self.predict_chunk(obs, task_instruction, chunk_size=self.action_chunk_size)
        if action_chunk is None or len(action_chunk) == 0:
            return np.zeros(8, dtype=np.float32)

        self.action_buffer = list(action_chunk)
        return self.action_buffer.pop(0)

    def predict_chunk(self, obs: Dict, task_instruction: str, chunk_size: int = 10) -> np.ndarray:
        """Predict a full action chunk from Pi0.5.

        Following the same preprocessing pipeline as rollout_interact_pi.py:
        - Resize images from (192, 320, 3) to (180, 320) first
        - Then apply resize_with_pad to 224x224
        """
        try:
            from openpi_client import image_tools

            # Preprocess images - same as rollout_interact_pi.py lines 231-239
            # NOTE: orchestrator already resizes images to (180, 320), so we don't need to resize again
            image1 = obs["image_primary"]  # exterior image (180, 320, 3)
            image2 = obs["image_wrist"]     # wrist image (180, 320, 3)

            # Build example dict - same as rollout_interact_pi.py lines 240-246
            example = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
                "observation/joint_position": obs["joint_pos"][:7],
                "observation/gripper_position": obs["joint_pos"][-1:],
                "prompt": task_instruction,
            }

            # Policy inference - same as rollout_interact_pi.py line 247
            with torch.no_grad():
                action_chunk = self.policy.infer(example)["actions"]  # (10, 8) velocity

            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()

            action_chunk = np.asarray(action_chunk)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            if action_chunk.shape[0] != chunk_size:
                if action_chunk.shape[0] < chunk_size:
                    pad = np.repeat(action_chunk[-1][None, :], chunk_size - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, pad], axis=0)
                action_chunk = action_chunk[:chunk_size]

            return action_chunk

        except Exception as e:
            print(f"Error in Pi0.5 forward pass: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((chunk_size, 8), dtype=np.float32)

    def predict_with_adapter(self, obs: Dict, task_instruction: str) -> tuple:
        """Predict action with action adapter and FK conversion.

        This follows the same pipeline as rollout_interact_pi.py forward_policy method (lines 228-291).

        Args:
            obs: Observation dict with:
                - image_primary: (192, 320, 3) RGB exterior image
                - image_wrist: (192, 320, 3) RGB wrist image
                - joint_pos: (8,) joint positions (7 joints + 1 gripper)
            task_instruction: Natural language instruction

        Returns:
            policy_in_out: Dict with 'joint_pos', 'joint_vel', 'state_fk'
            joint_pos_skip: (pred_step, 8) joint positions with gripper
            state_fk_skip: (pred_step, 7) cartesian poses (xyz, euler, gripper)
        """
        if self.dynamics_model is None:
            raise RuntimeError("Action adapter not loaded. Please provide action_adapter_path during initialization.")

        # Get action chunk from policy - same as rollout_interact_pi.py line 247
        action_chunk = self.predict_chunk(obs, task_instruction, chunk_size=self.action_chunk_size)

        # Action adapter - same as rollout_interact_pi.py lines 249-277
        joints = obs["joint_pos"]
        current_joint = joints[None, :][:, :7]
        current_gripper = joints[None, :][:, 7:]

        # Index strategy based on policy type - same as rollout_interact_pi.py lines 252-255
        if 'pi05' in self.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        # Policy output joint velocity and gripper position
        joint_vel = action_chunk[:, :7]  # (15, 7)
        gripper_pos = action_chunk[:, 7:]  # (15, 1)
        joint_vel = joint_vel[idx]  # (15, 7)
        gripper_pos = gripper_pos[idx]  # (15, 1)
        gripper_pos = np.clip(gripper_pos, 0, self.gripper_max)

        # Calculate future joint positions - same as rollout_interact_pi.py line 264
        joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)

        # FK - same as rollout_interact_pi.py lines 265-277
        state_fk = []
        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]  # (15, 7)
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)
        joint_vel = joint_vel  # (15, 7)
        for i in range(joint_pos.shape[0]):
            current_state_fk = self.get_fk_solution(joint_pos[i, :7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk)  # (15, 7)

        # Prepare output - same as rollout_interact_pi.py lines 279-290
        skip = self.policy_skip_step
        valid_num = int(skip * (self.pred_step - 1))
        policy_in_out = {
            'joint_pos': joint_pos[:valid_num],  # (12, 7)
            'joint_vel': joint_vel[:valid_num],  # (12, 7)
            'state_fk': state_fk[:valid_num],    # (12, 7)
        }
        state_fk_skip = state_fk[::skip][:self.pred_step]  # (5, 7)
        joint_pos_skip = joint_pos[::skip][:self.pred_step]  # (5, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)  # (5, 8) add gripper pos

        return policy_in_out, joint_pos_skip, state_fk_skip

    def reset(self):
        """Clear action buffer."""
        self.action_buffer = []
