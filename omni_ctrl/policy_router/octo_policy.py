"""Octo policy wrapper."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional
import jax
import jax.numpy as jnp

from .base_policy import BasePolicy


class OctoPolicy(BasePolicy):
    """Wrapper for Octo generalist robot policy.

    Octo is a transformer-based diffusion policy pretrained on 800k+ robot
    trajectories from the Open X-Embodiment dataset. It supports language
    instructions, multiple camera inputs, and outputs action chunks.

    Key features:
    - Action chunking (default chunk_size=4)
    - Observation history (default window_size=2)
    - Diffusion-based action prediction
    - Supports both language and goal-image conditioning
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_space: str = "cartesian_delta",
        horizon: int = 4,
        use_language: bool = True,
        use_goal_image: bool = False,
        image_size: int = 256,
    ):
        """Initialize Octo policy.

        Args:
            checkpoint_path: Path to Octo checkpoint (e.g., "hf://rail-berkeley/octo-small-1.5")
            device: Device to run model on (JAX will auto-detect GPU)
            action_space: Action space type (cartesian_delta, joint_pos, etc.)
            horizon: Action chunk size (default: 4)
            use_language: Whether to use language instructions
            use_goal_image: Whether to use goal images for conditioning
            image_size: Size to resize images to (default: 256)
        """
        self._device = device
        self._action_space = action_space
        self._horizon = horizon
        self._use_language = use_language
        self._use_goal_image = use_goal_image
        self._image_size = image_size

        # Import here to avoid dependency issues if Octo is not installed
        try:
            from octo.model.octo_model import OctoModel
        except ImportError as e:
            raise ImportError(
                f"Failed to import Octo: {e}. "
                "Please install Octo: pip install git+https://github.com/octo-models/octo.git"
            )

        # Load pretrained model
        print(f"Loading Octo model from {checkpoint_path}...")
        self._model = OctoModel.load_pretrained(checkpoint_path)
        print(f"✓ Octo policy loaded successfully")
        print(f"  Model spec:\n{self._model.get_pretty_spec()}")

        # Initialize RNG key for sampling
        self._rng = jax.random.PRNGKey(0)

        # Action buffer for chunking
        self._action_buffer = []

        # Observation history buffer (for window_size=2)
        self._obs_history = []
        self._window_size = 2  # Octo uses 2-frame history

    @property
    def name(self) -> str:
        """Return policy name."""
        return "octo"

    @property
    def action_space(self) -> str:
        """Return action space type."""
        return self._action_space

    def predict(self, obs: Dict, task_instruction: str) -> np.ndarray:
        """Predict single action from observation.

        Args:
            obs: Observation dict with:
                - image_primary: (H, W, 3) RGB image
                - image_wrist: (H, W, 3) RGB image (optional)
                - joint_pos: (7,) or (8,) joint positions
            task_instruction: Natural language task instruction

        Returns:
            Action array (action_dim,) - typically (7,) or (8,) for cartesian_delta
        """
        # If buffer has actions, return next
        if len(self._action_buffer) > 0:
            return self._action_buffer.pop(0)

        # Generate new action chunk
        action_chunk = self.predict_chunk(obs, task_instruction, chunk_size=self._horizon)
        if action_chunk is None or len(action_chunk) == 0:
            # Return zero action as fallback
            return np.zeros(7, dtype=np.float32)

        # Buffer remaining actions
        self._action_buffer = list(action_chunk[1:])
        return action_chunk[0]

    def predict_chunk(self, obs: Dict, task_instruction: str, chunk_size: int = 4) -> np.ndarray:
        """Predict action chunk from Octo model.

        Args:
            obs: Observation dictionary
            task_instruction: Natural language instruction
            chunk_size: Number of actions to predict (default: 4)

        Returns:
            Action chunk array of shape (chunk_size, action_dim)
        """
        # Prepare observation for Octo
        octo_obs = self._prepare_observation(obs)

        # Create task specification
        if self._use_language:
            task = self._model.create_tasks(texts=[task_instruction])
        elif self._use_goal_image:
            # TODO: Support goal image conditioning
            raise NotImplementedError("Goal image conditioning not yet implemented")
        else:
            raise ValueError("Must use either language or goal image conditioning")

        # Sample actions using diffusion model
        self._rng, sample_rng = jax.random.split(self._rng)
        action_chunk = self._model.sample_actions(
            octo_obs,
            task,
            rng=sample_rng,
        )

        # Convert JAX array to numpy
        action_chunk = np.asarray(action_chunk)

        # Remove batch dimension if present
        if action_chunk.ndim == 3:  # (1, chunk_size, action_dim)
            action_chunk = action_chunk[0]

        # Ensure correct chunk size
        if action_chunk.shape[0] != chunk_size:
            # Repeat or truncate to match requested chunk size
            if action_chunk.shape[0] < chunk_size:
                action_chunk = np.tile(action_chunk, (chunk_size // action_chunk.shape[0] + 1, 1))
            action_chunk = action_chunk[:chunk_size]

        return action_chunk

    def _prepare_observation(self, obs: Dict) -> Dict:
        """Convert observation dict to Octo format with history window.

        Octo expects:
        - image_primary: (1, window_size, 256, 256, 3) uint8
        - image_wrist: (1, window_size, 128, 128, 3) uint8 (optional)

        Args:
            obs: Raw observation dict

        Returns:
            Formatted observation dict for Octo
        """
        # Process current observation
        current_obs = {}

        # Process images
        image_primary = obs.get("image_primary")
        image_wrist = obs.get("image_wrist")

        if image_primary is not None:
            # Primary camera: 256x256
            current_obs["image_primary"] = self._process_image(image_primary, target_size=256)

        if image_wrist is not None:
            # Wrist camera: 128x128 (Octo expects different resolution)
            current_obs["image_wrist"] = self._process_image(image_wrist, target_size=128)

        # Add to history buffer
        self._obs_history.append(current_obs)

        # Keep only last window_size observations
        if len(self._obs_history) > self._window_size:
            self._obs_history = self._obs_history[-self._window_size:]

        # If history is not full, pad by repeating first observation
        while len(self._obs_history) < self._window_size:
            self._obs_history.insert(0, self._obs_history[0])

        # Stack history into window dimension
        octo_obs = {}

        # Stack image_primary: (window_size, H, W, 3) -> (1, window_size, H, W, 3)
        if "image_primary" in self._obs_history[0]:
            primary_stack = np.stack([o["image_primary"][0] for o in self._obs_history], axis=0)
            octo_obs["image_primary"] = primary_stack[None, ...]  # Add batch dim

        # Stack image_wrist: (window_size, H, W, 3) -> (1, window_size, H, W, 3)
        if "image_wrist" in self._obs_history[0]:
            wrist_stack = np.stack([o["image_wrist"][0] for o in self._obs_history], axis=0)
            octo_obs["image_wrist"] = wrist_stack[None, ...]  # Add batch dim

        # Add timestep pad mask (all True for valid timesteps)
        octo_obs["timestep_pad_mask"] = np.ones((1, self._window_size), dtype=bool)

        return octo_obs

    def _process_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Process image for Octo model.

        Args:
            image: RGB image (H, W, 3)
            target_size: Target size for resizing (256 for primary, 128 for wrist)

        Returns:
            Processed image (1, target_size, target_size, 3) uint8
        """
        import cv2

        # Convert to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Resize to target size
        if image.shape[:2] != (target_size, target_size):
            image = cv2.resize(
                image,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR
            )

        # Add batch dimension
        if image.ndim == 3:
            image = image[None, :]  # (1, H, W, 3)

        return image

    def reset(self):
        """Reset policy state."""
        self._action_buffer = []
        self._obs_history = []  # Clear observation history
        # Re-initialize RNG for reproducibility
        self._rng = jax.random.PRNGKey(0)
