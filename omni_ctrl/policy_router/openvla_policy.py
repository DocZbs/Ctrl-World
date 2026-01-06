"""OpenVLA policy wrapper."""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .base_policy import BasePolicy


class OpenVLAPolicy(BasePolicy):
    """Wrapper for OpenVLA-style models.

    This wrapper expects an OpenVLA-compatible model that provides either
    a `predict_action` or `infer` method. If those are not available, you
    will need to adapt `_predict_with_model` to your specific OpenVLA API.

    Note: OpenVLA models typically only support single-image input. For DROID
    datasets with multiple cameras, use image_key to select which camera to use.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_space: str = "cartesian",
        unnorm_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        image_key: Optional[str] = None,
        use_wrist_camera: bool = False,
        rescale_stats_path: Optional[str] = None,
    ):
        """Initialize OpenVLA policy.

        Args:
            checkpoint_path: Path to OpenVLA checkpoint
            device: Device to run model on
            action_space: Action space type (cartesian, cartesian_delta, joint_vel, etc.)
            unnorm_key: Unnormalization key for actions
            prompt_template: Template for instruction prompts
            image_key: Key to use for primary image (default: "image_primary")
            use_wrist_camera: If True, use image_wrist instead of image_primary
            rescale_stats_path: Path to rescaling statistics JSON
        """
        self._device = device
        self._action_space = action_space
        self._prompt_template = prompt_template or "In: What action should the robot take to {instruction}?\nOut:"
        self._use_wrist_camera = use_wrist_camera

        # Select image key based on use_wrist_camera flag
        if use_wrist_camera:
            self._image_key = "image_wrist"
        else:
            self._image_key = image_key or "image_primary"

        self.action_chunk_size = 1
        self._target_low = None
        self._target_high = None

        print(f"  OpenVLA will use: {self._image_key}")

        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        self._processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        self._dtype = next(self._model.parameters()).dtype

        norm_stats = getattr(self._model.config, "norm_stats", {}) or {}
        if unnorm_key:
            self._unnorm_key = unnorm_key
        elif "bridge_orig" in norm_stats:
            self._unnorm_key = "bridge_orig"
        elif norm_stats:
            self._unnorm_key = next(iter(norm_stats.keys()))
        else:
            raise ValueError("OpenVLA model has no norm_stats; cannot select unnorm_key.")

        if rescale_stats_path:
            with open(rescale_stats_path) as f:
                stats = json.load(f)
            self._target_low = np.array(stats["state_01"], dtype=np.float32)
            self._target_high = np.array(stats["state_99"], dtype=np.float32)

    @property
    def name(self) -> str:
        return "openvla"

    @property
    def action_space(self) -> str:
        return self._action_space

    def predict(self, obs: Dict, task_instruction: str) -> np.ndarray:
        action = self.predict_chunk(obs, task_instruction, chunk_size=1)
        return np.asarray(action[0])

    def predict_chunk(self, obs: Dict, task_instruction: str, chunk_size: int = 1) -> np.ndarray:
        """Predict action chunk from observation.

        Args:
            obs: Observation dict with keys:
                - image_primary: (H, W, 3) exterior camera RGB image
                - image_wrist: (H, W, 3) wrist camera RGB image (optional)
                - joint_pos: (8,) joint positions (will be ignored by OpenVLA)
            task_instruction: Natural language task instruction
            chunk_size: Number of actions to predict

        Returns:
            Action chunk array of shape (chunk_size, action_dim)

        Note:
            OpenVLA typically only uses a single image. DROID provides multiple
            cameras, but we select one based on self._image_key (configured via
            use_wrist_camera parameter).
        """
        # Get the selected image (primary or wrist)
        image = obs.get(self._image_key)
        if image is None:
            # Fallback: try other image keys
            if self._image_key == "image_wrist" and "image_primary" in obs:
                print(f"Warning: {self._image_key} not found, falling back to image_primary")
                image = obs["image_primary"]
            elif self._image_key == "image_primary" and "image_wrist" in obs:
                print(f"Warning: {self._image_key} not found, falling back to image_wrist")
                image = obs["image_wrist"]
            else:
                raise ValueError(
                    f"OpenVLA expects '{self._image_key}' in obs, but got keys: {list(obs.keys())}"
                )

        # Convert to numpy uint8 if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Prepare prompt and image
        prompt = self._prompt_template.format(instruction=task_instruction)
        image_pil = Image.fromarray(image)

        # Process inputs
        inputs = self._processor(prompt, image_pil, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            value = value.to(self._device)
            if torch.is_floating_point(value):
                value = value.to(dtype=self._dtype)
            moved[key] = value
        inputs = moved

        # Predict action
        with torch.no_grad():
            action = self._model.predict_action(**inputs, unnorm_key=self._unnorm_key, do_sample=False)

        # Post-process action
        action = np.asarray(action)
        if action.ndim == 1:
            action = action[None, :]
        action = self._maybe_rescale_action(action)

        # Ensure correct chunk size
        if action.shape[0] != chunk_size:
            action = np.tile(action, (chunk_size, 1))

        return action

    def _maybe_rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale OpenVLA actions to target stats if provided."""
        if self._action_space == "cartesian_delta":
            return action
        if self._target_low is None or self._target_high is None:
            return action

        norm_stats = getattr(self._model.config, "norm_stats", {}) or {}
        action_stats = norm_stats.get(self._unnorm_key, {}).get("action", {})
        if not action_stats:
            return action

        src_low = np.array(action_stats.get("q01"), dtype=np.float32)
        src_high = np.array(action_stats.get("q99"), dtype=np.float32)
        mask = np.array(action_stats.get("mask", np.ones_like(src_low, dtype=bool)))

        # Normalize to [-1, 1] using source stats
        eps = 1e-8
        normed = action.copy()
        normed = np.where(
            mask,
            2 * (normed - src_low) / (src_high - src_low + eps) - 1,
            normed,
        )

        # Denormalize to target stats
        target = 0.5 * (normed + 1) * (self._target_high - self._target_low) + self._target_low
        return target

    def reset(self):
        return
