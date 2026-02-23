"""World model wrapper for NEXUS.

This module wraps the existing Ctrl-World model to provide
a clean interface for the NEXUS orchestrator.
"""

import sys
import torch
import torch.nn.functional as F
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

        # Disable gradient checkpointing for inference
        if hasattr(self.model.unet, 'disable_gradient_checkpointing'):
            self.model.unet.disable_gradient_checkpointing()

        # Convert to bfloat16 for inference efficiency
        self.model.to(dtype=args.dtype, device=config.device)
        self.model.eval()

        # Set requires_grad to False for all parameters
        for param in self.model.parameters():
            param.requires_grad = False

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
        ground_truth_views: np.ndarray = None,
    ) -> Tuple[torch.Tensor, list, np.ndarray]:
        """Execute one step in world model.

        Following the same pipeline as rollout_interact_pi.py forward_wm method (lines 151-226).

        Args:
            action_seq: Action sequence (pred_step, 7) in cartesian space
            current_latent: Current latent state (4, 72, 40) or (1, 4, 72, 40)
            history_latents: List of history latents, each with shape (4, 72, 40) or (1, 4, 72, 40)
            history_actions: List of history actions, each with shape (7,)
            text: Task instruction text
            ground_truth_views: Optional GT RGB sequence (3, F, H, W, 3), used directly for top panel

        Returns:
            Tuple of:
                - next_latent: Predicted next latent (1, 4, 72, 40) - with batch dim
                - next_frames: List of decoded frames (with GT if provided)
                - camera_views: Camera views array (3, pred_step, H, W, 3)
        """
        # Normalize all latents to canonical Ctrl-World shape: (1, 4, 72, 40)
        current_latent = self._ensure_ctrlworld_latent(current_latent, "current_latent")

        normalized_history = []
        for lat in history_latents:
            normalized_history.append(self._ensure_ctrlworld_latent(lat, "history_latent"))
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

        # Convert to numpy: (3, 5, 3, H, W) -> (3, 5, H, W, 3)
        if videos.dtype == torch.bfloat16:
            videos = videos.float()
        videos_np = videos.cpu().numpy().transpose(0, 1, 3, 4, 2)  # (3 views, 5 frames, H, W, 3)
        videos_np = (videos_np * 255).astype(np.uint8)  # Convert to uint8

        videos_concat = np.concatenate([videos_np[i] for i in range(bsz)], axis=-2)  # (F, H, W*3, 3)

        # Prefer raw GT RGB sequence for top panel (sharper than latent decode)
        if ground_truth_views is not None:
            gt_views = np.asarray(ground_truth_views)
            if gt_views.ndim != 5 or gt_views.shape[0] != 3 or gt_views.shape[-1] != 3:
                raise ValueError(
                    f"ground_truth_views must be (3,F,H,W,3), got {tuple(gt_views.shape)}"
                )
            if gt_views.dtype != np.uint8:
                if np.issubdtype(gt_views.dtype, np.floating):
                    if float(np.nanmax(gt_views)) <= 1.0:
                        gt_views = np.clip(gt_views, 0.0, 1.0) * 255.0
                    gt_views = np.clip(gt_views, 0.0, 255.0).astype(np.uint8)
                else:
                    gt_views = gt_views.astype(np.uint8)

            gt_concat = np.concatenate([gt_views[i] for i in range(gt_views.shape[0])], axis=-2)

            if gt_concat.shape[0] < frame_num:
                pad = np.repeat(gt_concat[-1:, :, :, :], frame_num - gt_concat.shape[0], axis=0)
                gt_concat = np.concatenate([gt_concat, pad], axis=0)
            elif gt_concat.shape[0] > frame_num:
                gt_concat = gt_concat[:frame_num]

            if gt_concat.shape[1] != videos_concat.shape[1] or gt_concat.shape[2] != videos_concat.shape[2]:
                import cv2
                target_hw = (videos_concat.shape[2], videos_concat.shape[1])
                gt_concat = np.stack(
                    [cv2.resize(fr, target_hw, interpolation=cv2.INTER_AREA) for fr in gt_concat],
                    axis=0,
                )

            videos_concat = np.concatenate([gt_concat, videos_concat], axis=1)  # (F, H*2, W*3, 3)

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
        """Encode 3 camera views into Ctrl-World latent.

        Args:
            frame: Can be one of:
                - list/tuple of 3 RGB views, each (H, W, 3)
                - array (3, H, W, 3)
                - array (H, W*3, 3) where 3 views are concatenated horizontally
                - array (H*3, W, 3) where 3 views are concatenated vertically

        Returns:
            Latent tensor (4, 72, 40) - 3 cameras concatenated
        """
        import cv2

        def _split_three_views(frame_like) -> List[np.ndarray]:
            if isinstance(frame_like, (list, tuple)):
                if len(frame_like) != 3:
                    raise ValueError(f"Expected 3 views, got {len(frame_like)}")
                return [np.asarray(v) for v in frame_like]

            arr = np.asarray(frame_like)

            if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[-1] == 3:
                return [arr[0], arr[1], arr[2]]

            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"frame must be list[3] or array with RGB channels, got shape {arr.shape}"
                )

            h, w, _ = arr.shape

            # Horizontal layout: (H, 3W, 3)
            if w % 3 == 0 and (w // 3) >= 64:
                step = w // 3
                return [arr[:, i * step:(i + 1) * step, :] for i in range(3)]

            # Vertical layout: (3H, W, 3)
            if h % 3 == 0 and (h // 3) >= 64:
                step = h // 3
                return [arr[i * step:(i + 1) * step, :, :] for i in range(3)]

            raise ValueError(
                "Cannot infer 3-view layout from frame. "
                f"Got shape {arr.shape}; expected list[3], (3,H,W,3), (H,3W,3), or (3H,W,3)."
            )

        target_h, target_w = 192, 320
        views = _split_three_views(frame)

        encoded_views = []
        for view in views:
            view = np.asarray(view)

            if view.shape[0] != target_h or view.shape[1] != target_w:
                view = cv2.resize(view, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            view = view.astype(np.float32)
            if float(np.nanmax(view)) > 1.0:
                view = view / 255.0
            view = np.clip(view, 0.0, 1.0)

            view_tensor = (
                torch.from_numpy(view)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=self.dtype, device=self.device)
            )
            view_tensor = view_tensor * 2.0 - 1.0

            with torch.no_grad():
                latent = self.model.pipeline.vae.encode(view_tensor).latent_dist.sample()
                latent = latent * self.model.pipeline.vae.config.scaling_factor

            encoded_views.append(latent[0])  # (4, 24, 40)

        # Stack 3 views along latent height: (4, 72, 40)
        return torch.cat(encoded_views, dim=1)

    def encode_multiview_video(self, camera_views: np.ndarray, batch_size: int = 32) -> torch.Tensor:
        """Encode 3-view RGB sequence to Ctrl-World latent sequence.

        Args:
            camera_views: RGB views shaped (3, T, H, W, 3)
            batch_size: VAE encode batch size

        Returns:
            Latent sequence shaped (T, 4, 72, 40)
        """
        import cv2

        views = np.asarray(camera_views)
        if views.ndim != 5 or views.shape[0] != 3 or views.shape[-1] != 3:
            raise ValueError(f"camera_views must be (3,T,H,W,3), got {views.shape}")

        target_h, target_w = 192, 320
        vae = self.model.pipeline.vae
        orig_vae_dtype = vae.dtype
        if orig_vae_dtype != torch.float32:
            vae.to(dtype=torch.float32)
        try:
            encoded_per_view = []

            for cam_idx in range(3):
                v = views[cam_idx]
                if v.shape[1] != target_h or v.shape[2] != target_w:
                    resized = [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for frame in v]
                    v = np.stack(resized, axis=0)

                v = v.astype(np.float32)
                if float(np.nanmax(v)) > 1.0:
                    v = v / 255.0
                v = np.clip(v, 0.0, 1.0)

                x = torch.from_numpy(v).permute(0, 3, 1, 2).to(dtype=torch.float32, device=self.device)
                x = x * 2.0 - 1.0

                latents = []
                with torch.no_grad():
                    for i in range(0, x.shape[0], batch_size):
                        batch = x[i:i + batch_size]
                        latent = vae.encode(batch).latent_dist.sample()
                        latent = latent * vae.config.scaling_factor
                        latents.append(latent)

                encoded_per_view.append(torch.cat(latents, dim=0))  # (T, 4, 24, 40)

            return torch.cat(encoded_per_view, dim=2)  # (T, 4, 72, 40)
        finally:
            if orig_vae_dtype != torch.float32:
                vae.to(dtype=orig_vae_dtype)

    def _ensure_ctrlworld_latent(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize latent tensor to (1, 4, 72, 40).

        Accepts legacy shape (1, 4, 24, 120) and converts it by re-packing
        width-concatenated views into height-concatenated views.

        Also accepts height-concatenated 3-view layouts with non-standard per-view
        latent height (e.g., (66, 40) => 3x22). These are resized per view to
        canonical (24, 40) then re-concatenated to (72, 40).
        """
        if not torch.is_tensor(latent):
            latent = torch.as_tensor(latent)

        if latent.ndim == 3:
            latent = latent.unsqueeze(0)

        if latent.ndim != 4 or latent.shape[1] != 4:
            raise ValueError(
                f"{name} must be (4,H,W) or (1,4,H,W), got {tuple(latent.shape)}"
            )

        _, _, h, w = latent.shape
        if (h, w) == (72, 40):
            return latent.to(dtype=self.dtype, device=self.device)

        if (h, w) == (24, 120):
            latent = torch.cat(torch.chunk(latent, 3, dim=3), dim=2)
            return latent.to(dtype=self.dtype, device=self.device)

        if w == 40 and h % 3 == 0:
            per_view_h = h // 3
            if per_view_h > 0:
                latent = latent.to(dtype=self.dtype, device=self.device)
                b = latent.shape[0]

                # (B,4,3*H,40) -> (B*3,4,H,40)
                views = latent.view(b, 4, 3, per_view_h, 40)
                views = views.permute(0, 2, 1, 3, 4).reshape(b * 3, 4, per_view_h, 40)

                # Resize each camera latent to canonical per-view size (24,40)
                views = F.interpolate(views, size=(24, 40), mode="bilinear", align_corners=False)

                # (B*3,4,24,40) -> (B,4,72,40)
                views = views.reshape(b, 3, 4, 24, 40)
                latent = views.permute(0, 2, 1, 3, 4).reshape(b, 4, 72, 40)
                return latent

        raise ValueError(
            f"{name} has unsupported latent shape {(h, w)}; expected (72,40) "
            "or legacy (24,120), or 3-view stacked (3*k,40)."
        )

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
