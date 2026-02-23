"""NEXUS orchestrator: closed-loop task-to-policy self-evolution (Algorithm 1).

Standalone orchestrator that coordinates:
1. LLM-based task proposal (or template fallback)
2. Temperature-scaled scene retrieval with resampling
3. VLM-based expert routing with memory conditioning
4. World-model-based episode synthesis
5. VLM verification with 4-way failure attribution
6. Closed-loop retry based on failure attribution
"""

import random
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
import yaml

from ..task_generation.template_generator import TemplateTaskGenerator
from ..task_generation.task_schema import Task
from ..retrieval.droid_retriever import DROIDRetriever
from ..rollout.world_model_wrapper import WorldModelWrapper
from ..evaluation.openai_vlm_evaluator import OpenAIVLMEvaluator
from ..policy_router.policy_router import PolicyRouter
from ..policy_router.vlm_router import VLMPolicyRouter
from .episode import Episode
from .skill_library import SkillLibrary
from .nexus_memory import NexusMemory, MemoryEntry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.action_adapter.train2 import Dynamics
from models.utils import get_fk_solution
from scipy.spatial.transform import Rotation as R
OPENPI_CLIENT_ROOT = PROJECT_ROOT / "openpi" / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_ROOT.exists():
    sys.path.insert(0, str(OPENPI_CLIENT_ROOT))
from openpi_client import image_tools


class NexusOrchestrator:
    """NEXUS self-evolution loop implementing Algorithm 1.

    Coordinates all NEXUS modules in a closed-loop:
    for each task l:
        ground scene o_0 via temperature-scaled retrieval
        for each retry attempt:
            route expert m via VLM conditioned on memory M
            rollout episode tau via world model
            verify (r, kappa) via VLM evaluator
            update memory M
            if success: collect into verified set
            else: resample scene / reroute expert based on kappa
    """

    def __init__(self, config):
        print("=" * 60)
        print("Initializing NEXUS Orchestrator")
        print("=" * 60)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # --- Task generator ---
        print("\n[1/7] Initializing task generator...")
        task_gen_type = getattr(config.task_gen, "type", "template")
        if task_gen_type == "llm":
            from ..task_generation.llm_task_proposer import LLMTaskProposer
            self.task_generator = LLMTaskProposer(config.task_gen)
        else:
            self.task_generator = TemplateTaskGenerator(config.task_gen.templates_path)

        if getattr(config, "fixed_scenario", None) and config.fixed_scenario is not None:
            if getattr(config.fixed_scenario, "enabled", False):
                raise ValueError("fixed_scenario is not supported in NEXUS orchestrator")

        # --- Retriever ---
        print("\n[2/7] Initializing DROID retriever...")
        self.retriever = DROIDRetriever(config.retrieval)
        self.default_output_fps = 5
        droid_path = getattr(config.retrieval, "droid_path", None)
        if droid_path:
            info_path = Path(droid_path) / "meta" / "info.json"
            if info_path.exists():
                try:
                    with open(info_path) as f:
                        info = json.load(f)
                    fps = info.get("fps", None)
                    if isinstance(fps, (int, float)) and fps > 0:
                        self.default_output_fps = int(round(float(fps)))
                except (OSError, json.JSONDecodeError):
                    pass

        # --- Base policy router ---
        print("\n[3/7] Initializing policy router...")
        self.base_router = PolicyRouter(
            config.router,
            device=config.device,
            rollout_config=config.rollout,
        )
        available_policies = self.base_router.get_available_policy_names()
        if len(available_policies) == 1:
            policy_name = available_policies[0]
            print(f"Preloading single policy '{policy_name}'...")
            policy_obj = self.base_router.get_policy_by_name(policy_name)
            warmup_enabled = os.environ.get("CTRLWORLD_POLICY_WARMUP", "0") == "1"
            warmup_strict = os.environ.get("CTRLWORLD_POLICY_WARMUP_STRICT", "0") == "1"
            if warmup_enabled and hasattr(policy_obj, "warmup"):
                print(f"Warming up policy '{policy_name}'...", flush=True)
                try:
                    policy_obj.warmup()
                except Exception as e:
                    msg = f"Policy warmup failed: {e}"
                    if warmup_strict:
                        raise RuntimeError(msg)
                    print(f"Warning: {msg}. Continue without warmup.", flush=True)

        # --- VLM router (wraps base router) ---
        print("\n[4/7] Initializing VLM expert router...")
        vlm_routing_cfg = config.vlm_routing
        if getattr(vlm_routing_cfg, "enabled", True):
            self.vlm_router = VLMPolicyRouter(vlm_routing_cfg, self.base_router)
        else:
            self.vlm_router = None

        # --- World model ---
        print("\n[5/7] Initializing world model...")
        self.world_model = WorldModelWrapper(config.rollout)

        # --- Dynamics model (joint_vel -> joint_pos) ---
        print("\n[5.5/7] Initializing dynamics model...")
        self.dynamics_model = Dynamics(
            action_dim=7, action_num=15, hidden_size=512
        ).to(config.device)
        if getattr(config.rollout, "action_adapter_path", None):
            dynamics_ckpt = Path(config.rollout.action_adapter_path)
            if not dynamics_ckpt.is_absolute():
                dynamics_ckpt = PROJECT_ROOT / dynamics_ckpt
        else:
            dynamics_ckpt = PROJECT_ROOT / "models" / "action_adapter" / "model2_15_9.pth"
        if dynamics_ckpt.exists():
            self.dynamics_model.load_state_dict(
                torch.load(dynamics_ckpt, map_location=config.device)
            )
            print(f"  Dynamics model loaded from {dynamics_ckpt}")
        else:
            print(f"  Warning: Dynamics model not found at {dynamics_ckpt}")
            self.dynamics_model = None

        # --- Evaluator ---
        print("\n[6/7] Initializing evaluator...")
        if config.evaluation.vlm_type == "dummy":
            from ..evaluation.dummy_evaluator import DummyEvaluator
            self.evaluator = DummyEvaluator(config.evaluation)
        else:
            self.evaluator = OpenAIVLMEvaluator(config.evaluation)

        # --- NEXUS memory ---
        print("\n[7/7] Initializing NEXUS memory...")
        self.memory = NexusMemory(
            save_path=config.memory.save_path,
            max_entries=config.memory.max_entries,
            context_window=config.memory.context_window,
        )

        # --- Skill library ---
        self.skill_library = SkillLibrary(config.skill_library_path)

        # Store config and setup output
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.retrieval_temperature = config.retrieval_temperature
        self.top_k_candidates = config.top_k_candidates
        self.reward_threshold = config.reward_threshold
        self.max_retries = config.max_retries_per_task
        self.use_scenario_instruction = bool(
            getattr(config.task_gen, "use_scenario_instruction", False)
        )
        if self.max_retries < 1:
            raise ValueError("max_retries_per_task must be >= 1")

        self._save_config()
        self._setup_logging()

        # Statistics
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "wm_failures": 0,
            "task_failures": 0,
            "scene_failures": 0,
            "expert_failures": 0,
            "retries_total": 0,
            "scene_resamples": 0,
            "expert_reroutes": 0,
            "total_reward": 0.0,
            "rewards_history": [],
        }
        self._warned_cartesian_dim = False

        # Verified synthetic dataset (Algorithm 1 output)
        self.verified_set = []
        # Track used scenes within a run to reduce repeated retrieval.
        self._used_scene_ids = set()

        print("\n" + "=" * 60)
        print("NEXUS initialization complete!")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Main loop (Algorithm 1)
    # ------------------------------------------------------------------

    def run(self):
        """Run NEXUS Algorithm 1: closed-loop self-evolution."""
        print(f"Starting NEXUS evolution for {self.config.num_iterations} iterations...")
        print(f"Output directory: {self.output_dir}\n")

        for iteration in tqdm(range(self.config.num_iterations), desc="NEXUS"):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            try:
                self._nexus_iteration(iteration)
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if (iteration + 1) % self.config.log_frequency == 0:
                self._log_stats(iteration + 1)

        print("\n" + "=" * 60)
        print("NEXUS evolution complete!")
        print("=" * 60)
        self._log_stats(self.config.num_iterations)
        self._export_results()

    def _nexus_iteration(self, iteration: int):
        """Single NEXUS iteration implementing Algorithm 1 inner loop.

        Steps:
            1. Propose task l
            2. Ground scene o_0 via temperature-scaled retrieval
            3. For each retry attempt:
               a. Route expert m via VLM (conditioned on memory M)
               b. Rollout episode tau via world model
               c. Verify (r, kappa) via VLM evaluator
               d. Update memory M
               e. If r >= threshold: add to verified set, break
               f. Else: resample scene / reroute expert based on kappa
        """
        # Step 1: Generate a query task for retrieval
        query_task = self.task_generator.generate_task()

        # Step 2: Scene grounding (temperature-scaled retrieval)
        excluded_scene_ids = list(self._used_scene_ids) if self._used_scene_ids else None
        scenario = self.retriever.retrieve_with_temperature(
            query_task,
            temperature=self.retrieval_temperature,
            top_k_candidates=self.top_k_candidates,
            excluded_episode_ids=excluded_scene_ids,
        )
        self._used_scene_ids.add(scenario.episode_id)

        # Optional mode: run directly with scenario instruction (no random task instruction).
        if self.use_scenario_instruction:
            task_instruction = scenario.instruction or query_task.instruction
            task = Task(
                instruction=task_instruction,
                success_criteria=[task_instruction],
                object_categories=[],
                difficulty=getattr(query_task, "difficulty", 0.5),
                task_id=getattr(query_task, "task_id", f"task_{iteration + 1:04d}"),
            )
        else:
            task = query_task

        print(f"\n[Task] {task.instruction}")
        print(f"[Scene] Episode {scenario.episode_id}: {scenario.instruction}")

        failed_scenes = []
        failed_experts = []
        last_eval = None

        # Step 3: Closed-loop retry
        for attempt in range(self.max_retries):
            if attempt > 0:
                print(f"\n  --- Retry {attempt}/{self.max_retries - 1} ---")

            # 3a: VLM-based expert routing
            memory_ctx = self.memory.get_context_for_routing(
                task.instruction, scenario.episode_id
            )
            scene_image = self._get_scene_image(scenario)

            if self.vlm_router is not None:
                policy = self.vlm_router.select_expert(
                    task, scene_image, memory_ctx,
                    excluded_experts=failed_experts,
                )
            else:
                policy = self.base_router.select_policy(task)
            print(f"[Expert] {policy.name}")

            # 3b: World model rollout
            episode = self._execute_rollout(task, scenario, policy)
            print(f"[Rollout] {episode.get_length()} steps, video: {episode.video_path}")

            # 3c: VLM verification with 4-way attribution
            eval_result = self.evaluator.evaluate(episode.video_path, task)
            kappa = eval_result.failure_reason
            last_eval = eval_result
            print(f"[Eval] reward={eval_result.reward:.3f} kappa={kappa}")

            # 3d: Update memory
            entry = MemoryEntry(
                task_instruction=task.instruction,
                task_id=task.task_id,
                scenario_episode_id=scenario.episode_id,
                scenario_instruction=scenario.instruction,
                expert_name=policy.name,
                reward=eval_result.reward,
                failure_attribution=kappa,
                iteration=iteration,
                video_path=str(episode.video_path),
                reasoning=eval_result.reasoning[:200],
            )
            self.memory.append(entry)

            # Update router statistics
            if self.vlm_router is not None:
                self.vlm_router.update(policy.name, eval_result.reward)
            else:
                self.base_router.update(policy.name, eval_result.reward)

            # 3e: Check success
            if eval_result.reward >= self.reward_threshold:
                self.stats["successful_episodes"] += 1
                self.skill_library.add(task, episode, eval_result)
                self.verified_set.append({
                    "task": task.to_dict(),
                    "scenario_id": scenario.episode_id,
                    "expert": policy.name,
                    "reward": eval_result.reward,
                    "video_path": str(episode.video_path),
                })
                break

            # 3f: Failure-conditioned retry
            if kappa == "scene_failure":
                failed_scenes.append(scenario.episode_id)
                self.stats["scene_failures"] += 1
                self.stats["scene_resamples"] += 1
                excluded_for_resample = list(set(failed_scenes).union(self._used_scene_ids))
                scenario = self.retriever.retrieve_with_temperature(
                    task,
                    temperature=self.retrieval_temperature,
                    top_k_candidates=self.top_k_candidates,
                    excluded_episode_ids=excluded_for_resample,
                )
                self._used_scene_ids.add(scenario.episode_id)
                print(f"  [Resample scene] -> {scenario.episode_id}")
            elif kappa == "expert_failure":
                failed_experts.append(policy.name)
                self.stats["expert_failures"] += 1
                self.stats["expert_reroutes"] += 1
                print(f"  [Reroute expert] excluded: {failed_experts}")
            elif kappa == "task_failure":
                self.stats["task_failures"] += 1
                print("  [Skip] task is ill-defined")
                break
            elif kappa == "wm_failure":
                self.stats["wm_failures"] += 1
                print("  [Skip] world model issue")
                break

            self.stats["retries_total"] += 1

        # Update global stats
        self.stats["total_episodes"] += 1
        self.stats["total_reward"] += last_eval.reward if last_eval else 0.0
        if last_eval:
            self.stats["rewards_history"].append(last_eval.reward)

        # Save episode metadata
        episode.eval_result = {
            "success": last_eval.success if last_eval else False,
            "failure_reason": kappa if last_eval else "unknown",
            "reward": last_eval.reward if last_eval else 0.0,
            "reasoning": last_eval.reasoning[:200] if last_eval else "",
            "attempts": attempt + 1,
        }
        episode.save(str(self.output_dir / "episodes"))

    # ------------------------------------------------------------------
    # Scene image extraction
    # ------------------------------------------------------------------

    def _get_scene_image(self, scenario) -> Optional[np.ndarray]:
        """Extract a single scene image from scenario for VLM router."""
        if scenario.real_initial_frames is not None:
            frames = scenario.real_initial_frames
            if isinstance(frames, list) and len(frames) >= 2:
                return np.asarray(frames[1], dtype=np.uint8)
            if isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[0] >= 2:
                return frames[1].astype(np.uint8)
        return None

    def _load_reference_camera_views(
        self,
        video_root: str,
        episode_id: str,
        max_frames: int,
    ) -> Optional[np.ndarray]:
        """Load RGB GT camera sequence as (3, T, H, W, 3) uint8.

        Uses ffmpeg/ffprobe to decode a short prefix from the same video files used
        by retrieval, ensuring the top visualization panel is true original video.
        """
        video_root = Path(video_root)
        camera_names = ["exterior_1_left", "exterior_2_left", "wrist_left"]
        decoded = []

        for cam in camera_names:
            cam_video = video_root / f"observation.images.{cam}" / f"episode_{int(episode_id):06d}.mp4"
            if not cam_video.exists():
                if hasattr(self, "logger"):
                    self.logger.warning("GT camera video missing: %s", cam_video)
                return None

            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(cam_video),
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
            if probe.returncode != 0:
                if hasattr(self, "logger"):
                    self.logger.warning("ffprobe failed for %s: %s", cam_video, probe.stderr.strip())
                return None

            width_str, height_str = probe.stdout.strip().split(",")
            width, height = int(width_str), int(height_str)

            read_cmd = [
                "ffmpeg",
                "-i",
                str(cam_video),
                "-vframes",
                str(max_frames),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-v",
                "quiet",
                "pipe:1",
            ]
            out = subprocess.run(read_cmd, capture_output=True, timeout=45)
            if out.returncode != 0:
                if hasattr(self, "logger"):
                    self.logger.warning("ffmpeg failed for %s", cam_video)
                return None

            frame_size = width * height * 3
            if frame_size <= 0 or len(out.stdout) < frame_size:
                return None

            frame_count = len(out.stdout) // frame_size
            arr = np.frombuffer(out.stdout[: frame_count * frame_size], dtype=np.uint8)
            arr = arr.reshape(frame_count, height, width, 3)
            decoded.append(arr)

        if len(decoded) != 3:
            return None

        min_len = min(v.shape[0] for v in decoded)
        decoded = [v[:min_len] for v in decoded]
        return np.stack(decoded, axis=0)

    # ------------------------------------------------------------------
    # Rollout execution (adapted from the original orchestrator)
    # ------------------------------------------------------------------

    def _execute_rollout(self, task, scenario, policy) -> Episode:
        """Execute policy rollout in world model.

        Runs the policy-in-the-loop imagination loop:
        at each step, query policy for actions, convert to cartesian,
        step world model, decode frames, repeat.
        """
        episode = Episode(
            task_id=task.task_id,
            task_instruction=task.instruction,
            metadata={
                "scenario_id": scenario.episode_id,
                "policy_name": policy.name,
            },
        )

        # Initialize latent history buffer
        # If real_initial_frames are available but latent is zeros, encode them
        if scenario.real_initial_frames is not None and np.all(scenario.initial_frames[0] == 0):
            print("  Encoding initial frames from video...")
            real_views = scenario.real_initial_frames  # List of 3 frames (H, W, 3)
            # Encode 3 camera views to stacked latent (4, 72, 40)
            first_latent = self.world_model.encode_frame(real_views)
            # Convert to correct dtype
            first_latent = first_latent.to(dtype=self.world_model.dtype, device=self.world_model.device)
            # Replicate for history
            history_latents = [first_latent for _ in range(6 * 4)]
            current_latent = first_latent
        else:
            first_latent = torch.from_numpy(scenario.initial_frames[0]).float()
            first_latent = first_latent.to(dtype=self.world_model.dtype, device=self.world_model.device)
            history_latents = [first_latent for _ in range(6 * 4)]
            current_latent = first_latent
        joint_pos = self._ensure_joint_pos(scenario.initial_joints.copy())
        cartesian_pose = self._cartesian_from_joint_pos(joint_pos)

        history_actions = [scenario.initial_state.copy() for _ in range(6 * 4)]

        policy.reset()

        max_steps = self.config.rollout.max_steps
        pred_step = self.config.rollout.pred_step
        rollout_instruction = task.instruction

        frames_collected = []
        current_camera_views = None
        reference_views = None

        # Try loading short GT RGB sequence directly from source videos for top-panel overlay.
        # This keeps the upper panel as the original video instead of decoded latent.
        # Always attempt this when video_path is available; relying on
        # `real_initial_frames` may skip GT-video overlay even when source videos exist.
        if getattr(scenario, "video_path", None):
            try:
                reference_views = self._load_reference_camera_views(
                    scenario.video_path,
                    scenario.episode_id,
                    max_frames=max_steps + pred_step,
                )
            except Exception as e:
                self.logger.warning("Failed to load GT camera sequence: %s", e)
                reference_views = None

        if policy.action_space == "joint_vel" and self.dynamics_model is None:
            self.logger.warning(
                "Action adapter missing; joint_vel actions will become zero cartesian."
            )

        step_advance = max(pred_step - 1, 1)

        for interaction_idx, step in enumerate(range(0, max_steps, pred_step)):
            if interaction_idx % 10 == 0:
                print(f"  Rollout progress: interaction {interaction_idx}, step {step}/{max_steps}", flush=True)
            # Build observation for policy
            if current_camera_views is None:
                if scenario.real_initial_frames is not None:
                    real_views = np.stack(scenario.real_initial_frames, axis=0)
                    obs = self._camera_views_to_obs(real_views, joint_pos)
                else:
                    obs = self._latent_to_obs(current_latent, joint_pos)
            else:
                last_frames = current_camera_views[:, -1, :, :, :]
                obs = self._camera_views_to_obs(last_frames, joint_pos)

            # Policy predicts action and convert to cartesian for world model.
            # For joint-velocity experts, prefer policy's native adapter path
            # (same as rollout_interact_pi.py) to avoid conversion drift.
            if policy.action_space == "joint_vel" and hasattr(policy, "predict_with_adapter"):
                try:
                    policy_in_out, joint_pos_seq, action_seq = policy.predict_with_adapter(
                        obs, rollout_instruction
                    )
                    action_chunk = np.asarray(
                        policy_in_out.get("joint_vel", np.zeros((pred_step, 8), dtype=np.float32)),
                        dtype=np.float32,
                    )
                except Exception as e:
                    self.logger.warning(
                        "predict_with_adapter failed (%s), falling back to generic conversion.",
                        e,
                    )
                    action_chunk = self._infer_action_chunk(
                        policy, obs, rollout_instruction, pred_step
                    )
                    action_seq, joint_pos_seq = self._convert_actions(
                        policy, action_chunk, joint_pos, cartesian_pose, pred_step
                    )
            else:
                action_chunk = self._infer_action_chunk(
                    policy, obs, rollout_instruction, pred_step
                )
                action_seq, joint_pos_seq = self._convert_actions(
                    policy, action_chunk, joint_pos, cartesian_pose, pred_step
                )

            # Shape guards
            action_seq = np.asarray(action_seq, dtype=np.float32)
            if action_seq.ndim != 2 or action_seq.shape[1] != 7:
                raise ValueError(f"action_seq must be (pred_step, 7), got {action_seq.shape}")
            if action_seq.shape[0] != pred_step:
                action_seq = self._pad_or_trim_seq(action_seq, pred_step)

            if joint_pos_seq is not None:
                joint_pos_seq = np.asarray(joint_pos_seq, dtype=np.float32)
                if joint_pos_seq.ndim != 2 or joint_pos_seq.shape[1] != 8:
                    raise ValueError(
                        f"joint_pos_seq must be (pred_step, 8), got {joint_pos_seq.shape}"
                    )
                if joint_pos_seq.shape[0] != pred_step:
                    joint_pos_seq = self._pad_or_trim_seq(joint_pos_seq, pred_step)

            # World model step
            gt_start = interaction_idx * step_advance
            gt_end = gt_start + pred_step

            gt_views = None
            if reference_views is not None:
                gt_view_chunk = reference_views[:, gt_start:gt_end]
                if gt_view_chunk.shape[1] == 0:
                    gt_view_chunk = reference_views[:, -1:]
                if gt_view_chunk.shape[1] < pred_step:
                    pad = np.repeat(gt_view_chunk[:, -1:, :, :, :], pred_step - gt_view_chunk.shape[1], axis=1)
                    gt_view_chunk = np.concatenate([gt_view_chunk, pad], axis=1)
                gt_views = gt_view_chunk

            next_latent, next_frames, camera_views = self.world_model.step(
                action_seq=action_seq,
                current_latent=current_latent,
                history_latents=history_latents,
                history_actions=history_actions,
                text=rollout_instruction,
                ground_truth_views=gt_views,
            )

            current_camera_views = camera_views
            episode.add_step(obs, action_chunk)
            # Match rollout_interact_pi.py behavior: save pred_step-1 frames
            frames_collected.extend(next_frames[:step_advance])

            # Update state
            current_latent = next_latent
            history_latents.append(next_latent)
            history_actions.append(action_seq[-1].copy())

            if joint_pos_seq is not None:
                joint_pos = joint_pos_seq[-1].copy()
            else:
                joint_pos = self._update_joint_pos(joint_pos, action_chunk[0])

        # Keep output length bounded by max_steps for consistency
        frames_collected = frames_collected[:max_steps]
        video_path = self._save_video(frames_collected, task.task_id)
        episode.video_path = str(video_path)
        return episode

    # ------------------------------------------------------------------
    # Action conversion helpers
    # ------------------------------------------------------------------

    def _convert_actions(self, policy, action_chunk, joint_pos, cartesian_pose, pred_step):
        """Convert policy action chunk to cartesian sequence for world model.

        Returns:
            (action_seq, joint_pos_seq) where action_seq is (pred_step, 7)
            cartesian and joint_pos_seq is (pred_step, 8) or None.
        """
        if policy.action_space == "joint_vel":
            return self._adapt_action(action_chunk, joint_pos)
        elif policy.action_space == "cartesian":
            action_seq = self._ensure_cartesian_seq(action_chunk, pred_step)
            joint_pos_seq = self._update_joint_from_cartesian(joint_pos, action_seq)
            return action_seq, joint_pos_seq
        elif policy.action_space == "cartesian_delta":
            delta_seq = self._ensure_cartesian_seq(action_chunk, pred_step)
            action_seq, _ = self._integrate_cartesian_delta(
                cartesian_pose, delta_seq, pred_step
            )
            joint_pos_seq = self._update_joint_from_cartesian(joint_pos, action_seq)
            return action_seq, joint_pos_seq
        elif policy.action_space == "joint_pos":
            joint_pos_seq = self._ensure_joint_pos_seq(action_chunk, pred_step)
            action_seq = self._joint_pos_to_cartesian_seq(joint_pos_seq)
            return action_seq, joint_pos_seq
        else:
            raise ValueError(f"Unsupported action space: {policy.action_space}")

    def _adapt_action(self, action_joint_vel, current_joint_pos):
        """Convert joint velocity to cartesian action via dynamics model + FK."""
        if self.dynamics_model is None:
            pred_step = self.config.rollout.pred_step
            zero_actions = np.zeros((pred_step, 7), dtype=np.float32)
            jp = self._ensure_joint_pos(current_joint_pos)
            return zero_actions, np.tile(jp, (pred_step, 1))

        target_len = 15

        aj = np.asarray(action_joint_vel)
        if aj.ndim == 1:
            aj = aj[None, :]
        if aj.shape[1] < 8:
            aj = np.pad(aj, ((0, 0), (0, 8 - aj.shape[1])), mode="constant")
        elif aj.shape[1] > 8:
            aj = aj[:, :8]

        joint_vel = self._pad_or_trim_seq(aj[:, :7], target_len)
        gripper_pos = self._pad_or_trim_seq(aj[:, 7:], target_len)

        gripper_max = getattr(self.config.rollout, "gripper_max", 1.0)
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        current_joint = current_joint_pos[None, :][:, :7]
        current_gripper = np.array(
            [[current_joint_pos[7] if len(current_joint_pos) > 7 else 0.0]]
        )

        joint_pos_pred = self.dynamics_model(
            current_joint, joint_vel, None, training=False
        )
        if isinstance(joint_pos_pred, torch.Tensor):
            joint_pos_pred = joint_pos_pred.cpu().numpy()

        joint_pos_all = np.concatenate(
            [current_joint, joint_pos_pred], axis=0
        )[:target_len]
        gripper_pos_all = np.concatenate(
            [current_gripper, gripper_pos], axis=0
        )[:target_len]

        # FK: joint_pos -> cartesian_pose
        state_fk = []
        for i in range(joint_pos_all.shape[0]):
            fk_result = get_fk_solution(joint_pos_all[i, :7])
            xyz = fk_result[:3, 3]
            rot = R.from_matrix(fk_result[:3, :3])
            euler = rot.as_euler('xyz')
            state_fk.append(np.concatenate([xyz, euler, gripper_pos_all[i]], axis=0))
        state_fk = np.array(state_fk)

        skip = max(int(getattr(self.config.rollout, "policy_skip_step", 1)), 1)
        pred_step = self.config.rollout.pred_step
        state_fk_skip = state_fk[::skip][:pred_step]
        joint_pos_skip = joint_pos_all[::skip][:pred_step]
        joint_pos_skip = np.concatenate(
            [joint_pos_skip, gripper_pos_all[::skip][:pred_step]], axis=-1
        )
        return state_fk_skip, joint_pos_skip

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _camera_views_to_obs(self, camera_views, joint_pos):
        """Convert camera views (3, H, W, 3) to observation dict for policy."""
        camera_views = np.asarray(camera_views)
        if camera_views.ndim != 4 or camera_views.shape[0] < 3 or camera_views.shape[-1] != 3:
            raise ValueError(f"camera_views must be (3,H,W,3), got {camera_views.shape}")

        image_exterior = camera_views[1]
        image_wrist = camera_views[2]

        def _as_uint8(img):
            img = np.asarray(img)
            if img.dtype == np.uint8:
                return img
            if np.issubdtype(img.dtype, np.floating):
                if float(np.nanmax(img)) <= 1.0:
                    img = np.clip(img, 0.0, 1.0) * 255.0
                else:
                    img = np.clip(img, 0.0, 255.0)
            return img.astype(np.uint8)

        image_exterior = _as_uint8(image_exterior)
        image_wrist = _as_uint8(image_wrist)

        image_exterior = cv2.resize(image_exterior, (320, 180), interpolation=cv2.INTER_LINEAR)
        image_wrist = cv2.resize(image_wrist, (320, 180), interpolation=cv2.INTER_LINEAR)

        return {
            "image_primary": image_tools.resize_with_pad(image_exterior, 224, 224),
            "image_wrist": image_tools.resize_with_pad(image_wrist, 224, 224),
            "joint_pos": joint_pos,
        }

    def _latent_to_obs(self, latent, joint_pos):
        """Decode a single latent to observation dict (initial frame only)."""
        if latent.ndim != 3 or latent.shape[0] != 4 or latent.shape[1] % 3 != 0:
            raise ValueError(
                f"latent must be (4,72,40) with 3 views stacked, got {tuple(latent.shape)}"
            )

        import einops

        latent_split = einops.rearrange(
            latent.unsqueeze(0).unsqueeze(0),
            'b f c (m h) (n w) -> (b m n) f c h w',
            m=3, n=1
        )

        pipeline = self.world_model.model.pipeline
        decoded_views = []
        vae_dtype = getattr(pipeline.vae, 'dtype', None)
        if vae_dtype is None:
            vae_dtype = next(pipeline.vae.parameters()).dtype

        with torch.no_grad():
            for i in range(3):
                view_latent = latent_split[i, 0] / pipeline.vae.config.scaling_factor
                view_latent = view_latent.unsqueeze(0).to(device=pipeline.vae.device, dtype=vae_dtype)
                decoded = pipeline.vae.decode(view_latent, num_frames=1).sample
                decoded = ((decoded / 2.0 + 0.5).clamp(0, 1)).float()

                if decoded.ndim == 5:
                    frame = decoded[0, :, 0].cpu().numpy().transpose(1, 2, 0)
                elif decoded.ndim == 4:
                    frame = decoded[0].cpu().numpy().transpose(1, 2, 0)
                else:
                    raise ValueError(f"Unexpected decoded shape: {decoded.shape}")

                decoded_views.append((frame * 255).astype(np.uint8))

        return self._camera_views_to_obs(np.stack(decoded_views, axis=0), joint_pos)

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def _infer_action_chunk(self, policy, obs, instruction, pred_step: int) -> np.ndarray:
        """Get an action chunk from the selected policy."""
        chunk_size = getattr(policy, "action_chunk_size", pred_step)
        action_chunk = policy.predict_chunk(obs, instruction, chunk_size=chunk_size)
        if action_chunk is None:
            action_chunk = np.zeros((chunk_size, 8), dtype=np.float32)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = np.tile(action_chunk, (chunk_size, 1))
        if action_chunk.shape[1] < 8:
            action_chunk = np.pad(
                action_chunk, ((0, 0), (0, 8 - action_chunk.shape[1])), mode="constant"
            )
        elif action_chunk.shape[1] > 8:
            action_chunk = action_chunk[:, :8]
        if action_chunk.shape[0] != chunk_size:
            action_chunk = self._pad_or_trim_seq(action_chunk, int(chunk_size))
        if not np.isfinite(action_chunk).all():
            self.logger.warning("Non-finite values in action_chunk; replacing with zeros.")
            action_chunk = np.nan_to_num(action_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        return action_chunk

    # ------------------------------------------------------------------
    # Shared helpers (copied/adapted from the original orchestrator)
    # ------------------------------------------------------------------

    def _save_config(self):
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.yaml"
        config_dict = self._config_to_dict(self.config)
        config_dict["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.config.experiment_name,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Configuration saved to: {config_path}")

    def _config_to_dict(self, obj):
        """Recursively convert config object to dictionary."""
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith("_"):
                    result[key] = self._config_to_dict(value)
            return result
        if isinstance(obj, (list, tuple)):
            return [self._config_to_dict(item) for item in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def _setup_logging(self):
        """Setup logging to both console and file."""
        log_path = self.output_dir / "experiment.log"

        logger = logging.getLogger("nexus")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        self.logger = logger

        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 70)

        print(f"✓ Logging to: {log_path}")

    def _pad_or_trim_seq(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        if seq.shape[0] >= target_len:
            return seq[:target_len]
        pad = np.repeat(seq[-1][None, :], target_len - seq.shape[0], axis=0)
        return np.concatenate([seq, pad], axis=0)

    def _ensure_cartesian_seq(self, action_chunk: np.ndarray, pred_step: int) -> np.ndarray:
        seq = np.asarray(action_chunk)
        if seq.ndim == 1:
            seq = seq[None, :]
        if seq.shape[1] < 7:
            seq = np.pad(seq, ((0, 0), (0, 7 - seq.shape[1])))
        if seq.shape[1] > 7:
            if not self._warned_cartesian_dim:
                print(f"Warning: cartesian action dim {seq.shape[1]} > 7, truncating")
                self._warned_cartesian_dim = True
            seq = seq[:, :7]
        seq = self._pad_or_trim_seq(seq, pred_step)
        return seq

    def _cartesian_from_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        joint_pos = self._ensure_joint_pos(joint_pos)
        fk_result = get_fk_solution(joint_pos[:7])
        xyz = fk_result[:3, 3]
        rotation_matrix = fk_result[:3, :3]
        rot = R.from_matrix(rotation_matrix)
        euler = rot.as_euler("xyz")
        gripper = joint_pos[7] if joint_pos.shape[0] > 7 else 0.0
        return np.concatenate([xyz, euler, [gripper]], axis=0)

    def _integrate_cartesian_delta(
        self,
        current_pose: np.ndarray,
        delta_seq: np.ndarray,
        pred_step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate cartesian deltas into absolute poses."""
        seq = self._pad_or_trim_seq(delta_seq, pred_step)
        poses = np.zeros_like(seq)
        pose = np.array(current_pose, dtype=np.float32).copy()
        for i, delta in enumerate(seq):
            pose[:6] = pose[:6] + delta[:6]
            if delta.shape[0] >= 7:
                pose[6] = float(delta[6])
            poses[i] = pose
        return poses, pose

    def _ensure_joint_pos_seq(self, action_chunk: np.ndarray, pred_step: int) -> np.ndarray:
        seq = np.asarray(action_chunk)
        if seq.ndim == 1:
            seq = seq[None, :]
        if seq.shape[1] < 8:
            seq = np.pad(seq, ((0, 0), (0, 8 - seq.shape[1])))
        if seq.shape[1] > 8:
            seq = seq[:, :8]
        seq = self._pad_or_trim_seq(seq, pred_step)
        return seq

    def _joint_pos_to_cartesian_seq(self, joint_pos_seq: np.ndarray) -> np.ndarray:
        cartesian_seq = []
        for joint_pos in joint_pos_seq:
            fk_result = get_fk_solution(joint_pos[:7])
            xyz = fk_result[:3, 3]
            rotation_matrix = fk_result[:3, :3]
            rot = R.from_matrix(rotation_matrix)
            euler = rot.as_euler("xyz")
            gripper = joint_pos[7] if joint_pos.shape[0] > 7 else 0.0
            cartesian_seq.append(np.concatenate([xyz, euler, [gripper]], axis=0))
        return np.array(cartesian_seq)

    def _update_joint_from_cartesian(self, joint_pos: np.ndarray, action_seq: np.ndarray) -> np.ndarray:
        joint_pos = self._ensure_joint_pos(joint_pos)
        if action_seq.shape[1] >= 7:
            joint_pos[7] = float(action_seq[-1][-1])
        return np.tile(joint_pos, (action_seq.shape[0], 1))

    def _ensure_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        """Ensure joint position has 8 dims (including gripper)."""
        joint_pos = np.array(joint_pos, dtype=np.float32).copy()
        if joint_pos.shape[0] < 8:
            joint_pos = np.pad(joint_pos, (0, 8 - joint_pos.shape[0]))
        elif joint_pos.shape[0] > 8:
            joint_pos = joint_pos[:8]
        return joint_pos

    def _update_joint_pos(self, joint_pos, action):
        """Update joint positions given action (Euler integration fallback)."""
        dt = 0.1
        joint_pos = self._ensure_joint_pos(joint_pos)
        updated = joint_pos.copy()
        updated[:7] = joint_pos[:7] + action[:7] * dt
        return updated

    def _save_video(self, frames, task_id):
        """Save frames as MP4 video."""
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / f"{task_id}.mp4"
        output_fps = int(getattr(self.config.rollout, "output_fps", self.default_output_fps))
        output_fps = max(output_fps, 1)

        if len(frames) == 0:
            print("Warning: No frames to save")
            return video_path

        frames_uint8 = []
        for frame in frames:
            if frame.dtype == np.uint8:
                frames_uint8.append(frame)
                continue
            frame = np.clip(frame, 0.0, 1.0)
            frames_uint8.append((frame * 255).astype(np.uint8))

        def _write_opencv() -> None:
            h, w = frames_uint8[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, output_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter could not be opened (mp4v)")
            try:
                for fr in frames_uint8:
                    if fr.shape[:2] != (h, w):
                        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise RuntimeError("opencv wrote empty file")

        def _write_pyav() -> None:
            import av  # type: ignore

            h, w = frames_uint8[0].shape[:2]
            if video_path.exists():
                video_path.unlink()
            container = av.open(str(video_path), mode="w")
            try:
                stream = container.add_stream("libx264", rate=output_fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                for fr in frames_uint8:
                    if fr.shape[:2] != (h, w):
                        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                    av_frame = av.VideoFrame.from_ndarray(fr, format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise RuntimeError("pyav wrote empty file")

        writer_pref = os.environ.get("CTRLWORLD_VIDEO_WRITER", "auto").lower()
        backends: list[tuple[str, callable]] = []
        if writer_pref in {"pyav", "auto"}:
            backends.append(("pyav", _write_pyav))
            backends.append(("opencv", _write_opencv))
        else:
            backends.append(("opencv", _write_opencv))
            backends.append(("pyav", _write_pyav))

        errors = []
        for name, fn in backends:
            try:
                fn()
                errors = []
                break
            except Exception as e:
                errors.append(f"{name} failed: {e}")

        if errors:
            try:
                imageio.mimsave(str(video_path), frames_uint8, fps=output_fps)
                errors = []
            except Exception as e:
                errors.append(f"imageio failed: {e}")

        if errors:
            print("Warning: failed to save video:", " | ".join(errors))
            try:
                frames_dir = video_dir / "frames" / task_id
                frames_dir.mkdir(parents=True, exist_ok=True)
                for i, fr in enumerate(frames_uint8[: min(50, len(frames_uint8))]):
                    cv2.imwrite(
                        str(frames_dir / f"frame_{i:06d}.png"),
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                    )
                print(f"Saved first frames to: {frames_dir}")
            except Exception:
                pass

        return video_path

    def _log_stats(self, iteration):
        """Log current statistics."""
        total_eps = max(self.stats["total_episodes"], 1)
        success_rate = self.stats["successful_episodes"] / total_eps
        wm_failure_rate = self.stats["wm_failures"] / total_eps
        task_failure_rate = self.stats["task_failures"] / total_eps
        scene_failure_rate = self.stats["scene_failures"] / total_eps
        expert_failure_rate = self.stats["expert_failures"] / total_eps
        avg_reward = self.stats["total_reward"] / total_eps
        avg_retries = self.stats["retries_total"] / total_eps

        recent_rewards = self.stats["rewards_history"][-10:]
        recent_avg = float(np.mean(recent_rewards)) if recent_rewards else 0.0

        stats_msg = f"\n{'=' * 60}\n"
        stats_msg += f"Statistics (Iteration {iteration})\n"
        stats_msg += f"{'=' * 60}\n"
        stats_msg += f"Total episodes:     {self.stats['total_episodes']}\n"
        stats_msg += f"Successful:         {self.stats['successful_episodes']} ({success_rate:.1%})\n"
        stats_msg += f"WM failures:        {self.stats['wm_failures']} ({wm_failure_rate:.1%})\n"
        stats_msg += f"Task failures:      {self.stats['task_failures']} ({task_failure_rate:.1%})\n"
        stats_msg += f"Scene failures:     {self.stats['scene_failures']} ({scene_failure_rate:.1%})\n"
        stats_msg += f"Expert failures:    {self.stats['expert_failures']} ({expert_failure_rate:.1%})\n"
        stats_msg += f"Avg retries/task:   {avg_retries:.2f}\n"
        stats_msg += f"Scene resamples:    {self.stats['scene_resamples']}\n"
        stats_msg += f"Expert reroutes:    {self.stats['expert_reroutes']}\n"
        stats_msg += f"Average reward:     {avg_reward:.3f}\n"
        stats_msg += f"Recent avg (10):    {recent_avg:.3f}\n"
        stats_msg += f"Skills discovered:  {len(self.skill_library)}\n"
        stats_msg += f"Memory size:        {len(self.memory)}\n"
        stats_msg += f"Verified set size:  {len(self.verified_set)}\n"
        stats_msg += f"{'=' * 60}\n"

        print(stats_msg)
        if hasattr(self, "logger"):
            self.logger.info(stats_msg)

    def _export_results(self):
        """Export final results and statistics."""
        results_path = self.output_dir / "results.json"

        total_eps = max(self.stats["total_episodes"], 1)
        results = {
            "config": {
                "experiment_name": self.config.experiment_name,
                "num_iterations": self.config.num_iterations,
                "seed": self.config.seed,
            },
            "statistics": {
                "total_episodes": self.stats["total_episodes"],
                "successful_episodes": self.stats["successful_episodes"],
                "wm_failures": self.stats["wm_failures"],
                "task_failures": self.stats["task_failures"],
                "scene_failures": self.stats["scene_failures"],
                "expert_failures": self.stats["expert_failures"],
                "retries_total": self.stats["retries_total"],
                "scene_resamples": self.stats["scene_resamples"],
                "expert_reroutes": self.stats["expert_reroutes"],
                "success_rate": self.stats["successful_episodes"] / total_eps,
                "wm_failure_rate": self.stats["wm_failures"] / total_eps,
                "task_failure_rate": self.stats["task_failures"] / total_eps,
                "scene_failure_rate": self.stats["scene_failures"] / total_eps,
                "expert_failure_rate": self.stats["expert_failures"] / total_eps,
                "avg_reward": self.stats["total_reward"] / total_eps,
            },
            "skill_library": self.skill_library.get_stats(),
            "memory": self.memory.get_stats(),
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        verified_path = self.output_dir / "verified_set.json"
        with open(verified_path, "w") as f:
            json.dump(self.verified_set, f, indent=2)

        print(f"\nResults exported to: {results_path}")
        print(f"Verified set exported to: {verified_path}")
        if hasattr(self, "logger"):
            self.logger.info(f"Results exported to: {results_path}")
            self.logger.info(f"Verified set exported to: {verified_path}")
            self.logger.info("=" * 70)
            self.logger.info(
                f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info(
                f"Final success rate: {results['statistics']['success_rate']:.1%}"
            )
            self.logger.info("=" * 70)

        skill_summary_path = self.output_dir / "skill_library_summary.json"
        self.skill_library.export_summary(str(skill_summary_path))
