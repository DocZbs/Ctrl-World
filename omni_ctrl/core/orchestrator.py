"""Main orchestrator for Omni-Ctrl self-evolution loop."""

import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
import cv2
import imageio
import sys
import logging
from datetime import datetime
import shutil
import yaml

from ..task_generation.template_generator import TemplateTaskGenerator
from ..retrieval.droid_retriever import DROIDRetriever
from ..rollout.world_model_wrapper import WorldModelWrapper
from ..evaluation.openai_vlm_evaluator import OpenAIVLMEvaluator
from .episode import Episode
from .skill_library import SkillLibrary

# Import dynamics model and FK solver
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.action_adapter.train2 import Dynamics
from models.utils import get_fk_solution
from scipy.spatial.transform import Rotation as R
from openpi_client import image_tools


class OmniCtrlOrchestrator:
    """Main self-evolution loop coordinator.

    This is the MVP implementation that coordinates:
    1. Template-based task generation
    2. DROID scenario retrieval
    3. Single policy execution (Pi0.5)
    4. GPT-4V evaluation
    5. Skill library management

    Future versions will add:
    - LLM-based task generation
    - Multi-policy routing with SPO
    - Online policy updates
    """

    def __init__(self, config):
        """Initialize Omni-Ctrl orchestrator.

        Args:
            config: OmniCtrlConfig instance
        """
        print("=" * 60)
        print("Initializing Omni-Ctrl Orchestrator (MVP)")
        print("=" * 60)

        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize components
        print("\n[1/5] Initializing task generator...")
        self.task_generator = TemplateTaskGenerator(config.task_gen.templates_path)

        print("\n[2/5] Initializing DROID retriever...")
        self.retriever = DROIDRetriever(config.retrieval)

        print("\n[3/5] Initializing policy router...")
        from ..policy_router.policy_router import PolicyRouter

        self.router = PolicyRouter(config.router, device=config.device)

        print("\n[4/5] Initializing world model...")
        self.world_model = WorldModelWrapper(config.rollout)

        print("\n[4.5/5] Initializing dynamics model...")
        # Load dynamics model for joint_vel -> joint_pos conversion
        self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(config.device)
        if getattr(config.rollout, "action_adapter_path", None):
            dynamics_ckpt = Path(config.rollout.action_adapter_path)
            if not dynamics_ckpt.is_absolute():
                dynamics_ckpt = Path(__file__).parent.parent.parent / dynamics_ckpt
        else:
            dynamics_ckpt = Path(__file__).parent.parent.parent / "models" / "action_adapter" / "model2_15_9.pth"
        if dynamics_ckpt.exists():
            self.dynamics_model.load_state_dict(torch.load(dynamics_ckpt, map_location=config.device))
            print(f"  Dynamics model loaded from {dynamics_ckpt}")
        else:
            print(f"  ⚠️  Dynamics model checkpoint not found at {dynamics_ckpt}")
            print(f"  Will use zero actions (for testing)")
            self.dynamics_model = None

        print("\n[5/5] Initializing evaluator...")
        if config.evaluation.vlm_type == "dummy":
            from ..evaluation.dummy_evaluator import DummyEvaluator
            self.evaluator = DummyEvaluator(config.evaluation)
        else:
            self.evaluator = OpenAIVLMEvaluator(config.evaluation)

        # Skill library
        self.skill_library = SkillLibrary(config.skill_library_path)

        # Check if using fixed scenario
        self.use_fixed_scenario = (
            hasattr(config, 'fixed_scenario') and
            config.fixed_scenario is not None and
            config.fixed_scenario.enabled
        )
        if self.use_fixed_scenario:
            print("\n[NOTE] Using fixed scenario mode - retrieval will be skipped")
            self.fixed_scenario = self._load_fixed_scenario(
                config.fixed_scenario.annotation_path,
                config.fixed_scenario.droid_root
            )

        # Configuration
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration file to output directory
        self._save_config()

        # Setup logging to file
        self._setup_logging()

        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "wm_failures": 0,
            "vla_failures": 0,
            "total_reward": 0.0,
            "rewards_history": [],
        }
        self._warned_cartesian_dim = False

        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60 + "\n")

    def _save_config(self):
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.yaml"

        # Convert config to dict for saving
        config_dict = self._config_to_dict(self.config)

        # Add metadata
        config_dict['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.config.experiment_name,
        }

        # Save to YAML
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Configuration saved to: {config_path}")

    def _config_to_dict(self, obj):
        """Recursively convert config object to dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._config_to_dict(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def _setup_logging(self):
        """Setup logging to both console and file."""
        log_path = self.output_dir / "experiment.log"

        # Create logger
        logger = logging.getLogger('omni_ctrl')
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        self.logger = logger

        # Log experiment start
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 70)

        print(f"✓ Logging to: {log_path}")

    def _load_fixed_scenario(self, annotation_path, droid_root):
        """Load a fixed scenario from annotation file.

        Args:
            annotation_path: Path to annotation JSON file
            droid_root: Root directory of DROID dataset

        Returns:
            DROIDScenario object
        """
        import json
        from ..retrieval.droid_index import DROIDScenario

        # Load annotation
        anno_path = Path(annotation_path)
        with open(anno_path) as f:
            anno = json.load(f)

        episode_id = anno["episode_id"]
        instruction = anno["texts"][0] if anno.get("texts") else ""

        # Load latent videos
        latent_dir = Path(droid_root) / "latent_videos"
        if "train" in str(anno_path):
            split = "train"
        elif "val" in str(anno_path):
            split = "val"
        else:
            split = "unknown"

        latent_path = latent_dir / split / str(episode_id)

        # Load and concatenate 3 camera views
        latent_files = [latent_path / f"{i}.pt" for i in range(3)]
        if all(f.exists() for f in latent_files):
            latents = [torch.load(f) for f in latent_files]
            # Concatenate along height dimension: (T, 4, 24, 40) x3 -> (T, 4, 72, 40)
            combined_latent = torch.cat(latents, dim=2)
            initial_frames = combined_latent[:6].numpy()  # First 6 frames
        else:
            raise FileNotFoundError(f"Latent files not found in {latent_path}")

        # Load real RGB frames from video files
        # Try mediapy first (better compatibility), fall back to decord
        real_video_frames = []
        for cam_id in range(3):
            video_path = Path(droid_root) / anno["videos"][cam_id]["video_path"]
            try:
                # Try mediapy (better codec support)
                import mediapy
                video = mediapy.read_video(str(video_path))
                first_frame = video[0]  # (H, W, 3)
            except:
                # Fall back to decord
                from decord import VideoReader, cpu
                vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
                try:
                    first_frame = vr.get_batch([0]).asnumpy()[0]  # (H, W, 3)
                except:
                    first_frame = vr.get_batch([0]).numpy()[0]  # (H, W, 3)
            real_video_frames.append(first_frame)

        # Get initial cartesian state (7,)
        states = anno.get("states", [])
        if states:
            initial_state = np.array(states[0], dtype=np.float32)
        else:
            initial_state = np.zeros(7, dtype=np.float32)
        if initial_state.shape[0] < 7:
            initial_state = np.pad(initial_state, (0, 7 - initial_state.shape[0]))
        elif initial_state.shape[0] > 7:
            initial_state = initial_state[:7]

        # Extract joint positions (prefer "joints" for new_setup, fallback to DROID keys)
        joint_positions = None
        for key in ("joints", "observation.state.joint_position", "joint_position"):
            if key in anno:
                joint_positions = anno[key]
                break
        if joint_positions:
            initial_joints = np.array(joint_positions[0], dtype=np.float32)
        else:
            initial_joints = np.zeros(8, dtype=np.float32)
        if initial_joints.shape[0] < 8:
            initial_joints = np.pad(initial_joints, (0, 8 - initial_joints.shape[0]))
        elif initial_joints.shape[0] > 8:
            initial_joints = initial_joints[:8]

        scenario = DROIDScenario(
            episode_id=str(episode_id),
            instruction=instruction,
            initial_state=initial_state,
            initial_joints=initial_joints,
            initial_frames=initial_frames,  # latent frames (for history buffer)
            real_initial_frames=real_video_frames,  # RGB frames (for policy)
            video_path=str(Path(droid_root) / "videos" / split / str(episode_id)),
            latent_path=str(latent_path),
        )

        print(f"  Loaded fixed scenario: {episode_id}")
        print(f"  Instruction: {instruction}")
        print(f"  Initial frames shape: {initial_frames.shape}")

        return scenario

    def run(self):
        """Run the main self-evolution loop."""
        print(f"Starting Omni-Ctrl evolution for {self.config.num_iterations} iterations...")
        print(f"Output directory: {self.output_dir}\n")

        for iteration in tqdm(range(self.config.num_iterations), desc="Evolution"):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            try:
                # Step 1: Generate task
                task = self.task_generator.generate_task()
                print(f"\n[Task Generated]")
                print(f"  ID: {task.task_id}")
                print(f"  Instruction: {task.instruction}")
                print(f"  Difficulty: {task.difficulty:.2f}")

                # Step 2: Retrieve or use fixed scenario
                if self.use_fixed_scenario:
                    scenario = self.fixed_scenario
                    self._align_task_with_scenario(task, scenario)
                    print(f"\n[Using Fixed Scenario]")
                    print(f"  Episode ID: {scenario.episode_id}")
                    print(f"  Scenario instruction: {scenario.instruction}")
                    print(f"  Task instruction: {task.instruction}")
                else:
                    scenario = self.retriever.retrieve(task)
                    print(f"\n[Scenario Retrieved]")
                    print(f"  Episode ID: {scenario.episode_id}")
                    print(f"  Original instruction: {scenario.instruction}")

                # Step 3: Select policy and execute rollout
                policy = self.router.select_policy(task)
                print(f"\n[Executing Rollout]")
                print(f"  Policy: {policy.name}")
                episode = self._execute_rollout(task, scenario, policy)
                print(f"  Rollout completed: {episode.get_length()} steps")
                print(f"  Video saved: {episode.video_path}")

                # Step 4: Evaluate
                print(f"\n[Evaluating with GPT-4V]")
                eval_result = self.evaluator.evaluate(episode.video_path, task)
                print(f"  Success: {eval_result.success}")
                print(f"  Failure Reason: {eval_result.failure_reason}")
                print(f"  Consistency: {eval_result.physical_consistency:.2f}")
                print(f"  Reward: {eval_result.reward:.3f}")
                print(f"  Reasoning: {eval_result.reasoning[:100]}...")

                # Step 5: Update statistics and skill library
                self.stats["total_episodes"] += 1
                self.stats["total_reward"] += eval_result.reward
                self.stats["rewards_history"].append(eval_result.reward)

                if eval_result.success:
                    self.stats["successful_episodes"] += 1
                    self.skill_library.add(task, episode, eval_result)
                else:
                    # Track failure reasons
                    if eval_result.failure_reason == "wm_failure":
                        self.stats["wm_failures"] += 1
                    elif eval_result.failure_reason == "vla_failure":
                        self.stats["vla_failures"] += 1

                self.router.update(policy.name, eval_result.reward)

                # Save episode metadata
                episode.eval_result = eval_result.to_dict() if hasattr(eval_result, 'to_dict') else {
                    "success": eval_result.success,
                    "failure_reason": eval_result.failure_reason,
                    "consistency": eval_result.physical_consistency,
                    "reward": eval_result.reward,
                    "reasoning": eval_result.reasoning,
                }
                episode.save(str(self.output_dir / "episodes"))

                # Log statistics periodically
                if (iteration + 1) % self.config.log_frequency == 0:
                    self._log_stats(iteration + 1)

            except Exception as e:
                print(f"\nError in iteration {iteration}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print("\n" + "=" * 60)
        print("Evolution complete!")
        print("=" * 60)
        self._log_stats(self.config.num_iterations)
        self._export_results()

    def _execute_rollout(self, task, scenario, policy) -> Episode:
        """Execute policy rollout in world model.

        Args:
            task: Task object
            scenario: DROIDScenario object

        Returns:
            Episode object with trajectory
        """
        # Initialize episode
        episode = Episode(
            task_id=task.task_id,
            task_instruction=task.instruction,
            metadata={
                "scenario_id": scenario.episode_id,
                "policy_name": policy.name,
            },
        )

        # Initialize state from scenario
        # Following rollout_interact_pi.py: initialize history buffer with first frame repeated
        # num_history * 4 = 6 * 4 = 24 times (to allow history_idx = [0,0,-12,-9,-6,-3])
        first_latent = torch.from_numpy(scenario.initial_frames[0]).float()  # (4, 72, 40)
        history_latents = [first_latent for _ in range(6 * 4)]  # 24 frames in history
        current_latent = first_latent
        joint_pos = self._ensure_joint_pos(scenario.initial_joints.copy())
        cartesian_pose = self._cartesian_from_joint_pos(joint_pos)

        # Initialize history actions (use initial state repeated)
        history_actions = [scenario.initial_state.copy() for _ in range(6 * 4)]  # 24 actions

        # Reset policy
        policy.reset()

        # Rollout parameters
        max_steps = self.config.rollout.max_steps
        pred_step = self.config.rollout.pred_step
        rollout_instruction = scenario.instruction if self.use_fixed_scenario else task.instruction

        frames_collected = []
        current_camera_views = None  # Will be set after first world model step

        for step in range(0, max_steps, pred_step):
            # For first step, use initial scenario images by decoding latent
            # For subsequent steps, use world model predicted frames
            if current_camera_views is None:
                # First step: prefer real frames for policy input if available
                if scenario.real_initial_frames is not None:
                    real_views = np.stack(scenario.real_initial_frames, axis=0)
                    obs = self._camera_views_to_obs(real_views, joint_pos)
                else:
                    obs = self._latent_to_obs(current_latent, joint_pos)
            else:
                # Subsequent steps: use last frame from world model predictions
                # current_camera_views shape: (3, pred_step, H, W, 3)
                # Take last frame from each camera: (3, H, W, 3)
                last_frames = current_camera_views[:, -1, :, :, :]  # (3, H, W, 3)
                obs = self._camera_views_to_obs(last_frames, joint_pos)

            # Policy predicts action chunk
            action_chunk = self._infer_action_chunk(policy, obs, rollout_instruction, pred_step)

            # Convert to cartesian action sequence for world model
            if policy.action_space == "joint_vel":
                action_seq, joint_pos_seq = self._adapt_action(action_chunk, joint_pos)
            elif policy.action_space == "cartesian":
                action_seq = self._ensure_cartesian_seq(action_chunk, pred_step)
                joint_pos_seq = self._update_joint_from_cartesian(joint_pos, action_seq)
            elif policy.action_space == "cartesian_delta":
                delta_seq = self._ensure_cartesian_seq(action_chunk, pred_step)
                action_seq, cartesian_pose = self._integrate_cartesian_delta(cartesian_pose, delta_seq, pred_step)
                joint_pos_seq = self._update_joint_from_cartesian(joint_pos, action_seq)
            elif policy.action_space == "joint_pos":
                joint_pos_seq = self._ensure_joint_pos_seq(action_chunk, pred_step)
                action_seq = self._joint_pos_to_cartesian_seq(joint_pos_seq)
            else:
                raise ValueError(f"Unsupported action space: {policy.action_space}")

            # World model step (returns all pred_step frames and camera views)
            next_latent, next_frames, camera_views = self.world_model.step(
                action_seq=action_seq,
                current_latent=current_latent,
                history_latents=history_latents,
                history_actions=history_actions,
                text=rollout_instruction,
            )

            # Update current camera views for next iteration
            current_camera_views = camera_views  # (3, pred_step, H, W, 3)

            # Record step (using action chunk from policy)
            episode.add_step(obs, action_chunk)
            # Collect all predicted frames
            frames_collected.extend(next_frames)

            # Update state (following rollout_interact_pi.py)
            current_latent = next_latent
            # Append new latent to history buffer (keep growing, no pop)
            history_latents.append(next_latent)
            # Append last action in sequence to history
            history_actions.append(action_seq[-1].copy())

            # Update joint positions using action_chunk (first action's joint_vel)
            if joint_pos_seq is not None:
                joint_pos = joint_pos_seq[-1].copy()
            else:
                joint_pos = self._update_joint_pos(joint_pos, action_chunk[0])

        # Save video
        video_path = self._save_video(frames_collected, task.task_id)
        episode.video_path = str(video_path)

        return episode

    def _align_task_with_scenario(self, task, scenario) -> None:
        """Align task metadata to fixed scenario."""
        if scenario.instruction:
            task.instruction = scenario.instruction
            task.success_criteria = [scenario.instruction]
            task.object_categories = []

    def _camera_views_to_obs(self, camera_views, joint_pos):
        """Convert camera views to observation dict for policy.

        Args:
            camera_views: Array of shape (3, H, W, 3) - 3 camera views
            joint_pos: Joint positions array (7,)

        Returns:
            Observation dictionary for policy
        """
        # Following rollout_interact_pi.py line 230-246:
        # - videos[1] = exterior camera (camera 1)
        # - videos[2] = wrist camera (camera 2)
        image_exterior = camera_views[1]  # (H, W, 3) uint8
        image_wrist = camera_views[2]     # (H, W, 3) uint8

        # Resize to 180x320 (following rollout_interact_pi.py line 235-236)
        import torch
        image_exterior = torch.from_numpy(image_exterior).to(torch.uint8)
        image_wrist = torch.from_numpy(image_wrist).to(torch.uint8)

        image_exterior = torch.nn.functional.interpolate(
            image_exterior.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        image_wrist = torch.nn.functional.interpolate(
            image_wrist.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        from openpi_client import image_tools
        return {
            "image_primary": image_tools.resize_with_pad(image_exterior, 224, 224),
            "image_wrist": image_tools.resize_with_pad(image_wrist, 224, 224),
            "joint_pos": joint_pos,
        }

    def _latent_to_obs(self, latent, joint_pos):
        """Decode latent to observation dict (used for initial frame only).

        Args:
            latent: Latent tensor (4, 72, 40) - 3 views concatenated
            joint_pos: Joint positions array (7,)

        Returns:
            Observation dictionary for policy
        """
        import einops

        # Split the 3-view latent: (4, 72, 40) -> (3, 4, 24, 40)
        latent_split = einops.rearrange(
            latent.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 72, 40)
            'b f c (m h) (n w) -> (b m n) f c h w',
            m=3, n=1
        )  # (3, 1, 4, 24, 40)

        # Decode each view
        pipeline = self.world_model.model.pipeline
        decoded_views = []

        with torch.no_grad():
            for i in range(3):
                # view_latent shape: (1, 1, 4, 24, 40)
                # Need to squeeze frame dimension for single-frame decode: (1, 4, 24, 40)
                view_latent = latent_split[i, 0] / pipeline.vae.config.scaling_factor  # (4, 24, 40)
                view_latent = view_latent.unsqueeze(0).to(pipeline.vae.device)  # (1, 4, 24, 40) on correct device

                decoded = pipeline.vae.decode(view_latent, num_frames=1).sample  # output shape varies
                decoded = ((decoded / 2.0 + 0.5).clamp(0, 1))  # Normalize to [0, 1]

                # Handle different possible output shapes
                if decoded.ndim == 5:  # (B, C, T, H, W)
                    frame = decoded[0, :, 0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                elif decoded.ndim == 4:  # (B, C, H, W)
                    frame = decoded[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                else:
                    raise ValueError(f"Unexpected decoded shape: {decoded.shape}")

                frame_uint8 = (frame * 255).astype(np.uint8)
                decoded_views.append(frame_uint8)

        # Following rollout_interact_pi.py:
        # - videos[1] = exterior camera (camera 1)
        # - videos[2] = wrist camera (camera 2)
        image_exterior = decoded_views[1]  # (192, 320, 3)
        image_wrist = decoded_views[2]     # (192, 320, 3)

        # Resize to 180x320 then pad to 224x224 (following rollout_interact_pi.py line 235-236)
        image_exterior = torch.from_numpy(image_exterior).to(torch.uint8)
        image_wrist = torch.from_numpy(image_wrist).to(torch.uint8)

        image_exterior = torch.nn.functional.interpolate(
            image_exterior.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        image_wrist = torch.nn.functional.interpolate(
            image_wrist.permute(2, 0, 1).unsqueeze(0).float(),
            size=(180, 320), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        return {
            "image_primary": image_tools.resize_with_pad(image_exterior, 224, 224),
            "image_wrist": image_tools.resize_with_pad(image_wrist, 224, 224),
            "joint_pos": joint_pos,
        }

    def _adapt_action(self, action_joint_vel, current_joint_pos):
        """Convert joint velocity to cartesian action.

        Uses dynamics model to predict joint positions from velocities,
        then FK to convert to cartesian poses.

        Args:
            action_joint_vel: Joint velocity action chunk (N, 8) from policy
            current_joint_pos: Current joint positions (8,) including gripper

        Returns:
            Tuple of:
                - Cartesian action sequence (pred_step, 7)
                - Joint position sequence (pred_step, 8)
        """
        if self.dynamics_model is None:
            # Fallback: return zero actions and keep joints unchanged
            pred_step = self.config.rollout.pred_step
            zero_actions = np.zeros((pred_step, 7), dtype=np.float32)
            joint_pos = self._ensure_joint_pos(current_joint_pos)
            joint_seq = np.tile(joint_pos, (pred_step, 1))
            return zero_actions, joint_seq

        # Following rollout_interact_pi.py line 246-287
        # Policy outputs (10, 8) but we need 15 steps for dynamics model
        # Use idx to extend/repeat action chunks
        idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])  # for pi0.5, we need 15 steps

        # Separate joint_vel and gripper
        joint_vel = action_joint_vel[:, :7]  # (10, 7)
        gripper_pos = action_joint_vel[:, 7:]  # (10, 1)

        # Extend to 15 steps using modulo indexing
        idx_mod = idx % len(joint_vel)  # [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4]
        joint_vel = joint_vel[idx_mod]  # (15, 7)
        gripper_pos = gripper_pos[idx_mod]  # (15, 1)

        # Clip gripper
        gripper_max = getattr(self.config.rollout, "gripper_max", 1.0)
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)

        # Use dynamics model to predict future joint positions
        # Dynamics model expects numpy inputs (it will convert to torch internally)
        current_joint = current_joint_pos[None, :][:, :7]  # (1, 7)
        current_gripper = np.array([[current_joint_pos[7] if len(current_joint_pos) > 7 else 0.0]])

        # Call dynamics model with numpy inputs
        joint_pos_pred = self.dynamics_model(
            current_joint,  # numpy (1, 7)
            joint_vel,      # numpy (15, 7)
            None,
            training=False
        )  # output is torch tensor on device

        # Convert to numpy
        if isinstance(joint_pos_pred, torch.Tensor):
            joint_pos_pred = joint_pos_pred.cpu().numpy()  # (15, 7)

        # Concatenate current + predicted
        joint_pos_all = np.concatenate([current_joint, joint_pos_pred], axis=0)[:15]  # (15, 7)
        gripper_pos_all = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)

        # Forward kinematics: joint_pos -> cartesian_pose
        state_fk = []
        for i in range(joint_pos_all.shape[0]):
            fk_result = get_fk_solution(joint_pos_all[i, :7])  # (4, 4) transformation matrix
            xyz = fk_result[:3, 3]  # position
            rotation_matrix = fk_result[:3, :3]  # rotation
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')  # roll, pitch, yaw
            state_fk.append(np.concatenate([xyz, euler, gripper_pos_all[i]], axis=0))

        state_fk = np.array(state_fk)  # (15, 7)

        # Skip and select pred_step frames (following rollout_interact_pi.py line 286)
        skip = self.config.rollout.policy_skip_step
        pred_step = self.config.rollout.pred_step
        state_fk_skip = state_fk[::skip][:pred_step]  # (pred_step, 7)
        joint_pos_skip = joint_pos_all[::skip][:pred_step]  # (pred_step, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, gripper_pos_all[::skip][:pred_step]], axis=-1)  # (pred_step, 8)

        return state_fk_skip, joint_pos_skip

    def _infer_action_chunk(self, policy, obs, instruction, pred_step: int) -> np.ndarray:
        """Get an action chunk from the selected policy."""
        chunk_size = getattr(policy, "action_chunk_size", pred_step)
        action_chunk = policy.predict_chunk(obs, instruction, chunk_size=chunk_size)
        if action_chunk is None:
            action_chunk = np.zeros((chunk_size, 8), dtype=np.float32)
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim == 1:
            action_chunk = np.tile(action_chunk, (chunk_size, 1))
        return action_chunk

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
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('xyz')
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
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
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
        dt = 0.1  # Time step
        joint_pos = self._ensure_joint_pos(joint_pos)
        updated = joint_pos.copy()
        updated[:7] = joint_pos[:7] + action[:7] * dt
        return updated

    def _save_video(self, frames, task_id):
        """Save frames as MP4 video.

        Args:
            frames: List of RGB frames (H, W, 3) in [0, 1] or uint8 [0, 255]
            task_id: Task ID for filename

        Returns:
            Path to saved video
        """
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / f"{task_id}.mp4"

        if len(frames) == 0:
            print("Warning: No frames to save")
            return video_path

        # Normalize frames to uint8 [0, 255]
        frames_uint8 = []
        for frame in frames:
            if frame.dtype == np.uint8:
                frames_uint8.append(frame)
                continue
            frame = np.clip(frame, 0.0, 1.0)
            frames_uint8.append((frame * 255).astype(np.uint8))

        # Use imageio to save video (better codec compatibility)
        try:
            imageio.mimsave(
                str(video_path),
                frames_uint8,
                fps=5,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
        except Exception as e:
            print(f"Warning: imageio failed ({e}), falling back to OpenCV")
            # Fallback to OpenCV
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 5, (w, h))
            for frame in frames:
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()

        return video_path

    def _log_stats(self, iteration):
        """Log current statistics.

        Args:
            iteration: Current iteration number
        """
        total_eps = max(self.stats["total_episodes"], 1)
        success_rate = self.stats["successful_episodes"] / total_eps
        wm_failure_rate = self.stats["wm_failures"] / total_eps
        vla_failure_rate = self.stats["vla_failures"] / total_eps
        avg_reward = self.stats["total_reward"] / total_eps

        # Recent performance (last 10 episodes)
        recent_rewards = self.stats["rewards_history"][-10:]
        recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0

        stats_msg = f"\n{'=' * 60}\n"
        stats_msg += f"Statistics (Iteration {iteration})\n"
        stats_msg += f"{'=' * 60}\n"
        stats_msg += f"Total episodes:     {self.stats['total_episodes']}\n"
        stats_msg += f"Successful:         {self.stats['successful_episodes']} ({success_rate:.1%})\n"
        stats_msg += f"WM failures:        {self.stats['wm_failures']} ({wm_failure_rate:.1%})\n"
        stats_msg += f"VLA failures:       {self.stats['vla_failures']} ({vla_failure_rate:.1%})\n"
        stats_msg += f"Average reward:     {avg_reward:.3f}\n"
        stats_msg += f"Recent avg (10):    {recent_avg:.3f}\n"
        stats_msg += f"Skills discovered:  {len(self.skill_library)}\n"
        stats_msg += f"{'=' * 60}\n"

        print(stats_msg)
        if hasattr(self, 'logger'):
            self.logger.info(stats_msg)

    def _export_results(self):
        """Export final results and statistics."""
        results_path = self.output_dir / "results.json"

        import json

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
                "vla_failures": self.stats["vla_failures"],
                "success_rate": self.stats["successful_episodes"] / total_eps,
                "wm_failure_rate": self.stats["wm_failures"] / total_eps,
                "vla_failure_rate": self.stats["vla_failures"] / total_eps,
                "avg_reward": self.stats["total_reward"] / total_eps,
                "skills_discovered": len(self.skill_library),
            },
            "skill_library": self.skill_library.get_stats(),
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults exported to: {results_path}")
        if hasattr(self, 'logger'):
            self.logger.info(f"Results exported to: {results_path}")
            self.logger.info("=" * 70)
            self.logger.info(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Final success rate: {results['statistics']['success_rate']:.1%}")
            self.logger.info("=" * 70)

        # Export skill library summary
        skill_summary_path = self.output_dir / "skill_library_summary.json"
        self.skill_library.export_summary(str(skill_summary_path))
