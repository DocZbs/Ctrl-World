"""Main orchestrator for Omni-Ctrl self-evolution loop."""

import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import cv2

from ..task_generation.template_generator import TemplateTaskGenerator
from ..retrieval.droid_retriever import DROIDRetriever
from ..rollout.world_model_wrapper import WorldModelWrapper
from ..evaluation.gpt4v_evaluator import GPT4VEvaluator
from .episode import Episode
from .skill_library import SkillLibrary


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

        print("\n[3/5] Initializing policy...")
        # MVP: Use single policy (no router yet)
        from ..policy_router.pi05_policy import Pi05Policy

        if not config.router.available_policies:
            raise ValueError("No policies configured in router.available_policies")

        policy_config = config.router.available_policies[0]
        self.policy = Pi05Policy(policy_config.checkpoint, config.device)

        print("\n[4/5] Initializing world model...")
        self.world_model = WorldModelWrapper(config.rollout)

        print("\n[5/5] Initializing evaluator...")
        self.evaluator = GPT4VEvaluator(config.evaluation)

        # Skill library
        self.skill_library = SkillLibrary(config.skill_library_path)

        # Configuration
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_reward": 0.0,
            "rewards_history": [],
        }

        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60 + "\n")

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

                # Step 2: Retrieve scenario
                scenario = self.retriever.retrieve(task)
                print(f"\n[Scenario Retrieved]")
                print(f"  Episode ID: {scenario.episode_id}")
                print(f"  Original instruction: {scenario.instruction}")

                # Step 3: Execute rollout
                print(f"\n[Executing Rollout]")
                episode = self._execute_rollout(task, scenario)
                print(f"  Rollout completed: {episode.get_length()} steps")
                print(f"  Video saved: {episode.video_path}")

                # Step 4: Evaluate
                print(f"\n[Evaluating with GPT-4V]")
                eval_result = self.evaluator.evaluate(episode.video_path, task)
                print(f"  Success: {eval_result.success}")
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

                # Save episode metadata
                episode.eval_result = eval_result.to_dict() if hasattr(eval_result, 'to_dict') else {
                    "success": eval_result.success,
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

    def _execute_rollout(self, task, scenario) -> Episode:
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
                "policy_name": self.policy.name,
            },
        )

        # Initialize state from scenario
        history_latents = [torch.from_numpy(f).float() for f in scenario.initial_frames]
        current_latent = history_latents[-1]
        joint_pos = scenario.initial_joints.copy()

        # Reset policy
        self.policy.reset()

        # Rollout parameters
        max_steps = self.config.rollout.max_steps
        pred_step = self.config.rollout.pred_step

        frames_collected = []

        for step in range(0, max_steps, pred_step):
            # Get observation (for MVP, use placeholder decoded frames)
            obs = self._latent_to_obs(current_latent, joint_pos)

            # Policy predicts action
            action = self.policy.predict(obs, task.instruction)

            # Adapt action (joint_vel -> cartesian)
            # For MVP, use simplified conversion
            action_cartesian = self._adapt_action(action, joint_pos)

            # Prepare action sequence (repeat for pred_step)
            action_seq = np.tile(action_cartesian, (pred_step, 1))

            # World model step
            next_latent, next_frame = self.world_model.step(
                action_seq=action_seq,
                current_latent=current_latent,
                history_latents=history_latents,
                text=task.instruction,
            )

            # Record step
            episode.add_step(obs, action)
            frames_collected.append(next_frame)

            # Update state
            current_latent = next_latent
            history_latents.append(next_latent)

            # Update joint positions (simple integration)
            joint_pos = self._update_joint_pos(joint_pos, action)

        # Save video
        video_path = self._save_video(frames_collected, task.task_id)
        episode.video_path = str(video_path)

        return episode

    def _latent_to_obs(self, latent, joint_pos):
        """Decode latent to observation dict.

        For MVP, we use placeholder observations.
        Future versions will properly decode latents to RGB.

        Args:
            latent: Latent tensor
            joint_pos: Joint positions array

        Returns:
            Observation dictionary
        """
        # TODO: Properly decode latent using world model VAE
        # For now, return placeholder
        return {
            "image_primary": np.zeros((224, 224, 3), dtype=np.uint8),
            "image_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "joint_pos": joint_pos,
        }

    def _adapt_action(self, action_joint_vel, current_joint_pos):
        """Convert joint velocity to cartesian action.

        For MVP, we use a simplified conversion.
        Future versions will use the dynamics model or FK.

        Args:
            action_joint_vel: Joint velocity action (7 or 8,)
            current_joint_pos: Current joint positions (7,)

        Returns:
            Cartesian action (7,) - x, y, z, roll, pitch, yaw, gripper
        """
        # TODO: Use proper dynamics model from models/action_adapter/
        # For MVP, return placeholder cartesian action
        cartesian_action = np.zeros(7, dtype=np.float32)

        # If action has gripper, copy it
        if len(action_joint_vel) >= 8:
            cartesian_action[6] = action_joint_vel[7]  # Gripper

        return cartesian_action

    def _update_joint_pos(self, joint_pos, action):
        """Update joint positions given action.

        Simple Euler integration for MVP.

        Args:
            joint_pos: Current joint positions (7,)
            action: Joint velocity action

        Returns:
            Updated joint positions (7,)
        """
        dt = 0.1  # Time step
        return joint_pos + action[:7] * dt

    def _save_video(self, frames, task_id):
        """Save frames as MP4 video.

        Args:
            frames: List of RGB frames (H, W, 3) in [0, 1] range
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

        # Get frame dimensions
        h, w = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 5, (w, h))

        # Write frames
        for frame in frames:
            # Convert [0, 1] float to [0, 255] uint8
            frame_uint8 = (frame * 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
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
        avg_reward = self.stats["total_reward"] / total_eps

        # Recent performance (last 10 episodes)
        recent_rewards = self.stats["rewards_history"][-10:]
        recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0

        print(f"\n{'=' * 60}")
        print(f"Statistics (Iteration {iteration})")
        print(f"{'=' * 60}")
        print(f"Total episodes:     {self.stats['total_episodes']}")
        print(f"Successful:         {self.stats['successful_episodes']} ({success_rate:.1%})")
        print(f"Average reward:     {avg_reward:.3f}")
        print(f"Recent avg (10):    {recent_avg:.3f}")
        print(f"Skills discovered:  {len(self.skill_library)}")
        print(f"{'=' * 60}\n")

    def _export_results(self):
        """Export final results and statistics."""
        results_path = self.output_dir / "results.json"

        import json

        results = {
            "config": {
                "experiment_name": self.config.experiment_name,
                "num_iterations": self.config.num_iterations,
                "seed": self.config.seed,
            },
            "statistics": {
                "total_episodes": self.stats["total_episodes"],
                "successful_episodes": self.stats["successful_episodes"],
                "success_rate": (
                    self.stats["successful_episodes"] / max(self.stats["total_episodes"], 1)
                ),
                "avg_reward": (
                    self.stats["total_reward"] / max(self.stats["total_episodes"], 1)
                ),
                "skills_discovered": len(self.skill_library),
            },
            "skill_library": self.skill_library.get_stats(),
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults exported to: {results_path}")

        # Export skill library summary
        skill_summary_path = self.output_dir / "skill_library_summary.json"
        self.skill_library.export_summary(str(skill_summary_path))
