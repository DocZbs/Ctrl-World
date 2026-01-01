"""
Omni-Ctrl MVP Experiment Runner

This script runs the minimal viable product (MVP) version of Omni-Ctrl,
which implements the core self-evolution loop with:
- Template-based task generation
- DROID scenario retrieval
- Single policy execution (Pi0.5)
- GPT-4V evaluation
- Skill library management

Usage:
    python experiments/run_omni_ctrl_mvp.py
    python experiments/run_omni_ctrl_mvp.py --config path/to/config.yaml
    python experiments/run_omni_ctrl_mvp.py --iterations 50 --output experiments/my_run
"""

import sys
import argparse
from pathlib import Path

# Add Ctrl-World root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omni_ctrl.configs.base_config import (
    OmniCtrlConfig,
    PolicyConfig,
    TaskGenConfig,
    RetrievalConfig,
    RolloutConfig,
    EvaluationConfig,
    RouterConfig,
)
from omni_ctrl.core.orchestrator import OmniCtrlOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Omni-Ctrl MVP experiment")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (overrides config)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda:0 (overrides config)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-4V (overrides config)",
    )

    return parser.parse_args()


def create_default_config():
    """Create default configuration for MVP.

    YOU MUST UPDATE THESE PATHS BEFORE RUNNING!
    """
    config = OmniCtrlConfig(
        experiment_name="omni_ctrl_mvp",
        output_dir="experiments/omni_ctrl_mvp",
        num_iterations=20,  # Start small for testing
        seed=42,
        device="cuda:0",
        log_frequency=5,
    )

    # Task generation config
    config.task_gen = TaskGenConfig(
        type="template",
        templates_path="omni_ctrl/task_generation/templates.json",
    )

    # Retrieval config
    config.retrieval = RetrievalConfig(
        droid_path="dataset_example/droid_subset",
        index_path="experiments/droid_index",
        clip_model_path="openai/clip-vit-base-patch32",
        top_k=5,
        rebuild_index=False,
    )

    # Rollout config
    # ⚠️ UPDATE THESE PATHS TO YOUR ACTUAL MODEL CHECKPOINTS!
    config.rollout = RolloutConfig(
        wm_ckpt="/path/to/your/ctrl_world_checkpoint.pt",  # TODO: UPDATE THIS!
        svd_model_path="/path/to/your/svd_model",  # TODO: UPDATE THIS!
        clip_model_path="/path/to/your/clip_model",  # TODO: UPDATE THIS!
        data_stat_path="dataset_meta_info/droid/stat.json",
        max_steps=50,
        pred_step=5,
        policy_skip_step=2,
        action_adapter_path="models/action_adapter/model2_15_9.pth",
        device="cuda:0",
    )

    # Evaluation config
    # ⚠️ SET YOUR OPENAI API KEY!
    config.evaluation = EvaluationConfig(
        vlm_type="gpt4v",
        api_key="your-openai-api-key-here",  # TODO: UPDATE THIS!
        max_retries=3,
        timeout=60,
        success_weight=1.0,
        consistency_weight=0.5,
    )

    # Router config (single policy for MVP)
    # ⚠️ UPDATE PATH TO YOUR PI0.5 CHECKPOINT!
    config.router = RouterConfig(
        available_policies=[
            PolicyConfig(
                name="pi05",
                checkpoint="/path/to/your/pi05_checkpoint",  # TODO: UPDATE THIS!
                action_space="joint_vel",
            )
        ],
        selection_strategy="round_robin",
    )

    # Skill library
    config.skill_library_path = "experiments/omni_ctrl_mvp/skills"

    return config


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print(" " * 20 + "OMNI-CTRL MVP RUNNER")
    print("=" * 70)

    # Load configuration
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = OmniCtrlConfig.from_yaml(args.config)
    else:
        print("\nUsing default configuration")
        print("⚠️  Remember to update paths in create_default_config()!")
        config = create_default_config()

    # Override config with command line args
    if args.iterations is not None:
        config.num_iterations = args.iterations
        print(f"Overriding iterations: {args.iterations}")

    if args.output is not None:
        config.output_dir = args.output
        print(f"Overriding output dir: {args.output}")

    if args.seed is not None:
        config.seed = args.seed
        print(f"Overriding seed: {args.seed}")

    if args.device is not None:
        config.device = args.device
        config.rollout.device = args.device
        print(f"Overriding device: {args.device}")

    if args.api_key is not None:
        config.evaluation.api_key = args.api_key
        print("OpenAI API key provided via command line")

    # Validate critical paths
    print("\nValidating configuration...")
    validation_errors = []

    if "your-openai-api-key" in config.evaluation.api_key:
        validation_errors.append("OpenAI API key not set! Please update config.")

    if "/path/to/" in config.rollout.wm_ckpt:
        validation_errors.append("World model checkpoint path not set!")

    if "/path/to/" in str(config.router.available_policies[0].checkpoint):
        validation_errors.append("Policy checkpoint path not set!")

    if validation_errors:
        print("\n⚠️  Configuration Errors:")
        for error in validation_errors:
            print(f"  - {error}")
        print("\nPlease update the paths in create_default_config() or use a config file.")
        print("Exiting...\n")
        return

    print("Configuration validated ✓")

    # Print config summary
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Experiment:     {config.experiment_name}")
    print(f"Iterations:     {config.num_iterations}")
    print(f"Output dir:     {config.output_dir}")
    print(f"Device:         {config.device}")
    print(f"Seed:           {config.seed}")
    print(f"Task generator: {config.task_gen.type}")
    print(f"VLM evaluator:  {config.evaluation.vlm_type}")
    print(f"Policy:         {config.router.available_policies[0].name}")
    print("=" * 70 + "\n")

    # Initialize orchestrator
    try:
        orchestrator = OmniCtrlOrchestrator(config)
    except Exception as e:
        print(f"\n❌ Failed to initialize orchestrator: {e}")
        import traceback

        traceback.print_exc()
        return

    # Run evolution loop
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to: {config.output_dir}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Save final statistics even if interrupted
        print(f"\nFinal results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
