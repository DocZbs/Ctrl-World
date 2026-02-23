"""
NEXUS Experiment Runner

Usage:
    python nexus/run_nexus.py
    python nexus/run_nexus.py --config nexus/configs/default.yaml
    python nexus/run_nexus.py --iterations 50 --output experiments/my_nexus_run
"""

import os

# Configure JAX memory BEFORE any JAX imports
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"
if "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
print(
    "JAX configured: pre-allocation={}, memory fraction={}".format(
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
)

import sys
import argparse
from pathlib import Path

# Add Ctrl-World root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.configs.base_config import (
    NexusConfig,
    PolicyConfig,
    TaskGenConfig,
    RetrievalConfig,
    RolloutConfig,
    EvaluationConfig,
    RouterConfig,
    NexusMemoryConfig,
    NexusVLMRoutingConfig,
)


def _parse_cuda_index(device_str: str | None) -> int | None:
    if not device_str:
        return None
    if not device_str.startswith("cuda"):
        return None
    if ":" not in device_str:
        return 0
    try:
        return int(device_str.split(":", 1)[1])
    except ValueError:
        return None


def _remap_cuda_device(device_str: str | None, mapping: dict[int, int]) -> str | None:
    if not device_str:
        return None
    if not device_str.startswith("cuda"):
        return device_str
    idx = _parse_cuda_index(device_str)
    if idx is None:
        return device_str
    new_idx = mapping.get(idx, idx)
    return f"cuda:{new_idx}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run NEXUS experiment")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--iterations", type=int, default=None, help="Override iterations")
    parser.add_argument("--output", type=str, default=None, help="Override output dir")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--api-key", type=str, default=None, help="Override OpenAI API key")
    parser.add_argument(
        "--disable-vlm-routing",
        action="store_true",
        help="Disable VLM expert routing (falls back to statistical routing)",
    )

    return parser.parse_args()


def create_default_config() -> NexusConfig:
    """Create default configuration for NEXUS.

    YOU MUST UPDATE THESE PATHS BEFORE RUNNING.
    """
    config = NexusConfig(
        experiment_name="nexus",
        output_dir="experiments/nexus",
        num_iterations=20,
        seed=42,
        device="cuda:0",
        log_frequency=5,
    )

    config.task_gen = TaskGenConfig(
        type="template",
        templates_path="nexus/task_generation/templates.json",
    )

    config.retrieval = RetrievalConfig(
        droid_path="dataset_example/droid_subset",
        index_path="experiments/droid_index",
        clip_model_path="openai/clip-vit-base-patch32",
        top_k=5,
        rebuild_index=False,
    )

    config.rollout = RolloutConfig(
        wm_ckpt="/path/to/your/ctrl_world_checkpoint.pt",  # TODO
        svd_model_path="/path/to/your/svd_model",  # TODO
        clip_model_path="/path/to/your/clip_model",  # TODO
        data_stat_path="dataset_meta_info/droid/stat.json",
        max_steps=50,
        pred_step=5,
        policy_skip_step=2,
        action_adapter_path="models/action_adapter/model2_15_9.pth",
        device="cuda:0",
    )

    config.evaluation = EvaluationConfig(
        vlm_type="gpt-5",
        vlm_model="gpt-5",
        api_key="your-openai-api-key-here",  # TODO
        max_retries=3,
        timeout=60,
        success_weight=1.0,
        consistency_weight=0.5,
    )

    config.router = RouterConfig(
        available_policies=[
            PolicyConfig(
                name="pi05",
                checkpoint="/path/to/your/pi05_checkpoint",  # TODO
                action_space="joint_vel",
            ),
        ],
        selection_strategy="round_robin",
    )

    config.memory = NexusMemoryConfig(
        max_entries=10000,
        save_path="experiments/nexus/memory.json",
        context_window=20,
    )

    config.vlm_routing = NexusVLMRoutingConfig(
        enabled=True,
        vlm_model="gpt-4o",
        vlm_api_key="${OPENAI_API_KEY}",
        timeout=30,
    )

    config.retrieval_temperature = 0.1
    config.top_k_candidates = 20
    config.reward_threshold = 0.3
    config.max_retries_per_task = 3
    config.skill_library_path = "experiments/nexus/skills"

    return config


def _validate_config(config: NexusConfig) -> list[str]:
    errors: list[str] = []

    api_key = config.evaluation.api_key or ""
    if "your-openai-api-key" in api_key:
        errors.append("OpenAI API key not set! Please update config.")
    if api_key.startswith("${") and api_key.endswith("}"):
        env_key = api_key[2:-1]
        if not os.getenv(env_key, ""):
            errors.append(f"OpenAI API key env '{env_key}' is not set!")

    if "/path/to/" in (config.rollout.wm_ckpt or ""):
        errors.append("World model checkpoint path not set!")

    if not config.router.available_policies:
        errors.append("router.available_policies is empty!")
    else:
        for pol in config.router.available_policies:
            if "/path/to/" in (str(pol.checkpoint) if pol.checkpoint else ""):
                errors.append(f"Policy checkpoint path not set for {pol.name}!")

    if getattr(config, "vlm_routing", None) and getattr(config.vlm_routing, "enabled", False):
        routing_key = config.vlm_routing.vlm_api_key or ""
        if routing_key.startswith("${") and routing_key.endswith("}"):
            env_key = routing_key[2:-1]
            if not os.getenv(env_key, ""):
                errors.append(f"VLM routing API key env '{env_key}' is not set!")

    if getattr(config, "max_retries_per_task", 1) < 1:
        errors.append("max_retries_per_task must be >= 1")

    return errors


def _prepare_run_storage(config: NexusConfig) -> tuple[str, str]:
    """Prepare stable storage paths under output directory.

    Returns:
        Tuple of (base_output_dir, resolved_output_dir)
    """
    base_output_dir = Path(config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Keep all artifacts in a stable, non-timestamped layout.
    config.output_dir = str(base_output_dir)
    config.skill_library_path = str(base_output_dir / "skills")
    config.memory.save_path = str(base_output_dir / "memory.json")

    return str(base_output_dir), str(base_output_dir)


def main():
    args = parse_args()

    print("=" * 70)
    print(" " * 22 + "NEXUS RUNNER")
    print("=" * 70)

    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = NexusConfig.from_yaml(args.config)
    else:
        print("\nUsing default configuration")
        print("⚠️  Remember to update paths in create_default_config()!")
        config = create_default_config()

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

    if args.disable_vlm_routing:
        config.vlm_routing.enabled = False
        print("VLM routing disabled")

    base_output_dir, run_output_dir = _prepare_run_storage(config)
    if base_output_dir != run_output_dir:
        print(f"Output base dir: {base_output_dir}")
        print(f"Resolved run dir: {run_output_dir}")
    print(f"Skills dir: {config.skill_library_path}")
    print(f"Memory file: {config.memory.save_path}")

    # Limit visible GPUs to only those referenced by config
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        policy_devices = []
        if getattr(config, "router", None) and getattr(config.router, "available_policies", None):
            for pol in config.router.available_policies:
                dev = getattr(pol, "device", None)
                if dev:
                    policy_devices.append(dev)

        device_indices = []
        for dev in (config.device, config.rollout.device, *policy_devices):
            idx = _parse_cuda_index(dev)
            if idx is not None:
                device_indices.append(idx)
        unique_devices = sorted(set(device_indices))
        if unique_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in unique_devices)
            mapping = {old: new for new, old in enumerate(unique_devices)}
            config.device = _remap_cuda_device(config.device, mapping)
            config.rollout.device = _remap_cuda_device(config.rollout.device, mapping)
            if getattr(config, "router", None) and getattr(config.router, "available_policies", None):
                for pol in config.router.available_policies:
                    if getattr(pol, "device", None):
                        pol.device = _remap_cuda_device(pol.device, mapping)
            print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"Remapped device strings: policy={config.device}, rollout={config.rollout.device}")
    else:
        print(f"CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")

    policy_idx = _parse_cuda_index(config.device)
    if policy_idx is not None:
        os.environ["OPENPI_JAX_DEVICE"] = str(policy_idx)
        print(f"OPENPI_JAX_DEVICE set to: {os.environ['OPENPI_JAX_DEVICE']}")

    print("\nValidating configuration...")
    validation_errors = _validate_config(config)

    if validation_errors:
        print("\n⚠️  Configuration Errors:")
        for error in validation_errors:
            print(f"  - {error}")
        print("\nPlease update the paths in create_default_config() or use a config file.")
        print("Exiting...\n")
        return

    print("Configuration validated ✓")

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
    print(f"VLM routing:    {getattr(config.vlm_routing, 'enabled', True)}")
    print("Policies:")
    for pol in config.router.available_policies:
        print(f"  - {pol.name} ({pol.action_space})")
    print("=" * 70 + "\n")

    from nexus.core.nexus_orchestrator import NexusOrchestrator

    try:
        orchestrator = NexusOrchestrator(config)
    except Exception as e:
        print(f"\n❌ Failed to initialize orchestrator: {e}")
        import traceback

        traceback.print_exc()
        return

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
        print(f"\nFinal results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
