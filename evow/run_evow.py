"""
EvoW Experiment Runner

Usage:
    python evow/run_evow.py
    python evow/run_evow.py --config evow/configs/default.yaml
    python evow/run_evow.py --iterations 50 --output experiments/my_evow_run
"""

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from evow.utils.runtime import configure_jax_environment

# Configure JAX memory BEFORE any JAX imports
jax_env = configure_jax_environment()
print(
    "JAX configured: pre-allocation={}, memory fraction={}, allocator={}".format(
        jax_env["preallocate"],
        jax_env["mem_fraction"],
        jax_env["allocator"],
    )
)

import argparse

from evow.configs.base_config import (
    EvoWConfig,
    PolicyConfig,
    TaskGenConfig,
    RetrievalConfig,
    RolloutConfig,
    EvaluationConfig,
    RouterConfig,
    EvoWMemoryConfig,
    EvoWVLMRoutingConfig,
)
from evow.evaluation import DEFERRED_VLM_TYPES, is_qwen_vlm_type


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoW experiment")

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


def create_default_config() -> EvoWConfig:
    """Create default configuration for EvoW.

    YOU MUST UPDATE THESE PATHS BEFORE RUNNING.
    """
    config = EvoWConfig(
        experiment_name="evow",
        output_dir="experiments/evow",
        num_iterations=20,
        seed=42,
        device="cuda:0",
        log_frequency=5,
    )

    config.task_gen = TaskGenConfig(
        type="template",
        templates_path="evow/task_generation/templates.json",
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

    config.memory = EvoWMemoryConfig(
        max_entries=10000,
        save_path="experiments/evow/memory.json",
        context_window=20,
    )

    config.vlm_routing = EvoWVLMRoutingConfig(
        enabled=True,
        vlm_model="gpt-4o",
        vlm_api_key="${OPENAI_API_KEY}",
        timeout=30,
    )

    config.retrieval_temperature = 0.1
    config.top_k_candidates = 20
    config.reward_threshold = 0.3
    config.max_retries_per_task = 3
    config.skill_library_path = "experiments/evow/skills"

    return config


def _validate_config(config: EvoWConfig) -> list[str]:
    errors: list[str] = []

    vlm_type = str(getattr(config.evaluation, "vlm_type", "gpt-5") or "").strip().lower()
    is_dummy = vlm_type == "dummy"
    is_deferred = vlm_type in DEFERRED_VLM_TYPES
    is_qwen = is_qwen_vlm_type(vlm_type)

    api_key = config.evaluation.api_key or ""
    if not (is_dummy or is_deferred or is_qwen):
        if "your-openai-api-key" in api_key:
            errors.append("OpenAI API key not set! Please update config.")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_key = api_key[2:-1]
            if not os.getenv(env_key, ""):
                errors.append(f"OpenAI API key env '{env_key}' is not set!")

    if is_qwen:
        model_ref = str(getattr(config.evaluation, "vlm_model", "") or "").strip()
        if not model_ref:
            errors.append("For qwen evaluator, evaluation.vlm_model must be set to local/HF model path.")

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


def _collect_config_warnings(config: EvoWConfig) -> list[str]:
    warnings: list[str] = []

    use_scenario_instruction = bool(getattr(config.task_gen, "use_scenario_instruction", False))
    min_overlap = int(getattr(config.retrieval, "min_instruction_overlap_tokens", 0) or 0)
    instr_match_weight = float(getattr(config.retrieval, "instruction_match_weight", 0.0) or 0.0)
    retrieval_temperature = float(getattr(config, "retrieval_temperature", 0.1) or 0.1)
    top_k_candidates = int(getattr(config, "top_k_candidates", 20) or 20)
    deterministic_top1 = bool(getattr(config.retrieval, "deterministic_top1", False))

    if not use_scenario_instruction and min_overlap < 1:
        warnings.append(
            "task_gen.use_scenario_instruction=false but retrieval.min_instruction_overlap_tokens < 1; "
            "template task and retrieved scene may be semantically mismatched."
        )

    if (
        not use_scenario_instruction
        and min_overlap <= 0
        and instr_match_weight <= 0
        and (retrieval_temperature >= 0.4 or top_k_candidates >= 80)
        and not deterministic_top1
    ):
        warnings.append(
            "High-diversity retrieval without instruction alignment detected "
            f"(temperature={retrieval_temperature}, top_k_candidates={top_k_candidates}); "
            "this can create impossible tasks due to task-scene mismatch."
        )

    if not use_scenario_instruction and not deterministic_top1 and retrieval_temperature > 0:
        warnings.append(
            "task_gen.use_scenario_instruction=false with stochastic retrieval; "
            "set retrieval.deterministic_top1=true to always choose the most similar scene."
        )

    return warnings


def _prepare_run_storage(config: EvoWConfig) -> tuple[str, str]:
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
    print(" " * 22 + "EvoW RUNNER")
    print("=" * 70)

    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = EvoWConfig.from_yaml(args.config)
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

    validation_warnings = _collect_config_warnings(config)
    if validation_warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in validation_warnings:
            print(f"  - {warning}")

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

    from evow.core.evow_orchestrator import EvoWOrchestrator

    try:
        orchestrator = EvoWOrchestrator(config)
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
