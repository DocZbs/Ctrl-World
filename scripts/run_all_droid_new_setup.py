#!/usr/bin/env python3
"""Run all DROID new setup scenarios in a single process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omni_ctrl.configs.base_config import OmniCtrlConfig, FixedScenarioConfig


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


def _configure_cuda_env(config: OmniCtrlConfig) -> None:
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


def _reset_stats(orchestrator) -> None:
    orchestrator.stats = {
        "total_episodes": 0,
        "successful_episodes": 0,
        "wm_failures": 0,
        "vla_failures": 0,
        "total_reward": 0.0,
        "rewards_history": [],
    }
    orchestrator._warned_cartesian_dim = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run fixed scenarios in one process")
    parser.add_argument(
        "--config",
        default="omni_ctrl/configs/omni_ctrl_pi05_batch.yaml",
        help="Base config YAML",
    )
    parser.add_argument(
        "--ann-dir",
        default="dataset_example/droid_new_setup/annotation/val",
        help="Annotation directory",
    )
    parser.add_argument(
        "--out-base",
        default="experiments/omni_ctrl_fixed_scene_batch_pi_debug",
        help="Output directory base",
    )
    parser.add_argument(
        "--droid-root",
        default="dataset_example/droid_new_setup",
        help="DROID dataset root (for fixed scenario loading)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Iterations per scenario",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenarios with existing output video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmniCtrlConfig.from_yaml(args.config)
    config.num_iterations = args.iterations

    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    ann_paths = sorted(ann_dir.glob("*.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No annotation JSON files in {ann_dir}")

    _configure_cuda_env(config)

    from omni_ctrl.core.orchestrator import OmniCtrlOrchestrator
    from omni_ctrl.core.skill_library import SkillLibrary

    # Set output_dir to first scenario to avoid creating extra directory
    base_name = config.experiment_name
    first_sid = ann_paths[0].stem
    config.output_dir = str(Path(args.out_base) / first_sid)
    config.skill_library_path = str(Path(args.out_base) / first_sid / "skills")

    orchestrator = OmniCtrlOrchestrator(config)

    for ann_path in ann_paths:
        sid = ann_path.stem
        out_dir = Path(args.out_base) / sid
        if args.skip_existing and out_dir.exists():
            videos_dir = out_dir / "videos"
            has_video = videos_dir.exists() and any(videos_dir.glob("*.mp4"))
            has_results = (out_dir / "results.json").exists()
            has_episodes = (out_dir / "episodes").exists()
            if has_video or has_results or has_episodes:
                reason = "videos" if has_video else "results" if has_results else "episodes"
                print(f"Skipping {sid} (found existing {reason} in: {out_dir})")
                continue

        config.experiment_name = f"{base_name}_{sid}"
        config.output_dir = str(out_dir)
        config.skill_library_path = str(out_dir / "skills")
        if config.fixed_scenario is None:
            config.fixed_scenario = FixedScenarioConfig(
                enabled=True,
                annotation_path=str(ann_path),
                droid_root=args.droid_root,
            )
        # Always override with command line arguments
        config.fixed_scenario.droid_root = args.droid_root
        config.fixed_scenario.enabled = True
        config.fixed_scenario.annotation_path = str(ann_path)

        orchestrator.config = config
        orchestrator.output_dir = Path(config.output_dir)
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.skill_library = SkillLibrary(config.skill_library_path)
        if hasattr(orchestrator, "task_generator"):
            orchestrator.task_generator.task_counter = 0
        # Try to load scenario, skip if failed (e.g., corrupted video)
        try:
            orchestrator.fixed_scenario = orchestrator._load_fixed_scenario(
                config.fixed_scenario.annotation_path,
                config.fixed_scenario.droid_root,
            )
            orchestrator.use_fixed_scenario = True
            _reset_stats(orchestrator)
        except Exception as e:
            print(f"✗ Failed to load scenario {sid}: {e}")
            print(f"Skipping {sid}...\n")
            continue

        if hasattr(orchestrator.router, "_sticky_policy"):
            orchestrator.router._sticky_policy = None
        if hasattr(orchestrator.router, "_rr_index"):
            orchestrator.router._rr_index = 0
        orchestrator.router.reset()

        orchestrator.run()


if __name__ == "__main__":
    main()
