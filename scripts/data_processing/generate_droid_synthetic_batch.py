#!/usr/bin/env python3
"""
Generate synthetic DROID data from real DROID episodes using config file.
Architecture similar to run_all_droid_new_setup.py

Usage:
    python scripts/data_processing/generate_droid_synthetic_batch.py \
        --config omni_ctrl/configs/generate_synthetic_droid_1000.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _configure_cuda_env(config: dict) -> None:
    """Configure CUDA environment for multi-GPU usage"""
    policy_device = config.get('device', 'cuda:1')
    wm_device = config['rollout'].get('device', 'cuda:0')

    # Extract GPU indices
    policy_idx = int(policy_device.split(':')[1]) if 'cuda:' in policy_device else 0
    wm_idx = int(wm_device.split(':')[1]) if 'cuda:' in wm_device else 0

    unique_devices = sorted(set([policy_idx, wm_idx]))

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in unique_devices)
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

        # Remap device indices
        device_mapping = {old: new for new, old in enumerate(unique_devices)}
        config['device'] = f"cuda:{device_mapping[policy_idx]}"
        config['rollout']['device'] = f"cuda:{device_mapping[wm_idx]}"

        print(f"Remapped devices: Policy={config['device']}, WM={config['rollout']['device']}")
    else:
        print(f"CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Set JAX device for policy
    policy_gpu_idx = config['device'].split(':')[1] if 'cuda:' in config['device'] else '0'
    os.environ["OPENPI_JAX_DEVICE"] = policy_gpu_idx
    print(f"OPENPI_JAX_DEVICE set to: {os.environ['OPENPI_JAX_DEVICE']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic DROID data from config")
    parser.add_argument(
        "--config",
        default="omni_ctrl/configs/generate_synthetic_droid_1000.yaml",
        help="Config YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Synthetic DROID Data Generation")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Output: {config['output_dir']}")
    print()

    # Configure CUDA environment
    _configure_cuda_env(config)

    # Import after CUDA setup
    from scripts.inference.run_inference_on_droid_data import DroidDataInference

    # Extract configuration
    source_config = config['source_data']
    policy_config_item = config['router']['available_policies'][0]
    rollout_config = config['rollout']

    print("Loading models...")
    print(f"  Policy: {policy_config_item['checkpoint']}")
    print(f"  World Model: {rollout_config['wm_ckpt']}")
    print()

    # Create data generator
    generator = DroidDataInference(
        pi_ckpt=policy_config_item['checkpoint'],
        policy_type=policy_config_item['name'],
        device=config['device'],  # Policy device
        wm_device=rollout_config['device'],  # World model device
        generate_data=True,
        wm_ckpt=rollout_config['wm_ckpt'],
        svd_model_path=rollout_config['svd_model_path'],
        clip_model_path=rollout_config['clip_model_path'],
        data_stat_path=rollout_config['data_stat_path'],
        action_adapter_path=rollout_config.get('action_adapter_path'),
    )

    print(f"✓ Models loaded")
    print(f"  Policy on {config['device']}")
    print(f"  World Model on {rollout_config['device']}")
    print()

    # Run generation
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting batch generation...")
    print(f"  Episodes: {source_config['start_episode']} - {source_config['end_episode']}")
    print(f"  Max steps per episode: {rollout_config['max_steps']}")
    print()

    results, errors = generator.run_inference(
        data_dir=source_config['data_dir'],
        output_dir=output_dir,
        start_episode=source_config['start_episode'],
        end_episode=source_config['end_episode'],
        save_every=source_config['save_every'],
    )

    # Print summary
    print()
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Total episodes: {source_config['end_episode'] - source_config['start_episode']}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(errors)}")
    if len(results) > 0:
        print(f"  Success rate: {len(results) / (source_config['end_episode'] - source_config['start_episode']) * 100:.1f}%")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Generated episodes: {output_dir / 'generated_episodes'}")
    if errors:
        print(f"  - Errors: {output_dir / 'errors.json'}")
    print()


if __name__ == '__main__':
    main()
