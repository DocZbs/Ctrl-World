#!/usr/bin/env python3
"""
Batch generate synthetic DROID data from real DROID episodes.

This script loads real DROID episodes, uses a policy to generate actions,
and uses a world model to generate corresponding video frames and states.
The output is saved in DROID format for downstream training.

Usage:
    python scripts/inference/generate_synthetic_droid_batch.py \
        --data-dir /path/to/droid_data \
        --output-dir synthetic_data/generated \
        --wm-ckpt /path/to/world_model.pth \
        --start-episode 0 \
        --end-episode 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the DroidDataInference class using absolute path resolution
inference_script_path = Path(__file__).resolve().parent / "run_inference_on_droid_data.py"

if not inference_script_path.exists():
    raise FileNotFoundError(f"Cannot find run_inference_on_droid_data.py at {inference_script_path}")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_inference_on_droid_data",
    inference_script_path
)
run_inference_module = importlib.util.module_from_spec(spec)
sys.modules['run_inference_on_droid_data'] = run_inference_module
spec.loader.exec_module(run_inference_module)
DroidDataInference = run_inference_module.DroidDataInference


def setup_cuda_env(wm_device: str, policy_device: str):
    """Setup CUDA environment for multi-GPU usage.

    Important: JAX will typically pick the first visible GPU (cuda:0). To ensure the policy runs on the requested
    GPU, we order CUDA_VISIBLE_DEVICES such that the *policy GPU is first* and the WM GPU is second.
    """
    # If the caller isn't using CUDA at all (e.g., cpu smoke tests), don't touch CUDA_VISIBLE_DEVICES.
    if "cuda:" not in wm_device and "cuda:" not in policy_device:
        print("Non-CUDA devices requested; skipping CUDA_VISIBLE_DEVICES setup.")
        return wm_device, policy_device

    wm_idx = int(wm_device.split(':')[1]) if 'cuda:' in wm_device else 0
    policy_idx = int(policy_device.split(':')[1]) if 'cuda:' in policy_device else 0

    # Preserve a deterministic order: policy first (so JAX uses cuda:0), then WM.
    unique_devices = []
    for d in (policy_idx, wm_idx):
        if d not in unique_devices:
            unique_devices.append(d)

    # Only set CUDA_VISIBLE_DEVICES if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in unique_devices)
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

        # Remap devices to 0-indexed after filtering (policy should become cuda:0).
        device_mapping = {old: new for new, old in enumerate(unique_devices)}
        wm_device_remapped = f"cuda:{device_mapping[wm_idx]}"
        policy_device_remapped = f"cuda:{device_mapping[policy_idx]}"
        print(f"Remapped devices: WM={wm_device_remapped}, Policy={policy_device_remapped}")

        return wm_device_remapped, policy_device_remapped
    else:
        print(f"CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")
        # Parse existing CUDA_VISIBLE_DEVICES
        visible_gpus = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

        # Map original indices to visible indices
        device_mapping = {gpu: idx for idx, gpu in enumerate(visible_gpus)}

        if wm_idx not in device_mapping or policy_idx not in device_mapping:
            raise ValueError(
                f"Requested GPUs ({wm_idx}, {policy_idx}) not in CUDA_VISIBLE_DEVICES ({visible_gpus})"
            )

        wm_device_remapped = f"cuda:{device_mapping[wm_idx]}"
        policy_device_remapped = f"cuda:{device_mapping[policy_idx]}"
        print(f"Using remapped devices: WM={wm_device_remapped}, Policy={policy_device_remapped}")

        return wm_device_remapped, policy_device_remapped


def main():
    default_data_stat = str(PROJECT_ROOT / "dataset_meta_info" / "droid" / "stat.json")
    if not Path(default_data_stat).exists():
        default_data_stat = None
    default_action_adapter = str(PROJECT_ROOT / "models" / "action_adapter" / "model2_15_9.pth")
    if not Path(default_action_adapter).exists():
        default_action_adapter = None

    parser = argparse.ArgumentParser(
        description='Batch generate synthetic DROID data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to source DROID data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--pi-ckpt',
        type=str,
        default='/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid',
        help='Policy checkpoint path'
    )
    parser.add_argument(
        '--policy-type',
        type=str,
        default='pi05',
        help='Policy type (pi05, pi0fast, pi0)'
    )
    parser.add_argument(
        '--wm-ckpt',
        type=str,
        required=True,
        help='World model checkpoint path'
    )
    parser.add_argument(
        '--start-episode',
        type=int,
        default=0,
        help='Start episode index'
    )
    parser.add_argument(
        '--end-episode',
        type=int,
        default=1000,
        help='End episode index (exclusive)'
    )
    parser.add_argument(
        '--wm-device',
        type=str,
        default='cuda:0',
        help='Device for world model'
    )
    parser.add_argument(
        '--policy-device',
        type=str,
        default='cuda:1',
        help='Device for policy (JAX will use this GPU)'
    )
    parser.add_argument(
        '--svd-model-path',
        type=str,
        default='stabilityai/stable-video-diffusion-img2vid',
        help='SVD model path'
    )
    parser.add_argument(
        '--clip-model-path',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model path'
    )
    parser.add_argument(
        '--data-stat-path',
        type=str,
        default=default_data_stat,
        help='Data statistics path for normalization'
    )
    parser.add_argument(
        '--action-adapter-path',
        type=str,
        default=default_action_adapter,
        help='Action adapter checkpoint path'
    )
    parser.add_argument(
        '--max-gen-steps',
        type=int,
        default=5,
        help='Maximum steps to generate per episode'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save progress every N episodes'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip episodes that have already been generated'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only load policy/world model and exit (no episode generation)'
    )

    args = parser.parse_args()

    # Setup CUDA environment BEFORE importing any models
    print("Setting up CUDA environment...")
    wm_device, policy_device = setup_cuda_env(args.wm_device, args.policy_device)

    print("=" * 60)
    print("Batch Synthetic DROID Data Generation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Source data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Policy: {args.pi_ckpt} ({args.policy_type})")
    print(f"  World model: {args.wm_ckpt}")
    print(f"  Episode range: {args.start_episode} - {args.end_episode}")
    print(f"  WM device: {wm_device}")
    print(f"  Policy device: {policy_device}")
    print(f"  Max steps per episode: {args.max_gen_steps}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data generator
    print("Loading models...")
    print()

    # Pass both devices to the constructor
    generator = DroidDataInference(
        pi_ckpt=args.pi_ckpt,
        policy_type=args.policy_type,
        device=policy_device,  # Policy device
        wm_device=wm_device,  # World model device
        generate_data=True,
        wm_ckpt=args.wm_ckpt,
        svd_model_path=args.svd_model_path,
        clip_model_path=args.clip_model_path,
        data_stat_path=args.data_stat_path,
        action_adapter_path=args.action_adapter_path,
    )

    print(f"✓ Policy on {policy_device}, World Model on {wm_device}")

    if args.dry_run:
        print("Dry-run requested; exiting after successful model load.")
        return

    print()
    print("Starting batch generation...")
    print()

    # Run batch generation
    results, errors = generator.run_inference(
        data_dir=args.data_dir,
        output_dir=output_dir,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        save_every=args.save_every,
        max_gen_steps=args.max_gen_steps,
    )

    # Save final statistics
    stats = {
        'total_episodes': args.end_episode - args.start_episode,
        'successful': len(results),
        'failed': len(errors),
        'success_rate': len(results) / (args.end_episode - args.start_episode) if (args.end_episode - args.start_episode) > 0 else 0,
        'config': {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'pi_ckpt': args.pi_ckpt,
            'policy_type': args.policy_type,
            'wm_ckpt': args.wm_ckpt,
            'start_episode': args.start_episode,
            'end_episode': args.end_episode,
            'max_gen_steps': args.max_gen_steps,
            'wm_device': wm_device,
            'policy_device': policy_device,
        }
    }

    stats_path = output_dir / "generation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print()
    print(f"Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - Generation stats: {stats_path}")
    print(f"  - Generated episodes: {output_dir / 'generated_episodes'}")
    if errors:
        print(f"  - Errors: {output_dir / 'errors.json'}")
    print()


if __name__ == '__main__':
    main()
