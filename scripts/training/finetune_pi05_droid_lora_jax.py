#!/usr/bin/env python3
"""
Fine-tune Pi0.5-DROID with LoRA on synthetic data using JAX

This script provides LoRA (Low-Rank Adaptation) finetuning for Pi0.5-DROID using JAX,
which is the native, well-tested framework for OpenPI.

LoRA Benefits:
- Uses ~10x less GPU memory
- Faster training
- Only trains a small number of parameters
- Can be easily merged back to base model
- Native JAX format - no conversion issues

Usage:
    python scripts/finetune_pi05_droid_lora_jax.py \
        --repo-id local/synthetic_pickplace_0002 \
        --exp-name pickplace_lora_jax \
        --num-train-steps 5000 \
        --batch-size 32 \
        --lora-rank 16
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "openpi" / "src"))

import openpi.training.config as config
import openpi.training.weight_loaders as weight_loaders
import openpi.models.pi0_config as pi0_config
from openpi.training.config import (
    TrainConfig,
    LeRobotDROIDDataConfig,
    DataConfig,
    AssetsConfig,
)


def register_lora_config(lora_rank: int = 16):
    """Register Pi0.5-DROID LoRA finetuning configuration for JAX"""

    # Create model config with LoRA
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    # Create freeze filter to only train LoRA parameters
    freeze_filter = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter()

    # Register the config
    lora_config = TrainConfig(
        name="pi05_droid_lora_jax",
        model=model_config,
        data=LeRobotDROIDDataConfig(
            repo_id="local/synthetic_pickplace_0002",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Use original DROID norm stats from JAX checkpoint
                assets_dir="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        # Load pretrained Pi0.5-DROID weights (JAX format)
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid/params"
        ),

        # Freeze all parameters except LoRA
        freeze_filter=freeze_filter,

        # Training hyperparameters optimized for LoRA
        num_train_steps=5000,
        batch_size=32,

        # Turn off EMA for LoRA finetuning
        ema_decay=None,

        # Learning rate - slightly higher for LoRA
        lr_schedule=config._optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
    )

    # Add to configs if not already present
    config_names = [c.name for c in config._CONFIGS]
    if "pi05_droid_lora_jax" not in config_names:
        config._CONFIGS.append(lora_config)
        print(f"Registered Pi0.5-DROID LoRA JAX config")
    else:
        print(f"Pi0.5-DROID LoRA JAX config already registered")

    return lora_config


def train_lora_jax(
    repo_id: str,
    exp_name: str,
    num_train_steps: int = 5000,
    batch_size: int = 32,
    lora_rank: int = 16,
):
    """
    Train Pi0.5-DROID with LoRA using JAX

    Args:
        repo_id: LeRobot dataset repository ID
        exp_name: Experiment name
        num_train_steps: Number of training steps
        batch_size: Batch size
        lora_rank: LoRA rank (higher = more capacity, more memory)
    """
    print(f"\n{'='*80}")
    print("Pi0.5-DROID LoRA Fine-tuning (JAX)")
    print(f"{'='*80}\n")

    # Register the LoRA config
    register_lora_config(lora_rank=lora_rank)

    print(f"Configuration:")
    print(f"  Dataset: {repo_id}")
    print(f"  Experiment: {exp_name}")
    print(f"  Training steps: {num_train_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Base model: /mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid (JAX)")
    print(f"  Framework: JAX (native)")
    print()

    # Use the OpenPI venv Python
    python_exe = "/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/bin/python"

    # Build command for JAX training
    cmd = [
        python_exe,
        "openpi/scripts/train.py",
        "pi05_droid_lora",  # Use existing config
        "--exp-name", exp_name,
        "--data.repo-id", repo_id,
        "--num-train-steps", str(num_train_steps),
        "--batch-size", str(batch_size),
        "--weight-loader.params-path", "/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid/params",
        "--overwrite",  # Overwrite existing checkpoint directory
        "--keep-period", "1000",  # Keep checkpoints every 1000 steps (all of them)
    ]

    print(f"Running JAX training command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run training
    subprocess.run(cmd, check=True)

    print(f"\nLoRA training complete!")
    print(f"  Checkpoints saved to: openpi/checkpoints/{exp_name}/")
    print(f"  Format: JAX (native, no conversion needed)")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Pi0.5-DROID with LoRA on synthetic data using JAX"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/synthetic_pickplace_0002",
        help="LeRobot dataset repository ID",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16, higher = more capacity)",
    )

    args = parser.parse_args()

    train_lora_jax(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
    )

    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Evaluate the model on your task")
    print("2. Use the JAX checkpoint directly for inference")
    print("3. No conversion needed - native JAX format")


if __name__ == "__main__":
    main()
