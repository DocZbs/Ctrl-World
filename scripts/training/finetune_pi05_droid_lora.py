#!/usr/bin/env python3
"""
Fine-tune Pi0.5-DROID with LoRA on synthetic data

This script provides LoRA (Low-Rank Adaptation) finetuning for Pi0.5-DROID,
which is much more memory-efficient than full finetuning.

LoRA Benefits:
- Uses ~10x less GPU memory
- Faster training
- Only trains a small number of parameters
- Can be easily merged back to base model

Usage:
    # Train with LoRA
    python scripts/finetune_pi05_droid_lora.py \
        --repo-id local/synthetic_pickplace_0002 \
        --exp-name pickplace_lora \
        --num-train-steps 5000 \
        --batch-size 32 \
        --lora-rank 16

    # Or use the config name directly
    python openpi/scripts/train_pytorch.py pi05_droid_lora \
        --exp-name pickplace_lora \
        --data.repo-id local/synthetic_pickplace_0002 \
        --num-train-steps 5000
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add the openpi training config to the path
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


def register_lora_config(lora_rank: int = 16, lora_alpha: float = 32):
    """Register Pi0.5-DROID LoRA finetuning configuration"""

    # Create model config with LoRA
    # For Pi0.5, we use gemma_2b_lora for the vision-language model
    # and gemma_300m_lora for the action expert
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",  # Use LoRA for PaliGemma
        action_expert_variant="gemma_300m_lora",  # Use LoRA for action expert
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
        name="pi05_droid_lora",
        model=model_config,
        data=LeRobotDROIDDataConfig(
            repo_id="local/synthetic_pickplace_0002",  # Default, can be overridden
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Reuse original DROID norm stats (copied during conversion)
                assets_dir="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/assets",
                asset_id="droid",
            ),
        ),
        # Load pretrained Pi0.5-DROID weights (PyTorch format)
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/params"
        ),
        pytorch_weight_path="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch",

        # Freeze all parameters except LoRA
        freeze_filter=freeze_filter,

        # Training hyperparameters optimized for LoRA
        num_train_steps=5000,
        batch_size=32,  # Can use larger batch size with LoRA

        # Turn off EMA for LoRA finetuning (recommended)
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
    if "pi05_droid_lora" not in config_names:
        config._CONFIGS.append(lora_config)
        print(f"✓ Registered Pi0.5-DROID LoRA config")
    else:
        print(f"✓ Pi0.5-DROID LoRA config already registered")

    return lora_config


def train_lora(
    repo_id: str,
    exp_name: str,
    num_gpus: int = 1,
    num_train_steps: int = 5000,
    batch_size: int = 32,
    lora_rank: int = 16,
):
    """
    Train Pi0.5-DROID with LoRA

    Args:
        repo_id: LeRobot dataset repository ID
        exp_name: Experiment name
        num_gpus: Number of GPUs to use
        num_train_steps: Number of training steps
        batch_size: Batch size (can be larger with LoRA)
        lora_rank: LoRA rank (higher = more capacity, more memory)
    """
    print(f"\n{'='*80}")
    print("Pi0.5-DROID LoRA Fine-tuning")
    print(f"{'='*80}\n")

    # Register the LoRA config
    register_lora_config(lora_rank=lora_rank)

    print(f"Configuration:")
    print(f"  Dataset: {repo_id}")
    print(f"  Experiment: {exp_name}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Training steps: {num_train_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Base model: /mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid")
    print()

    # Use the OpenPI venv Python
    python_exe = "/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/.venv/bin/python"

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={num_gpus}",
            "openpi/scripts/train_pytorch.py",
            "pi05_droid_lora",
            "--exp-name", exp_name,
            "--data.repo-id", repo_id,
            "--num-train-steps", str(num_train_steps),
            "--batch-size", str(batch_size),
            "--pytorch-training-precision", "float32",
        ]
    else:
        cmd = [
            python_exe,
            "openpi/scripts/train_pytorch.py",
            "pi05_droid_lora",
            "--exp-name", exp_name,
            "--data.repo-id", repo_id,
            "--num-train-steps", str(num_train_steps),
            "--batch-size", str(batch_size),
            "--pytorch-training-precision", "float32",
        ]

    print(f"Running training command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run training
    subprocess.run(cmd, check=True)

    print(f"\n✓ LoRA training complete!")
    print(f"  Checkpoints saved to: openpi/checkpoints/{exp_name}/")
    print(f"  LoRA weights are much smaller than full model weights")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Pi0.5-DROID with LoRA on synthetic data"
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
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
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
        help="Batch size (default: 32, can be larger with LoRA)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16, higher = more capacity)",
    )

    args = parser.parse_args()

    train_lora(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        num_gpus=args.num_gpus,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
    )

    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Evaluate the model on your task")
    print("2. If needed, merge LoRA weights back to base model")
    print("3. Deploy the finetuned model")


if __name__ == "__main__":
    main()
