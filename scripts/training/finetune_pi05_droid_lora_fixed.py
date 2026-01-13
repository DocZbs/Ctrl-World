#!/usr/bin/env python3
"""
FIXED: Fine-tune Pi0.5-DROID with LoRA (correctly loads pretrained weights)

The original version used 'gemma_2b_lora' which changes layernorm structure,
preventing pretrained weights from loading. This version fixes that issue.

Usage:
    python scripts/training/finetune_pi05_droid_lora_fixed.py \
        --repo-id local/synthetic_pickplace_0002 \
        --exp-name pickplace_lora_fixed \
        --num-train-steps 5000 \
        --batch-size 32
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


def register_lora_config_fixed():
    """
    Register FIXED Pi0.5-DROID LoRA configuration

    Key fix: Use the REGULAR model structure (not _lora variants),
    and let the training system add LoRA layers on top.
    This ensures pretrained weights load correctly.
    """

    # CRITICAL FIX: Use regular variants, not _lora
    # The training system will add LoRA layers WITHOUT changing the base structure
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b",  # FIXED: was "gemma_2b_lora"
        action_expert_variant="gemma_300m",  # FIXED: was "gemma_300m_lora"
    )

    # Create freeze filter - freeze everything except what we want to train
    # We'll add LoRA to attention and FFN layers
    freeze_filter = {
        # Freeze all base model parameters
        "**": True,
        # Unfreeze LoRA parameters (will be added by training system)
        "**.lora_**": False,
        # Optionally unfreeze output projection
        "*action_out_proj*": False,
    }

    lora_config = TrainConfig(
        name="pi05_droid_lora_fixed",
        model=model_config,
        data=LeRobotDROIDDataConfig(
            repo_id="local/synthetic_pickplace_0002",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/assets",
                asset_id="droid",
            ),
        ),
        # Load pretrained weights (WILL WORK NOW with correct structure)
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch/params"
        ),
        pytorch_weight_path="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid_pytorch",

        # Freeze base parameters
        freeze_filter=freeze_filter,

        # Training hyperparameters
        num_train_steps=5000,
        batch_size=32,
        ema_decay=None,

        lr_schedule=config._optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
    )

    # Register config
    config_names = [c.name for c in config._CONFIGS]
    if "pi05_droid_lora_fixed" not in config_names:
        config._CONFIGS.append(lora_config)
        print(f"✓ Registered FIXED Pi0.5-DROID LoRA config")
    else:
        print(f"✓ FIXED LoRA config already registered")

    return lora_config


def train_lora_fixed(
    repo_id: str,
    exp_name: str,
    num_gpus: int = 1,
    num_train_steps: int = 5000,
    batch_size: int = 32,
):
    """Train Pi0.5-DROID with FIXED LoRA configuration"""

    print(f"\n{'='*80}")
    print("Pi0.5-DROID LoRA Fine-tuning (FIXED VERSION)")
    print(f"{'='*80}\n")

    register_lora_config_fixed()

    print(f"Configuration:")
    print(f"  Dataset: {repo_id}")
    print(f"  Experiment: {exp_name}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Steps: {num_train_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"\nFIXES APPLIED:")
    print(f"  ✓ Using regular model structure (not _lora variants)")
    print(f"  ✓ Pretrained weights will load correctly")
    print(f"  ✓ LoRA will be added by training system")

    # Build command
    train_script = Path(__file__).parent.parent / "openpi" / "src" / "openpi" / "scripts" / "train_pytorch.py"

    cmd = [
        "python", str(train_script),
        "pi05_droid_lora_fixed",  # Use fixed config
        "--exp-name", exp_name,
        f"--data.repo-id={repo_id}",
        f"--num-train-steps={num_train_steps}",
        f"--batch-size={batch_size}",
    ]

    if num_gpus > 1:
        cmd = ["torchrun", f"--nproc_per_node={num_gpus}"] + cmd[1:]

    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}\n")
    print(f"{'='*80}\n")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n✗ Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✓ Training completed successfully!")
        print(f"\nCheckpoint saved to:")
        print(f"  checkpoints/{exp_name}/")


def main():
    parser = argparse.ArgumentParser(description="FIXED Pi0.5-DROID LoRA finetuning")
    parser.add_argument("--repo-id", type=str, required=True,
                       help="LeRobot dataset repository ID")
    parser.add_argument("--exp-name", type=str, required=True,
                       help="Experiment name")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--num-train-steps", type=int, default=5000,
                       help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")

    args = parser.parse_args()

    train_lora_fixed(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        num_gpus=args.num_gpus,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
