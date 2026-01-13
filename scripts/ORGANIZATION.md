# Scripts Directory Organization

This directory contains all scripts for the Ctrl-World project, organized by purpose.

## Directory Structure

```
scripts/
├── training/           # Model training and finetuning scripts
├── data_processing/    # Data generation, conversion, and preprocessing
├── evaluation/         # Testing, validation, and analysis scripts
├── inference/          # Rollout and inference scripts
├── utils/              # Utility scripts and patches
├── deprecated/         # Old/unused scripts (kept for reference)
└── ORGANIZATION.md     # This file
```

## Training Scripts (`training/`)

Scripts for training and finetuning models:

- `finetune_pi05_droid_lora_jax.py` - **[RECOMMENDED]** LoRA finetuning using JAX (native format)
- `finetune_pi05_droid_lora.py` - LoRA finetuning using PyTorch (has conversion issues)
- `finetune_pi05_synthetic.py` - Finetune Pi0.5 on synthetic data
- `finetune_by_scene.py` - Finetune by scene
- `finetune_by_task.py` - Finetune by task
- `finetune_single_episode.py` - Finetune on single episode
- `train_wm.py` - Train world model
- `train_action_adapter.sh` - Train action adapter
- `run_training.sh` - General training runner
- `run_lora_finetune_cuda1.sh` - **[MAIN]** Run LoRA finetuning on GPU 1 (uses JAX)
- `fix_and_retrain_pi05.sh` - Fix and retrain Pi0.5

**Quick Start:**
```bash
# Run LoRA finetuning (recommended)
bash training/run_lora_finetune_cuda1.sh
```

## Data Processing Scripts (`data_processing/`)

Scripts for generating, converting, and preprocessing data:

- `generate_synthetic_trajectories.py` - Generate synthetic trajectories
- `generate_synthetic_trajectories.sh` - Shell wrapper for trajectory generation
- `convert_synthetic_to_lerobot.py` - Convert synthetic data to LeRobot format
- `convert_checkpoint_format.py` - Convert checkpoint formats
- `compute_synthetic_norm_stats.py` - Compute normalization statistics
- `label_trajectories.py` - Label trajectory data
- `prepare_validation_data.py` - Prepare validation datasets
- `gen_latent_chunk.py` - Generate latent chunks
- `gen_latent_simple.py` - Simple latent generation
- `create_latent_symlinks.py` - Create symlinks for latent data

**Quick Start:**
```bash
# Generate synthetic trajectories
python data_processing/generate_synthetic_trajectories.py --config <config>
```

## Evaluation Scripts (`evaluation/`)

Scripts for testing, validation, and analysis:

- `test_success_rate.py` - Test model success rate
- `test_synthetic_generation.sh` - Test synthetic data generation
- `analyze_data_quality.py` - Analyze data quality
- `analyze_success_rate.py` - Analyze success rates
- `compare_model_loss.py` - Compare model losses
- `validate_chunk_000.sh` - Validate data chunks

**Quick Start:**
```bash
# Test success rate
python evaluation/test_success_rate.py --model <model_path>
```

## Inference Scripts (`inference/`)

Scripts for running inference and rollouts:

- `run_all_droid_new_setup.py` - **[MAIN]** Run inference with DROID setup
- `run_all_droid_new_setup.sh` - Shell wrapper for DROID inference
- `run_all_droid_new_setup_finetuned.sh` - Run with finetuned model
- `run_all_pi05.sh` - Run Pi0.5 inference
- `run_all_pi0_fast.sh` - Run Pi0 fast inference
- `rollout_interact_pi.py` - Interactive rollout with Pi models
- `rollout_key_board.py` - Keyboard-controlled rollout
- `rollout_replay_traj.py` - Replay trajectories

**Quick Start:**
```bash
# Run inference with Pi0.5
python inference/run_all_droid_new_setup.py --config <config>
```

## Utility Scripts (`utils/`)

Utility scripts and patches:

- `patch_lerobot.py` - Patch LeRobot library
- `preprocess.sh` - Preprocessing utilities
- `process_chunk_000.sh` - Process data chunks

## Migration Notes

All scripts have been moved to subdirectories. Update your commands:

**Old:**
```bash
python scripts/finetune_pi05_droid_lora_jax.py
```

**New:**
```bash
python scripts/training/finetune_pi05_droid_lora_jax.py
```

Or use relative paths from the scripts directory:
```bash
cd scripts
python training/finetune_pi05_droid_lora_jax.py
```

## Recommended Workflow

1. **Data Generation:**
   ```bash
   python data_processing/generate_synthetic_trajectories.py
   ```

2. **Training:**
   ```bash
   bash training/run_lora_finetune_cuda1.sh
   ```

3. **Evaluation:**
   ```bash
   python evaluation/test_success_rate.py
   ```

4. **Inference:**
   ```bash
   python inference/run_all_droid_new_setup.py
   ```

## Important Notes

- **Use JAX training** (`finetune_pi05_droid_lora_jax.py`) instead of PyTorch to avoid conversion issues
- All training scripts save checkpoints to `openpi/checkpoints/<exp_name>/`
- JAX checkpoints are in native format and work directly for inference
- PyTorch checkpoints may have key mismatch issues

## Troubleshooting

If you encounter import errors after reorganization:
1. Use absolute paths from project root
2. Or update `sys.path` in scripts if needed
3. Check that you're running from the correct directory

For questions or issues, refer to the main project README.
