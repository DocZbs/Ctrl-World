# Scripts Directory

This directory contains scripts for testing, training, and running Ctrl-World experiments.

## 📊 Experiment Output Structure

Every experiment automatically saves configuration and logs:

```
experiments/your_experiment_name/
├── config.yaml                    # 📄 Configuration snapshot (with timestamp)
├── experiment.log                 # 📝 Full execution log
├── results.json                   # 📊 Final statistics
├── skill_library_summary.json     # 🎯 Discovered skills
├── episodes/                      # Episode metadata
│   ├── task_0001.json
│   └── ...
├── videos/                        # Generated videos
│   ├── task_0001.mp4
│   └── ...
└── skills/                        # Skill library
```

**Key files**:
- **config.yaml** - Exact configuration used (includes timestamp and metadata)
- **experiment.log** - Complete log with timestamps, statistics, and errors
- **results.json** - Summary statistics (success rate, failures, rewards)

You can review past experiments by checking these files:
```bash
# View configuration used
cat experiments/omni_ctrl_octo_batch/0001/config.yaml

# Monitor live experiment log
tail -f experiments/omni_ctrl_octo_batch/0001/experiment.log

# Check final results
cat experiments/omni_ctrl_octo_batch/0001/results.json
```

## 🚀 Quick Start

### Test a Policy

Test if a policy is working correctly:

```bash
# Test Octo policy
python scripts/test_policy.py --policy octo --device cuda:7

# Test Pi0.5 policy
python scripts/test_policy.py --policy pi05 --device cuda:6

# Test OpenVLA policy
python scripts/test_policy.py --policy openvla --device cuda:0

# Use small model (Octo only)
python scripts/test_policy.py --policy octo --small
```

### Run Batch Experiments

Run all validation scenarios with a specific policy:

```bash
# Octo policy
bash scripts/run_all_octo.sh

# Pi0.5 policy
bash scripts/run_all_pi05.sh

# Custom configuration
bash scripts/run_batch.sh omni_ctrl/configs/your_config.yaml
```

### Advanced Batch Options

```bash
# Specify custom output directory
OUT_BASE=experiments/my_experiment bash scripts/run_batch.sh config.yaml

# Specify custom annotation directory
ANN_DIR=path/to/annotations bash scripts/run_batch.sh config.yaml

# Run multiple iterations per scenario
ITERATIONS=3 bash scripts/run_batch.sh config.yaml
```

## 📁 Script Organization

### Core Scripts

| Script | Purpose |
|--------|---------|
| `test_policy.py` | Unified policy testing (Octo, Pi0.5, OpenVLA) |
| `run_batch.sh` | Unified batch runner for all policies |
| `run_all_droid_new_setup.py` | Python backend for batch runs |

### Policy-Specific Wrappers

| Script | Purpose |
|--------|---------|
| `run_all_octo.sh` | Run Octo batch experiments |
| `run_all_pi05.sh` | Run Pi0.5 batch experiments |
| `run_all_droid_new_setup.sh` | Legacy wrapper (uses default config) |

### Interactive & Replay

| Script | Purpose |
|--------|---------|
| `rollout_interact_pi.py` | Interactive rollout with Pi0.5 |
| `rollout_key_board.py` | Keyboard-controlled rollout |
| `rollout_replay_traj.py` | Replay recorded trajectories |

### Data Processing

| Script | Purpose |
|--------|---------|
| `gen_latent_simple.py` | Generate latent features (simple) |
| `gen_latent_chunk.py` | Generate latent features (chunked) |
| `prepare_validation_data.py` | Prepare validation data |
| `create_latent_symlinks.py` | Create symlinks for latents |
| `process_all_chunks_parallel.sh` | Parallel chunk processing |
| `validate_chunks_parallel.sh` | Parallel validation |

### Environment Setup

| Script | Purpose |
|--------|---------|
| `setup_octo_env.sh` | Setup Octo environment |
| `run_with_env.sh` | Run with specific environment |

### Training

| Script | Purpose |
|--------|---------|
| `train_wm.py` | Train world model |

## 📝 Usage Examples

### Example 1: Test Octo policy before running experiments

```bash
# Quick test with small model
python scripts/test_policy.py --policy octo --small --device cuda:7

# If tests pass, run batch experiments
bash scripts/run_all_octo.sh
```

### Example 2: Run experiments on GPU 6 and 7

Make sure your config file has:
```yaml
device: "cuda:7"  # Orchestrator
rollout:
  device: "cuda:6"  # World Model
router:
  available_policies:
    - device: "cuda:7"  # Policy
```

Then run:
```bash
bash scripts/run_all_octo.sh
```

### Example 3: Run specific scenarios

```bash
# Only run scenarios 0001-0005
mkdir -p /tmp/subset_annotations
cp dataset_example/droid_new_setup/annotation/val/000[1-5].json /tmp/subset_annotations/

ANN_DIR=/tmp/subset_annotations bash scripts/run_batch.sh omni_ctrl/configs/omni_ctrl_octo.yaml
```

### Example 4: Resume interrupted batch run

The `--skip-existing` flag (enabled by default) will skip scenarios that already have results:

```bash
# This will automatically skip completed scenarios
bash scripts/run_all_octo.sh
```

### Example 5: Human label rollout videos (auto-save)

For manual success/failure annotation on GR00T rollout videos:

```bash
python scripts/utils/human_annotation_server.py \
  --root /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/Exp_Groot \
  --output /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/Exp_Groot/human_annotations.json \
  --port 18765
```

Then open:

```text
http://127.0.0.1:18765/__annotate__/
```

Features:
- Auto-save after each label click
- Auto-save when editing comments
- If marked as failure, select a required failure reason
- Result JSON includes real-time `sr` (success / (success + failure))
- Summary includes failure-reason counts/distribution
- Resume from existing annotation JSON
- Keyboard shortcuts for fast labeling (`1/0/2/3`, `←/→`, `U`, `C`)

## 🔧 Troubleshooting

### OOM (Out of Memory) Errors

1. Check GPU memory: `nvidia-smi`
2. Use different GPUs in config
3. Use smaller model variant (e.g., octo-small)

### JAX CUDA Warnings (Octo)

If you see "Falling back to cpu", install CUDA-enabled JAX:
```bash
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### API Connection Errors (Evaluator)

If using OpenAI VLM evaluator:
1. Set API key: `export OPENAI_API_KEY=your-key`
2. Use VPN if needed
3. Or use DummyEvaluator for testing (modify config)

### Local Qwen-VL Evaluator

You can run local trajectory evaluation without OpenAI API by setting:

```yaml
evaluation:
  vlm_type: qwen-vl
  vlm_model: /mnt/nvme-fast/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
  eval_num_frames: 8
  frame_crop: bottom
```

Optional runtime overrides for local Qwen evaluator:
- `EVOW_QWEN_DEVICE` (e.g. `cuda:1`)
- `EVOW_QWEN_TORCH_DTYPE` (`auto` / `bf16` / `fp16` / `fp32`)
- `EVOW_QWEN_MAX_NEW_TOKENS`

For `Qwen3-VL-*` checkpoints, use a recent `transformers` version (>= 4.57).

## 🗑️ Deprecated Scripts

The following scripts are deprecated and may be removed in future versions:

- `run_all_pi05_batch.py` - Use `run_batch.sh` with pi05 config instead
- `test_octo_policy.py` - Use `test_policy.py --policy octo` instead
- `test_octo_on_droid.py` - Use `test_policy.py --policy octo` instead
