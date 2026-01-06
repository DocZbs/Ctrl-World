# Octo Policy Setup Guide

## ⚠️ Dependency Conflict Notice

Octo and OpenPI (Pi0.5) have conflicting JAX version requirements:
- **Octo** requires JAX 0.4.20
- **OpenPI** requires JAX 0.5.3

## Solution Options

### Option 1: Separate Environments (Recommended for Production)

Create separate conda environments for different policies:

```bash
# Environment for Pi0.5
conda create -n ctrl-world-pi05 python=3.11
conda activate ctrl-world-pi05
pip install -r requirements.txt
# Install OpenPI dependencies

# Environment for Octo
conda create -n ctrl-world-octo python=3.10
conda activate ctrl-world-octo
pip install -r requirements.txt
cd /tmp && git clone https://github.com/octo-models/octo.git
cd octo && pip install -e . && pip install -r requirements.txt
```

Then run experiments with the appropriate environment:
```bash
# For Pi0.5 experiments
conda activate ctrl-world-pi05
python experiments/run_omni_ctrl_mvp.py --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml

# For Octo experiments
conda activate ctrl-world-octo
python experiments/run_omni_ctrl_mvp.py --config omni_ctrl/configs/omni_ctrl_octo.yaml
```

### Option 2: Use Octo with Relaxed Dependencies (Quick Testing)

For quick testing, you can try upgrading JAX and accepting dependency conflicts:

```bash
conda activate ctrl-world
pip install --upgrade "jax[cuda12]>=0.4.38"
```

**Note**: This may cause issues with OpenPI/Pi0.5 policy. Use this only if you're testing Octo standalone.

### Option 3: Use PyTorch Backend (Experimental)

Octo has experimental PyTorch support which avoids JAX conflicts:

```bash
pip install git+https://github.com/emb-ai/octo-pytorch.git
```

Then modify `OctoPolicy` to use the PyTorch backend (see implementation notes below).

## Downloading Octo Models

Octo models will automatically download from HuggingFace on first use:

```python
from octo.model.octo_model import OctoModel

# This will download ~100MB for octo-small or ~400MB for octo-base
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
```

Available checkpoints:
- `hf://rail-berkeley/octo-small-1.5` (27M params, ~100MB)
- `hf://rail-berkeley/octo-base-1.5` (93M params, ~400MB)

Models are cached in `~/.cache/huggingface/` and won't re-download on subsequent runs.

## Manual Model Download

If you prefer to download manually:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/rail-berkeley/octo-small-1.5

# Then update config to use local path:
# checkpoint: "/path/to/octo-small-1.5"
```

## Verifying Installation

Test your setup with:

```bash
python scripts/test_octo_policy.py
```

This will:
1. Check if Octo can be imported
2. Load a pretrained model (downloads if needed)
3. Run inference with dummy observations
4. Test PolicyRouter integration

## Usage Example

```python
from omni_ctrl.configs import OmniCtrlConfig
from omni_ctrl.core import OmniCtrlOrchestrator

# Load config with Octo policy
config = OmniCtrlConfig.from_yaml("omni_ctrl/configs/omni_ctrl_octo.yaml")

# Run experiment
orchestrator = OmniCtrlOrchestrator(config)
orchestrator.run()
```

## Multi-Policy Ablation Studies

For ablation studies comparing multiple VLA models:

### Approach 1: Sequential Runs
Run each policy in its own environment sequentially:

```bash
# Run Pi0.5 baseline
conda activate ctrl-world-pi05
python experiments/run_omni_ctrl_mvp.py --config configs/pi05.yaml --output results/pi05

# Run Octo
conda activate ctrl-world-octo
python experiments/run_omni_ctrl_mvp.py --config configs/octo.yaml --output results/octo

# Run OpenVLA
conda activate ctrl-world-pi05  # OpenVLA compatible with Pi0.5 env
python experiments/run_omni_ctrl_mvp.py --config configs/openvla.yaml --output results/openvla
```

### Approach 2: Containerized Policies (Advanced)
Use Docker containers for each policy to avoid dependency conflicts entirely.

## Troubleshooting

### Issue: `RuntimeError: jaxlib version X is incompatible with jax version Y`

**Solution**: You're mixing JAX versions from Octo and OpenPI. Use separate environments (Option 1).

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: Install Octo dependencies:
```bash
cd /tmp/octo
pip install -r requirements.txt
```

### Issue: Slow model loading

**Solution**: Models download on first use. Subsequent runs will be faster as models are cached.

### Issue: CUDA out of memory

**Solution**:
- Use `octo-small` instead of `octo-base`
- Reduce batch size in rollout config
- Use a GPU with more memory

## Additional Resources

- [Octo GitHub Repository](https://github.com/octo-models/octo)
- [Octo Paper](https://arxiv.org/abs/2405.12213)
- [HuggingFace Models](https://huggingface.co/rail-berkeley)
- [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/)
