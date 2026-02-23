#!/usr/bin/env bash
set -euo pipefail

# Batch generate synthetic DROID data from real DROID episodes
# Similar to run_all_droid_new_setup_finetuned.sh but for data generation

# Disable HuggingFace mirror and use local cache to avoid proxy issues
# unset HF_ENDPOINT
# export HF_HUB_OFFLINE=1  # Comment out to allow downloads if models not cached
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration
DATA_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data"
OUTPUT_DIR="synthetic_data/droid_1000_generated"
PI_CKPT="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid"
POLICY_TYPE="pi05"
START_EPISODE=0
END_EPISODE=999
WM_DEVICE="cuda:0"
POLICY_DEVICE="cuda:1"

# World model configuration
WM_CKPT="/mnt/nvme-fast/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt"
DATA_STAT_PATH="dataset_meta_info/droid/stat.json"
ACTION_ADAPTER_PATH="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/models/action_adapter/model2_15_9.pth"
SVD_MODEL_PATH="/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP_MODEL_PATH="/mnt/nvme-fast/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MAX_GEN_STEPS=50
SAVE_EVERY=10

# Check if required paths are set
if [ -z "$WM_CKPT" ]; then
    echo "Error: WM_CKPT (world model checkpoint) must be set"
    echo "Please edit this script and set WM_CKPT to your world model checkpoint path"
    exit 1
fi

echo "=========================================="
echo "Batch Generate Synthetic DROID Data"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Source data: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Policy: $PI_CKPT ($POLICY_TYPE)"
echo "  World model: $WM_CKPT"
echo "  Episode range: $START_EPISODE - $END_EPISODE"
echo "  WM device: $WM_DEVICE"
echo "  Policy device: $POLICY_DEVICE"
echo "  Max steps per episode: $MAX_GEN_STEPS"
echo ""

# Check required directories
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR/videos/chunk-000" ]; then
    echo "Error: Videos directory not found: $DATA_DIR/videos/chunk-000"
    exit 1
fi

if [ ! -d "$DATA_DIR/data/chunk-000" ]; then
    echo "Error: Data directory not found: $DATA_DIR/data/chunk-000"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting batch generation..."
echo ""

# Run generation
python scripts/inference/generate_synthetic_droid_batch.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pi-ckpt "$PI_CKPT" \
    --policy-type "$POLICY_TYPE" \
    --wm-ckpt "$WM_CKPT" \
    --start-episode $START_EPISODE \
    --end-episode $END_EPISODE \
    --wm-device "$WM_DEVICE" \
    --policy-device "$POLICY_DEVICE" \
    --max-gen-steps $MAX_GEN_STEPS \
    --save-every $SAVE_EVERY \
    --svd-model-path "$SVD_MODEL_PATH" \
    --clip-model-path "$CLIP_MODEL_PATH" \
    ${DATA_STAT_PATH:+--data-stat-path "$DATA_STAT_PATH"} \
    ${ACTION_ADAPTER_PATH:+--action-adapter-path "$ACTION_ADAPTER_PATH"}

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check statistics:"
echo "  cat $OUTPUT_DIR/stats.json"
echo ""
echo "Check errors (if any):"
echo "  cat $OUTPUT_DIR/errors.json"
echo ""
echo "Generated data structure:"
echo "  $OUTPUT_DIR/generated_episodes/"
echo "    ├── videos/chunk-000/"
echo "    │   ├── observation.images.exterior_1_left/"
echo "    │   ├── observation.images.exterior_2_left/"
echo "    │   └── observation.images.wrist_left/"
echo "    └── data/chunk-000/"
echo "        └── episode_*.parquet"
echo ""
