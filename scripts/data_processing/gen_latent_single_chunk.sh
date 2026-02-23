#!/bin/bash

# Generate latents for a single chunk
# Usage: ./gen_latent_single_chunk.sh <chunk_id> [device]
# Example: ./gen_latent_single_chunk.sh 001 cuda:1

CHUNK_ID=$1
DEVICE=${2:-cuda:1}

if [ -z "$CHUNK_ID" ]; then
    echo "Usage: $0 <chunk_id> [device]"
    echo "Example: $0 001 cuda:1"
    exit 1
fi

SVD_PATH="/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
DROID_ROOT="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data"

chunk_name="chunk-${CHUNK_ID}"
video_dir="${DROID_ROOT}/videos/${chunk_name}"
output_dir="${DROID_ROOT}/latents/${chunk_name}"

echo "=========================================="
echo "Processing ${chunk_name}"
echo "=========================================="
echo "Device: $DEVICE"
echo ""

if [ ! -d "$video_dir" ]; then
    echo "Error: Video directory not found: $video_dir"
    exit 1
fi

# Count episodes
num_episodes=$(ls "${video_dir}/observation.images.exterior_1_left/" | wc -l)
echo "Found $num_episodes episodes"

# Get episode ID range
first_episode=$(ls "${video_dir}/observation.images.exterior_1_left/" | head -1 | sed 's/episode_\([0-9]*\)\.mp4/\1/' | sed 's/^0*//')
last_episode=$(ls "${video_dir}/observation.images.exterior_1_left/" | tail -1 | sed 's/episode_\([0-9]*\)\.mp4/\1/' | sed 's/^0*//')

echo "Episode range: ${first_episode} to ${last_episode}"
echo ""

# Generate latents
python scripts/data_processing/gen_latent_chunk.py \
    --chunk-dir "${video_dir}" \
    --output-dir "${output_dir}" \
    --svd-path "${SVD_PATH}" \
    --start-id ${first_episode} \
    --end-id ${last_episode} \
    --device "${DEVICE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ${chunk_name} completed successfully"
    echo "Latents saved to: ${output_dir}"
else
    echo ""
    echo "✗ ${chunk_name} failed"
    exit 1
fi
