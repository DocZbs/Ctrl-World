#!/bin/bash

# Generate latents for DROID chunks 001-010
# Based on the existing chunk-000 latent generation

SVD_PATH="/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
DROID_ROOT="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data"
DEVICE="${1:-cuda:1}"

echo "=========================================="
echo "DROID Latent Generation for Chunks 001-010"
echo "=========================================="
echo "SVD Model: $SVD_PATH"
echo "DROID Root: $DROID_ROOT"
echo "Device: $DEVICE"
echo ""

for chunk_id in {001..010}; do
    chunk_name="chunk-${chunk_id}"
    video_dir="${DROID_ROOT}/videos/${chunk_name}"
    output_dir="${DROID_ROOT}/latents/${chunk_name}"

    echo ""
    echo "=========================================="
    echo "Processing ${chunk_name}"
    echo "=========================================="

    if [ ! -d "$video_dir" ]; then
        echo "Warning: Video directory not found: $video_dir"
        echo "Skipping ${chunk_name}..."
        continue
    fi

    # Count episodes in this chunk
    num_episodes=$(ls "${video_dir}/observation.images.exterior_1_left/" | wc -l)
    echo "Found $num_episodes episodes in ${chunk_name}"

    # Get episode ID range from filenames
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
        echo "✓ ${chunk_name} completed successfully"
    else
        echo "✗ ${chunk_name} failed"
    fi
done

echo ""
echo "=========================================="
echo "All chunks processed!"
echo "=========================================="
