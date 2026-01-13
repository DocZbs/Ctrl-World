#!/bin/bash
# Process chunk-000 videos to generate latents

SVD_PATH="/mnt/eb20f54b-f016-4501-a5b2-5dd6afff9d90/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CHUNK_DIR="droid_data/videos/chunk-000"
OUTPUT_DIR="droid_data/latents/chunk-000"
DEVICE="cuda:1"

# Process all episodes in chunk-000 (0-999)
python scripts/gen_latent_chunk.py \
    --chunk-dir "$CHUNK_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --svd-path "$SVD_PATH" \
    --start-id 0 \
    --end-id 999 \
    --device "$DEVICE"
