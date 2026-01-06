#!/bin/bash
# Process chunk-000 videos to generate latents

SVD_PATH="/data1/zbs_files/data/HF/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CHUNK_DIR="data/videos/chunk-001"
OUTPUT_DIR="data/latents/chunk-001"
DEVICE="cuda:4"

# Process all episodes in chunk-000 (0-999)
python scripts/gen_latent_chunk.py \
    --chunk-dir "$CHUNK_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --svd-path "$SVD_PATH" \
    --start-id 1000 \
    --end-id 1999 \
    --device "$DEVICE"
