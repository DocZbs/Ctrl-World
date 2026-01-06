#!/bin/bash
# Process chunk-001 to chunk-010 in parallel using multiple GPUs

SVD_PATH="/data1/zbs_files/data/HF/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"

# Function to process a single chunk
process_chunk() {
    chunk_id=$1
    gpu_id=$2

    CHUNK_NUM=$(printf "%03d" "$chunk_id")
    CHUNK_DIR="data/videos/chunk-${CHUNK_NUM}"
    OUTPUT_DIR="data/latents/chunk-${CHUNK_NUM}"

    START_ID=$((chunk_id * 1000))
    END_ID=$((START_ID + 999))
    DEVICE="cuda:${gpu_id}"

    echo "[$(date)] Starting chunk-${CHUNK_NUM} on ${DEVICE}"

    python scripts/gen_latent_chunk.py \
        --chunk-dir "$CHUNK_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --svd-path "$SVD_PATH" \
        --start-id "$START_ID" \
        --end-id "$END_ID" \
        --device "$DEVICE" \
        > logs/chunk_${CHUNK_NUM}.log 2>&1

    echo "[$(date)] Completed chunk-${CHUNK_NUM} on ${DEVICE}"
}

# Create logs directory
mkdir -p logs

# Process chunks in parallel using 5 GPUs (2,4,5,6,7)
echo "Starting parallel processing with 5 GPUs (cuda:2,4,5,6,7)..."

# Batch 1: chunk 1-5 (GPU 2,4,5,6,7)
process_chunk 1 2 &
process_chunk 2 4 &
process_chunk 3 5 &
process_chunk 4 6 &
process_chunk 5 7 &
wait
echo "Batch 1 (chunks 1-5) completed!"

# Batch 2: chunk 6-10 (GPU 2,4,5,6,7)
process_chunk 6 2 &
process_chunk 7 4 &
process_chunk 8 5 &
process_chunk 9 6 &
process_chunk 10 7 &
wait
echo "Batch 2 (chunks 6-10) completed!"

echo ""
echo "All chunks (001-010) processed!"
echo "Check logs in logs/ directory for details"
