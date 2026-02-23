#!/bin/bash

# Generate latents for chunk-001 to chunk-009 (episodes 1000-9999)

# Activate conda environment
source /mnt/nvme-fast/zbs/miniconda3/bin/activate ctrl-world

DROID_ROOT="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data"
SVD_PATH="/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
DEVICE="cuda:1"

# Get episodes from 1000 to 9999
ANNO_DIR="$DROID_ROOT/annotation/train"

# Create episode list file
EPISODE_LIST="/tmp/episodes_001_009.txt"
rm -f "$EPISODE_LIST"
touch "$EPISODE_LIST"

for i in $(seq 1000 9999); do
    if [ -f "$ANNO_DIR/${i}.json" ]; then
        printf "%06d\n" $i >> "$EPISODE_LIST"
    fi
done

echo "Found $(wc -l < $EPISODE_LIST) episodes in range 001000-009999"

# Generate latents
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World
python scripts/data_processing/gen_latents_batch.py \
    --droid-root "$DROID_ROOT" \
    --svd-path "$SVD_PATH" \
    --device "$DEVICE" \
    --episode-file "$EPISODE_LIST"

echo "Done!"
