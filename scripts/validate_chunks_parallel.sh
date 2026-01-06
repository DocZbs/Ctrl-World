#!/bin/bash
# Parallel validation for chunk-001, chunk-002, chunk-003
# Using GPU pairs: 2-3, 4-5, 6-7

# Create logs directory
mkdir -p logs/validation

# Function to validate a single chunk
validate_chunk() {
    local chunk_id=$1
    local gpu_devices=$2
    local log_file="logs/validation/chunk_$(printf "%03d" $chunk_id).log"

    echo "Starting validation for chunk-$chunk_id on GPU $gpu_devices"
    echo "Log file: $log_file"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_devices

        # Activate conda environment
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ctrl-world

        echo "=== Validation for chunk-$chunk_id started at $(date) ===" > "$log_file" 2>&1
        echo "Using GPUs: $gpu_devices" >> "$log_file" 2>&1
        echo "" >> "$log_file" 2>&1

        # Step 1: Prepare validation annotations
        echo "Step 1: Preparing validation annotations..." >> "$log_file" 2>&1
        gg python scripts/prepare_validation_data.py \
            --data-root data \
            --chunk-id $chunk_id \
            --output-dir data/validation >> "$log_file" 2>&1

        echo "" >> "$log_file" 2>&1
        echo "Step 2: Creating symlinks for latents..." >> "$log_file" 2>&1
        gg python scripts/create_latent_symlinks.py \
            --data-root data \
            --chunk-id $chunk_id \
            --split val >> "$log_file" 2>&1

        echo "" >> "$log_file" 2>&1
        echo "Step 3: Running validation..." >> "$log_file" 2>&1
        gg python scripts/run_all_droid_new_setup.py \
            --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
            --ann-dir data/validation/annotation/val \
            --droid-root data \
            --out-base experiments/validation_chunk_$(printf "%03d" $chunk_id) \
            --iterations 1 >> "$log_file" 2>&1

        echo "" >> "$log_file" 2>&1
        echo "=== Validation for chunk-$chunk_id completed at $(date) ===" >> "$log_file" 2>&1
        echo "Results in experiments/validation_chunk_$(printf "%03d" $chunk_id)/" >> "$log_file" 2>&1
    ) &
}

# Launch 3 parallel validation tasks
echo "Launching parallel validation tasks..."
echo "================================================"

validate_chunk 1 "2,3"
# validate_chunk 2 "4,5"
# validate_chunk 3 "6,7"