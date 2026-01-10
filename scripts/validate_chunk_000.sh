#!/bin/bash
# Validate chunk-000 episodes with non-empty tasks

# Step 1: Prepare validation annotations
echo "Step 1: Preparing validation annotations..."
python scripts/prepare_validation_data.py \
    --data-root droid_data \
    --chunk-id 0 \
    --output-dir droid_data/validation

echo ""
echo "Step 2: Creating symlinks for latents..."
python scripts/create_latent_symlinks.py \
    --data-root droid_data \
    --chunk-id 0 \
    --split val

echo ""
echo "Step 3: Running validation..."
# Step 3: Run validation
python scripts/run_all_droid_new_setup.py \
    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
    --ann-dir droid_data/validation/annotation/val \
    --droid-root droid_data \
    --out-base experiments/validation_chunk_000 \
    --iterations 1

echo ""
echo "Validation complete! Results in experiments/validation_chunk_000/"
