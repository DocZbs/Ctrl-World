#!/bin/bash
# Generate synthetic trajectories for Pi0.5 DROID finetuning

# Run from project root directory
cd "$(dirname "$0")/../.."

# Configuration
ANNOTATION_FILE="dataset_example/droid_new_setup/annotation/val/0002.json"
DATASET_ROOT="dataset_example/droid_new_setup"
OUTPUT_DIR="synthetic_data/pickplace_fixed"
NUM_ROLLOUTS=100
MAX_STEPS=30


# Run generation
python scripts/data_processing/generate_synthetic_trajectories.py \
    --annotation-file "$ANNOTATION_FILE" \
    --dataset-root "$DATASET_ROOT" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --output-dir "$OUTPUT_DIR" \
    --no-vlm \
    --max-steps "$MAX_STEPS" \
    --save-every 1



# Verify generated data
python scripts/data_processing/verify_data_format.py \
    "$OUTPUT_DIR/annotation/synthetic/"

echo ""
echo "Done! Data saved to: $OUTPUT_DIR"
