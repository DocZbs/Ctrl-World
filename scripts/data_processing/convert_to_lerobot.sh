#!/bin/bash
# Convert synthetic data to LeRobot format (Parquet)

set -e

cd "$(dirname "$0")/../.."

INPUT_DIR="synthetic_data/pickplace_0002"
OUTPUT_NAME="synthetic_pickplace_0002"

echo "Converting synthetic data to LeRobot format (Parquet)..."
echo "Input: $INPUT_DIR"
echo "Output: local/$OUTPUT_NAME"
echo ""

python scripts/data_processing/push_to_lerobot_simple.py \
    --input-dir "$INPUT_DIR" \
    --output-name "$OUTPUT_NAME"

echo ""
echo "Done! You can now train with:"
echo "  --data.repo-id local/$OUTPUT_NAME"
