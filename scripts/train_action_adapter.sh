#!/bin/bash

# Train Action Adapter on DROID Dataset

# Default configuration
DATA_PATH="dataset_example/droid_subset"
META_PATH="dataset_meta_info/droid_subset"
OUTPUT_DIR="models/action_adapter/checkpoints"
NUM_EPOCHS=10
BATCH_SIZE=128
LR=1e-4
NUM_WORKERS=8
NUM_FRAMES=15
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --meta_path)
            META_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Training Action Adapter"
echo "========================================================================"
echo "Data Path:      $DATA_PATH"
echo "Meta Path:      $META_PATH"
echo "Output Dir:     $OUTPUT_DIR"
echo "Num Epochs:     $NUM_EPOCHS"
echo "Batch Size:     $BATCH_SIZE"
echo "Learning Rate:  $LR"
echo "Num Workers:    $NUM_WORKERS"
echo "Num Frames:     $NUM_FRAMES"
echo "Device:         $DEVICE"
echo "========================================================================"
echo ""

# Run training
python models/action_adapter/train_adapter.py \
    --data_path "$DATA_PATH" \
    --meta_path "$META_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --num_frames "$NUM_FRAMES" \
    --device "$DEVICE"
