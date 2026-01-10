#!/bin/bash
# 使用多GPU加速合成数据生成

set -e

echo "=========================================="
echo "Multi-GPU Synthetic Data Generation"
echo "=========================================="
echo "GPU 0: World Model (Ctrl-World)"
echo "GPU 1: Policy (π0.5)"
echo "=========================================="

# 配置
TASK_NAME="pickplace_0002"
ANNOTATION_FILE="dataset_example/droid_new_setup/annotation/val/0002.json"
NUM_ROLLOUTS=400
OUTPUT_DIR="synthetic_data/${TASK_NAME}"

# 检查GPU
echo ""
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# 显示任务信息
echo ""
echo "Task information:"
TASK_TEXT=$(cat $ANNOTATION_FILE | python3 -c "import sys, json; print(json.load(sys.stdin)['texts'][0])")
echo "  Task: $TASK_TEXT"
echo "  Rollouts: $NUM_ROLLOUTS"
echo "  Output: $OUTPUT_DIR"

# 准备指令变体
INSTRUCTION_VARIANTS=(
    "pick the blue block and place it in plate"
    "pick up the blue block and put it inside the plate"
    "grasp the blue cube and move it to the plate"
    "take the blue object and place it in the dish"
)

echo ""
echo "Instruction variants:"
for i in "${!INSTRUCTION_VARIANTS[@]}"; do
    echo "  $((i+1)). ${INSTRUCTION_VARIANTS[$i]}"
done

# 运行生成（使用多GPU）
echo ""
echo "Starting generation with multi-GPU acceleration..."
echo "This will take a few hours for 400 rollouts..."
echo ""

python scripts/generate_synthetic_trajectories.py \
    --annotation-file $ANNOTATION_FILE \
    --num-rollouts $NUM_ROLLOUTS \
    --output-dir $OUTPUT_DIR \
    --wm-device cuda:0 \
    --policy-device cuda:1 \
    --instruction-variants "${INSTRUCTION_VARIANTS[@]}"

# 检查结果
echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="

if [ -d "$OUTPUT_DIR" ]; then
    TRAJ_COUNT=$(ls -d $OUTPUT_DIR/syn_* 2>/dev/null | wc -l)
    echo "Generated: $TRAJ_COUNT trajectories"
    echo "Location: $OUTPUT_DIR"

    # 显示磁盘使用
    DISK_USAGE=$(du -sh $OUTPUT_DIR | cut -f1)
    echo "Disk usage: $DISK_USAGE"

    echo ""
    echo "Next steps:"
    echo "1. Label trajectories:"
    echo "   python scripts/label_trajectories.py --input-dir $OUTPUT_DIR --output-file $OUTPUT_DIR/labels.json"
    echo ""
    echo "2. Filter successful ones:"
    echo "   python scripts/label_trajectories.py --input-dir $OUTPUT_DIR --labels-file $OUTPUT_DIR/labels.json --filter-success --output-dir ${OUTPUT_DIR}_success"
else
    echo "Error: Output directory not created"
    exit 1
fi

echo "=========================================="
