#!/bin/bash
# 测试合成数据生成流程（少量rollout）

set -e

echo "=========================================="
echo "Testing Synthetic Data Generation"
echo "=========================================="

# 配置
TASK_NAME="pickplace_0002_test"
ANNOTATION_FILE="dataset_example/droid_new_setup/annotation/val/0002.json"
NUM_ROLLOUTS=10  # 先测试10条
OUTPUT_DIR="synthetic_data/${TASK_NAME}"

# 检查annotation文件
echo ""
echo "[1/4] Checking annotation file..."
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Error: Annotation file not found: $ANNOTATION_FILE"
    exit 1
fi

# 显示任务信息
TASK_TEXT=$(cat $ANNOTATION_FILE | python3 -c "import sys, json; print(json.load(sys.stdin)['texts'][0])")
echo "Task: $TASK_TEXT"

# 准备指令变体
echo ""
echo "[2/4] Preparing instruction variants..."
INSTRUCTION_VARIANTS=(
    "$TASK_TEXT"
    "pick up the blue block and put it inside the plate"
    "grasp the blue cube and move it to the plate"
    "take the blue object and place it in the dish"
)

echo "Instruction variants:"
for i in "${!INSTRUCTION_VARIANTS[@]}"; do
    echo "  $((i+1)). ${INSTRUCTION_VARIANTS[$i]}"
done

# 生成合成轨迹
echo ""
echo "[3/4] Generating $NUM_ROLLOUTS synthetic trajectories..."
echo "This will take a few minutes..."

python scripts/generate_synthetic_trajectories.py \
    --annotation-file $ANNOTATION_FILE \
    --num-rollouts $NUM_ROLLOUTS \
    --output-dir $OUTPUT_DIR \
    --instruction-variants "${INSTRUCTION_VARIANTS[@]}"

# 检查输出
echo ""
echo "[4/4] Checking output..."
if [ -d "$OUTPUT_DIR" ]; then
    TRAJ_COUNT=$(ls -d $OUTPUT_DIR/syn_* 2>/dev/null | wc -l)
    echo "✓ Generated $TRAJ_COUNT trajectories in $OUTPUT_DIR"

    # 显示第一个轨迹的信息
    FIRST_TRAJ=$(ls -d $OUTPUT_DIR/syn_* 2>/dev/null | head -1)
    if [ -n "$FIRST_TRAJ" ]; then
        echo ""
        echo "Sample trajectory: $(basename $FIRST_TRAJ)"
        if [ -f "$FIRST_TRAJ/metadata.json" ]; then
            echo "Metadata:"
            cat "$FIRST_TRAJ/metadata.json" | python3 -m json.tool | head -20
        fi
    fi
else
    echo "✗ Output directory not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review the generated trajectories"
echo "2. If quality is good, run full 400 rollouts"
echo "3. Label and filter successful trajectories"
echo "=========================================="
