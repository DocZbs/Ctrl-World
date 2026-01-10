# Ctrl-World + π0.5 合成数据微调完整流程

## 概述

本文档描述如何实现论文中的完整流程：**在世界模型中生成合成数据，然后用这些数据微调π0.5策略**。

### 核心思想

1. **Policy-in-the-loop Imagination Rollout**: 在世界模型中让策略与环境交互，生成合成轨迹
2. **人工偏好标注**: 人工判断哪些轨迹成功，哪些失败
3. **监督学习微调**: 用成功的轨迹对π0.5进行监督学习（MSE loss）

### 论文中的关键数字

- 每个任务生成 **~400条** 合成轨迹
- 保留其中 **25-50条** 成功的轨迹
- 微调 **2k steps**，使用 **4张H100**

---

## 完整流程

### Step 1: 生成合成轨迹

使用世界模型和策略生成多样化的合成轨迹。

#### 1.1 选择初始场景

从 `dataset_example/droid_new_setup` 中选择一个场景：

```bash
ls /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/dataset_example/droid_new_setup/annotation/val/
# 输出: 0001.json, 0002.json, ..., 0013.json
```

查看场景任务：
```bash
cat /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/dataset_example/droid_new_setup/annotation/val/0002.json | grep texts
# "texts": ["pick the blue block and place it in plate"]
```

#### 1.2 准备指令变体（可选）

为了增加多样性，可以用LLM改写指令：

```python
# 原始指令
"pick the blue block and place it in plate"

# 变体
[
    "pick the blue block and place it in plate",
    "pick up the blue block and put it inside the plate",
    "grasp the blue cube and move it to the plate",
    "take the blue object and place it in the dish",
]
```

#### 1.3 运行合成轨迹生成

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World

python scripts/generate_synthetic_trajectories.py \
    --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json \
    --dataset-root dataset_example/droid_new_setup \
    --num-rollouts 400 \
    --output-dir synthetic_data/pickplace_0002 \
    --instruction-variants \
        "pick the blue block and place it in plate" \
        "pick up the blue block and put it inside the plate" \
        "grasp the blue cube and move it to the plate" \
        "take the blue object and place it in the dish"
```

**参数说明**:
- `--annotation-file`: 初始场景的annotation文件
- `--num-rollouts`: 生成轨迹数量（论文中是400）
- `--instruction-variants`: 指令变体列表（增加多样性）

**输出**:
```
synthetic_data/pickplace_0002/
├── syn_000001/
│   ├── metadata.json
│   ├── video.mp4
│   └── images/
│       ├── 0000.png
│       ├── 0001.png
│       └── ...
├── syn_000002/
└── ...
```

**预计时间**: 400条轨迹约需要几小时（取决于GPU和轨迹长度）

---

### Step 2: 标注成功轨迹

人工观看视频并标注哪些轨迹成功。

#### 2.1 交互式标注

```bash
python scripts/label_trajectories.py \
    --input-dir synthetic_data/pickplace_0002 \
    --output-file synthetic_data/pickplace_0002/labels.json
```

**操作说明**:
- 按 `s` 标记为成功
- 按 `f` 标记为失败
- 按 `q` 跳过当前视频
- 按 `n` 跳到下一个

**输出**: `labels.json`
```json
{
  "syn_000001": true,
  "syn_000002": false,
  "syn_000003": true,
  ...
}
```

#### 2.2 筛选成功轨迹

```bash
python scripts/label_trajectories.py \
    --input-dir synthetic_data/pickplace_0002 \
    --labels-file synthetic_data/pickplace_0002/labels.json \
    --filter-success \
    --output-dir synthetic_data/pickplace_0002_success
```

**目标**: 保留 **25-50条** 成功的轨迹（论文中的数字）

---

### Step 3: 转换为LeRobot格式

将成功的轨迹转换为LeRobot格式，用于π0.5微调。

```bash
python scripts/convert_synthetic_to_lerobot.py \
    --input-dir synthetic_data/pickplace_0002_success \
    --output-dir lerobot_synthetic/pickplace_0002 \
    --repo-id "local://lerobot_synthetic/pickplace_0002"
```

**输出**:
```
lerobot_synthetic/pickplace_0002/
├── episode_000000/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── episode_000001/
├── episodes.json
└── meta.json
```

---

### Step 4: 配置π0.5微调

#### 4.1 修改训练配置

编辑 `openpi/src/openpi/training/config.py`，添加新配置：

```python
TrainConfig(
    name="pi05_synthetic_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
    ),
    data=LeRobotDROIDDataConfig(
        repo_id="local:///mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/lerobot_synthetic/pickplace_0002",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            # 重要：使用原始DROID的归一化统计量
            assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
            asset_id="droid",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"
    ),
    num_train_steps=2_000,  # 论文中是2k steps
    batch_size=32,  # 根据GPU调整
    lr_schedule=_optimizer.LRSchedule(
        warmup_steps=100,
        decay_steps=2_000,
        peak_lr=1e-5,  # 较低的学习率用于微调
    ),
),
```

#### 4.2 或者使用现有配置

如果不想修改config.py，可以使用 `pi05_droid_finetune` 配置，只需修改 `repo_id`。

---

### Step 5: 运行π0.5微调

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi

# 设置环境变量
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 运行微调
uv run scripts/train.py pi05_synthetic_finetune \
    --exp-name=pickplace_0002_finetune \
    --overwrite
```

**训练参数**:
- **Steps**: 2,000（论文中的设置）
- **Batch size**: 32（根据GPU调整）
- **Learning rate**: 1e-5（较低的学习率）
- **GPU**: 4x H100（论文中的设置，可以用更少的GPU但会更慢）

**预计时间**:
- 4x H100: ~1-2小时
- 1x H100: ~4-8小时
- 1x A100: ~8-12小时

**输出**:
```
openpi/checkpoints/pickplace_0002_finetune/
├── params/
│   ├── checkpoint_001000
│   ├── checkpoint_002000
│   └── ...
├── assets/
└── wandb_id.txt
```

---

### Step 6: 测试微调后的模型

#### 6.1 在Ctrl-World中测试

修改 `omni_ctrl/configs/omni_ctrl_pi05_batch.yaml`:

```yaml
router:
  available_policies:
    - name: "pi05"
      checkpoint: "/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/checkpoints/pickplace_0002_finetune"
      action_space: "joint_vel"
```

运行测试：
```bash
python scripts/run_all_droid_new_setup.py \
    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
    --ann-dir dataset_example/droid_new_setup/annotation/val \
    --droid-root dataset_example/droid_new_setup \
    --out-base experiments/test_finetuned_pi05
```

#### 6.2 分析成功率

```bash
python scripts/analyze_success_rate.py \
    experiments/test_finetuned_pi05
```

---

## 完整示例脚本

创建一个端到端的脚本 `scripts/full_synthetic_finetune_pipeline.sh`:

```bash
#!/bin/bash
# 完整的合成数据微调流程

set -e

TASK_NAME="pickplace_0002"
ANNOTATION_FILE="dataset_example/droid_new_setup/annotation/val/0002.json"
NUM_ROLLOUTS=400
TARGET_SUCCESS=30  # 目标成功轨迹数

echo "=========================================="
echo "Synthetic Data Fine-tuning Pipeline"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Target rollouts: $NUM_ROLLOUTS"
echo "Target successful trajectories: $TARGET_SUCCESS"
echo "=========================================="

# Step 1: Generate synthetic trajectories
echo ""
echo "[Step 1/6] Generating synthetic trajectories..."
python scripts/generate_synthetic_trajectories.py \
    --annotation-file $ANNOTATION_FILE \
    --num-rollouts $NUM_ROLLOUTS \
    --output-dir synthetic_data/$TASK_NAME \
    --instruction-variants \
        "pick the blue block and place it in plate" \
        "pick up the blue block and put it inside the plate" \
        "grasp the blue cube and move it to the plate"

# Step 2: Label trajectories
echo ""
echo "[Step 2/6] Labeling trajectories (interactive)..."
echo "Please watch videos and label success/failure"
python scripts/label_trajectories.py \
    --input-dir synthetic_data/$TASK_NAME \
    --output-file synthetic_data/$TASK_NAME/labels.json

# Step 3: Filter successful trajectories
echo ""
echo "[Step 3/6] Filtering successful trajectories..."
python scripts/label_trajectories.py \
    --input-dir synthetic_data/$TASK_NAME \
    --labels-file synthetic_data/$TASK_NAME/labels.json \
    --filter-success \
    --output-dir synthetic_data/${TASK_NAME}_success

# Check if we have enough successful trajectories
SUCCESS_COUNT=$(ls synthetic_data/${TASK_NAME}_success | wc -l)
echo "Found $SUCCESS_COUNT successful trajectories"

if [ $SUCCESS_COUNT -lt 10 ]; then
    echo "Warning: Only $SUCCESS_COUNT successful trajectories (recommended: 25-50)"
    echo "Consider generating more rollouts or adjusting labeling criteria"
fi

# Step 4: Convert to LeRobot format
echo ""
echo "[Step 4/6] Converting to LeRobot format..."
python scripts/convert_synthetic_to_lerobot.py \
    --input-dir synthetic_data/${TASK_NAME}_success \
    --output-dir lerobot_synthetic/$TASK_NAME \
    --repo-id "local://lerobot_synthetic/$TASK_NAME"

# Step 5: Fine-tune π0.5
echo ""
echo "[Step 5/6] Fine-tuning π0.5..."
echo "Note: Make sure you've updated config.py with the correct repo_id"
echo "Press Enter to continue or Ctrl+C to abort..."
read

cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_synthetic_finetune \
    --exp-name=${TASK_NAME}_finetune \
    --overwrite

cd ..

# Step 6: Test fine-tuned model
echo ""
echo "[Step 6/6] Testing fine-tuned model..."
python scripts/run_all_droid_new_setup.py \
    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
    --ann-dir dataset_example/droid_new_setup/annotation/val \
    --droid-root dataset_example/droid_new_setup \
    --out-base experiments/test_${TASK_NAME}_finetuned

# Analyze results
python scripts/analyze_success_rate.py \
    experiments/test_${TASK_NAME}_finetuned

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Synthetic data: synthetic_data/${TASK_NAME}_success"
echo "LeRobot dataset: lerobot_synthetic/$TASK_NAME"
echo "Fine-tuned checkpoint: openpi/checkpoints/${TASK_NAME}_finetune"
echo "Test results: experiments/test_${TASK_NAME}_finetuned"
echo "=========================================="
```

---

## 关键参数调整

### 生成阶段

| 参数 | 论文值 | 建议范围 | 说明 |
|------|--------|---------|------|
| `num_rollouts` | 400 | 200-600 | 生成的轨迹总数 |
| `instruction_variants` | 3-5个 | 2-10个 | 指令变体数量 |
| `state_perturbation_std` | 0.05 | 0.01-0.1 | 初始状态扰动 |

### 标注阶段

| 指标 | 论文值 | 建议范围 |
|------|--------|---------|
| 成功轨迹数 | 25-50 | 20-100 |
| 成功率 | 6-12% | 5-25% |

### 微调阶段

| 参数 | 论文值 | 建议范围 | 说明 |
|------|--------|---------|------|
| `num_train_steps` | 2,000 | 1k-5k | 训练步数 |
| `batch_size` | 32 | 16-64 | 批大小 |
| `learning_rate` | 1e-5 | 1e-6 to 1e-4 | 学习率 |
| `GPUs` | 4x H100 | 1-8 | GPU数量 |

---

## 故障排查

### Q1: 生成的轨迹质量差？

**可能原因**:
- 世界模型不准确
- Action adapter不匹配
- 初始状态不合适

**解决**:
- 检查世界模型checkpoint是否正确
- 确认action adapter已微调
- 尝试不同的初始场景

### Q2: 成功率太低（<5%）？

**可能原因**:
- 任务太难
- 策略不适合该任务
- 标注标准太严格

**解决**:
- 选择更简单的任务
- 增加rollout数量
- 放宽成功标准

### Q3: 微调后性能没提升？

**可能原因**:
- 成功轨迹太少
- 学习率不合适
- 训练步数不够

**解决**:
- 确保至少有25条成功轨迹
- 调整学习率（1e-6 to 1e-4）
- 增加训练步数到5k

### Q4: 内存不足？

**解决**:
```bash
# 减小batch size
batch_size=16

# 限制JAX内存
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# 使用梯度累积
# 在config.py中设置 gradient_accumulation_steps=2
```

---

## 总结

### 完整流程回顾

1. ✓ **生成**: 400条合成轨迹（几小时）
2. ✓ **标注**: 筛选25-50条成功轨迹（人工）
3. ✓ **转换**: 转为LeRobot格式（几分钟）
4. ✓ **微调**: 2k steps监督学习（1-8小时）
5. ✓ **测试**: 在Ctrl-World中评估

### 预期效果

根据论文，这个流程可以：
- 提升策略在特定任务上的成功率
- 改善指令跟随能力
- 增强对初始状态变化的鲁棒性

### 下一步

- 尝试不同的任务
- 实验不同的多样性策略
- 对比微调前后的性能

祝微调成功！
