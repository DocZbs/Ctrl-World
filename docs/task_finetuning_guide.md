# 按任务微调Action Adapter指南

## 概述

DROID数据集包含**758个不同的任务**，每个任务由自然语言指令描述（如"Close the drawer"、"Put the marker in the cup"等）。你可以选择一个特定任务进行微调，使模型专门适配该任务。

## 快速开始

### 1. 查看可用任务

```bash
# 列出前50个最常见的任务
python scripts/finetune_by_task.py --list-tasks

# 列出前100个任务
python scripts/finetune_by_task.py --list-tasks --top-n 100
```

输出示例：
```
  1. [  3 eps] Remove the marker from the mug
  2. [  3 eps] Remove the marker from the cup
  3. [  3 eps] Close the oven door
  4. [  3 eps] Remove the marker from the cup and put it on the table
  5. [  3 eps] Move the cloth
  ...
```

### 2. 选择任务进行微调

选择episode数量较多的任务（如3个以上）效果更好：

```bash
# 在"Close the drawer"任务上微调
python scripts/finetune_by_task.py --task "Close the drawer"

# 在"Remove the marker from the mug"任务上微调
python scripts/finetune_by_task.py --task "Remove the marker from the mug"
```

### 3. 自定义训练参数

```bash
python scripts/finetune_by_task.py \
    --task "Close the oven door" \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | (必需) | 任务指令（必须完全匹配） |
| `--list-tasks` | - | 列出所有可用任务 |
| `--top-n` | 50 | 列出前N个最常见任务 |
| `--data-root` | `droid_data` | DROID数据根目录 |
| `--chunks` | `[0]` | 使用的chunk列表 |
| `--pretrained-path` | `models/action_adapter/model2_15_9.pth` | 预训练模型 |
| `--output-dir` | `train_adapter/by_task` | 输出目录 |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--num-frames` | 15 | 预测帧数 |
| `--device` | cuda | 训练设备 |

## 输出文件

训练完成后会生成：

```
train_adapter/by_task/
├── adapter_task_a1b2c3d4_best.pth      # 最佳模型
├── adapter_task_a1b2c3d4_epoch1.pth    # 每个epoch的checkpoint
├── adapter_task_a1b2c3d4_epoch2.pth
├── ...
├── adapter_task_a1b2c3d4_history.json  # 训练历史
└── task_mapping.json                    # 任务名称映射
```

### 任务映射文件

`task_mapping.json` 记录了hash到任务名称的映射：
```json
{
  "a1b2c3d4": "Close the drawer",
  "e5f6g7h8": "Remove the marker from the mug"
}
```

## 使用示例

### 示例1: 微调"Close the drawer"任务

```bash
# 1. 查看该任务有多少episodes
python scripts/finetune_by_task.py --list-tasks | grep "Close the drawer"
# 输出: [  2 eps] Close the drawer

# 2. 开始微调
python scripts/finetune_by_task.py \
    --task "Close the drawer" \
    --epochs 30 \
    --batch-size 32

# 3. 查看训练历史
cat train_adapter/by_task/adapter_task_*_history.json
```

### 示例2: 微调多个相关任务

```bash
# 微调所有"marker"相关任务
python scripts/finetune_by_task.py --task "Remove the marker from the mug" --epochs 30
python scripts/finetune_by_task.py --task "Remove the marker from the cup" --epochs 30
python scripts/finetune_by_task.py --task "Put the marker in the pot" --epochs 30
```

### 示例3: 使用多个chunks

如果你有多个chunks的数据：

```bash
python scripts/finetune_by_task.py \
    --task "Close the drawer" \
    --chunks 0 1 2 \
    --epochs 20
```

## 训练建议

### 1. 任务选择

- **推荐**: 选择有3个以上episodes的任务
- **可行**: 2个episodes也可以，但可能过拟合
- **不推荐**: 只有1个episode的任务

### 2. 训练参数

根据任务的episode数量调整：

| Episodes数量 | Epochs | Batch Size | Learning Rate |
|-------------|--------|------------|---------------|
| 2-3         | 30-50  | 16-32      | 1e-4          |
| 4-10        | 20-30  | 32-64      | 1e-4          |
| 10+         | 10-20  | 64-128     | 1e-5          |

### 3. 避免过拟合

- 监控训练loss和验证loss的差距
- 如果验证loss不下降，减少epochs
- 使用较低的学习率（1e-5）

## 使用微调后的模型

### 方法1: 修改配置文件

```yaml
# omni_ctrl/configs/your_config.yaml
rollout:
  action_adapter_path: "train_adapter/by_task/adapter_task_a1b2c3d4_best.pth"
```

### 方法2: 命令行指定

```bash
python scripts/run_experiment.py \
    --config omni_ctrl/configs/your_config.yaml \
    --action-adapter train_adapter/by_task/adapter_task_a1b2c3d4_best.pth
```

## 对比不同任务的效果

创建一个批量测试脚本：

```bash
#!/bin/bash
# test_tasks.sh

tasks=(
    "Close the drawer"
    "Remove the marker from the mug"
    "Put the marker in the pot"
)

for task in "${tasks[@]}"; do
    echo "Testing task: $task"

    # 微调
    python scripts/finetune_by_task.py --task "$task" --epochs 30

    # 测试（需要你自己实现测试逻辑）
    # python scripts/test_adapter.py --task "$task"
done
```

## 常见问题

### Q1: 找不到指定的任务？

**错误**: `No episodes found for task: ...`

**原因**: 任务名称必须完全匹配（包括大小写、标点符号）

**解决**:
```bash
# 先列出任务，复制准确的名称
python scripts/finetune_by_task.py --list-tasks | grep -i "drawer"
```

### Q2: 任务的episodes太少？

**建议**:
- 选择有3个以上episodes的任务
- 或者使用多个chunks: `--chunks 0 1 2`
- 或者选择相似的任务一起训练

### Q3: 训练loss不下降？

**解决**:
1. 检查数据是否正确加载
2. 提高学习率: `--lr 1e-3`
3. 增加epochs: `--epochs 50`
4. 减小batch_size: `--batch-size 16`

### Q4: 如何知道哪个模型对应哪个任务？

查看 `task_mapping.json`:
```bash
cat train_adapter/by_task/task_mapping.json
```

或者查看训练历史：
```bash
cat train_adapter/by_task/adapter_task_*_history.json | grep task_instruction
```

## 高级用法

### 1. 任务聚类

将相似的任务归类，一起训练：

```python
# 例如：所有"marker"相关任务
marker_tasks = [
    "Remove the marker from the mug",
    "Remove the marker from the cup",
    "Put the marker in the pot",
    "Put the marker on the table",
]

# 分别微调每个任务
for task in marker_tasks:
    # 微调...
```

### 2. 任务难度分析

统计每个任务的成功率，选择难度适中的任务进行微调。

### 3. 迁移学习

先在数据多的任务上训练，再在数据少的相似任务上微调：

```bash
# 1. 在"Close the drawer"上训练（2 episodes）
python scripts/finetune_by_task.py --task "Close the drawer" --epochs 30

# 2. 使用该模型作为预训练，在相似任务上继续训练
python scripts/finetune_by_task.py \
    --task "Close the bottom drawer of the right cabinet" \
    --pretrained-path train_adapter/by_task/adapter_task_xxx_best.pth \
    --epochs 20
```

## 总结

- DROID有**758个不同任务**
- 推荐选择有**3个以上episodes**的任务
- 使用 `--list-tasks` 查看所有可用任务
- 训练参数根据数据量调整
- 注意避免过拟合

祝微调顺利！
