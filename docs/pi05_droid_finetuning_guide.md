# π0.5 DROID 微调指南

## 概述

π0.5是Physical Intelligence开发的视觉-语言-动作(VLA)模型，在DROID数据集上训练。本指南介绍如何在DROID数据集的子集或特定任务上微调π0.5模型。

## 两种微调方式

### 方式1: 在自定义DROID子集上微调（推荐用于单任务）

适用于：
- 小规模数据集（<10小时数据）
- 特定任务的微调
- 快速实验和原型开发

### 方式2: 在完整DROID数据集上微调

适用于：
- 大规模数据集（>10小时数据）
- 多任务训练
- 需要最佳性能

---

## 方式1: 自定义DROID子集微调

### 前置条件

1. 已安装openpi环境
2. 有DROID格式的数据或已转换为LeRobot格式

### Step 1: 准备数据

#### 选项A: 使用现有DROID数据

如果你已经有DROID parquet格式的数据（如`droid_data/data/chunk-000/`），需要转换为LeRobot格式：

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi

# 转换DROID数据到LeRobot格式
uv run examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir /path/to/your/droid/episodes \
    --output_dir /path/to/lerobot/dataset
```

#### 选项B: 按任务筛选数据

如果你想只在特定任务上微调，可以先筛选出该任务的episodes：

```python
# scripts/filter_task_episodes.py
import pandas as pd
from pathlib import Path
import shutil

task = "Close the drawer"
droid_root = Path("/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data")
output_dir = Path("/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_task_subset")
output_dir.mkdir(exist_ok=True)

# 找到所有该任务的episodes
for chunk_id in [0]:  # 可以添加更多chunks
    chunk_dir = droid_root / "data" / f"chunk-{chunk_id:03d}"

    for ep_file in chunk_dir.glob("episode_*.parquet"):
        df = pd.read_parquet(ep_file)
        if len(df) > 0:
            instruction = df.iloc[0].get('language_instruction', '')
            if instruction == task:
                # 复制episode文件
                shutil.copy(ep_file, output_dir / ep_file.name)
                print(f"Copied {ep_file.name}")
```

### Step 2: 转换为LeRobot格式

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi

# 转换筛选后的数据
uv run examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_task_subset \
    --output_dir /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/lerobot_task_dataset
```

### Step 3: 修改训练配置

编辑 `openpi/src/openpi/training/config.py`，找到 `pi05_droid_finetune` 配置：

```python
TrainConfig(
    name="pi05_droid_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,  # π0.5使用32维动作
        action_horizon=16,
    ),
    data=LeRobotDROIDDataConfig(
        # 修改为你的数据集路径
        repo_id="local://path/to/lerobot_task_dataset",  # 或上传到HuggingFace
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
    num_train_steps=20_000,  # 根据数据量调整
    batch_size=32,
)
```

### Step 4: 运行微调

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi

# 启动训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_droid_finetune \
    --exp-name=my_task_finetune \
    --overwrite
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_steps` | 20,000 | 训练步数 |
| `batch_size` | 32 | 批大小 |
| `--exp-name` | - | 实验名称 |
| `--overwrite` | - | 覆盖已存在的checkpoint |

---

## 方式2: 完整DROID数据集微调

### 前置条件

1. 1.8TB磁盘空间
2. 8x H100 GPU（或等效算力）
3. 安装RLDS依赖

### Step 1: 安装RLDS依赖

```bash
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi
uv sync --group rlds
```

### Step 2: 下载DROID数据集

```bash
# 需要先安装gsutil
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 /path/to/droid/1.0.1
```

### Step 3: 修改配置

编辑 `openpi/src/openpi/training/config.py`，修改 `rlds_data_dir`:

```python
rlds_data_dir: str | None = "/path/to/droid/1.0.1"
```

### Step 4: 计算归一化统计量

```bash
uv run --group rlds scripts/compute_norm_stats.py \
    --config-name pi05_full_droid_finetune \
    --max-frames 10_000_000
```

### Step 5: 运行训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py \
    pi05_full_droid_finetune \
    --exp-name=my_full_droid_finetune \
    --overwrite
```

### 计算需求

- **时间**: 约2天（8x H100 GPU）
- **迭代**: 100k iterations
- **Batch size**: 256
- **数据**: 约1 epoch

---

## 按任务微调的完整工作流

### 示例：微调"Close the drawer"任务

```bash
#!/bin/bash
# finetune_close_drawer.sh

TASK="Close the drawer"
DROID_ROOT="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_data"
TASK_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/droid_close_drawer"
LEROBOT_DIR="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/lerobot_close_drawer"

# 1. 筛选任务episodes
echo "Step 1: Filtering episodes for task: $TASK"
python3 << EOF
import pandas as pd
from pathlib import Path
import shutil

task = "$TASK"
droid_root = Path("$DROID_ROOT")
output_dir = Path("$TASK_DIR")
output_dir.mkdir(exist_ok=True)

count = 0
for chunk_id in [0]:
    chunk_dir = droid_root / "data" / f"chunk-{chunk_id:03d}"
    for ep_file in chunk_dir.glob("episode_*.parquet"):
        df = pd.read_parquet(ep_file)
        if len(df) > 0:
            instruction = df.iloc[0].get('language_instruction', '')
            if instruction == task:
                shutil.copy(ep_file, output_dir / ep_file.name)
                count += 1
                print(f"Copied {ep_file.name}")

print(f"Total episodes: {count}")
EOF

# 2. 转换为LeRobot格式
echo "Step 2: Converting to LeRobot format"
cd /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi
uv run examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir $TASK_DIR \
    --output_dir $LEROBOT_DIR

# 3. 修改配置（手动或通过脚本）
echo "Step 3: Please update config.py with:"
echo "  repo_id='local://$LEROBOT_DIR'"

# 4. 运行训练
echo "Step 4: Starting training"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_droid_finetune \
    --exp-name=close_drawer_finetune \
    --overwrite
```

---

## 训练监控

### 使用WandB

训练会自动记录到WandB：

```python
# 在config.py中
wandb_enabled=True
project_name="openpi"
```

### 查看checkpoint

```bash
ls /mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/openpi/checkpoints/close_drawer_finetune/
```

---

## 使用微调后的模型

### 方法1: 在Ctrl-World中使用

修改 `omni_ctrl/configs/omni_ctrl_pi05_batch.yaml`:

```yaml
router:
  available_policies:
    - name: "pi05"
      checkpoint: "/path/to/your/finetuned/checkpoint"
      action_space: "joint_vel"
```

### 方法2: 直接使用

```python
from openpi_client import Pi05Policy

policy = Pi05Policy(
    checkpoint_path="/path/to/your/finetuned/checkpoint",
    device="cuda:0"
)

# 使用policy进行推理
action = policy.predict(observation, instruction)
```

---

## 常见问题

### Q1: 数据量太少怎么办？

**建议**:
- 至少需要3-5个episodes
- 使用较低的学习率
- 减少训练步数（5k-10k steps）
- 考虑数据增强

### Q2: 内存不足？

**解决**:
```bash
# 减小batch size
batch_size=16

# 限制JAX内存使用
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
```

### Q3: 如何选择训练步数？

| 数据量 | 训练步数 | Batch Size |
|--------|---------|------------|
| 2-5 episodes | 5k-10k | 16-32 |
| 5-20 episodes | 10k-20k | 32-64 |
| 20+ episodes | 20k-50k | 64-128 |

### Q4: 训练太慢？

**优化**:
- 使用更大的batch size
- 使用多GPU训练
- 减少数据增强
- 使用LoRA（虽然效果可能不如全量微调）

---

## 高级技巧

### 1. 多任务联合微调

将多个相关任务的数据合并：

```python
tasks = [
    "Close the drawer",
    "Close the bottom drawer of the right cabinet",
    "Close the right cabinet door"
]

# 合并所有任务的episodes
for task in tasks:
    # 筛选并复制episodes
    ...
```

### 2. 渐进式微调

先在数据多的任务上训练，再在数据少的任务上微调：

```bash
# 1. 在任务A上训练（10 episodes）
uv run scripts/train.py pi05_droid_finetune --exp-name=task_a

# 2. 使用task_a的checkpoint继续在任务B上训练（2 episodes）
# 修改weight_loader指向task_a的checkpoint
uv run scripts/train.py pi05_droid_finetune --exp-name=task_b
```

### 3. 评估微调效果

```bash
# 在Ctrl-World中测试
python scripts/run_all_droid_new_setup.py \
    --config omni_ctrl/configs/omni_ctrl_pi05_batch.yaml \
    --ann-dir droid_data/validation/annotation/val
```

---

## 参考资料

- [OpenPI官方文档](https://github.com/Physical-Intelligence/openpi)
- [DROID数据集](https://droid-dataset.github.io/)
- [LeRobot](https://github.com/huggingface/lerobot)
- [训练脚本](openpi/scripts/train.py)
- [配置文件](openpi/src/openpi/training/config.py)

---

## 总结

- **小数据集**: 使用LeRobot格式 + `pi05_droid_finetune` 配置
- **大数据集**: 使用RLDS格式 + `pi05_full_droid_finetune` 配置
- **单任务**: 先筛选episodes，再转换为LeRobot格式
- **训练时间**: 小数据集几小时，大数据集2天（8x H100）
- **最小数据量**: 建议至少3-5个episodes

祝微调顺利！
