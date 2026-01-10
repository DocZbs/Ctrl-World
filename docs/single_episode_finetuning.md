# DROID数据格式与单场景微调指南

## DROID数据集格式

### 目录结构
```
droid_data/
├── data/
│   ├── chunk-000/          # Episodes 0-999
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/          # Episodes 1000-1999
│   └── ...
├── latents/                # 预提取的latent特征
│   └── chunk-000/
│       ├── episode_000000/
│       │   ├── 0.pt
│       │   ├── 1.pt
│       │   └── 2.pt
│       └── ...
└── videos/                 # 原始视频
    └── chunk-000/
        └── episode_000000/
            ├── exterior_image_1_left.mp4
            ├── exterior_image_2_left.mp4
            └── wrist_image_left.mp4
```

### Parquet文件格式

每个episode是一个parquet文件，包含以下关键字段：

#### 观测数据 (Observation)
- `observation.state.joint_position`: 关节位置 (7维)
- `observation.state.cartesian_position`: 笛卡尔位置 (6维: x,y,z,roll,pitch,yaw)
- `observation.state.gripper_position`: 夹爪位置

#### 动作数据 (Action)
- `action.joint_velocity`: 关节速度 (7维) - **这是训练的目标**
- `action.joint_position`: 关节位置
- `action.cartesian_velocity`: 笛卡尔速度
- `action.gripper_position`: 夹爪动作

#### 任务信息
- `language_instruction`: 任务描述文本
- `task_category`: 任务类别
- `is_episode_successful`: 是否成功完成

#### 其他
- `frame_index`: 帧索引
- `timestamp`: 时间戳
- `reward`: 奖励值

### 数据示例

```python
import pandas as pd

df = pd.read_parquet("droid_data/data/chunk-000/episode_000000.parquet")

# 查看第一帧
frame_0 = df.iloc[0]

# 关节位置 (7维)
joint_pos = frame_0['observation.state.joint_position']
# [-0.22476004, -0.42106023, -0.12811285, -2.3547568, -0.19623408, 2.2180023, 0.02638818]

# 关节速度 (7维) - 这是action adapter要预测的
joint_vel = frame_0['action.joint_velocity']
# [0.02907034, 0.00067504, -0.0262411, 0.01411975, -0.01116389, -0.01882628, 0.00623559]

# 任务描述
instruction = frame_0['language_instruction']
# "pick up the blue block"
```

---

## 单场景微调

### 为什么要单场景微调？

1. **快速适应**: 针对特定场景快速优化模型
2. **调试测试**: 验证模型在特定场景的表现
3. **场景特化**: 为特定任务创建专用模型
4. **数据效率**: 不需要大量数据就能看到效果

### 使用方法

#### 1. 基本用法

在episode 0上微调：
```bash
python scripts/finetune_single_episode.py --episode-id 0
```

在episode 123上微调：
```bash
python scripts/finetune_single_episode.py --episode-id 123
```

#### 2. 自定义参数

```bash
python scripts/finetune_single_episode.py \
    --episode-id 0 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --output-dir train_adapter/single_episode
```

#### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episode-id` | (必需) | Episode ID (0-999 for chunk-000) |
| `--data-root` | `droid_data` | DROID数据根目录 |
| `--pretrained-path` | `models/action_adapter/model2_15_9.pth` | 预训练模型路径 |
| `--output-dir` | `train_adapter/single_episode` | 输出目录 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 32 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--num-frames` | 15 | 预测帧数 |
| `--device` | cuda | 训练设备 |

### 输出文件

训练完成后会生成：

1. **模型文件**: `train_adapter/single_episode/adapter_episode_000000.pth`
2. **训练历史**: `train_adapter/single_episode/episode_000000_history.json`

训练历史包含：
```json
{
  "episode_id": 0,
  "num_epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "best_loss": 0.001234,
  "final_loss": 0.001456,
  "loss_history": [0.01, 0.008, 0.006, ...]
}
```

### 使用微调后的模型

在配置文件中指定微调后的模型：

```yaml
# omni_ctrl/configs/your_config.yaml
rollout:
  action_adapter_path: "train_adapter/single_episode/adapter_episode_000000.pth"
```

---

## 完整工作流示例

### 场景1: 快速测试单个episode

```bash
# 1. 在episode 0上微调
python scripts/finetune_single_episode.py --episode-id 0 --epochs 50

# 2. 测试微调后的模型
# 修改配置文件使用新模型
# 然后运行测试
```

### 场景2: 批量微调多个episodes

```bash
# 创建批量微调脚本
for ep_id in 0 5 10 15 20; do
    echo "Training on episode $ep_id..."
    python scripts/finetune_single_episode.py \
        --episode-id $ep_id \
        --epochs 50 \
        --output-dir train_adapter/multi_episode
done
```

### 场景3: 对比不同episode的微调效果

```bash
# 1. 微调多个episodes
python scripts/finetune_single_episode.py --episode-id 0 --epochs 50
python scripts/finetune_single_episode.py --episode-id 100 --epochs 50
python scripts/finetune_single_episode.py --episode-id 500 --epochs 50

# 2. 对比loss
cat train_adapter/single_episode/episode_*_history.json | grep best_loss
```

---

## 训练技巧

### 1. 学习率选择

- **快速收敛**: `lr=1e-3` (可能不稳定)
- **稳定训练**: `lr=1e-4` (推荐)
- **精细调整**: `lr=1e-5` (收敛慢)

### 2. Epoch数量

- **短episode (<100帧)**: 50-100 epochs
- **中等episode (100-300帧)**: 30-50 epochs
- **长episode (>300帧)**: 20-30 epochs

### 3. 批大小

- **小数据集**: batch_size=16-32
- **中等数据集**: batch_size=32-64
- **大数据集**: batch_size=64-128

### 4. 过拟合检测

单场景微调容易过拟合，注意：
- Loss降到很低但在其他场景表现差 → 过拟合
- 建议在相似场景上测试泛化能力

---

## 常见问题

### Q1: Episode文件不存在？
```
FileNotFoundError: Episode file not found
```

**解决**: 检查episode ID是否正确，chunk-000包含episodes 0-999

### Q2: 内存不足？
```
RuntimeError: CUDA out of memory
```

**解决**: 降低batch_size
```bash
python scripts/finetune_single_episode.py --episode-id 0 --batch-size 16
```

### Q3: 训练太慢？
**解决**:
- 减少epochs: `--epochs 20`
- 增加batch_size: `--batch-size 64`
- 使用更快的GPU

### Q4: Loss不下降？
**解决**:
- 提高学习率: `--lr 1e-3`
- 检查数据是否正确加载
- 确认预训练模型路径正确

---

## 参考

- 原始训练脚本: `train_adapter/finetune_adapter.py`
- Dynamics模型定义: `models/action_adapter/train2.py`
- 预训练模型: `models/action_adapter/model2_15_9.pth`
- DROID数据集: https://droid-dataset.github.io/
