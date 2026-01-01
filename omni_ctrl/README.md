# Omni-Ctrl: Self-Evolving Robot Learning Framework

自我演化的机器人学习框架，扩展 Ctrl-World 实现自动化任务生成、场景检索、策略路由和 VLM 评估。

## 快速开始

### 1. 安装依赖

```bash
pip install -r omni_ctrl/requirements.txt
```

### 2. 配置路径

编辑 `experiments/run_omni_ctrl_mvp.py`，更新以下路径：

```python
# 世界模型检查点
wm_ckpt="/path/to/ctrl_world_checkpoint.pt"

# SVD 和 CLIP 模型
svd_model_path="/path/to/svd_model"
clip_model_path="/path/to/clip_model"


# 策略检查点
checkpoint="/path/to/pi05_checkpoint"

# OpenAI API 密钥
api_key="your-openai-api-key"
```

### 3. 运行实验

```bash
# 默认配置（20次迭代）
python experiments/run_omni_ctrl_mvp.py

# 自定义迭代次数
python experiments/run_omni_ctrl_mvp.py --iterations 100

# 指定输出目录
python experiments/run_omni_ctrl_mvp.py --output experiments/my_run
```

## 架构概览

```
任务生成 → DROID检索 → 策略执行(世界模型) → VLM评估 → 技能库 → 循环
```

### 核心组件

- **任务生成** (`task_generation/`) - 模板/LLM生成任务
- **场景检索** (`retrieval/`) - 基于CLIP的DROID场景检索
- **策略路由** (`policy_router/`) - 统一策略接口，支持Pi0.5/OpenVLA等
- **世界模型** (`rollout/`) - Ctrl-World包装器
- **VLM评估** (`evaluation/`) - GPT-4V评估成功率和物理一致性
- **主循环** (`core/`) - 自我演化协调器

## 目录结构

```
omni_ctrl/
├── configs/           # 配置系统
├── task_generation/   # 任务生成（15个模板）
├── retrieval/         # DROID场景检索（FAISS索引）
├── policy_router/     # 策略管理
├── rollout/           # 世界模型执行
├── evaluation/        # VLM评估
├── core/              # 核心协调器
└── utils/             # 工具函数
```

## 配置说明

### 使用YAML配置

创建 `my_config.yaml`:

```yaml
experiment_name: "my_experiment"
num_iterations: 100
device: "cuda:0"

task_gen:
  type: "template"
  templates_path: "omni_ctrl/task_generation/templates.json"

retrieval:
  droid_path: "dataset_example/droid_subset"
  clip_model_path: "openai/clip-vit-base-patch32"

evaluation:
  vlm_type: "gpt4v"
  api_key: "${OPENAI_API_KEY}"

router:
  available_policies:
    - name: "pi05"
      checkpoint: "/path/to/pi05"
      action_space: "joint_vel"
```

运行：
```bash
export OPENAI_API_KEY="your-key"
python experiments/run_omni_ctrl_mvp.py --config my_config.yaml
```

### Python API

```python
from omni_ctrl.configs import OmniCtrlConfig
from omni_ctrl.core import OmniCtrlOrchestrator

# 创建配置
config = OmniCtrlConfig(
    experiment_name="test",
    num_iterations=10,
    device="cuda:0"
)

# 更新配置
config.evaluation.api_key = "your-api-key"
config.rollout.wm_ckpt = "/path/to/checkpoint.pt"

# 运行
orchestrator = OmniCtrlOrchestrator(config)
orchestrator.run()
```

## 结果输出

结果保存在 `experiments/omni_ctrl_mvp/`:

```
experiments/omni_ctrl_mvp/
├── videos/              # 生成的视频
├── episodes/            # 轨迹元数据
├── skills/
│   └── index.json      # 发现的技能目录
└── results.json        # 统计数据
```

## 开发路线

### ✅ Phase 1: MVP (已完成)
- [x] 模板任务生成（15个任务）
- [x] DROID场景检索
- [x] Pi0.5策略集成
- [x] GPT-4V评估
- [x] 技能库管理

### 🔄 Phase 2: 多策略路由
- [ ] OpenVLA策略包装
- [ ] Diffusion Policy包装
- [ ] SPO自我对弈优化
- [ ] UCB策略选择

### 📋 Phase 3: LLM任务生成
- [ ] GPT-4任务生成器
- [ ] 任务多样性跟踪
- [ ] 基于DROID的任务锚定

## 常见问题

### FAISS索引问题

重建索引：
```bash
python experiments/run_omni_ctrl_mvp.py --rebuild-index
```

### 内存不足

减少步数：
```python
config.rollout.max_steps = 30  # 默认50
```

### API错误

检查API密钥：
```bash
export OPENAI_API_KEY="sk-..."
python -c "import openai; print(openai.OpenAI().models.list())"
```

## 引用

如果使用本框架，请引用：

```bibtex
@article{omnictrl2025,
  title={Omni-Ctrl: Scaling Robot Intelligence via Self-Evolving Generative World Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 基于

- Ctrl-World (生成式世界模型)
- DROID (机器人操作数据集)
- OpenPI (Pi0 系列VLA模型)
- OpenAI GPT-4V (评估)
