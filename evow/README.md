# EvoW: Open-ended Task-to-Policy Self-Evolution

`evow/` 是 Ctrl-World 仓库中承载 `EvoW` 的子模块，负责开放式任务生成、场景检索、策略路由、世界模型 rollout 与 VLM 评估的闭环编排。

虽然实现代码仍位于 `evow/` 目录下，但该子系统对外统一称为 `EvoW`；运行时仍会复用仓库根目录下的 `models/`、`config.py` 与 `openpi/`。

## Quick Start

1) 安装依赖：

```bash
pip install -r evow/requirements.txt
```

2) 根据你的本地环境，先补齐模型与数据路径。最少需要检查：

- `router.available_policies[*].checkpoint`
- `rollout.wm_ckpt`
- `rollout.svd_model_path`
- `rollout.clip_model_path`
- `evaluation.api_key` 或对应环境变量

3) 使用默认入口运行：

```bash
python -m evow
```

也兼容直接运行脚本：

```bash
python evow/run_evow.py
```

4) 或使用 YAML 配置：

```bash
python evow/run_evow.py --config evow/configs/default.yaml
```

如果希望使用本地、无需 API 的评估方式，可将：

- `evaluation.vlm_type` 设为 `qwen-vl`
- `evaluation.vlm_model` 设为本地或 HuggingFace 的 Qwen-VL 模型路径

若使用 `Qwen3-VL-*`，建议确保 `transformers>=4.57`。

## 目录结构

```text
evow/
├── __main__.py               # `python -m evow` 入口
├── run_evow.py              # CLI 入口与配置校验
├── configs/                  # dataclass 配置定义与 YAML 示例
├── core/                     # 主编排循环、episode、memory、skill library
├── evaluation/               # GPT/Qwen/dummy/deferred 评估器
├── policy_router/            # Pi 系列策略封装与专家路由
├── retrieval/                # DROID 场景索引与检索
├── rollout/                  # Ctrl-World 世界模型封装
├── task_generation/          # 模板/LLM 任务生成
└── utils/                    # 运行时路径与环境初始化工具
```

## 运行流程

`run_evow.py` 的职责主要分成四步：

1. 解析 CLI 参数并加载 `EvoWConfig`（EvoW 配置对象）
2. 校验模型路径、API key 和路由设置
3. 处理输出目录与 GPU 可见设备映射
4. 构造 `evow.core.EvoWOrchestrator`（EvoW 主循环）并启动闭环迭代

主循环位于 `evow/core/evow_orchestrator.py`，串联以下模块：

- 任务生成：`evow.task_generation`
- 场景检索：`evow.retrieval`
- 策略路由：`evow.policy_router`
- 世界模型 rollout：`evow.rollout`
- VLM 评估：`evow.evaluation`
- 经验记忆：`evow.core.evow_memory`

## 配置建议

- 优先从 `evow/configs/default.yaml` 或你自己的实验 YAML 启动，不建议直接修改 `create_default_config()` 来长期维护实验。
- `run_evow.py` 会自动整理输出目录，并将 memory/skills 固定写到 `output_dir` 下。
- 若未显式设置 `CUDA_VISIBLE_DEVICES`，入口会根据策略与 rollout 配置自动裁剪可见 GPU。
- `evow/utils/runtime.py` 统一处理 Ctrl-World 根目录、`openpi/` 和 `openpi-client` 的导入路径初始化。

## Key Components

- Task generation: `evow.task_generation`
- Retrieval: `evow.retrieval`
- Policy routing: `evow.policy_router`（当前以 Pi 系列为主）
- World model rollout: `evow.rollout`
- VLM evaluation: `evow.evaluation`
- Core loop: `evow.core.EvoWOrchestrator`（EvoW 主循环）
