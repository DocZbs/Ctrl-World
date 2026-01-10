# 合成数据生成 - 多样性策略说明

## 问题：如何确保400次rollout都不同？

### 多样性来源（4种策略）

#### 1. **不同的随机种子** ✓
每次rollout使用不同的随机种子：
```python
seed = 42 + i  # i从0到399
np.random.seed(seed)
torch.manual_seed(seed)
```

**效果**: 即使初始状态和指令相同，策略的随机性也会导致不同的动作序列。

#### 2. **指令变体循环** ✓
提供多个指令变体，循环使用：
```python
instruction_variants = [
    "pick the blue block and place it in plate",
    "pick up the blue block and put it inside the plate",
    "grasp the blue cube and move it to the plate",
    "take the blue object and place it in the dish",
]
instruction = instruction_variants[i % len(instruction_variants)]
```

**效果**: 不同的指令表述可能导致策略采取不同的行为。

#### 3. **变化的状态扰动** ✓
每次rollout使用不同程度的初始状态扰动：
```python
# 扰动幅度在0.5x到1.5x之间随机变化
perturbation_scale = state_perturbation_std * (0.5 + np.random.rand())
perturbed_state[:7] += np.random.normal(0, perturbation_scale, 7)
```

**效果**: 不同的初始关节位置导致不同的轨迹。

#### 4. **夹爪状态扰动** ✓
对夹爪状态也添加小的随机扰动：
```python
perturbed_state[6] = np.clip(perturbed_state[6] + np.random.normal(0, 0.01), 0, 1)
```

**效果**: 夹爪的初始状态也会影响后续动作。

---

## 多样性验证

### 理论分析

假设：
- 4个指令变体
- 每个rollout有不同的随机种子
- 状态扰动是连续的随机变量

**组合数**:
- 指令: 4种
- 随机种子: 400种（每个都不同）
- 状态扰动: 无限种（连续分布）

**结论**: 理论上400次rollout几乎不可能完全相同。

### 实际验证方法

生成后可以检查多样性：

```python
# 检查轨迹的多样性
import json
from pathlib import Path

traj_dirs = list(Path("synthetic_data/pickplace_0002").glob("syn_*"))

# 1. 检查指令多样性
instructions = []
for traj_dir in traj_dirs:
    with open(traj_dir / "metadata.json") as f:
        meta = json.load(f)
        instructions.append(meta['task_instruction'])

print(f"Unique instructions: {len(set(instructions))}")

# 2. 检查初始状态多样性
initial_states = []
for traj_dir in traj_dirs:
    with open(traj_dir / "metadata.json") as f:
        meta = json.load(f)
        initial_states.append(tuple(meta['initial_state'][:7]))

print(f"Unique initial states: {len(set(initial_states))}")

# 3. 检查轨迹长度多样性
lengths = []
for traj_dir in traj_dirs:
    with open(traj_dir / "metadata.json") as f:
        meta = json.load(f)
        lengths.append(meta['metadata']['num_steps'])

print(f"Trajectory length range: {min(lengths)} to {max(lengths)}")
print(f"Unique lengths: {len(set(lengths))}")
```

---

## 论文中的多样性策略

根据Ctrl-World论文，他们使用了两种主要策略：

### 1. Instruction Rephrase
> "用LLM把同一任务换个说法"

**我们的实现**: ✓ 支持
- 通过 `--instruction-variants` 参数提供多个变体
- 可以手动编写或用LLM生成

### 2. Reset Init State
> "在world model中把机械臂随机移动到不同初始位姿"

**我们的实现**: ✓ 支持
- 对初始关节位置添加高斯噪声
- 扰动幅度可调（`state_perturbation_std`）

### 额外的多样性来源

我们还添加了：
- **随机种子**: 每次rollout不同的随机种子
- **变化的扰动**: 扰动幅度本身也是随机的

---

## 使用建议

### 最小配置（基础多样性）
```bash
python scripts/generate_synthetic_trajectories.py \
    --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json \
    --num-rollouts 400 \
    --output-dir synthetic_data/pickplace_0002
```

**多样性来源**: 随机种子 + 状态扰动

### 推荐配置（高多样性）
```bash
python scripts/generate_synthetic_trajectories.py \
    --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json \
    --num-rollouts 400 \
    --output-dir synthetic_data/pickplace_0002 \
    --instruction-variants \
        "pick the blue block and place it in plate" \
        "pick up the blue block and put it inside the plate" \
        "grasp the blue cube and move it to the plate" \
        "take the blue object and place it in the dish"
```

**多样性来源**: 随机种子 + 状态扰动 + 指令变体

### 最大多样性配置
```bash
python scripts/generate_synthetic_trajectories.py \
    --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json \
    --num-rollouts 400 \
    --output-dir synthetic_data/pickplace_0002 \
    --instruction-variants \
        "pick the blue block and place it in plate" \
        "pick up the blue block and put it inside the plate" \
        "grasp the blue cube and move it to the plate" \
        "take the blue object and place it in the dish" \
        "move the blue block into the plate" \
        "place the blue cube in the dish" \
        "put the blue object inside the plate" \
        "transfer the blue block to the plate"
```

**多样性来源**: 随机种子 + 状态扰动 + 8个指令变体

---

## 常见问题

### Q: 为什么需要这么多多样性？

**A**: 论文中提到，在固定初始观测+指令下，VLA行为往往很确定（老是抓同一个东西）。增加多样性可以：
1. 探索更多可能成功的轨迹
2. 提高策略的鲁棒性
3. 覆盖更多的状态空间

### Q: 400条轨迹会不会有重复？

**A**: 几乎不可能。即使指令相同，由于：
- 每次使用不同的随机种子
- 状态扰动是连续随机变量
- 策略本身可能有随机性

两条轨迹完全相同的概率极低。

### Q: 如何验证多样性？

**A**: 生成后可以：
1. 检查metadata中的`random_seed`和`perturbation_scale`
2. 比较初始状态（`initial_state`）
3. 观看视频，看轨迹是否明显不同
4. 使用上面的验证脚本

### Q: 多样性太大会不会影响质量？

**A**: 可能会。如果：
- 状态扰动太大（`state_perturbation_std > 0.1`）
- 指令变体偏离原意太多

可能导致成功率降低。建议：
- 从小的扰动开始（0.05）
- 指令变体保持语义一致

---

## 总结

✓ **我们的实现确保了400次rollout的多样性**

通过4种策略的组合：
1. 不同的随机种子（400种）
2. 指令变体循环（可配置）
3. 变化的状态扰动（连续分布）
4. 夹爪状态扰动

**理论上几乎不可能有两条完全相同的轨迹。**
