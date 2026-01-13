#!/usr/bin/env python3
"""
统计实验结果的成功率
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_experiment_results(exp_dir):
    """分析实验目录中的所有结果"""

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"错误: 目录不存在 {exp_dir}")
        return

    results_files = list(exp_path.glob("*/results.json"))

    if not results_files:
        print(f"未找到任何 results.json 文件在 {exp_dir}")
        return

    total = 0
    success_count = 0
    failed_count = 0

    success_episodes = []
    failed_episodes = []

    success_scores = []

    for result_file in sorted(results_files):
        episode_id = result_file.parent.name

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            total += 1

            # Check statistics.successful_episodes instead of top-level success
            stats = data.get('statistics', {})
            successful_eps = stats.get('successful_episodes', 0)
            avg_reward = stats.get('avg_reward', 0.0)
            vla_failures = stats.get('vla_failures', 0)
            wm_failures = stats.get('wm_failures', 0)

            if successful_eps > 0:
                success_count += 1
                success_episodes.append((episode_id, avg_reward))
                success_scores.append(avg_reward)
            else:
                failed_count += 1
                failed_episodes.append((episode_id, vla_failures, wm_failures))

        except Exception as e:
            print(f"警告: 无法读取 {result_file}: {e}")
            continue

    print("=" * 80)
    print(f"实验结果统计: {exp_dir}")
    print("=" * 80)
    print()

    print(f"总测试数:     {total}")
    print(f"成功数:       {success_count}")
    print(f"失败数:       {failed_count}")
    print()

    if total > 0:
        success_rate = (success_count / total) * 100
        print(f"成功率:       {success_rate:.2f}%")
        print()

    if success_scores:
        avg_score = sum(success_scores) / len(success_scores)
        min_score = min(success_scores)
        max_score = max(success_scores)
        print(f"成功案例平均分数: {avg_score:.3f}")
        print(f"分数范围:         {min_score:.3f} - {max_score:.3f}")
        print()

    if success_episodes:
        print(f"成功的episodes ({len(success_episodes)}):")
        for ep_id, reward in success_episodes:
            print(f"  ✓ {ep_id} (reward={reward:.3f})")
        print()

    if failed_episodes:
        print(f"失败的episodes ({len(failed_episodes)}):")
        for ep_id, vla_fail, wm_fail in failed_episodes:
            print(f"  ✗ {ep_id} (VLA failures={vla_fail}, WM failures={wm_fail})")
        print()

    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = "/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/experiments/omni_ctrl_pi05_batch_myadapter"

    analyze_experiment_results(exp_dir)
