#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _task_sort_key(p: Path):
    stem = p.stem
    if stem.startswith("task_"):
        num = stem.split("_", 1)[1]
        if num.isdigit():
            return int(num)
    return stem


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{ts}")
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def _memory_stats(entries: list[dict]) -> dict:
    total = len(entries)
    if total == 0:
        return {"total_entries": 0}
    attribution_counts = {}
    rewards = []
    for e in entries:
        k = e.get("failure_attribution", "unknown")
        attribution_counts[k] = attribution_counts.get(k, 0) + 1
        rewards.append(float(e.get("reward", 0.0)))
    return {
        "total_entries": total,
        "attribution_counts": attribution_counts,
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "success_count": attribution_counts.get("success", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate existing EvoW videos and rewrite memory/results.")
    parser.add_argument("--exp-dir", required=True, help="Experiment directory, e.g. experiments/xxx")
    parser.add_argument("--config", default=None, help="Config YAML path (default: <exp-dir>/config.yaml)")
    parser.add_argument("--timeout", type=int, default=60, help="Override evaluation timeout seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Override evaluation API retries")
    parser.add_argument("--vlm-type", default=None, help="Override evaluator type (e.g. gpt-5, qwen-vl)")
    parser.add_argument("--vlm-model", default=None, help="Override VLM model id/path")
    parser.add_argument("--eval-num-frames", type=int, default=None, help="Override number of sampled frames")
    parser.add_argument(
        "--frame-crop",
        default=None,
        choices=["bottom", "bottom_right", "bottom_center", "bottom_left", "top", "full"],
        help="Override frame crop policy",
    )
    parser.add_argument("--qwen-device", default=None, help="Override local Qwen device, e.g. cuda:1")
    parser.add_argument("--qwen-torch-dtype", default=None, help="Override local Qwen dtype: auto/bf16/fp16/fp32")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=None, help="Override local Qwen decode length")
    parser.add_argument("--qwen-temperature", type=float, default=None, help="Override local Qwen temperature")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but do not write files")
    parser.add_argument("--no-results", action="store_true", help="Do not rewrite results.json / verified_set.json")
    args = parser.parse_args()

    from evow.evaluation import create_vlm_evaluator
    from evow.task_generation.task_schema import Task

    project_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    config_path = Path(args.config).resolve() if args.config else (exp_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    episodes_dir = exp_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

    cfg = _load_yaml(config_path)
    eval_cfg_raw = cfg.get("evaluation", {}) or {}

    eval_cfg = SimpleNamespace(
        vlm_type=args.vlm_type if args.vlm_type is not None else eval_cfg_raw.get("vlm_type", "gpt-5"),
        api_key=eval_cfg_raw.get("api_key", "${OPENAI_API_KEY}"),
        vlm_model=args.vlm_model if args.vlm_model is not None else eval_cfg_raw.get("vlm_model", "gpt-5.2"),
        vlm_fallback_model=eval_cfg_raw.get("vlm_fallback_model", "gpt-4o-mini"),
        max_retries=args.max_retries,
        timeout=args.timeout,
        success_weight=float(eval_cfg_raw.get("success_weight", 1.0)),
        consistency_weight=float(eval_cfg_raw.get("consistency_weight", 0.5)),
        eval_num_frames=(
            args.eval_num_frames
            if args.eval_num_frames is not None
            else eval_cfg_raw.get("eval_num_frames", 8)
        ),
        frame_crop=args.frame_crop if args.frame_crop is not None else eval_cfg_raw.get("frame_crop", "bottom_right"),
        qwen_device=args.qwen_device if args.qwen_device is not None else eval_cfg_raw.get("qwen_device"),
        qwen_torch_dtype=(
            args.qwen_torch_dtype
            if args.qwen_torch_dtype is not None
            else eval_cfg_raw.get("qwen_torch_dtype")
        ),
        qwen_max_new_tokens=(
            args.qwen_max_new_tokens
            if args.qwen_max_new_tokens is not None
            else eval_cfg_raw.get("qwen_max_new_tokens")
        ),
        qwen_temperature=(
            args.qwen_temperature
            if args.qwen_temperature is not None
            else eval_cfg_raw.get("qwen_temperature")
        ),
        qwen_trust_remote_code=eval_cfg_raw.get("qwen_trust_remote_code", True),
    )

    reward_threshold = float(cfg.get("reward_threshold", 0.3))

    old_memory_path = exp_dir / "memory.json"
    old_memory_by_task = {}
    if old_memory_path.exists():
        try:
            old_memory = json.loads(old_memory_path.read_text(encoding="utf-8"))
            if isinstance(old_memory, list):
                old_memory_by_task = {str(e.get("task_id", "")): e for e in old_memory}
        except Exception:
            pass

    evaluator = create_vlm_evaluator(eval_cfg, allow_deferred=False)
    print(f"Using evaluator: {evaluator.__class__.__name__} ({eval_cfg.vlm_model})")

    episode_files = sorted(episodes_dir.glob("task_*.json"), key=_task_sort_key)
    if not episode_files:
        raise RuntimeError(f"No task_*.json found in {episodes_dir}")

    new_memory = []
    verified_set = []

    wm_failures = 0
    task_failures = 0
    scene_failures = 0
    expert_failures = 0
    successful = 0
    total_reward = 0.0

    for idx, ep_path in enumerate(episode_files):
        ep = json.loads(ep_path.read_text(encoding="utf-8"))
        task_id = str(ep.get("task_id", ep_path.stem))
        instruction = str(ep.get("task_instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Empty task_instruction in {ep_path}")

        task = Task(
            instruction=instruction,
            success_criteria=[instruction],
            object_categories=[],
            difficulty=0.5,
            task_id=task_id,
        )

        video_path_raw = ep.get("video_path", "")
        video_path = Path(video_path_raw)
        if not video_path.is_absolute():
            video_path = (project_root / video_path_raw).resolve()

        result = evaluator.evaluate(str(video_path), task)

        ep["eval_result"] = {
            "success": bool(result.success),
            "failure_reason": str(result.failure_reason),
            "reward": float(result.reward),
            "reasoning": str(result.reasoning),
            "attempts": 1,
        }

        meta = ep.get("metadata", {}) or {}
        old_entry = old_memory_by_task.get(task_id, {})

        scenario_id = str(meta.get("scenario_id", old_entry.get("scenario_episode_id", "")))
        policy_name = str(meta.get("policy_name", old_entry.get("expert_name", "pi05")))
        scenario_instruction = str(old_entry.get("scenario_instruction", ""))

        mem_entry = {
            "task_instruction": instruction,
            "task_id": task_id,
            "scenario_episode_id": scenario_id,
            "scenario_instruction": scenario_instruction,
            "expert_name": policy_name,
            "reward": float(result.reward),
            "failure_attribution": str(result.failure_reason),
            "iteration": idx,
            "video_path": str(ep.get("video_path", "")),
            "reasoning": str(result.reasoning)[:200],
        }
        new_memory.append(mem_entry)

        total_reward += float(result.reward)
        if result.reward >= reward_threshold:
            successful += 1
            verified_set.append(
                {
                    "task": asdict(task),
                    "scenario_id": scenario_id,
                    "expert": policy_name,
                    "reward": float(result.reward),
                    "video_path": str(ep.get("video_path", "")),
                }
            )

        fr = str(result.failure_reason)
        if fr == "wm_failure":
            wm_failures += 1
        elif fr == "task_failure":
            task_failures += 1
        elif fr == "scene_failure":
            scene_failures += 1
        elif fr == "expert_failure":
            expert_failures += 1

        print(f"[{idx+1}/{len(episode_files)}] {task_id}: reward={result.reward:.3f} failure={fr}")

        if not args.dry_run:
            ep_path.write_text(json.dumps(ep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    # backup + rewrite memory
    if old_memory_path.exists():
        bkp = _backup(old_memory_path)
        print(f"Backed up memory: {bkp}")
    old_memory_path.write_text(json.dumps(new_memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rewrote memory: {old_memory_path}")

    if not args.no_results:
        total = len(episode_files)
        total_safe = max(total, 1)
        results_path = exp_dir / "results.json"
        verified_path = exp_dir / "verified_set.json"

        existing_results = {}
        if results_path.exists():
            try:
                existing_results = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                existing_results = {}

        statistics = {
            "total_episodes": total,
            "successful_episodes": successful,
            "wm_failures": wm_failures,
            "task_failures": task_failures,
            "scene_failures": scene_failures,
            "expert_failures": expert_failures,
            "retries_total": 0,
            "scene_resamples": 0,
            "expert_reroutes": 0,
            "success_rate": successful / total_safe,
            "wm_failure_rate": wm_failures / total_safe,
            "task_failure_rate": task_failures / total_safe,
            "scene_failure_rate": scene_failures / total_safe,
            "expert_failure_rate": expert_failures / total_safe,
            "avg_reward": total_reward / total_safe,
        }

        existing_results.setdefault("config", {
            "experiment_name": cfg.get("experiment_name", exp_dir.name),
            "num_iterations": cfg.get("num_iterations", total),
            "seed": cfg.get("seed", 42),
        })
        existing_results["statistics"] = statistics
        existing_results["memory"] = _memory_stats(new_memory)

        if results_path.exists():
            bkp = _backup(results_path)
            print(f"Backed up results: {bkp}")
        results_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if verified_path.exists():
            bkp = _backup(verified_path)
            print(f"Backed up verified_set: {bkp}")
        verified_path.write_text(json.dumps(verified_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Rewrote results: {results_path}")
        print(f"Rewrote verified_set: {verified_path}")


if __name__ == "__main__":
    main()
