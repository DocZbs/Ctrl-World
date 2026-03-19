#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evow.evaluation.openai_vlm_evaluator import OpenAIVLMEvaluator
from evow.task_generation.task_schema import Task


def _extract_traj_id(name: str) -> int | None:
    m = re.search(r"traj_(\d+)_", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _build_maps(rollout_dir: Path):
    info_dir = rollout_dir / "info"
    video_dir = rollout_dir / "video"

    if not info_dir.exists() or not video_dir.exists():
        raise FileNotFoundError(
            f"Expected subdirs 'info' and 'video' under {rollout_dir}"
        )

    info_map = {}
    for p in sorted(info_dir.glob("*.json")):
        traj_id = _extract_traj_id(p.name)
        if traj_id is None:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        instruction = str(data.get("instructions", "")).strip()
        if instruction:
            info_map[traj_id] = {
                "instruction": instruction,
                "info_path": str(p),
            }

    video_map = {}
    for p in sorted(video_dir.glob("*.mp4")):
        traj_id = _extract_traj_id(p.name)
        if traj_id is None:
            continue
        video_map[traj_id] = str(p)

    return info_map, video_map


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T rollout videos and compute success rate."
    )
    parser.add_argument(
        "--rollout-dir",
        required=True,
        help="Path containing info/ and video/ (e.g. .../Rollouts_interact_gr00t)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <rollout-dir>/vlm_eval_results.json)",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--fallback-model", default="gpt-4o-mini")
    parser.add_argument("--eval-num-frames", type=int, default=8)
    parser.add_argument(
        "--frame-crop",
        default="bottom_right",
        choices=["bottom", "bottom_right", "bottom_center", "bottom_left", "top", "full"],
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else rollout_dir / "vlm_eval_results.json"
    )

    info_map, video_map = _build_maps(rollout_dir)
    common_ids = sorted(set(info_map.keys()) & set(video_map.keys()))

    if not common_ids:
        raise RuntimeError("No matched traj_id between info/*.json and video/*.mp4")

    if args.limit is not None:
        common_ids = common_ids[: max(args.limit, 0)]

    eval_cfg = SimpleNamespace(
        api_key="${OPENAI_API_KEY}",
        vlm_model=args.model,
        vlm_fallback_model=args.fallback_model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        success_weight=1.0,
        consistency_weight=0.5,
        eval_num_frames=args.eval_num_frames,
        frame_crop=args.frame_crop,
    )
    evaluator = OpenAIVLMEvaluator(eval_cfg)

    total = 0
    success = 0
    failure_counts = {
        "wm_failure": 0,
        "task_failure": 0,
        "scene_failure": 0,
        "expert_failure": 0,
    }
    reward_sum = 0.0
    per_video = []

    for idx, traj_id in enumerate(common_ids, start=1):
        instruction = info_map[traj_id]["instruction"]
        video_path = video_map[traj_id]

        task = Task(
            instruction=instruction,
            success_criteria=[instruction],
            object_categories=[],
            difficulty=0.5,
            task_id=f"traj_{traj_id:04d}",
        )

        result = evaluator.evaluate(video_path, task)

        total += 1
        reward_sum += float(result.reward)
        if bool(result.success):
            success += 1

        fr = str(result.failure_reason)
        if fr in failure_counts:
            failure_counts[fr] += 1

        per_video.append(
            {
                "traj_id": traj_id,
                "instruction": instruction,
                "video_path": video_path,
                "success": bool(result.success),
                "reward": float(result.reward),
                "failure_reason": fr,
                "reasoning": str(result.reasoning),
            }
        )

        print(
            f"[{idx}/{len(common_ids)}] traj_{traj_id:04d}: "
            f"success={bool(result.success)} reward={float(result.reward):.3f} failure={fr}"
        )

    total_safe = max(total, 1)
    summary = {
        "rollout_dir": str(rollout_dir),
        "total": total,
        "successful": success,
        "success_rate": success / total_safe,
        "avg_reward": reward_sum / total_safe,
        "failure_counts": failure_counts,
        "eval_num_frames": int(args.eval_num_frames),
        "frame_crop": args.frame_crop,
        "model": args.model,
    }

    output = {
        "summary": summary,
        "results": per_video,
    }

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n================ Summary ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
