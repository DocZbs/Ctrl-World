"""Smoke test for OpenAI VLM evaluator."""

import argparse
import os

from omni_ctrl.configs.base_config import EvaluationConfig
from omni_ctrl.evaluation.openai_vlm_evaluator import OpenAIVLMEvaluator
from omni_ctrl.task_generation.task_schema import Task


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenAI VLM evaluation on a video.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--instruction", required=True, help="Task instruction")
    parser.add_argument("--success", nargs="*", default=None, help="Success criteria list")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model name")
    parser.add_argument("--fallback", default="gpt-4o-mini", help="Fallback model name")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    cfg = EvaluationConfig(
        vlm_type=args.model,
        vlm_model=args.model,
        vlm_fallback_model=args.fallback,
        api_key=api_key,
        max_retries=3,
        timeout=60,
        success_weight=1.0,
        consistency_weight=0.5,
    )

    evaluator = OpenAIVLMEvaluator(cfg)
    task = Task(
        instruction=args.instruction,
        success_criteria=args.success or [args.instruction],
        object_categories=[],
        difficulty=0.3,
        task_id="test_task",
    )
    result = evaluator.evaluate(args.video, task)
    print(result)


if __name__ == "__main__":
    main()
