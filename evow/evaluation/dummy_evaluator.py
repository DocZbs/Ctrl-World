"""Dummy evaluator for testing without API access."""

import random
from .openai_vlm_evaluator import EvalResult


class DummyEvaluator:
    """Dummy evaluator that returns random results for testing.

    Use this when you don't have API access or want to test the pipeline.
    """

    def __init__(self, config):
        """Initialize dummy evaluator.

        Args:
            config: EvaluationConfig instance (unused)
        """
        print("⚠️  DummyEvaluator initialized (for testing only)")
        self.success_weight = config.success_weight
        self.consistency_weight = config.consistency_weight

    def evaluate(self, video_path: str, task) -> EvalResult:
        """Return random evaluation result.

        Args:
            video_path: Path to episode video file
            task: Task object

        Returns:
            EvalResult with random values
        """
        # Random success (60% success rate for testing)
        success = random.random() > 0.4
        consistency = random.uniform(0.6, 1.0) if success else random.uniform(0.3, 0.7)

        if success:
            reward = self.success_weight * consistency
            reasoning = f"[DUMMY] Task '{task.instruction}' appears to be completed successfully."
            failure_reason = "success"
        else:
            reward = 0.0
            # Randomly assign failure to WM or VLA (50/50)
            failure_reason = "wm_failure" if random.random() > 0.5 else "vla_failure"
            if failure_reason == "wm_failure":
                reasoning = f"[DUMMY] Task '{task.instruction}' failed due to world model issues."
            else:
                reasoning = f"[DUMMY] Task '{task.instruction}' failed due to policy issues."

        return EvalResult(
            success=success,
            physical_consistency=consistency,
            reward=reward,
            reasoning=reasoning,
            raw_response={"dummy": True},
            failure_reason=failure_reason,
        )

    def get_stats(self):
        """Get evaluator statistics."""
        return {
            "vlm_model": "dummy (for testing)",
            "max_retries": 0,
            "timeout": 0,
        }
