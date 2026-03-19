"""Deferred evaluator: skip online VLM evaluation during rollout.

This evaluator is intended for fast generation runs where online API calls are
undesired. It writes deterministic placeholder eval results that can be
replaced later by offline re-evaluation.
"""

from .openai_vlm_evaluator import EvalResult


class DeferredEvaluator:
    """Return deterministic placeholder results for deferred evaluation."""

    def __init__(self, config):
        self.success_weight = getattr(config, "success_weight", 1.0)
        self.consistency_weight = getattr(config, "consistency_weight", 0.5)
        print("⚠️  DeferredEvaluator initialized (online VLM evaluation disabled)")

    def evaluate(self, video_path: str, task) -> EvalResult:
        return EvalResult(
            success=False,
            physical_consistency=0.0,
            reward=0.0,
            reasoning="[DEFERRED_EVAL] Online evaluation skipped; run offline re-evaluation later.",
            raw_response={"deferred": True},
            failure_reason="expert_failure",
        )

    def get_stats(self):
        return {
            "vlm_model": "deferred (offline re-eval required)",
            "max_retries": 0,
            "timeout": 0,
        }
