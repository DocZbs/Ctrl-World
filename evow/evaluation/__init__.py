"""Evaluation module and evaluator factory."""

from __future__ import annotations

QWEN_VLM_TYPES = {
    "qwen",
    "qwen_vl",
    "qwen-vl",
    "qwen3-vl",
    "qwen3_vl",
}

DEFERRED_VLM_TYPES = {"deferred", "skip", "none"}


def is_qwen_vlm_type(vlm_type: str | None) -> bool:
    return str(vlm_type or "").strip().lower() in QWEN_VLM_TYPES


def create_vlm_evaluator(config, *, allow_deferred: bool = True):
    """Create evaluator backend from config.vlm_type.

    Args:
        config: Evaluation config-like object.
        allow_deferred: Whether to return DeferredEvaluator when
            ``vlm_type`` is deferred/skip/none.
    """
    vlm_type = str(getattr(config, "vlm_type", "gpt-5") or "").strip().lower()

    if vlm_type == "dummy":
        from .dummy_evaluator import DummyEvaluator

        return DummyEvaluator(config)

    if allow_deferred and vlm_type in DEFERRED_VLM_TYPES:
        from .deferred_evaluator import DeferredEvaluator

        return DeferredEvaluator(config)

    if is_qwen_vlm_type(vlm_type):
        from .qwen_vlm_evaluator import QwenVLRewardEvaluator

        return QwenVLRewardEvaluator(config)

    from .openai_vlm_evaluator import OpenAIVLMEvaluator

    return OpenAIVLMEvaluator(config)


__all__ = [
    "QWEN_VLM_TYPES",
    "DEFERRED_VLM_TYPES",
    "is_qwen_vlm_type",
    "create_vlm_evaluator",
]
