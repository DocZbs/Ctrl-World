"""EvoW experience memory for closed-loop self-evolution.

Stores tuples (l, o_0, m, r_psi, kappa) from past attempts.
Provides query methods for VLM router context, failed scene lookup,
and failed expert lookup.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class MemoryEntry:
    """Single memory entry mapping to paper notation (l, o_0, m, r_psi, kappa)."""
    task_instruction: str           # l
    task_id: str
    scenario_episode_id: str        # o_0 identifier
    scenario_instruction: str
    expert_name: str                # m
    reward: float                   # r_psi
    failure_attribution: str        # kappa: success/task_failure/scene_failure/expert_failure/wm_failure
    iteration: int = 0
    video_path: str = ""
    reasoning: str = ""


class EvoWMemory:
    """Experience memory M for EvoW closed-loop evolution.

    Stores past outcomes and provides query methods for:
    - VLM router context (recent entries for a given task/scene)
    - Scene resampling (which scenes failed for which tasks)
    - Expert rerouting (which experts failed for which tasks)
    """

    VALID_ATTRIBUTIONS = {"success", "task_failure", "scene_failure", "expert_failure", "wm_failure"}

    def __init__(self, save_path: str, max_entries: int = 10000, context_window: int = 20):
        self.save_path = Path(save_path)
        self.max_entries = max_entries
        self.context_window = context_window
        self.entries: List[MemoryEntry] = []
        self._load()

    def append(self, entry: MemoryEntry):
        """Append new experience to memory and persist."""
        if entry.failure_attribution not in self.VALID_ATTRIBUTIONS:
            raise ValueError(
                f"Invalid failure_attribution '{entry.failure_attribution}', "
                f"must be one of {self.VALID_ATTRIBUTIONS}"
            )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._save()

    def get_context_for_routing(
        self, task_instruction: str, scenario_id: str, k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get recent memory entries relevant to routing decision.

        Returns entries matching the same task instruction or scenario,
        plus the most recent general entries, up to k total.
        """
        k = k or self.context_window
        relevant = [
            e for e in self.entries
            if e.task_instruction == task_instruction
            or e.scenario_episode_id == scenario_id
        ]
        if len(relevant) >= k:
            return relevant[-k:]
        # Fill remaining slots with most recent general entries
        remaining = k - len(relevant)
        relevant_set = set(id(e) for e in relevant)
        recent = [e for e in reversed(self.entries) if id(e) not in relevant_set]
        return relevant + recent[:remaining]

    def get_failed_scenes(self, task_instruction: str) -> List[str]:
        """Get episode IDs of scenes that failed with scene_failure for this task."""
        return [
            e.scenario_episode_id for e in self.entries
            if e.failure_attribution == "scene_failure"
            and e.task_instruction == task_instruction
        ]

    def get_failed_experts(self, task_instruction: str, scenario_id: str) -> List[str]:
        """Get expert names that failed with expert_failure for this task+scene."""
        return [
            e.expert_name for e in self.entries
            if e.failure_attribution == "expert_failure"
            and e.task_instruction == task_instruction
            and e.scenario_episode_id == scenario_id
        ]

    def get_stats(self) -> Dict:
        """Return summary statistics of the memory."""
        if not self.entries:
            return {"total_entries": 0}
        attributions = {}
        for e in self.entries:
            attributions[e.failure_attribution] = attributions.get(e.failure_attribution, 0) + 1
        rewards = [e.reward for e in self.entries]
        return {
            "total_entries": len(self.entries),
            "attribution_counts": attributions,
            "avg_reward": sum(rewards) / len(rewards),
            "success_count": attributions.get("success", 0),
        }

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if self.save_path.exists():
            try:
                with open(self.save_path) as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**d) for d in data]
                print(f"Loaded {len(self.entries)} memory entries from {self.save_path}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to load memory from {self.save_path}: {e}")
                self.entries = []

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"EvoWMemory(entries={len(self.entries)}, path={self.save_path})"
