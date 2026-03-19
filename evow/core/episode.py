"""Episode dataclass for storing rollout trajectories."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import json
from pathlib import Path


@dataclass
class Episode:
    """Rollout episode with observations, actions, and evaluation.

    Stores the complete trajectory of a single task execution,
    including observations, actions, and evaluation results.

    Attributes:
        task_id: Unique identifier for the task
        task_instruction: Natural language task description
        observations: List of observation dictionaries at each step
        actions: List of action arrays taken
        video_path: Path to saved episode video
        eval_result: Evaluation result dictionary
        metadata: Additional metadata (policy name, scenario, etc.)
    """

    task_id: str
    task_instruction: str
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    video_path: str = ""
    eval_result: Dict = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, obs: Dict[str, Any], action: np.ndarray):
        """Record one step of the episode.

        Args:
            obs: Observation dictionary
            action: Action array
        """
        self.observations.append(obs)
        self.actions.append(action)

    def get_length(self) -> int:
        """Get episode length in steps.

        Returns:
            Number of steps in episode
        """
        return len(self.actions)

    def save(self, save_dir: str):
        """Save episode metadata to disk.

        Note: Large arrays (observations, actions) are not saved to reduce
        storage. Only metadata and video path are saved.

        Args:
            save_dir: Directory to save episode info
        """
        save_path = Path(save_dir) / f"{self.task_id}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "num_steps": len(self.actions),
            "video_path": self.video_path,
            "eval_result": self.eval_result,
            "metadata": self.metadata,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary (excluding large arrays).

        Returns:
            Dictionary with episode metadata
        """
        return {
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "num_steps": self.get_length(),
            "video_path": self.video_path,
            "eval_result": self.eval_result,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, load_path: str) -> "Episode":
        """Load episode from saved JSON file.

        Args:
            load_path: Path to episode JSON file

        Returns:
            Episode instance (without observations/actions)
        """
        with open(load_path) as f:
            data = json.load(f)

        return cls(
            task_id=data["task_id"],
            task_instruction=data["task_instruction"],
            video_path=data["video_path"],
            eval_result=data.get("eval_result", {}),
            metadata=data.get("metadata", {}),
        )
