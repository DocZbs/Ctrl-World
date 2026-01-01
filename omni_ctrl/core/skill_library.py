"""Skill library for storing and managing discovered skills."""

from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any


class SkillLibrary:
    """Store and manage successfully discovered skills.

    The skill library maintains a catalog of all tasks that were
    successfully completed, along with their videos and evaluation results.
    This enables analysis of what skills the system has learned.
    """

    def __init__(self, save_dir: str):
        """Initialize skill library.

        Args:
            save_dir: Directory to store skill library
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.skills = []
        self._load_existing()

    def _load_existing(self):
        """Load existing skills from disk."""
        index_path = self.save_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self.skills = json.load(f)
                print(f"Loaded {len(self.skills)} existing skills from library")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to load skill library: {e}")
                self.skills = []

    def add(self, task, episode, eval_result):
        """Add a successful skill to the library.

        Args:
            task: Task object
            episode: Episode object
            eval_result: EvalResult object with evaluation details
        """
        skill = {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "success_criteria": task.success_criteria,
            "object_categories": task.object_categories,
            "difficulty": task.difficulty,
            "video_path": episode.video_path,
            "reward": eval_result.reward,
            "consistency": eval_result.physical_consistency,
            "reasoning": eval_result.reasoning,
            "metadata": episode.metadata,
        }

        self.skills.append(skill)

        # Save updated index
        self._save_index()

        print(f"Added skill to library: {task.instruction[:50]}...")

    def _save_index(self):
        """Save skill index to disk."""
        with open(self.save_dir / "index.json", "w") as f:
            json.dump(self.skills, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics.

        Returns:
            Dictionary with statistics about discovered skills
        """
        if not self.skills:
            return {
                "total_skills": 0,
                "avg_reward": 0.0,
                "avg_consistency": 0.0,
            }

        rewards = [s["reward"] for s in self.skills]
        consistencies = [s["consistency"] for s in self.skills]

        return {
            "total_skills": len(self.skills),
            "avg_reward": float(np.mean(rewards)),
            "avg_consistency": float(np.mean(consistencies)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }

    def get_skills_by_object(self, object_category: str) -> List[Dict]:
        """Get all skills involving a specific object category.

        Args:
            object_category: Object category to filter by

        Returns:
            List of skill dictionaries
        """
        return [
            s
            for s in self.skills
            if object_category in s.get("object_categories", [])
        ]

    def get_top_skills(self, k: int = 10) -> List[Dict]:
        """Get top K skills by reward.

        Args:
            k: Number of top skills to return

        Returns:
            List of top skill dictionaries
        """
        sorted_skills = sorted(self.skills, key=lambda s: s["reward"], reverse=True)
        return sorted_skills[:k]

    def export_summary(self, output_path: str):
        """Export a summary of the skill library.

        Args:
            output_path: Path to save summary JSON
        """
        summary = {
            "statistics": self.get_stats(),
            "top_skills": self.get_top_skills(k=10),
            "skills_by_difficulty": self._group_by_difficulty(),
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Skill library summary exported to {output_path}")

    def _group_by_difficulty(self) -> Dict[str, int]:
        """Group skills by difficulty level.

        Returns:
            Dictionary mapping difficulty ranges to counts
        """
        ranges = {
            "easy (0.0-0.3)": 0,
            "medium (0.3-0.7)": 0,
            "hard (0.7-1.0)": 0,
        }

        for skill in self.skills:
            diff = skill.get("difficulty", 0.5)
            if diff < 0.3:
                ranges["easy (0.0-0.3)"] += 1
            elif diff < 0.7:
                ranges["medium (0.3-0.7)"] += 1
            else:
                ranges["hard (0.7-1.0)"] += 1

        return ranges

    def __len__(self):
        """Get number of skills in library."""
        return len(self.skills)

    def __repr__(self):
        """String representation."""
        return f"SkillLibrary(path={self.save_dir}, skills={len(self.skills)})"
