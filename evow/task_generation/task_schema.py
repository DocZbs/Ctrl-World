"""Task schema and dataclasses."""

from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    """Task description with success criteria.

    Attributes:
        instruction: Natural language task instruction
        success_criteria: List of conditions that define task success
        object_categories: List of object types involved in the task
        difficulty: Task difficulty score (0.0 to 1.0)
        task_id: Unique identifier for this task
    """

    instruction: str
    success_criteria: List[str]
    object_categories: List[str]
    difficulty: float = 0.5
    task_id: str = ""

    def __post_init__(self):
        """Validate task attributes."""
        if not self.instruction:
            raise ValueError("Task instruction cannot be empty")
        if not self.success_criteria:
            raise ValueError("Task must have at least one success criterion")
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("Difficulty must be between 0.0 and 1.0")

    def to_dict(self):
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "success_criteria": self.success_criteria,
            "object_categories": self.object_categories,
            "difficulty": self.difficulty,
        }
