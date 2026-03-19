"""Template-based task generator for MVP."""

import json
import random
from pathlib import Path
from .task_schema import Task


class TemplateTaskGenerator:
    """Generate tasks from fixed templates.

    This is the MVP implementation that samples from predefined task templates.
    Will be replaced by LLM-based generation in Phase 3.
    """

    def __init__(self, templates_path: str):
        """Initialize template generator.

        Args:
            templates_path: Path to JSON file containing task templates
        """
        templates_path = Path(templates_path)
        if not templates_path.exists():
            raise FileNotFoundError(f"Templates file not found: {templates_path}")

        with open(templates_path) as f:
            self.templates = json.load(f)

        if not self.templates:
            raise ValueError("Templates file is empty")

        self.task_counter = 0

    def generate_task(self) -> Task:
        """Sample a random task from templates.

        Returns:
            Task object with randomly sampled template
        """
        template = random.choice(self.templates)

        self.task_counter += 1
        return Task(
            instruction=template["instruction"],
            success_criteria=template["success_criteria"],
            object_categories=template.get("objects", []),
            difficulty=template.get("difficulty", 0.5),
            task_id=f"task_{self.task_counter:04d}",
        )

    def get_stats(self):
        """Get statistics about task generation.

        Returns:
            Dictionary with generator statistics
        """
        return {
            "total_templates": len(self.templates),
            "tasks_generated": self.task_counter,
        }
