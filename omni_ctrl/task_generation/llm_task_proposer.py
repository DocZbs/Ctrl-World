"""LLM-based task proposer for NEXUS open-ended task generation.

Generates diverse manipulation tasks using GPT-4o, optionally grounded
in available DROID scene descriptions.
"""

import json
import os
from pathlib import Path
from typing import List

import openai

from .task_schema import Task


class LLMTaskProposer:
    """Generate diverse, open-ended tasks using an LLM.

    Replaces TemplateTaskGenerator for NEXUS. Generates batches of tasks
    via GPT-4o and buffers them for sequential consumption.
    """

    def __init__(self, config):
        self.model = getattr(config, "llm_model", "gpt-4o")
        api_key = getattr(config, "llm_api_key", "") or ""
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.getenv(api_key[2:-1], "")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")

        self.client = openai.OpenAI(api_key=api_key)
        self.num_tasks_per_batch = getattr(config, "num_tasks_per_batch", 10)
        self.task_counter = 0
        self.task_buffer: List[Task] = []

        droid_meta_path = getattr(config, "droid_meta_path", "")
        self._scene_descriptions = self._load_scene_descriptions(droid_meta_path)
        print(f"LLMTaskProposer initialized (model={self.model}, scenes={len(self._scene_descriptions)})")

    def _load_scene_descriptions(self, droid_meta_path: str) -> List[str]:
        """Load DROID scene descriptions for grounding task proposals."""
        descriptions = []
        if not droid_meta_path:
            return descriptions
        anno_dir = Path(droid_meta_path) / "annotation" / "train"
        if not anno_dir.exists():
            anno_dir = Path(droid_meta_path) / "annotation"
        if not anno_dir.exists():
            return descriptions
        for json_path in sorted(anno_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                texts = data.get("texts", [])
                if texts:
                    descriptions.append(texts[0])
                else:
                    instruction = data.get("language_instruction", "")
                    if instruction:
                        descriptions.append(instruction)
            except (json.JSONDecodeError, KeyError):
                continue
        return descriptions

    def generate_task(self) -> Task:
        """Generate a single task, refilling buffer from LLM if empty."""
        if not self.task_buffer:
            self.task_buffer = self._generate_batch()
        if not self.task_buffer:
            raise RuntimeError("LLM task generation returned no valid tasks")
        return self.task_buffer.pop(0)

    def _generate_batch(self) -> List[Task]:
        """Call LLM to generate a batch of diverse tasks."""
        prompt = self._build_proposal_prompt()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.9,
            )
            text = response.choices[0].message.content
            return self._parse_tasks(text)
        except Exception as e:
            print(f"Warning: LLM task generation failed: {e}")
            return []

    def _build_proposal_prompt(self) -> str:
        scene_block = ""
        if self._scene_descriptions:
            # Include a sample of scene descriptions for grounding
            sample = self._scene_descriptions[:20]
            scene_list = "\n".join(f"- {s}" for s in sample)
            scene_block = (
                f"\nHere are some example manipulation tasks from the robot's dataset "
                f"to give you a sense of the robot's capabilities:\n{scene_list}\n"
            )

        return f"""You are a task designer for a robot manipulation system.
Generate {self.num_tasks_per_batch} diverse robot manipulation tasks.

Requirements:
- Tasks should be realistic tabletop manipulation tasks a robot arm can perform
- Include a variety of skills: pick-and-place, pushing, folding, stacking, opening, closing, wiping, etc.
- Each task needs a clear natural language instruction
- Each task needs 1-3 specific success criteria
- Each task needs a list of involved object categories
- Assign a difficulty score from 0.0 (easy) to 1.0 (hard)
- Tasks should be diverse in objects, actions, and complexity
{scene_block}
Respond ONLY with a JSON array. Each element must have exactly these fields:
```json
[
  {{
    "instruction": "pick up the red block and place it on the plate",
    "success_criteria": ["robot grasps the red block", "red block is placed on the plate"],
    "objects": ["red block", "plate"],
    "difficulty": 0.4
  }}
]
```

Generate exactly {self.num_tasks_per_batch} tasks. No text outside the JSON array."""

    def _parse_tasks(self, response_text: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        # Extract JSON from response
        text = response_text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find JSON array in the text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end])
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse LLM task response: {e}")
                    return []
            else:
                print(f"Warning: No JSON array found in LLM response: {e}")
                return []

        if not isinstance(items, list):
            return []

        tasks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction", "")
            success_criteria = item.get("success_criteria", [])
            objects = item.get("objects", [])
            difficulty = item.get("difficulty", 0.5)

            if not instruction or not success_criteria:
                continue

            difficulty = max(0.0, min(1.0, float(difficulty)))
            self.task_counter += 1
            tasks.append(Task(
                instruction=instruction,
                success_criteria=success_criteria,
                object_categories=objects,
                difficulty=difficulty,
                task_id=f"nexus_{self.task_counter:04d}",
            ))
        return tasks

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "tasks_generated": self.task_counter,
            "buffer_size": len(self.task_buffer),
            "scene_descriptions": len(self._scene_descriptions),
        }
