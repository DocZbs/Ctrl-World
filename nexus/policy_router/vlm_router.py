"""VLM-based policy router for NEXUS mixture-of-experts selection (Pi-only)."""

import os
import json
import base64
import io
import numpy as np
from typing import List, Optional

from PIL import Image

from .policy_router import PolicyRouter


# Expert descriptions for the VLM prompt (Pi-only)
EXPERT_DESCRIPTIONS = {
    "pi05": "Pi0.5 -- state-of-the-art VLA model trained on diverse DROID manipulation data. "
            "Outputs 15-step joint velocity action chunks. Strong generalist.",
    "pi0": "Pi0 -- VLA model for DROID manipulation. Outputs 10-step joint velocity chunks. "
           "Good baseline performance.",
    "pi0fast": "Pi0-FAST -- faster variant of Pi0 with 10-step joint velocity chunks. "
               "Lower latency, slightly less precise.",
}


class VLMPolicyRouter:
    """VLM-conditioned expert router q_omega(m | l, o_0, M).

    Uses GPT-4o to select the best VLA expert given task description,
    initial scene observation, and experience memory context.
    Falls back to statistical routing (UCB) if VLM call fails.
    """

    def __init__(self, config, base_router: PolicyRouter):
        self.base_router = base_router
        self.vlm_model = getattr(config, "vlm_model", "gpt-4o")
        self.timeout = getattr(config, "timeout", 30)

        api_key = getattr(config, "vlm_api_key", "") or ""
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.getenv(api_key[2:-1], "")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")

        import openai
        self.client = openai.OpenAI(api_key=api_key)

        self.available_experts = base_router.get_available_policy_names()
        print(f"VLMPolicyRouter initialized (model={self.vlm_model}, experts={self.available_experts})")

    def select_expert(
        self,
        task,
        scene_image: Optional[np.ndarray],
        memory_context: list,
        excluded_experts: Optional[list] = None,
    ):
        """Select best expert using VLM reasoning.

        Args:
            task: Task object with instruction and success_criteria
            scene_image: Initial scene observation (H, W, 3) uint8, or None
            memory_context: List of MemoryEntry objects for context
            excluded_experts: Expert names to exclude (failed previously)

        Returns:
            Policy object from base_router
        """
        candidates = [
            e for e in self.available_experts
            if e not in (excluded_experts or [])
        ]

        if not candidates:
            candidates = list(self.available_experts)

        # Single candidate -- no need to call VLM
        if len(candidates) == 1:
            return self.base_router.get_policy_by_name(candidates[0])

        try:
            chosen_name = self._vlm_select(task, scene_image, memory_context, candidates)
            if chosen_name not in candidates:
                print(f"Warning: VLM chose '{chosen_name}' not in candidates, falling back")
                return self.base_router.select_policy(task)
            return self.base_router.get_policy_by_name(chosen_name)
        except Exception as e:
            print(f"Warning: VLM routing failed ({e}), falling back to statistical selection")
            return self.base_router.select_policy(task)

    def _vlm_select(
        self,
        task,
        scene_image: Optional[np.ndarray],
        memory_context: list,
        candidates: list,
    ) -> str:
        """Call VLM to select expert. Returns chosen expert name."""
        prompt = self._build_routing_prompt(task, memory_context, candidates)

        message_content = [{"type": "text", "text": prompt}]

        # Attach scene image if available
        if scene_image is not None:
            img_b64 = self._encode_image(scene_image)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })

        response = self.client.chat.completions.create(
            model=self.vlm_model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=200,
            temperature=0.3,
            timeout=self.timeout,
        )

        text = response.choices[0].message.content.strip()
        return self._parse_expert_choice(text, candidates)

    def _build_routing_prompt(self, task, memory_context: list, candidates: list) -> str:
        # Build expert descriptions block
        expert_block = "\n".join(
            f"- **{name}**: {EXPERT_DESCRIPTIONS.get(name, 'No description available.')}"
            for name in candidates
        )

        # Build memory context block
        memory_block = "No previous attempts recorded."
        if memory_context:
            lines = []
            for entry in memory_context[-10:]:  # Last 10 entries
                lines.append(
                    f"  - Task: \"{entry.task_instruction[:60]}\" | "
                    f"Expert: {entry.expert_name} | "
                    f"Reward: {entry.reward:.2f} | "
                    f"Result: {entry.failure_attribution}"
                )
            memory_block = "Recent outcomes:\n" + "\n".join(lines)

        return f"""You are a robot policy routing expert. Select the best VLA expert for this task.

**Task:** {task.instruction}
**Success Criteria:** {', '.join(task.success_criteria)}

**Available Experts:**
{expert_block}

**Experience Memory:**
{memory_block}

Based on the task requirements, scene (if image provided), and past experience,
which expert is most likely to succeed?

Respond with ONLY the expert name (one of: {', '.join(candidates)}). No explanation."""

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 JPEG string."""
        img = Image.fromarray(image)
        max_size = 512
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(d * ratio) for d in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_expert_choice(self, text: str, candidates: list) -> str:
        """Parse VLM response to extract expert name."""
        text_lower = text.strip().lower()
        # Direct match
        for name in candidates:
            if name == text_lower:
                return name
        # Substring match
        for name in candidates:
            if name in text_lower:
                return name
        # Try JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                choice = data.get("expert", data.get("name", ""))
                choice = choice.strip().lower()
                for name in candidates:
                    if name in choice:
                        return name
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ValueError(f"Could not parse expert choice from VLM response: '{text}'")

    def update(self, policy_name: str, reward: float):
        """Update base router statistics."""
        self.base_router.update(policy_name, reward)
