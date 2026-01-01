"""GPT-4V based evaluation for robot manipulation tasks."""

import base64
import json
import io
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List

try:
    import openai
except ImportError:
    raise ImportError("Please install openai: pip install openai")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install Pillow: pip install Pillow")

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python: pip install opencv-python")


@dataclass
class EvalResult:
    """Evaluation result from VLM.

    Attributes:
        success: Whether the task was successfully completed
        physical_consistency: Physical plausibility score (0-1)
        reward: Computed reward signal for router update
        reasoning: VLM's textual reasoning/explanation
        raw_response: Raw VLM response dict
    """

    success: bool
    physical_consistency: float
    reward: float
    reasoning: str
    raw_response: dict


class GPT4VEvaluator:
    """Evaluate robot manipulation episodes using GPT-4V.

    Provides both task success evaluation and physical consistency checking.
    """

    def __init__(self, config):
        """Initialize GPT-4V evaluator.

        Args:
            config: EvaluationConfig instance
        """
        self.client = openai.OpenAI(api_key=config.api_key)
        self.max_retries = config.max_retries
        self.timeout = config.timeout
        self.success_weight = config.success_weight
        self.consistency_weight = config.consistency_weight

        print("GPT-4V Evaluator initialized")

    def evaluate(self, video_path: str, task) -> EvalResult:
        """Evaluate episode for task success from video.

        Args:
            video_path: Path to episode video file
            task: Task object with instruction and success criteria

        Returns:
            EvalResult with success, consistency, and reward
        """
        # Extract frames from video
        frames = self._extract_frames(video_path, num_frames=8)

        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return EvalResult(
                success=False,
                physical_consistency=0.0,
                reward=0.0,
                reasoning="Failed to extract video frames",
                raw_response={},
            )

        # Build evaluation prompt
        prompt = self._build_eval_prompt(task)

        # Call GPT-4V with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_gpt4v(frames, prompt)
                break
            except Exception as e:
                print(f"GPT-4V API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return EvalResult(
                        success=False,
                        physical_consistency=0.0,
                        reward=0.0,
                        reasoning=f"API call failed: {e}",
                        raw_response={},
                    )

        # Parse response
        eval_result = self._parse_response(response, task)
        return eval_result

    def _build_eval_prompt(self, task) -> str:
        """Build evaluation prompt for GPT-4V.

        Args:
            task: Task object

        Returns:
            Prompt string
        """
        prompt = f"""You are evaluating a robot manipulation task.

**Task Instruction:** {task.instruction}

**Success Criteria:**
{chr(10).join(f'- {c}' for c in task.success_criteria)}

**Your Task:**
Watch the sequence of video frames (shown chronologically) and evaluate:

1. **Task Success**: Did the robot successfully complete the task according to ALL success criteria? (Yes/No)

2. **Physical Consistency**: Are there any physically impossible events or hallucinations?
   - Examples: Objects disappearing, teleporting, defying gravity, unrealistic motions
   - Score: 0.0 (many issues) to 1.0 (perfectly consistent)

3. **Reasoning**: Provide 2-3 sentences explaining your evaluation. Be specific about what you observed.

**Respond ONLY in valid JSON format:**
```json
{{
  "success": true/false,
  "consistency": 0.0-1.0,
  "reasoning": "Your explanation here..."
}}
```

Do not include any text outside the JSON block."""

        return prompt

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """Extract evenly-spaced frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            List of RGB frames (H, W, 3)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Warning: Video has no frames: {video_path}")
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def _call_gpt4v(self, frames: List[np.ndarray], prompt: str) -> str:
        """Call GPT-4V API with video frames.

        Args:
            frames: List of RGB frames
            prompt: Evaluation prompt

        Returns:
            GPT-4V response text
        """
        # Build message with images
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        # Add frames as base64-encoded images
        for frame in frames:
            # Convert numpy array to PIL Image
            img = Image.fromarray(frame)

            # Resize if too large (GPT-4V has size limits)
            max_size = 512
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

        # API call
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500,
            timeout=self.timeout,
        )

        return response.choices[0].message.content

    def _parse_response(self, response: str, task) -> EvalResult:
        """Parse GPT-4V response into EvalResult.

        Args:
            response: GPT-4V response text
            task: Original task object

        Returns:
            EvalResult object
        """
        # Try to extract JSON from response
        try:
            # Find JSON block
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response

            result_dict = json.loads(json_str)

            success = result_dict.get("success", False)
            consistency = float(result_dict.get("consistency", 0.0))
            reasoning = result_dict.get("reasoning", "")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Response: {response}")

            # Fallback parsing
            success = any(word in response.lower() for word in ["yes", "true", "successful", "completed"])
            consistency = 0.5  # Default medium consistency
            reasoning = response[:200]  # Take first 200 chars

        # Compute reward
        reward = self._compute_reward(success, consistency)

        return EvalResult(
            success=success,
            physical_consistency=consistency,
            reward=reward,
            reasoning=reasoning,
            raw_response={"text": response},
        )

    def _compute_reward(self, success: bool, consistency: float) -> float:
        """Compute scalar reward for router update.

        Args:
            success: Task success (True/False)
            consistency: Physical consistency score (0-1)

        Returns:
            Reward value (0-1)
        """
        if not success:
            return 0.0

        # Success reward weighted by consistency
        reward = self.success_weight * (1.0 * consistency)

        # Ensure in [0, 1] range
        reward = np.clip(reward, 0.0, 1.0)

        return float(reward)

    def get_stats(self):
        """Get evaluator statistics.

        Returns:
            Dictionary with evaluator info
        """
        return {
            "vlm_model": "gpt-4-vision-preview",
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
