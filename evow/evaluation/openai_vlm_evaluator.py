"""OpenAI VLM evaluation for robot manipulation tasks."""

import base64
import json
import io
import os
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
        failure_reason: Categorized failure reason - one of:
            - "success": Task completed successfully
            - "wm_failure": Failed due to world model (unrealistic physics, hallucinations)
            - "vla_failure": Failed due to VLA policy (wrong actions, poor planning)
    """

    success: bool
    physical_consistency: float
    reward: float
    reasoning: str
    raw_response: dict
    # One of: "success", "task_failure", "scene_failure", "expert_failure", "wm_failure"
    failure_reason: str = "success"


class OpenAIVLMEvaluator:
    """Evaluate robot manipulation episodes using OpenAI VLMs.

    Provides both task success evaluation and physical consistency checking.
    """

    def __init__(self, config):
        """Initialize GPT-4V evaluator.

        Args:
            config: EvaluationConfig instance
        """
        api_key = getattr(config, "api_key", None)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_key = api_key[2:-1]
            api_key = os.getenv(env_key, "")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = getattr(config, "vlm_model", "gpt-4o-mini")
        self.fallback_model = getattr(config, "vlm_fallback_model", None)
        self.max_retries = config.max_retries
        self.timeout = config.timeout
        self.success_weight = config.success_weight
        self.consistency_weight = config.consistency_weight

        eval_num_frames = getattr(config, "eval_num_frames", None)
        if eval_num_frames is None:
            eval_num_frames = os.getenv("EVOW_VLM_NUM_FRAMES", "8")
        try:
            eval_num_frames = int(eval_num_frames)
        except (TypeError, ValueError):
            eval_num_frames = 8
        self.eval_num_frames = max(eval_num_frames, 1)

        # Frame crop policy for evaluation input:
        #   - "bottom": evaluate generated rollout panel (bottom half)
        #   - "bottom_right": evaluate only right camera view in bottom panel
        #   - "bottom_center": evaluate only center camera view in bottom panel
        #   - "bottom_left": evaluate only left camera view in bottom panel
        #   - "top": evaluate only reference/GT panel
        #   - "full": evaluate full frame
        frame_crop = getattr(config, "frame_crop", None)
        if frame_crop is None:
            frame_crop = os.getenv("EVOW_VLM_FRAME_CROP", "bottom_right")
        frame_crop = str(frame_crop).strip().lower()

        aliases = {
            "right_bottom": "bottom_right",
            "left_bottom": "bottom_left",
            "center_bottom": "bottom_center",
            "pred": "bottom",
            "pred_right": "bottom_right",
        }
        frame_crop = aliases.get(frame_crop, frame_crop)

        valid_crops = {
            "bottom",
            "bottom_right",
            "bottom_center",
            "bottom_left",
            "top",
            "full",
        }
        if frame_crop not in valid_crops:
            print(f"Warning: invalid frame_crop={frame_crop}, fallback to bottom")
            frame_crop = "bottom"
        self.frame_crop = frame_crop

        print(
            "OpenAI VLM Evaluator initialized "
            f"(frame_crop={self.frame_crop}, eval_num_frames={self.eval_num_frames})"
        )

    def evaluate(self, video_path: str, task) -> EvalResult:
        """Evaluate episode for task success from video.

        Args:
            video_path: Path to episode video file
            task: Task object with instruction and success criteria

        Returns:
            EvalResult with success, consistency, and reward
        """
        # Extract frames from video
        frames = self._extract_frames(video_path, num_frames=self.eval_num_frames)

        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return EvalResult(
                success=False,
                physical_consistency=0.0,
                reward=0.0,
                reasoning="Failed to extract video frames",
                raw_response={},
                failure_reason="wm_failure",  # Video extraction failure indicates WM issue
            )

        # Build evaluation prompt
        prompt = self._build_eval_prompt(task)

        # Call GPT-4V with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_vlm(frames, prompt)
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
                        failure_reason="wm_failure",  # API failure means we can't evaluate
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
   - **Temporal rule**: If all success criteria are clearly satisfied at any point in the sequence, mark success = true (even if this happens mid-sequence, not only at the final frame).
   - Do not require the final frame to remain successful.
   - Ignore single-frame visual glitches; require clear visual evidence across at least two nearby frames when possible.

2. **Physical Consistency**: Are there any physically impossible events or hallucinations?
   - Examples: Objects disappearing, teleporting, defying gravity, unrealistic motions
   - Score: 0.0 (many issues) to 1.0 (perfectly consistent)

3. **Failure Attribution** (only if task failed):
   - "task_failure": The task itself is ill-defined, ambiguous, or physically impossible given the scene
   - "scene_failure": The retrieved scene does not match the task requirements (wrong objects, wrong setup, missing required items)
   - "expert_failure": The robot policy made wrong actions, poor planning, or didn't follow the instruction correctly
   - "wm_failure": The world model produced unrealistic physics, visual hallucinations, or impossible object behaviors
   - "success": Task completed successfully (use this if success is true)

4. **Reasoning**: Provide 2-3 sentences explaining your evaluation. Be specific about what you observed.

**Respond ONLY in valid JSON format:**
```json
{{
  "success": true/false,
  "consistency": 0.0-1.0,
  "failure_reason": "success" or "task_failure" or "scene_failure" or "expert_failure" or "wm_failure",
  "reasoning": "Your explanation here..."
}}
```

Do not include any text outside the JSON block."""

        return prompt

    def _extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
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
                frame_rgb = self._crop_frame_for_eval(frame_rgb)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def _crop_frame_for_eval(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Crop frame according to configured panel policy.

        For Ctrl-World videos with GT+Pred vertical layout, use bottom/pred panel by default.
        Also supports selecting a single camera view in the bottom panel.
        """
        if self.frame_crop == "full":
            return frame_rgb

        h = int(frame_rgb.shape[0])

        # First, choose vertical panel.
        base = frame_rgb
        if self.frame_crop == "top":
            if h >= 240:
                half = max(h // 2, 1)
                base = frame_rgb[:half, :, :]
            return base

        # Bottom-family crops prefer predicted panel when stacked GT+Pred is present.
        if h >= 240:
            half = max(h // 2, 1)
            base = frame_rgb[half:, :, :]

        # Optional horizontal single-view crop in bottom panel.
        if self.frame_crop in {"bottom_left", "bottom_center", "bottom_right"}:
            w = int(base.shape[1])
            if w >= 3:
                one_third = max(w // 3, 1)
                if self.frame_crop == "bottom_left":
                    x0, x1 = 0, one_third
                elif self.frame_crop == "bottom_center":
                    x0, x1 = one_third, min(one_third * 2, w)
                else:
                    x0, x1 = min(one_third * 2, w - 1), w
                return base[:, x0:x1, :]

        return base

    def _call_vlm(self, frames: List[np.ndarray], prompt: str) -> str:
        """Call OpenAI VLM API with video frames.

        Args:
            frames: List of RGB frames
            prompt: Evaluation prompt

        Returns:
            GPT-4V response text
        """
        # Build message with images
        message_content = [{"type": "input_text", "text": prompt}]

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

            message_content.append(
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}
            )

        # API call (Responses API preferred)
        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": message_content}],
                max_output_tokens=500,
                timeout=self.timeout,
            )
            text = self._extract_response_text(response)
            if not text and self.fallback_model:
                response = self.client.responses.create(
                    model=self.fallback_model,
                    input=[{"role": "user", "content": message_content}],
                    max_output_tokens=500,
                    timeout=self.timeout,
                )
                text = self._extract_response_text(response)
            if not text:
                raise RuntimeError("OpenAI response did not contain any text output.")
            return text

        # Fallback: chat.completions for older SDKs
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {"type": "image_url", "image_url": {"url": c["image_url"]}}
                            for c in message_content[1:]
                        ],
                    ],
                }
            ],
            max_tokens=500,
            timeout=self.timeout,
        )
        text = self._extract_response_text(response)
        if not text and self.fallback_model:
            response = self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[
                                {"type": "image_url", "image_url": {"url": c["image_url"]}}
                                for c in message_content[1:]
                            ],
                        ],
                    }
                ],
                max_tokens=500,
                timeout=self.timeout,
            )
            text = self._extract_response_text(response)
        if not text:
            raise RuntimeError("OpenAI response did not contain any text output.")
        return text

    def _extract_response_text(self, response) -> str:
        """Extract text from different OpenAI SDK response shapes."""
        if response is None:
            return ""

        text = getattr(response, "output_text", None)
        if text:
            return text

        output = getattr(response, "output", None)
        if isinstance(output, list):
            texts = []
            for item in output:
                content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if isinstance(content, list):
                    for part in content:
                        part_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if part_text:
                            texts.append(part_text)
                elif isinstance(content, str):
                    texts.append(content)
            if texts:
                return "\n".join(texts)

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if msg and msg.get("content"):
                    return msg["content"]
            else:
                msg = getattr(first, "message", None)
                if msg and getattr(msg, "content", None):
                    return msg.content

        return ""

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
            failure_reason = result_dict.get("failure_reason", "success" if success else "expert_failure")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Response: {response}")

            # Fallback parsing
            success = any(word in response.lower() for word in ["yes", "true", "successful", "completed"])
            consistency = 0.5  # Default medium consistency
            reasoning = response[:200]  # Take first 200 chars
            failure_reason = "success" if success else "expert_failure"

        # Normalize legacy failure_reason values
        if failure_reason == "vla_failure":
            failure_reason = "expert_failure"

        # Ensure failure_reason is valid
        valid_reasons = {"success", "task_failure", "scene_failure", "expert_failure", "wm_failure"}
        if failure_reason not in valid_reasons:
            failure_reason = "success" if success else "expert_failure"

        # Auto-correct failure_reason based on success
        if success and failure_reason != "success":
            failure_reason = "success"
        elif not success and failure_reason == "success":
            failure_reason = "expert_failure"

        # Compute reward
        reward = self._compute_reward(success, consistency)

        return EvalResult(
            success=success,
            physical_consistency=consistency,
            reward=reward,
            reasoning=reasoning,
            raw_response={"text": response},
            failure_reason=failure_reason,
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
            "vlm_model": self.model,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
