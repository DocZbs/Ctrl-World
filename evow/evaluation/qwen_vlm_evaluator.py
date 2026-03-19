"""Local Qwen-VL evaluator for robot manipulation trajectories."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("Please install opencv-python: pip install opencv-python") from exc

try:
    import torch
except ImportError as exc:
    raise ImportError("Please install torch: pip install torch") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("Please install Pillow: pip install Pillow") from exc

try:
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
    from transformers import __version__ as TRANSFORMERS_VERSION
except ImportError as exc:
    raise ImportError("Please install transformers: pip install transformers") from exc


@dataclass
class EvalResult:
    """Evaluation result from VLM."""

    success: bool
    physical_consistency: float
    reward: float
    reasoning: str
    raw_response: dict
    failure_reason: str = "success"


class QwenVLRewardEvaluator:
    """Evaluate rollouts with a local Qwen-VL style model."""

    def __init__(self, config):
        self.model = getattr(config, "vlm_model", None) or os.getenv(
            "EVOW_QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.max_retries = int(getattr(config, "max_retries", 3))
        self.timeout = int(getattr(config, "timeout", 60))
        self.success_weight = float(getattr(config, "success_weight", 1.0))
        self.consistency_weight = float(getattr(config, "consistency_weight", 0.5))

        eval_num_frames = getattr(config, "eval_num_frames", None)
        if eval_num_frames is None:
            eval_num_frames = os.getenv("EVOW_VLM_NUM_FRAMES", "8")
        try:
            eval_num_frames = int(eval_num_frames)
        except (TypeError, ValueError):
            eval_num_frames = 8
        self.eval_num_frames = max(eval_num_frames, 1)

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

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = str(getattr(config, "qwen_device", None) or os.getenv("EVOW_QWEN_DEVICE", default_device))
        self.max_new_tokens = int(
            getattr(config, "qwen_max_new_tokens", None) or os.getenv("EVOW_QWEN_MAX_NEW_TOKENS", "512")
        )
        self.temperature = float(
            getattr(config, "qwen_temperature", None) or os.getenv("EVOW_QWEN_TEMPERATURE", "0.0")
        )
        dtype_name = str(
            getattr(config, "qwen_torch_dtype", None) or os.getenv("EVOW_QWEN_TORCH_DTYPE", "auto")
        ).lower()
        trust_remote_code = str(
            getattr(config, "qwen_trust_remote_code", None)
            or os.getenv("EVOW_QWEN_TRUST_REMOTE_CODE", "1")
        ).lower() in {"1", "true", "yes", "y", "on"}

        torch_dtype = self._resolve_dtype(dtype_name)
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=trust_remote_code)
        self._model = self._load_model(load_kwargs)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._dtype = next(self._model.parameters()).dtype

        print(
            "Qwen-VL Evaluator initialized "
            f"(model={self.model}, device={self.device}, frame_crop={self.frame_crop}, "
            f"eval_num_frames={self.eval_num_frames})"
        )

    def evaluate(self, video_path: str, task) -> EvalResult:
        frames = self._extract_frames(video_path, num_frames=self.eval_num_frames)
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return EvalResult(
                success=False,
                physical_consistency=0.0,
                reward=0.0,
                reasoning="Failed to extract video frames",
                raw_response={},
                failure_reason="wm_failure",
            )

        prompt = self._build_eval_prompt(task)

        for attempt in range(self.max_retries):
            try:
                response = self._call_vlm(frames, prompt)
                break
            except Exception as e:
                print(f"Qwen-VL call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return EvalResult(
                        success=False,
                        physical_consistency=0.0,
                        reward=0.0,
                        reasoning=f"Qwen-VL call failed: {e}",
                        raw_response={},
                        failure_reason="wm_failure",
                    )

        return self._parse_response(response, task)

    def _build_eval_prompt(self, task) -> str:
        return f"""You are evaluating a robot manipulation task.

**Task Instruction:** {task.instruction}

**Success Criteria:**
{chr(10).join(f'- {c}' for c in task.success_criteria)}

**Frame Layout Note:**
The image is a contact sheet of chronological rollout frames, ordered left-to-right, top-to-bottom.

**Your Task:**
Watch the sequence of video frames and evaluate:

1. **Task Success**: Did the robot successfully complete the task according to ALL success criteria? (Yes/No)
   - **Temporal rule**: If all success criteria are clearly satisfied at any point in the sequence, mark success = true.
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

    def _extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = self._crop_frame_for_eval(frame_rgb)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def _crop_frame_for_eval(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.frame_crop == "full":
            return frame_rgb

        h = int(frame_rgb.shape[0])

        base = frame_rgb
        if self.frame_crop == "top":
            if h >= 240:
                half = max(h // 2, 1)
                base = frame_rgb[:half, :, :]
            return base

        if h >= 240:
            half = max(h // 2, 1)
            base = frame_rgb[half:, :, :]

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

    def _load_model(self, load_kwargs: dict):
        try:
            return AutoModelForVision2Seq.from_pretrained(self.model, **load_kwargs)
        except Exception as first_error:
            try:
                return AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)
            except Exception as second_error:
                extra_hint = ""
                combined_err = f"{first_error}\n{second_error}"
                if "qwen3_vl" in combined_err:
                    extra_hint = (
                        " Qwen3-VL checkpoints require newer Transformers "
                        f"(detected {TRANSFORMERS_VERSION}, expected >= 4.57)."
                    )
                raise RuntimeError(
                    f"Failed to load Qwen-VL model '{self.model}'. "
                    f"Vision2Seq error: {first_error}; CausalLM fallback error: {second_error}"
                    f"{extra_hint}"
                ) from second_error

    def _resolve_dtype(self, dtype_name: str):
        if dtype_name in {"", "none", "auto"}:
            if self.device.startswith("cuda"):
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_name in {"fp32", "float32"}:
            return torch.float32
        print(f"Warning: unsupported qwen_torch_dtype={dtype_name}, fallback to auto")
        return self._resolve_dtype("auto")

    def _make_frame_grid(self, frames: List[np.ndarray]) -> Image.Image:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        cols = min(4, max(len(pil_frames), 1))
        rows = int(math.ceil(len(pil_frames) / cols))

        base_w, base_h = pil_frames[0].size
        max_side = max(base_w, base_h)
        if max_side > 448:
            scale = 448.0 / float(max_side)
            tile_w = max(int(round(base_w * scale)), 1)
            tile_h = max(int(round(base_h * scale)), 1)
        else:
            tile_w, tile_h = base_w, base_h

        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
        for idx, image in enumerate(pil_frames):
            resized = image.resize((tile_w, tile_h), Image.Resampling.BICUBIC)
            x = (idx % cols) * tile_w
            y = (idx // cols) * tile_h
            canvas.paste(resized, (x, y))
        return canvas

    def _prepare_inputs(self, prompt: str, grid_image: Image.Image):
        user_content = [
            {"type": "image", "image": grid_image},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": user_content}]

        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.processor(
                text=[prompt_text],
                images=[grid_image],
                return_tensors="pt",
                padding=True,
            )

        return self.processor(
            text=prompt,
            images=grid_image,
            return_tensors="pt",
        )

    def _call_vlm(self, frames: List[np.ndarray], prompt: str) -> str:
        if not frames:
            raise ValueError("No frames provided for Qwen-VL evaluation")

        grid_image = self._make_frame_grid(frames)
        inputs = self._prepare_inputs(prompt, grid_image)

        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if torch.is_floating_point(value):
                    value = value.to(dtype=self._dtype)
            moved[key] = value
        inputs = moved

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        if "input_ids" in inputs:
            trimmed_ids = []
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids):
                trimmed_ids.append(out_ids[in_ids.shape[0]:])
        else:
            trimmed_ids = output_ids

        text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response = text[0].strip() if text else ""
        if not response:
            raise RuntimeError("Qwen-VL response is empty")
        return response

    def _parse_response(self, response: str, task) -> EvalResult:
        try:
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

            success = any(word in response.lower() for word in ["yes", "true", "successful", "completed"])
            consistency = 0.5
            reasoning = response[:200]
            failure_reason = "success" if success else "expert_failure"

        if failure_reason == "vla_failure":
            failure_reason = "expert_failure"

        valid_reasons = {"success", "task_failure", "scene_failure", "expert_failure", "wm_failure"}
        if failure_reason not in valid_reasons:
            failure_reason = "success" if success else "expert_failure"

        if success and failure_reason != "success":
            failure_reason = "success"
        elif not success and failure_reason == "success":
            failure_reason = "expert_failure"

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
        if not success:
            return 0.0

        reward = self.success_weight * (1.0 * consistency)
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)

    def get_stats(self):
        return {
            "vlm_model": self.model,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "device": self.device,
            "backend": "qwen_vl_local",
        }
