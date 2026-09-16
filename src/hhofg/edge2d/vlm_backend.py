from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class VLMResult:
    s2d_score: float | None
    raw_response: str
    model_id: str
    error: str = ""
    metadata: dict[str, Any] | None = None


def extract_json_object(text: str) -> dict[str, Any]:
    s = str(text).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    while start >= 0:
        depth = 0
        for idx in range(start, len(s)):
            if s[idx] == "{":
                depth += 1
            elif s[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : idx + 1])
                    except Exception:
                        break
        start = s.find("{", start + 1)
    raise ValueError(f"VLM response is not valid JSON: {s[:500]}")


def parse_s2d_score(text: str) -> tuple[float, dict[str, Any]]:
    try:
        data = extract_json_object(text)
    except ValueError:
        # LLaVA can emit the requested numeric field first and then truncate a
        # verbose reason at max_new_tokens. Recover only that explicit field;
        # never infer a score from free-form prose.
        match = re.search(r'["\']s2d_score["\']\s*:\s*["\']?([-+]?\d*\.?\d+)', str(text))
        if not match:
            raise
        data = {"s2d_score": float(match.group(1)), "truncated_json_recovered": True}
    value = data.get("s2d_score")
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+", value)
        if not m:
            raise ValueError(f"s2d_score is not numeric: {value!r}")
        value = float(m.group(0))
    score = float(value)
    if not np.isfinite(score):
        raise ValueError("s2d_score is not finite")
    return float(np.clip(score, 0.01, 0.99)), data


class TransformersLlavaBackend:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_path = str(cfg.get("model_path", "checkpoints/llava-v1.6-mistral-7b-hf"))
        self.model_id = str(cfg.get("model_id", Path(self.model_path).name))
        self.local_files_only = bool(cfg.get("local_files_only", True))
        self.device = str(cfg.get("device", "cuda"))
        self.torch_dtype = str(cfg.get("torch_dtype", "float16"))
        self.device_map = cfg.get("device_map", "auto")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 96))
        self.temperature = float(cfg.get("temperature", 0.0))
        self._loaded = False
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(self.model_path, local_files_only=self.local_files_only)
        model_type = str(getattr(config, "model_type", ""))
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
        if model_type == "llava_onevision":
            from transformers import LlavaOnevisionForConditionalGeneration

            cls = LlavaOnevisionForConditionalGeneration
        elif model_type == "llava_next":
            from transformers import LlavaNextForConditionalGeneration

            cls = LlavaNextForConditionalGeneration
        else:
            raise RuntimeError(f"Unsupported LLaVA model_type: {model_type}")
        self.model = cls.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._loaded = True

    def generate_json(self, *, image: Image.Image, prompt: str) -> VLMResult:
        self._load()
        import torch

        assert self.processor is not None and self.model is not None
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        first_device = next(self.model.parameters()).device
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                if key in {"pixel_values"}:
                    dtype = torch.float16 if self.torch_dtype == "float16" and first_device.type == "cuda" else None
                    inputs[key] = value.to(first_device, dtype=dtype) if dtype is not None else value.to(first_device)
                else:
                    inputs[key] = value.to(first_device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.temperature > 1e-6,
                temperature=self.temperature if self.temperature > 1e-6 else None,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        input_len = int(inputs["input_ids"].shape[1])
        raw = self.processor.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
        try:
            score, parsed = parse_s2d_score(raw)
            return VLMResult(score, raw, self.model_id, metadata={"parsed": parsed})
        except Exception as exc:
            return VLMResult(None, raw, self.model_id, error=str(exc))
