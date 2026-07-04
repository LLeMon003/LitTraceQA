from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .config import PipelineConfig, is_api_key_configured


class AnswerClient:
    def __init__(self, config: PipelineConfig, retries: int = 2, model_override: str | None = None) -> None:
        self.config = config
        self.retries = retries
        self.model = model_override or config.answer_model

    def supports_text_generation(self) -> bool:
        return is_api_key_configured(self.config.answer_api_key) and bool(self.model)

    def supports_image_input(self) -> bool:
        model = self.model.lower()
        return any(marker in model for marker in ("vl", "vision", "qwen3-vl", "qwen2-vl", "glm-4.1v", "deepseek-vl"))

    def _image_to_data_url(self, image_path: str | Path, max_side: int = 1600, quality: int = 85) -> str:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = min(1.0, max_side / max(width, height))
            if scale < 1.0:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _with_images(self, messages: list[dict[str, Any]], image_paths: list[str | Path] | None) -> list[dict[str, Any]]:
        if not image_paths or not self.supports_image_input():
            return messages
        unique_paths: list[Path] = []
        seen: set[str] = set()
        for image_path in image_paths:
            path = Path(image_path)
            key = str(path)
            if key in seen or not path.exists():
                continue
            seen.add(key)
            unique_paths.append(path)
        if not unique_paths:
            return messages
        converted = [dict(message) for message in messages]
        last_user_index = max((idx for idx, message in enumerate(converted) if message.get("role") == "user"), default=-1)
        if last_user_index < 0:
            return converted
        original_content = converted[last_user_index].get("content", "")
        content: list[dict[str, Any]] = [{"type": "text", "text": str(original_content)}]
        for path in unique_paths:
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(path), "detail": "high"}})
        converted[last_user_index]["content"] = content
        return converted

    def generate_json(self, messages: list[dict[str, str]], image_paths: list[str | Path] | None = None) -> dict[str, Any]:
        if not self.supports_text_generation():
            raise RuntimeError("Answer generation model is not configured.")
        url = self.config.answer_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._with_images(messages, image_paths),
            "temperature": self.config.answer_temperature,
            "max_tokens": self.config.answer_max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.answer_api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.answer_timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"raw_response": data, "content": content}
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Answer generation failed: {last_error}") from last_error


def probe_chat_generation(
    *,
    model: str,
    api_key: str | None,
    base_url: str,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    if not is_api_key_configured(api_key):
        return {"model": model, "can_generate": False, "reason": "API key missing or placeholder.", "raw_response": None}
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": 'Return exactly this JSON object: {"ok": true}'},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
        return {
            "model": model,
            "can_generate": bool(content.strip()),
            "reason": "chat completion returned content" if content.strip() else "chat completion returned empty content",
            "content": content,
            "raw_response": data,
        }
    except Exception as exc:
        return {"model": model, "can_generate": False, "reason": str(exc), "raw_response": None}


def resolve_answer_generation_model(config: PipelineConfig) -> dict[str, Any]:
    vlm_probe = probe_chat_generation(
        model=config.answer_model,
        api_key=config.answer_api_key,
        base_url=config.answer_base_url,
        timeout_seconds=min(config.answer_timeout_seconds, 30),
    )
    if vlm_probe.get("can_generate"):
        return {
            "answer_model": config.answer_model,
            "source": "configured_vlm_answer_model_generation_capable",
            "vlm_generation_probe": vlm_probe,
        }
    return {
        "answer_model": config.base_generation_model,
        "source": "base_generation_model_after_vlm_probe_failed",
        "vlm_generation_probe": vlm_probe,
    }
