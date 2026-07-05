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


def _model_looks_vision_capable(model: str) -> bool:
    return any(marker in model.lower() for marker in ("vl", "vision", "qwen3-vl", "qwen2-vl", "glm-4.1v", "deepseek-vl", "gpt-4o"))


class VLMParserClient:
    def __init__(self, config: PipelineConfig, retries: int = 2) -> None:
        self.config = config
        self.retries = retries

    def supports_image_input(self) -> bool:
        return is_api_key_configured(self.config.parser_api_key) and _model_looks_vision_capable(self.config.parser_model)

    def _image_to_data_url(self, image_path: str | Path, max_side: int = 1800, quality: int = 88) -> str:
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

    def _with_image(self, messages: list[dict[str, Any]], image_path: str | Path) -> list[dict[str, Any]]:
        converted = [dict(message) for message in messages]
        last_user = max((idx for idx, message in enumerate(converted) if message.get("role") == "user"), default=-1)
        if last_user < 0:
            raise ValueError("Parser messages must include a user message.")
        original = converted[last_user].get("content", "")
        converted[last_user]["content"] = [
            {"type": "text", "text": str(original)},
            {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path), "detail": "high"}},
        ]
        return converted

    def generate_page_structure(self, messages: list[dict[str, Any]], image_path: str | Path) -> dict[str, Any]:
        if not self.supports_image_input():
            raise RuntimeError("Parser VLM image input is not configured or the model name is not vision-capable.")
        url = self.config.parser_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.parser_model,
            "messages": self._with_image(messages, image_path),
            "temperature": self.config.parser_temperature,
            "max_tokens": self.config.parser_max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.parser_api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.parser_timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"raw_response": data, "content": content}
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Parser VLM call failed: {last_error}") from last_error
