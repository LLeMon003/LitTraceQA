from __future__ import annotations

import base64
import io
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import PipelineConfig, is_api_key_configured

try:
    from PIL import Image
except Exception:  # pragma: no cover - skip-generation and text-only runs do not need PIL
    Image = None  # type: ignore[assignment]


def _model_looks_vision_capable(model: str) -> bool:
    lowered = (model or "").lower()
    return any(marker in lowered for marker in ["vl", "vision", "qwen-vl", "llava", "gpt-4o", "gemini"])


class AnswerVLMError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "unknown", status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable


class VLMAnswerClient:
    def __init__(self, config: PipelineConfig, retries: int = 2) -> None:
        self.config = config
        self.retries = retries

    def supports_text_generation(self) -> bool:
        return is_api_key_configured(self.config.answer_api_key) and bool(self.config.answer_model)

    def supports_image_input(self, model: str | None = None) -> bool:
        return self.supports_text_generation() and _model_looks_vision_capable(model or self.config.answer_model)

    def _image_to_data_url(self, image_path: str | Path, max_side: int = 1600, quality: int = 85) -> str:
        if Image is None:
            raise RuntimeError("PIL is required for cropped_image VLM-2 mode.")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.width < 32 or image.height < 32:
                padded = Image.new("RGB", (max(32, image.width), max(32, image.height)), "white")
                padded.paste(image, ((padded.width - image.width) // 2, (padded.height - image.height) // 2))
                image = padded
            width, height = image.size
            scale = min(1.0, max_side / max(width, height))
            if scale < 1.0:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    def _with_images(self, messages: list[dict[str, Any]], image_paths: list[str | Path] | None, *, model: str | None = None) -> list[dict[str, Any]]:
        if not image_paths or not self.supports_image_input(model):
            return messages
        converted = [dict(message) for message in messages]
        last_user = max((idx for idx, message in enumerate(converted) if message.get("role") == "user"), default=-1)
        if last_user < 0:
            return converted
        content: list[dict[str, Any]] = [{"type": "text", "text": str(converted[last_user].get("content", ""))}]
        seen: set[str] = set()
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists() or str(path) in seen:
                continue
            seen.add(str(path))
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(path), "detail": "high"}})
        converted[last_user]["content"] = content
        return converted

    def generate_prediction(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[str | Path] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not self.supports_text_generation():
            raise RuntimeError("Answer VLM text generation is not configured.")
        url = self.config.answer_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model or self.config.answer_model,
            "messages": self._with_images(messages, image_paths, model=model),
            "temperature": self.config.answer_temperature if temperature is None else temperature,
            "max_tokens": self.config.answer_max_tokens if max_tokens is None else max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.answer_api_key}"}
        last_error: Exception | None = None
        rate_limit_retries = 0
        rate_limit_backoff = max(0.0, self.config.generation_429_initial_backoff_seconds)
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.answer_timeout_seconds if timeout_seconds is None else timeout_seconds,
                ) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"raw_response": data, "content": content}
            except urllib.error.HTTPError as exc:
                last_error = exc
                try:
                    response_detail = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    response_detail = ""
                if exc.code == 429 and self.config.generation_retry_on_429 and rate_limit_retries < self.config.generation_429_max_retries:
                    sleep_seconds = min(rate_limit_backoff, self.config.generation_429_max_backoff_seconds)
                    if self.config.generation_cooldown_after_429_seconds > 0:
                        sleep_seconds = max(sleep_seconds, self.config.generation_cooldown_after_429_seconds)
                    time.sleep(max(0.0, sleep_seconds))
                    rate_limit_retries += 1
                    rate_limit_backoff *= max(1.0, self.config.generation_429_backoff_multiplier)
                    continue
                if exc.code in {500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                kind = "rate_limit" if exc.code == 429 else "context_length" if exc.code in {400, 413} else "http"
                retryable = exc.code in {429, 500, 502, 503, 504}
                detail_suffix = f": {response_detail[:1000]}" if response_detail else ""
                raise AnswerVLMError(
                    f"Answer VLM call failed: HTTP Error {exc.code}: {exc.reason}{detail_suffix}",
                    kind=kind,
                    status_code=exc.code,
                    retryable=retryable,
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise AnswerVLMError(f"Answer VLM call failed: timeout: {exc}", kind="timeout", retryable=True) from exc
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise AnswerVLMError(f"Answer VLM call failed: {last_error}", kind="unknown", retryable=False) from last_error
        raise AnswerVLMError(f"Answer VLM call failed: {last_error}", kind="unknown", retryable=False) from last_error


def probe_text_json(model: str, api_key: str | None, base_url: str, timeout_seconds: float = 30) -> dict[str, Any]:
    if not is_api_key_configured(api_key):
        return {"model": model, "can_generate": False, "reason": "API key missing or placeholder."}
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": '{"ok": true}'}],
        "temperature": 0,
        "max_tokens": 32,
    }
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return {"model": model, "can_generate": bool(content.strip()), "content": content, "raw_response": data}
    except Exception as exc:
        return {"model": model, "can_generate": False, "reason": str(exc)}
