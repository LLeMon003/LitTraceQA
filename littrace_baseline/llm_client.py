from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Config


class SiliconFlowClient:
    def __init__(self, config: Config, retries: int = 2) -> None:
        self.config = config
        self.retries = retries

    def chat_completions(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(content or ""), data
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"SiliconFlow chat completion failed after retries: {last_error}") from last_error

