from __future__ import annotations

import base64
import http.client
import io
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - text-only reranking does not need PIL
    Image = None  # type: ignore[assignment]


class LLMRerankError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


def _parse_scores(response: dict[str, Any], document_count: int) -> list[float]:
    scores: list[float | None] = [None] * document_count
    for result in response.get("results") or []:
        if not isinstance(result, dict):
            continue
        index = result.get("index")
        score = result.get("relevance_score")
        if isinstance(index, int) and 0 <= index < document_count and isinstance(score, (int, float)):
            scores[index] = float(score)
    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise ValueError(f"Rerank response omitted document indexes: {missing}")
    return [float(score) for score in scores if score is not None]


class LLMRerankClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float = 120,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _image_to_data_url(image_path: str | Path, max_side: int = 1600, quality: int = 85) -> str:
        if Image is None:
            raise RuntimeError("PIL is required for multimodal reranking.")
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

    def _prepare_document(self, document: dict[str, Any]) -> str | dict[str, Any]:
        text = str(document.get("text") or "")
        image_path = Path(str(document.get("image_path") or ""))
        if image_path.is_file():
            if text:
                raise ValueError("SiliconFlow /rerank does not support joint text+image document objects.")
            return {"image": self._image_to_data_url(image_path)}
        return text

    def score_documents(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        instruction: str,
    ) -> dict[str, Any]:
        if not self.api_key or self.api_key.lower() in {"your_api_key", "replace_me", "none"}:
            raise LLMRerankError("LLM reranker API key is not configured.", attempts=0)
        if not documents:
            return {"scores": [], "raw_response": {}, "attempts": 0}
        payload = {
            "model": self.model,
            "query": query,
            "documents": [self._prepare_document(document) for document in documents],
            "instruction": instruction,
            "top_n": len(documents),
            "return_documents": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        url = self.base_url.rstrip("/") + "/rerank"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                return {
                    "scores": _parse_scores(data, len(documents)),
                    "raw_response": data,
                    "attempts": attempt + 1,
                }
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise LLMRerankError(
                        f"LLM reranker call failed: HTTP {exc.code}: {exc.reason}",
                        attempts=attempt + 1,
                        status_code=exc.code,
                    ) from exc
            except (
                TimeoutError,
                socket.timeout,
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                OSError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMRerankError(
                        f"LLM reranker call failed: {exc}",
                        attempts=attempt + 1,
                    ) from exc
            time.sleep(min(8.0, 2.0**attempt))
        raise LLMRerankError(f"LLM reranker call failed: {last_error}", attempts=self.max_retries + 1)


__all__ = ["LLMRerankClient", "LLMRerankError", "_parse_scores"]
