from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from metadata_only_baseline.config import is_api_key_configured, load_config as load_base_config, mask_api_key


@dataclass(frozen=True)
class PipelineConfig:
    env_path: Path
    env_exists: bool
    ocr_model: str
    ocr_provider: str
    ocr_api_key: str | None
    ocr_base_url: str
    ocr_device: str
    ocr_max_pages_per_paper: int
    answer_provider: str
    answer_api_key: str | None
    answer_base_url: str
    answer_model: str
    base_generation_model: str
    answer_temperature: float
    answer_max_tokens: int
    answer_timeout_seconds: float


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_pipeline_config(env_path: str | Path = ".env") -> PipelineConfig:
    path = Path(env_path)
    env = _read_env(path)
    base = load_base_config(path)

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or env.get(name) or default

    base_generation_model = get("BASE_GENERATION_MODEL") or base.model
    answer_model = get("ANSWER_MODEL") or get("ANSWER_GENERATION_MODEL") or base_generation_model
    answer_key = get("ANSWER_API_KEY") or base.api_key
    answer_base = get("ANSWER_BASE_URL") or base.base_url
    ocr_key = get("OCR_API_KEY") or base.api_key
    ocr_base = get("OCR_BASE_URL") or base.base_url
    return PipelineConfig(
        env_path=path,
        env_exists=path.exists(),
        ocr_model=get("OCR_MODEL") or get("OCR_PROCESS_MODEL", "deepseek-ai/DeepSeek-OCR"),
        ocr_provider=get("OCR_PROVIDER", "siliconflow"),
        ocr_api_key=ocr_key,
        ocr_base_url=ocr_base,
        ocr_device=get("OCR_DEVICE", "auto"),
        ocr_max_pages_per_paper=int(get("OCR_MAX_PAGES_PER_PAPER", "0") or "0"),
        answer_provider=get("ANSWER_PROVIDER", "siliconflow"),
        answer_api_key=answer_key,
        answer_base_url=answer_base,
        answer_model=answer_model,
        base_generation_model=base_generation_model,
        answer_temperature=float(get("ANSWER_TEMPERATURE", "0") or "0"),
        answer_max_tokens=int(get("ANSWER_MAX_TOKENS", "3000") or "3000"),
        answer_timeout_seconds=float(get("ANSWER_TIMEOUT_SECONDS", "120") or "120"),
    )


__all__ = ["PipelineConfig", "is_api_key_configured", "load_pipeline_config", "mask_api_key"]
