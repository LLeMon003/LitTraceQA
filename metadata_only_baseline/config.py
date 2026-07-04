from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


@dataclass(frozen=True)
class Config:
    env_path: Path
    env_exists: bool
    api_key: str | None
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_no}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_config(env_path: str | Path = ".env") -> Config:
    path = Path(env_path)
    env_values: dict[str, str] = {}
    if path.exists():
        try:
            from dotenv import dotenv_values  # type: ignore

            env_values = {k: v for k, v in dotenv_values(path).items() if k and v is not None}
        except Exception:
            env_values = _parse_env_file(path)

    def get(name: str, default: str | None = None) -> str | None:
        return os.environ.get(name) or env_values.get(name) or default

    return Config(
        env_path=path,
        env_exists=path.exists(),
        api_key=get("SILICONFLOW_API_KEY"),
        base_url=get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=get("SILICONFLOW_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        temperature=float(get("SILICONFLOW_TEMPERATURE", "0") or "0"),
        max_tokens=int(get("SILICONFLOW_MAX_TOKENS", "3000") or "3000"),
        timeout_seconds=float(get("SILICONFLOW_TIMEOUT_SECONDS", "120") or "120"),
    )


def mask_api_key(key: str) -> str:
    if not key:
        return "<missing>"
    if len(key) <= 8:
        return f"{key[:2]}****"
    return f"{key[:3]}****{key[-4:]}"


def is_api_key_configured(key: str | None) -> bool:
    if key is None:
        return False
    value = key.strip()
    if not value:
        return False
    lowered = value.lower()
    placeholders = {
        "...",
        "your_key",
        "your_api_key",
        "sk-xxx",
        "replace_me",
        "put_your_new_key_here",
        "put-your-key-here",
        "api_key",
        "none",
        "null",
    }
    if lowered in placeholders:
        return False
    if "your" in lowered and "key" in lowered:
        return False
    if "replace" in lowered or "placeholder" in lowered:
        return False
    if lowered.startswith("sk-") and set(lowered[3:]) <= {"x", "*"}:
        return False
    return len(value) >= 12

