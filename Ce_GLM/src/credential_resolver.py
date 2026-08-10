"""Central, secret-safe SiliconFlow credential and configuration resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CREDENTIAL_NAMES = ("SILICONFLOW_API_KEY", "SILICONFLOW_TOKEN", "SILICONFLOW_KEY")
DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class CredentialUnavailable(RuntimeError):
    """Non-secret diagnostic for unavailable provider credentials."""


@dataclass(frozen=True)
class Credential:
    value: str
    variable_name: str
    source: str

    def status(self) -> dict[str, str | bool]:
        return {"credential_present": True, "variable_name": self.variable_name, "source": self.source}

    def __repr__(self) -> str:
        return f"Credential(variable_name={self.variable_name!r}, source={self.source!r})"


@dataclass(frozen=True)
class ProviderConfig:
    endpoint: str
    model: str
    credential: Credential

    def status(self) -> dict[str, str | bool]:
        return {**self.credential.status(), "endpoint": self.endpoint, "model": self.model}


def _clean(value: object) -> str:
    text = str(value).lstrip("\ufeff").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _clean(value)
    return values


def _first_credential(values: Mapping[str, object]) -> tuple[str, str] | None:
    for name in CREDENTIAL_NAMES:
        value = _clean(values.get(name, ""))
        if value:
            return name, value
    return None


def _sources(environ: Mapping[str, str], home: Path) -> list[tuple[str, Mapping[str, str]]]:
    result: list[tuple[str, Mapping[str, str]]] = [("process_environment", environ)]
    pointer = _clean(environ.get("LITTRACEQA_SECRETS_FILE", ""))
    if pointer:
        result.append(("configured_secret_file", _read_env_file(Path(pointer))))
    result.append(("default_user_secret_file", _read_env_file(home / ".littraceqa" / "secrets.env")))
    return result


def resolve_credential(*, environ: Mapping[str, str] | None = None, home: Path | None = None) -> Credential:
    values = os.environ if environ is None else environ
    user_home = Path.home() if home is None else home
    for source, candidate in _sources(values, user_home):
        found = _first_credential(candidate)
        if found:
            return Credential(value=found[1], variable_name=found[0], source=source)
    raise CredentialUnavailable(
        "SiliconFlow credential unavailable; checked process environment, configured secret file, and default user secret file"
    )


def resolve_provider_config(*, environ: Mapping[str, str] | None = None, home: Path | None = None) -> ProviderConfig:
    values = os.environ if environ is None else environ
    user_home = Path.home() if home is None else home
    source_values = dict(_sources(values, user_home))
    endpoint = _clean(values.get("SILICONFLOW_BASE_URL", "")) or _clean(source_values.get("configured_secret_file", {}).get("SILICONFLOW_BASE_URL", "")) or DEFAULT_ENDPOINT
    model = _clean(values.get("SILICONFLOW_TEXT_MODEL", values.get("SILICONFLOW_MODEL", ""))) or _clean(source_values.get("configured_secret_file", {}).get("SILICONFLOW_TEXT_MODEL", "")) or DEFAULT_MODEL
    return ProviderConfig(endpoint=endpoint, model=model, credential=resolve_credential(environ=values, home=user_home))
