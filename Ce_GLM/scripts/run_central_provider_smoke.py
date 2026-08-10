"""Secret-safe SiliconFlow smoke utility using the Ver3 central resolver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src.credential_resolver import CredentialUnavailable, resolve_provider_config


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def client() -> tuple[OpenAI, dict[str, str | bool]]:
    config = resolve_provider_config()
    return OpenAI(api_key=config.credential.value, base_url=config.endpoint, timeout=30), config.status()


def run_models() -> None:
    api, status = client()
    api.models.list()
    emit({"status": "ok", "operation": "models", **status})


def run_tiny_json(snippet: str, output_dir: Path) -> None:
    api, status = client()
    prompt = (
        "Return JSON only with one key, label. Copy the exact release label from the supplied snippet. "
        "Do not add explanation.\nSnippet: " + snippet
    )
    response = api.chat.completions.create(
        model=str(status["model"]),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=40,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(raw)
    label = parsed.get("label")
    valid = isinstance(label, str) and label and label in snippet
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    (output_dir / "status.json").write_text(
        json.dumps({"status": "ok", "operation": "tiny_json", "structured_value_valid": valid, **status}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    emit({"status": "ok", "operation": "tiny_json", "structured_value_valid": valid, **status})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("status", "models", "tiny-json"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "status":
            emit({"status": "ok", "operation": "status", **resolve_provider_config().status()})
        elif args.mode == "models":
            run_models()
        else:
            if args.output_dir is None:
                parser.error("--output-dir is required for tiny-json")
            run_tiny_json("The release label is RT-17.", args.output_dir)
    except CredentialUnavailable as exc:
        emit({"status": "credential_unavailable", "operation": args.mode, "diagnostic": str(exc)})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
