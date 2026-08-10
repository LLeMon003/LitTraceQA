"""Run a secret-safe synthetic JSON/grounding qualification for one text model.

This utility never reads official inputs, gold, predictions, evaluator output,
or historical artifacts. It stores only a compact status envelope; raw provider
responses are intentionally discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.credential_resolver import CredentialUnavailable, resolve_provider_config


SNIPPET = "Synthetic source: the calibrated release value is 17 m/s."
PROMPT = (
    "Return a native JSON object only with keys value, unit, quote. Extract the exact "
    "value and unit from the source. quote must be an exact substring of the source. "
    f"Source: {SNIPPET}"
)


def validate_response(raw: str) -> tuple[str, int]:
    """Return the non-sensitive outcome and UTF-8 response size."""
    size = len(raw.encode("utf-8"))
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return "MALFORMED_JSON", size
    if not isinstance(value, dict):
        return "SCHEMA_REJECTED", size
    quote = value.get("quote")
    extracted_value = value.get("value")
    if (
        isinstance(extracted_value, bool)
        or str(extracted_value) != "17"
        or value.get("unit") != "m/s"
        or not isinstance(quote, str)
    ):
        return "GROUNDING_REJECTED", size
    return ("PASS" if quote in SNIPPET else "GROUNDING_REJECTED"), size


def make_client(config: Any, timeout_seconds: int) -> OpenAI:
    """Disable SDK retries so the caller's timeout remains a real bound."""
    return OpenAI(
        api_key=config.credential.value,
        base_url=config.endpoint,
        timeout=timeout_seconds,
        max_retries=0,
    )


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def qualify(model: str, timeout_seconds: int) -> dict[str, object]:
    started = time.monotonic()
    try:
        config = resolve_provider_config()
        client = make_client(config, timeout_seconds)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=64,
            response_format={"type": "json_object"},
        )
        status, response_bytes = validate_response(response.choices[0].message.content or "")
    except CredentialUnavailable:
        status, response_bytes = "CREDENTIAL_UNAVAILABLE", 0
    except Exception as exc:  # provider exceptions are intentionally classified without details.
        status, response_bytes = type(exc).__name__, 0
    return {
        "schema_version": 1,
        "classification": "SYNTHETIC_MODEL_CHANNEL_QUALIFICATION",
        "model": model,
        "status": status,
        "latency_seconds": round(time.monotonic() - started, 3),
        "response_bytes": response_bytes,
        "official_input_used": False,
        "gold_used": False,
        "evaluator_invoked": False,
        "raw_response_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing a nonempty qualification directory")
    output.mkdir(parents=True, exist_ok=True)
    result = qualify(args.model, args.timeout_seconds)
    write_atomic(output / "status.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
