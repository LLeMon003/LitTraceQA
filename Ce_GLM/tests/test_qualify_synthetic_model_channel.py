"""Tests for secret-safe synthetic model-channel qualification parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthetic_qualifier", ROOT / "scripts" / "qualify_synthetic_model_channel.py"
)
assert SPEC and SPEC.loader
QUALIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFIER)


def test_valid_synthetic_response_is_accepted_without_exposing_content() -> None:
    status, size = QUALIFIER.validate_response(
        '{"value":"17","unit":"m/s","quote":"the calibrated release value is 17 m/s"}'
    )

    assert status == "PASS"
    assert size > 0

    numeric_status, _ = QUALIFIER.validate_response(
        '{"value":17,"unit":"m/s","quote":"the calibrated release value is 17 m/s"}'
    )
    assert numeric_status == "PASS"


def test_qualification_fails_closed_for_malformed_or_unsupported_values() -> None:
    assert QUALIFIER.validate_response("not-json")[0] == "MALFORMED_JSON"
    assert QUALIFIER.validate_response('{"value":"18","unit":"m/s","quote":"17 m/s"}')[0] == "GROUNDING_REJECTED"
    assert QUALIFIER.validate_response('{"value":"17","unit":"m/s","quote":"not in source"}')[0] == "GROUNDING_REJECTED"


def test_qualification_disables_sdk_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(QUALIFIER, "OpenAI", FakeClient)
    config = SimpleNamespace(credential=SimpleNamespace(value="test-only"), endpoint="https://example.invalid/v1")

    QUALIFIER.make_client(config, 90)

    assert captured["timeout"] == 90
    assert captured["max_retries"] == 0
