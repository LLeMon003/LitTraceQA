"""Contract checks for the score-prioritized fresh profile."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profiles" / "fresh_authoritative_v4.yaml"


def profile_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def test_fresh_authoritative_profile_is_explicit_and_fail_closed() -> None:
    values = profile_values()

    assert values["profile"] == "FRESH_AUTHORITATIVE_V4"
    assert values["mode"] == "raw-fresh"
    assert values["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert values["temperature"] == "0"
    assert "no resume" in values["fresh_boundary"]
    assert "no sealed cache" in values["fresh_boundary"]
    assert "gold" in values["fresh_boundary"]
    assert "evaluator" in values["fresh_boundary"]
    assert "none" in values["score_guarantee"]
