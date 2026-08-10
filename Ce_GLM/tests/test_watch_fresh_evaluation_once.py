"""Focused checks for the post-freeze evaluation watcher contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fresh_eval_watcher", ROOT / "scripts" / "watch_fresh_evaluation_once.py")
assert SPEC and SPEC.loader
WATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCHER)


def write_receipt(prediction: Path, *, digest: str | None = None) -> None:
    receipt = {
        "classification": "FRESH_FROZEN_CACHE_EXACT_REPLAY",
        "prediction_sha256": digest or WATCHER.sha256(prediction),
        "records": 55,
        "unique_query_ids": 55,
        "provider_calls": 0,
        "gold_used": False,
        "evaluator_invoked": False,
    }
    prediction.with_suffix(prediction.suffix + ".replay.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_frozen_replay_receipt_requires_exact_hash_and_contract(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(
        "".join(json.dumps({"query_id": f"synthetic-{index}"}) + "\n" for index in range(55)),
        encoding="utf-8",
    )
    write_receipt(prediction)

    assert WATCHER.frozen_replay_is_valid(prediction)

    write_receipt(prediction, digest="0" * 64)
    assert not WATCHER.frozen_replay_is_valid(prediction)


def test_frozen_replay_receipt_is_fail_closed_when_absent(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text('{"query_id":"synthetic"}\n', encoding="utf-8")

    assert not WATCHER.frozen_replay_is_valid(prediction)


def test_frozen_replay_receipt_rejects_non_55_record_prediction(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text('{"query_id":"synthetic"}\n', encoding="utf-8")
    write_receipt(prediction)

    assert not WATCHER.frozen_replay_is_valid(prediction)
