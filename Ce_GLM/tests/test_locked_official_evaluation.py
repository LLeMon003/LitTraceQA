#!/usr/bin/env python3
"""Focused no-evaluator tests for the official evaluation lock."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "records" / "OFFICIAL_EVALUATION_LOCK.json"
if not LOCK_PATH.is_file():
    pytest.skip("requires the intentionally external official-evaluation lock fixture", allow_module_level=True)
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
SPEC = importlib.util.spec_from_file_location("locked_eval", ROOT / "scripts" / "run_locked_official_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def expect_error(lock: dict, fragment: str) -> None:
    try:
        MODULE.validate_contract(lock)
    except MODULE.ContractError as exc:
        assert fragment.lower() in str(exc).lower(), (fragment, str(exc))
    else:
        raise AssertionError(f"expected ContractError containing {fragment!r}")


def main() -> int:
    original_hashes = {key: hash_file(Path(LOCK[f"{key}_path"])) for key in ("prediction", "gold", "evaluator")}
    passed: list[str] = []

    result = MODULE.validate_contract(LOCK)
    assert result["contract_verified"] and result["record_count"] == 55
    passed.append("correct_official_gold_accepted")

    raw_lock = copy.deepcopy(LOCK)
    raw = ROOT.parent / "littraceqa_baseline_Ver.2" / "inputs" / "validation_inputs.jsonl"
    raw_lock["gold_path"] = str(raw)
    raw_lock["gold_sha256"] = hash_file(raw)
    expect_error(raw_lock, "gold lacks")
    passed.append("raw_validation_input_rejected")

    checkpoint_lock = copy.deepcopy(LOCK)
    checkpoint = ROOT / "outputs" / "cached_exact" / "predictions.jsonl"
    checkpoint_lock["gold_path"] = str(checkpoint)
    checkpoint_lock["gold_sha256"] = hash_file(checkpoint)
    expect_error(checkpoint_lock, "gold lacks")
    passed.append("prediction_checkpoint_rejected_as_gold")

    bad = copy.deepcopy(LOCK)
    bad["evaluator_sha256"] = "0" * 64
    expect_error(bad, "evaluator hash mismatch")
    passed.append("wrong_evaluator_hash_rejected")

    bad = copy.deepcopy(LOCK)
    bad["prediction_sha256"] = "0" * 64
    expect_error(bad, "prediction hash mismatch")
    passed.append("wrong_prediction_hash_rejected")

    with tempfile.TemporaryDirectory(prefix="v23_eval_contract_") as temp_name:
        temp = Path(temp_name)
        gold_rows = MODULE.load_jsonl(Path(LOCK["gold_path"]))
        prediction_rows = MODULE.load_jsonl(Path(LOCK["prediction_path"]))

        mismatch = copy.deepcopy(prediction_rows)
        mismatch[0]["query_id"] = "q_nonexistent"
        mismatch_path = temp / "mismatch.jsonl"
        write_jsonl(mismatch_path, mismatch)
        mismatch_lock = copy.deepcopy(LOCK)
        mismatch_lock["prediction_path"] = str(mismatch_path)
        mismatch_lock["prediction_sha256"] = hash_file(mismatch_path)
        expect_error(mismatch_lock, "query-ID mismatch")
        passed.append("query_id_mismatch_rejected")

        duplicate = copy.deepcopy(prediction_rows)
        duplicate[1]["query_id"] = duplicate[0]["query_id"]
        duplicate_path = temp / "duplicate.jsonl"
        write_jsonl(duplicate_path, duplicate)
        duplicate_lock = copy.deepcopy(LOCK)
        duplicate_lock["prediction_path"] = str(duplicate_path)
        duplicate_lock["prediction_sha256"] = hash_file(duplicate_path)
        expect_error(duplicate_lock, "duplicate query IDs")
        passed.append("duplicate_ids_rejected")

        missing_answer = copy.deepcopy(gold_rows)
        missing_answer[0].pop("answer")
        missing_path = temp / "missing_answer.jsonl"
        write_jsonl(missing_path, missing_answer)
        missing_lock = copy.deepcopy(LOCK)
        missing_lock["gold_path"] = str(missing_path)
        missing_lock["gold_sha256"] = hash_file(missing_path)
        expect_error(missing_lock, "answer-reference")
        passed.append("missing_answer_reference_fields_rejected")

        table_bad = copy.deepcopy(prediction_rows)
        table_id = next(str(row["query_id"]) for row in gold_rows if "table" in row["answer_types"])
        next(row for row in table_bad if str(row["query_id"]) == table_id)["answer"]["table"].pop("schema")
        table_path = temp / "table_bad.jsonl"
        write_jsonl(table_path, table_bad)
        table_lock = copy.deepcopy(LOCK)
        table_lock["prediction_path"] = str(table_path)
        table_lock["prediction_sha256"] = hash_file(table_path)
        expect_error(table_lock, "table contract mismatch")
        passed.append("table_contract_mismatch_rejected")

    final_hashes = {key: hash_file(Path(LOCK[f"{key}_path"])) for key in ("prediction", "gold", "evaluator")}
    assert original_hashes == final_hashes
    passed.append("no_input_file_modified")
    assert len(passed) == 10
    print(json.dumps({"status": "PASS", "tests": passed, "evaluator_invocations": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
