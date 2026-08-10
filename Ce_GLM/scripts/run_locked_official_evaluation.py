#!/usr/bin/env python3
"""Hash-locked, contract-checked single LitTraceQA evaluation wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ContractError(f"invalid JSONL at physical line {number}: {type(exc).__name__}") from None
            if not isinstance(row, dict):
                raise ContractError(f"non-object JSONL record at physical line {number}")
            rows.append(row)
    return rows


def require_hash(label: str, path: Path, expected: str) -> str:
    if not path.is_file():
        raise ContractError(f"{label} does not exist")
    actual = sha256(path)
    if actual != expected.upper():
        raise ContractError(f"{label} hash mismatch")
    return actual


def _ids(label: str, rows: list[dict[str, Any]], count: int) -> list[str]:
    if len(rows) != count:
        raise ContractError(f"{label} record-count mismatch")
    ids = [str(row.get("query_id") or "").strip() for row in rows]
    if any(not query_id for query_id in ids):
        raise ContractError(f"{label} contains a missing query ID")
    duplicates = [query_id for query_id, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ContractError(f"{label} contains duplicate query IDs")
    return ids


def validate_contract(lock: dict[str, Any]) -> dict[str, Any]:
    prediction = Path(lock["prediction_path"])
    gold = Path(lock["gold_path"])
    evaluator = Path(lock["evaluator_path"])
    count = int(lock["record_count"])

    hashes = {
        "prediction": require_hash("prediction", prediction, lock["prediction_sha256"]),
        "gold": require_hash("gold", gold, lock["gold_sha256"]),
        "evaluator": require_hash("evaluator", evaluator, lock["evaluator_sha256"]),
    }
    if prediction.resolve() == gold.resolve() or hashes["prediction"] == hashes["gold"]:
        raise ContractError("prediction checkpoint rejected as gold")

    gold_rows = load_jsonl(gold)
    prediction_rows = load_jsonl(prediction)
    gold_ids = _ids("gold", gold_rows, count)
    prediction_ids = _ids("prediction", prediction_rows, count)
    if set(gold_ids) != set(prediction_ids):
        raise ContractError("query-ID mismatch")

    gold_by_id = {str(row["query_id"]): row for row in gold_rows}
    pred_by_id = {str(row["query_id"]): row for row in prediction_rows}
    family_counts = Counter()
    for query_id, gold_row in gold_by_id.items():
        if not isinstance(gold_row.get("answer_types"), list):
            raise ContractError("gold lacks answer_types; raw input or prediction checkpoint rejected")
        if not isinstance(gold_row.get("answer"), dict):
            raise ContractError("gold lacks answer-reference fields")
        if not isinstance(gold_row.get("gold_papers"), list):
            raise ContractError("gold lacks paper-reference fields")
        if not isinstance(gold_row.get("evidence"), list):
            raise ContractError("gold lacks evidence-reference fields")
        pred_row = pred_by_id[query_id]
        if not isinstance(pred_row.get("papers"), list) and not isinstance(pred_row.get("gold_papers"), list):
            raise ContractError("prediction lacks paper selections")
        if not isinstance(pred_row.get("evidence"), list):
            raise ContractError("prediction lacks evidence")
        pred_answer = pred_row.get("answer")
        if not isinstance(pred_answer, dict):
            raise ContractError("prediction lacks answer object")
        gold_answer = gold_row["answer"]
        for family in gold_row["answer_types"]:
            family_counts[str(family)] += 1
            if family not in gold_answer:
                raise ContractError(f"gold lacks {family} answer reference")
            if family not in pred_answer:
                raise ContractError(f"prediction lacks {family} answer field")
        if "table" in gold_row["answer_types"]:
            gold_table = gold_answer.get("table")
            pred_table = pred_answer.get("table")
            if not isinstance(gold_table, dict) or not isinstance(gold_table.get("schema"), list) or not isinstance(gold_table.get("rows"), list):
                raise ContractError("gold table contract mismatch")
            if not isinstance(pred_table, dict) or not isinstance(pred_table.get("rows"), list):
                raise ContractError("prediction table contract mismatch")
            if "schema" in pred_table and not isinstance(pred_table["schema"], list):
                raise ContractError("prediction table schema must be a list when present")

    return {
        "record_count": count,
        "unique_query_ids": len(set(gold_ids)),
        "query_id_alignment": True,
        "answer_type_counts": dict(sorted(family_counts.items())),
        "hashes": hashes,
        "contract_verified": True,
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    contract = validate_contract(lock)
    if args.check_only:
        print(json.dumps({"status": "CONTRACT_VALID", **contract}, indent=2))
        return 0

    output = Path(lock["authoritative_result_path"])
    if output.exists():
        raise ContractError("authoritative result already exists; repeat evaluation refused")
    prediction = Path(lock["prediction_path"])
    gold = Path(lock["gold_path"])
    evaluator = Path(lock["evaluator_path"])
    python = Path(lock["python_executable"])
    command = [str(python), str(evaluator), "--gold", str(gold), "--pred", str(prediction)]
    before = {key: sha256(Path(lock[f"{key}_path"])) for key in ("prediction", "gold", "evaluator")}
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=300, check=False)
    runtime = time.monotonic() - started
    after = {key: sha256(Path(lock[f"{key}_path"])) for key in ("prediction", "gold", "evaluator")}
    if before != after:
        raise ContractError("an immutable evaluation input changed during execution")
    if process.returncode != 0:
        raise RuntimeError(f"official evaluator failed with return code {process.returncode}")
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("official evaluator returned invalid JSON") from None
    envelope = {
        "status": "AUTHORITATIVE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "invocation_count": 1,
        "command": command,
        "runtime_seconds": runtime,
        "prediction_path": str(prediction),
        "prediction_sha256": after["prediction"],
        "gold_path": str(gold),
        "gold_sha256": after["gold"],
        "evaluator_path": str(evaluator),
        "evaluator_sha256": after["evaluator"],
        "contract": contract,
        "result": result,
        "stderr_empty": not bool(process.stderr.strip()),
        "inputs_unchanged": True,
    }
    write_atomic(output, envelope)
    metrics = result.get("metrics", {})
    print(json.dumps({"status": "AUTHORITATIVE", "result_path": str(output), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
