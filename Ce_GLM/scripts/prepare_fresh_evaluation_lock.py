#!/usr/bin/env python3
"""Prepare a hash-locked, single-use evaluator contract for a frozen fresh prediction.

This utility never invokes the evaluator and never reads or prints gold contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}")
    return resolved


def jsonl_query_id_count(path: Path) -> int:
    count = 0
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            if not query_id or query_id in identifiers:
                raise ValueError(f"Prediction has invalid query IDs at line {number}")
            identifiers.add(query_id)
            count += 1
    return count


def write_atomic(path: Path, value: dict[str, object]) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()

    prediction = require_file(args.prediction, "prediction")
    gold = require_file(args.gold, "gold")
    evaluator = require_file(args.evaluator, "evaluator")
    python = require_file(args.python, "Python executable")
    result = args.result.expanduser().resolve()
    lock = args.lock.expanduser().resolve()
    if result.exists() or lock.exists():
        raise FileExistsError("Refusing to overwrite an evaluation result or lock")
    count = jsonl_query_id_count(prediction)
    if count != 55:
        raise ValueError("Fresh prediction must contain exactly 55 unique query IDs")
    value: dict[str, object] = {
        "schema_version": 1,
        "prediction_path": str(prediction),
        "prediction_sha256": sha256(prediction),
        "gold_path": str(gold),
        "gold_sha256": sha256(gold),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256(evaluator),
        "python_executable": str(python),
        "authoritative_result_path": str(result),
        "record_count": count,
    }
    write_atomic(lock, value)
    print(json.dumps({"status": "LOCK_PREPARED", "record_count": count, "lock": str(lock)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
