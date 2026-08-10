#!/usr/bin/env python3
"""Wait for one raw-fresh run, then prepare and execute one locked evaluation.

The watcher intentionally neither reads prediction content nor opens the gold
file.  The existing lock preparer and evaluator wrapper own those contracts.
Its compact status file never contains provider output, credentials, or raw
evaluator stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path: Path, value: dict[str, object]) -> None:
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


def load_run_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run manifest unavailable: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise RuntimeError("run manifest is not an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def prediction_id_profile(path: Path) -> tuple[int, int] | None:
    """Return record and unique-ID counts without reading answer content."""
    identifiers: set[str] = set()
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return None
                query_id = str(row.get("query_id") or "").strip()
                if not query_id or query_id in identifiers:
                    return None
                identifiers.add(query_id)
                count += 1
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return count, len(identifiers)


def frozen_replay_is_valid(prediction: Path) -> bool:
    """Return true only for the receipt emitted by the fresh replay utility."""
    receipt_path = prediction.with_suffix(prediction.suffix + ".replay.json")
    if not prediction.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    profile = prediction_id_profile(prediction)
    return (
        isinstance(receipt, dict)
        and receipt.get("classification") == "FRESH_FROZEN_CACHE_EXACT_REPLAY"
        and receipt.get("prediction_sha256") == sha256(prediction)
        and receipt.get("records") == 55
        and receipt.get("unique_query_ids") == 55
        and receipt.get("provider_calls") == 0
        and receipt.get("gold_used") is False
        and receipt.get("evaluator_invoked") is False
        and profile == (55, 55)
    )


def run_checked(command: list[str], label: str) -> None:
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if process.returncode:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument(
        "--frozen-replay",
        type=Path,
        help="Evaluate this verified fresh freeze/replay output after the raw run completes.",
    )
    parser.add_argument(
        "--freeze-cache-dir",
        type=Path,
        help="Freeze and replay the completed raw prediction here before evaluation; requires --frozen-replay.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.freeze_cache_dir is not None and args.frozen_replay is None:
        parser.error("--freeze-cache-dir requires --frozen-replay")
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    prediction = args.frozen_replay.resolve() if args.frozen_replay else run_root / "predictions.jsonl"
    freeze_cache = args.freeze_cache_dir.resolve() if args.freeze_cache_dir else None
    status_path = run_root / "fresh_evaluation_watcher_status.json"
    result = run_root / "authoritative_fresh_evaluation.json"
    lock = run_root / "fresh_evaluation.lock.json"
    if status_path.exists() or result.exists() or lock.exists():
        raise FileExistsError("watcher refuses an existing status, result, or lock")
    preparer = repo_root / "scripts" / "prepare_fresh_evaluation_lock.py"
    evaluator_runner = repo_root / "scripts" / "run_locked_official_evaluation.py"
    freezer = repo_root / "scripts" / "freeze_fresh_cache_exact.py"
    for required in (args.python, preparer, evaluator_runner, freezer, args.gold, args.evaluator):
        if not required.is_file():
            raise FileNotFoundError("required watcher input is missing")

    write_atomic(status_path, {"status": "WAITING_FOR_RAW_FRESH", "updated_at_utc": utc_now()})
    while True:
        manifest = load_run_manifest(run_root / "run_manifest.json")
        state = str(manifest.get("status") or "")
        if state == "completed":
            if freeze_cache is not None:
                raw_prediction = run_root / "predictions.jsonl"
                cache_manifest = freeze_cache / "manifest.json"
                if not cache_manifest.is_file():
                    write_atomic(status_path, {"status": "FREEZING_FRESH_PREDICTION", "updated_at_utc": utc_now()})
                    run_checked(
                        [str(args.python), str(freezer), "freeze", "--prediction", str(raw_prediction), "--cache-dir", str(freeze_cache)],
                        "fresh prediction freeze",
                    )
                if not prediction.is_file():
                    write_atomic(status_path, {"status": "REPLAYING_FRESH_PREDICTION", "updated_at_utc": utc_now()})
                    run_checked(
                        [str(args.python), str(freezer), "replay", "--cache-dir", str(freeze_cache), "--output", str(prediction)],
                        "fresh prediction replay",
                    )
            if not prediction.is_file():
                if args.frozen_replay:
                    write_atomic(status_path, {"status": "WAITING_FOR_FROZEN_REPLAY", "updated_at_utc": utc_now()})
                    time.sleep(args.poll_seconds)
                    continue
                raise RuntimeError("completed raw-fresh run lacks final prediction")
            if args.frozen_replay and not frozen_replay_is_valid(prediction):
                write_atomic(status_path, {"status": "WAITING_FOR_VALID_FROZEN_REPLAY", "updated_at_utc": utc_now()})
                time.sleep(args.poll_seconds)
                continue
            write_atomic(status_path, {"status": "PREPARING_LOCK", "updated_at_utc": utc_now()})
            run_checked([
                str(args.python), str(preparer), "--prediction", str(prediction),
                "--gold", str(args.gold), "--evaluator", str(args.evaluator),
                "--python", str(args.python), "--result", str(result), "--lock", str(lock),
            ], "evaluation lock preparation")
            write_atomic(status_path, {"status": "RUNNING_SINGLE_EVALUATION", "updated_at_utc": utc_now()})
            run_checked([str(args.python), str(evaluator_runner), "--lock", str(lock)], "locked evaluation")
            write_atomic(status_path, {"status": "EVALUATION_COMPLETE", "updated_at_utc": utc_now()})
            return 0
        if state in {"failed", "completed_at_stop_stage"}:
            write_atomic(status_path, {"status": "RAW_FRESH_DID_NOT_COMPLETE", "raw_status": state, "updated_at_utc": utc_now()})
            return 2
        write_atomic(status_path, {"status": "WAITING_FOR_RAW_FRESH", "raw_status": state, "updated_at_utc": utc_now()})
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
