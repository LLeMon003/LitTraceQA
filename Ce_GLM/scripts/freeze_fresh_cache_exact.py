#!/usr/bin/env python3
"""Freeze a completed fresh 55-record prediction and replay it exactly.

This utility never generates answers, contacts a provider, reads official gold,
or invokes an evaluator.  Its cache is a record of *one fresh result*, not a
route to the separate historical cache-exact diagnostic score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def profile_prediction(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction: {path}")
    identifiers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"prediction has invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict) or not str(row.get("query_id") or "").strip():
                raise ValueError(f"prediction has no query_id at line {line_number}")
            identifiers.append(str(row["query_id"]))
    if len(identifiers) != 55 or len(set(identifiers)) != 55:
        raise ValueError("prediction must contain exactly 55 unique query IDs")
    return {"records": len(identifiers), "unique_query_ids": len(set(identifiers))}


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze(prediction: Path, cache_dir: Path) -> dict[str, Any]:
    source = prediction.resolve()
    cache = cache_dir.resolve()
    profile = profile_prediction(source)
    if cache.exists():
        if any(cache.iterdir()):
            raise FileExistsError("refusing to overwrite a fresh cache directory")
    else:
        cache.mkdir(parents=True)
    cached_prediction = cache / "prediction.jsonl"
    copy_atomic(source, cached_prediction)
    source_hash = sha256(source)
    if sha256(cached_prediction) != source_hash:
        raise RuntimeError("fresh cache copy hash mismatch")
    manifest = {
        "schema_version": 1,
        "classification": "FRESH_FROZEN_CACHE_EXACT_REPLAY",
        "generation_claim": "exact replay of this frozen fresh prediction only; not historical cache-exact diagnostic replay",
        "prediction_sha256": source_hash,
        **profile,
        "provider_calls": 0,
        "gold_used": False,
        "evaluator_invoked": False,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_json_atomic(cache / "manifest.json", manifest)
    return {"status": "frozen", **{key: manifest[key] for key in ("classification", "prediction_sha256", "records", "unique_query_ids")}}


def replay(cache_dir: Path, output: Path) -> dict[str, Any]:
    cache = cache_dir.resolve()
    manifest_path = cache / "manifest.json"
    cached_prediction = cache / "prediction.jsonl"
    if not manifest_path.is_file() or not cached_prediction.is_file():
        raise FileNotFoundError("fresh cache requires manifest.json and prediction.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classification") != "FRESH_FROZEN_CACHE_EXACT_REPLAY":
        raise ValueError("cache does not declare the fresh frozen replay contract")
    profile = profile_prediction(cached_prediction)
    cached_hash = sha256(cached_prediction)
    if cached_hash != str(manifest.get("prediction_sha256") or "") or any(profile[key] != manifest.get(key) for key in profile):
        raise ValueError("fresh cache manifest does not match its prediction")
    destination = output.resolve()
    if destination.exists():
        raise FileExistsError("refusing to overwrite replay output")
    copy_atomic(cached_prediction, destination)
    if sha256(destination) != cached_hash:
        raise RuntimeError("fresh cache replay hash mismatch")
    receipt = {
        "schema_version": 1,
        "classification": "FRESH_FROZEN_CACHE_EXACT_REPLAY",
        "prediction_sha256": cached_hash,
        **profile,
        "provider_calls": 0,
        "gold_used": False,
        "evaluator_invoked": False,
    }
    write_json_atomic(destination.with_suffix(destination.suffix + ".replay.json"), receipt)
    return {"status": "replayed", **{key: receipt[key] for key in ("classification", "prediction_sha256", "records", "unique_query_ids")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--prediction", required=True, type=Path)
    freeze_parser.add_argument("--cache-dir", required=True, type=Path)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--cache-dir", required=True, type=Path)
    replay_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = freeze(args.prediction, args.cache_dir) if args.command == "freeze" else replay(args.cache_dir, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
