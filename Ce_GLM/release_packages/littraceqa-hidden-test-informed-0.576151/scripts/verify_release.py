#!/usr/bin/env python3
"""Independently verify release hashes, composition, and optional official schema."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import build


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_snapshot(snapshot: Path, manifest: dict) -> None:
    for relative, expected in manifest["official_snapshot"]["files"].items():
        path = snapshot / relative
        actual = build.sha256(path)
        if actual != expected:
            raise AssertionError(f"Official snapshot hash mismatch for {relative}: {actual} != {expected}")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def index(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = build.read_jsonl(path)
    ids = [str(row.get("query_id") or "") for row in rows]
    if len(ids) != 71 or len(set(ids)) != 71:
        raise AssertionError(f"Expected 71 unique query IDs: {path}")
    return ids, dict(zip(ids, rows, strict=True))


def verify_composition(prediction: Path, paths: dict[str, Path]) -> int:
    parent_ids, parent = index(paths["parent"])
    output_ids, output = index(prediction)
    if output_ids != parent_ids:
        raise AssertionError("Prediction query order differs from frozen B")

    components: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("mc", "paper", "evidence", "table"):
        ids, rows = index(paths[name])
        if ids != parent_ids:
            raise AssertionError(f"{name} query order differs from frozen B")
        components[name] = rows

    allowed_top_level = {"gold_papers", "evidence", "answer"}
    for query_id in parent_ids:
        base = parent[query_id]
        row = output[query_id]
        for key in set(base) | set(row):
            if key not in allowed_top_level and canonical(base.get(key)) != canonical(row.get(key)):
                raise AssertionError(f"Unexpected top-level drift: {query_id}/{key}")

        if canonical(row.get("gold_papers")) != canonical(components["paper"][query_id].get("gold_papers")):
            raise AssertionError(f"Paper component mismatch: {query_id}")
        if canonical(row.get("evidence")) != canonical(components["evidence"][query_id].get("evidence")):
            raise AssertionError(f"Evidence component mismatch: {query_id}")

        expected_answer = dict(base.get("answer") or {})
        mc_answer = (components["mc"][query_id].get("answer") or {}).get("multiple_choice")
        table_answer = (components["table"][query_id].get("answer") or {}).get("table")
        if mc_answer is not None:
            expected_answer["multiple_choice"] = mc_answer
        if table_answer is not None:
            expected_answer["table"] = table_answer
        if canonical(row.get("answer") or {}) != canonical(expected_answer):
            raise AssertionError(f"Answer component mismatch: {query_id}")

        selected_papers = {
            item.get("paper_id")
            for item in row.get("gold_papers") or []
            if isinstance(item, dict)
        }
        if any(
            item.get("paper_id") not in selected_papers
            for item in row.get("evidence") or []
            if isinstance(item, dict)
        ):
            raise AssertionError(f"Evidence containment failure: {query_id}")
    return len(parent_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default="scored_output/test_predictions.jsonl")
    parser.add_argument("--official-snapshot", help="Optional directory created by fetch_official_snapshot.py")
    args = parser.parse_args()

    manifest = build.load_manifest()
    paths = build.artifact_paths(manifest)
    build.verify_frozen_hashes(manifest, paths)

    prediction = resolve(args.prediction)
    actual_hash = build.sha256(prediction)
    expected_hash = manifest["reproduction"]["expected_sha256"]
    if actual_hash != expected_hash:
        raise AssertionError(f"Prediction hash mismatch: {actual_hash} != {expected_hash}")

    records_checked = verify_composition(prediction, paths)

    official_validation = "not requested"
    if args.official_snapshot:
        snapshot = resolve(args.official_snapshot)
        verify_snapshot(snapshot, manifest)
        command = [
            sys.executable,
            str(snapshot / "scripts" / "validate_submission.py"),
            "--input",
            str(snapshot / "data" / "test.jsonl"),
            "--pred",
            str(prediction),
            "--paper-metadata",
            str(snapshot / "data" / "paper_metadata.jsonl"),
            "--require-evidence",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        official_validation = completed.stdout.strip()

    report = {
        "status": "PASS",
        "records_checked": records_checked,
        "prediction_sha256": actual_hash,
        "checks": [
            "all frozen artifact hashes match",
            "71 unique IDs preserve frozen B order",
            "only designated component fields are composed",
            "evidence is contained in the selected paper set",
            "prediction is byte-identical to the scored submission",
        ],
        "official_validation": official_validation,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
