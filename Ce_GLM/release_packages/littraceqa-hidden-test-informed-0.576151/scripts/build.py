#!/usr/bin/env python3
"""Rebuild the exact 0.576151 LitTraceQA submission from frozen components."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def paper_set(row: dict[str, Any]) -> set[str]:
    return {
        str(item.get("paper_id") or "")
        for item in row.get("gold_papers") or []
        if isinstance(item, dict) and item.get("paper_id")
    }


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def artifact_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        name: ROOT / item["path"]
        for name, item in manifest["artifacts"].items()
    }


def verify_frozen_hashes(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        expected = manifest["artifacts"][name]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"Frozen artifact hash mismatch for {name}: {actual} != {expected}")


def indexed(path: Path, expected_ids: list[str] | None = None) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    ids = [str(row.get("query_id") or "") for row in rows]
    if len(ids) != 71 or len(set(ids)) != 71:
        raise AssertionError(f"Expected 71 unique query IDs: {path}")
    if expected_ids is not None and ids != expected_ids:
        raise AssertionError(f"Query ID/order drift: {path}")
    return ids, dict(zip(ids, rows, strict=True))


def compose(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    paths = artifact_paths(manifest)
    verify_frozen_hashes(manifest, paths)
    parent_rows = read_jsonl(paths["parent"])
    parent_ids = [str(row.get("query_id") or "") for row in parent_rows]
    if len(parent_ids) != 71 or len(set(parent_ids)) != 71:
        raise AssertionError("Frozen parent must contain 71 unique query IDs")

    components = {
        name: indexed(paths[name], parent_ids)[1]
        for name in ("mc", "paper", "evidence", "table")
    }
    combined: list[dict[str, Any]] = []
    for parent in parent_rows:
        query_id = str(parent["query_id"])
        row = copy.deepcopy(parent)

        mc_answer = components["mc"][query_id].get("answer") or {}
        if "multiple_choice" in mc_answer:
            row.setdefault("answer", {})["multiple_choice"] = copy.deepcopy(mc_answer["multiple_choice"])

        row["gold_papers"] = copy.deepcopy(components["paper"][query_id].get("gold_papers") or [])
        row["evidence"] = copy.deepcopy(components["evidence"][query_id].get("evidence") or [])

        table_answer = components["table"][query_id].get("answer") or {}
        if "table" in table_answer:
            row.setdefault("answer", {})["table"] = copy.deepcopy(table_answer["table"])

        selected = paper_set(row)
        outside = [
            item
            for item in row.get("evidence") or []
            if isinstance(item, dict) and item.get("paper_id") not in selected
        ]
        if outside:
            raise AssertionError(f"Evidence outside selected paper set for {query_id}: {canonical(outside)}")
        combined.append(row)
    return combined


def resolve_output(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="build/test_predictions.jsonl")
    args = parser.parse_args()

    manifest = load_manifest()
    output = resolve_output(args.output)
    write_jsonl(output, compose(manifest))
    actual = sha256(output)
    expected = manifest["reproduction"]["expected_sha256"]
    if actual != expected:
        raise AssertionError(f"Rebuilt output hash mismatch: {actual} != {expected}")
    print(json.dumps({"status": "PASS", "rows": 71, "output": str(output), "sha256": actual}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
