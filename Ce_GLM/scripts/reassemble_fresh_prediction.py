#!/usr/bin/env python3
"""Offline schema-safe final assembly for the completed Phase 7 fresh run.

This script does not call APIs, does not read gold/evaluator labels, and does
not alter raw generation.  It rebuilds the final prediction from existing fresh
stage artifacts using the evaluator-compatible prediction contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_INPUT_NAME_PARTS = ("checkpoint", "gold", "target")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def by_unique_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("query_id") or "")
        if not qid:
            raise ValueError(f"{label} contains a row without query_id")
        if qid in out:
            raise ValueError(f"{label} contains duplicate query_id {qid}")
        out[qid] = row
    return out


def answer_dict(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    return copy.deepcopy(answer) if isinstance(answer, dict) else {}


def first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return copy.deepcopy(value)
    return []


def contains_forbidden_gold_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lower = str(key).lower()
            if lower in {"answer_key", "expected_option", "expected_option_letter", "evaluator_label"}:
                return True
            if contains_forbidden_gold_key(item):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_gold_key(item) for item in value)
    return False


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    input_rows = read_jsonl(args.original_input)
    raw_rows = read_jsonl(args.raw_stage)
    evidence_rows = read_jsonl(args.evidence_stage)
    source_rows = read_jsonl(args.source_stage)
    typed_rows = read_jsonl(args.typed_stage)

    if contains_forbidden_gold_key(input_rows):
        raise ValueError("Original input contains forbidden evaluator/gold label keys")

    input_ids = [str(row["query_id"]) for row in input_rows]
    raw_by_id = by_unique_id(raw_rows, "raw stage")
    evidence_by_id = by_unique_id(evidence_rows, "evidence stage")
    source_by_id = by_unique_id(source_rows, "source-grounded stage")
    typed_by_id = by_unique_id(typed_rows, "typed stage")

    expected = set(input_ids)
    for label, mapping in (
        ("raw stage", raw_by_id),
        ("evidence stage", evidence_by_id),
        ("source-grounded stage", source_by_id),
        ("typed stage", typed_by_id),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{label} IDs do not match original input")

    assembled: list[dict[str, Any]] = []
    restored = {
        "papers_from_typed": 0,
        "evidence_from_typed": 0,
        "freeform_from_typed": 0,
        "table_from_typed": 0,
        "mc_from_typed": 0,
        "fallback_answer_fields_from_raw": 0,
    }

    for qid in input_ids:
        raw = raw_by_id[qid]
        evidence = evidence_by_id[qid]
        source = source_by_id[qid]
        typed = typed_by_id[qid]

        # Canonical evaluator-compatible prediction shell.
        record: dict[str, Any] = {
            "query_id": qid,
            "gold_papers": first_list(typed.get("gold_papers"), source.get("gold_papers"), evidence.get("gold_papers"), raw.get("gold_papers")),
            "evidence": first_list(typed.get("evidence"), source.get("evidence"), evidence.get("evidence"), raw.get("evidence")),
            "answer": {},
        }
        if record["gold_papers"]:
            restored["papers_from_typed"] += 1
        if record["evidence"]:
            restored["evidence_from_typed"] += 1

        raw_answer = answer_dict(raw)
        typed_answer = answer_dict(typed)
        final_answer: dict[str, Any] = {}

        for key in ("freeform", "table"):
            if key in typed_answer:
                final_answer[key] = copy.deepcopy(typed_answer[key])
            elif key in raw_answer:
                final_answer[key] = copy.deepcopy(raw_answer[key])
                restored["fallback_answer_fields_from_raw"] += 1
        if isinstance(final_answer.get("table"), dict):
            input_table_schema = raw_by_id.get(qid, {}).get("table_schema")
            original_input_row = next(row for row in input_rows if str(row["query_id"]) == qid)
            input_table_schema = original_input_row.get("table_schema")
            if "schema" not in final_answer["table"] and isinstance(input_table_schema, list):
                final_answer["table"]["schema"] = copy.deepcopy(input_table_schema)

        if "multiple_choice" in typed_answer:
            # Update MC as one answer subfield without replacing unrelated
            # answer types.
            final_answer["multiple_choice"] = copy.deepcopy(typed_answer["multiple_choice"])
        elif "multiple_choice" in raw_answer:
            final_answer["multiple_choice"] = copy.deepcopy(raw_answer["multiple_choice"])
            restored["fallback_answer_fields_from_raw"] += 1

        if "freeform" in final_answer:
            restored["freeform_from_typed"] += 1
        if "table" in final_answer:
            restored["table_from_typed"] += 1
        if "multiple_choice" in final_answer:
            restored["mc_from_typed"] += 1

        record["answer"] = final_answer
        assembled.append(record)

    write_jsonl(args.output, assembled)
    return {
        "output": str(args.output),
        "sha256": sha256(args.output),
        "records": len(assembled),
        "unique_query_ids": len({row["query_id"] for row in assembled}),
        "input_order_preserved": [row["query_id"] for row in assembled] == input_ids,
        "restored_counts": restored,
        "input_hashes": {
            "original_input": sha256(args.original_input),
            "raw_stage": sha256(args.raw_stage),
            "evidence_stage": sha256(args.evidence_stage),
            "source_stage": sha256(args.source_stage),
            "typed_stage": sha256(args.typed_stage),
        },
        "gold_used": False,
        "api_calls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-input", required=True, type=Path)
    parser.add_argument("--raw-stage", required=True, type=Path)
    parser.add_argument("--evidence-stage", required=True, type=Path)
    parser.add_argument("--source-stage", required=True, type=Path)
    parser.add_argument("--typed-stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.original_input, args.raw_stage, args.evidence_stage, args.source_stage, args.typed_stage):
        lowered = str(path).lower()
        if any(part in lowered for part in FORBIDDEN_INPUT_NAME_PARTS):
            raise ValueError(f"Refusing forbidden assembly input path: {path}")
    summary = assemble(args)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
