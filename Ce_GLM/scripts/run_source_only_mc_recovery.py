#!/usr/bin/env python3
"""Fill only blank MC fields from paper-constrained source objects.

This command never reads gold/evaluator files and deliberately ignores the
``query_ids`` field in the Docling index. Raw prompts and provider replies are
written only under the caller-supplied external artifact directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.credential_resolver import resolve_provider_config
from src.source_only_mc_recovery import blank_mc, contains_forbidden_key, parent_locators, proposal_status, rank_bundle, selected_paper_ids, validate_proposal


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_index_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read source objects while fail-closing malformed external index rows."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                malformed += 1
    return rows, malformed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def request_payload(question: str, options: dict[str, str], bundle: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "instruction": "Choose an option only when directly supported by the supplied source objects. Return JSON exactly with letter, citations, confidence. citations must contain only supplied object_id values.",
            "question": question,
            "options": options,
            "source_objects": bundle,
        },
        ensure_ascii=False,
    )


def call_model(api: OpenAI, model: str, payload: str, timeout: float) -> tuple[str, str | None]:
    for _ in range(2):
        try:
            response = api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return one JSON object only. Do not use knowledge outside the supplied source objects."},
                    {"role": "user", "content": payload},
                ],
                temperature=0,
                max_tokens=220,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
            return response.choices[0].message.content or "{}", None
        except Exception as exc:  # bounded infrastructure retry; never serialize exception text
            diagnostic = type(exc).__name__
    return "{}", diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--object-index", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_records is not None and args.max_records <= 0:
        raise ValueError("timeout and max-records must be positive")
    if args.artifact_dir.exists():
        raise FileExistsError("refusing to overwrite artifact directory")
    prediction_rows = read_jsonl(args.predictions)
    input_rows = read_jsonl(args.inputs)
    if any(contains_forbidden_key(row) for row in input_rows):
        raise ValueError("input contains forbidden evaluator/gold key")
    inputs = {str(row.get("query_id") or ""): row for row in input_rows}
    if len(prediction_rows) != 55 or len({str(row.get("query_id") or "") for row in prediction_rows}) != 55:
        raise ValueError("predictions must contain 55 unique query IDs")
    candidates = [row for row in prediction_rows if blank_mc(row) and str(row.get("query_id") or "") in inputs]
    candidates.sort(key=lambda row: str(row["query_id"]))
    if args.max_records is not None:
        candidates = candidates[: args.max_records]
    args.artifact_dir.mkdir(parents=True)
    index_rows, malformed_index_rows = read_index_jsonl(args.object_index)
    config = resolve_provider_config()
    api = OpenAI(api_key=config.credential.value, base_url=config.endpoint, timeout=args.timeout)
    model = args.model or config.model
    decisions: dict[str, dict[str, Any]] = {}
    decision_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row in candidates:
        query_id = str(row["query_id"])
        input_row = inputs[query_id]
        options = dict((input_row.get("multiple_choice") or {}).get("options") or {})
        bundle = rank_bundle(
            str(input_row.get("question") or ""),
            options,
            selected_paper_ids(row),
            index_rows,
            locators=parent_locators(row),
        )
        if not options or not bundle:
            decision_rows.append({"query_id": query_id, "status": "NO_SOURCE_BUNDLE"})
            continue
        payload = request_payload(str(input_row.get("question") or ""), options, bundle)
        raw, diagnostic = call_model(api, model, payload, args.timeout)
        raw_rows.append({"query_id": query_id, "prompt": payload, "raw_response": raw})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        proposal = validate_proposal(parsed, options, bundle)
        if proposal is None:
            decision_rows.append({"query_id": query_id, "status": diagnostic or proposal_status(parsed, options, bundle)})
            continue
        decisions[query_id] = proposal
        decision_rows.append({"query_id": query_id, "status": "ACCEPTED", **proposal})
    output_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        copied = copy.deepcopy(row)
        proposal = decisions.get(str(copied.get("query_id") or ""))
        if proposal is not None and blank_mc(copied):
            copied["answer"]["multiple_choice"]["gold"] = proposal["letter"]
        output_rows.append(copied)
    candidate_path = args.artifact_dir / "candidate_predictions.jsonl"
    write_jsonl(candidate_path, output_rows)
    write_jsonl(args.artifact_dir / "decisions.jsonl", decision_rows)
    write_jsonl(args.artifact_dir / "raw_provider_io.jsonl", raw_rows)
    status_counts = Counter(str(row["status"]) for row in decision_rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "prediction_sha256": sha256(args.predictions),
        "candidate_sha256": sha256(candidate_path),
        "record_count": len(output_rows),
        "blank_candidates": len(candidates),
        "accepted": len(decisions),
        "rejected_or_unavailable": len(candidates) - len(decisions),
        "model": model,
        "endpoint": config.endpoint,
        "gold_used": False,
        "query_id_index_used": False,
        "evaluator_calls": 0,
        "malformed_index_rows_skipped": malformed_index_rows,
        "decision_status_counts": dict(sorted(status_counts.items())),
    }
    (args.artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "blank_candidates", "accepted", "rejected_or_unavailable", "gold_used", "query_id_index_used", "evaluator_calls")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
