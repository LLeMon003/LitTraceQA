from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_io import extract_answer_contract, find_official_file, read_jsonl, write_jsonl
from .parser import normalize_prediction, strip_internal_grounding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate official predictions from existing internal predictions with the current normalizer.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--source-output-dir", required=True, help="Existing run directory containing internal_predictions/candidates/selected contexts.")
    parser.add_argument("--output-dir", required=True, help="Directory to write renormalized predictions and report.")
    parser.add_argument("--symbolic-evidence-standardization", choices=["true", "false"], default="true")
    return parser.parse_args()


def _latest_by_query_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            result[query_id] = row
    return result


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    validation_by_query = {str(row.get("query_id") or ""): row for row in validation_rows if row.get("query_id")}
    internal_by_query = _latest_by_query_id(read_jsonl(source_dir / "internal_predictions.jsonl"))
    candidate_by_query = _latest_by_query_id(read_jsonl(source_dir / "candidate_papers.jsonl"))
    selected_by_query = _latest_by_query_id(read_jsonl(source_dir / "selected_symbolic_contexts.prompt.jsonl"))
    standardize = args.symbolic_evidence_standardization.lower() == "true"

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stats = {
        "official_dir": str(args.official_dir),
        "source_output_dir": str(source_dir),
        "output_dir": str(output_dir),
        "query_count": len(validation_rows),
        "renormalized_predictions": 0,
        "missing_internal_predictions": 0,
        "symbolic_evidence_standardization": standardize,
        "error_type_counts": {},
    }

    for sample in validation_rows:
        query_id = str(sample.get("query_id") or "")
        internal = internal_by_query.get(query_id)
        if not internal:
            stats["missing_internal_predictions"] += 1
            continue
        candidates = candidate_by_query.get(query_id, {}).get("candidates") or []
        if not isinstance(candidates, list):
            candidates = []
        selected_evidence = selected_by_query.get(query_id, {}).get("selected_evidence") or []
        if not isinstance(selected_evidence, list):
            selected_evidence = []
        answer_contract = extract_answer_contract(sample, validation_by_query.get(query_id))
        prediction, normalization_errors = normalize_prediction(
            internal,
            sample,
            [str(candidate.get("paper_id") or "") for candidate in candidates if isinstance(candidate, dict)],
            answer_contract=answer_contract,
            selected_evidence=selected_evidence,
            symbolic_evidence_standardization=standardize,
            candidate_records=[candidate for candidate in candidates if isinstance(candidate, dict)],
        )
        predictions.append(strip_internal_grounding(prediction))
        stats["renormalized_predictions"] += 1
        for error in normalization_errors:
            errors.append(error)
            error_type = str(error.get("type") or "unknown")
            stats["error_type_counts"][error_type] = int(stats["error_type_counts"].get(error_type, 0)) + 1

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "renormalization_errors.jsonl", errors)
    (output_dir / "renormalization_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
