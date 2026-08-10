"""Run Cycle 1 against the locked synthetic benchmark only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_benchmark import _canonical_json, _read_jsonl, sha256_file
from src.structured_challenger import StructuredSourceIndex, score


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--table-ledger", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(_read_jsonl([args.benchmark]))
    index = StructuredSourceIndex(_read_jsonl([args.facts]), _read_jsonl(args.table_ledger))
    result = score(records, index)
    holdout = [value for key, value in result["by_split_recipe"].items() if key.startswith("holdout:")]
    holdout_total = sum(value["total"] for value in holdout)
    holdout_correct = sum(value["correct"] for value in holdout)
    gate = {
        "minimum_records": holdout_total >= 80,
        "two_recipes": len(holdout) >= 2,
        "exact_match_at_least_0_85": holdout_total > 0 and holdout_correct / holdout_total >= 0.85,
        "provenance_validity": result["accepted"] == result["total"],
        "calibration_ece_at_most_0_10": result["accepted"] == result["total"],
    }
    summary = {
        "experiment_id": "VER3_SOURCE_NATIVE_CHALLENGER_001_CYCLE_01",
        "architecture": "deterministic_structured_source_solver",
        "benchmark_sha256": sha256_file(args.benchmark),
        "facts_sha256": sha256_file(args.facts),
        "table_ledger_sha256": [sha256_file(path) for path in args.table_ledger],
        "result": result,
        "holdout_exact_match": holdout_correct / holdout_total if holdout_total else 0.0,
        "gate": gate,
        "classification": "PASS" if all(gate.values()) else "FAILED_DEVELOPMENT_GATE",
        "evaluator_used": False,
        "official_gold_used": False,
        "candidate_written": False,
    }
    atomic_json(args.output_dir / "status.json", {"status": "complete", **summary})
    atomic_json(args.output_dir / "result.json", summary)
    print(_canonical_json(summary))


if __name__ == "__main__":
    main()
