"""Run Architecture 1 only on deterministic target-aligned development records."""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import OpenAI
from src.credential_resolver import resolve_provider_config
from src.model_challenger import assess, metrics


def rows(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sample(records: list[dict], split: str, per_operator: int) -> list[dict]:
    by_operator: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["split"] == split:
            by_operator[record["reasoning_operator"]].append(record)
    return [record for operator in sorted(by_operator) for record in sorted(by_operator[operator], key=lambda item: item["record_id"])[:per_operator]]


def threshold(calibration: dict, proposals: list[dict]) -> float:
    for candidate in (0.90, 0.95, 0.99):
        retained = [item for item in proposals if item["confidence"] >= candidate]
        if len(retained) >= 20:
            retained_metrics = metrics(calibration["records"], retained)
            if retained_metrics["selective_exact_match"] >= 0.85:
                return candidate
    return 1.01


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_cached_raw(path: Path, phase: str) -> dict[int, str]:
    cached: dict[int, str] = {}
    if not path.is_file():
        return cached
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("phase") not in {"calibration", "holdout"} or not isinstance(row.get("batch_index"), int) or not isinstance(row.get("raw"), str):
                raise ValueError(f"invalid cached raw record at {path}:{line_number}")
            if row["phase"] != phase:
                continue
            if row["batch_index"] in cached:
                raise ValueError(f"duplicate cached {phase} batch {row['batch_index']}")
            cached[row["batch_index"]] = row["raw"]
    return cached


def write_status(path: Path, **values: object) -> None:
    path.write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--table-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-operator", type=int, default=25)
    parser.add_argument("--operators", help="comma-separated development operators for a bounded diagnostic")
    parser.add_argument("--resume", action="store_true", help="Reuse atomically persisted completed two-record batches in output-dir.")
    parser.add_argument("--model", help="Explicit provider model for this synthetic-only development run; defaults to the central provider configuration.")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="Per-request provider timeout with SDK retries disabled.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Strict-JSON completion cap for each at-most-two-record batch.")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = list(rows([args.benchmark]))
    if args.operators:
        permitted = set(args.operators.split(","))
        benchmark = [record for record in benchmark if record["reasoning_operator"] in permitted]
    calibration_records = sample(benchmark, "calibration", args.per_operator)
    holdout_records = sample(benchmark, "holdout", args.per_operator)
    corpus_by_paper: dict[str, list[dict]] = defaultdict(list)
    for row in rows([args.facts, *args.table_ledger]):
        if row.get("record_hash") or row.get("object_uid"):
            corpus_by_paper[str(row.get("paper_id"))].append(row)
    config = resolve_provider_config()
    active_model = args.model or config.model
    api = OpenAI(api_key=config.credential.value, base_url=config.endpoint, timeout=args.timeout_seconds, max_retries=0)
    def call(payload: str) -> str:
        response = api.chat.completions.create(model=active_model, messages=[{"role": "system", "content": "Return JSON only: {results:[{record_id,answer,source_object_ids,evidence_quote,confidence}]}. evidence_quote must be an exact nonempty substring from one cited source object. Never use knowledge outside supplied evidence."}, {"role": "user", "content": payload}], temperature=0, max_tokens=args.max_tokens, response_format={"type": "json_object"})
        return response.choices[0].message.content or "{}"
    raw_path = args.output_dir / "raw_responses.jsonl"
    proposal_path = args.output_dir / "proposals.jsonl"
    status_path = args.output_dir / "status.json"

    def run_phase(phase: str, records: list[dict]) -> list[dict]:
        cached = load_cached_raw(raw_path, phase)
        def checkpoint(batch_index: int, raw: str, proposals: list[dict], reused: bool) -> None:
            if reused:
                return
            append_jsonl(raw_path, {"phase": phase, "batch_index": batch_index, "raw": raw})
            for proposal in proposals:
                append_jsonl(proposal_path, {"phase": phase, "batch_index": batch_index, **proposal})
            write_status(status_path, status="in_progress", phase=phase, completed_batches=len(load_cached_raw(raw_path, phase)), official_gold_used=False)
        proposals, _ = assess(records, corpus_by_paper, call, reused_raw=cached, on_batch=checkpoint)
        return proposals

    try:
        write_status(status_path, status="in_progress", phase="calibration", official_gold_used=False)
        calibration_proposals = run_phase("calibration", calibration_records)
        frozen_threshold = threshold({"records": calibration_records}, calibration_proposals)
        write_status(status_path, status="in_progress", phase="holdout", official_gold_used=False)
        holdout_proposals = run_phase("holdout", holdout_records)
    except Exception as exc:
        write_status(status_path, status="infrastructure_failure", diagnostic_type=type(exc).__name__, official_gold_used=False)
        raise
    holdout_proposals = [item for item in holdout_proposals if item["confidence"] >= frozen_threshold]
    holdout_metrics = metrics(holdout_records, holdout_proposals)
    gate = {"accepted_at_least_100": holdout_metrics["accepted"] >= 100, "two_families": len(holdout_metrics["families"]) >= 2, "three_operators": len(holdout_metrics["operators"]) >= 3, "selective_exact_match_at_least_0_85": holdout_metrics["selective_exact_match"] >= 0.85, "all_grounded": holdout_metrics["all_grounded"]}
    report = {"status": "complete", "architecture": "structured_bundle_constrained_json", "calibration": {"record_count": len(calibration_records), "metrics": metrics(calibration_records, calibration_proposals), "frozen_confidence_threshold": frozen_threshold}, "holdout": {"record_count": len(holdout_records), "metrics": holdout_metrics}, "gate": gate, "classification": "PASS" if all(gate.values()) else "FAILED_DEVELOPMENT_GATE", "provider": {"endpoint": config.endpoint, "model": active_model, "model_source": "explicit_argument" if args.model else "central_provider_config", "timeout_seconds": args.timeout_seconds, "max_tokens": args.max_tokens, "sdk_retries": 0}, "official_gold_used": False, "candidate_written": False}
    (args.output_dir / "result.json").write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    write_status(status_path, status="complete", official_gold_used=False)
    print(json.dumps({"status": "complete", "classification": report["classification"], "metrics": holdout_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
