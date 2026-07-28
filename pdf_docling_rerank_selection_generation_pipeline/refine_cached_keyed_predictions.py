"""Run fail-soft raw-L0 refinement over completed keyed generation drafts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .data_io import find_official_file, read_jsonl, write_jsonl
from .generate_from_cached_selection import (
    _candidate_map,
    _refine_keyed_draft,
    _restrict_prediction_to_visible_evidence,
    build_generation_provenance,
)
from .parser import normalize_prediction, strip_internal_grounding
from .vlm_answer_client import VLMAnswerClient


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine cached keyed predictions with resolved L0 evidence only.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--predictions-input", required=True)
    parser.add_argument("--internal-predictions-input", required=True)
    parser.add_argument("--candidate-papers-input", required=True)
    parser.add_argument("--answer-contracts-input", required=True)
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _effective_contract(sample: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Restore public answer-type metadata omitted from redacted contracts."""
    merged = dict(contract)
    answer_types = sample.get("answer_types")
    if isinstance(answer_types, list):
        merged["answer_types"] = [str(value) for value in answer_types]
    return merged


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    client = VLMAnswerClient(config, retries=config.generation_request_retries)
    validation_inputs_path = find_official_file(args.official_dir, "validation_inputs.jsonl")
    inputs = {str(row.get("query_id") or ""): row for row in read_jsonl(validation_inputs_path)}
    base_predictions = {str(row.get("query_id") or ""): row for row in read_jsonl(args.predictions_input)}
    internal = {str(row.get("query_id") or ""): row for row in read_jsonl(args.internal_predictions_input)}
    contracts = {str(row.get("query_id") or ""): row for row in read_jsonl(args.answer_contracts_input)}
    hierarchies = {
        str(row.get("query_id") or ""): row.get("hierarchy")
        for row in read_jsonl(args.hierarchy_input)
        if isinstance(row.get("hierarchy"), dict)
    }
    candidates = _candidate_map(Path(args.candidate_papers_input))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = build_generation_provenance(
        validation_inputs_path=validation_inputs_path,
        selected_contexts_path=Path(args.predictions_input),
        candidate_papers_path=Path(args.candidate_papers_input),
        hierarchy_path=Path(args.hierarchy_input),
        answer_contracts_path=Path(args.answer_contracts_input),
        env_path=args.env_path,
        generation_parameters={"mode": "cached_l0_refinement", "internal_predictions": str(Path(args.internal_predictions_input).resolve())},
    )
    (output / "refinement_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    predictions: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for query_id, base in base_predictions.items():
        sample = inputs.get(query_id)
        hierarchy = hierarchies.get(query_id)
        draft = internal.get(query_id)
        if not sample or not hierarchy or not draft:
            predictions.append(base)
            audit.append({"query_id": query_id, "status": "preserved_missing_cached_input"})
            continue
        if args.dry_run:
            predictions.append(base)
            audit.append({"query_id": query_id, "status": "dry_run"})
            continue
        try:
            effective_contract = _effective_contract(sample, contracts.get(query_id, {}))
            refined, response = _refine_keyed_draft(
                client,
                sample,
                candidates.get(query_id, []),
                effective_contract,
                draft,
                hierarchy,
            )
            prediction, errors = normalize_prediction(
                refined,
                sample,
                [str(row.get("paper_id") or "") for row in candidates.get(query_id, [])],
                answer_contract=effective_contract,
                selected_evidence=[dict(row) for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)],
                symbolic_evidence_standardization=config.symbolic_evidence_standardization,
                candidate_records=candidates.get(query_id, []),
            )
            prediction, outside_removed = _restrict_prediction_to_visible_evidence(
                prediction,
                [dict(row) for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)],
            )
            predictions.append(strip_internal_grounding(prediction))
            raw.append({"query_id": query_id, **(response or {})})
            audit.append({"query_id": query_id, "status": "refined", "normalization_errors": [str(error) for error in errors], "outside_removed": outside_removed})
        except Exception as exc:
            predictions.append(base)
            audit.append({"query_id": query_id, "status": "preserved_refinement_failure", "error": str(exc)})
        print(json.dumps({"query_id": query_id, "completed": len(predictions)}, ensure_ascii=False), flush=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "raw_refinement_responses.jsonl", raw)
    write_jsonl(output / "refinement_audit.jsonl", audit)
    print(json.dumps({"predictions": len(predictions), "refined": sum(row.get("status") == "refined" for row in audit), "preserved": sum(str(row.get("status") or "").startswith("preserved") for row in audit), "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
