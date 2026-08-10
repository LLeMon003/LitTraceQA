from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


EXPECTED_SOURCE_SHA = "A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281"
EXPECTED_CANDIDATE_SHA = "98C9AF8031E0CA4CEF59E21554A53660740C2C28703DE5E49D725A0E5A0A6597"
EXPECTED_DECISION_SHA = "DA0175EAAB1FA4236E7A0AED5D7C0173CE5916A3C510327F76C424D9BCB1C37C"


def artifact_root() -> Path:
    value = os.environ.get("LITTRACEQA_V23_ARTIFACT_ROOT")
    if not value:
        raise RuntimeError("LITTRACEQA_V23_ARTIFACT_ROOT is required")
    return Path(value)


def source_prediction() -> Path:
    value = os.environ.get("LITTRACEQA_V23_SOURCE_PREDICTION")
    if not value:
        raise RuntimeError("LITTRACEQA_V23_SOURCE_PREDICTION is required")
    return Path(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_source_prediction_hash() -> None:
    assert sha256(source_prediction()) == EXPECTED_SOURCE_SHA


def test_candidate_prediction_hash() -> None:
    assert sha256(artifact_root() / "mc_candidate_predictions.jsonl") == EXPECTED_CANDIDATE_SHA


def test_decision_hash() -> None:
    assert sha256(artifact_root() / "mc_recovery_decisions.jsonl") == EXPECTED_DECISION_SHA


def test_source_has_55_unique_records() -> None:
    source = rows(source_prediction())
    assert len(source) == 55 == len({row["query_id"] for row in source})


def test_candidate_has_55_unique_records() -> None:
    candidate = rows(artifact_root() / "mc_candidate_predictions.jsonl")
    assert len(candidate) == 55 == len({row["query_id"] for row in candidate})


def test_source_has_exactly_11_blank_mc_answers() -> None:
    blank = []
    for row in rows(source_prediction()):
        mc = (row.get("answer") or {}).get("multiple_choice")
        if isinstance(mc, dict) and not str(mc.get("gold") or "").strip():
            blank.append(row["query_id"])
    assert len(blank) == 11


def test_decisions_cover_exactly_the_blank_mc_answers() -> None:
    source = rows(source_prediction())
    blank = {
        row["query_id"] for row in source
        if isinstance((row.get("answer") or {}).get("multiple_choice"), dict)
        and not str(row["answer"]["multiple_choice"].get("gold") or "").strip()
    }
    decisions = rows(artifact_root() / "mc_recovery_decisions.jsonl")
    assert {row["query_id"] for row in decisions} == blank


def test_candidate_fills_all_mc_answers() -> None:
    candidate = rows(artifact_root() / "mc_candidate_predictions.jsonl")
    mc_rows = [row for row in candidate if "multiple_choice" in (row.get("answer") or {})]
    assert len(mc_rows) == 41
    assert all(str(row["answer"]["multiple_choice"].get("gold") or "").strip() for row in mc_rows)


def test_only_previously_blank_mc_fields_changed() -> None:
    source = rows(source_prediction())
    candidate = rows(artifact_root() / "mc_candidate_predictions.jsonl")
    changed = 0
    for before, after in zip(source, candidate, strict=True):
        assert before["query_id"] == after["query_id"]
        if before == after:
            continue
        before_copy = copy.deepcopy(before)
        after_letter = after["answer"]["multiple_choice"]["gold"]
        assert not str(before_copy["answer"]["multiple_choice"].get("gold") or "").strip()
        before_copy["answer"]["multiple_choice"]["gold"] = after_letter
        assert before_copy == after
        changed += 1
    assert changed == 11


def test_freeze_hashes_match_current_artifacts() -> None:
    freeze = json.loads((artifact_root() / "MC_CANDIDATE_FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_PRE_EVALUATION"
    assert freeze["gold_used"] is False
    assert freeze["candidate_prediction_sha256"] == EXPECTED_CANDIDATE_SHA
    assert freeze["decision_sha256"] == EXPECTED_DECISION_SHA
    assert freeze["source_prediction_sha256"] == EXPECTED_SOURCE_SHA


def test_official_result_is_single_invocation_for_frozen_candidate() -> None:
    result = json.loads((artifact_root() / "mc_official_evaluation.json").read_text(encoding="utf-8"))
    assert result["status"] == "AUTHORITATIVE"
    assert result["invocation_count"] == 1
    assert result["prediction_sha256"] == EXPECTED_CANDIDATE_SHA
    assert result["inputs_unchanged"] is True


def test_authoritative_metrics_are_component_isolated() -> None:
    result = json.loads((artifact_root() / "mc_official_evaluation.json").read_text(encoding="utf-8"))
    metrics = result["result"]["metrics"]
    assert metrics["multiple_choice_accuracy"] == 33 / 41
    assert metrics["paper_f1_macro"] == 1.0
    assert metrics["evidence_f1_macro"] == 0.5815597533779352
    assert metrics["freeform_exact_match"] == 16 / 26
    assert metrics["table_row_f1_macro"] == 0.8753968253968254
    assert metrics["table_cell_accuracy_micro"] == 14 / 27
