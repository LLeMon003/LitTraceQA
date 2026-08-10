from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mc_blank_recovery", ROOT / "scripts" / "run_mc_blank_recovery.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_blank_mc_query_ids_only_selects_empty_mc_fields() -> None:
    rows = [
        {"query_id": "q1", "answer": {"multiple_choice": {"gold": ""}}},
        {"query_id": "q2", "answer": {"multiple_choice": {"gold": "A"}}},
        {"query_id": "q3", "answer": {"freeform": {"text": "x"}}},
    ]
    assert MODULE.blank_mc_query_ids(rows) == ["q1"]


def test_validate_answer_rejects_non_option_letter() -> None:
    assert MODULE.validate_answer({"letter": "b", "semantic_answer": "x"}, ["A", "B"])["letter"] == "B"
    try:
        MODULE.validate_answer({"letter": "C"}, ["A", "B"])
    except ValueError as exc:
        assert "invalid option letter" in str(exc)
    else:
        raise AssertionError("invalid letter was accepted")


def test_selected_paper_ids_accepts_structured_predictions() -> None:
    assert MODULE.selected_paper_ids([{"paper_id": "p1"}, {"paper_id": "p2"}]) == {"p1", "p2"}


def test_render_context_prioritizes_evidence_page_and_overlap() -> None:
    evidence = [{"paper_id": "p", "locator": {"page": 2}}]
    objects = [
        {"paper_id": "p", "page": 1, "object_type": "paragraph", "text": "target metric"},
        {"paper_id": "p", "page": 2, "object_type": "paragraph", "text": "target metric value 42"},
    ]
    rendered = MODULE.render_context("target metric?", {"A": "42", "B": "7"}, evidence, objects)
    assert rendered.splitlines()[0].startswith("[paper=p page=2")
