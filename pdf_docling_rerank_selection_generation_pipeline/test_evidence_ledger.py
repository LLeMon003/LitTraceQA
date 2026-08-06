from .generate_from_cached_selection import _evidence_ledger
from .vlm_answer_prompt_builder import _compact_evidence_ledger, build_symbolic_answer_prompt


def test_evidence_ledger_keeps_ref_and_extractable_context_once() -> None:
    records = [
        {
            "evidence_ref": "E0001",
            "paper_id": "paper_a",
            "page": 3,
            "source_type": "table",
            "label": "Table 2",
            "section_id": "sec_results",
            "section_title": "Results",
            "text": "Method | Score\nOurs | 91.2",
        },
        {
            "evidence_ref": "E0002",
            "paper_id": "paper_a",
            "page": 3,
            "source_type": "text_span",
            "section_id": "sec_results",
            "section_title": "Results",
            "text": "The score is reported on the test set.",
        },
    ]
    ledger = _evidence_ledger(records)
    assert ledger.count("PAPER paper_a") == 1
    assert ledger.count("SECTION Results") == 1
    assert "E0001\tp3\ttable\tTable 2" in ledger
    assert "E0002\tp3\ttext_span" in ledger
    assert _compact_evidence_ledger(records) == ledger


def test_multi_paper_ledger_protocol_has_bounded_nonduplicated_evidence() -> None:
    messages = build_symbolic_answer_prompt(
        {"query_id": "q", "question": "Compare methods.", "task_family": "multi_paper", "answer_types": ["freeform"]},
        [{"paper_id": "paper_a", "title": "A"}],
        {"selected_evidence": [], "evidence_ledger": "PAPER paper_a\nE0001\tp1\ttext_span\tfact"},
        answer_contract={"answer_types": ["freeform"], "multiple_choice": {"options": []}, "table": {"table_schema": None}},
    )
    prompt = messages[-1]["content"]
    assert '"evidence_output_limit":16' in prompt
    assert "contributing_papers" in prompt
    assert '"query_id"' not in prompt
    assert '"evidence_refs"' in prompt
