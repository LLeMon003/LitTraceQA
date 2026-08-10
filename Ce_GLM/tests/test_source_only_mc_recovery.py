import importlib.util
from pathlib import Path

from src.source_only_mc_recovery import contains_forbidden_key, index_object, parent_locators, proposal_status, rank_bundle, selected_paper_ids, validate_proposal


def test_index_projection_ignores_query_ids_and_ranks_only_selected_papers():
    row = {"object_uid": "o1", "paper_id": "p1", "text": "Alpha metric is 7", "query_ids": ["official-q"]}
    projected = index_object(row)
    assert projected and "query_ids" not in projected
    bundle = rank_bundle("Which option reports alpha metric?", {"A": "7", "B": "9"}, {"p1"}, [row, {"object_uid": "o2", "paper_id": "p2", "text": "Alpha metric is 9"}])
    assert [item["object_id"] for item in bundle] == ["o1"]


def test_validate_proposal_requires_allowed_citation_and_textual_support():
    bundle = [{"object_id": "o1", "paper_id": "p", "content": "The reported alpha value is 7."}]
    options = {"A": "7", "B": "9"}
    assert validate_proposal({"letter": "A", "citations": ["o1"], "confidence": 0.9}, options, bundle)
    assert validate_proposal({"letter": "B", "citations": ["o1"], "confidence": 0.9}, options, bundle) is None
    assert validate_proposal({"letter": "A", "citations": ["missing"], "confidence": 0.9}, options, bundle) is None
    assert proposal_status({"letter": "A", "citations": ["missing"], "confidence": 0.9}, options, bundle) == "OUT_OF_BUNDLE_CITATION"
    assert proposal_status({"letter": "", "citations": [], "confidence": 0.0}, options, bundle) == "MODEL_ABSTAIN"


def test_selected_papers_and_forbidden_input_guard():
    assert selected_paper_ids({"gold_papers": [{"paper_id": "p1"}, {"paper_id": "p2"}]}) == {"p1", "p2"}
    assert contains_forbidden_key({"multiple_choice": {"options": {"A": "x"}}}) is False
    assert contains_forbidden_key({"multiple_choice": {"gold": "A"}}) is True


def test_malformed_docling_rows_are_skipped_without_exposing_content(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_source_only_mc_recovery.py"
    spec = importlib.util.spec_from_file_location("source_only_mc_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "index.jsonl"
    path.write_text('{"object_uid":"o","paper_id":"p","text":"ok"}\n{broken\n', encoding="utf-8")
    rows, malformed = module.read_index_jsonl(path)
    assert len(rows) == 1
    assert malformed == 1


def test_parent_evidence_locator_boosts_only_matching_source_family():
    parent = {"evidence": [{"paper_id": "p1", "source_type": "table", "locator": {"page": 3, "table_id": "Table 2"}}]}
    locators = parent_locators(parent)
    assert locators == [{"paper_id": "p1", "source_type": "table", "page": 3, "table_id": "Table 2", "figure_id": "", "algorithm_id": ""}]
    index = [
        {"object_uid": "lexical", "paper_id": "p1", "page": 1, "object_type": "paragraph", "text": "Alpha metric appears here."},
        {"object_uid": "anchored", "paper_id": "p1", "page": 3, "object_type": "table_cell", "object_label": "Table 2", "text": "unrelated cell"},
        {"object_uid": "wrong-family", "paper_id": "p1", "page": 3, "object_type": "paragraph", "text": "Alpha metric appears here."},
    ]
    bundle = rank_bundle("Which option reports alpha metric?", {"A": "7"}, {"p1"}, index, locators=locators)
    assert [item["object_id"] for item in bundle][:2] == ["anchored", "lexical"]
