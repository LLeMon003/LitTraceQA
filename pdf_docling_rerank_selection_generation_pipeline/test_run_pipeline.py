import json
import tempfile
import unittest
from pathlib import Path

from .run_pipeline import _completed_query_ids, _has_required_table_rows, _load_slot_plans, _messages_for_json_retry, _table_focused_context, parse_args


class RunPipelineTests(unittest.TestCase):
    def test_parse_args_exposes_inputs_override(self) -> None:
        args = parse_args(["--inputs", "official_dev/data/test.jsonl"])
        self.assertEqual(args.inputs, "official_dev/data/test.jsonl")
        defaults = parse_args([])
        self.assertEqual(defaults.inputs, "")

    def test_completed_query_ids_uses_predictions_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"query_id": "q_001"}) + "\n" + json.dumps({"query_id": "q_002"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_completed_query_ids(path), {"q_001", "q_002"})

    def test_load_slot_plans_keeps_the_latest_valid_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot_plans.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q_001", "plan": {"version": "old"}}),
                        json.dumps({"query_id": "q_001", "plan": {"version": "new"}}),
                        json.dumps({"query_id": "q_002", "plan": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_load_slot_plans(path), {"q_001": {"version": "new"}})

    def test_table_retry_requires_nonempty_rows(self) -> None:
        sample = {"answer_types": ["table"]}
        self.assertFalse(_has_required_table_rows({"answer": {"table": {"rows": []}}}, sample))
        self.assertTrue(_has_required_table_rows({"answer": {"table": {"rows": [{"method": "A"}]}}}, sample))
        retry = _messages_for_json_retry([{"role": "user", "content": "INPUT"}], 12, require_table_rows=True)
        self.assertIn("non-empty table_answer_plan", retry[0]["content"])
        partial = _messages_for_json_retry(
            [{"role": "user", "content": "INPUT"}], 12,
            require_table_rows=True, allow_partial_table_rows=True,
        )
        self.assertIn("Score every requested row independently", partial[0]["content"])

    def test_table_focused_context_keeps_the_best_matching_table_once(self) -> None:
        selected = {"selected_evidence": [
            {"global_record_id": "other", "source_type": "text_span", "text": "unrelated introduction"},
            {"global_record_id": "table", "source_type": "table", "label": "Table 10", "text": "Random FVT SCIQ 200 3000"},
            {"global_record_id": "table", "source_type": "table", "label": "Table 10", "text": "Random FVT SCIQ 200 3000"},
        ], "compact_chunk_packets": [{"unused": True}], "evidence_hierarchy": {"unused": True}}
        focused = _table_focused_context(selected, "What SCIQ score does Random obtain at 200 steps?", limit=2)
        self.assertEqual([row["global_record_id"] for row in focused["selected_evidence"]], ["table", "other"])
        self.assertEqual(focused["compact_chunk_packets"], [])
        self.assertNotIn("evidence_hierarchy", focused)

    def test_table_focused_context_filters_to_multiple_routed_papers(self) -> None:
        selected = {"targeted_candidate_paper_ids": ["target_a", "target_b"], "selected_evidence": [
            {"global_record_id": "target", "paper_id": "target_a", "source_type": "text_span", "text": "Target method speedup 3x"},
            {"global_record_id": "homonym", "paper_id": "homonym", "source_type": "text_span", "text": "DISCO speedup 9x"},
        ]}
        focused = _table_focused_context(selected, "What is the target method speedup?", limit=12)
        self.assertEqual([row["global_record_id"] for row in focused["selected_evidence"]], ["target"])
