from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .parser import _label_to_locator, _resolve_evidence_ref_echo, normalize_prediction
from .evidence_hierarchy import keyed_hierarchy_prompt_projection
from .symbolic_schema import grounding_label_from_record
from .generate_from_cached_selection import _posthoc_ground_keyed_prediction
from .run_pipeline import _acquire_output_lock, _messages_for_json_retry, _remove_query_rows
from .symbolic_context_selector import _compact_package_packets, audit_selected_context
from .vlm_answer_client import VLMAnswerClient
from .vlm_answer_prompt_builder import build_symbolic_answer_prompt


class PromptGroundingTests(unittest.TestCase):
    def test_keyed_table_projection_respects_configured_row_and_view_limits(self):
        hierarchy = {
            "keyed_table_structure_enabled": True,
            "keyed_table_view_limit": 2,
            "keyed_table_view_rows": 3,
            "l0_catalog": [
                {"evidence_ref": f"E000{index}", "global_record_id": f"paper::{index}", "paper_id": "paper", "section_id": "sec", "page": 1, "source_type": "table", "label": f"Table {index}", "text": "table"}
                for index in range(1, 4)
            ],
            "l2_evidence_cards": [
                {
                    "support_refs": [f"E000{index}"],
                    "table_view": {"columns": ["metric"], "rows": [{"row_label": str(row), "values": {"metric": str(row)}} for row in range(4)]},
                }
                for index in range(1, 4)
            ],
        }
        cards = keyed_hierarchy_prompt_projection(hierarchy)["l2_cards"]
        views = [card["table_view"] for card in cards if card["table_view"]]
        self.assertEqual(len(views), 2)
        self.assertTrue(all(len(view["rows"]) == 3 for view in views))

    def test_prompt_audit_requires_exact_selected_record_alignment(self):
        selected = [{
            "evidence_ref": "E0001",
            "paper_id": "paper",
            "page": 3,
            "source_type": "figure",
            "label": "Figure 2",
            "locator": {"page": 3, "figure_id": "Figure 2"},
            "text": "caption",
            "section_id": "sec",
            "section_title": "Method",
            "section_type": "method",
            "section_path": ["Method"],
            "reading_order": 4,
            "document_order": 4,
        }]
        packet = {
            "paper_id": "paper",
            "section_id": "sec",
            "section_title": "Method",
            "section_type": "method",
            "section_path": ["Method"],
            "record_defaults": {"page": 3, "source_type": "figure"},
            "records": [{
                "evidence_ref": "E0001",
                "label": "Figure 2",
                "locator": {"figure_id": "Figure 2"},
                "text": "caption",
                "reading_order": 4,
                "document_order": 4,
            }],
        }
        self.assertTrue(audit_selected_context(selected, [packet], [])["passed"])
        packet["records"][0]["text"] = "wrong"
        with self.assertRaisesRegex(ValueError, "field mismatch: text"):
            audit_selected_context(selected, [packet], [])

    def test_package_packets_emit_overlapping_neighbors_once(self):
        first = {
            "global_record_id": "paper::r1", "evidence_ref": "E0001", "paper_id": "paper",
            "page": 1, "source_type": "text_span", "text": "first",
        }
        shared = {
            "global_record_id": "paper::r2", "evidence_ref": "E0002", "paper_id": "paper",
            "page": 1, "source_type": "figure", "label": "Figure 1", "text": "shared",
        }
        packages = [
            {"package_id": "pkg::one", "paper_id": "paper", "records": [first, shared]},
            {"package_id": "pkg::two", "paper_id": "paper", "records": [shared]},
        ]
        packets = _compact_package_packets(packages, [first, shared])
        self.assertEqual(len(packets), 1)
        self.assertEqual([item["evidence_ref"] for item in packets[0]["records"]], ["E0001", "E0002"])

    def test_package_packets_preserve_cross_section_record_metadata(self):
        reference = {
            "global_record_id": "paper::reference", "evidence_ref": "E0001", "paper_id": "paper",
            "page": 9, "source_type": "citation_context", "locator": {"page": 9, "citation_id": "7"}, "text": "[7] Reference entry",
            "section_id": "references", "section_title": "References", "section_type": "references",
            "section_path": ["References"],
        }
        context = {
            "global_record_id": "paper::context", "evidence_ref": "E0002", "paper_id": "paper",
            "page": 2, "source_type": "citation_context", "locator": {"page": 2, "citation_id": "7"}, "text": "We follow [7].",
            "section_id": "method", "section_title": "Method", "section_type": "method",
            "section_path": ["Method"],
        }
        packets = _compact_package_packets(
            [{"package_id": "pkg::paper::reference", "anchor_record_id": "paper::reference", "paper_id": "paper", "section_id": "references", "records": [reference, context]}],
            [reference, context],
        )
        self.assertEqual([packet["section_id"] for packet in packets], ["references", "method"])
        self.assertTrue(audit_selected_context([reference, context], packets, [])["passed"])

    def test_evidence_ref_echo_restores_exact_selected_locator(self):
        selected = [{
            "evidence_ref": "E0007",
            "paper_id": "paper",
            "source_type": "table",
            "locator": {"page": 8, "table_id": "Table 3"},
        }]
        evidence, errors = _resolve_evidence_ref_echo(
            {"evidence": [{"evidence_ref": "E0007"}]},
            selected,
        )
        self.assertEqual(
            evidence,
            [{
                "paper_id": "paper",
                "source_type": "table",
                "locator": {"page": 8, "table_id": "Table 3"},
            }],
        )
        self.assertEqual(errors, ["evidence_ref_echo_resolved"])
        prediction, _ = normalize_prediction(
            {
                "gold_papers": [{"paper_id": "paper"}],
                "evidence": [{"evidence_ref": "E0007"}],
                "answer": {"freeform": {"text": "x"}},
            },
            {
                "query_id": "q",
                "task_family": "hidden_source_single_paper",
                "primary_evidence_type": "table",
                "answer_types": ["freeform"],
                "question": "value?",
            },
            ["paper"],
            selected_evidence=selected,
        )
        self.assertEqual(prediction["evidence"], evidence)

    def test_citation_locator_is_canonicalized_for_old_selection_cache(self):
        selected = [{
            "evidence_ref": "E0024", "paper_id": "paper", "source_type": "citation_context",
            "label": "Reference 24", "locator": {"page": 12, "citation_id": "Reference 24"},
        }]
        evidence, _ = _resolve_evidence_ref_echo({"evidence_refs": ["E0024"]}, selected)
        self.assertEqual(evidence[0]["locator"], {"page": 12, "citation_id": "24"})
        self.assertEqual(_label_to_locator("citation_context", "[24]"), {"citation_id": "24"})
        self.assertEqual(grounding_label_from_record("citation_context", "Ref. 24"), {"type": "citation_id", "value": "24"})

    def test_table_shaped_answer_can_ground_to_figure_card(self):
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::figure", "paper_id": "paper",
                "page": 2, "source_type": "figure", "label": "Figure 1",
                "locator": {"page": 2, "figure_id": "Figure 1"},
                "text": "Figure 1: Method MASTER uses MCTS.",
            }],
            "l2_evidence_cards": [{
                "card_id": "C001", "proposition": "Figure 1: Method MASTER uses MCTS.",
                "source_type": "figure", "support_refs": ["E0001"],
            }],
            "query_claims": [{"claim_id": "Q01", "claim": "Which method uses MCTS?"}],
            "l1_contexts": [], "l3_navigation": {},
        }
        grounded, audit = _posthoc_ground_keyed_prediction(
            {
                "claim_to_support_keys": {"Q01": ["C001"]},
                "table_answer_plan": [{"row_support_key": "C001", "values": {"Method": "MASTER"}}],
                "answer": {"table": {"rows": [{"Method": "MASTER"}]}, "freeform": {"text": "MASTER"}},
            },
            hierarchy,
        )
        self.assertEqual(grounded["table_answer_plan"][0]["row_evidence_ref"], "E0001")
        self.assertEqual(grounded["answer"]["freeform"]["text"], "MASTER")
        self.assertEqual(grounded["answer"]["table"]["rows"], [{"Method": "MASTER"}])
        self.assertTrue(any(row.get("status") == "table_plan_grounded" for row in audit))

    def test_unsupported_table_draft_is_removed_after_keyed_grounding(self):
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::caption", "paper_id": "paper",
                "page": 2, "source_type": "figure", "label": "Figure 1",
                "locator": {"page": 2, "figure_id": "Figure 1"}, "text": "Figure 1: MASTER uses MCTS.",
            }],
            "l2_evidence_cards": [{
                "card_id": "C001", "proposition": "Figure 1: MASTER uses MCTS.",
                "source_type": "figure", "support_refs": ["E0001"],
            }],
            "query_claims": [{"claim_id": "Q01", "claim": "Which method uses MCTS?"}],
            "l1_contexts": [], "l3_navigation": {},
        }
        grounded, audit = _posthoc_ground_keyed_prediction(
            {
                "claim_to_support_keys": {"Q01": ["C001"]},
                "table_answer_plan": [{"row_support_key": "C001", "values": {"Method": "Invented"}}],
                "answer": {"table": {"rows": [{"Method": "Invented"}]}},
            },
            hierarchy,
        )
        self.assertEqual(grounded["table_answer_plan"], [])
        self.assertEqual(grounded["answer"]["table"]["rows"], [])
        self.assertTrue(any(row.get("status") == "unsupported_table_row_removed" for row in audit))

    def test_verified_visual_card_can_ground_visible_figure_fact(self):
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::figure", "paper_id": "paper",
                "page": 2, "source_type": "figure", "label": "Figure 1",
                "locator": {"page": 2, "figure_id": "Figure 1"}, "text": "Figure 1: Learning curve.",
            }],
            "l2_evidence_cards": [{
                "card_id": "V001", "proposition": "The red curve reaches 80% accuracy.",
                "source_type": "figure", "support_refs": ["E0001"],
                "verification": {"status": "visual_verified"},
            }],
            "query_claims": [{"claim_id": "Q01", "claim": "What accuracy does the red curve reach?"}],
            "l1_contexts": [], "l3_navigation": {},
        }
        grounded, audit = _posthoc_ground_keyed_prediction(
            {"claim_to_support_keys": {"Q01": ["C001"]}, "answer": {"freeform": {"text": "80%"}}}, hierarchy
        )
        self.assertEqual(grounded["answer"]["freeform"]["text"], "80%")
        self.assertEqual(grounded["visual_support_card_keys"], ["C001"])
        self.assertTrue(any(row.get("status") == "directly_grounded" for row in audit))

    def test_paper_title_table_cell_accepts_only_the_grounded_paper_id(self):
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::figure", "paper_id": "paper",
                "page": 2, "source_type": "figure", "label": "Figure 1",
                "locator": {"page": 2, "figure_id": "Figure 1"}, "text": "Figure 1: MCTS framework.",
            }],
            "l2_evidence_cards": [{"card_id": "C001", "proposition": "Figure 1: MCTS framework.", "source_type": "figure", "support_refs": ["E0001"]}],
            "query_claims": [{"claim_id": "Q01", "claim": "Which paper?"}], "l1_contexts": [], "l3_navigation": {},
        }
        grounded, _ = _posthoc_ground_keyed_prediction(
            {"claim_to_support_keys": {"Q01": ["C001"]}, "table_answer_plan": [{"row_support_key": "C001", "values": {"Paper Title": "paper"}}]}, hierarchy
        )
        self.assertEqual(grounded["table_answer_plan"][0]["values"], {"Paper Title": "paper"})

    def test_table_plan_key_must_be_linked_to_a_claim(self):
        hierarchy = {
            "primary_evidence_type": "figure",
            "l0_catalog": [{"evidence_ref": "E0001", "global_record_id": "paper::figure", "paper_id": "paper", "page": 2, "source_type": "figure", "label": "Figure 1", "locator": {"page": 2, "figure_id": "Figure 1"}, "text": "Figure 1: MCTS framework."}],
            "l2_evidence_cards": [
                {"card_id": "C001", "proposition": "Figure 1: MCTS framework.", "source_type": "figure", "support_refs": ["E0001"]},
                {"card_id": "C002", "proposition": "Figure 1: MCTS framework.", "source_type": "figure", "support_refs": ["E0001"]},
            ],
            "query_claims": [{"claim_id": "Q01", "claim": "Which paper uses MCTS?"}], "l1_contexts": [], "l3_navigation": {},
        }
        grounded, audit = _posthoc_ground_keyed_prediction(
            {"claim_to_support_keys": {"Q01": ["C001"]}, "gold_papers": [{"paper_id": "paper"}, {"paper_id": "unsupported"}], "table_answer_plan": [{"row_support_key": "C002", "values": {"Paper Title": "paper"}}]}, hierarchy
        )
        self.assertEqual(grounded["table_answer_plan"], [])
        self.assertEqual(grounded["evidence_refs"], ["E0001"])
        self.assertEqual(grounded["gold_papers"], [{"paper_id": "paper"}])
        self.assertTrue(any(row.get("status") == "table_plan_unlinked_to_claim_removed" for row in audit))
        self.assertTrue(any(row.get("status") == "unsupported_paper_removed" for row in audit))

    def test_explicit_venue_and_object_constraints_filter_impossible_card_keys(self):
        hierarchy = {
            "primary_evidence_type": "figure",
            "l0_catalog": [
                {"evidence_ref": "E0001", "global_record_id": "naacl2025_1::figure", "paper_id": "naacl2025_1", "page": 1, "source_type": "figure", "label": "Figure 1", "locator": {"page": 1, "figure_id": "Figure 1"}, "text": "MCTS figure."},
                {"evidence_ref": "E0002", "global_record_id": "acl2025_1::figure", "paper_id": "acl2025_1", "page": 1, "source_type": "figure", "label": "Figure 1", "locator": {"page": 1, "figure_id": "Figure 1"}, "text": "MCTS figure."},
                {"evidence_ref": "E0003", "global_record_id": "naacl2025_2::figure", "paper_id": "naacl2025_2", "page": 1, "source_type": "figure", "label": "Figure 1", "locator": {"page": 1, "figure_id": "Figure 1"}, "text": "Other figure."},
            ],
            "l2_evidence_cards": [
                {"card_id": "C001", "proposition": "Figure explicitly names MCTS.", "source_type": "figure", "support_refs": ["E0001"]},
                {"card_id": "C002", "proposition": "Figure explicitly names MCTS.", "source_type": "figure", "support_refs": ["E0002"]},
                {"card_id": "C003", "proposition": "Figure shows a method diagram.", "source_type": "figure", "support_refs": ["E0003"]},
            ],
            "query_claims": [{"claim_id": "Q01", "claim": "Which NAACL 2025 papers explicitly mention MCTS in a figure?"}], "l1_contexts": [], "l3_navigation": {},
        }
        grounded, audit = _posthoc_ground_keyed_prediction(
            {"claim_to_support_keys": {"Q01": ["C001", "C002", "C003"]}, "answer": {"freeform": {"text": ""}}}, hierarchy
        )
        self.assertEqual(grounded["evidence_refs"], ["E0001"])
        self.assertEqual(grounded["claim_to_support_keys"], {"Q01": ["C001"]})
        self.assertEqual(sum(row.get("status") == "claim_key_constraint_removed" for row in audit), 2)

    def test_answer_client_attaches_real_image(self):
        config = SimpleNamespace(answer_api_key="test-key-123456", answer_model="Qwen3-VL")
        client = VLMAnswerClient(config)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            messages = client._with_images(
                [{"role": "user", "content": "inspect IMG001"}],
                [image_path],
            )
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "inspect IMG001"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as attached:
            self.assertEqual(attached.size, (32, 32))

    def test_json_retry_requires_complete_sparse_evidence(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "original"}]
        retried = _messages_for_json_retry(messages, 12)
        self.assertEqual(messages[1]["content"], "original")
        self.assertIn("complete valid JSON", retried[1]["content"])
        self.assertIn("at most 12", retried[1]["content"])

    def test_keyed_prompt_hides_runtime_card_metadata(self):
        hierarchy = {
            "prompt_mode": "keyed_l2_only",
            "l0_catalog": [{"evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper", "page": 1, "source_type": "text_span", "text": "Evidence text."}],
            "l2_evidence_cards": [{"card_id": "C001", "proposition": "Evidence proposition.", "support_refs": ["E0001"], "verification": {"status": "visual_verified"}}],
            "query_claims": [{"claim_id": "Q01", "claim": "Question"}], "l1_contexts": [], "l3_navigation": {},
        }
        sample = {"query_id": "q", "task_family": "hidden_source_single_paper", "primary_evidence_type": "text_span", "question": "Question", "answer_types": ["freeform"]}
        messages = build_symbolic_answer_prompt(sample, [{"paper_id": "paper", "title": "Hidden title"}], {"evidence_hierarchy": hierarchy}, answer_contract={"answer_types": ["freeform"]})
        self.assertNotIn("_card_metadata", messages[-1]["content"])
        self.assertNotIn("visual_verified", messages[-1]["content"])

    def test_zero_micro_index_uses_final_prompt_ceiling_not_legacy_cap(self):
        import inspect
        from .generate_from_cached_selection import _fit_hierarchy_to_prompt
        source = inspect.getsource(_fit_hierarchy_to_prompt)
        self.assertIn("configured_micro_chars is None or int(configured_micro_chars) == 0", source)

    def test_keyed_hierarchy_does_not_disable_cropped_image_mode(self):
        import inspect
        from .generate_from_cached_selection import _fit_hierarchy_to_prompt
        source = inspect.getsource(_fit_hierarchy_to_prompt)
        self.assertIn("context_mode=config.vlm2_context_mode", source)
        self.assertNotIn('context_mode="text_only",\n            max_images=0', source)
        self.assertIn("max_images=max_images", source)

    def test_keyed_context_persists_image_map_for_posthoc_audit(self):
        from .generate_from_cached_selection import _selected_context
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            context = _selected_context(
                [{"evidence_ref": "E0001", "paper_id": "paper", "source_type": "figure", "crop_path": str(image_path), "text": "caption"}],
                context_mode="cropped_image",
                max_images=1,
                evidence_hierarchy={"l0_catalog": [{"evidence_ref": "E0001"}]},
            )
        self.assertEqual(context["attached_image_refs"], [{"image_ref": "IMG001", "evidence_refs": ["E0001"]}])
        self.assertEqual(context["evidence_hierarchy"]["image_map"], context["attached_image_refs"])

    def test_json_draft_error_retains_raw_attempts(self):
        from .generate_from_cached_selection import JSONDraftError, _extract_json_object_with_suffix_repair, _generate_json_draft
        error = JSONDraftError("bad json", [{"content": "not json"}])
        self.assertEqual(str(error), "bad json")
        self.assertEqual(error.raw_attempts, [{"content": "not json"}])

        class BrokenClient:
            def generate_prediction(self, messages, image_paths=None):
                return {"content": "not json", "raw_response": {"content": "not json"}}

        with self.assertRaises(JSONDraftError) as raised:
            _generate_json_draft(BrokenClient(), [{"role": "user", "content": "q"}], ["image.png"], 2)
        self.assertEqual(len(raised.exception.raw_attempts), 3)

        parsed, repaired = _extract_json_object_with_suffix_repair('{"answer":{"freeform":{"text":"x"}}')
        self.assertTrue(repaired)
        self.assertEqual(parsed["answer"]["freeform"]["text"], "x")
        with self.assertRaises(Exception):
            _extract_json_object_with_suffix_repair('{"answer":"unterminated}')

    def test_refinement_uses_same_suffix_repair_contract(self):
        import inspect
        from .generate_from_cached_selection import _refine_keyed_draft
        source = inspect.getsource(_refine_keyed_draft)
        self.assertIn("_extract_json_object_with_suffix_repair", source)

    def test_primary_table_enables_keyed_table_structure_for_freeform_contract(self):
        source = Path(__file__).with_name("generate_from_cached_selection.py").read_text(encoding="utf-8")
        self.assertIn('or str(sample.get("primary_evidence_type") or "") == "table"', source)

    def test_query_replacement_preserves_other_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            runtime_path = Path(directory) / "symbolic_records.runtime.jsonl"
            path.write_text(
                "\n".join(json.dumps({"query_id": query_id}) for query_id in ("q_001", "q_020", "q_001")) + "\n",
                encoding="utf-8",
            )
            runtime_path.write_text(json.dumps({"paper_id": "paper"}) + "\n", encoding="utf-8")
            _remove_query_rows(
                {"predictions": path, "symbolic_records_runtime": runtime_path},
                {"q_001"},
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            runtime_row = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(rows, [{"query_id": "q_020"}])
        self.assertEqual(runtime_row, {"paper_id": "paper"})

    def test_query_replacement_drops_unscoped_prior_run_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"type": "run_failed", "error": "old"}),
                    json.dumps({"query_id": "q_001", "type": "old_query_error"}),
                    json.dumps({"query_id": "q_020", "type": "other_query_error"}),
                ]) + "\n",
                encoding="utf-8",
            )
            _remove_query_rows({"errors": path}, {"q_001"})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [{"query_id": "q_020", "type": "other_query_error"}])

    def test_output_lock_rejects_concurrent_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _acquire_output_lock(Path(directory))
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    _acquire_output_lock(Path(directory))
            finally:
                first.close()


if __name__ == "__main__":
    unittest.main()
