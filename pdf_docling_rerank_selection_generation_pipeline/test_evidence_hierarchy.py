from __future__ import annotations

import unittest

from .evidence_hierarchy import (
    attach_cards,
    build_l0_l1_l3,
    hierarchy_metrics,
    hierarchy_prompt_projection,
    keyed_hierarchy_prompt_projection,
    resolve_claim_support_keys,
    verify_llm_cards,
    _query_aware_extractive_proposition,
)
from .generate_from_cached_selection import _fact_is_directly_supported, _restrict_prediction_to_visible_evidence
from .parser import _evidence_from_table_answer_plan


class EvidenceHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = {
            "paper_id": "paper", "global_record_id": "paper::r2", "page": 2,
            "source_type": "table", "record_type": "table", "label": "Table 1",
            "locator": {"page": 2, "table_id": "Table 1"}, "section_id": "results",
            "section_title": "Results", "section_path": ["Results"], "document_order": 2,
            "text": "Table 1: scores\n| Method | F1 |\n| Ours | 91.2 |",
        }
        self.before = {
            "paper_id": "paper", "global_record_id": "paper::r1", "page": 2,
            "source_type": "text_span", "record_type": "paragraph", "locator": {"page": 2},
            "section_id": "results", "section_title": "Results", "section_path": ["Results"], "document_order": 1,
            "text": "We report test-set F1 scores for every method.",
        }
        self.after = {
            "paper_id": "paper", "global_record_id": "paper::r3", "page": 2,
            "source_type": "text_span", "record_type": "paragraph", "locator": {"page": 2},
            "section_id": "results", "section_title": "Results", "section_path": ["Results"], "document_order": 3,
            "text": "Higher F1 indicates better performance.",
        }

    def _hierarchy(self):
        return build_l0_l1_l3(
            [self.anchor], [{"paper_id": "paper", "title": "Paper", "abstract": "Abstract."}],
            {"paper": [self.before, self.anchor, self.after]}, l1_max_chars=240, l3_paper_chars=100,
        )

    def test_l0_preserves_anchor_and_l1_context_is_provenanced(self):
        hierarchy = self._hierarchy()
        metrics = hierarchy_metrics(hierarchy)
        self.assertEqual(metrics["exact_anchor_retention"], 1.0)
        self.assertEqual(metrics["l1_context_count"], 1)
        refs = {row["evidence_ref"] for row in hierarchy["l0_catalog"]}
        self.assertTrue(set(hierarchy["l1_contexts"][0]["neighbor_refs"]).issubset(refs))
        self.assertIn("table_context", hierarchy["l1_contexts"][0])

    def test_explicit_navigation_adds_only_named_paper_records(self):
        hardware = {
            "paper_id": "paper", "global_record_id": "paper::r4", "page": 9,
            "source_type": "text_span", "record_type": "paragraph", "locator": {"page": 9},
            "section_id": "setup", "section_title": "Experimental Setup", "section_path": ["Experimental Setup"], "document_order": 4,
            "text": "All experiments use the hardware configuration of a single NVIDIA RTX 4090 GPU.",
        }
        hierarchy = build_l0_l1_l3(
            [self.anchor], [{"paper_id": "paper", "title": "S-RAG: Auditing Data Provenance"}],
            {"paper": [self.before, self.anchor, self.after, hardware]},
            question="In the S-RAG framework, what hardware was used?", l1_max_chars=240, l3_paper_chars=100,
        )
        routed = [row for row in hierarchy["l0_catalog"] if row["global_record_id"] == "paper::r4"]
        self.assertEqual(len(routed), 1)
        self.assertIn("lexical_text_navigation", routed[0]["navigation_reasons"])

    def test_llm_card_requires_exact_quote_and_traceable_values(self):
        hierarchy = self._hierarchy()
        ref = next(row["evidence_ref"] for row in hierarchy["l0_catalog"] if row["global_record_id"] == "paper::r2")
        accepted, claims, rejected = verify_llm_cards({
            "claims": [{"claim_id": "Q01", "claim": "Find the score."}],
            "cards": [{
                "claim_ids": ["Q01"], "proposition": "Ours has F1 91.2.",
                "entities": ["Ours"], "values": ["91.2"], "conditions": [],
                "support_refs": [ref], "support_quotes": [{"evidence_ref": ref, "quote": "| Ours | 91.2 |"}],
            }],
        }, hierarchy, 6, 24)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        self.assertEqual(claims[0]["claim_id"], "Q01")

    def test_prompt_projection_omits_repeated_navigation_anchor_lists(self):
        hierarchy = attach_cards("What is the F1?", self._hierarchy(), mode="extractive", max_claims=6, max_cards=3, primary_evidence_type="table")
        projection = hierarchy_prompt_projection(hierarchy)
        self.assertNotIn("anchor_refs", projection["l3_navigation"]["sections"][0])
        refs = {row["evidence_ref"] for row in hierarchy["l0_catalog"]}
        self.assertTrue({ref for card in projection["l2_evidence_cards"] for ref in card["support_refs"]}.issubset(refs))

    def test_post_parse_gate_removes_table_plan_locator_not_in_l2_support(self):
        prediction, removed = _restrict_prediction_to_visible_evidence(
            {"evidence": [
                {"paper_id": "paper", "source_type": "table", "locator": {"page": 2, "table_id": "Table 1"}},
                {"paper_id": "paper", "source_type": "table", "locator": {"page": 9, "table_id": "Table 9"}},
            ]},
            [{"paper_id": "paper", "source_type": "table", "locator": {"page": 2, "table_id": "Table 1"}}],
        )
        self.assertTrue(removed)
        self.assertEqual(len(prediction["evidence"]), 1)

    def test_keyed_projection_hides_raw_provenance_and_resolves_cards(self):
        hierarchy = attach_cards("What is the F1?", self._hierarchy(), mode="extractive", max_claims=6, max_cards=3, primary_evidence_type="table")
        projection = keyed_hierarchy_prompt_projection(hierarchy)
        serialized = str({key: value for key, value in projection.items() if not key.startswith("_")})
        self.assertNotIn("E000", serialized)
        self.assertNotIn("support_quotes", serialized)
        self.assertNotIn("support_refs", serialized)
        self.assertNotIn("locator", serialized)
        self.assertTrue(projection["l2_cards"])
        card_key = projection["l2_cards"][0]["key"]
        padded_key = f"C0{int(card_key[1:]):03d}"
        refs, audit = resolve_claim_support_keys({"Q01": [padded_key, "C999"]}, hierarchy)
        self.assertTrue(refs)
        self.assertEqual(audit[0]["valid_card_keys"], [card_key])

    def test_keyed_projection_links_attached_crop_without_exposing_evidence_ref(self):
        hierarchy = attach_cards("What is the F1?", self._hierarchy(), mode="extractive", max_claims=6, max_cards=3, primary_evidence_type="table")
        table_ref = next(row["evidence_ref"] for row in hierarchy["l0_catalog"] if row["source_type"] == "table")
        hierarchy["image_map"] = [{"image_ref": "IMG001", "evidence_refs": [table_ref]}]
        projection = keyed_hierarchy_prompt_projection(hierarchy)
        serialized = str({key: value for key, value in projection.items() if not key.startswith("_")})
        self.assertIn("IMG001", serialized)
        self.assertNotIn(table_ref, serialized)

    def test_stability_query_aware_projection_prioritizes_ensemble_anchor(self):
        stable = dict(self.anchor, global_record_id="paper::stable", text="unrelated stable context", cached_selection_sources=["run_a", "run_b"])
        query_aligned = dict(self.anchor, global_record_id="paper::new", text="F1 score 91.2", cached_selection_sources=["run_c"])
        hierarchy = build_l0_l1_l3(
            [stable, query_aligned], [{"paper_id": "paper", "title": "Paper"}],
            {"paper": [stable, query_aligned]}, l1_max_chars=120, l3_paper_chars=80,
        )
        hierarchy.update({"task_family": "hidden_source_single_paper", "primary_evidence_type": "table", "keyed_micro_order": "stability_query_aware", "prompt_micro_text_chars": 40, "prompt_micro_index_chars": 1000})
        projection = keyed_hierarchy_prompt_projection(hierarchy)
        self.assertEqual(projection["l2_micro_rows"][0][-1], "unrelated stable context")



    def test_direct_fact_gate_accepts_values_and_rejects_unseen_entities(self):
        self.assertTrue(_fact_is_directly_supported("91.2", "| Ours | 91.2 |"))
        self.assertTrue(_fact_is_directly_supported("Ours", "| Ours | 91.2 |"))
        self.assertFalse(_fact_is_directly_supported("UnseenMethod", "| Ours | 91.2 |"))

    def test_table_answer_plan_preserves_non_table_source_type(self):
        evidence = _evidence_from_table_answer_plan(
            [{"row_source": {"paper_id": "paper", "page": 2, "source_type": "figure", "label": "Figure 1"}, "values": {"Method": "Ours"}}],
            {"paper"},
        )
        self.assertEqual(evidence, [{"paper_id": "paper", "source_type": "figure", "locator": {"page": 2, "figure_id": "Figure 1"}}])

    def test_candidate_paper_id_canonicalizes_to_title_in_table_cell(self):
        from .parser import postprocess_table_rows
        rows, errors = postprocess_table_rows(
            [{"Paper Title": "paper_001"}], ["Paper Title"], {},
            [{"paper_id": "paper_001", "title": "Canonical Paper Title"}],
        )
        self.assertEqual(rows, [{"Paper Title": "Canonical Paper Title"}])
        self.assertIn("table_paper_title_row_key_canonicalized", errors)

    def test_micro_proposition_selects_query_relevant_sentence_not_prefix(self):
        record = {
            "text": "Proceedings of the conference. We use an AdamW optimizer with learning rate 0.001 for all runs.",
            "source_type": "text_span",
        }
        proposition = _query_aware_extractive_proposition(record, {"optimizer", "learning", "rate"}, 80)
        self.assertIn("AdamW", proposition)
        self.assertIn("0.001", proposition)

    def test_query_aware_micro_order_keeps_one_anchor_per_paper_first(self):
        hierarchy = self._hierarchy()
        second_paper = dict(self.anchor)
        second_paper.update({"paper_id": "paper_two", "global_record_id": "paper_two::r1", "text": "MCTS is used in the framework.", "selection_rank": 2})
        hierarchy["l0_catalog"].append({**second_paper, "evidence_ref": "E0004", "role": "selected_anchor"})
        hierarchy["selected_anchor_refs"] = ["E0002", "E0004"]
        hierarchy["task_family"] = "multi_paper"
        hierarchy["primary_evidence_type"] = "text_span"
        hierarchy["keyed_micro_order"] = "query_aware"
        projection = hierarchy_prompt_projection(hierarchy)
        refs = [row["support_ref"] for row in projection["l2_micro_evidence"]]
        self.assertIn("E0004", refs[:2])

    def test_table_micro_proposition_keeps_relevant_row_values(self):
        hierarchy = self._hierarchy()
        hierarchy["query_claims"] = [{"claim_id": "Q01", "claim": "What F1 does Ours achieve?"}]
        hierarchy["prompt_micro_text_chars"] = 120
        projection = hierarchy_prompt_projection(hierarchy)
        table_row = next(row for row in projection["l2_micro_evidence"] if row["support_ref"] == "E0002")
        self.assertIn("Ours", table_row["extractive_proposition"])
        self.assertIn("91.2", table_row["extractive_proposition"])

    def test_table_card_proposition_keeps_header_and_cell_together(self):
        hierarchy = attach_cards("What F1 does Ours achieve?", self._hierarchy(), mode="extractive", max_claims=6, max_cards=3, primary_evidence_type="table")
        card = next(card for card in hierarchy["l2_evidence_cards"] if card.get("source_type") == "table")
        self.assertIn("Columns: Method | F1", card["proposition"])
        self.assertIn("Row Ours: Method=Ours; F1=91.2", card["proposition"])

    def test_table_view_keeps_group_condition_with_repeated_method(self):
        table = dict(self.anchor)
        table["table_structure"] = {
            "caption": "FID by step count.",
            "header_rows": [["Method", "FID"]],
            "columns": ["Method", "FID"],
            "rows": [
                {"row_index": 1, "row_label": "1-step", "values": {"Method": "1-step", "FID": ""}, "row_section": True},
                {"row_index": 2, "row_label": "TCM", "values": {"Method": "TCM", "FID": "2.46"}, "row_section": False},
                {"row_index": 3, "row_label": "2-step", "values": {"Method": "2-step", "FID": ""}, "row_section": True},
                {"row_index": 4, "row_label": "TCM", "values": {"Method": "TCM", "FID": "2.05"}, "row_section": False},
            ],
            "cells": [],
            "footnotes": [],
        }
        hierarchy = build_l0_l1_l3(
            [table], [{"paper_id": "paper", "title": "Paper"}], {"paper": [table]},
            l1_max_chars=240, l3_paper_chars=100,
        )
        hierarchy = attach_cards("What is the 2-step FID for TCM?", hierarchy, mode="extractive", max_claims=3, max_cards=3, primary_evidence_type="table")
        card = hierarchy["l2_evidence_cards"][0]
        self.assertIn("Group 2-step; Row TCM", card["proposition"])
        self.assertLess(card["proposition"].index("FID=2.05"), card["proposition"].index("FID=2.46"))

    def test_selected_anchor_is_refreshed_from_current_processed_record(self):
        stale = dict(self.anchor)
        stale.pop("table_structure", None)
        current = dict(self.anchor)
        current["text"] = "Table 1: current structured text"
        current["table_structure"] = {
            "format": "docling_table_cells_v1",
            "caption": "Table 1: current structured text",
            "header_rows": [["Method", "F1"]],
            "columns": ["Method", "F1"],
            "rows": [{"row_index": 1, "row_label": "Ours", "values": {"Method": "Ours", "F1": "91.2"}}],
            "cells": [],
            "footnotes": ["† uses validation data."],
        }
        hierarchy = build_l0_l1_l3(
            [stale], [{"paper_id": "paper", "title": "Paper"}],
            {"paper": [self.before, current, self.after]}, l1_max_chars=240, l3_paper_chars=100,
        )
        table = next(row for row in hierarchy["l0_catalog"] if row["global_record_id"] == "paper::r2")
        self.assertEqual(table["text"], "Table 1: current structured text")
        self.assertEqual(table["table_structure"]["format"], "docling_table_cells_v1")

    def test_keyed_micro_budget_does_not_charge_hidden_section_path(self):
        hierarchy = self._hierarchy()
        hierarchy["keyed_micro_index_chars"] = 1000
        hierarchy["keyed_micro_text_chars"] = 80
        hierarchy["l0_catalog"][1]["section_path"] = ["A very long hidden section path " * 20]
        keyed = keyed_hierarchy_prompt_projection(hierarchy)
        self.assertTrue(keyed["l2_micro_rows"])

    def test_zero_keyed_micro_budget_means_unbounded_before_prompt_fit(self):
        hierarchy = self._hierarchy()
        hierarchy["keyed_micro_index_chars"] = 0
        hierarchy["keyed_micro_text_chars"] = 80
        keyed = keyed_hierarchy_prompt_projection(hierarchy)
        self.assertTrue(keyed["l2_micro_rows"])


if __name__ == "__main__":
    unittest.main()
