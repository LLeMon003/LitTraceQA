"""Tests for query-visible task structure after the 2026-08-05 schema change."""
from __future__ import annotations

import unittest

from .task_structure import (
    as_source_types,
    derive_task_structure,
    explicit_source_type_mentions,
    fallback_is_multi_paper,
)


class TaskStructureTest(unittest.TestCase):
    def test_new_format_sample_without_gold_fields(self) -> None:
        sample = {
            "query_id": "q_001",
            "question": "Among the two prompt compression methods, how much does 500xCompressor outperform ICAE on NaturalQ?",
            "answer_types": ["freeform", "multiple_choice"],
            "multiple_choice_options": [{"label": "A", "text": "1"}],
        }
        structure = derive_task_structure(sample)
        self.assertFalse(structure.is_multi_paper)
        self.assertEqual(structure.preferred_source_types, ())
        self.assertIsNone(structure.inferred_paper_count)

    def test_table_contract_does_not_determine_routing_or_source(self) -> None:
        sample = {"question": "Report the numbers.", "answer_types": ["table"]}
        structure = derive_task_structure(sample)
        self.assertFalse(structure.is_multi_paper)
        self.assertEqual(structure.preferred_source_types, ())

    def test_explicit_query_locator_is_kept(self) -> None:
        sample = {"question": "What value is in Figure 3 of the paper?", "answer_types": ["freeform"]}
        self.assertEqual(explicit_source_type_mentions(sample["question"]), ("figure",))
        structure = derive_task_structure(sample)
        self.assertEqual(structure.preferred_source_types, ("figure",))

    def test_generic_object_word_and_legacy_type_do_not_route_source(self) -> None:
        sample = {
            "question": "Summarize the table result.", "answer_types": ["table"],
            "primary_evidence_type": "table",
        }
        self.assertEqual(explicit_source_type_mentions(sample["question"]), ())
        self.assertEqual(derive_task_structure(sample).preferred_source_types, ())

    def test_plan_structured_analysis_drives_multi_routing(self) -> None:
        sample = {"question": "Compare two methods.", "answer_types": ["freeform"]}
        plan = {
            "slots": [{"id": "S001", "required_source_types": ["table"], "entities": ["IMM", "D-FINE"]}],
            "requires_cross_paper_synthesis": False,
            "query_analysis": {
                "entities": ["IMM", "D-FINE"],
                "comparison_targets": ["IMM", "D-FINE"],
                "inferred_paper_count": 2,
                "cross_paper_synthesis_required": True,
            },
        }
        structure = derive_task_structure(sample, plan)
        self.assertTrue(structure.is_multi_paper)
        self.assertEqual(structure.task_family, "multi_paper")
        self.assertEqual(structure.preferred_source_types, ("table",))
        self.assertEqual(structure.inferred_paper_count, 2)

    def test_deterministic_rules_override_llm_flag(self) -> None:
        # LLM says cross-paper not required, but two comparison targets are a
        # structured signal: the deterministic router must still route multi.
        sample = {"question": "Compare A and B.", "answer_types": ["freeform"]}
        plan = {
            "slots": [],
            "query_analysis": {
                "entities": ["A", "B"],
                "comparison_targets": ["A", "B"],
                "inferred_paper_count": 1,
                "cross_paper_synthesis_required": False,
            },
        }
        self.assertTrue(derive_task_structure(sample, plan).is_multi_paper)

    def test_validated_plan_cross_flag_overrides_raw_analysis_false(self) -> None:
        sample = {"question": "Report each method.", "answer_types": ["table"]}
        plan = {
            "slots": [{"required_source_types": ["text_span"], "entities": ["A"]}],
            "requires_cross_paper_synthesis": True,
            "query_analysis": {"entities": ["A", "B"], "comparison_targets": [], "inferred_paper_count": None, "cross_paper_synthesis_required": False},
        }
        self.assertTrue(derive_task_structure(sample, plan).is_multi_paper)

    def test_legacy_primary_type_is_not_a_source_hint(self) -> None:
        sample = {
            "question": "Report the number.",
            "answer_types": ["freeform"],
            "task_family": "hidden_source_single_paper",
            "primary_evidence_type": "table",
        }
        structure = derive_task_structure(sample)
        self.assertFalse(structure.is_multi_paper)
        self.assertEqual(structure.preferred_source_types, ())

    def test_plan_preferred_types_win_over_question(self) -> None:
        sample = {"question": "What value is in Table 2?", "answer_types": ["freeform"]}
        plan = {"slots": [{"required_source_types": ["figure"]}], "query_analysis": {}}
        self.assertEqual(derive_task_structure(sample, plan).preferred_source_types, ("figure",))

    def test_fallback_multi_patterns(self) -> None:
        self.assertTrue(fallback_is_multi_paper({"question": "Compare A and B across papers.", "answer_types": ["freeform"]}))
        self.assertTrue(fallback_is_multi_paper({"question": "List the papers that use X.", "answer_types": ["freeform"]}))
        self.assertFalse(fallback_is_multi_paper({"question": "What is the F1 score?", "answer_types": ["freeform"]}))

    def test_as_source_types_normalises_aliases(self) -> None:
        self.assertEqual(as_source_types("text"), ("text_span",))
        self.assertEqual(as_source_types(["equation", "algorithm"]), ("equation_algorithm",))
        self.assertEqual(as_source_types(None), ())


if __name__ == "__main__":
    unittest.main()
