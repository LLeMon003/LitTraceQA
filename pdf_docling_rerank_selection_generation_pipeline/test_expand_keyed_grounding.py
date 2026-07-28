from __future__ import annotations

import unittest

from .expand_keyed_grounding import expand_prediction


class ExpandKeyedGroundingTests(unittest.TestCase):
    def test_adds_one_card_per_claim_without_changing_answer(self) -> None:
        hierarchy = {
            "query_claims": [{"claim_id": "Q01"}, {"claim_id": "Q02"}],
            "l0_catalog": [
                {"evidence_ref": "E0001", "paper_id": "paper_a", "source_type": "text_span", "page": 1, "locator": {"page": 1}, "text": "a"},
                {"evidence_ref": "E0002", "paper_id": "paper_b", "source_type": "table", "page": 2, "locator": {"page": 2, "table_id": "Table 1"}, "text": "b"},
                {"evidence_ref": "E0003", "paper_id": "paper_a", "source_type": "text_span", "page": 3, "locator": {"page": 3}, "text": "c"},
            ],
            "l2_evidence_cards": [
                {"claim_ids": ["Q01"], "support_refs": ["E0001"], "proposition": "a"},
                {"claim_ids": ["Q02"], "support_refs": ["E0002"], "proposition": "b"},
                {"claim_ids": ["Q01"], "support_refs": ["E0003"], "proposition": "c"},
            ],
        }
        prediction = {"query_id": "q", "gold_papers": [{"paper_id": "paper_a"}], "evidence": [], "answer": {"freeform": {"text": "unchanged"}}}
        internal = {"claim_to_support_keys": {"Q01": ["C001"]}}
        expanded, audit = expand_prediction(prediction, internal, hierarchy, task_family="multi_paper", max_cards=4, expand_papers="multi")
        self.assertEqual(expanded["answer"], prediction["answer"])
        self.assertEqual(len(expanded["evidence"]), 2)
        self.assertEqual({row["paper_id"] for row in expanded["gold_papers"]}, {"paper_a", "paper_b"})
        self.assertEqual(audit["added_evidence_count"], 2)
