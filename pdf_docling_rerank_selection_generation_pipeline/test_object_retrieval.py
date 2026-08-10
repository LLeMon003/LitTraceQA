from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from . import object_retrieval


VER3_ROOT = Path(__file__).resolve().parents[1] / "GroundLM-Ver3-ver3-cache-exact-complete-solution-001"


class ObjectRetrievalTests(unittest.TestCase):
    def test_paper_constraint_and_answer_bearing_recall(self) -> None:
        good = {"paper_id": "p", "record_hash": "wanted", "source_hash": "s", "page": 1,
                "evaluator_visible_table_id": "Table 1", "row_index": 2, "column_index": 3,
                "normalized_cell_value": "42"}
        other = {**good, "paper_id": "other", "record_hash": "wrong"}
        record = {"question": "Which value is reported in Table 1 at row 2 and column 3?",
                  "source_paper": "p", "reasoning_operator": "structured_lookup",
                  "source_objects": [{"object_id": "wanted"}]}
        self.assertEqual(object_retrieval.retrieve(record["question"], "p", [good, other])[0]["object_id"], "wanted")
        self.assertEqual(object_retrieval.score([record], [good, other])["answer_bearing_recall"], 1.0)

    def test_fact_object_uid_is_the_retrieval_identity(self) -> None:
        fact = {"paper_id": "p", "object_uid": "fact-wanted", "record_hash": "audit-hash", "source_hash": "s",
                "page": 3, "object_type": "equation_block", "normalized_value": "x"}
        record = {"question": "According to the equation_block in paper p on page 3, what text is stated?",
                  "source_paper": "p", "reasoning_operator": "direct_extraction",
                  "source_objects": [{"object_id": "fact-wanted"}]}
        self.assertEqual(object_retrieval.score([record], [fact])["answer_bearing_recall"], 1.0)

    @unittest.skipUnless(VER3_ROOT.is_dir(), "Ver3 solution directory not present")
    def test_port_behavior_is_identical_to_ver3_original(self) -> None:
        """Feed identical inputs to the Ver3 original and the port; outputs must match."""
        sys.path.insert(0, str(VER3_ROOT))
        try:
            ver3 = importlib.import_module("src.target_retrieval")
        finally:
            sys.path.pop(0)
        corpus = [
            {"paper_id": "p", "record_hash": "wanted", "source_hash": "s", "page": 1,
             "evaluator_visible_table_id": "Table 1", "row_index": 2, "column_index": 3,
             "normalized_cell_value": "42", "table_caption": "Results"},
            {"paper_id": "p", "object_uid": "fact-wanted", "record_hash": "audit-hash", "source_hash": "s",
             "page": 3, "object_type": "equation_block", "normalized_value": "x"},
            {"paper_id": "other", "record_hash": "wrong", "source_hash": "s", "page": 1,
             "normalized_cell_value": "99"},
        ]
        records = [
            {"question": "Which value is reported in Table 1 at row 2 and column 3?",
             "source_paper": "p", "reasoning_operator": "structured_lookup",
             "source_objects": [{"object_id": "wanted"}]},
            {"question": "According to the equation_block in paper p on page 3, what text is stated?",
             "source_paper": "p", "reasoning_operator": "direct_extraction",
             "source_objects": [{"object_id": "fact-wanted"}]},
        ]
        self.assertEqual(object_retrieval.score(records, corpus), ver3.score(records, corpus))
        for record in records:
            self.assertEqual(
                [x["object_id"] for x in object_retrieval.retrieve(record["question"], record["source_paper"], corpus)],
                [x["object_id"] for x in ver3.retrieve(record["question"], record["source_paper"], corpus)],
            )


if __name__ == "__main__":
    unittest.main()
