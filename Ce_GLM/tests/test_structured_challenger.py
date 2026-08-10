import unittest

from src.structured_challenger import StructuredSourceIndex, score
from src.hybrid_challenger import ProvenanceCoalescingHybridIndex


class StructuredChallengerTests(unittest.TestCase):
    def test_solves_unique_fact_and_table_and_rejects_ambiguity(self):
        facts = [{"ambiguity_status": "accepted", "paper_id": "p1", "page": 1, "object_type": "equation", "normalized_value": "x=y", "object_uid": "o"}]
        cells = [
            {"provenance_status": "accepted", "paper_id": "p2", "page": 2, "evaluator_visible_table_id": "Table 1", "row_index": 1, "column_index": 2, "normalized_cell_value": "42", "is_column_header": False, "is_row_header": False},
            {"provenance_status": "accepted", "paper_id": "p2", "page": 2, "evaluator_visible_table_id": "Table 1", "row_index": 3, "column_index": 2, "normalized_cell_value": "a", "is_column_header": False, "is_row_header": False},
            {"provenance_status": "accepted", "paper_id": "p2", "page": 2, "evaluator_visible_table_id": "Table 1", "row_index": 3, "column_index": 2, "normalized_cell_value": "b", "is_column_header": False, "is_row_header": False},
        ]
        index = StructuredSourceIndex(facts, cells)
        self.assertEqual("x=y", index.solve("What exact source text is recorded by the equation in paper p1 on page 1?").text)
        self.assertEqual("42", index.solve("In paper p2, page 2, Table 1, what value appears at zero-based row 1, column 2?").text)
        self.assertEqual("not_unique_or_missing", index.solve("In paper p2, page 2, Table 1, what value appears at zero-based row 3, column 2?").status)

    def test_score_uses_question_only(self):
        index = StructuredSourceIndex([], [{"provenance_status": "accepted", "paper_id": "p", "page": 1, "evaluator_visible_table_id": "Table 1", "row_index": 1, "column_index": 1, "normalized_cell_value": "vv", "is_column_header": False, "is_row_header": False}])
        result = score([{"recipe": "table_coordinate_lookup", "split": "holdout", "question": "In paper p, page 1, Table 1, what value appears at zero-based row 1, column 1?", "answer": {"text": "vv"}}], index)
        self.assertEqual(1.0, result["by_split_recipe"]["holdout:table_coordinate_lookup"]["exact_match"])

    def test_hybrid_coalesces_identical_values_but_rejects_conflicts(self):
        base = {"provenance_status": "accepted", "paper_id": "p", "page": 1, "evaluator_visible_table_id": "Table 1", "row_index": 1, "column_index": 1, "is_column_header": False, "is_row_header": False}
        same = ProvenanceCoalescingHybridIndex([], [{**base, "normalized_cell_value": "42", "record_hash": "a"}, {**base, "normalized_cell_value": "42", "record_hash": "b"}])
        conflict = ProvenanceCoalescingHybridIndex([], [{**base, "normalized_cell_value": "42"}, {**base, "normalized_cell_value": "43"}])
        question = "In paper p, page 1, Table 1, what value appears at zero-based row 1, column 1?"
        self.assertEqual("42", same.solve(question).text)
        self.assertEqual("not_unique_or_missing", conflict.solve(question).status)


if __name__ == "__main__":
    unittest.main()
