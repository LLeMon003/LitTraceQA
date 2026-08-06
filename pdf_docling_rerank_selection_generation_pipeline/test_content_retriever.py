from __future__ import annotations

import unittest

from .content_retriever import build_retriever_pool, explicit_object_labels, hybrid_retriever_scores


class ContentRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = [
            {"unit_type": "object_figure", "text": "Figure 4: The average latency and SLO attainment rate.", "section_id": "s1"},
            {"unit_type": "object_equation_algorithm", "text": "Equation 6: J = sum ( W ( i ) )", "section_id": "s1"},
            {"unit_type": "object_table", "text": "Table 1: FID, NFE and param.", "section_id": "s1"},
            {"unit_type": "text_chunk", "text": "We report the 1-step FID of TCM on CIFAR-10 in Table 1.", "section_id": "s1"},
            {"unit_type": "text_chunk", "text": "The appendix contains unrelated proofs and page numbers.", "section_id": "s2"},
        ]
        self.sections = [
            {"section_id": "s1", "section_title": "Results"},
            {"section_id": "s2", "section_title": "Appendix"},
        ]

    def test_special_objects_always_preserved(self) -> None:
        scores = [0.0] * len(self.units)
        pool = build_retriever_pool(self.units, scores, budget=1)
        self.assertIn(0, pool)  # figure
        self.assertIn(1, pool)  # equation
        self.assertIn(2, pool)  # table

    def test_explicit_object_labels(self) -> None:
        self.assertEqual(explicit_object_labels("What does Table 4 show?"), {"table 4"})
        self.assertEqual(explicit_object_labels("How many subfigures in Figure 4?"), {"figure 4"})
        self.assertEqual(explicit_object_labels("Equation 6 and Fig. 2"), {"equation 6", "figure 2"})

    def test_hybrid_boost_ranks_explicit_label_higher(self) -> None:
        scores = hybrid_retriever_scores("What is the 1-step FID in Table 1?", self.units, self.sections)
        table_unit = 2
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), table_unit)

    def test_budget_selects_top_nonspecial(self) -> None:
        scores = [float(i) for i in range(len(self.units))]
        pool = build_retriever_pool(self.units, scores, budget=2)
        self.assertIn(4, pool)  # highest-scored text unit


if __name__ == "__main__":
    unittest.main()
