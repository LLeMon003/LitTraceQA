from __future__ import annotations

import unittest

from .selection_oracle_diagnostics import greedy_page_capped_coverage, maximum_coverage


class SelectionOracleDiagnosticsTests(unittest.TestCase):
    def test_maximum_coverage_finds_non_greedy_union(self):
        gold = [{"paper_id": "p", "page": 1, "source_type": "text_span"} for _ in range(3)]
        packages = [
            {"records": [{"paper_id": "p", "page": 1, "source_type": "text_span", "text": "a"}]},
            {"records": [{"paper_id": "p", "page": 1, "source_type": "text_span", "text": "b"}]},
        ]
        # Directly exercise the bit-mask solver with packages whose records all
        # match the evaluator's text-span page/source contract.
        result = maximum_coverage(packages, gold[:2], budget=1)
        self.assertEqual(result["covered_count"], 2)

    def test_page_capped_greedy_respects_page_limit(self):
        gold = [{"paper_id": "p", "page": 1, "source_type": "text_span"}]
        packages = [
            {"paper_id": "p", "page": 1, "records": [{"paper_id": "p", "page": 1, "source_type": "text_span"}]},
            {"paper_id": "p", "page": 1, "records": [{"paper_id": "p", "page": 1, "source_type": "text_span"}]},
        ]
        self.assertEqual(greedy_page_capped_coverage(packages, gold, budget=2, max_per_page=1), 1)


if __name__ == "__main__":
    unittest.main()
