import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pdf_docling_rerank_selection_generation_pipeline.paper_conditioned_claims import (
    PaperConditionedClaimsConfig,
    generate_evidence_plan,
)


class EvidencePlanningTests(unittest.TestCase):
    def test_plan_keeps_candidate_ids_and_source_types(self):
        client = Mock()
        client.generate_prediction.return_value = {"content": json.dumps({
            "cross_paper": True,
            "plans": [
                {"paper_id": "a", "source_types": ["table", "bad"], "retrieval_query": "benchmark metric"},
                {"paper_id": "outside", "source_types": ["figure"], "retrieval_query": "ignored"},
            ],
        })}
        with tempfile.TemporaryDirectory() as directory:
            plan, warnings, cached = generate_evidence_plan(
                query_id="q", query="question", primary_evidence_type="table",
                candidate_papers=[{"paper_id": "a", "title": "A", "abstract": ""}],
                config=PaperConditionedClaimsConfig(enabled=True, cache_enabled=False),
                client=client, cache_root=Path(directory),
            )
        self.assertFalse(cached)
        self.assertTrue(plan["cross_paper"])
        self.assertEqual(plan["plans"], [{"paper_id": "a", "source_types": ["table"], "retrieval_query": "benchmark metric"}])
        self.assertFalse(warnings)


if __name__ == "__main__":
    unittest.main()
