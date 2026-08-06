import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pdf_docling_rerank_selection_generation_pipeline.multi_paper_hyde import MultiPaperHyDEConfig, _generation_cache_path, validate_claims
from pdf_docling_rerank_selection_generation_pipeline.paper_conditioned_claims import PaperConditionedClaimsConfig
from pdf_docling_rerank_selection_generation_pipeline.section_relevance import SectionRelevanceConfig
from pdf_docling_rerank_selection_generation_pipeline.symbolic_context_selector import select_symbolic_contexts


class MultiPaperHyDETests(unittest.TestCase):
    def test_claim_validation_deduplicates_and_routes(self):
        claims, warnings = validate_claims(json.dumps({"claims": [
            {"hypothetical_evidence": "The requested metric is [VALUE].", "expected_source_types": ["table", "unsupported"]},
            {"hypothetical_evidence": "The requested metric is [VALUE].", "expected_source_types": ["table"]},
        ]}), 4)
        self.assertEqual(claims[0]["expected_source_types"], ["table"])
        self.assertEqual(len(claims), 1)
        self.assertTrue(warnings)

    def test_cache_key_includes_query_and_candidates(self):
        config = MultiPaperHyDEConfig()
        first = _generation_cache_path(Path("/tmp/cache"), query_id="q", query="one", candidate_papers=["p"], config=config)
        second = _generation_cache_path(Path("/tmp/cache"), query_id="q", query="two", candidate_papers=["p"], config=config)
        self.assertNotEqual(first, second)

    def test_single_paper_never_generates_hyde_routes(self):
        record = {"paper_id": "p", "global_record_id": "p::r", "source_type": "text_span", "text": "evidence", "page": 1}
        trace = {"ranked_units": [{"record_ids": ["p::r"], "score_contract": {"local_relevance": 0.9}}], "sections": []}
        relevance = {"ranked_sections": [], "selected_sections": [], "expanded_records": [record], "trace": trace}
        with patch("pdf_docling_rerank_selection_generation_pipeline.symbolic_context_selector.retrieve_section_relevance", return_value=relevance), patch("pdf_docling_rerank_selection_generation_pipeline.symbolic_context_selector.generate_evidence_plan") as generate:
            result = select_symbolic_contexts("q", [record], ".", "docling", context_selection_mode="section_relevance", section_relevance_config=SectionRelevanceConfig(backend="llmrerank"), multi_paper_hyde_config=MultiPaperHyDEConfig(enabled=True), is_multi_paper_task=False, hyde_client=Mock())
        generate.assert_not_called()
        self.assertEqual(result["selected_record_count"], 1)

    def test_multi_paper_plan_becomes_a_routing_requirement(self):
        direct = {"paper_id": "p1", "global_record_id": "p1::direct", "source_type": "text_span", "text": "direct answer", "page": 1}
        auxiliary = {"paper_id": "p2", "global_record_id": "p2::aux", "source_type": "text_span", "text": "baseline schedule", "page": 1}
        relevance = {
            "ranked_sections": [],
            "selected_sections": [],
            "expanded_records": [direct, auxiliary],
            "trace": {
                "ranked_units": [
                    {"record_ids": ["p1::direct"], "score_contract": {"local_relevance": 0.9}},
                    {"record_ids": ["p2::aux"], "score_contract": {"local_relevance": 0.1}},
                ],
                "sections": [],
            },
        }
        with patch("pdf_docling_rerank_selection_generation_pipeline.symbolic_context_selector.retrieve_section_relevance", return_value=relevance), patch(
            "pdf_docling_rerank_selection_generation_pipeline.symbolic_context_selector.generate_evidence_plan",
            return_value=({"cross_paper": True, "plans": [{"paper_id": "p2", "source_types": ["text_span"], "retrieval_query": "baseline schedule"}]}, [], True),
        ) as generate:
            result = select_symbolic_contexts(
                "direct answer",
                [direct, auxiliary],
                ".",
                "docling",
                context_selection_mode="section_relevance",
                section_relevance_config=SectionRelevanceConfig(backend="llmrerank"),
                paper_conditioned_claims_config=PaperConditionedClaimsConfig(enabled=True),
                is_multi_paper_task=True,
                hyde_client=Mock(),
                evidence_package_budget=2,
                evidence_package_min_budget=2,
                multi_paper_min_distinct_papers=2,
            )
        generate.assert_called_once()
        self.assertEqual(result["section_relevance_trace"]["paper_conditioned_claims"]["mode"], "evidence_planning")
        self.assertEqual(result["section_relevance_trace"]["package_selection"]["paper_conditioned_claim_route_count"], 1)
        self.assertEqual({record["paper_id"] for record in result["selected_evidence"]}, {"p1", "p2"})


if __name__ == "__main__":
    unittest.main()
