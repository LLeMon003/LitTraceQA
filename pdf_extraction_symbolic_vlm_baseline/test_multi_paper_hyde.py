import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde import (
    MultiPaperHyDEConfig,
    _generation_cache_path,
    apply_multi_paper_hyde,
    fuse_scores,
    minmax_normalize,
    validate_claims,
)
from pdf_extraction_symbolic_vlm_baseline.section_relevance import SectionRelevanceConfig
from pdf_extraction_symbolic_vlm_baseline.symbolic_context_selector import select_symbolic_contexts


def _section(index, score):
    paper = f"paper_{index // 2}"
    section_id = f"sec_{index}"
    record = {
        "paper_id": paper,
        "global_record_id": f"{paper}::r{index}",
        "record_id": f"r{index}",
        "record_type": "paragraph",
        "source_type": "text_span",
        "text": f"section evidence {index}",
        "page": index + 1,
    }
    return {
        "paper_id": paper,
        "section_id": section_id,
        "section_title": "Experiments",
        "section_type": "experiments",
        "page_start": index + 1,
        "page_end": index + 1,
        "artifact_fingerprint": paper,
        "records": [record],
        "record_ids": [record["global_record_id"]],
        "assessment": {"relevance": {"score": score, "rank": index + 1, "backend": "llmrerank"}},
    }


def _relevance():
    sections = [_section(0, 0.9), _section(1, 0.8), _section(2, 0.2)]
    return {
        "ranked_sections": sections,
        "selected_sections": sections[:2],
        "expanded_records": [record for section in sections[:2] for record in section["records"]],
        "trace": {
            "sections": [
                {
                    "paper_id": section["paper_id"],
                    "section_id": section["section_id"],
                    "record_ids": section["record_ids"],
                    "assessment": copy.deepcopy(section["assessment"]),
                    "selected": index < 2,
                    "expanded_record_count": 1 if index < 2 else 0,
                }
                for index, section in enumerate(sections)
            ],
            "selected_section_ids": [f"{section['paper_id']}::{section['section_id']}" for section in sections[:2]],
            "expanded_record_ids": [section["record_ids"][0] for section in sections[:2]],
        },
    }


class _Tokenizer:
    version = "test"

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class MultiPaperHyDETests(unittest.TestCase):
    def test_valid_decomposition_deduplicates_and_rescues_source_types(self):
        raw = json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "x",
                        "hypothetical_evidence": "The requested metric is [VALUE].",
                        "expected_source_types": ["table", "unsupported"],
                    },
                    {
                        "claim_id": "y",
                        "hypothetical_evidence": "The requested metric is [VALUE].",
                        "expected_source_types": ["table"],
                    },
                    {
                        "claim_id": "z",
                        "hypothetical_evidence": "The method uses [OPTIMIZER].",
                        "expected_source_types": [],
                    },
                ]
            }
        )
        claims, warnings = validate_claims(raw, 4)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["expected_source_types"], ["table"])
        self.assertEqual(set(claims[1]["expected_source_types"]), {
            "citation_context", "equation_algorithm", "figure", "table", "text_span"
        })
        self.assertTrue(any("duplicate" in warning for warning in warnings))
        self.assertTrue(any("unsupported" in warning for warning in warnings))

    def test_malformed_or_unmasked_decomposition_fails(self):
        with self.assertRaises(ValueError):
            validate_claims("not json", 4)
        with self.assertRaises(ValueError):
            validate_claims('{"claims":[{"hypothetical_evidence":"The answer is Adam."}]}', 4)

    def test_score_normalization_and_weighted_fusion(self):
        self.assertEqual(minmax_normalize([2.0, 4.0]), [0.0, 1.0])
        original, hyde, fused = fuse_scores([0.9, 0.1], [0.0, 1.0], 0.7, 0.3)
        self.assertEqual(original, [1.0, 0.0])
        self.assertEqual(hyde, [0.0, 1.0])
        self.assertEqual(fused, [0.7, 0.3])
        _, _, baseline = fuse_scores([0.9, 0.1], [1.0, 0.0], 1.0, 0.0)
        self.assertGreater(baseline[0], baseline[1])

    def test_cache_key_changes_with_prompt_or_schema_input(self):
        config = MultiPaperHyDEConfig()
        root = Path("/tmp/cache")
        first = _generation_cache_path(root, query_id="q", query="one", candidate_papers=["p"], config=config)
        second = _generation_cache_path(root, query_id="q", query="two", candidate_papers=["p"], config=config)
        third = _generation_cache_path(root, query_id="q", query="one", candidate_papers=["p", "p2"], config=config)
        with patch("pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.PROMPT_VERSION", "changed"):
            fourth = _generation_cache_path(root, query_id="q", query="one", candidate_papers=["p"], config=config)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertNotEqual(first, fourth)

    def test_single_paper_path_does_not_invoke_hyde(self):
        baseline = _relevance()
        with patch(
            "pdf_extraction_symbolic_vlm_baseline.symbolic_context_selector.retrieve_section_relevance",
            return_value=baseline,
        ), patch(
            "pdf_extraction_symbolic_vlm_baseline.symbolic_context_selector.apply_multi_paper_hyde",
        ) as hyde:
            result = select_symbolic_contexts(
                query="q",
                candidate_records=[baseline["expanded_records"][0]],
                processed_root=".",
                parser_model_slug="docling",
                context_selection_mode="section_relevance",
                section_relevance_config=SectionRelevanceConfig(backend="llmrerank"),
                section_relevance_top_k=2,
                multi_paper_hyde_config=MultiPaperHyDEConfig(enabled=True),
                is_multi_paper_task=False,
                hyde_client=Mock(),
            )
        hyde.assert_not_called()
        self.assertEqual(result["selected_record_count"], 2)

    def test_disabled_and_non_qwen_paths_preserve_baseline(self):
        baseline = _relevance()
        self.assertIs(
            apply_multi_paper_hyde(
                relevance=baseline,
                query="q",
                query_id="q",
                primary_evidence_type="text_span",
                processed_root=".",
                section_config=SectionRelevanceConfig(backend="llmrerank"),
                hyde_config=MultiPaperHyDEConfig(enabled=False),
                client=Mock(),
                selection_budget=2,
            ),
            baseline,
        )
        original_ids = list(baseline["trace"]["selected_section_ids"])
        result = apply_multi_paper_hyde(
            relevance=baseline,
            query="q",
            query_id="q",
            primary_evidence_type="text_span",
            processed_root=".",
            section_config=SectionRelevanceConfig(backend="bm25"),
            hyde_config=MultiPaperHyDEConfig(enabled=True),
            client=Mock(),
            selection_budget=2,
        )
        self.assertEqual(result["trace"]["selected_section_ids"], original_ids)
        self.assertTrue(result["trace"]["hyde"]["fallback_used"])

    def test_generation_failure_falls_back_to_complete_baseline(self):
        baseline = _relevance()
        before = copy.deepcopy(baseline["trace"]["selected_section_ids"])
        with patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.generate_claims",
            side_effect=ValueError("malformed"),
        ):
            result = apply_multi_paper_hyde(
                relevance=baseline,
                query="q",
                query_id="q",
                primary_evidence_type="text_span",
                processed_root=".",
                section_config=SectionRelevanceConfig(backend="llmrerank"),
                hyde_config=MultiPaperHyDEConfig(enabled=True),
                client=Mock(),
                selection_budget=2,
            )
        self.assertEqual(result["trace"]["selected_section_ids"], before)
        self.assertEqual(result["trace"]["hyde"]["fallback_stage"], "generation_or_validation")

    def test_fusion_keeps_fixed_budget_and_writes_complete_audit(self):
        baseline = _relevance()
        claims = [{
            "claim_id": "claim_1",
            "hypothetical_evidence": "The requested result is [VALUE].",
            "expected_source_types": ["citation_context"],
        }]
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.generate_claims",
            return_value=("{}", claims, [], False),
        ), patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde._load_tokenizer",
            return_value=_Tokenizer(),
        ), patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.e5_score_queries",
            return_value=([[0.1, 0.2, 0.99]], [False, False, False], "test"),
        ):
            result = apply_multi_paper_hyde(
                relevance=baseline,
                query="q",
                query_id="q",
                primary_evidence_type="text_span",
                processed_root=directory,
                section_config=SectionRelevanceConfig(backend="llmrerank"),
                hyde_config=MultiPaperHyDEConfig(enabled=True, original_weight=0.2, claim_weight=0.8),
                client=Mock(),
                selection_budget=2,
            )
        self.assertEqual(len(result["selected_sections"]), 2)
        self.assertEqual(result["selected_sections"][0]["section_id"], "sec_2")
        audit = result["trace"]["hyde"]
        for key in (
            "parsed_claims", "original_qwen_section_scores", "per_claim_retrieval_hits",
            "hyde_section_scores", "normalization_trace", "fusion_weights",
            "fused_section_scores", "baseline_rank_vs_fused_rank", "selected_sections",
            "selection_budget", "fallback_used",
        ):
            self.assertIn(key, audit)
        self.assertFalse(audit["fallback_used"])
        self.assertEqual(audit["selection_budget"], 2)
        self.assertEqual(audit["routing_rescue_count"], 1)

    def test_record_aware_fusion_reselects_units_with_fused_section_score(self):
        baseline = _relevance()
        for section_index, section in enumerate(baseline["ranked_sections"]):
            chunk = {
                "paper_id": section["paper_id"], "section_id": section["section_id"],
                "section_index": section_index, "chunk_index": 0, "unit_type": "text_chunk",
                "record_ids": section["record_ids"], "anchor_record_ids": [],
                "score_contract": {"local_relevance": 0.5, "local_rank": 0, "final_rank": 0},
            }
            section["chunks"] = [chunk]
            section["_context_chunk_indexes"] = [0]
            section["_context_record_ids"] = section["record_ids"]
            baseline["trace"]["sections"][section_index]["chunks"] = [{"chunk_index": 0}]
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.generate_claims",
            return_value=("{}", [{"claim_id": "claim_1", "hypothetical_evidence": "The result is [VALUE].", "expected_source_types": ["citation_context"]}], [], False),
        ), patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde._load_tokenizer", return_value=_Tokenizer(),
        ), patch(
            "pdf_extraction_symbolic_vlm_baseline.multi_paper_hyde.e5_score_queries",
            return_value=([[0.1, 0.2, 0.99]], [False, False, False], "test"),
        ):
            result = apply_multi_paper_hyde(
                relevance=baseline, query="q", query_id="q", primary_evidence_type="text_span",
                processed_root=directory,
                section_config=SectionRelevanceConfig(
                    backend="llmrerank", unit_mode="record_aware", retrieval_unit_top_k=1,
                ),
                hyde_config=MultiPaperHyDEConfig(enabled=True, original_weight=0.2, claim_weight=0.8),
                client=Mock(), selection_budget=2,
            )
        self.assertEqual([section["section_id"] for section in result["selected_sections"]], ["sec_2"])
        self.assertEqual(result["trace"]["ranked_units"][0]["section_id"], "sec_2")
        self.assertEqual(result["trace"]["expanded_record_ids"], ["paper_1::r2"])


if __name__ == "__main__":
    unittest.main()
