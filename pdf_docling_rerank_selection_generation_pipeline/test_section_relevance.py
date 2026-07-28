import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf_docling_rerank_selection_generation_pipeline.section_relevance import (
    SectionRelevanceConfig,
    _prefix,
    _record_line,
    query_object_targets,
    retrieve_section_relevance,
)


def _record(identifier, source_type, text, *, page=1, label="", order=1):
    return {
        "paper_id": "paper", "global_record_id": identifier, "section_id": "method",
        "section_title": "Method", "section_type": "method", "section_path": ["Method"],
        "source_type": source_type, "record_type": "text" if source_type == "text_span" else source_type,
        "page": page, "label": label, "locator": {"page": page}, "text": text, "document_order": order,
    }


class SectionRelevanceTests(unittest.TestCase):
    def test_query_object_targets_preserve_explicit_locators(self):
        targets = query_object_targets("Compare Table 2, Fig. 4, Eq. (6), and the 24th reference.")
        self.assertEqual(targets["table"], 2)
        self.assertEqual(targets["figure"], 4)
        self.assertEqual(targets["equation_algorithm"], 6)
        self.assertEqual(targets["citation_context"], 24)

    def test_bm25_object_unit_uses_explicit_object_narration_before_neighbors(self):
        records = [
            _record("p::before", "text_span", "before the table", order=1),
            _record("p::table", "table", "target metric 42", label="Table 1", order=2),
            _record("p::after", "text_span", "after the table", order=3),
            _record("p::narration", "text_span", "Table 1 reports the target metric of 42.", page=2, order=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = retrieve_section_relevance(
                "What is in Table 1?", records, Path(directory),
                SectionRelevanceConfig(backend="bm25", unit_target_tokens=32, unit_max_tokens=64, object_neighbor_records=1),
            )
        object_unit = next(item for item in result["trace"]["ranked_units"] if item["unit_type"] == "object_table")
        self.assertEqual(object_unit["anchor_record_ids"], ["p::table"])
        self.assertEqual(object_unit["record_ids"], ["p::table", "p::narration"])

    def test_compact_pair_projection_omits_internal_provenance_fields(self):
        section = {
            "paper_id": "internal-paper-id", "paper_title": "", "paper_aliases": [],
            "section_title": "Experiments", "section_type": "results", "section_path": ["3", "Experiments"],
            "page_start": 4, "page_end": 9, "object_labels": ["Table 1", "Figure 2"],
        }
        record = _record("internal::r42", "table", "Table 1 has F1 91.2.", page=5, label="Table 1", order=42)
        record["bbox"] = [0, 0, 1, 1]
        record["parser_confidence"] = 0.2
        projection = _prefix(section) + _record_line(record)
        for forbidden in ("internal-paper-id", "internal::r42", "Pages:", "Objects:", "locator", "bbox", "parser_confidence"):
            self.assertNotIn(forbidden, projection)
        self.assertIn("Section: Experiments", projection)
        self.assertIn("[table: Table 1]", projection)

    def test_trace_reports_compact_pair_token_cost(self):
        records = [_record("p::one", "text_span", "A concise relevant paragraph.")]
        config = SectionRelevanceConfig(
            backend="llmrerank", llmrerank_api_key="configured", unit_target_tokens=32, unit_max_tokens=64,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_docling_rerank_selection_generation_pipeline.section_relevance.LLMRerankClient.score_documents",
            return_value={"scores": [0.5], "raw_response": {}, "attempts": 1},
        ):
            result = retrieve_section_relevance(
                "What is relevant?", records, Path(directory), config,
            )
        stats = result["trace"]["llmrerank"]
        self.assertEqual(stats["logical_text_pair_count"], 1)
        self.assertGreater(stats["logical_text_pair_tokens"], 0)
        unit = result["ranked_sections"][0]["chunks"][0]
        self.assertNotIn("Paper ID:", unit["text"])
        self.assertNotIn("[p::one]", unit["text"])

    def test_qwen_scores_every_unit_without_sparse_prefilter(self):
        records = [
            _record("p::one", "text_span", "first record", order=1),
            _record("p::two", "figure", "Figure 1: target", label="Figure 1", order=2),
        ]
        config = SectionRelevanceConfig(
            backend="llmrerank",
            llmrerank_api_key="configured",
            llmrerank_instruction_version="v3_complete_support",
            unit_target_tokens=32,
            unit_max_tokens=64,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_docling_rerank_selection_generation_pipeline.section_relevance.LLMRerankClient.score_documents",
            return_value={"scores": [0.2, 0.8], "raw_response": {}, "attempts": 1},
        ):
            result = retrieve_section_relevance("target", records, Path(directory), config)
        trace = result["trace"]
        self.assertEqual(trace["candidate_union"]["mode"], "full_qwen")
        self.assertEqual(trace["candidate_union"]["qwen_scored_unit_count"], len(trace["ranked_units"]))
        self.assertEqual(set(trace["candidate_union"]["qwen_scored_record_ids"]), {"p::one", "p::two"})
        self.assertEqual(trace["candidate_union"]["qwen_unscored_record_ids"], [])
        self.assertEqual(trace["llmrerank"]["failed_pair_count"], 0)
        self.assertEqual(trace["llmrerank"]["instruction_version"], "v3_complete_support")

    def test_candidate_paper_identity_is_included_in_unit_projection(self):
        records = [_record("imm::one", "text_span", "kernel width details")]
        records[0]["paper_id"] = "imm"
        with tempfile.TemporaryDirectory() as directory:
            result = retrieve_section_relevance(
                "What does IMM use?",
                records,
                Path(directory),
                SectionRelevanceConfig(backend="bm25", unit_target_tokens=32, unit_max_tokens=64, llmrerank_include_paper_identity=True),
                candidate_paper_metadata=[{"paper_id": "imm", "title": "Inductive Moment Matching"}],
            )
        text = result["ranked_sections"][0]["chunks"][0]["text"]
        self.assertIn("Paper: Inductive Moment Matching", text)
        self.assertIn("Method aliases: IMM", text)

    def test_rejects_deleted_backends_and_prefilter_variants(self):
        with self.assertRaisesRegex(ValueError, "bm25 or llmrerank"):
            retrieve_section_relevance("q", [], ".", SectionRelevanceConfig(backend="e5_base_v2"))
        with self.assertRaisesRegex(ValueError, "none or bm25"):
            retrieve_section_relevance("q", [], ".", SectionRelevanceConfig(backend="llmrerank", llmrerank_failure_fallback="hybrid"))


if __name__ == "__main__":
    unittest.main()
