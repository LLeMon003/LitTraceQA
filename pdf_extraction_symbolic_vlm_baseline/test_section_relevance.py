import http.client
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf_extraction_symbolic_vlm_baseline.section_relevance import SectionRelevanceConfig, _chunk_section, _deduplicate_candidate_records, _llmrerank_scores, context_records_for_section, deterministic_locator_only_indexes, explicit_target_unit_indexes, pool_scores, pool_section_units, query_section_targets, retrieve_section_relevance, structural_section_key
from pdf_extraction_symbolic_vlm_baseline.llmrerank_client import LLMRerankClient, LLMRerankError, _parse_scores


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        return list(range(len(tokens) + (2 if add_special_tokens else 0)))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join("token" for _ in token_ids)


class _RoundTripTokenizer:
    def encode(self, text, add_special_tokens=False):
        values = text.split()
        return [*values, "<s>", "</s>"] if add_special_tokens else values

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token for token in token_ids if token not in {"<s>", "</s>"})


class SectionRelevanceTests(unittest.TestCase):
    def test_log_mean_exp_is_bounded_by_mean_and_max(self):
        pooled = pool_scores([0.0, 1.0], "log_mean_exp", 3.0)
        self.assertGreaterEqual(pooled["log_mean_exp"], pooled["mean"])
        self.assertLessEqual(pooled["log_mean_exp"], pooled["max"])

    def test_empty_pool_is_zero(self):
        self.assertEqual(pool_scores([], "max", 3.0)["selected"], 0.0)

    def test_top_k_mean_pooling(self):
        self.assertEqual(pool_scores([0.1, 0.9, 0.7], "top_k_mean", 5.0, top_k=2)["selected"], 0.8)

    def test_section_pooling_uses_only_top_units(self):
        pooled = pool_section_units([0.9, 0.8, 0.7, 0.0, 0.0], "log_mean_exp", 5.0, 3)
        self.assertEqual(pooled["unit_count"], 5)
        self.assertEqual(pooled["aggregated_unit_count"], 3)
        self.assertGreater(pooled["selected"], 0.7)

    def test_rerank_response_is_restored_to_input_order(self):
        response = {
            "results": [
                {"index": 2, "relevance_score": 0.7},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.1},
            ]
        }
        self.assertEqual(_parse_scores(response, 3), [0.9, 0.1, 0.7])

    def test_reranker_retries_remote_disconnect_and_wraps_error(self):
        client = LLMRerankClient(
            api_key="configured",
            base_url="https://example.invalid",
            model="reranker",
            timeout_seconds=1,
            max_retries=1,
        )
        with patch(
            "pdf_extraction_symbolic_vlm_baseline.llmrerank_client.urllib.request.urlopen",
            side_effect=http.client.RemoteDisconnected("disconnected"),
        ), patch("pdf_extraction_symbolic_vlm_baseline.llmrerank_client.time.sleep"):
            with self.assertRaises(LLMRerankError) as caught:
                client.score_documents(
                    query="query",
                    documents=[{"text": "document"}],
                    instruction="instruction",
                )
        self.assertEqual(caught.exception.attempts, 2)
        self.assertIn("disconnected", str(caught.exception))

    def test_long_object_prefix_updates_original_section_trace_fields(self):
        section = {
            "paper_id": "paper",
            "section_title": "Results",
            "section_type": "results",
            "page_start": 1,
            "page_end": 2,
            "object_labels": [f"Table {index}" for index in range(100)],
            "content": "short content",
        }
        chunks = _chunk_section(section, _WhitespaceTokenizer(), SectionRelevanceConfig(chunk_max_tokens=64))
        self.assertTrue(chunks)
        self.assertIn("full_representation", section)
        self.assertEqual(len(section["object_labels"]), 100)

    def test_query_section_targets_only_explicit_section_or_reference_intent(self):
        self.assertEqual(query_section_targets("How many papers were cited in Introduction Section?"), [{"target": "introduction", "reason": "explicit_section:introduction"}])
        self.assertEqual(query_section_targets("Who is the first author of the 24th reference cited?"), [{"target": "references", "reason": "reference_index_or_count"}])
        self.assertEqual(query_section_targets("Who is the first author of the third reference cited?"), [{"target": "references", "reason": "reference_index_or_count"}])
        self.assertEqual(query_section_targets("Which method uses a frozen reference model?"), [])

    def test_explicit_reference_ordinal_forces_matching_object_unit(self):
        chunks = [
            {"unit_type": "object_citation_context", "paper_id": "p", "text": "label=Reference 23; text"},
            {"unit_type": "object_citation_context", "paper_id": "p", "text": "label=Reference 24; text"},
        ]
        self.assertEqual(explicit_target_unit_indexes("Who wrote the 24th reference?", chunks), {1})
        self.assertEqual(explicit_target_unit_indexes("What is the last reference?", chunks), {1})

    def test_explicit_numbered_objects_force_matching_units(self):
        chunks = [
            {"unit_type": "object_table", "paper_id": "p", "text": "label=Table 2; text"},
            {"unit_type": "object_figure", "paper_id": "p", "text": "label=Figure 4; text"},
            {"unit_type": "object_equation_algorithm", "paper_id": "p", "text": "label=Equation 6; text"},
            {"unit_type": "object_equation_algorithm", "paper_id": "p", "text": "label=Algorithm 3; text"},
        ]
        self.assertEqual(explicit_target_unit_indexes("Compare Table 2 and Figure 4", chunks), {0, 1})
        self.assertEqual(explicit_target_unit_indexes("What is Equation (6)?", chunks), {2})
        self.assertEqual(explicit_target_unit_indexes("Explain Algorithm 3", chunks), {3})

    def test_deterministic_locator_only_requires_named_unique_paper(self):
        chunks = [
            {"unit_type": "object_table", "paper_id": "p1", "text": "label=Table 3; text"},
            {"unit_type": "object_table", "paper_id": "p2", "text": "label=Table 3; text"},
        ]
        scoped, audit = deterministic_locator_only_indexes(
            "In Exact Paper Title, what is Table 3?",
            chunks,
            [{"paper_id": "p1", "title": "Exact Paper Title"}, {"paper_id": "p2", "title": "Another Paper"}],
        )
        self.assertEqual(scoped, {0})
        self.assertEqual(audit["status"], "active")
        scoped, audit = deterministic_locator_only_indexes(
            "What is Table 3?",
            chunks,
            [{"paper_id": "p1", "title": "Exact Paper Title"}, {"paper_id": "p2", "title": "Another Paper"}],
        )
        self.assertIsNone(scoped)
        self.assertEqual(audit["reason"], "paper_identity_not_unique")

    def test_structural_section_key_keeps_fine_grained_title_prior(self):
        self.assertEqual(structural_section_key({"section_title": "Ablation Study", "section_type": "results"}), "ablation")
        self.assertEqual(structural_section_key({"section_title": "Implementation Details", "section_type": "experiments"}), "implementation_details")

    def test_context_records_follow_selected_chunks_in_document_order(self):
        section = {
            "records": [
                {"global_record_id": "p::r3", "page": 3, "document_order": 3},
                {"global_record_id": "p::r1", "page": 1, "document_order": 1},
                {"global_record_id": "p::r2", "page": 2, "document_order": 2},
            ],
            "_context_record_ids": ["p::r3", "p::r1"],
        }
        config = SectionRelevanceConfig(backend="llmrerank", llmrerank_context_top_k_chunks=3)
        self.assertEqual(
            [record["global_record_id"] for record in context_records_for_section(section, config)],
            ["p::r1", "p::r3"],
        )

    def test_record_aware_units_keep_objects_atomic_with_text_neighbors(self):
        records = [
            {"paper_id": "p", "global_record_id": "p::t1", "source_type": "text_span", "record_type": "text", "text": "before context words", "page": 1, "document_order": 1},
            {"paper_id": "p", "global_record_id": "p::o1", "source_type": "table", "record_type": "table", "text": "table value " + "x " * 40, "page": 1, "document_order": 2},
            {"paper_id": "p", "global_record_id": "p::t2", "source_type": "text_span", "record_type": "text", "text": "after context words", "page": 1, "document_order": 3},
        ]
        section = {
            "paper_id": "p", "section_title": "Results", "section_type": "results", "section_path": ["Results"],
            "page_start": 1, "page_end": 1, "object_labels": ["Table 1"], "content": "content", "records": records,
        }
        config = SectionRelevanceConfig(
            backend="llmrerank", unit_mode="record_aware", unit_target_tokens=24, unit_max_tokens=32,
            unit_overlap_records=1, object_units_enabled=True, object_neighbor_records=1,
        )
        units = _chunk_section(section, _RoundTripTokenizer(), config)
        objects = [unit for unit in units if unit["unit_type"] == "object_table"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["anchor_record_ids"], ["p::o1"])
        self.assertEqual(objects[0]["record_ids"], ["p::t1", "p::o1", "p::t2"])
        self.assertTrue(objects[0]["oversized_atomic_object"])
        text_units = [unit for unit in units if unit["unit_type"].startswith("text_")]
        self.assertTrue(all("p::o1" not in unit["record_ids"] for unit in text_units))
        self.assertFalse(any(unit.get("oversized_record_split") for unit in text_units))

    def test_record_aware_retrieval_emits_auditable_unit_scores(self):
        records = [
            {"paper_id": "p", "global_record_id": "p::r1", "section_id": "s1", "section_title": "Results", "section_type": "results", "section_path": ["Results"], "source_type": "text_span", "record_type": "text", "text": "unrelated setup", "page": 1, "document_order": 1},
            {"paper_id": "p", "global_record_id": "p::r2", "section_id": "s1", "section_title": "Results", "section_type": "results", "section_path": ["Results"], "source_type": "table", "record_type": "table", "label": "Table 1", "locator": {"page": 1, "table_id": "Table 1"}, "text": "target metric is 42", "page": 1, "document_order": 2},
        ]
        config = SectionRelevanceConfig(
            backend="bm25", unit_mode="record_aware", unit_target_tokens=32, unit_max_tokens=48,
            aggregation_top_k=3, section_bonus_weight=0.1, object_section_bonus_weight=0.15,
            section_bonus_max=0.1, retrieval_unit_top_k=1, object_neighbor_records=0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_extraction_symbolic_vlm_baseline.section_relevance._load_tokenizer",
            return_value=_RoundTripTokenizer(),
        ):
            result = retrieve_section_relevance("target metric", records, Path(directory), config, top_k_sections=1)
        ranked = result["trace"]["ranked_units"]
        self.assertTrue(ranked)
        self.assertTrue(all(set(item["score_contract"]) >= {"local_relevance", "section_relevance", "section_bonus", "object_bonus", "final_relevance", "local_rank", "final_rank"} for item in ranked))
        self.assertEqual(ranked[0]["unit_type"], "object_table")
        self.assertGreaterEqual(ranked[0]["score_contract"]["final_relevance"], ranked[0]["score_contract"]["local_relevance"])
        self.assertEqual([record["global_record_id"] for record in result["expanded_records"]], ["p::r2"])

    def test_exact_duplicate_candidate_records_are_collapsed_without_conflicts(self):
        record = {
            "paper_id": "p", "global_record_id": "p::r1", "page": 1, "section_id": "s",
            "record_type": "text", "source_type": "text_span", "locator": {"page": 1}, "text": "same",
        }
        canonical, stats = _deduplicate_candidate_records([record, dict(record)])
        self.assertEqual(len(canonical), 1)
        self.assertEqual(stats["exact_duplicate_record_count"], 1)
        self.assertEqual(stats["conflicting_global_record_id_count"], 0)

    def test_canonical_text_and_image_projections_share_one_qwen_score(self):
        chunks = [
            {
                "text": "same projection", "image_paths": ["/tmp/same-figure.png"], "token_count": 5,
                "artifact_fingerprint": {"artifact_version": "v1"}, "record_ids": ["p::r1"],
            },
            {
                "text": "same projection", "image_paths": ["/tmp/same-figure.png"], "token_count": 5,
                "artifact_fingerprint": {"artifact_version": "v1"}, "record_ids": ["p::r2"],
            },
        ]
        config = SectionRelevanceConfig(backend="llmrerank", llmrerank_api_key="configured")
        with tempfile.TemporaryDirectory() as directory, patch(
            "pdf_extraction_symbolic_vlm_baseline.section_relevance.LLMRerankClient.score_documents",
            return_value={"scores": [0.4, 0.8], "raw_response": {}, "attempts": 1},
        ) as score_documents:
            scores, details, stats = _llmrerank_scores("query", chunks, config, Path(directory))
        self.assertEqual(score_documents.call_count, 1)
        self.assertEqual(len(score_documents.call_args.kwargs["documents"]), 2)
        self.assertEqual(scores, [0.8, 0.8])
        self.assertEqual(stats["canonical_text_projection_count"], 1)
        self.assertEqual(stats["canonical_image_projection_count"], 1)
        self.assertEqual(stats["deduplicated_text_projection_count"], 1)
        self.assertEqual(stats["deduplicated_image_projection_count"], 1)
        self.assertEqual(details[0]["canonical_projection_ids"], details[1]["canonical_projection_ids"])


if __name__ == "__main__":
    unittest.main()
