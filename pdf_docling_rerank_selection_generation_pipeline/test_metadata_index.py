from __future__ import annotations

import unittest

from .metadata_index import build_metadata_records, retrieve_candidates


class MetadataIndexTopicRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = build_metadata_records([
            {
                "paper_id": "p_consistency",
                "title": "Consistency Models Made Easy",
                "abstract": "Consistency models offer faster sampling for diffusion models; few-step distillation on CIFAR-10 reduces FID.",
                "authors": [], "venue": "ICLR", "year": 2025,
            },
            {
                "paper_id": "p_unrelated",
                "title": "A Study of Database Query Optimizers",
                "abstract": "We benchmark join order and cost models for analytical SQL workloads.",
                "authors": [], "venue": "SIGMOD", "year": 2024,
            },
        ])

    def test_topic_method_recovers_domain_paper(self) -> None:
        hits = retrieve_candidates(
            "What 1-step FID do consistency-distilled models achieve on CIFAR-10?",
            self.records,
            top_k=2,
            method="hybrid_alias_topic_optin",
        )
        self.assertEqual(hits[0]["paper_id"], "p_consistency")
        self.assertEqual(hits[0]["retrieval_method"], "hybrid_alias_topic_optin")
        self.assertGreater(hits[0]["retrieval_score_components"].get("topic_overlap", 0.0), 0.0)

    def test_plain_hybrid_still_works(self) -> None:
        hits = retrieve_candidates(
            "What 1-step FID do consistency-distilled models achieve on CIFAR-10?",
            self.records,
            top_k=2,
            method="hybrid_alias",
        )
        self.assertEqual(hits[0]["retrieval_method"], "hybrid_alias")
        self.assertNotIn("topic_overlap", hits[0]["retrieval_score_components"])


if __name__ == "__main__":
    unittest.main()
