import unittest

from pdf_docling_rerank_selection_generation_pipeline.evidence_packages import EvidencePackageConfig, build_packages, select_packages


def _record(identifier, paper, source_type, text, page=1, label="", order=1):
    return {
        "global_record_id": identifier,
        "paper_id": paper,
        "source_type": source_type,
        "text": text,
        "page": page,
        "label": label,
        "section_id": "results",
        "section_title": "Results",
        "document_order": order,
    }


class EvidencePackageTests(unittest.TestCase):
    def test_object_package_keeps_neighboring_text_and_canonical_record(self):
        records = [
            _record("p::t1", "p", "text_span", "before table", order=1),
            _record("p::tab", "p", "table", "table rows", label="Table 3", order=2),
            _record("p::t2", "p", "text_span", "after table", order=3),
            _record("p::tab", "p", "table", "short duplicate", label="Table 3", order=2),
        ]
        packages = build_packages(records, {"ranked_units": [{"record_ids": ["p::tab"], "score_contract": {"local_relevance": 0.9}}]}, EvidencePackageConfig())
        table = next(package for package in packages if package["anchor_record_id"] == "p::tab")
        self.assertEqual([record["global_record_id"] for record in table["records"]], ["p::t1", "p::tab", "p::t2"])

    def test_object_package_prefers_explicit_label_references_over_adjacent_layout_text(self):
        records = [
            _record("p::near-before", "p", "text_span", "generic preceding paragraph", page=3, order=10),
            _record("p::table", "p", "table", "table rows", page=3, label="Table 3", order=11),
            _record("p::near-after", "p", "text_span", "generic following paragraph", page=3, order=12),
            _record("p::narration", "p", "text_span", "As shown in Table 3, the proposed method improves F1.", page=4, order=2),
            _record("p::other", "p", "text_span", "Table 4 contains ablations.", page=3, order=13),
        ]
        package = next(item for item in build_packages(records, {}, EvidencePackageConfig()) if item["anchor_record_id"] == "p::table")
        identifiers = {record["global_record_id"] for record in package["records"]}
        self.assertIn("p::narration", identifiers)
        self.assertNotIn("p::near-before", identifiers)
        self.assertNotIn("p::near-after", identifiers)
        self.assertNotIn("p::other", identifiers)

    def test_figure_and_equation_reference_patterns_accept_common_abbreviations(self):
        records = [
            _record("p::fig", "p", "figure", "caption", label="Figure 2", order=2),
            _record("p::fig-text", "p", "text_span", "Fig. 2 illustrates the architecture.", order=9),
            _record("p::eq", "p", "equation_algorithm", "formula", label="Equation 11", order=12),
            _record("p::eq-text", "p", "text_span", "We optimize the objective in Eq. (11).", order=20),
        ]
        packages = {item["anchor_record_id"]: item for item in build_packages(records, {}, EvidencePackageConfig())}
        self.assertIn("p::fig-text", {record["global_record_id"] for record in packages["p::fig"]["records"]})
        self.assertIn("p::eq-text", {record["global_record_id"] for record in packages["p::eq"]["records"]})

    def test_object_reference_patterns_accept_appendix_and_suffix_labels(self):
        records = [
            _record("p::table", "p", "table", "rows", label="Table A.1", order=2),
            _record("p::table-text", "p", "text_span", "Tab. A.1 lists the appendix settings.", order=7),
            _record("p::eq", "p", "equation_algorithm", "formula", label="Equation 3a", order=12),
            _record("p::eq-text", "p", "text_span", "The bound follows directly from Eq. 3a.", order=15),
        ]
        packages = {item["anchor_record_id"]: item for item in build_packages(records, {}, EvidencePackageConfig())}
        self.assertIn("p::table-text", {record["global_record_id"] for record in packages["p::table"]["records"]})
        self.assertIn("p::eq-text", {record["global_record_id"] for record in packages["p::eq"]["records"]})

    def test_unscored_narration_is_context_only_not_a_new_package(self):
        anchor = _record("p::table", "p", "table", "rows", label="Table 5", order=2)
        narration = _record("p::text", "p", "text_span", "Table 5 reports the final accuracy.", page=3, order=1)
        packages = build_packages([anchor], {}, EvidencePackageConfig(), context_records=[anchor, narration])
        self.assertEqual([package["anchor_record_id"] for package in packages], ["p::table"])
        self.assertIn("p::text", {record["global_record_id"] for record in packages[0]["records"]})

    def test_multi_paper_selection_reserves_one_package_per_paper(self):
        records = [
            _record("a::one", "a", "text_span", "matching evidence"),
            _record("b::one", "b", "text_span", "matching evidence"),
            _record("b::two", "b", "text_span", "other evidence", order=2),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["a::one"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["b::one"], "score_contract": {"local_relevance": 0.8}},
            {"record_ids": ["b::two"], "score_contract": {"local_relevance": 0.7}},
        ]}
        result = select_packages(
            query="matching evidence",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2)),
            primary_evidence_type="text_span",
            is_multi_paper_task=True,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=2),
        )
        self.assertEqual({package["paper_id"] for package in result["packages"]}, {"a", "b"})

    def test_explicit_table_modality_is_selected_before_higher_text_score(self):
        records = [
            _record("p::text", "p", "text_span", "unrelated but high score", order=1),
            _record("p::table", "p", "table", "metric rows", label="Table 2", order=2),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::text"], "score_contract": {"local_relevance": 0.99}},
            {"record_ids": ["p::table"], "score_contract": {"local_relevance": 0.10}},
        ]}
        result = select_packages(
            query="What does Table 2 report?",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=1)),
            primary_evidence_type="table",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=1),
        )
        self.assertEqual(result["packages"][0]["anchor_record_id"], "p::table")
        self.assertEqual(result["trace"]["requested_modalities"], ["table"])

    def test_page_cap_prevents_near_duplicate_packages_from_filling_budget(self):
        records = [
            _record("p::one", "p", "text_span", "matching evidence", page=1, order=1),
            _record("p::two", "p", "text_span", "matching evidence", page=1, order=2),
            _record("p::three", "p", "text_span", "matching evidence", page=2, order=1),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::one"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["p::two"], "score_contract": {"local_relevance": 0.8}},
            {"record_ids": ["p::three"], "score_contract": {"local_relevance": 0.7}},
        ]}
        result = select_packages(
            query="matching evidence",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=3, max_packages_per_page=1)),
            primary_evidence_type="text_span",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=3, max_packages_per_page=1),
        )
        self.assertEqual([package["page"] for package in result["packages"]], [1, 2])

    def test_page_text_audit_route_preserves_a_text_anchor_per_page(self):
        records = [
            _record("p::one", "p", "text_span", "first page", page=1, order=1),
            _record("p::two", "p", "text_span", "second page", page=2, order=1),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::one"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["p::two"], "score_contract": {"local_relevance": 0.1}},
        ]}
        result = select_packages(
            query="first page",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2, page_text_anchors_per_page=1)),
            primary_evidence_type="figure",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=2, page_text_anchors_per_page=1),
        )
        self.assertEqual({package["anchor_record_id"] for package in result["packages"]}, {"p::one", "p::two"})

    def test_qwen_text_and_visual_tracks_are_retained_separately(self):
        records = [_record("p::fig", "p", "figure", "caption", label="Figure 1")]
        trace = {
            "ranked_units": [{"record_ids": ["p::fig"], "score_contract": {"local_relevance": 0.8}}],
            "sections": [{"chunks": [{
                "record_ids": ["p::fig"],
                "llmrerank_call": {"modality_scores": [
                    {"modality": "text", "score": 0.2},
                    {"modality": "image", "score": 0.9},
                ]},
            }]}],
        }
        package = build_packages(records, trace, EvidencePackageConfig())[0]
        self.assertEqual(package["qwen"], 0.8)
        self.assertEqual(package["qwen_text"], 0.2)
        self.assertEqual(package["qwen_visual"], 0.9)

    def test_primary_order_does_not_compare_visual_track_to_total_qwen_score(self):
        records = [
            _record("p::visual", "p", "figure", "gold method figure", label="Figure 2"),
            _record("p::text_only", "p", "figure", "unrelated reference figure", page=2, label="Figure 9"),
        ]
        trace = {
            "ranked_units": [
                {"record_ids": ["p::visual"], "score_contract": {"local_relevance": 0.8}},
                {"record_ids": ["p::text_only"], "score_contract": {"local_relevance": 0.2}},
            ],
            "sections": [{"chunks": [{
                "record_ids": ["p::visual"],
                "llmrerank_call": {"modality_scores": [{"modality": "image", "score": 0.01}]},
            }]}],
        }
        result = select_packages(
            query="Which Figure 2 shows the method?",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=1)),
            primary_evidence_type="figure",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=1),
        )
        self.assertEqual(result["packages"][0]["anchor_record_id"], "p::visual")

    def test_context_budget_never_splits_a_package(self):
        records = [
            _record("p::t1", "p", "text_span", "before", order=1),
            _record("p::table", "p", "table", "table-content", label="Table 1", order=2),
            _record("p::t2", "p", "text_span", "after", order=3),
            _record("p::other", "p", "text_span", "other-support", page=2, order=1),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::table"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["p::other"], "score_contract": {"local_relevance": 0.8}},
        ]}
        result = select_packages(
            query="Table 1",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2, max_context_chars=20)),
            primary_evidence_type="table",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=2, max_context_chars=20),
        )
        self.assertIn("p::table", [record["global_record_id"] for record in result["records"]])
        self.assertNotIn("p::other", [record["global_record_id"] for record in result["records"]])

    def test_hyde_route_claim_reserves_its_best_package(self):
        records = [
            _record("p::direct", "p", "text_span", "direct question answer"),
            _record("p::aux", "p", "text_span", "baseline training schedule"),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::direct"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["p::aux"], "score_contract": {"local_relevance": 0.1}},
        ]}
        result = select_packages(
            query="direct question",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2)),
            primary_evidence_type="text_span",
            is_multi_paper_task=True,
            route_queries=["baseline training schedule [VALUE]"],
            config=EvidencePackageConfig(package_budget=2),
        )
        self.assertEqual({package["anchor_record_id"] for package in result["packages"]}, {"p::direct", "p::aux"})
        self.assertEqual(result["trace"]["hyde_claim_route_count"], 1)

    def test_paper_local_route_reserves_local_bm25_anchor(self):
        records = [
            _record("a::direct", "a", "text_span", "named method answer"),
            _record("b::baseline", "b", "text_span", "optimizer schedule baseline", order=2),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["a::direct"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["b::baseline"], "score_contract": {"local_relevance": 0.1}},
        ]}
        result = select_packages(
            query="named method answer",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2)),
            primary_evidence_type="text_span",
            is_multi_paper_task=True,
            route_queries=[],
            paper_local_route_queries=[("b", "optimizer schedule baseline")],
            config=EvidencePackageConfig(package_budget=2),
        )
        self.assertIn("b::baseline", {package["anchor_record_id"] for package in result["packages"]})
        self.assertEqual(result["trace"]["paper_local_bm25_route_count"], 1)

    def test_each_present_source_type_has_an_independent_candidate_route(self):
        records = [
            _record("p::text", "p", "text_span", "ordinary evidence"),
            _record("p::figure", "p", "figure", "diagram caption", label="Figure 1"),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["p::text"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["p::figure"], "score_contract": {"local_relevance": 0.1}},
        ]}
        result = select_packages(
            query="ordinary evidence",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=2, candidate_pool_per_route=1)),
            primary_evidence_type="text_span",
            is_multi_paper_task=False,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=2, candidate_pool_per_route=1),
        )
        self.assertEqual(result["trace"]["source_route_count"], 2)
        self.assertEqual(result["trace"]["candidate_package_count"], 2)

    def test_adaptive_stop_uses_minimum_cross_paper_coverage_not_max_budget(self):
        records = [
            _record(f"p{index}::r", f"p{index}", "text_span", "shared evidence")
            for index in range(5)
        ]
        trace = {"ranked_units": [
            {"record_ids": [f"p{index}::r"], "score_contract": {"local_relevance": 1.0 - index * 0.1}}
            for index in range(5)
        ]}
        result = select_packages(
            query="shared evidence",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=8, min_package_budget=4, min_distinct_papers=4)),
            primary_evidence_type="text_span",
            is_multi_paper_task=True,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=8, min_package_budget=4, min_distinct_papers=4),
        )
        self.assertEqual(len(result["packages"]), 4)
        self.assertEqual(result["trace"]["adaptive_stop_reason"], "coverage_complete_no_marginal_gain")

    def test_explicit_method_figure_keeps_method_section_package_per_paper(self):
        records = [
            {**_record("a::intro", "a", "figure", "overview", page=1, label="Figure 1"), "section_type": "introduction"},
            {**_record("a::method", "a", "figure", "method diagram", page=3, label="Figure 2"), "section_type": "method"},
            {**_record("b::intro", "b", "figure", "overview", page=1, label="Figure 1"), "section_type": "introduction"},
            {**_record("b::method", "b", "figure", "method diagram", page=3, label="Figure 2"), "section_type": "method"},
        ]
        trace = {"ranked_units": [
            {"record_ids": ["a::intro"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["a::method"], "score_contract": {"local_relevance": 0.2}},
            {"record_ids": ["b::intro"], "score_contract": {"local_relevance": 0.8}},
            {"record_ids": ["b::method"], "score_contract": {"local_relevance": 0.1}},
        ]}
        result = select_packages(
            query="Which papers show MCTS in their primary method framework figure?",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=6, min_distinct_papers=2, modality_packages_per_paper=1)),
            primary_evidence_type="figure",
            is_multi_paper_task=True,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=6, min_distinct_papers=2, modality_packages_per_paper=1),
        )
        self.assertTrue({"a::method", "b::method"} <= {package["anchor_record_id"] for package in result["packages"]})
        self.assertEqual(result["trace"]["requested_layout_section_types"], ["method"])

    def test_multi_paper_supporting_text_route_keeps_text_anchor_for_object_question(self):
        records = [
            _record("a::fig", "a", "figure", "method figure", label="Figure 1", order=1),
            _record("a::text", "a", "text_span", "the required training condition", order=2),
            _record("b::fig", "b", "figure", "method figure", label="Figure 1", order=1),
            _record("b::text", "b", "text_span", "the required training condition", order=2),
        ]
        trace = {"ranked_units": [
            {"record_ids": ["a::fig"], "score_contract": {"local_relevance": 0.9}},
            {"record_ids": ["a::text"], "score_contract": {"local_relevance": 0.6}},
            {"record_ids": ["b::fig"], "score_contract": {"local_relevance": 0.8}},
            {"record_ids": ["b::text"], "score_contract": {"local_relevance": 0.5}},
        ]}
        result = select_packages(
            query="What does Figure 1 show?",
            packages=build_packages(records, trace, EvidencePackageConfig(package_budget=4, min_distinct_papers=2, modality_packages_per_paper=1, supporting_text_packages_per_paper=1)),
            primary_evidence_type="figure",
            is_multi_paper_task=True,
            route_queries=[],
            config=EvidencePackageConfig(package_budget=4, min_distinct_papers=2, modality_packages_per_paper=1, supporting_text_packages_per_paper=1),
        )
        self.assertTrue({"a::text", "b::text"} <= {package["anchor_record_id"] for package in result["packages"]})


if __name__ == "__main__":
    unittest.main()
