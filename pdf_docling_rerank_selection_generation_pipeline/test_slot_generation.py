from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .slot_generation import (
    align_composed_values,
    bind_composition_support,
    deterministic_count_extraction,
    ensure_slot_cards,
    plan_augmented_rerank_query,
    plan_package_routes,
    plan_paper_package_routes,
    slot_cards,
    slot_composition_messages,
    slot_extraction_messages,
    slot_image_paths,
    slot_plan_messages,
    validate_slot_extraction,
    validate_slot_plan,
)
from . import generate_from_cached_selection


class SlotGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "query_id": "q_001", "task_family": "hidden_source_single_paper",
            "primary_evidence_type": "table", "question": "What F1 does Method A achieve on Set X?",
        }
        self.candidates = [{"paper_id": "paper", "title": "Paper"}]
        self.hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                "section_id": "results", "page": 3, "source_type": "table", "label": "Table 1",
                "text": "Method A on Set X has F1 71.16.",
            }],
            "l2_evidence_cards": [{
                "support_refs": ["E0001"], "proposition": "Method A on Set X has F1 71.16.",
                "verification": {"status": "extractive"},
            }],
        }

    def test_plan_ignores_legacy_paper_scope_and_keeps_typed_slot(self) -> None:
        plan, audit = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "direct", "paper_scope": ["paper", "unknown"],
            "required_source_types": ["table", "bad"], "entities": ["Method A"],
            "required_conditions": ["F1", "Set X"],
        }]}, self.sample)
        self.assertFalse(plan["fallback"])
        self.assertNotIn("paper_scope", plan["slots"][0])
        self.assertEqual(plan["slots"][0]["required_source_types"], ["table"])
        self.assertEqual(audit[0]["status"], "slot_accepted")
        self.assertTrue(audit[0]["legacy_paper_scope_ignored"])

    def test_plan_accepts_literal_as_direct_operation(self) -> None:
        plan, audit = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "literal", "required_source_types": ["equation_algorithm"],
            "entities": ["RomanTex"], "required_conditions": [],
        }]}, self.sample)
        self.assertEqual(plan["slots"][0]["operation"], "direct")
        self.assertEqual(audit[0]["status"], "slot_accepted")

    def test_package_routes_keep_each_slot_source_type_and_do_not_use_paper_scope(self) -> None:
        routes = plan_package_routes({"slots": [
            {"id": "S001", "paper_scope": ["leaky"], "required_source_types": ["table"], "entities": ["Method A"], "required_conditions": ["F1"]},
            {"id": "S002", "required_source_types": ["figure"], "entities": ["Method B"], "required_conditions": ["architecture"]},
        ]}, "Compare Method A and Method B")
        self.assertEqual(routes, [
            {"slot_id": "S001", "record_types": ["table"], "query": "table Method A F1"},
            {"slot_id": "S002", "record_types": ["figure"], "query": "figure Method B architecture"},
        ])

    def test_package_route_keeps_explicit_parenthetical_entity_condition(self) -> None:
        routes = plan_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"],
            "entities": ["ECM-XL (100k iterations)"], "required_conditions": ["CIFAR-10"],
        }]}, "What is ECM-XL (100k iterations) on CIFAR-10?")
        self.assertEqual(routes, [
            {"slot_id": "S001", "record_types": ["table"], "query": "table ECM-XL (100k iterations) CIFAR-10"},
            {"slot_id": "S001", "record_types": ["table"], "query": "table ECM-XL 100k iterations", "catalog_fallback": True},
        ])

    def test_package_route_uses_catalog_fallback_for_long_literal_condition_list(self) -> None:
        routes = plan_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"], "entities": ["Kitchen"],
            "required_conditions": ["SUN RGB-D", "ARKitScenes", "Hypersim", "Objectron"],
        }]}, "What is Kitchen across these datasets?")
        self.assertEqual(routes, [{
            "slot_id": "S001", "record_types": ["table"],
            "query": "table Kitchen SUN RGB-D ARKitScenes Hypersim Objectron", "catalog_fallback": True,
        }])

    def test_paper_package_route_requires_one_distinctive_literal_title_match(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["equation_algorithm"],
            "entities": ["RomanTex", "NeRF"], "required_conditions": ["rotation matrix"],
        }]}, [
            {"paper_id": "roman", "title": "RomanTex: Rotary Positional Encoding for Textures"},
            {"paper_id": "nerf", "title": "NeRF Editing"},
        ])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "roman", "record_types": ["equation_algorithm"],
            "query": "equation_algorithm RomanTex rotation matrix",
        }])

    def test_paper_package_route_binds_unique_short_title_initialism(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"],
            "entities": ["TCM"], "required_conditions": ["CIFAR-10"],
        }]}, [
            {"paper_id": "tcm", "title": "Truncated Consistency Models"},
            {"paper_id": "other", "title": "Consistency Models Made Easy"},
        ])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "tcm", "record_types": ["table"],
            "query": "table TCM CIFAR-10",
        }])

    def test_paper_package_route_binds_initialism_after_generic_title_prefix(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"], "entities": ["NCFM", "ATT", "DEDA"],
            "required_conditions": ["Tiny ImageNet"],
        }]}, [
            {"paper_id": "ncfm", "title": "Dataset Distillation with Neural Characteristic Function: A Minmax Perspective"},
            {"paper_id": "att", "title": "Dataset Distillation by Automatic Training Trajectories"},
            {"paper_id": "deda", "title": "Diversity-Enhanced Distribution Alignment for Dataset Distillation"},
            {"paper_id": "distractor", "title": "AegisGuard: RL-Guided Adapter Tuning"},
        ])
        self.assertEqual([route["paper_id"] for route in routes], ["ncfm", "att", "deda"])

    def test_paper_package_route_binds_unique_one_letter_hyphenated_method_typo(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"], "entities": ["AP-BPTT"],
            "required_conditions": ["Tiny ImageNet"],
        }]}, [{
            "paper_id": "at_bptt", "title": "Beyond Random", "abstract": "We propose AT-BPTT.",
        }])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "at_bptt", "record_types": ["table"],
            "query": "table AP-BPTT Tiny ImageNet",
        }])

    def test_paper_package_route_binds_unique_short_colon_title(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["table"],
            "entities": ["MoST"], "required_conditions": ["ModelNet40"],
        }]}, [
            {"paper_id": "most", "title": "MoST: Efficient Sparse Tuning"},
            {"paper_id": "other", "title": "Most Efficient Tuning"},
        ])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "most", "record_types": ["table"],
            "query": "table MoST ModelNet40",
        }])

    def test_paper_package_route_uses_unique_title_overlap_and_citation_text(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["citation_context"],
            "entities": ["BRIDGE multi-view clustering paper"], "required_conditions": ["ICCV 2025", "APADC"],
        }]}, [
            {"paper_id": "iccv2025_00038", "title": "A Unified Framework to BRIDGE Complete and Incomplete Deep Multi-View Clustering"},
            {"paper_id": "iccv2025_00519", "title": "Deep Incomplete Multi-view Clustering with Distribution Recovery"},
        ])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "iccv2025_00038", "record_types": ["citation_context", "text_span"],
            "query": "citation_context BRIDGE multi-view clustering paper ICCV 2025 APADC",
        }])

    def test_paper_package_route_does_not_bind_generic_description_by_title_overlap(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["text_span"],
            "entities": ["Gaussian splatting editing"], "required_conditions": [],
        }]}, [{
            "paper_id": "gaussian", "title": "NeRF Editing with Gaussian Splatting",
        }])
        self.assertEqual(routes, [])

    def test_paper_package_route_uses_single_question_venue_to_disambiguate_title(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["text_span"],
            "entities": ["DisCo"], "required_conditions": ["token reduction"],
        }]}, [
            {"paper_id": "iccv2025_00591", "title": "DisCo: Visual Encapsulation in Video MLLMs"},
            {"paper_id": "neurips2025_01146", "title": "DISCO: Discrete Noise for Conditional Control"},
        ], "Across these ICCV 2025 papers, report the DisCo result.")
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "iccv2025_00591", "record_types": ["text_span"],
            "query": "text_span DisCo token reduction",
        }])

    def test_paper_package_route_uses_unique_exact_method_name_in_abstract(self) -> None:
        routes = plan_paper_package_routes({"slots": [{
            "id": "S001", "required_source_types": ["equation_algorithm"],
            "entities": ["ERASE"], "required_conditions": ["update expression"],
        }]}, [
            {"paper_id": "erase", "title": "Language Modeling with Editable External Knowledge", "abstract": "We introduce ERASE for editable knowledge."},
            {"paper_id": "other", "title": "Other paper", "abstract": "No named method here."},
        ])
        self.assertEqual(routes, [{
            "slot_id": "S001", "paper_id": "erase", "record_types": ["equation_algorithm"],
            "query": "equation_algorithm ERASE update expression",
        }])

    def test_plan_drops_paper_identity_conditions(self) -> None:
        plan, audit = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "count", "required_source_types": ["figure"],
            "entities": ["Figure 4"], "required_conditions": ["DynaPipe paper"],
        }]}, self.sample, candidates=[{
            "paper_id": "neurips2025_01307",
            "title": "DynaPipe: Dynamic Layer Redistribution for Efficient Serving of LLMs with Pipeline Parallelism",
        }])
        self.assertEqual(plan["slots"][0]["required_conditions"], [])
        self.assertEqual(audit[0]["paper_identity_conditions_dropped"], ["DynaPipe paper"])

    def test_visual_crop_value_does_not_require_text_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "figure_0004.png"
            crop.write_bytes(b"image")
            hierarchy = {
                "l0_catalog": [{
                    "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                    "section_id": "results", "page": 7, "source_type": "figure", "label": "Figure 4",
                    "text": "Figure 4: The average latency and SLO attainment rate.", "crop_path": str(crop),
                }],
                "l2_evidence_cards": [{
                    "support_refs": ["E0001"],
                    "proposition": "Figure 4: The average latency and SLO attainment rate.",
                    "verification": {"status": "extractive"},
                }],
            }
            accepted, _ = validate_slot_extraction({
                "slot_id": "S001", "status": "supported", "value": "8", "support_keys": ["C001"],
            }, {"id": "S001", "operation": "count", "required_source_types": ["figure"]}, hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertTrue(accepted["validation"]["numeric_ok"])
        self.assertTrue(accepted["validation"]["visual_supported"])
        self.assertEqual(accepted["value"], "8")

    def test_count_operation_rejects_non_integer_value(self) -> None:
        slot = {"id": "S001", "operation": "count", "required_source_types": ["table"]}
        rejected, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "8.0", "support_keys": ["C001"],
        }, slot, self.hierarchy)
        self.assertEqual(rejected["status"], "partial")
        self.assertFalse(rejected["validation"]["numeric_ok"])

    def test_list_value_with_digits_is_not_rejected_by_numeric_containment(self) -> None:
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "P05, P10", "support_keys": ["C001"],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, self.hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertTrue(accepted["validation"]["numeric_ok"])

    def test_table_rows_extraction_accepts_row_list(self) -> None:
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"],
            "table_rows": [
                {"values": {"Method Name": "LOGO", "Training Objective Equation ID": "Equation 3"}, "support_keys": ["C001"]},
            ],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, self.hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertEqual(len(accepted["table_rows"]), 1)
        self.assertEqual(accepted["table_rows"][0]["values"]["Method Name"], "LOGO")

    def test_table_rows_reject_unbound_support_keys(self) -> None:
        rejected, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None,
            "table_rows": [{"values": {"Method Name": "LOGO"}, "support_keys": ["C999"]}],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, self.hierarchy)
        self.assertEqual(rejected["status"], "partial")

    def test_table_rows_reconstructed_from_evidence_values_pairs(self) -> None:
        schema = ["Method Name", "Training Objective Equation ID"]
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"],
            "evidence_values": [
                {"name": "Method Name", "value": "LOGO", "support_keys": ["C001"]},
                {"name": "Training Objective Equation ID", "value": "Equation 3", "support_keys": ["C001"]},
                {"name": "Method Name", "value": "FPO", "support_keys": ["C001"]},
                {"name": "Training Objective Equation ID", "value": "Equation 8", "support_keys": ["C001"]},
            ],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, self.hierarchy, table_schema=schema)
        self.assertEqual(accepted["status"], "supported")
        self.assertEqual(len(accepted["table_rows"]), 2)
        self.assertEqual(accepted["table_rows"][0]["values"]["Method Name"], "LOGO")
        self.assertEqual(accepted["table_rows"][1]["values"]["Training Objective Equation ID"], "Equation 8")

    def test_table_rows_remap_variant_columns_to_schema(self) -> None:
        schema = ["dataset", "ndcg_at_10", "map_at_10"]
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"],
            "table_rows": [
                {"values": {"dataset": "SciFact", "NDCG@10": "62.5", "MAP@10": "31.0"}, "support_keys": ["C001"]},
            ],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, self.hierarchy, table_schema=schema)
        values = accepted["table_rows"][0]["values"]
        self.assertEqual(values.get("ndcg_at_10"), "62.5")
        self.assertEqual(values.get("map_at_10"), "31.0")
        self.assertNotIn("NDCG@10", values)

    def _citation_hierarchy(self, records: list[dict]) -> dict:
        return {
            "l0_catalog": [
                {
                    "evidence_ref": f"E{index:04d}", "global_record_id": f"paper::r{index}", "paper_id": "paper",
                    "section_id": "sec_refs", "section_title": "References", "page": 14,
                    "source_type": "citation_context", "label": f"Reference {record.get('citation_id')}",
                    "locator": {"citation_id": record.get("citation_id")}, "text": record.get("text"),
                }
                for index, record in enumerate(records, start=1)
            ],
            "l2_evidence_cards": [
                {"support_refs": [f"E{index:04d}"], "proposition": record.get("text"),
                 "verification": {"status": "extractive"}}
                for index, record in enumerate(records, start=1)
            ],
        }

    def test_deterministic_last_reference_index(self) -> None:
        polluted = self._citation_hierarchy([
            {"citation_id": 67, "text": "Annette J. Dobson and Adrian G. Barnett. An Introduction to Generalized Linear Models, 2008."},
            {"citation_id": 69, "text": "Question: Do the main claims made in the abstract and introduction accurately reflect the paper?"},
        ])
        result = deterministic_count_extraction(
            {"id": "S001", "operation": "count"}, polluted,
            "What is the index of the last reference in FedRACE: A Hierarchical Framework?",
            candidates=[{"paper_id": "paper", "title": "FedRACE: A Hierarchical and Statistical Framework"}],
        )
        self.assertIsNone(result)
        clean = self._citation_hierarchy([
            {"citation_id": 1, "text": "Author A. Title one, 2010."},
            {"citation_id": 2, "text": "Author B. Title two, 2011."},
        ])
        result = deterministic_count_extraction(
            {"id": "S001", "operation": "count"}, clean,
            "What is the index of the last reference in FedRACE: A Hierarchical Framework?",
            candidates=[{"paper_id": "paper", "title": "FedRACE: A Hierarchical and Statistical Framework"}],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], "2")
        self.assertEqual(result["status"], "supported")

    def test_deterministic_parentheses_count(self) -> None:
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                "section_id": "sec", "section_title": "Method", "page": 7,
                "source_type": "equation_algorithm", "label": "Equation 6",
                "locator": {"page": 7, "equation_id": "Equation 6"},
                "text": "J = ( ( W ( i 1 , i 2 ) ) - W ( j 1 , j 2 ) ) , (6)",
            }],
            "l2_evidence_cards": [{"support_refs": ["E0001"], "proposition": "J = ( ( W ( i 1 , i 2 ) ) - W ( j 1 , j 2 ) ) , (6)", "verification": {"status": "extractive"}}],
        }
        result = deterministic_count_extraction(
            {"id": "S001", "operation": "count"}, hierarchy,
            "How many Parentheses does it have in equation 6 of the Continuity-Preserving paper?",
            candidates=[{"paper_id": "paper", "title": "Continuity-Preserving Convolutional Autoencoders"}],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], "4")

    def test_deterministic_author_reference_count(self) -> None:
        hierarchy = self._citation_hierarchy([
            {"citation_id": 5, "text": "Bell, J. H., Bonawitz, K. A. Secure aggregation, 2017."},
            {"citation_id": 6, "text": "Bonawitz, K., Ivanov, V. Practical secure aggregation, 2017."},
            {"citation_id": 8, "text": "Bonawitz, K., Eichner, H. Practical secure aggregation for federated learning."},
            {"citation_id": 67, "text": "Table 6. Computation and communication complexity of existing SecAgg algorithms."},
        ])
        result = deterministic_count_extraction(
            {"id": "S001", "operation": "count"}, hierarchy,
            "How many references in the SecEmb paper include Bonawitz as an author?",
            candidates=[{"paper_id": "paper", "title": "SecEmb: Sparsity-Aware Secure Federated Learning"}],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], "3")

    def test_deterministic_section_citation_count(self) -> None:
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                "section_id": "sec_intro", "section_title": "Introduction", "page": 1,
                "source_type": "text_span", "label": None,
                "locator": {"page": 1}, "text": "Various methods exist (Koren et al., 2009; Xue et al., 2017; Rendle, 2010).",
            }],
            "l2_evidence_cards": [{"support_refs": ["E0001"], "proposition": "Various methods exist (Koren et al., 2009; Xue et al., 2017; Rendle, 2010).", "verification": {"status": "extractive"}}],
        }
        result = deterministic_count_extraction(
            {"id": "S001", "operation": "count"}, hierarchy,
            "How many papers were cited in Introduction Section of paper SecEmb?",
            candidates=[{"paper_id": "paper", "title": "SecEmb: Sparsity-Aware Secure Federated Learning"}],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], "3")

    def test_table_alignment_recovers_row_labels_from_bare_values(self) -> None:
        schema = ["Method", "FID"]
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                "section_id": "results", "page": 8, "source_type": "table", "label": "Table 1",
                "locator": {"page": 8, "table_id": "Table 1"},
                "text": "| Method | FID |\n| iCT-deep | 3.25 |\n| SiD | 1.92 |",
            }],
            "l2_evidence_cards": [{"support_refs": ["E0001"], "proposition": "Columns: Method | FID Row iCT-deep: Method=iCT-deep; FID=3.25 Row SiD: Method=SiD; FID=1.92", "verification": {"status": "extractive"}}],
        }
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"],
            "evidence_values": [
                {"name": "operand", "value": "3.25", "support_keys": ["C001"]},
                {"name": "operand", "value": "1.92", "support_keys": ["C001"]},
            ],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, hierarchy, table_schema=schema)
        self.assertEqual(accepted["status"], "supported")
        rows = {row["values"].get("Method"): row["values"].get("FID") for row in accepted["table_rows"]}
        self.assertEqual(rows.get("iCT-deep"), "3.25")
        self.assertEqual(rows.get("SiD"), "1.92")

    def test_table_alignment_picks_schema_column_over_model_value(self) -> None:
        schema = ["Method", "$AP^{nus}_{3D}$"]
        hierarchy = {
            "l0_catalog": [{
                "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                "section_id": "results", "page": 4, "source_type": "table", "label": "Table 2",
                "locator": {"page": 4, "table_id": "Table 2"},
                "text": "| Method | AP kit 3D | AP nus 3D |\n| DetAny3D (ours) w/ Ground Truth | 38.68 | 37.55 |",
            }],
            "l2_evidence_cards": [{"support_refs": ["E0001"], "proposition": "Columns: Method | AP kit 3D | AP nus 3D Row DetAny3D (ours) w/ Ground Truth: Method=DetAny3D (ours) w/ Ground Truth; AP kit 3D=38.68; AP nus 3D=37.55", "verification": {"status": "extractive"}}],
        }
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"],
            "evidence_values": [{"name": "operand", "value": "38.68", "support_keys": ["C001"]}],
        }, {"id": "S001", "operation": "list", "required_source_types": ["table"]}, hierarchy, table_schema=schema)
        self.assertEqual(accepted["status"], "supported")
        row = accepted["table_rows"][0]["values"]
        self.assertEqual(row.get("Method"), "DetAny3D (ours) w/ Ground Truth")
        self.assertEqual(row.get("$AP^{nus}_{3D}$"), "37.55")

    def test_table_rows_flow_into_prediction_table(self) -> None:
        class FakeClient:
            responses = iter([
                {"slots": [{"role": "direct_answer", "operation": "list", "required_source_types": ["table"], "entities": ["Method Name"]}]},
                {"slot_id": "S001", "status": "supported", "value": None, "support_keys": ["C001"], "table_rows": [
                    {"values": {"Method Name": "LOGO", "Training Objective Equation ID": "Equation 3"}, "support_keys": ["C001"]},
                ], "missing_conditions": []},
                {"gold_papers": [{"paper_id": "paper"}], "claim_to_slot_ids": {"Q01": ["S001"]}, "answer": {"table": {"rows": []}}},
            ])

            def __init__(self, *_args, **_kwargs):
                pass

            def generate_prediction(self, *_args, **_kwargs):
                return {"content": json.dumps(next(self.responses)), "raw_response": {}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official" / "data"
            official.mkdir(parents=True)
            (official / "validation_inputs.jsonl").write_text(json.dumps({
                "query_id": "q_table", "task_family": "multi_paper", "primary_evidence_type": "table",
                "question": "What method and equation do ICML papers propose?",
                "answer_types": ["table"],
                "table_schema": [
                    {"name": "Method Name", "type": "string", "is_row_key": True},
                    {"name": "Training Objective Equation ID", "type": "string", "is_row_key": False},
                ],
            }) + "\n", encoding="utf-8")
            selected = root / "selected.jsonl"
            selected.write_text(json.dumps({"query_id": "q_table", "selected_records": []}) + "\n", encoding="utf-8")
            candidates = root / "candidates.jsonl"
            candidates.write_text(json.dumps({"query_id": "q_table", "paper_id": "paper", "title": "Paper", "rank": 1}) + "\n", encoding="utf-8")
            hierarchy = root / "hierarchy.jsonl"
            hierarchy.write_text(json.dumps({"query_id": "q_table", "hierarchy": self.hierarchy}) + "\n", encoding="utf-8")
            output = root / "output"
            argv = [
                "generate_from_cached_selection", "--official-dir", str(official.parent),
                "--selected-contexts-input", str(selected), "--candidate-papers-input", str(candidates),
                "--hierarchy-input", str(hierarchy), "--hierarchy-prompt-mode", "keyed", "--generation-mode", "slots",
                "--output-dir", str(output), "--env-path", str(root / "missing.env"),
            ]
            with patch.object(generate_from_cached_selection, "VLMAnswerClient", FakeClient), patch.object(sys, "argv", argv):
                self.assertEqual(generate_from_cached_selection.main(), 0)
            prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            rows = (prediction.get("answer") or {}).get("table", {}).get("rows") or []
            self.assertEqual(rows, [{"Method Name": "LOGO", "Training Objective Equation ID": "Equation 3"}])

    def test_plan_prompt_excludes_candidate_metadata(self) -> None:
        text = slot_plan_messages(self.sample, {"answer_types": ["freeform"]})[-1]["content"]
        self.assertNotIn("candidate_papers", text)
        self.assertNotIn("Paper", text)
        self.assertNotIn("task_family", text)
        self.assertNotIn("primary_evidence_type", text)
        self.assertIn("query_analysis", text)
        self.assertIn("inferred_paper_count", text)

    def test_plan_prompt_excludes_gold_shaped_fields_for_new_format_sample(self) -> None:
        sample = {
            "query_id": "q_001",
            "question": "Compare the two methods.",
            "answer_types": ["freeform"],
        }
        text = slot_plan_messages(sample, {"answer_types": ["freeform"]})[-1]["content"]
        self.assertNotIn("task_family", text)
        self.assertNotIn("primary_evidence_type", text)

    def test_validate_slot_plan_keeps_structured_query_analysis(self) -> None:
        plan, audit = validate_slot_plan({
            "slots": [{
                "role": "direct_answer", "operation": "difference",
                "required_source_types": ["table"], "entities": ["IMM", "D-FINE"],
                "required_conditions": ["F1"],
            }],
            "query_analysis": {
                "entities": ["IMM", "D-FINE"],
                "comparison_targets": ["IMM", "D-FINE"],
                "inferred_paper_count": 2,
                "cross_paper_synthesis_required": True,
            },
        }, {"query_id": "q", "question": "Compare IMM and D-FINE.", "answer_types": ["freeform"]})
        self.assertFalse(plan["fallback"])
        self.assertTrue(plan["requires_cross_paper_synthesis"])
        self.assertEqual(plan["query_analysis"]["inferred_paper_count"], 2)
        self.assertEqual(plan["query_analysis"]["comparison_targets"], ["IMM", "D-FINE"])

    def test_cross_paper_list_plan_splits_entities_into_slots(self) -> None:
        plan, audit = validate_slot_plan({
            "slots": [{
                "role": "direct_answer", "operation": "list", "required_source_types": ["table"],
                "entities": ["Method A", "Method B"], "required_conditions": [],
            }],
            "query_analysis": {"inferred_paper_count": 2},
        }, {"query_id": "q", "question": "Compare Method A and Method B.", "answer_types": ["table"]})
        self.assertEqual([slot["entities"] for slot in plan["slots"]], [["Method A"], ["Method B"]])
        self.assertTrue(any(row.get("status") == "cross_paper_list_slot_split" for row in audit))

    def test_across_papers_list_is_split_even_when_llm_omits_cross_flag(self) -> None:
        plan, _ = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "list", "required_source_types": ["text_span"],
            "entities": ["DisCo", "DiTFastAttnV2", "DLFR-Gen"], "required_conditions": [],
        }], "requires_cross_paper_synthesis": False, "query_analysis": {
            "entities": ["DisCo", "DiTFastAttnV2", "DLFR-Gen"], "comparison_targets": [],
            "inferred_paper_count": None, "cross_paper_synthesis_required": False,
        }}, {"query_id": "q", "question": "Across these efficiency papers, report each headline claim.", "answer_types": ["table"]})
        self.assertTrue(plan["requires_cross_paper_synthesis"])
        self.assertEqual([slot["entities"] for slot in plan["slots"]], [["DisCo"], ["DiTFastAttnV2"], ["DLFR-Gen"]])

    def test_validate_slot_plan_deterministic_routing_overrides_llm_flag(self) -> None:
        plan, audit = validate_slot_plan({
            "slots": [{
                "role": "direct_answer", "operation": "difference",
                "required_source_types": ["table"], "entities": ["A", "B"],
                "required_conditions": [],
            }],
            "requires_cross_paper_synthesis": False,
            "query_analysis": {
                "entities": ["A", "B"],
                "comparison_targets": ["A", "B"],
                "inferred_paper_count": 1,
                "cross_paper_synthesis_required": False,
            },
        }, {"query_id": "q", "question": "Compare A and B.", "answer_types": ["freeform"]})
        self.assertTrue(plan["requires_cross_paper_synthesis"])
        self.assertTrue(any(row.get("status") == "cross_paper_routing_deterministic" for row in audit))

    def test_validate_slot_plan_fills_source_types_from_query_contract(self) -> None:
        plan, audit = validate_slot_plan({
            "slots": [{
                "role": "direct_answer", "operation": "direct",
                "required_source_types": [], "entities": ["value"],
                "required_conditions": [],
            }],
            "query_analysis": {},
        }, {"query_id": "q", "question": "Report the value in Table 2.", "answer_types": ["freeform"]})
        self.assertIn("table", plan["slots"][0]["required_source_types"])

    def test_table_output_contract_does_not_override_slot_source_type(self) -> None:
        plan, audit = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "direct",
            "required_source_types": ["equation"], "entities": ["loss"],
            "required_conditions": [],
        }], "query_analysis": {}}, {
            "query_id": "q", "question": "Report the loss.", "answer_types": ["table"],
        })
        self.assertEqual(plan["slots"][0]["required_source_types"], ["equation_algorithm"])
        self.assertFalse(any(row.get("status") == "slot_contract_aligned_table" for row in audit))

    def test_slot_explicit_object_corrects_generic_text_plan(self) -> None:
        plan, audit = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "direct",
            "required_source_types": ["text_span"], "entities": ["Training Objective Equation ID"],
            "required_conditions": [],
        }], "query_analysis": {}}, {
            "query_id": "q", "question": "Which method is used in Equation 4?", "answer_types": ["table"],
        })
        self.assertEqual(plan["slots"][0]["required_source_types"], ["equation_algorithm"])
        self.assertTrue(any(row.get("status") == "slot_source_type_explicitly_corrected" for row in audit))

    def test_fallback_plan_includes_query_analysis(self) -> None:
        plan, audit = validate_slot_plan({"slots": []}, {
            "query_id": "q", "question": "Compare A and B across papers.", "answer_types": ["freeform"],
        })
        self.assertTrue(plan["fallback"])
        self.assertTrue(plan["requires_cross_paper_synthesis"])
        self.assertEqual(plan["query_analysis"]["inferred_paper_count"], 2)
        self.assertEqual(audit[-1]["status"], "slot_plan_fallback")

    def test_plan_augmented_rerank_query_appends_slot_terms(self) -> None:
        plan = {"slots": [{"entities": ["MCTS", "ICAE"], "required_conditions": ["NAACL 2025", "F1"]}]}
        query = plan_augmented_rerank_query(plan, "Which papers mention MCTS?")
        self.assertTrue(query.startswith("Which papers mention MCTS?"))
        self.assertIn("NAACL 2025", query)
        self.assertIn("F1", query)
        # Deduplicated: repeated terms appear once.
        self.assertEqual(query.count("MCTS"), 2)  # question + entity

    def test_extraction_requires_visible_key_matching_type_and_number(self) -> None:
        slot = {"id": "S001", "required_source_types": ["table"]}
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "71.16", "support_keys": ["C001"],
        }, slot, self.hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertTrue(accepted["validation"]["numeric_ok"])
        rejected, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "99.99", "support_keys": ["C001"],
        }, slot, self.hierarchy)
        self.assertEqual(rejected["status"], "partial")
        self.assertFalse(rejected["validation"]["numeric_ok"])

    def test_difference_is_calculated_from_bound_operands(self) -> None:
        slot = {"id": "S001", "operation": "difference", "required_source_types": ["table"]}
        hierarchy = {**self.hierarchy, "l2_evidence_cards": [{
            **self.hierarchy["l2_evidence_cards"][0], "proposition": "Method A 71.16 and Method B 26.16.",
        }]}
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "999",
            "support_keys": ["C001"], "evidence_values": [
                {"name": "Method A", "value": "71.16", "support_keys": ["C001"]},
                {"name": "Method B", "value": "26.16", "support_keys": ["C001"]},
            ],
        }, slot, hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertEqual(accepted["value"], "45")
        self.assertTrue(accepted["validation"]["derived_value"])

    def test_reported_table_difference_beats_rounded_operand_subtraction(self) -> None:
        slot = {"id": "S001", "operation": "difference", "required_source_types": ["table"]}
        hierarchy = {**self.hierarchy, "l2_evidence_cards": [{
            **self.hierarchy["l2_evidence_cards"][0],
            "proposition": "Method A 41.36 // Method B 26.65 // Absolute Delta 14.70.",
        }]}
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "14.71", "reported_value": "14.70",
            "support_keys": ["C001"], "evidence_values": [
                {"name": "Method A", "value": "41.36", "support_keys": ["C001"]},
                {"name": "Method B", "value": "26.65", "support_keys": ["C001"]},
            ],
        }, slot, hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertEqual(accepted["value"], "14.70")
        self.assertTrue(accepted["validation"]["reported_difference"])
        self.assertFalse(accepted["validation"]["derived_value"])

    def test_difference_operand_conditions_must_cover_required_terms(self) -> None:
        slot = {
            "id": "S001", "operation": "difference", "required_source_types": ["table"],
            "required_conditions": ["F1", "NaturalQ", "500-to-1 compression setting"],
        }
        hierarchy = {**self.hierarchy, "l2_evidence_cards": [{
            **self.hierarchy["l2_evidence_cards"][0],
            "proposition": "Ours 500 to 1 NaturalQ F1 41.36 // ICAE 500 to 1 NaturalQ F1 26.65.",
        }]}
        rejected, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "21.93",
            "support_keys": ["C001"], "evidence_values": [
                {"name": "minuend", "value": "41.36", "support_keys": ["C001"], "conditions": {"dataset": "NaturalQ", "metric": "F1", "setting": "500 to 1"}},
                {"name": "subtrahend", "value": "19.43", "support_keys": ["C001"], "conditions": {"dataset": "Other", "metric": "F1", "setting": "500 to 1"}},
            ],
        }, slot, hierarchy)
        self.assertEqual(rejected["status"], "partial")
        self.assertFalse(rejected["validation"]["condition_ok"])

    def test_difference_operands_must_cooccur_in_same_table_record(self) -> None:
        slot = {
            "id": "S001", "operation": "difference", "required_source_types": ["table"],
            "required_conditions": ["F1", "NaturalQ"],
        }
        rejected, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "21.93",
            "support_keys": ["C001"], "evidence_values": [
                {"name": "minuend", "value": "41.36", "support_keys": ["C001"], "conditions": {"dataset": "NaturalQ", "metric": "F1"}},
                {"name": "subtrahend", "value": "19.43", "support_keys": ["C001"], "conditions": {"dataset": "NaturalQ", "metric": "F1"}},
            ],
        }, slot, self.hierarchy)
        self.assertEqual(rejected["status"], "partial")
        self.assertFalse(rejected["validation"]["same_table_operands"])

    def test_difference_correct_same_setting_operands_pass_and_use_reported_delta(self) -> None:
        slot = {
            "id": "S001", "operation": "difference", "required_source_types": ["table"],
            "required_conditions": ["F1", "NaturalQ", "500-to-1 compression setting"],
        }
        hierarchy = {**self.hierarchy, "l0_catalog": [{
            **self.hierarchy["l0_catalog"][0],
            "text": "Ours 500 to 1 | 41.36\nICAE 500 to 1 | 26.65\nAbsolute Delta | 14.70",
        }], "l2_evidence_cards": [{
            **self.hierarchy["l2_evidence_cards"][0],
            "proposition": "Ours 500 to 1 41.36 // ICAE 500 to 1 26.65 // Absolute Delta 14.70.",
        }]}
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "21.93",
            "support_keys": ["C001"], "evidence_values": [
                {"name": "minuend", "value": "41.36", "support_keys": ["C001"], "conditions": {"dataset": "NaturalQ", "metric": "F1", "setting": "500 to 1"}},
                {"name": "subtrahend", "value": "26.65", "support_keys": ["C001"], "conditions": {"dataset": "NaturalQ", "metric": "F1", "setting": "500 to 1"}},
            ],
        }, slot, hierarchy)
        self.assertEqual(accepted["status"], "supported")
        self.assertTrue(accepted["validation"]["condition_ok"])
        self.assertTrue(accepted["validation"]["same_table_operands"])
        self.assertEqual(accepted["value"], "14.70")

    def test_bound_table_delta_is_used_when_model_omits_it(self) -> None:
        slot = {"id": "S001", "operation": "difference", "required_source_types": ["table"]}
        hierarchy = {**self.hierarchy, "l0_catalog": [{
            **self.hierarchy["l0_catalog"][0],
            "text": "| Method A | 41.36 |\n| Method B | 26.65 |\n| Absolute Delta | 14.70 |",
        }]}
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "14.71", "support_keys": ["C001"],
            "evidence_values": [
                {"name": "Method A", "value": "41.36", "support_keys": ["C001"]},
                {"name": "Method B", "value": "26.65", "support_keys": ["C001"]},
            ],
        }, slot, hierarchy)
        self.assertEqual(accepted["value"], "14.70")
        self.assertEqual(accepted["validation"]["reported_difference_source"], "deterministic_table_delta")

    def test_slot_reservoir_adds_readable_card_even_for_l2_support_refs(self) -> None:
        hierarchy = {**self.hierarchy, "l0_catalog": [
            *self.hierarchy["l0_catalog"],
            {"evidence_ref": "E0002", "global_record_id": "paper::r2", "paper_id": "paper", "section_id": "method", "page": 2,
             "source_type": "equation_algorithm", "label": "Equation 6", "text": "f(x) = x + 1"},
        ]}
        updated = ensure_slot_cards(hierarchy)
        self.assertEqual(len(updated["l2_evidence_cards"]), 3)
        self.assertIn("EQUATION Equation 6 // f(x) = x + 1", updated["l2_evidence_cards"][-1]["proposition"])
        cards = slot_cards({"required_source_types": ["equation_algorithm"], "entities": ["Equation 6"]}, updated, "What is Equation 6?")
        self.assertTrue(any("EQUATION Equation 6" in str(card.get("proposition")) for card in cards))

    def test_plan_normalizes_common_source_and_operation_aliases(self) -> None:
        plan, _ = validate_slot_plan({"slots": [{
            "role": "direct_answer", "operation": "literal_extraction",
            "required_source_types": ["equation"], "entities": ["Equation 6"],
        }]}, self.sample)
        self.assertEqual(plan["slots"][0]["operation"], "direct")
        self.assertEqual(plan["slots"][0]["required_source_types"], ["equation_algorithm"])

    def test_empty_plan_fallback_retains_question_focus_for_card_routing(self) -> None:
        plan, _ = validate_slot_plan({"slots": []}, {
            **self.sample, "question": "What hardware configuration was used in the S-RAG framework?",
        })
        self.assertTrue(plan["fallback"])
        self.assertTrue(plan["slots"][0]["entities"][0].startswith("hardware configuration"))

    def test_empty_plan_infers_count_operation_from_question_form(self) -> None:
        plan, _ = validate_slot_plan({"slots": []}, {**self.sample, "question": "How many parentheses are in Equation 6?"})
        self.assertEqual(plan["slots"][0]["operation"], "count")

    def test_direct_slot_does_not_require_optional_operand_objects(self) -> None:
        accepted, _ = validate_slot_extraction({
            "slot_id": "S001", "status": "supported", "value": "Method A", "support_keys": ["C001"],
            "evidence_values": [{"name": "unbound", "value": "Method A"}],
        }, {"id": "S001", "operation": "direct", "required_source_types": ["table"]}, self.hierarchy)
        self.assertEqual(accepted["status"], "supported")

    def test_composition_binds_only_supported_slots(self) -> None:
        composed, audit = bind_composition_support({
            "claim_to_slot_ids": {"Q01": ["S001", "S999"]},
            "table_answer_plan": [{"row_slot_ids": ["S001"], "values": {"Score": 71.16}}],
            "answer": {"freeform": {"text": "71.16"}},
        }, [{"slot_id": "S001", "status": "supported", "support_keys": ["C001"]}])
        self.assertEqual(composed["claim_to_support_keys"], {"Q01": ["C001"]})
        self.assertEqual(composed["table_answer_plan"][0]["row_support_key"], "C001")
        self.assertTrue(any(row["status"] == "composition_claim_bound" for row in audit))

    def test_numeric_composition_is_reduced_to_the_verified_slot_value(self) -> None:
        composed, audit = align_composed_values({
            "answer": {"freeform": {"text": "Method A is higher by 45 F1 points."}},
        }, {**self.sample, "question": "By how much does Method A outperform Method B in F1?"}, [{
            "slot_id": "S001", "status": "supported", "value": "45", "support_keys": ["C001"],
            "validation": {"derived_value": True},
        }])
        self.assertEqual(composed["answer"]["freeform"]["text"], "45")
        self.assertEqual(audit[0]["status"], "composition_numeric_value_normalized")

    def test_derived_numeric_value_survives_keyed_grounding(self) -> None:
        internal, audit = generate_from_cached_selection._posthoc_ground_keyed_prediction({
            "gold_papers": [{"paper_id": "paper"}],
            "claim_to_support_keys": {"Q01": ["C001"]},
            "answer": {"freeform": {"text": "45"}},
            "_validated_slots": [{
                "slot_id": "S001", "status": "supported", "value": "45", "support_keys": ["C001"],
                "validation": {"derived_value": True},
            }],
        }, self.hierarchy)
        self.assertEqual(internal["answer"]["freeform"]["text"], "45")
        self.assertTrue(any(row.get("status") == "derived_value_grounded" for row in audit))

    def test_visual_slot_value_survives_keyed_grounding(self) -> None:
        internal, audit = generate_from_cached_selection._posthoc_ground_keyed_prediction({
            "gold_papers": [{"paper_id": "paper"}],
            "claim_to_support_keys": {"Q01": ["C001"]},
            "answer": {"freeform": {"text": "8"}},
            "_validated_slots": [{
                "slot_id": "S001", "status": "supported", "value": "8", "support_keys": ["C001"],
                "validation": {"visual_supported": True},
            }],
        }, self.hierarchy)
        self.assertEqual(internal["answer"]["freeform"]["text"], "8")
        self.assertTrue(any(row.get("status") == "visual_slot_grounded" for row in audit))

    def test_slot_supported_value_survives_keyed_grounding(self) -> None:
        internal, audit = generate_from_cached_selection._posthoc_ground_keyed_prediction({
            "gold_papers": [{"paper_id": "paper"}],
            "claim_to_support_keys": {"Q01": ["C001"]},
            "answer": {"freeform": {"text": "P05, P10"}},
            "_validated_slots": [{
                "slot_id": "S001", "status": "supported", "value": "P05, P10", "support_keys": ["C001"],
                "validation": {"visual_supported": False},
            }],
        }, self.hierarchy)
        self.assertEqual(internal["answer"]["freeform"]["text"], "P05, P10")
        self.assertTrue(any(row.get("status") == "slot_supported_grounded" for row in audit))

    def test_venue_mention_does_not_remove_evidence(self) -> None:
        hierarchy = {**self.hierarchy, "l0_catalog": [{
            **self.hierarchy["l0_catalog"][0], "paper_id": "acl2025_01429",
        }]}
        internal, audit = generate_from_cached_selection._posthoc_ground_keyed_prediction({
            "gold_papers": [{"paper_id": "acl2025_01429"}],
            "claim_to_support_keys": {"Q01": ["C001"]},
            "answer": {"freeform": {"text": "P05"}},
        }, {**hierarchy, "query_claims": [{"claim_id": "Q01", "claim": "Which NAACL 2025 papers use MCTS in a figure?"}]})
        self.assertEqual(internal["claim_to_support_keys"], {"Q01": ["C001"]})
        self.assertEqual(internal["gold_papers"], [{"paper_id": "acl2025_01429"}])

    def test_composition_prompt_contains_validated_slots_not_l0_catalog(self) -> None:
        messages = slot_composition_messages(
            self.sample, self.candidates, {"answer_types": ["freeform"]},
            [{"slot_id": "S001", "status": "supported", "value": "71.16", "support_keys": ["C001"]}],
        )
        text = messages[-1]["content"]
        self.assertIn("validated_slots", text)
        self.assertNotIn("E0001", text)

    def test_slot_packet_exposes_titles_as_navigation_not_paper_ids(self) -> None:
        text = slot_extraction_messages(
            self.sample, {"required_source_types": ["table"]}, self.hierarchy,
            [{"paper_id": "paper", "title": "Named Paper"}],
        )[-1]["content"]
        self.assertIn("paper_navigation", text)
        self.assertIn("Named Paper", text)
        self.assertNotIn('"paper_id"', text)

    def test_unique_named_candidate_title_routes_cards_to_that_paper(self) -> None:
        hierarchy = {**self.hierarchy, "l0_catalog": [
            {**self.hierarchy["l0_catalog"][0], "paper_id": "named"},
            {**self.hierarchy["l0_catalog"][0], "evidence_ref": "E0002", "global_record_id": "other::r1", "paper_id": "other", "text": "Method A has F1 99."},
        ], "l2_evidence_cards": [
            {"support_refs": ["E0001"], "proposition": "Named paper score."},
            {"support_refs": ["E0002"], "proposition": "Other paper score."},
        ]}
        cards = slot_cards(
            {"required_source_types": ["table"]}, hierarchy, "In Named Paper, what is the score?",
            candidates=[{"paper_id": "named", "title": "Named Paper"}, {"paper_id": "other", "title": "Other Work"}],
        )
        self.assertEqual(cards[0]["paper_key"], "P01")

    def test_routed_paper_id_filters_slot_cards(self) -> None:
        hierarchy = {**self.hierarchy, "l0_catalog": [
            {**self.hierarchy["l0_catalog"][0], "paper_id": "target"},
            {**self.hierarchy["l0_catalog"][0], "evidence_ref": "E0002", "global_record_id": "other::r1", "paper_id": "other", "text": "Other equation f(x)=9."},
        ], "l2_evidence_cards": [
            {"support_refs": ["E0001"], "proposition": "Target equation f(x)=1."},
            {"support_refs": ["E0002"], "proposition": "Other equation f(x)=9."},
        ]}
        cards = slot_cards(
            {"required_source_types": ["table"], "routed_paper_id": "target"}, hierarchy, "What is the equation?"
        )
        self.assertTrue(cards)
        self.assertTrue(all(card.get("paper_key") == "P01" for card in cards))

    def test_slot_images_are_limited_to_its_visible_cards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "figure.png"
            crop.write_bytes(b"not-decoded-by-this-routing-test")
            hierarchy = {
                "l0_catalog": [{
                    "evidence_ref": "E0001", "global_record_id": "paper::r1", "paper_id": "paper",
                    "section_id": "results", "page": 3, "source_type": "figure", "label": "Figure 4",
                    "text": "Figure 4: DynaPipe result.", "crop_path": str(crop),
                }],
                "l2_evidence_cards": [{"support_refs": ["E0001"], "proposition": "Figure 4: DynaPipe result."}],
            }
            paths = slot_image_paths(
                {**self.sample, "question": "What does Figure 4 show?"},
                {"required_source_types": ["figure"], "entities": ["Figure 4"]}, hierarchy,
                self.candidates, max_images=1,
            )
        self.assertEqual(paths, [str(crop)])

    def test_slot_packet_indexes_an_attached_crop_by_card_key(self) -> None:
        messages = slot_extraction_messages(
            self.sample, {"required_source_types": ["table"]}, self.hierarchy, self.candidates,
            [{"path": "/tmp/ignored.png", "support_card_keys": ["C001"]}],
        )
        text = messages[-1]["content"]
        self.assertIn('"image_index":1', text)
        self.assertIn('"support_card_keys":["C001"]', text)

    def test_slot_mode_runs_plan_extract_compose_and_restores_l0(self) -> None:
        class FakeClient:
            responses = iter([
                {"slots": [{"role": "direct_answer", "operation": "direct", "required_source_types": ["table"], "entities": ["Method A"], "required_conditions": ["F1", "Set X"]}]},
                {"slot_id": "S001", "status": "supported", "value": "71.16", "conditions": {"metric": "F1"}, "support_keys": ["C001"], "missing_conditions": []},
                {"gold_papers": [{"paper_id": "paper"}], "claim_to_slot_ids": {"Q01": ["S001"]}, "answer": {"freeform": {"text": "71.16"}}},
            ])

            def __init__(self, *_args, **_kwargs):
                pass

            def generate_prediction(self, *_args, **_kwargs):
                return {"content": json.dumps(next(self.responses)), "raw_response": {}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official" / "data"
            official.mkdir(parents=True)
            (official / "validation_inputs.jsonl").write_text(json.dumps({
                "query_id": "q_001", "task_family": "hidden_source_single_paper", "primary_evidence_type": "table",
                "question": "What F1 does Method A achieve on Set X?", "answer_types": ["freeform"],
            }) + "\n", encoding="utf-8")
            selected = root / "selected.jsonl"
            selected.write_text(json.dumps({"query_id": "q_001", "selected_records": []}) + "\n", encoding="utf-8")
            candidates = root / "candidates.jsonl"
            candidates.write_text(json.dumps({"query_id": "q_001", "paper_id": "paper", "title": "Paper", "rank": 1}) + "\n", encoding="utf-8")
            hierarchy = root / "hierarchy.jsonl"
            hierarchy.write_text(json.dumps({"query_id": "q_001", "hierarchy": self.hierarchy}) + "\n", encoding="utf-8")
            output = root / "output"
            argv = [
                "generate_from_cached_selection", "--official-dir", str(official.parent),
                "--selected-contexts-input", str(selected), "--candidate-papers-input", str(candidates),
                "--hierarchy-input", str(hierarchy), "--hierarchy-prompt-mode", "keyed", "--generation-mode", "slots",
                "--output-dir", str(output), "--env-path", str(root / "missing.env"),
            ]
            with patch.object(generate_from_cached_selection, "VLMAnswerClient", FakeClient), patch.object(sys, "argv", argv):
                self.assertEqual(generate_from_cached_selection.main(), 0)
            prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            internal = json.loads((output / "internal_predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prediction["answer"]["freeform"]["text"], "71.16")
            self.assertEqual(internal["claim_to_support_keys"], {"Q01": ["C001"]})
            self.assertEqual(len((output / "slot_extractions.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            diagnostics = (output / "errors.jsonl").read_text(encoding="utf-8")
            self.assertIn("evidence_ref_echo_resolved", diagnostics)


if __name__ == "__main__":
    unittest.main()
