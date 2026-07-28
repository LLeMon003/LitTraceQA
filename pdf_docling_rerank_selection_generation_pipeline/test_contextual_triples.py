from __future__ import annotations

import unittest

from .contextual_triples import (
    attach_contextual_triple_graph,
    structural_sufficiency_precheck,
    sufficiency_messages,
    triple_generation_messages,
    validate_sufficiency_decision,
)


class ContextualTripleTests(unittest.TestCase):
    def _hierarchy(self) -> dict:
        return {
            "selected_anchor_refs": ["E0001"],
            "l0_catalog": [{
                "evidence_ref": "E0001", "paper_id": "paper", "page": 2,
                "source_type": "table", "label": "Table 1", "section_path": ["Results"],
                "text": "Table 1: F1 scores. | Method | F1 | | Ours | 91.2 |",
            }, {
                "evidence_ref": "E0002", "paper_id": "paper", "page": 2,
                "source_type": "text_span", "section_path": ["Results"],
                "text": "Higher F1 is better.",
            }],
            "l1_contexts": [{
                "anchor_ref": "E0001", "neighbor_refs": ["E0002"],
                "table_context": {"caption": "Table 1: F1 scores.", "columns": ["Method", "F1"], "rows": [{"row_label": "Ours", "values": {"Method": "Ours", "F1": "91.2"}}]},
            }],
            "l2_evidence_cards": [{
                "card_id": "C001", "claim_ids": ["Q01"], "proposition": "Ours has F1 91.2.",
                "entities": ["Ours"], "values": ["91.2"], "conditions": [],
                "support_refs": ["E0001"], "support_quotes": [{"evidence_ref": "E0001", "quote": "| Ours | 91.2 |"}],
            }],
            "l3_navigation": {"unresolved_claims": []},
        }

    def test_windows_and_triples_keep_raw_provenance(self) -> None:
        hierarchy = attach_contextual_triple_graph("What is Ours F1?", self._hierarchy())
        window = hierarchy["l1_evidence_windows"][0]
        triple = hierarchy["l2_contextual_triples"][0]
        self.assertEqual(window["support_refs"], ["E0001", "E0002"])
        self.assertEqual(window["table_schema"]["columns"], ["Method", "F1"])
        self.assertTrue(window["table_lines"])
        self.assertEqual(triple["support_refs"], ["E0001", "E0002"])
        self.assertEqual(triple["support_quotes"][0]["evidence_ref"], "E0001")
        self.assertEqual(hierarchy["l3_triple_navigation"]["claim_to_triples"]["Q01"], ["T001"])

    def test_llm_payload_uses_l1_not_raw_catalog(self) -> None:
        hierarchy = attach_contextual_triple_graph("What is Ours F1?", self._hierarchy())
        messages = triple_generation_messages("What is Ours F1?", hierarchy)
        payload = messages[-1]["content"]
        self.assertIn('"windows"', payload)
        self.assertIn('"table_schema"', payload)
        self.assertNotIn('"l0_catalog"', payload)

    def test_rich_cards_bound_triple_count_when_l0_union_is_large(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["selected_anchor_refs"] = ["E0001"] * 500
        hierarchy = attach_contextual_triple_graph("What is Ours F1?", hierarchy)
        self.assertEqual(len(hierarchy["l1_evidence_windows"]), 1)
        self.assertEqual(len(hierarchy["l2_contextual_triples"]), 1)

    def test_sufficiency_gate_only_allows_known_triple_keys(self) -> None:
        hierarchy = attach_contextual_triple_graph("What is Ours F1?", self._hierarchy())
        self.assertEqual(structural_sufficiency_precheck(hierarchy)["status"], "ready_for_semantic_judge")
        message = sufficiency_messages("What is Ours F1?", hierarchy)[-1]["content"]
        self.assertIn('"candidate_paths"', message)
        decision = validate_sufficiency_decision({
            "sufficient": False, "expand_l1_triple_ids": ["T001", "T999"], "expand_l0_triple_ids": ["T001"],
        }, hierarchy)
        self.assertEqual(decision["expand_l1_triple_ids"], ["T001"])
        self.assertEqual(decision["expand_l0_triple_ids"], ["T001"])

    def test_text_triple_payload_excludes_crop_backed_figures(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["l0_catalog"].append({
            "evidence_ref": "E0003", "paper_id": "paper", "source_type": "figure", "section_path": ["Results"],
            "crop_path": "/tmp/figure.png", "text": "Figure 2 caption.",
        })
        hierarchy["selected_anchor_refs"].append("E0003")
        hierarchy["l2_evidence_cards"].append({"card_id": "C002", "support_refs": ["E0003"], "proposition": "Figure 2 caption."})
        hierarchy = attach_contextual_triple_graph("What does Figure 2 show?", hierarchy)
        payload = triple_generation_messages("What does Figure 2 show?", hierarchy)[-1]["content"]
        self.assertNotIn('Figure 2 caption.', payload)

    def test_explicit_citation_seed_reaches_l1_without_a_rich_card(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["l0_catalog"].append({
            "evidence_ref": "E0003", "paper_id": "paper", "page": 9, "source_type": "citation_context",
            "section_path": ["References"], "locator": {"citation_id": "24"},
            "text": "Freda Shi et al. Language models are multilingual reasoners.",
        })
        hierarchy["l3_navigation"] = {"papers": [{"paper_id": "paper", "title": "EasySpec: Efficient Decoding"}]}
        hierarchy = attach_contextual_triple_graph("Who is the first author of the 24th reference in EasySpec?", hierarchy)
        window = next(row for row in hierarchy["l1_evidence_windows"] if row["anchor_ref"] == "E0003")
        self.assertIn("explicit_citation_id:24", window["seed_reasons"])

    def test_hardware_query_expansion_seeds_gpu_record(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["l0_catalog"].append({
            "evidence_ref": "E0003", "paper_id": "paper", "page": 9, "source_type": "text_span",
            "section_path": ["Experimental Setup"], "text": "All experiments ran on one NVIDIA RTX 4090 GPU.",
        })
        hierarchy = attach_contextual_triple_graph("What hardware was used?", hierarchy)
        window = next(row for row in hierarchy["l1_evidence_windows"] if row["anchor_ref"] == "E0003")
        self.assertIn("idf_lexical_navigation", window["seed_reasons"])

    def test_first_author_citation_has_deterministic_provenance(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["l0_catalog"].append({
            "evidence_ref": "E0003", "paper_id": "paper", "page": 9, "source_type": "citation_context",
            "section_path": ["References"], "locator": {"citation_id": "24"},
            "text": "Freda Shi, Mirac Suzgun, et al. Language models are multilingual reasoners.",
        })
        hierarchy["l3_navigation"] = {"papers": [{"paper_id": "paper", "title": "EasySpec: Efficient Decoding"}]}
        hierarchy = attach_contextual_triple_graph("Who is the first author of the 24th reference in EasySpec?", hierarchy)
        triple = next(item for item in hierarchy["l2_contextual_triples"] if item["verification"]["status"] == "deterministic_citation_relation")
        self.assertEqual(triple["object"], "Freda Shi")
        self.assertEqual(triple["support_refs"], ["E0003"])

    def test_hardware_relation_uses_exact_device_phrase(self) -> None:
        hierarchy = self._hierarchy()
        hierarchy["l0_catalog"].append({
            "evidence_ref": "E0003", "paper_id": "paper", "page": 9, "source_type": "text_span",
            "section_path": ["Experimental Setup"], "text": "All experiments are run on a single NVIDIA RTX 4090 GPU.",
        })
        hierarchy = attach_contextual_triple_graph("What hardware configuration was used?", hierarchy)
        triple = next(item for item in hierarchy["l2_contextual_triples"] if item["verification"]["status"] == "deterministic_hardware_relation")
        self.assertEqual(triple["object"], "a single NVIDIA RTX 4090 GPU")
        self.assertEqual(triple["support_refs"], ["E0003"])
