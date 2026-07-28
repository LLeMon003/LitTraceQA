from __future__ import annotations

import unittest
from pathlib import Path


class CachedRefinementTests(unittest.TestCase):
    def test_runner_uses_existing_draft_and_fail_soft_fallback(self) -> None:
        from . import refine_cached_keyed_predictions as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("_refine_keyed_draft", source)
        self.assertIn("preserved_refinement_failure", source)
        self.assertIn("predictions.append(base)", source)

    def test_runner_restores_public_answer_types_before_normalization(self) -> None:
        from .refine_cached_keyed_predictions import _effective_contract

        effective = _effective_contract(
            {"answer_types": ["freeform", "multiple_choice"]},
            {"query_id": "q_001", "multiple_choice": {"options": []}},
        )
        self.assertEqual(effective["answer_types"], ["freeform", "multiple_choice"])
        self.assertEqual(effective["multiple_choice"], {"options": []})
