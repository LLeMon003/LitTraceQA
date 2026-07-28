from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .generate_from_cached_selection import _write_or_validate_provenance, build_generation_provenance


class GenerationProvenanceTests(unittest.TestCase):
    def test_manifest_hashes_only_inference_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("validation_inputs.jsonl", "selected.jsonl", "candidates.jsonl", "hierarchy.jsonl", "contracts.jsonl"):
                path = root / name
                path.write_text('{"query_id":"q"}\n', encoding="utf-8")
                paths[name] = path
            manifest = build_generation_provenance(
                validation_inputs_path=paths["validation_inputs.jsonl"],
                selected_contexts_path=paths["selected.jsonl"],
                candidate_papers_path=paths["candidates.jsonl"],
                hierarchy_path=paths["hierarchy.jsonl"],
                answer_contracts_path=paths["contracts.jsonl"],
                env_path=".env",
                generation_parameters={"hierarchy_prompt_mode": "keyed", "max_prompt_chars": 20000},
            )
            self.assertTrue(manifest["inference_inputs_only"])
            self.assertNotIn("validation.jsonl", {Path(item["path"]).name for item in manifest["sources"].values()})
            _write_or_validate_provenance(root, manifest, resume=False)
            self.assertEqual(json.loads((root / "generation_provenance.json").read_text(encoding="utf-8")), manifest)
            _write_or_validate_provenance(root, manifest, resume=True)

    def test_rejects_validation_gold_as_a_generation_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "validation.jsonl"
            gold.write_text('{"query_id":"q"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                build_generation_provenance(
                    validation_inputs_path=gold,
                    selected_contexts_path=gold,
                    candidate_papers_path=gold,
                    hierarchy_path=None,
                    answer_contracts_path=None,
                    env_path=".env",
                    generation_parameters={"hierarchy_prompt_mode": "keyed"},
                )
