from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .parser import _resolve_evidence_ref_echo, normalize_prediction
from .run_pdf_extraction_symbolic_vlm_baseline import _acquire_output_lock, _messages_for_json_retry, _remove_query_rows
from .symbolic_context_selector import audit_selected_context
from .vlm_answer_client import VLMAnswerClient


class PromptGroundingTests(unittest.TestCase):
    def test_prompt_audit_requires_exact_selected_record_alignment(self):
        selected = [{
            "evidence_ref": "E0001",
            "paper_id": "paper",
            "page": 3,
            "source_type": "figure",
            "label": "Figure 2",
            "locator": {"page": 3, "figure_id": "Figure 2"},
            "text": "caption",
            "section_id": "sec",
            "section_title": "Method",
            "section_type": "method",
            "section_path": ["Method"],
            "reading_order": 4,
            "document_order": 4,
        }]
        packet = {
            "paper_id": "paper",
            "section_id": "sec",
            "section_title": "Method",
            "section_type": "method",
            "section_path": ["Method"],
            "record_defaults": {"page": 3, "source_type": "figure"},
            "records": [{
                "evidence_ref": "E0001",
                "label": "Figure 2",
                "locator": {"figure_id": "Figure 2"},
                "text": "caption",
                "reading_order": 4,
                "document_order": 4,
            }],
        }
        self.assertTrue(audit_selected_context(selected, [packet], [])["passed"])
        packet["records"][0]["text"] = "wrong"
        with self.assertRaisesRegex(ValueError, "field mismatch: text"):
            audit_selected_context(selected, [packet], [])

    def test_evidence_ref_echo_restores_exact_selected_locator(self):
        selected = [{
            "evidence_ref": "E0007",
            "paper_id": "paper",
            "source_type": "table",
            "locator": {"page": 8, "table_id": "Table 3"},
        }]
        evidence, errors = _resolve_evidence_ref_echo(
            {"evidence": [{"evidence_ref": "E0007"}]},
            selected,
        )
        self.assertEqual(
            evidence,
            [{
                "paper_id": "paper",
                "source_type": "table",
                "locator": {"page": 8, "table_id": "Table 3"},
            }],
        )
        self.assertEqual(errors, ["evidence_ref_echo_resolved"])
        prediction, _ = normalize_prediction(
            {
                "gold_papers": [{"paper_id": "paper"}],
                "evidence": [{"evidence_ref": "E0007"}],
                "answer": {"freeform": {"text": "x"}},
            },
            {
                "query_id": "q",
                "task_family": "hidden_source_single_paper",
                "primary_evidence_type": "table",
                "answer_types": ["freeform"],
                "question": "value?",
            },
            ["paper"],
            selected_evidence=selected,
        )
        self.assertEqual(prediction["evidence"], evidence)

    def test_answer_client_attaches_real_image(self):
        config = SimpleNamespace(answer_api_key="test-key-123456", answer_model="Qwen3-VL")
        client = VLMAnswerClient(config)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            messages = client._with_images(
                [{"role": "user", "content": "inspect IMG001"}],
                [image_path],
            )
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "inspect IMG001"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as attached:
            self.assertEqual(attached.size, (32, 32))

    def test_json_retry_requires_complete_sparse_evidence(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "original"}]
        retried = _messages_for_json_retry(messages, 12)
        self.assertEqual(messages[1]["content"], "original")
        self.assertIn("complete valid JSON", retried[1]["content"])
        self.assertIn("at most 12", retried[1]["content"])

    def test_query_replacement_preserves_other_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            runtime_path = Path(directory) / "symbolic_records.runtime.jsonl"
            path.write_text(
                "\n".join(json.dumps({"query_id": query_id}) for query_id in ("q_001", "q_020", "q_001")) + "\n",
                encoding="utf-8",
            )
            runtime_path.write_text(json.dumps({"paper_id": "paper"}) + "\n", encoding="utf-8")
            _remove_query_rows(
                {"predictions": path, "symbolic_records_runtime": runtime_path},
                {"q_001"},
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            runtime_row = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(rows, [{"query_id": "q_020"}])
        self.assertEqual(runtime_row, {"paper_id": "paper"})

    def test_output_lock_rejects_concurrent_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _acquire_output_lock(Path(directory))
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    _acquire_output_lock(Path(directory))
            finally:
                first.close()


if __name__ == "__main__":
    unittest.main()
