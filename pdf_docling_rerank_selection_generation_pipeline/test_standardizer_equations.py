from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from .transcription_backends.base import TranscribedElement
from .transcription_backends.base import TranscribedDocument
from .transcription_backends.standardizer import _record_type_and_source, standardize_transcribed_document
from .data_io import read_jsonl


class StandardizerEquationDetectionTests(unittest.TestCase):
    def test_prose_equation_reference_remains_text(self) -> None:
        element = TranscribedElement(
            paper_id="demo",
            element_id="p4-text-1",
            element_type="text",
            raw_backend_type="text",
            page=4,
            text="As shown in Equation (11), the margin is adaptive.",
        )
        self.assertEqual(
            _record_type_and_source(element),
            ("paragraph", "text_span", ["default_text_to_text_span"]),
        )

    def test_formula_remains_equation_without_a_display_label(self) -> None:
        element = TranscribedElement(
            paper_id="demo",
            element_id="p4-formula-1",
            element_type="text",
            raw_backend_type="formula",
            page=4,
            text="L = x + y",
        )
        self.assertEqual(
            _record_type_and_source(element),
            ("equation", "equation_algorithm", ["docling_formula_to_equation_record"]),
        )

    def test_numbered_algorithm_code_block_remains_algorithm(self) -> None:
        element = TranscribedElement(
            paper_id="demo",
            element_id="p5-code-1",
            element_type="code",
            raw_backend_type="code",
            page=5,
            text="Algorithm 1 Training procedure",
        )
        self.assertEqual(
            _record_type_and_source(element),
            ("algorithm", "equation_algorithm", ["docling_code_algorithm_label_rule"]),
        )

    def test_unnumbered_display_formula_uses_equation_zero(self) -> None:
        document = TranscribedDocument(
            paper_id="demo",
            pdf_path="demo.pdf",
            backend="docling",
            page_count=1,
            elements=[
                TranscribedElement(
                    paper_id="demo", page=1, element_id="formula-1", element_type="formula",
                    raw_backend_type="formula", text="L = x + y",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            standardize_transcribed_document(document, directory)
            records = read_jsonl(Path(directory) / "symbolic_records.runtime.jsonl")
        equation = next(record for record in records if record.get("source_type") == "equation_algorithm")
        self.assertEqual(equation["label"], "Equation 0")
        self.assertEqual(equation["locator"], {"page": 1, "equation_id": "Equation 0"})

    def test_missing_formula_reference_adds_auditable_pointer(self) -> None:
        document = TranscribedDocument(
            paper_id="demo",
            pdf_path="demo.pdf",
            backend="docling",
            page_count=1,
            elements=[
                TranscribedElement(
                    paper_id="demo", page=1, element_id="text-1", element_type="text",
                    raw_backend_type="text", text="As shown in Equation (11), the margin is adaptive.",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            standardize_transcribed_document(document, directory)
            records = read_jsonl(Path(directory) / "symbolic_records.debug.jsonl")
        text_record = next(record for record in records if record.get("record_type") == "paragraph")
        pointer = next(record for record in records if record.get("record_type") == "equation_reference")
        self.assertEqual(text_record["source_type"], "text_span")
        self.assertEqual(pointer["locator"], {"page": 1, "equation_id": "Equation 11"})
        self.assertIn("displayed_equation_body_unavailable", pointer["warnings"])

    def test_numbered_formula_adds_generic_equation_zero_alias(self) -> None:
        document = TranscribedDocument(
            paper_id="demo", pdf_path="demo.pdf", backend="docling", page_count=1,
            elements=[TranscribedElement(
                paper_id="demo", page=1, element_id="formula-1", element_type="formula",
                raw_backend_type="formula", text="L = x + y, (6)", label="Equation 6",
            )],
        )
        with tempfile.TemporaryDirectory() as directory:
            standardize_transcribed_document(document, directory)
            records = read_jsonl(Path(directory) / "symbolic_records.debug.jsonl")
        alias = next(record for record in records if record.get("label") == "Equation 0")
        self.assertEqual(alias["locator"], {"page": 1, "equation_id": "Equation 0"})
        self.assertIn("generic_equation_zero_locator_alias", alias["warnings"])

    def test_math_rich_text_adds_equation_zero_only_when_no_formula_exists(self) -> None:
        document = TranscribedDocument(
            paper_id="demo", pdf_path="demo.pdf", backend="docling", page_count=1,
            elements=[TranscribedElement(
                paper_id="demo", page=1, element_id="text-1", element_type="text", raw_backend_type="text",
                text="The objective is U(y|x) = (pi_theta(y|x))^alpha with beta > 0.",
            )],
        )
        with tempfile.TemporaryDirectory() as directory:
            standardize_transcribed_document(document, directory)
            records = read_jsonl(Path(directory) / "symbolic_records.debug.jsonl")
        fallback = next(record for record in records if record.get("label") == "Equation 0")
        self.assertIn("math_rich_text_to_equation_zero_fallback", fallback["standardization_rules"])
