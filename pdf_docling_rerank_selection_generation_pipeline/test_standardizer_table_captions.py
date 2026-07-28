import tempfile
import unittest
from pathlib import Path

from pdf_docling_rerank_selection_generation_pipeline.data_io import read_jsonl
from pdf_docling_rerank_selection_generation_pipeline.transcription_backends.base import (
    TranscribedDocument,
    TranscribedElement,
)
from pdf_docling_rerank_selection_generation_pipeline.transcription_backends.standardizer import (
    standardize_transcribed_document,
)


class TableCaptionStandardizerTests(unittest.TestCase):
    def test_unclaimed_caption_becomes_table_anchor_without_duplicate(self):
        document = TranscribedDocument(
            paper_id="demo",
            pdf_path="demo.pdf",
            backend="docling",
            page_count=2,
            elements=[
                TranscribedElement(
                    paper_id="demo",
                    page=1,
                    element_id="caption-1",
                    element_type="caption",
                    raw_backend_type="caption",
                    text="Table 1: Caption retained although the table object is absent.",
                    bbox=[10, 10, 100, 20],
                ),
                TranscribedElement(
                    paper_id="demo",
                    page=2,
                    element_id="table-2",
                    element_type="table",
                    label="Table 2",
                    text="Method | Score\nA | 1",
                    bbox=[10, 10, 100, 100],
                ),
                TranscribedElement(
                    paper_id="demo",
                    page=2,
                    element_id="caption-2",
                    element_type="caption",
                    raw_backend_type="caption",
                    text="Table 2: The real structured table above.",
                    bbox=[10, 101, 100, 120],
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            standardize_transcribed_document(document, directory)
            records = read_jsonl(Path(directory) / "symbolic_records.runtime.jsonl")

        tables = [record for record in records if record.get("source_type") == "table"]
        by_label = {}
        for record in tables:
            by_label.setdefault(record.get("label"), []).append(record)
        self.assertEqual(len(by_label["Table 1"]), 1)
        self.assertEqual(by_label["Table 1"][0]["record_type"], "table_caption")
        self.assertEqual(by_label["Table 1"][0]["locator"], {"page": 1, "table_id": "Table 1"})
        self.assertEqual(len(by_label["Table 2"]), 1)
        self.assertEqual(by_label["Table 2"][0]["record_type"], "table")


if __name__ == "__main__":
    unittest.main()
