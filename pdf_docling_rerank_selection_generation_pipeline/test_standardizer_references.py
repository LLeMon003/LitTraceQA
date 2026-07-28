import unittest

from pdf_docling_rerank_selection_generation_pipeline.transcription_backends.standardizer import _split_reference_entries


class ReferenceStandardizerTests(unittest.TestCase):
    def test_preserves_printed_reference_number_and_merges_continuation(self):
        records = [
            {"global_record_id": "p::header", "record_type": "section_header", "source_type": "text_span", "text": "References"},
            {"global_record_id": "p::one", "record_type": "paragraph", "source_type": "text_span", "text": "[12] First author. Main reference."},
            {"global_record_id": "p::continuation", "record_type": "paragraph", "source_type": "text_span", "text": "continuation on the next column."},
            {"global_record_id": "p::two", "record_type": "paragraph", "source_type": "text_span", "text": "13. Second author. Another reference."},
        ]
        output, debug = _split_reference_entries(records, {row["global_record_id"]: dict(row) for row in records})
        references = [row for row in output if row.get("record_type") == "reference_entry"]
        self.assertEqual([row["label"] for row in references], ["Reference 12", "Reference 13"])
        self.assertIn("continuation on the next column.", references[0]["text"])
        self.assertIn("references_continuation_merged", debug["p::one"]["standardization_rules"])


if __name__ == "__main__":
    unittest.main()
