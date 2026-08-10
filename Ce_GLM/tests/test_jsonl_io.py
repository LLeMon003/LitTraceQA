from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.jsonl_io import JSONLParseError, inspect_jsonl, read_jsonl, write_jsonl


class JSONLTransportTests(unittest.TestCase):
    def test_semantic_round_trip_unicode_and_controls(self) -> None:
        records = [
            {
                "query_id": "q_001",
                "u2028": "left\u2028right",
                "u2029": "left\u2029right",
                "unicode_minus": "−3.5",
                "name": "Zoë 李",
                "controls": "tab\tline\nnext",
                "empty_string": "",
                "empty_list": [],
                "empty_object": {},
                "null": None,
                "nested": {"array": [1, {"text": "café"}]},
            },
            {"query_id": "q_002", "large": "λ" * 1_000_000},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.jsonl"
            write_jsonl(path, records)
            self.assertEqual(read_jsonl(path), records)
            raw = path.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(len(raw.splitlines()), len(records))
            self.assertIn("\u2028".encode("utf-8"), raw)
            self.assertIn("\u2029".encode("utf-8"), raw)
            self.assertIn(b"\\n", raw)
            self.assertIn(b"\\t", raw)

    def test_inspection_counts_schema_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(path, [{"query_id": "a", "x": 1}, {"query_id": "b", "x": 2}])
            result = inspect_jsonl(path)
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["unique_query_ids"], 2)
            self.assertEqual(result["duplicate_query_ids"], [])
            self.assertEqual(result["schema_counts"], {"query_id|x": 2})

    def test_line_numbered_parser_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_bytes(b'{"query_id":"ok"}\n{"query_id":}\n')
            with self.assertRaises(JSONLParseError) as raised:
                read_jsonl(path)
            self.assertEqual(raised.exception.location.line, 2)
            self.assertIsNotNone(raised.exception.location.column)
            self.assertIn(str(path), str(raised.exception))

    def test_blank_line_and_non_object_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "blank.jsonl"
            blank.write_text('{"a":1}\n\n', encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(JSONLParseError, "blank physical line"):
                read_jsonl(blank)
            array = Path(directory) / "array.jsonl"
            array.write_text(json.dumps([1, 2]) + "\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(JSONLParseError, "must be a JSON object"):
                read_jsonl(array)

    def test_invalid_utf8_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_utf8.jsonl"
            path.write_bytes(b'{"text":"\xff"}\n')
            with self.assertRaisesRegex(JSONLParseError, "invalid UTF-8"):
                read_jsonl(path)

    def test_non_dictionary_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_write.jsonl"
            with self.assertRaisesRegex(TypeError, "must be a dictionary"):
                write_jsonl(path, [{"ok": True}, ["not", "an", "object"]])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
