from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "844C6248E8A353783BE7600050BBB247A7716931B24C3C2C10FD173B04CE6914"


class ReproducibleReleaseTest(unittest.TestCase):
    def test_clean_rebuild_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test_predictions.jsonl"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build.py"), "--output", str(output)],
                cwd=ROOT,
                check=True,
            )
            digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
            self.assertEqual(digest, EXPECTED)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "verify_release.py"), "--prediction", str(output)],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
