import tempfile
import unittest
from pathlib import Path

from .transcription_backends.docling_backend import _cached_artifacts_path


class DoclingBackendTests(unittest.TestCase):
    def test_uses_cache_only_when_required_models_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            self.assertIsNone(_cached_artifacts_path(cache_dir, None))
            models = cache_dir / "models"
            (models / "docling-project--docling-layout-heron").mkdir(parents=True)
            self.assertIsNone(_cached_artifacts_path(cache_dir, None))
            (models / "docling-project--docling-models").mkdir()
            self.assertEqual(_cached_artifacts_path(cache_dir, None), models)

