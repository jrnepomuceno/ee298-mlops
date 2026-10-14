from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from pi5_vcm.dataset_source import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_DOWNLOAD_PATTERNS,
    download_training_data,
)


class DatasetSourceTests(unittest.TestCase):
    @patch("pi5_vcm.dataset_source.snapshot_download")
    def test_download_uses_pinned_revision_and_training_files_only(
        self, snapshot_download: unittest.mock.MagicMock
    ) -> None:
        snapshot_download.return_value = "/cache/datasets/snapshot"

        result = download_training_data(cache_dir="/tmp/hf-cache")

        self.assertEqual(result, Path("/cache/datasets/snapshot"))
        snapshot_download.assert_called_once_with(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=list(DATASET_DOWNLOAD_PATTERNS),
            cache_dir="/tmp/hf-cache",
        )


if __name__ == "__main__":
    unittest.main()
