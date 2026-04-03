from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hybrid_quant.data.cli import main


class _FakeDownloadService:
    def __init__(self) -> None:
        self.request = None
        self.dataset_path = None
        self.split_dir = None
        self.allow_gaps = None

    def ingest(
        self,
        *,
        request,
        dataset_path,
        split_config,
        split_output_dir,
        allow_gaps,
    ):
        self.request = request
        self.dataset_path = dataset_path
        self.split_dir = split_output_dir
        self.allow_gaps = allow_gaps

        return type(
            "Result",
            (),
            {
                "rows_downloaded": 42,
                "rows_cleaned": 40,
                "dataset_path": dataset_path,
                "validation_report": type(
                    "Validation",
                    (),
                    {
                        "gap_count": 0,
                        "start": "2024-01-01T00:00:00+00:00",
                        "end": "2024-01-02T00:00:00+00:00",
                    },
                )(),
                "split_paths": {},
            },
        )()


class _FakeSplitService:
    def __init__(self) -> None:
        self.input_path = None
        self.output_dir = None
        self.split_config = None

    def split_existing_dataset(self, *, input_path, output_dir, split_config):
        self.input_path = input_path
        self.output_dir = output_dir
        self.split_config = split_config
        return {
            "train": Path(output_dir) / "train.parquet",
            "validation": Path(output_dir) / "validation.parquet",
            "test": Path(output_dir) / "test.parquet",
        }


class DataCliTests(unittest.TestCase):
    def test_download_command_uses_config_defaults_and_arguments(self) -> None:
        service = _FakeDownloadService()
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        with patch("hybrid_quant.data.cli.build_ingestion_service", return_value=service):
            with patch("builtins.print"):
                exit_code = main(
                    [
                        "download",
                        "--config-dir",
                        str(config_dir),
                        "--start",
                        "2024-01-01T00:00:00+00:00",
                        "--limit",
                        "500",
                        "--skip-splits",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.request.symbol, "BTCUSDT")
        self.assertEqual(service.request.interval, "5m")
        self.assertEqual(service.request.limit, 500)
        self.assertTrue(str(service.dataset_path).endswith("data\\raw\\BTCUSDT\\5m\\ohlcv.parquet"))
        self.assertIsNone(service.split_dir)
        self.assertFalse(service.allow_gaps)

    def test_split_command_uses_default_output_dir(self) -> None:
        service = _FakeSplitService()
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "ohlcv.parquet"
            with patch("hybrid_quant.data.cli.build_ingestion_service", return_value=service):
                with patch("builtins.print"):
                    exit_code = main(
                        [
                            "split",
                            "--config-dir",
                            str(config_dir),
                            "--input-path",
                            str(input_path),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.input_path, str(input_path))
        self.assertEqual(service.output_dir, input_path.parent / "splits")
        self.assertAlmostEqual(service.split_config.train_ratio, 0.7)
        self.assertAlmostEqual(service.split_config.validation_ratio, 0.15)
        self.assertAlmostEqual(service.split_config.test_ratio, 0.15)

