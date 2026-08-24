import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import profile_predict
from heights.metrics import collect_inference_metrics
from heights.utils import resolve_device


class ProfilePredictTest(unittest.TestCase):
    def test_resolve_cpu_device(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_find_input_files_returns_only_sorted_top_level_txt_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "b.txt").touch()
            (folder / "a.txt").touch()
            (folder / "ignored.csv").touch()
            (folder / "nested").mkdir()
            (folder / "nested" / "c.txt").touch()

            files = profile_predict.find_input_files(folder)

            self.assertEqual([path.name for path in files], ["a.txt", "b.txt"])

    def test_find_input_files_rejects_empty_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No .txt input files"):
                profile_predict.find_input_files(temp_dir)

    @patch("heights.metrics.get_hardware_name", return_value="Test CPU")
    @patch("heights.metrics.get_peak_rss_mb", return_value=100.0)
    def test_collect_inference_metrics_aggregates_records(self, _peak, _hardware):
        records = [
            {
                "tiles_count": 9,
                "preprocess_sec": 1.0,
                "inference_sec": 2.0,
                "postprocess_sec": 3.0,
            },
            {
                "tiles_count": 16,
                "preprocess_sec": 4.0,
                "inference_sec": 5.0,
                "postprocess_sec": 6.0,
            },
        ]

        metrics = collect_inference_metrics(
            torch.device("cpu"), records, total_sec=21.0, images_count=2
        )

        self.assertEqual(metrics["images_count"], 2)
        self.assertEqual(metrics["tiles_count"], 25)
        self.assertEqual(metrics["preprocess_sec"], 5.0)
        self.assertEqual(metrics["inference_sec"], 7.0)
        self.assertEqual(metrics["postprocess_sec"], 9.0)

    def test_main_loads_models_once_for_all_images(self):
        args = SimpleNamespace(
            input_folder="inputs",
            output_folder="outputs",
            device="cpu",
            areas_model_checkpoint="areas.ckpt",
            height_model_checkpoint="heights.ckpt",
        )
        record = {
            "tiles_count": 1,
            "preprocess_sec": 1.0,
            "inference_sec": 1.0,
            "postprocess_sec": 1.0,
        }

        with (
            patch.object(profile_predict, "parse_args", return_value=args),
            patch.object(
                profile_predict,
                "find_input_files",
                return_value=[Path("a.txt"), Path("b.txt")],
            ),
            patch.object(profile_predict, "load_models", return_value=(Mock(), Mock())) as load,
            patch.object(profile_predict, "predict_file", return_value=record) as predict,
            patch.object(profile_predict, "collect_inference_metrics", return_value={}),
            patch.object(profile_predict, "print_inference_metrics"),
        ):
            profile_predict.main()

        load.assert_called_once()
        self.assertEqual(predict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
