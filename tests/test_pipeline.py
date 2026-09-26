"""Integration checks for inference artifacts and reference evaluation."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rootscope.batch import MODEL_PATH, run_batch
from rootscope.engine import InferenceEngine, evaluate
from rootscope.models import catalog, model_path, model_spec


class RootScopePipelineTests(unittest.TestCase):
    def test_model_and_batch_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            image = np.full((256, 256), 215, dtype=np.uint8)
            cv2.line(image, (120, 15), (122, 238), 35, 3)
            source = base / "sample.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            result = run_batch([source], [], base / "results", make_zip=True)
            self.assertEqual(len(result["results"]), 1)
            self.assertFalse(result["errors"])
            folder = Path(result["folder"])
            self.assertTrue(Path(result["archive"]).is_file())
            self.assertTrue((folder / "results.csv").is_file())
            self.assertTrue((folder / "provenance.json").is_file())
            with (folder / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertIn("dominant_path_length", rows[0])
            self.assertIn("mean_local_acute_angle_deg", rows[0])
            result_json = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result_json), 1)
            image_folder = folder / "images" / result_json[0]["image_id"]
            mask = cv2.imread(str(image_folder / "prediction_mask.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, image.shape)
            self.assertTrue((image_folder / "overlay.png").is_file())
            self.assertTrue((image_folder / "skeleton.png").is_file())

    def test_identical_reference_scores_one(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[4:58, 29:34] = 1
        scores = evaluate(mask, mask)
        for key in ("precision", "recall", "dice", "iou", "hard_cldice"):
            self.assertAlmostEqual(scores[key], 1.0)

    def test_model_hash_and_output_shape(self):
        engine = InferenceEngine(MODEL_PATH)
        probability = engine.predict(np.zeros((256, 256), dtype=np.uint8))
        self.assertEqual(probability.shape, (256, 256))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_all_five_locked_models_load_and_predict(self):
        items = catalog()["models"]
        self.assertEqual([item["id"] for item in items], [f"fold_{fold}" for fold in range(5)])
        for item in items:
            with self.subTest(model=item["id"]):
                self.assertTrue(model_path(item["id"]).is_file())
                engine = InferenceEngine(model_path(item["id"]), item["sha256"])
                prediction = engine.predict(np.zeros((256, 256), dtype=np.uint8))
                self.assertEqual(prediction.shape, (256, 256))
                self.assertTrue(np.isfinite(prediction).all())

    def test_selected_fold_threshold_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "sample.png"
            self.assertTrue(cv2.imwrite(str(source), np.full((256, 256), 180, np.uint8)))
            result = run_batch([source], [], base / "results", model_id="fold_1", make_zip=False)
            self.assertFalse(result["errors"])
            provenance = json.loads((Path(result["folder"]) / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["model_id"], "fold_1")
            self.assertEqual(provenance["study_fold"], 1)
            self.assertEqual(provenance["used_threshold"], model_spec("fold_1")["threshold"])


if __name__ == "__main__":
    unittest.main()
