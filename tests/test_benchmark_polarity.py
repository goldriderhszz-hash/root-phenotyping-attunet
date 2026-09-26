"""Guard the input and exported-mask polarity of the RVE comparison."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from benchmark.evaluate_rve_comparison import (
    rve_foreground,
    scores,
    verify_rve_input,
    verify_rve_metadata,
)


class TestBenchmarkPolarity(unittest.TestCase):
    def test_inverted_input_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            prepared = root / "prepared" / "rve_validation_fold0"
            source.mkdir()
            prepared.mkdir(parents=True)
            image = np.array([[210, 210], [180, 210]], dtype=np.uint8)
            cv2.imwrite(str(source / "example.tif"), image)
            cv2.imwrite(str(prepared / "example.tif"), image)
            verify_rve_input(source, root / "prepared", "validation", "example")
            cv2.imwrite(str(prepared / "example.tif"), 255 - image)
            with self.assertRaisesRegex(ValueError, "polarity"):
                verify_rve_input(source, root / "prepared", "validation", "example")

    def test_black_export_is_root_foreground(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "example_seg.png"
            cv2.imwrite(str(path), np.array([[255, 255], [0, 255]], dtype=np.uint8))
            foreground = rve_foreground(path)
            reference = np.array([[False, False], [True, False]])
            self.assertEqual(scores(foreground, reference)["dice"], 1.0)
            self.assertEqual(int(foreground.sum()), 1)

    def test_inversion_setting_is_checked(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.csv"
            path.write_text(
                "RhizoVision Explorer Version, 2.0.3\n"
                "Root type, Whole root\n"
                "Image Thresholding Level, 190\n"
                "Invert images, true\n"
                "Keep largest component, true\n"
                "Save segmented images, true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invert images"):
                verify_rve_metadata(Path(folder), 190)


if __name__ == "__main__":
    unittest.main()
