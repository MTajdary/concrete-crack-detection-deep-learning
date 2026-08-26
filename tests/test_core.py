import unittest
import numpy as np
import cv2
from core.measurement import CrackMeasurer
from core.structural_rules import StructuralRulesEngine
from core.detector import CrackDetector

class TestCoreModules(unittest.TestCase):
    def setUp(self):
        self.measurer = CrackMeasurer(pixel_to_mm_ratio=0.1)
        self.detector = CrackDetector()

    def test_measurement_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        res = self.measurer.measure(mask)
        self.assertFalse(res["detected"])
        self.assertEqual(res["length_mm"], 0.0)

    def test_measurement_known_geometry(self):
        # 100px vertical line, 5px thickness drawn as rectangle (width 6px, height 100px)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 97:103] = 255 # 100px height, 6px width
        res = self.measurer.measure(mask)
        self.assertTrue(res["detected"])
        # Expected length approx 100px * 0.1 = 10mm
        self.assertAlmostEqual(res["length_mm"], 10.0, delta=2.0)
        # Expected max width approx 6px * 0.1 = 0.6mm
        self.assertAlmostEqual(res["max_width_mm"], 0.6, delta=0.2)

    def test_structural_rules(self):
        # Test minor crack
        eval_minor = StructuralRulesEngine.evaluate(max_width_mm=0.15, avg_width_mm=0.1, length_mm=20.0, exposure="humid_air")
        self.assertEqual(eval_minor.risk_code, "LOW-1")

        # Test critical crack
        eval_crit = StructuralRulesEngine.evaluate(max_width_mm=0.8, avg_width_mm=0.6, length_mm=150.0, exposure="humid_air")
        self.assertEqual(eval_crit.risk_code, "HIGH-3")

    def test_detector_overlay(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        overlay = self.detector.create_overlay(img, mask)
        self.assertEqual(overlay.shape, (50, 50, 3))

if __name__ == '__main__':
    unittest.main()
