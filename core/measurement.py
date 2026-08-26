"""
Geometric Quantification of Concrete Cracks using Computer Vision
Calculates: Skeletonization (Zhang-Suen Thinning), Euclidean Distance Transform, and Damage Area.
"""

import cv2
import numpy as np
from typing import Dict, Any

class CrackMeasurer:
    def __init__(self, pixel_to_mm_ratio: float = 0.2):
        """
        :param pixel_to_mm_ratio: Millimeters represented by one pixel (mm/px).
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Zhang-Suen morphological thinning algorithm for true 1-pixel wide centerline.
        """
        img = (binary_mask > 127).astype(np.uint8)
        
        while True:
            # 8-neighbor shifting
            P2 = np.pad(img[:-2, 1:-1], ((1,1),(1,1)), 'constant')
            P3 = np.pad(img[:-2, 2:], ((1,1),(1,1)), 'constant')
            P4 = np.pad(img[1:-1, 2:], ((1,1),(1,1)), 'constant')
            P5 = np.pad(img[2:, 2:], ((1,1),(1,1)), 'constant')
            P6 = np.pad(img[2:, 1:-1], ((1,1),(1,1)), 'constant')
            P7 = np.pad(img[2:, :-2], ((1,1),(1,1)), 'constant')
            P8 = np.pad(img[1:-1, :-2], ((1,1),(1,1)), 'constant')
            P9 = np.pad(img[:-2, :-2], ((1,1),(1,1)), 'constant')

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            cond1 = (B >= 2) & (B <= 6)

            P = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros(img.shape, dtype=np.uint8)
            for i in range(8):
                A += ((P[i] == 0) & (P[i+1] == 1)).astype(np.uint8)
            cond2 = (A == 1)

            # Step 1 conditions
            cond3_1 = (P2 * P4 * P6 == 0)
            cond4_1 = (P4 * P6 * P8 == 0)
            del_mask1 = (img == 1) & cond1 & cond2 & cond3_1 & cond4_1
            img[del_mask1] = 0

            # Recalculate for Step 2
            P2 = np.pad(img[:-2, 1:-1], ((1,1),(1,1)), 'constant')
            P3 = np.pad(img[:-2, 2:], ((1,1),(1,1)), 'constant')
            P4 = np.pad(img[1:-1, 2:], ((1,1),(1,1)), 'constant')
            P5 = np.pad(img[2:, 2:], ((1,1),(1,1)), 'constant')
            P6 = np.pad(img[2:, 1:-1], ((1,1),(1,1)), 'constant')
            P7 = np.pad(img[2:, :-2], ((1,1),(1,1)), 'constant')
            P8 = np.pad(img[1:-1, :-2], ((1,1),(1,1)), 'constant')
            P9 = np.pad(img[:-2, :-2], ((1,1),(1,1)), 'constant')

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            cond1 = (B >= 2) & (B <= 6)

            P = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros(img.shape, dtype=np.uint8)
            for i in range(8):
                A += ((P[i] == 0) & (P[i+1] == 1)).astype(np.uint8)
            cond2 = (A == 1)

            cond3_2 = (P2 * P4 * P8 == 0)
            cond4_2 = (P2 * P6 * P8 == 0)
            del_mask2 = (img == 1) & cond1 & cond2 & cond3_2 & cond4_2
            img[del_mask2] = 0

            if not np.any(del_mask1) and not np.any(del_mask2):
                break

        return (img * 255).astype(np.uint8)

    def measure(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract complete geometric profile from binary segmentation mask.
        """
        _, thresh = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        crack_pixel_count = np.count_nonzero(thresh)

        if crack_pixel_count == 0:
            return {
                "detected": False,
                "length_mm": 0.0,
                "max_width_mm": 0.0,
                "avg_width_mm": 0.0,
                "area_mm2": 0.0,
                "skeleton_map": np.zeros_like(thresh),
                "dist_map": np.zeros_like(thresh, dtype=np.float32)
            }

        # 1. Total Damage Area
        area_mm2 = crack_pixel_count * (self.pixel_to_mm_ratio ** 2)

        # 2. Skeleton & Centerline Length
        skeleton = self.skeletonize(thresh)
        skeleton_pixels = np.count_nonzero(skeleton)
        length_mm = skeleton_pixels * self.pixel_to_mm_ratio

        # 3. Width Profile via Euclidean Distance Transform
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        skeleton_distances = dist_transform[skeleton > 0]

        if len(skeleton_distances) > 0:
            max_width_px = float(np.max(skeleton_distances) * 2.0)
            avg_width_px = float(np.mean(skeleton_distances) * 2.0)
        else:
            max_width_px = 0.0
            avg_width_px = 0.0

        max_width_mm = max_width_px * self.pixel_to_mm_ratio
        avg_width_mm = avg_width_px * self.pixel_to_mm_ratio

        return {
            "detected": True,
            "length_mm": round(length_mm, 2),
            "max_width_mm": round(max_width_mm, 2),
            "avg_width_mm": round(avg_width_mm, 2),
            "area_mm2": round(area_mm2, 2),
            "skeleton_map": skeleton,
            "dist_map": dist_transform
        }
