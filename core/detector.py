"""
Detector Interface & Segmentation Wrapper
Provides standard pipeline for processing RGB inspection images into binary crack masks.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

class CrackDetector:
    """
    Crack Segmentation engine. Supports deep learning weights or morphological pre-processor.
    """
    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self.is_dl_loaded = weights_path is not None

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Accepts RGB image, returns binary mask (0 for background, 255 for crack).
        """
        if self.is_dl_loaded:
            # Place holder for YOLOv8-seg / PyTorch model inference
            pass
        
        # Adaptive thresholding and edge morphological detector as robust baseline
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Illumination equalization via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to smooth texture while keeping crack edges sharp
        blurred = cv2.bilateralFilter(enhanced, 7, 50, 50)
        
        # Adaptive thresholding to isolate darker fissures
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 6
        )
        
        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        min_size = 40  # Minimum pixel area for a valid crack component
        cleaned_mask = np.zeros_like(opened)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255
                
        return cleaned_mask

    def create_overlay(self, original_rgb: np.ndarray, binary_mask: np.ndarray, skeleton: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generates visual overlay: Red mask for crack, Green line for skeleton.
        """
        overlay = original_rgb.copy()
        
        # Red highlight on crack body
        overlay[binary_mask > 0] = [255, 60, 60]
        blended = cv2.addWeighted(original_rgb, 0.65, overlay, 0.35, 0)
        
        # Green centerline on skeleton
        if skeleton is not None:
            blended[skeleton > 0] = [0, 255, 0]
            
        return blended
