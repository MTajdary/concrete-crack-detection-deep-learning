"""
Concrete Crack AI Inspection - CLI & Execution Pipeline
Performs end-to-end inspection: Image -> Segmentation -> Geometric Metrics -> Structural Rules -> Inspection PDF Report.
"""

import os
import sys
from pathlib import Path

# Add project root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import matplotlib.pyplot as plt
from datetime import datetime

from core.detector import CrackDetector
from core.measurement import CrackMeasurer
from core.structural_rules import StructuralRulesEngine
from report.generator import InspectionReportGenerator

def run_inspection(image_path: str, 
                   output_dir: str = "output", 
                   pixel_to_mm_ratio: float = 0.25, 
                   exposure: str = "humid_air",
                   element_id: str = "RC-Girder-G12"):
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Loading inspection image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"[!] Error: File not found at {image_path}")
        return

    # 1. Read input image (RGB)
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 2. Crack Detection & Segmentation
    detector = CrackDetector()
    binary_mask = detector.segment(rgb_img)

    # 3. Geometric Measurement
    measurer = CrackMeasurer(pixel_to_mm_ratio=pixel_to_mm_ratio)
    metrics = measurer.measure(binary_mask)

    print("\n--- Geometric Quantification Results ---")
    print(f"Crack Detected:   {metrics['detected']}")
    print(f"Crack Length:     {metrics['length_mm']} mm")
    print(f"Max Crack Width:  {metrics['max_width_mm']} mm")
    print(f"Avg Crack Width:  {metrics['avg_width_mm']} mm")
    print(f"Damaged Area:     {metrics['area_mm2']} mm²")

    # 4. Structural Risk Evaluation (ACI 224R / Eurocode 2)
    evaluation = StructuralRulesEngine.evaluate(
        max_width_mm=metrics["max_width_mm"],
        avg_width_mm=metrics["avg_width_mm"],
        length_mm=metrics["length_mm"],
        exposure=exposure
    )

    print("\n--- Structural Risk Evaluation ---")
    print(f"Severity Level:   {evaluation.severity_level}")
    print(f"Risk Code:        {evaluation.risk_code}")
    print(f"Standard:         {evaluation.standard_reference}")
    print(f"Action:           {evaluation.recommended_action}")

    # 5. Generate Visual Diagnostic Diagnostic Plot
    overlay_img = detector.create_overlay(rgb_img, binary_mask, metrics["skeleton_map"])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Inspection Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title("Binary Segmentation Mask", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title(f"Overlay (Red: Mask, Green: Skeleton)\nMax Width: {metrics['max_width_mm']} mm", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "diagnostic_overlay.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Diagnostic plot saved: {plot_path}")

    # 6. Generate Structural Inspection PDF Report
    pdf_path = os.path.join(output_dir, "Structural_Inspection_Report.pdf")
    generator = InspectionReportGenerator(output_path=pdf_path)
    
    metadata = {
        "project_name": "Civil Infrastructure Health Monitoring",
        "element_id": element_id,
        "exposure": exposure,
        "scale": pixel_to_mm_ratio,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    eval_dict = {
        "severity_level": evaluation.severity_level,
        "risk_code": evaluation.risk_code,
        "standard_reference": evaluation.standard_reference,
        "serviceability_impact": evaluation.serviceability_impact,
        "recommended_action": evaluation.recommended_action
    }

    generator.generate(
        metadata=metadata,
        metrics=metrics,
        evaluation=eval_dict,
        overlay_image_path=plot_path
    )
    print(f"[+] Formal Inspection PDF Report generated: {pdf_path}")
    print("\n[✓] Pipeline execution finished successfully.")

if __name__ == "__main__":
    sample_img = "data/samples/sample_concrete_crack.jpg"
    run_inspection(sample_img, output_dir="output", pixel_to_mm_ratio=0.25)
