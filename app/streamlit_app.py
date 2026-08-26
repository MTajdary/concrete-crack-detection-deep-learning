"""
Interactive Web Application for Structural Crack Inspection (Streamlit)
Allows real-time image upload, parameter tuning, diagnostic overlay, and instant PDF report download.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.detector import CrackDetector
from core.measurement import CrackMeasurer
from core.structural_rules import StructuralRulesEngine
from report.generator import InspectionReportGenerator

st.set_page_config(page_title="Concrete Crack AI Inspector", page_icon="🏗️", layout="wide")

st.title("🏗️ AI-Based Quantitative Concrete Crack Inspection")
st.markdown("Automated Structural Health Monitoring (SHM) conforming to **ACI 224R-01** and **Eurocode 2** standards.")

# Sidebar Settings
st.sidebar.header("⚙️ Inspection Parameters")
pixel_ratio = st.sidebar.slider("Calibration Ratio (mm/pixel)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
exposure = st.sidebar.selectbox(
    "Environmental Exposure Class (ACI 224R Table 4.1)",
    ["humid_air", "dry_air", "deicing_chemicals", "seawater", "water_retaining"],
    index=0
)
element_id = st.sidebar.text_input("Structural Element ID", value="RC-Beam-B04")

# File Uploader
uploaded_file = st.file_uploader("Upload Concrete Surface Inspection Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    rgb_img = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(rgb_img, use_column_width=True)

    with st.spinner("Analyzing concrete fissure geometry and assessing structural risk..."):
        # 1. Detection & Segmentation
        detector = CrackDetector()
        binary_mask = detector.segment(rgb_img)

        # 2. Measurement
        measurer = CrackMeasurer(pixel_to_mm_ratio=pixel_ratio)
        metrics = measurer.measure(binary_mask)

        # 3. Structural Rules Evaluation
        evaluation = StructuralRulesEngine.evaluate(
            max_width_mm=metrics["max_width_mm"],
            avg_width_mm=metrics["avg_width_mm"],
            length_mm=metrics["length_mm"],
            exposure=exposure
        )

        # 4. Diagnostic Overlay
        overlay_img = detector.create_overlay(rgb_img, binary_mask, metrics["skeleton_map"])

    with col2:
        st.subheader("🔍 Diagnostic Overlay (Mask & Skeleton)")
        st.image(overlay_img, use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Quantitative Structural Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Crack Length ($L_c$)", f"{metrics['length_mm']} mm")
    m2.metric("Max Width ($w_{max}$)", f"{metrics['max_width_mm']} mm")
    m3.metric("Avg Width ($w_{avg}$)", f"{metrics['avg_width_mm']} mm")
    m4.metric("Damaged Area ($A_d$)", f"{metrics['area_mm2']} mm²")

    st.subheader("🛡️ Engineering Risk & Remedial Recommendations")
    sev = evaluation.severity_level
    if "Critical" in sev:
        st.error(f"**Severity Level:** {sev} | **Risk Code:** {evaluation.risk_code}")
    elif "Moderate" in sev:
        st.warning(f"**Severity Level:** {sev} | **Risk Code:** {evaluation.risk_code}")
    else:
        st.success(f"**Severity Level:** {sev} | **Risk Code:** {evaluation.risk_code}")

    st.info(f"**Standard Reference:** {evaluation.standard_reference}\n\n"
            f"**Serviceability Impact:** {evaluation.serviceability_impact}\n\n"
            f"**Recommended Action:** {evaluation.recommended_action}")

    # 5. Generate and Download PDF Report
    os.makedirs("output", exist_ok=True)
    plot_path = "output/temp_overlay.png"
    cv2.imwrite(plot_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    
    pdf_path = "output/Inspection_Report.pdf"
    generator = InspectionReportGenerator(output_path=pdf_path)
    metadata = {
        "project_name": "Civil Infrastructure Inspection",
        "element_id": element_id,
        "exposure": exposure,
        "scale": pixel_ratio,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    eval_dict = {
        "severity_level": evaluation.severity_level,
        "risk_code": evaluation.risk_code,
        "standard_reference": evaluation.standard_reference,
        "serviceability_impact": evaluation.serviceability_impact,
        "recommended_action": evaluation.recommended_action
    }
    generator.generate(metadata, metrics, eval_dict, plot_path)

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Official Structural Inspection Report (PDF)",
            data=pdf_file,
            file_name=f"Inspection_Report_{element_id}.pdf",
            mime="application/pdf"
        )
