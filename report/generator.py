"""
Structural Inspection Report Generator (PDF)
Generates industry-grade engineering evaluation reports using ReportLab.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class InspectionReportGenerator:
    def __init__(self, output_path: str = "report_output.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=6
        )
        self.subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=12
        )
        self.section_heading = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=10,
            spaceAfter=6
        )
        self.body_style = ParagraphStyle(
            'ReportBody',
            parent=self.styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#2D3748")
        )
        self.bold_body = ParagraphStyle(
            'BoldBody',
            parent=self.body_style,
            fontName='Helvetica-Bold'
        )

    def generate(self, 
                 metadata: dict, 
                 metrics: dict, 
                 evaluation: dict, 
                 overlay_image_path: str):
        """
        Builds the PDF document.
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=36, leftMargin=36,
            topMargin=36, bottomMargin=36
        )
        story = []

        # Header Title
        story.append(Paragraph("AI-BASED STRUCTURAL CRACK INSPECTION REPORT", self.title_style))
        story.append(Paragraph("Automated Computer Vision Quantitative Assessment & Risk Analysis", self.subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=14))

        # Metadata Table
        meta_data = [
            [Paragraph("<b>Project:</b>", self.body_style), Paragraph(metadata.get("project_name", "Infrastructure Inspection"), self.body_style),
             Paragraph("<b>Date/Time:</b>", self.body_style), Paragraph(metadata.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")), self.body_style)],
            [Paragraph("<b>Structure Element:</b>", self.body_style), Paragraph(metadata.get("element_id", "RC Beam - B04"), self.body_style),
             Paragraph("<b>Exposure Class:</b>", self.body_style), Paragraph(metadata.get("exposure", "Humid Air / Soil"), self.body_style)],
            [Paragraph("<b>Inspector / Firm:</b>", self.body_style), Paragraph(metadata.get("inspector", "Automated AI Vision System"), self.body_style),
             Paragraph("<b>Scale Ratio:</b>", self.body_style), Paragraph(f"{metadata.get('scale', 0.2)} mm/px", self.body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))

        # Section 1: Visual Inspection Overlay
        story.append(Paragraph("1. Visual Diagnostic Overlay", self.section_heading))
        if os.path.exists(overlay_image_path):
            story.append(RLImage(overlay_image_path, width=540, height=220))
        story.append(Spacer(1, 10))

        # Section 2: Quantitative Geometric Metrics
        story.append(Paragraph("2. Geometric Quantification (OpenCV Engine)", self.section_heading))
        metric_data = [
            [Paragraph("<b>Metric Parameter</b>", self.bold_body), Paragraph("<b>Value</b>", self.bold_body), Paragraph("<b>Engineering Significance</b>", self.bold_body)],
            [Paragraph("Total Crack Length ($L_c$)", self.body_style), Paragraph(f"<b>{metrics.get('length_mm', 0)} mm</b>", self.body_style), Paragraph("Centerline medial axis tracking", self.body_style)],
            [Paragraph("Maximum Crack Width ($w_{max}$)", self.body_style), Paragraph(f"<b>{metrics.get('max_width_mm', 0)} mm</b>", self.body_style), Paragraph("Critical ACI 224R / Eurocode 2 metric", self.body_style)],
            [Paragraph("Average Crack Width ($w_{avg}$)", self.body_style), Paragraph(f"<b>{metrics.get('avg_width_mm', 0)} mm</b>", self.body_style), Paragraph("Permeability profile estimation", self.body_style)],
            [Paragraph("Total Damaged Area", self.body_style), Paragraph(f"<b>{metrics.get('area_mm2', 0)} mm²</b>", self.body_style), Paragraph("Surface spalling and damage extent", self.body_style)],
        ]
        metric_table = Table(metric_data, colWidths=[170, 100, 270])
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(metric_table)
        story.append(Spacer(1, 12))

        # Section 3: Engineering Risk & Remedial Recommendations
        story.append(Paragraph("3. Structural Risk Assessment & Action Plan", self.section_heading))
        
        # Color code severity
        sev = evaluation.get("severity_level", "")
        sev_color = colors.HexColor("#38A169") if "Minor" in sev or "Negligible" in sev else (colors.HexColor("#DD6B20") if "Moderate" in sev else colors.HexColor("#E53E3E"))

        eval_data = [
            [Paragraph("<b>Severity Classification:</b>", self.bold_body), Paragraph(f"<font color='{sev_color.hexval()}'><b>{sev}</b></font>", self.body_style)],
            [Paragraph("<b>Risk Code:</b>", self.bold_body), Paragraph(evaluation.get("risk_code", "N/A"), self.body_style)],
            [Paragraph("<b>Governing Standard:</b>", self.bold_body), Paragraph(evaluation.get("standard_reference", "ACI 224R-01"), self.body_style)],
            [Paragraph("<b>Serviceability Impact:</b>", self.bold_body), Paragraph(evaluation.get("serviceability_impact", "None"), self.body_style)],
            [Paragraph("<b>Recommended Action:</b>", self.bold_body), Paragraph(f"<b>{evaluation.get('recommended_action', 'N/A')}</b>", self.body_style)],
        ]
        eval_table = Table(eval_data, colWidths=[150, 390])
        eval_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
            ('BOX', (0, 0), (-1, -1), 1.0, sev_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(eval_table)

        doc.build(story)
        return self.output_path
