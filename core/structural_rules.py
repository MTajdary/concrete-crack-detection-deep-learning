"""
Structural Rules Engine for Crack Assessment
Based on ACI 224R-01 (Control of Cracking in Concrete Structures) and Eurocode 2 (EN 1992-1-1).
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvaluationResult:
    severity_level: str
    risk_code: str
    recommended_action: str
    standard_reference: str
    serviceability_impact: str

class StructuralRulesEngine:
    """
    Evaluates detected crack metrics against international civil engineering standards.
    """
    
    # ACI 224R-01 Table 4.1: Tolerable crack widths for reinforced concrete
    ACI_LIMITS = {
        "dry_air": 0.41,          # mm (Dry air or protective membrane)
        "humid_air": 0.30,        # mm (Humidity, moist air, soil)
        "deicing_chemicals": 0.18,# mm (Deicing chemicals)
        "seawater": 0.15,         # mm (Seawater and seawater spray; wetting and drying)
        "water_retaining": 0.10   # mm (Water-retaining structures)
    }

    @classmethod
    def evaluate(cls, max_width_mm: float, avg_width_mm: float, length_mm: float, exposure: str = "humid_air") -> EvaluationResult:
        """
        Evaluate structural crack condition.
        :param max_width_mm: Maximum measured crack width in mm
        :param avg_width_mm: Average crack width in mm
        :param length_mm: Total crack length in mm
        :param exposure: Environmental exposure condition
        """
        tolerable_limit = cls.ACI_LIMITS.get(exposure, 0.30)
        
        if max_width_mm <= 0.05:
            return EvaluationResult(
                severity_level="Negligible / Micro-crack",
                risk_code="LOW-0",
                recommended_action="Periodic visual monitoring. No structural intervention required.",
                standard_reference="ACI 224R-01 (Micro-cracking)",
                serviceability_impact="No impact on durability or load-bearing capacity."
            )
        elif max_width_mm <= tolerable_limit:
            return EvaluationResult(
                severity_level="Minor (Within Permissible Limit)",
                risk_code="LOW-1",
                recommended_action=f"Routine maintenance and surface sealing to prevent moisture ingress. Limit for '{exposure}' is {tolerable_limit} mm.",
                standard_reference=f"ACI 224R-01 Table 4.1 ({exposure})",
                serviceability_impact="Low risk under current environmental exposure. Monitor during scheduled inspections."
            )
        elif max_width_mm <= tolerable_limit * 1.5:
            return EvaluationResult(
                severity_level="Moderate (Exceeds Limit)",
                risk_code="MED-2",
                recommended_action="Epoxy or polyurethane resin injection recommended to prevent rebar corrosion and freeze-thaw degradation.",
                standard_reference="ACI 224.1R-07 (Causes, Evaluation, and Repair of Cracks)",
                serviceability_impact="Increased permeability. Rebar passivity layer at risk of carbonation/chloride attack."
            )
        else:
            return EvaluationResult(
                severity_level="Critical / Structural Concern",
                risk_code="HIGH-3",
                recommended_action="Immediate non-destructive testing (NDT/Ultrasonic), core testing, and structural engineering appraisal for potential strengthening (FRP/jacketing).",
                standard_reference="Eurocode 2 / ACI 318 / ACI 224.1R",
                serviceability_impact="High risk to structural durability, integrity, and shear/flexural serviceability."
            )
