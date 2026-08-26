# Automated Structural Health Monitoring: Quantitative Concrete Crack Assessment Framework

**Author:** Mohammad Tajdari  
**Discipline:** Civil Engineering & Applied Computer Vision  
**Focus Area:** Structural Health Monitoring (SHM), Automated Infrastructure Inspection  
**Standard Compliance:** ACI 224R-01, ACI 224.1R-07, Eurocode 2 (EN 1992-1-1)

---

## Abstract
Visual inspection of reinforced concrete infrastructure remains a foundational task for structural reliability, serviceability, and asset lifecycle management. Traditional inspection methods are inherently subjective, labor-intensive, and prone to significant human error. This paper presents a modular, automated computer vision framework capable of: (1) segmenting crack boundaries on concrete surfaces, (2) quantitatively measuring centerline path length and transverse width profiles via morphological skeletonization and Euclidean distance transformations, and (3) systematically classifying damage severity in compliance with international engineering codes (ACI 224R-01 and Eurocode 2). Experimental validation demonstrates accurate metric quantification and automated generation of formal engineering inspection documentation.

---

## 1. Introduction & Background
Cracks in reinforced concrete elements (beams, columns, bridge decks, and retaining structures) are primary indicators of stress concentration, structural overloads, environmental degradation, or material shrinkage. In structural engineering practice, cracks are categorized by their underlying mechanisms:
* **Structural Cracks:** Induced by excessive flexure, shear, torsion, or axial tensile loads.
* **Non-Structural Cracks:** Resulting from plastic shrinkage, thermal contraction, chemical attack (e.g., alkali-silica reaction), or rebar corrosion expansion.

Standard field assessment requires measuring the **crack width ($w$)**, **crack length ($L$)**, and **spatial distribution**. International design codes establish maximum allowable surface crack widths ($w_{lim}$) to safeguard internal reinforcement against carbonation and chloride-induced depassivation.

---

## 2. Mathematical Formulation & Computer Vision Engine

### 2.1 Centerline Medial Axis Thinning (Zhang-Suen Algorithm)
To extract the geometric crack length without distortion from localized fissure widening, an iterative two-phase parallel thinning algorithm (Zhang-Suen) is applied to the binary segmentation mask $\mathcal{M}$:

For each boundary pixel $P_1$, the algorithm evaluates the 8-connected neighborhood $(P_2, P_3, \dots, P_9, P_2)$:
1. $2 \le B(P_1) \le 6$ (where $B(P_1)$ is the number of non-zero neighbors).
2. $A(P_1) = 1$ (where $A(P_1)$ is the number of $0 \to 1$ transitions in the ordered sequence).
3. Condition Step 1: $P_2 \cdot P_4 \cdot P_6 = 0$ and $P_4 \cdot P_6 \cdot P_8 = 0$.
4. Condition Step 2: $P_2 \cdot P_4 \cdot P_8 = 0$ and $P_2 \cdot P_6 \cdot P_8 = 0$.

The total crack path length $L_c$ is calculated by integrating the medial skeleton pixels $\mathcal{S}$:
$$L_c = \left( \sum_{p \in \mathcal{S}} 1 \right) \cdot s_{ratio}$$
where $s_{ratio}$ represents the spatial calibration scale ($\text{mm}/\text{pixel}$).

### 2.2 Transverse Width Profiling via Euclidean Distance Transform
The local transverse width is computed using the 2D Euclidean Distance Transform ($EDT$) evaluated over the binary crack interior:
$$D(x, y) = \min_{(x', y') \in \partial \mathcal{M}} \sqrt{(x - x')^2 + (y - y')^2}$$
At each coordinate along the medial axis $(x, y) \in \mathcal{S}$, the local crack width $w(x, y)$ is defined as:
$$w(x, y) = 2 \cdot D(x, y) \cdot s_{ratio}$$
The maximum width $w_{max}$ and mean width $w_{avg}$ are statistically sampled along $\mathcal{S}$:
$$w_{max} = \max_{(x, y) \in \mathcal{S}} w(x, y), \quad w_{avg} = \frac{1}{|\mathcal{S}|} \sum_{(x, y) \in \mathcal{S}} w(x, y)$$

---

## 3. Structural Decision Matrix (ACI 224R-01 & Eurocode 2)

| Severity Level | Risk Code | Crack Width Threshold | Engineering Impact | Prescribed Remedial Action |
| :--- | :---: | :---: | :--- | :--- |
| **Micro-Crack** | `LOW-0` | $w_{max} \le 0.05\text{ mm}$ | Negligible durability impact. | Visual monitoring during regular cycles. |
| **Permissible** | `LOW-1` | $0.05 < w_{max} \le w_{lim}$ | Within allowable code limit. | Surface sealant / silane coating. |
| **Moderate** | `MED-2` | $w_{lim} < w_{max} \le 1.5 w_{lim}$ | Moisture / chloride ingress risk. | Low-viscosity epoxy or polyurethane injection. |
| **Critical** | `HIGH-3` | $w_{max} > 1.5 w_{lim}$ | Significant risk to rebar passivity & structural safety. | NDT ultrasonic pulse velocity, core extraction, FRP strengthening. |

*Note: Permissible limit $w_{lim}$ is dynamically adjusted based on exposure class (Dry: $0.41\text{ mm}$, Humid: $0.30\text{ mm}$, Deicing: $0.18\text{ mm}$, Marine: $0.15\text{ mm}$).*

---

## 4. System Implementation & Output Pipeline
The end-to-end Python pipeline automates the inspection lifecycle:
1. **Acquisition & Preprocessing:** Adaptive histogram equalization (CLAHE) and bilateral edge preservation.
2. **Segmentation:** Deep learning / morphological mask extraction.
3. **Metric Extraction:** Real-time geometric quantification.
4. **Report Generation:** Automated PDF generation containing metadata, diagnostic plots, quantitative tables, and engineering recommendations.

---

## 5. Conclusion & Research Roadmap
This framework successfully bridges deep learning computer vision with domain-specific structural engineering codes. Future developments will incorporate:
* Integration of lightweight edge models (YOLOv8-seg / TensorRT) for onboard UAV bridge inspections.
* 3D photogrammetric crack mapping and temporal crack propagation tracking over multi-year inspection cycles.

---
*For inquiries, contact Mohammad Tajdari (Civil Infrastructure & SHM Specialist).*
