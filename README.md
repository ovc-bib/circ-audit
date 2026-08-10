# CircAudit 🔬

**Circularity-Aware Audit Tool for Biomedical Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

CircAudit is an open-source framework for systematically detecting and quantifying **circularity** (knowledge leakage) in biomedical prediction tasks. It implements the four-type circularity taxonomy (C1–C4) and the seven-rule evaluation protocol described in our paper:

> **Circularity in Biomedical Prediction: An Empirical Audit Across Three Domains**
> Briefings in Bioinformatics, 2026.

---

## Why CircAudit?

High-performing biomedical prediction methods (AUROC > 0.9) often achieve their performance through **circularity** — features that encode knowledge already present in the evaluation labels or gold-standard databases. CircAudit helps you:

- 🕵️ **Detect** circularity in your prediction pipeline
- 📊 **Quantify** how much performance is genuine vs. circularly inflated
- 📈 **Visualize** circularity–performance trade-offs across methods
- 📝 **Report** audit findings in HTML or text format

---

## Installation

```bash
# From PyPI (recommended)
pip install circ-audit

# With optional dependencies
pip install circ-audit[xgboost,plotly]

# From source
git clone https://github.com/ovc-bib/circ-audit.git
cd circ-audit
pip install -e .
```

---

## Quick Start

### 1. Define Feature Metadata

```python
from circ_audit_tool import CircAudit

feature_metadata = {
    "crispr_score": {
        "source": "DepMap",
        "circ_type": None,           # Non-circular
        "exposure_score": 0.03,
    },
    "ppi_score": {
        "source": "STRING",
        "circ_type": "C2",           # Feature contamination
        "exposure_score": 0.52,
    },
    "lit_score": {
        "source": "PubMed",
        "circ_type": "C2",
        "exposure_score": 0.80,
    },
    "go_sim": {
        "source": "GO",
        "circ_type": "C4",           # Selection bias
        "exposure_score": 0.25,
    },
}
```

### 2. Create Auditor & Classify Features

```python
audit = CircAudit(
    task_name="SL Prediction",
    feature_metadata=feature_metadata,
    exposure_threshold=0.3,
)

# Classify features as circular vs. non-circular
classification = audit.classify_features()
print(classification["non_circular"])  # ['crispr_score']
print(classification["circular"])      # ['ppi_score', 'lit_score', 'go_sim']
```

### 3. Audit a Method (from summary statistics)

```python
# Audit from published metrics (no raw predictions needed)
report = audit.audit_from_summary(
    method_name="SINaTRA",
    auroc=0.957,
    circ_rho=0.561,
    c1=False, c2=True, c3=False, c4=True,
    evidence={"C2": "Uses STRING PPI features", "C4": "GO similarity bias"},
)
print(report.severity)  # 'severe'
```

### 4. Audit with Raw Predictions

```python
# Full audit with empirical circularity computation
report = audit.audit_method(
    method_name="CrisprSL",
    y_true=y_true_array,
    y_pred=y_pred_array,
    exposure_scores=exposure_array,
    y_pred_noncirc=y_pred_noncirc_array,  # non-circular baseline
)
```

### 5. PPI Ceiling Analysis

```python
# Check whether a method exceeds the PPI-only ceiling
ceiling = audit.ceiling_analysis(
    auroc_ppi_only=0.891,
    auroc_method=0.917,
)
print(ceiling["is_above_ceiling"])  # False
print(ceiling["interpretation"])    # "AUROC is within PPI ceiling"
```

### 6. Generate Report

```python
# Save and report results
audit.save_results("output_dir/")

from circ_audit_tool import ReportGenerator
reporter = ReportGenerator(audit)
reporter.generate_html_report(output_path="audit_report.html")
reporter.generate_text_report(output_path="audit_report.txt")
```

### 7. Visualize Results

```python
from circ_audit_tool import CircAuditVisualizer

viz = CircAuditVisualizer(output_dir="figures/")

# Core figure: Circularity vs AUROC scatter
viz.plot_circ_vs_auroc(summary_df, format="both")

# Feature exposure heatmap
viz.plot_feature_heatmap(feature_df, format="both")

# Cross-domain synthesis
viz.plot_cross_domain_synthesis(domain_results, format="both")
```

---

## Circularity Taxonomy

| Type | Name | Mechanism | Example |
|------|------|-----------|---------|
| C1 | Label Leakage | Features directly encode the target label | y = f(x) where x includes y |
| C2 | Feature Contamination | Features correlate with knowledge-base exposure | PPI score, literature count |
| C3 | Evaluation Bias | Test set distribution mirrors training due to KB overlap | Known pairs in both train/test |
| C4 | Selection Bias | Feature values correlate with KB coverage (degree) | GO similarity of well-studied genes |

---

## Seven-Rule Evaluation Protocol

CircAudit implements the CDRL (Circularity-Disentangled Reporting Label) — a 7-rule protocol for honest evaluation:

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Knowledge exposure computed | ρ(S, E_K) for each method |
| R2 | C1–C4 flags assigned | Per-method flagging |
| R3 | Non-circular baseline trained | Model on non-circular features only |
| R4 | Genuine fraction computed | (AUROC_all − AUROC_nc) / AUROC_all |
| R5 | PPI ceiling applied | AUROC ≤ AUROC_PPI_ceiling |
| R6 | Severity label assigned | clean / mild / moderate / severe |
| R7 | Nested CV or temporal split | Evaluation rigor check |

---

## API Reference

### `CircAudit`

Main auditor class. Key methods:

- `classify_features()` — Classify features as circular/non-circular
- `compute_circularity(predictions, exposure)` — Compute Circ(M, T, K) = ρ(S_M, E_K)
- `audit_method()` — Full audit of a single method with raw predictions
- `audit_from_summary()` — Audit from published metrics (no raw data needed)
- `ceiling_analysis()` — PPI ceiling analysis
- `generate_summary()` — DataFrame of all audited methods
- `save_results()` — Export to CSV + JSON

### `CircAuditVisualizer`

Publication-quality figures:

- `plot_circ_vs_auroc()` — Core Circ(ρ) vs AUROC scatter (Fig 3)
- `plot_feature_heatmap()` — Feature exposure heatmap (Fig 4)
- `plot_degree_stratified()` — Degree-stratified AUROC comparison
- `plot_cross_domain_synthesis()` — Multi-domain Circ vs AUROC (Fig 7)
- `plot_circ_type_radar()` — C1–C4 radar chart

### `ReportGenerator`

Formatted reports:

- `generate_html_report()` — Styled HTML report with tables, severity badges, and findings
- `generate_text_report()` — Plain text summary

---

## Framework Modules

The package includes a self-contained framework sub-package:

- `circ_audit_tool.framework.circularity` — Def 1–3, C1–C4 taxonomy, scorer
- `circ_audit_tool.framework.info_theory` — PPI Ceiling Theorem, MI ↔ AUROC conversion, marginal info gain
- `circ_audit_tool.framework.audit_algorithm` — 5-step audit pipeline, feature exposure, label independence

---

## Running Tests

```bash
# End-to-end test suite
python test_e2e.py

# Or with pytest
pytest test_e2e.py -v
```

---

## Citation

If you use CircAudit in your research, please cite:

```bibtex
@article{cui2026circularity,
  title={Circularity in Biomedical Prediction: An Empirical Audit Across Three Domains},
  author={Cui, Lei},
  journal={Briefings in Bioinformatics},
  year={2026},
  volume={27},
  doi={10.1093/bib/bbae492}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## Contact

For questions and feedback, please open a [GitHub Issue](https://github.com/ovc-bib/circ-audit/issues).
