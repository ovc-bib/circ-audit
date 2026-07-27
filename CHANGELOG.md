# Changelog

All notable changes to CircAudit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-27

### Added
- `CircAudit` — main auditor class with full audit pipeline
- `CircularityScorer` — computes Circ(M, T, K) = ρ(S_M, E_K) using Spearman correlation
- `CircularityInflation` — computes performance inflation (Δ AUROC between all-features and non-circular baseline)
- `CircularityReport` — structured report with severity classification (clean/mild/moderate/severe)
- `PPICeilingAnalyzer` — PPI Ceiling Theorem implementation (MI ↔ AUROC conversion)
- `MarginalInfoGainAnalyzer` — marginal information gain for feature source decomposition
- `FeatureInfoDecomposer` — decomposes prediction info into circular and genuine components
- `C1–C4` circularity taxonomy: Label Leakage, Feature Contamination, Evaluation Bias, Selection Bias
- `audit_from_summary()` — audit methods from published metrics (no raw predictions needed)
- `audit_method()` — full audit with empirical circularity computation
- `ceiling_analysis()` — PPI ceiling analysis
- `CircAuditVisualizer` — publication-quality figures (scatter, heatmap, degree-stratified, cross-domain, radar)
- `ReportGenerator` — HTML and text report generation
- Seven-rule evaluation protocol (CDRL)
- End-to-end test suite (5 test categories)
