# Changelog

All notable changes to CircAudit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-29

### Added
- `circ_audit_tool.temporal` — the temporal-evaluation instruments of the
  manuscript revision: evidence dating pipeline (`first_evidence_year`,
  `cutoff_split`), snapshot rebuilders for archived database releases
  (`rank_normalised_degree`, `personalised_pagerank`, `bfs_proximity`,
  `build_snapshot_features`, `pair_transform`) and the vintage rank-stability
  diagnostic with absorption (`auroc_pairwise`, `rank_stability`, `absorption`)
- `data/` — dated gene and pair gold standards, negative pools, per-gene and
  per-pair scores, and the result tables behind every manuscript figure and
  table (36 CSV files)
- `analysis/` — the t1-t12, a/b/c analysis scripts, figure generation and the
  manuscript number cross-check, relocatable via `CIRCAUDIT_HOME`
- `test_temporal.py` — 17 end-to-end checks of the temporal instruments

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
- Seven-rule evaluation protocol (CDRL; updated to the temporal-reporting edition with the v11 manuscript)
- End-to-end test suite (5 test categories)
