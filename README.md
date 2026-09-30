# CircAudit 🔬

**Temporal-evaluation and circularity-audit instruments for database-driven predictors**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

CircAudit is the released instrument of the manuscript *"A temporal
evaluation of feature-side inflation in synthetic lethality prediction"*
(BIB-26-2126, under revision). It ships the evidence-dating pipeline, the
snapshot rebuilders for archived database releases, the vintage
rank-stability diagnostic and the graduated seven-rule reporting protocol,
together with a circularity-audit framework (feature exposure, taxonomy
scoring, ceiling analysis) for database-driven predictors.

---

## Why CircAudit?

Temporal label splits cannot detect leakage that arrives through features:
features computed from today's databases already carry much of the
information that will define tomorrow's gold standard. CircAudit lets you

- 📅 **date** every gold entry by the publication year of its first
  experimental evidence,
- 🕰️ **rebuild** feature channels from archived database snapshots with
  identical code, so vintages are comparable,
- 📊 **diagnose** whether a knowledge channel reorders gene prominence
  between vintages (the quantity that tracks feature-side inflation),
- 🕵️ **audit** circularity (knowledge exposure, taxonomy flags, ceiling
  analysis) when raw predictions are available.

---

## Installation

```bash
git clone https://github.com/ovc-bib/circ-audit.git
cd circ-audit
pip install -e .

# with optional dependencies
pip install -e ".[xgboost,plotly]"
```

---

## Quick Start: temporal evaluation

```python
from circ_audit_tool.temporal import (
    first_evidence_year, cutoff_split,           # dating, Eq (1)
    build_snapshot_features, rank_stability,     # snapshots, Eq (3)-(5), (8)
    auroc_pairwise, absorption,                  # evaluation, Eq (7), (9)
)

# 1. date every gold gene by its first experimental evidence
years = first_evidence_year(evidence_df)          # gene, year, source
split = cutoff_split(years, cutoff=2016)          # train <= 2016 < test

# 2. rebuild the three network features on any archived release
#    with identical code (degree rank, PageRank, BFS proximity)
t2016 = build_snapshot_features(hetionet_edges, candidates, hr_seeds)
t2021 = build_snapshot_features(string_v11_edges, candidates, hr_seeds)

# 3. the diagnostic: how much does the channel reorder gene prominence
#    between vintages? (low rho => the channel absorbs future discoveries)
rho = rank_stability(t2016["degree_rank"], t2021["degree_rank"])

# 4. feature-side absorption with protocol, labels, learner and test
#    rows fixed, only the snapshot moved
delta = absorption(auroc_pairwise(y, s_v2016), auroc_pairwise(y, s_v2021))
```

| Instrument | Function | Manuscript |
|---|---|---|
| Evidence dating | `first_evidence_year(evidence)` | Eq (1) |
| Cutoff split | `cutoff_split(years, cutoff)` | train <= T < test |
| Rank-normalised degree | `rank_normalised_degree(edges, nodes)` | Eq (3) |
| Personalised PageRank | `personalised_pagerank(edges, nodes, seeds)` | Eq (4), alpha = 0.15 |
| BFS proximity | `bfs_proximity(edges, nodes, seeds)` | Eq (5) |
| All three features | `build_snapshot_features(edges, nodes, seeds)` | Feature vintages |
| Pair transforms | `pair_transform(x_u, x_v)` | Eq (6) |
| Pairwise AUROC | `auroc_pairwise(y_true, scores)` | Eq (7) |
| Rank stability | `rank_stability(f_a, f_b)` | Eq (8) |
| Feature-side absorption | `absorption(A_a, A_b)` | Eq (9) |

---

## Seven-Rule Reporting Protocol

CircAudit ships a graduated seven-rule reporting protocol for honest
evaluation of database-driven predictors, rated from problematic (0-1
rules satisfied) through concerning (2-3), acceptable (4) and good (5) to
excellent (6-7):

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Positive definition stated | Evidence sources listed for the gold standard |
| R2 | Positives dated | First experimental evidence year per positive |
| R3 | Labels restricted to a stated cutoff | Training uses only knowledge available at T |
| R4 | Feature vintages stated | Vintage of every knowledge-derived feature column |
| R5 | Rank stability reported | Spearman rho between each channel's vintage and the present |
| R6 | Non-knowledge floor included | At least one non-knowledge feature family evaluated |
| R7 | Shortlist yield reported | Hits at a fixed depth against the random expectation |

---

## Revision Data and Analysis Scripts

- `data/` — the evidence-dated gene and pair gold standards, the negative
  pools, per-gene and per-pair scores, and the result table behind every
  figure and table of the manuscript (36 CSV files + 1 JSON).
- `analysis/` — the exact scripts that produced them (t1-t12, a/b/c series,
  figure generation and the number cross-check). They reference the original
  project layout through `tm_utils.V11`; set the `CIRCAUDIT_HOME`
  environment variable to relocate, and see the Data and code availability
  statement of the manuscript for the raw database releases the pipeline
  consumes.

---

## Circularity-Audit Framework

The package also includes the audit layer used for exposure and
selection-bias reporting, with a four-type circularity taxonomy:

| Type | Name | Mechanism | Example |
|------|------|-----------|---------|
| C1 | Label Leakage | Features directly encode the target label | y = f(x) where x includes y |
| C2 | Feature Contamination | Features correlate with knowledge-base exposure | PPI score, literature count |
| C3 | Evaluation Bias | Test set distribution mirrors training due to KB overlap | Known pairs in both train/test |
| C4 | Selection Bias | Feature values correlate with KB coverage (degree) | GO similarity of well-studied genes |

```python
from circ_audit_tool import CircAudit, ReportGenerator

audit = CircAudit(
    task_name="SL prediction",
    feature_metadata={
        "crispr_score": {"source": "DepMap", "circ_type": None,
                          "exposure_score": 0.03},
        "ppi_score": {"source": "STRING", "circ_type": "C2",
                      "exposure_score": 0.52},
    },
    exposure_threshold=0.3,
)

classification = audit.classify_features()
report = audit.audit_from_summary(
    method_name="example", auroc=0.93, circ_rho=0.55,
    c1=False, c2=True, c3=False, c4=True,
    evidence={"C2": "PPI features"},
)
ceiling = audit.ceiling_analysis(auroc_ppi_only=0.89, auroc_method=0.92)

audit.save_results("output_dir/")
ReportGenerator(audit).generate_html_report("audit_report.html")
```

Visualisation of audit results (circularity-vs-AUROC scatter, exposure
heatmap, degree-stratified comparison, taxonomy radar) is available through
`CircAuditVisualizer`; see `circ_audit_tool/visualize.py`.

### Framework modules

- `circ_audit_tool.temporal` — dating pipeline, snapshot rebuilders,
  rank-stability and absorption diagnostics (the manuscript instruments)
- `circ_audit_tool.framework.circularity` — definitions, C1-C4 taxonomy, scorer
- `circ_audit_tool.framework.info_theory` — ceiling theorem, MI-AUROC conversion
- `circ_audit_tool.framework.audit_algorithm` — audit pipeline, feature
  exposure, label-independence checks

---

## Running Tests

```bash
python test_e2e.py        # circularity-audit suite
python test_temporal.py   # temporal-instrument suite (17 checks)
```

---

## Citation

The instrument accompanies *"A temporal evaluation of feature-side
inflation in synthetic lethality prediction"* (BIB-26-2126, under
revision). A citation entry will be added on acceptance.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For
major changes, please open an issue first to discuss what you would like
to change.

---

## Contact

For questions and feedback, please open a [GitHub Issue](https://github.com/ovc-bib/circ-audit/issues).
