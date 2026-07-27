"""
report.py — CircAudit Report Generator
========================================
Generate HTML and text reports from audit results.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional

try:
    from .auditor import CircAudit
except ImportError:
    CircAudit = None


class ReportGenerator:
    """Generate formatted reports from CircAudit results."""

    def __init__(self, audit: Optional[CircAudit] = None):
        self.audit = audit

    def generate_html_report(
        self,
        summary_df: Optional[pd.DataFrame] = None,
        output_path: str = "circ_audit_report.html",
        task_name: str = "Biomedical Prediction",
    ) -> str:
        """
        Generate an HTML report with audit results.
        """
        if summary_df is None and self.audit is not None:
            summary_df = self.audit.generate_summary()
        if summary_df is None:
            raise ValueError("No audit data available")

        html = self._html_header(task_name)
        html += self._html_summary_table(summary_df)
        html += self._html_severity_breakdown(summary_df)
        html += self._html_findings(summary_df)
        html += self._html_footer()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Report saved to {output_path}")
        return output_path

    def generate_text_report(
        self,
        summary_df: Optional[pd.DataFrame] = None,
        output_path: str = "circ_audit_report.txt",
        task_name: str = "Biomedical Prediction",
    ) -> str:
        """Generate a plain text report."""
        if summary_df is None and self.audit is not None:
            summary_df = self.audit.generate_summary()
        if summary_df is None:
            raise ValueError("No audit data available")

        lines = []
        lines.append("=" * 70)
        lines.append(f"CircAudit Report — {task_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Summary table
        lines.append("METHOD SUMMARY")
        lines.append("-" * 70)
        for _, row in summary_df.iterrows():
            lines.append(
                f"  {row['method']:20s}  AUROC={row['auroc']:.3f}  "
                f"Circ={row.get('circ_rho', float('nan')):.3f}  "
                f"Severity={row.get('severity', 'N/A')}"
            )
        lines.append("")

        # Severity breakdown
        lines.append("SEVERITY BREAKDOWN")
        lines.append("-" * 70)
        for sev in ["clean", "mild", "moderate", "severe"]:
            count = len(summary_df[summary_df.get("severity", "") == sev])
            methods = summary_df[summary_df.get("severity", "") == sev]["method"].tolist()
            lines.append(f"  {sev.capitalize():10s}: {count} method(s) — {', '.join(methods)}")
        lines.append("")

        # Key findings
        lines.append("KEY FINDINGS")
        lines.append("-" * 70)
        valid = summary_df.dropna(subset=["circ_rho"])
        if len(valid) > 0:
            # Correlation between circ and performance
            from scipy.stats import spearmanr
            rho, p = spearmanr(valid["circ_rho"], valid["auroc"])
            lines.append(f"  Circularity-Performance correlation: ρ={rho:.3f} (p={p:.4f})")

            # Clean vs circular performance gap
            clean = summary_df[summary_df.get("severity", "") == "clean"]["auroc"]
            circular = summary_df[summary_df.get("severity", "").isin(["moderate", "severe"])]["auroc"]
            if len(clean) > 0 and len(circular) > 0:
                lines.append(f"  Clean methods mean AUROC:    {clean.mean():.3f}")
                lines.append(f"  Circular methods mean AUROC: {circular.mean():.3f}")
                lines.append(f"  Performance gap: {circular.mean() - clean.mean():.3f}")

        text = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Text report saved to {output_path}")
        return output_path

    # =========================================================================
    # HTML Components
    # =========================================================================

    def _html_header(self, task_name: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CircAudit Report — {task_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #fafafa; }}
  h1 {{ color: #1565c0; border-bottom: 2px solid #1565c0; padding-bottom: 10px; }}
  h2 {{ color: #333; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #1565c0; color: white; font-weight: 600; }}
  tr:hover {{ background: #e3f2fd; }}
  .clean {{ color: #388e3c; font-weight: bold; }}
  .mild {{ color: #fbc02d; font-weight: bold; }}
  .moderate {{ color: #f57c00; font-weight: bold; }}
  .severe {{ color: #d32f2f; font-weight: bold; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
  .badge-clean {{ background: #e8f5e9; color: #388e3c; }}
  .badge-mild {{ background: #fff9c4; color: #f9a825; }}
  .badge-moderate {{ background: #fff3e0; color: #f57c00; }}
  .badge-severe {{ background: #ffebee; color: #d32f2f; }}
  .finding {{ background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #1565c0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 0 4px 4px 0; }}
</style>
</head>
<body>
<h1>🔬 CircAudit Report</h1>
<p><strong>Task:</strong> {task_name} |
<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    def _html_summary_table(self, df: pd.DataFrame) -> str:
        html = "<h2>Method Summary</h2><table><tr>"
        headers = ["Method", "AUROC", "Circ (ρ)", "C1", "C2", "C3", "C4", "Severity"]
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr>"

        for _, row in df.iterrows():
            sev = row.get("severity", "unknown")
            html += "<tr>"
            html += f"<td><strong>{row['method']}</strong></td>"
            html += f"<td>{row['auroc']:.3f}</td>"

            circ_rho = row.get("circ_rho", float("nan"))
            html += f"<td>{circ_rho:.3f}</td>" if not pd.isna(circ_rho) else "<td>—</td>"

            for flag in ["C1_label_leakage", "C2_feature_contamination",
                         "C3_evaluation_bias", "C4_selection_bias"]:
                val = row.get(flag, False)
                html += f"<td>{'✓' if val else '—'}</td>"

            badge_class = f"badge-{sev}"
            html += f'<td><span class="badge {badge_class}">{sev.upper()}</span></td>'
            html += "</tr>"

        html += "</table>"
        return html

    def _html_severity_breakdown(self, df: pd.DataFrame) -> str:
        html = "<h2>Severity Breakdown</h2>"
        for sev in ["clean", "mild", "moderate", "severe"]:
            group = df[df.get("severity", "") == sev]
            if len(group) == 0:
                continue
            methods = ", ".join(group["method"].tolist())
            html += f"""
            <div class="finding">
              <span class="badge badge-{sev}">{sev.upper()}</span>
              ({len(group)} method{'' if len(group)==1 else 's'})
              <br><strong>Methods:</strong> {methods}
            </div>"""
        return html

    def _html_findings(self, df: pd.DataFrame) -> str:
        html = "<h2>Key Findings</h2>"
        valid = df.dropna(subset=["circ_rho"])

        if len(valid) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(valid["circ_rho"], valid["auroc"])
            html += f"""
            <div class="finding">
              <strong>Circularity-Performance Correlation</strong><br>
              Spearman ρ = {rho:.3f} (p = {p:.4f})<br>
              <em>{'Significant' if p < 0.05 else 'Not significant'}:
              Higher circularity {'is' if rho > 0 else 'is not'} associated with higher AUROC</em>
            </div>"""

        # Clean vs circular
        clean = df[df.get("severity", "") == "clean"]
        circular = df[df.get("severity", "").isin(["moderate", "severe"])]
        if len(clean) > 0 and len(circular) > 0:
            html += f"""
            <div class="finding">
              <strong>Clean vs Circular Methods</strong><br>
              Clean methods mean AUROC: {clean['auroc'].mean():.3f}<br>
              Circular methods mean AUROC: {circular['auroc'].mean():.3f}<br>
              <em>Gap of {circular['auroc'].mean() - clean['auroc'].mean():.3f} may be
              attributable to circularity inflation</em>
            </div>"""

        return html

    def _html_footer(self) -> str:
        return """
<hr>
<p style="color: #999; font-size: 0.85em;">
  Generated by <strong>CircAudit</strong> —
  Circularity-Aware Biomedical Prediction Framework
</p>
</body></html>"""
