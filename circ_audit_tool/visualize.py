"""
visualize.py — CircAudit Visualization
========================================
Generate publication-quality figures for circularity audit results.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple

# Try plotly for interactive, fallback to matplotlib
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class CircAuditVisualizer:
    """Generate visualizations for circularity audit results."""

    # Color palette (consistent, accessible)
    COLORS = {
        "severe": "#d32f2f",     # Red
        "moderate": "#f57c00",   # Orange
        "mild": "#fbc02d",       # Yellow
        "clean": "#388e3c",      # Green
        "highlight": "#1565c0",  # Blue
    }

    SEVERITY_ORDER = ["clean", "mild", "moderate", "severe"]

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Figure 3: Circularity vs AUROC Scatter Plot (THE KEY FIGURE)
    # =========================================================================

    def plot_circ_vs_auroc(
        self,
        methods_df: pd.DataFrame,
        title: str = "Circularity vs Performance in SL Prediction",
        output_name: str = "fig3_circ_vs_auroc",
        format: str = "both",
    ) -> str:
        """
        Core figure: Circ(ρ) vs AUROC scatter plot.

        Parameters
        ----------
        methods_df : DataFrame with columns: method, auroc, circ_rho, severity
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping plot")
            return ""

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot each severity group
        for severity in self.SEVERITY_ORDER:
            group = methods_df[methods_df["severity"] == severity]
            if len(group) == 0:
                continue

            color = self.COLORS[severity]
            ax.scatter(
                group["circ_rho"],
                group["auroc"],
                c=color,
                s=150,
                alpha=0.85,
                edgecolors="white",
                linewidth=1.5,
                label=severity.capitalize(),
                zorder=5,
            )

            # Label each point
            for _, row in group.iterrows():
                ax.annotate(
                    row["method"],
                    (row["circ_rho"], row["auroc"]),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=9,
                    fontweight="bold" if severity == "clean" else "normal",
                    color=color,
                )

        # Regression line (if enough data points with circ_rho)
        valid = methods_df.dropna(subset=["circ_rho"])
        if len(valid) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(valid["circ_rho"], valid["auroc"])
            z = np.polyfit(valid["circ_rho"], valid["auroc"], 1)
            p_fn = np.poly1d(z)
            x_line = np.linspace(valid["circ_rho"].min() - 0.05, valid["circ_rho"].max() + 0.05, 100)
            ax.plot(
                x_line, p_fn(x_line),
                "--", color="gray", alpha=0.5, linewidth=1.5,
                label=f"ρ={rho:.3f} (p={p:.3f})",
            )

        # Annotations
        ax.axhline(y=0.82, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(0.05, 0.825, "PPI Ceiling ≈ 0.82", fontsize=8, color="gray", alpha=0.6)

        ax.set_xlabel("Circularity Score (ρ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUROC", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        paths = self._save_figure(fig, output_name, format)
        plt.close(fig)
        return paths

    # =========================================================================
    # Figure 4: Feature Exposure Heatmap
    # =========================================================================

    def plot_feature_heatmap(
        self,
        heatmap_df: pd.DataFrame,
        title: str = "Feature Category Exposure by Method",
        output_name: str = "fig4_feature_heatmap",
        format: str = "both",
    ) -> str:
        """
        Feature category exposure heatmap.
        Rows = methods, Columns = feature categories.
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping plot")
            return ""

        # Prepare data
        plot_df = heatmap_df.set_index("method")
        plot_df = plot_df.fillna(-0.1)  # NaN = method doesn't use this category

        fig, ax = plt.subplots(figsize=(10, 6))

        # Custom colormap: white=not used, green=clean, red=circular
        im = ax.imshow(plot_df.values, cmap="RdYlGn_r", aspect="auto", vmin=-0.1, vmax=0.9)

        # Labels
        ax.set_xticks(range(len(plot_df.columns)))
        ax.set_xticklabels(plot_df.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_yticklabels(plot_df.index, fontsize=10)

        # Value annotations
        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                val = plot_df.values[i, j]
                if val < 0:
                    text = "—"
                    color = "lightgray"
                elif val < 0.2:
                    text = f"{val:.2f}"
                    color = "darkgreen"
                elif val < 0.4:
                    text = f"{val:.2f}"
                    color = "black"
                else:
                    text = f"{val:.2f}"
                    color = "darkred"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Mean Exposure Score", shrink=0.8)

        plt.tight_layout()
        paths = self._save_figure(fig, output_name, format)
        plt.close(fig)
        return paths

    # =========================================================================
    # Supplementary: Degree-stratified comparison
    # =========================================================================

    def plot_degree_stratified(
        self,
        deg_df: pd.DataFrame,
        title: str = "Degree-Stratified AUROC: PPI vs Fusion",
        output_name: str = "supp_degree_stratified",
        format: str = "both",
    ) -> str:
        """Bar chart comparing PPI-only vs Fusion across degree bins."""
        if not HAS_MPL:
            return ""

        # Pivot
        pivot = deg_df.pivot(index="degree_bin", columns="method", values="auroc")

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(pivot.index))
        width = 0.2
        colors = ["#ef5350", "#1565c0", "#388e3c", "#ffa726"]

        for i, col in enumerate(pivot.columns):
            vals = pivot[col].values
            ax.bar(x + i * width, vals, width, label=col, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xlabel("Degree Bin", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(pivot.index, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", alpha=0.2)

        plt.tight_layout()
        paths = self._save_figure(fig, output_name, format)
        plt.close(fig)
        return paths

    # =========================================================================
    # Figure 7: Cross-domain synthesis
    # =========================================================================

    def plot_cross_domain_synthesis(
        self,
        domain_results: Dict[str, pd.DataFrame],
        title: str = "Circularity-Performance Trade-off Across Domains",
        output_name: str = "fig7_cross_domain",
        format: str = "both",
    ) -> str:
        """
        Cross-domain scatter plot showing Circ vs AUROC for all domains.
        """
        if not HAS_MPL:
            return ""

        fig, ax = plt.subplots(figsize=(10, 7))

        domain_colors = {
            "SL": "#d32f2f",
            "DTI": "#1565c0",
            "DGA": "#388e3c",
        }
        domain_markers = {
            "SL": "o",
            "DTI": "s",
            "DGA": "^",
        }

        for domain_name, df in domain_results.items():
            color = domain_colors.get(domain_name, "gray")
            marker = domain_markers.get(domain_name, "o")

            valid = df.dropna(subset=["circ_rho", "auroc"])
            ax.scatter(
                valid["circ_rho"],
                valid["auroc"],
                c=color,
                marker=marker,
                s=120,
                alpha=0.8,
                edgecolors="white",
                linewidth=1.2,
                label=f"{domain_name} Prediction",
                zorder=5,
            )

            for _, row in valid.iterrows():
                ax.annotate(
                    row["method"],
                    (row["circ_rho"], row["auroc"]),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=8,
                    color=color,
                )

        ax.set_xlabel("Circularity Score (ρ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUROC", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        paths = self._save_figure(fig, output_name, format)
        plt.close(fig)
        return paths

    # =========================================================================
    # C1-C4 Radar Chart
    # =========================================================================

    def plot_circ_type_radar(
        self,
        methods_df: pd.DataFrame,
        title: str = "Circularity Type Profile",
        output_name: str = "supp_radar",
        format: str = "both",
    ) -> str:
        """Radar chart showing C1-C4 flags for each method."""
        if not HAS_MPL:
            return ""

        categories = ["C1\nLabel", "C2\nFeature", "C3\nEvaluation", "C4\nSelection"]

        # Select top methods for clarity
        plot_methods = methods_df["method"].unique()[:6]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors = ["#d32f2f", "#1565c0", "#388e3c", "#f57c00", "#7b1fa2", "#00838f"]

        for i, method in enumerate(plot_methods):
            row = methods_df[methods_df["method"] == method].iloc[0]
            values = [
                float(row.get("C1_label_leakage", 0)),
                float(row.get("C2_feature_contamination", 0)),
                float(row.get("C3_evaluation_bias", 0)),
                float(row.get("C4_selection_bias", 0)),
            ]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, color=colors[i % len(colors)], label=method)
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["No", "", "Yes"], fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        paths = self._save_figure(fig, output_name, format)
        plt.close(fig)
        return paths

    # =========================================================================
    # Helper
    # =========================================================================

    def _save_figure(
        self, fig, name: str, format: str = "both"
    ) -> str:
        """Save figure in specified format(s)."""
        paths = []
        if format in ("png", "both"):
            png_path = os.path.join(self.output_dir, f"{name}.png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            paths.append(png_path)
        if format in ("pdf", "both"):
            pdf_path = os.path.join(self.output_dir, f"{name}.pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            paths.append(pdf_path)
        if format in ("tiff", "both"):
            tiff_path = os.path.join(self.output_dir, f"{name}.tiff")
            fig.savefig(tiff_path, dpi=300, bbox_inches="tight")
            paths.append(tiff_path)
        print(f"  Saved: {', '.join(paths)}")
        return paths[0] if len(paths) == 1 else "; ".join(paths)
