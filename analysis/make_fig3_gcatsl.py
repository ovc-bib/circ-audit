"""
make_fig3_gcatsl.py — Figure 3: the original GCATSL pipeline under the time
machine (R2 revision, Reviewer 2's requested experiment)
==============================================================================
(a) prospective AUROC of the authors' released implementation at the three
    cutoffs under both feature vintages (near chance), against its
    retrospective cross-validated AUROC on the same pairs and universe
    (0.977) and its released-example five-fold validation (0.931);
(b) the paired feature-vintage contrasts with 95% bootstrap intervals.

Data: results/t13_gcatsl_pair.csv (evaluated on the frozen 6,400-gene
universe with the shared pool test negatives).
Style follows make_figures_v11.py (NPG palette, boxed frames).
Outputs: figures/fig3_gcatsl.{png,pdf,tiff,svg}
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U

V11 = U.V11
FIG = f"{V11}/figures"
PAL = {"red": "#E64B35", "blue": "#4DBBD5", "green": "#00A087",
       "navy": "#3C5488", "grey": "#B0B0B0", "dark": "#333333",
       "gold": "#D7A50F"}
FRAME = "#3B3B3B"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "axes.linewidth": 0.8, "axes.edgecolor": FRAME,
    "xtick.color": FRAME, "ytick.color": FRAME,
    "axes.labelcolor": FRAME, "text.color": FRAME,
    "figure.dpi": 150, "savefig.dpi": 300, "svg.fonttype": "none",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.axisbelow": True,
})


def style_ax(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(FRAME)
    ax.grid(axis="y", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    ax.tick_params(length=2.5, width=0.8)


def main():
    t13 = pd.read_csv(f"{V11}/results/t13_gcatsl_pair.csv")
    t13["cutoff"] = t13["cutoff"].astype(str)

    def v(model, c, col="auroc"):
        return float(t13[(t13.model == model) & (t13.cutoff == str(c))]
                     .iloc[0][col])

    cutoffs = [2012, 2014, 2016]
    het = [v("GCATSL original (hetionet2016)", c) for c in cutoffs]
    cur = [v("GCATSL original (string2021)", c) for c in cutoffs]
    d = [v("GCATSL absorption (2021 minus 2016 vintage)", c)
         for c in cutoffs]
    lo = [v("GCATSL absorption (2021 minus 2016 vintage)", c, "ci_lo")
          for c in cutoffs]
    hi = [v("GCATSL absorption (2021 minus 2016 vintage)", c, "ci_hi")
          for c in cutoffs]
    p = [v("GCATSL absorption (2021 minus 2016 vintage)", c, "p")
         for c in cutoffs]
    cv_cur = v("GCATSL original CV-regime (string2021)", "cv")
    cv_16 = v("GCATSL original CV-regime (hetionet2016)", "cv")
    toy = 0.931

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) the collapse ------------------------------------------------
    ax = axes[0]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, het, w, color=PAL["blue"], edgecolor=FRAME,
           linewidth=0.6, label="2016-vintage inputs")
    ax.bar(x + w / 2, cur, w, color=PAL["navy"], edgecolor=FRAME,
           linewidth=0.6, label="current-vintage inputs")
    ax.axhline(0.5, color=PAL["grey"], lw=0.8, ls=(0, (4, 3)))
    ax.axhline(toy, color=PAL["gold"], lw=0.9)
    ax.axhline(cv_cur, color=PAL["red"], lw=0.9)
    ax.text(2.52, cv_cur - 0.028,
            "cross validation, same pairs and universe,\n"
            f"either feature vintage ({cv_cur:.3f})", fontsize=6.4,
            ha="right", va="top", color=PAL["red"])
    ax.text(2.52, toy - 0.024, f"released example, five-fold ({toy:.3f})",
            fontsize=6.4, ha="right", va="top", color=PAL["gold"])
    ax.text(-0.42, 0.512, "chance", fontsize=6.0, color=PAL["grey"],
            va="bottom")
    for xi, (h, c) in enumerate(zip(het, cur)):
        ax.text(xi - w / 2, h + 0.012, f"{h:.3f}", ha="center", fontsize=6.2)
        ax.text(xi + w / 2, c + 0.012, f"{c:.3f}", ha="center", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{c}" for c in cutoffs])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUROC on future pairs")
    ax.set_title("Authors' implementation, temporal regime", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    style_ax(ax)
    ax.text(-0.14, 1.10, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=PAL["dark"])

    # (b) vintage contrasts -------------------------------------------
    ax = axes[1]
    y = np.arange(3)[::-1]
    for yi, dv, l, h, pv in zip(y, d, lo, hi, p):
        col = PAL["navy"] if abs(dv) < 0.005 else PAL["red"]
        ax.plot([l, h], [yi, yi], color=col, lw=1.4, solid_capstyle="butt")
        ax.plot(dv, yi, "o", ms=4.5, mfc=col, mec=FRAME, mew=0.6)
    ax.axvline(0, color=PAL["grey"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"@{c}" for c in cutoffs])
    for yi, dv, pv in zip(y, d, p):
        lab = "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"
        ax.text(0.062, yi, lab, fontsize=6.4, va="center", color=PAL["dark"])
    ax.set_xlim(-0.13, 0.13)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Paired AUROC difference "
                  "(current minus 2016 vintage)")
    ax.set_title("Feature-vintage contrast, identical test rows",
                 fontweight="bold")
    style_ax(ax)
    ax.grid(axis="x", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.text(-0.17, 1.10, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=PAL["dark"])

    fig.tight_layout(w_pad=2.2)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf", "tiff", "svg"):
        fig.savefig(f"{FIG}/fig3_gcatsl.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"fig3 saved to {FIG}/fig3_gcatsl.{{png,pdf,tiff,svg}}")
    print(f"  CV-regime: current {cv_cur:.3f}, 2016-vintage {cv_16:.3f}")
    print(f"  temporal:  {het} vs {cur}")
    print(f"  contrasts: {d} (p: {p})")


if __name__ == "__main__":
    main()
