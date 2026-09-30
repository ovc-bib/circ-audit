"""
make_figures_v11.py — v11 manuscript figures, Elsevier/IPM-style upgrade
=========================================================================
7-figure set, data identical to results/*.csv|json.

Style: boxed thin frames, shaded CI bands, bold lowercase panel letters,
NPG muted palette, rounded schematic boxes, direct annotation over legend.
Outputs: figures/fig{1..7}_*.{png,pdf,tiff} (TIFF 300 dpi)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U

V11 = U.V11
FIG = f"{V11}/figures"
os.makedirs(FIG, exist_ok=True)

# NPG palette (muted, colorblind-considerate)
PAL = {"red": "#E64B35", "blue": "#4DBBD5", "green": "#00A087",
       "navy": "#3C5488", "orange": "#F39B7F", "purple": "#8491B4",
       "grey": "#B0B0B0", "dark": "#333333", "teal": "#16A085",
       "wine": "#943D3D", "gold": "#D7A50F"}
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


def style_ax(ax, grid=True):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(FRAME)
    if grid:
        ax.grid(axis="y", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    ax.tick_params(length=2.5, width=0.8)


def panel(ax, letter, dx=-0.02, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=PAL["dark"],
            ha="left", va="bottom")


def save(fig, name):
    for ext in ("png", "pdf", "tiff", "svg"):
        out = f"{FIG}/{name}.{ext}"
        try:
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        except PermissionError:
            alt = f"{FIG}/{name}_locked.{ext}"
            fig.savefig(alt, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"  WARNING: {out} locked by a viewer, wrote {alt}")
    plt.close(fig)
    print(f"  saved {name}")


def fmt3(x):
    """3-decimal label with half-up rounding, so 0.9015 renders as 0.902
    and matches Table 1 (Python's f-format gives 0.901 via binary repr)."""
    return f"{round(float(x) + 1e-9, 3):.3f}"


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, alpha=1.0, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, alpha=alpha,
                                mutation_aspect=1.0))


def arrow(ax, x0, y0, x1, y1, color, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------
# Fig 1 — protocol
# ------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    split = pd.read_csv(f"{U.V11}/../v10_bioinformatics/data/gold_temporal_split.csv")
    fy = split["first_year_experimental"].dropna()

    # (a) evidence-dated gold standard
    ax = axes[0][0]
    style_ax(ax)
    ax.hist(fy, bins=np.arange(1984, 2021), color=PAL["navy"], alpha=0.9,
            edgecolor="white", lw=0.4)
    top = ax.get_ylim()[1]
    for c, lab, col in ((2012, "train \u22642012", PAL["red"]),
                        (2014, "train \u22642014", PAL["orange"]),
                        (2016, "train \u22642016", PAL["gold"])):
        ax.axvline(c + 0.5, color=col, lw=1.2, ls="--")
        ax.text(c + 1.0, top * 0.95, lab, color=col, fontsize=6.5,
                rotation=90, va="top")
    ax.set_xlabel("year of first experimental evidence")
    ax.set_ylabel("gold genes")
    ax.set_title("Evidence-dated gold standard", loc="left", style="italic")
    panel(ax, "a")

    # (b) design schematic, boxes fill the left space
    ax = axes[0][1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("The temporal evaluation", loc="left", style="italic")

    # past / future zones
    ax.add_patch(plt.Rectangle((0.015, 0.02), 0.575, 0.84, fc="#EDF3F9",
                               ec="none", zorder=0))
    ax.add_patch(plt.Rectangle((0.625, 0.02), 0.36, 0.84, fc="#FBEBE8",
                               ec="none", zorder=0))
    ax.text(0.30, 0.925, "past", ha="center", fontsize=7.5, color="#7A94B8",
            style="italic")
    ax.text(0.805, 0.925, "future", ha="center", fontsize=7.5,
            color="#C9958C", style="italic")

    # timeline
    ax.annotate("", xy=(0.965, 0.87), xytext=(0.03, 0.87),
                arrowprops=dict(arrowstyle="-|>", color=FRAME, lw=1.1))
    for xv, lab in ((0.12, "2012"), (0.32, "2014"), (0.50, "2016")):
        ax.plot([xv], [0.87], "o", ms=4, color=FRAME)
        ax.text(xv, 0.897, lab, ha="center", fontsize=7.2, color=FRAME)
    # freeze boundary
    ax.plot([0.60, 0.60], [0.04, 0.87], color=PAL["red"], lw=2.0,
            ls=(0, (4, 3)), zorder=2)
    ax.text(0.618, 0.79, "T", fontsize=10.5, fontweight="bold",
            color=PAL["red"])
    ax.text(0.583, 0.45, "freeze", rotation=90, ha="right", va="center",
            fontsize=6.8, color=PAL["red"])

    # knowledge box (past): drawn at (0.02, 0.391, 0.565, 0.445)
    for dx, dy, fc, ec in ((0.008, 0.018, "#D8D8D8", "#D8D8D8"),
                           (0.0, 0.036, "#FFFFFF", PAL["navy"])):
        rbox(ax, 0.02 + dx, 0.355 + dy, 0.565, 0.445, fc, ec, 1.2)
    ax.text(0.3025, 0.775, "knowledge at T", ha="center", fontsize=9,
            fontweight="bold", color=PAL["navy"])
    for i, nm in enumerate(("BioGRID", "STRING", "GO", "DepMap")):
        cx = 0.038 + i * 0.136
        rbox(ax, cx, 0.615, 0.128, 0.080, "#DCE8F4", PAL["navy"], 0.8)
        ax.text(cx + 0.064, 0.655, nm, ha="center", va="center",
                fontsize=6.8, color=PAL["navy"])
    ax.text(0.3025, 0.585, "vintage features", ha="center", fontsize=6.5,
            color="#5B7DA8")
    ax.scatter(np.linspace(0.055, 0.55, 8), [0.505] * 8, s=20,
               color=PAL["gold"], edgecolor="white", lw=0.5, zorder=3)
    ax.text(0.3025, 0.452, "dated labels (gold \u2264 T)", ha="center",
            fontsize=6.5, color="#8A6D0B")

    # model box: drawn at (0.03, 0.066, 0.47, 0.26)
    for dx, dy, fc, ec in ((0.008, 0.005, "#D8D8D8", "#D8D8D8"),
                           (0.0, 0.016, "#FFFFFF", PAL["grey"])):
        rbox(ax, 0.03 + dx, 0.05 + dy, 0.47, 0.26, fc, ec, 1.2)
    ax.text(0.265, 0.268, "train model", ha="center", fontsize=8.5,
            fontweight="bold", color=PAL["dark"])
    npos = [(0.15, 0.205), (0.26, 0.170), (0.37, 0.205), (0.20, 0.128),
            (0.315, 0.128)]
    for (x1, y1), (x2, y2) in ((npos[0], npos[1]), (npos[1], npos[2]),
                                (npos[0], npos[3]), (npos[2], npos[4]),
                                (npos[3], npos[4])):
        ax.plot([x1, x2], [y1, y2], color="#9AA5B1", lw=0.9, zorder=2)
    ax.scatter(*zip(*npos), s=17, color=PAL["navy"], edgecolor="white",
               lw=0.5, zorder=3)

    # discovery box (future): drawn at (0.635, 0.436, 0.335, 0.38)
    for dx, dy, fc, ec in ((0.008, 0.018, "#D8D8D8", "#D8D8D8"),
                           (0.0, 0.036, "#FFFFFF", PAL["red"])):
        rbox(ax, 0.635 + dx, 0.40 + dy, 0.335, 0.38, fc, ec, 1.2)
    ax.text(0.8025, 0.745, "discoveries\nafter T", ha="center", fontsize=8.5,
            fontweight="bold", color=PAL["red"], linespacing=1.1)
    ax.scatter(np.linspace(0.665, 0.94, 6), [0.575] * 6, s=26,
               color=PAL["red"], marker="*", edgecolor="white", lw=0.4,
               zorder=3)
    ax.text(0.8025, 0.505, "held-out future gold", ha="center", fontsize=6.8,
            color=PAL["dark"])

    # arrows: knowledge -> model -> (across the freeze line) -> discoveries
    arrow(ax, 0.3025, 0.391, 0.30, 0.326, PAL["navy"], lw=1.6)
    arrow(ax, 0.50, 0.19, 0.79, 0.436, PAL["red"], lw=1.6)
    ax.text(0.66, 0.315, "score", fontsize=7, color=PAL["red"],
            rotation=28, ha="center")
    panel(ax, "b", dx=0.0, dy=1.10)

    # (c) split sizes per cutoff
    ax = axes[1][0]
    style_ax(ax)
    x = np.arange(3)
    old = [415, 591, 630]
    new = [273, 97, 58]
    ax.bar(x - 0.19, old, 0.36, color=PAL["navy"], label="train (\u2264 T)",
           edgecolor="white", lw=0.5)
    ax.bar(x + 0.19, new, 0.36, color=PAL["red"], label="test (> T)",
           edgecolor="white", lw=0.5)
    for xi, o, n in zip(x, old, new):
        ax.text(xi - 0.19, o + 12, str(o), ha="center", fontsize=6.5,
                color=PAL["navy"])
        ax.text(xi + 0.19, n + 12, str(n), ha="center", fontsize=6.5,
                color=PAL["red"])
    ax.set_xticks(x)
    ax.set_xticklabels(["2012", "2014", "2016"])
    ax.set_xlabel("cutoff")
    ax.set_ylabel("gold genes")
    ax.set_ylim(0, 760)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Split sizes", loc="left", style="italic")
    panel(ax, "c")

    # (d) knowledge-channel vintage map
    ax = axes[1][1]
    style_ax(ax, grid=False)
    channels = [
        ("BioGRID physical", [2013, 2015, 2024], PAL["navy"]),
        ("Hetionet", [2016], PAL["blue"]),
        ("STRING (bulk)", [2016, 2021, 2023], PAL["red"]),
        ("GO annotations", [2016, 2026], PAL["green"]),
        ("PubMed counts", [2017, 2026], PAL["teal"]),
        ("DepMap (core block)", [2019, 2024], PAL["purple"]),
    ]
    for i, (name, yrs, col) in enumerate(channels):
        ax.plot([min(yrs), max(yrs)], [i, i], color=col, lw=1.2, alpha=0.45,
                zorder=1)
        ax.scatter(yrs, [i] * len(yrs), s=24, color=col, zorder=3,
                   edgecolor="white", lw=0.6)
    for c in (2012, 2014, 2016):
        ax.axvline(c + 0.5, color="#DDDDDD", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([c[0] for c in channels], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(2011.5, 2027.5)
    ax.set_xticks([2012, 2016, 2020, 2024])
    ax.set_xlabel("snapshot vintage (year)")
    ax.set_title("Knowledge-channel vintages", loc="left", style="italic")
    ax.annotate("reordering transitions absorb", xy=(2018.5, 2.0),
                xytext=(2019.0, 4.1), fontsize=6.2, color=PAL["red"],
                arrowprops=dict(arrowstyle="-", color=PAL["red"], lw=0.7))
    panel(ax, "d")
    fig.subplots_adjust(wspace=0.40, hspace=0.55)
    save(fig, "fig1_protocol")

def fig2():
    t1 = pd.read_csv(f"{V11}/results/t1_prospective_leaderboard.csv")
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.85],
                          hspace=0.78)

    # (a) dumbbell: retrospective vs prospective @2016
    ax = fig.add_subplot(gs[0])
    style_ax(ax, grid=False)
    ax.grid(axis="x", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    show = ["SINaTRA", "Fusion", "PPI_GBM", "PPI_Proximity",
            "CRISPR_Only", "KG4SL", "SL2MF", "SLant"]
    lab = {"SINaTRA": "SINaTRA-style", "SINaTRA_noSLDB": "SINaTRA (−SL counts)",
           "Fusion": "Fusion",
           "PPI_GBM": "PPI-GBM", "PPI_Proximity": "PPI proximity",
           "CRISPR_Only": "CRISPR-only", "KG4SL": "KG4SL-style",
           "SL2MF": "SL2MF-style", "SLant": "SLant"}
    rows = []
    for m in show:
        r = t1[(t1.method == m) & (t1.cutoff == 2016)].iloc[0]
        if pd.notna(r["retro_cv"]):
            rows.append((m, float(r["retro_cv"]), float(r["auroc"])))
    rows.sort(key=lambda t: t[1])
    for i, (m, retro, pros) in enumerate(rows):
        col = PAL["green"] if m == "CRISPR_Only" else (
            PAL["red"] if m == "SINaTRA" else PAL["navy"])
        ax.plot([retro, pros], [i, i], color=col, lw=1.7, alpha=0.75,
                solid_capstyle="round", zorder=2)
        ax.scatter([retro], [i], s=30, color="white", edgecolor="#888888",
                   lw=1.1, zorder=3)
        ax.scatter([pros], [i], s=44, color=col, edgecolor="white", lw=0.7,
                   zorder=4)
        ax.text(1.005, i, f"{pros - retro:+.3f}", fontsize=6.2, va="center",
                color=col)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab[r[0]] for r in rows], fontsize=6.8)
    ax.set_xlim(0.53, 1.0)
    ax.set_xlabel("AUROC    (open circle = retrospective CV, "
                  "filled = prospective, label = difference)")
    ax.scatter([], [], s=30, color="white", edgecolor="#888888", lw=1.1,
               label="retrospective CV")
    ax.scatter([], [], s=44, color=PAL["navy"], label="prospective")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    ax.annotate("CRISPR-only: the only family whose\nconnector points the other way",
                xy=(0.735, 5.3), xytext=(0.80, 4.35), fontsize=6.3,
                color=PAL["green"],
                arrowprops=dict(arrowstyle="-", color=PAL["green"], lw=0.7))
    ax.set_title("Restricting labels changes nothing", loc="left",
                 style="italic")
    panel(ax, "a")

    # (b) heatmap: methods x (retro + cutoffs)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_auc = LinearSegmentedColormap.from_list(
        "auc", ["#3C5488", "#F7F7F7", "#E64B35"])
    ax = fig.add_subplot(gs[1])
    hm = ["SINaTRA", "SINaTRA_noSLDB", "SINaTRA_topo", "Fusion", "PPI_GBM",
          "PPI_Proximity", "CRISPR_Only", "KG4SL", "SL2MF", "SLant"]
    hlab = [lab.get(m, "SINaTRA (topo+omics)") for m in hm]
    M = np.full((len(hm), 4), np.nan)
    for i, m in enumerate(hm):
        r0 = t1[(t1.method == m) & (t1.cutoff == 2016)].iloc[0]
        M[i, 0] = r0["retro_cv"]
        for j, c in enumerate((2012, 2014, 2016), start=1):
            M[i, j] = t1[(t1.method == m) & (t1.cutoff == c)].iloc[0]["auroc"]
    im = ax.imshow(M, cmap=cmap_auc, vmin=0.55, vmax=1.0, aspect="auto")
    for i in range(len(hm)):
        for j in range(4):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, fmt3(v), ha="center", va="center",
                        fontsize=6.4,
                        color="white" if (v > 0.88 or v < 0.62) else "#333333")
    ax.axvline(0.5, color="white", lw=2.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["retro (CV)", "prospective @2012", "@2014", "@2016"],
                       fontsize=6.8)
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels(hlab, fontsize=6.6)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hm), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("AUROC", fontsize=6.5)
    ax.set_title("Prospective leaderboard across cutoffs", loc="left",
                 style="italic")
    panel(ax, "b")

    # (c) SINaTRA input-family ablation
    ax = fig.add_subplot(gs[2])
    style_ax(ax)
    b5 = pd.read_csv(f"{U.V11}/results/b5_sinatra_vintage.csv")
    variants = [("SINaTRA", "full inputs", PAL["red"]),
                ("SINaTRA_noSLDB", "\u2212 SL counts", PAL["orange"]),
                ("SINaTRA_topo", "topology + omics", PAL["navy"])]
    x = np.arange(3)
    for k, (m, lab_, col) in enumerate(variants):
        sub = t1[t1.method == m].sort_values("cutoff")
        ax.bar(x + (k - 1.5) * 0.21, sub["auroc"], 0.19, color=col,
               label=lab_, edgecolor="white", lw=0.5)
        for bxi, v in zip(x + (k - 1.5) * 0.21, sub["auroc"]):
            ax.text(bxi, v + 0.004, fmt3(v), ha="center", fontsize=5.2,
                    color=col)
    sub = b5.sort_values("cutoff")
    ax.bar(x + 1.5 * 0.21, sub["auroc"], 0.19, color=PAL["teal"],
           label="topo, 2016 networks", edgecolor="white", lw=0.5)
    for bxi, v in zip(x + 1.5 * 0.21, sub["auroc"]):
        ax.text(bxi, v + 0.004, fmt3(v), ha="center", fontsize=5.2,
                color=PAL["teal"])
    ax.axhline(0.976, ls=":", lw=1.0, color=PAL["grey"])
    ax.text(2.42, 0.979, "retro 0.976", fontsize=6, color=PAL["grey"],
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["@2012", "@2014", "@2016"])
    ax.set_xlabel("training cutoff")
    ax.set_ylabel("prospective AUROC")
    ax.set_ylim(0.70, 1.005)
    ax.legend(frameon=False, loc="lower left", fontsize=5.6, ncol=2)
    ax.set_title("SINaTRA-style: input-family ablation", loc="left",
                 style="italic")
    panel(ax, "c")
    save(fig, "fig2_leaderboard")


# ------------------------------------------------------------------
# Fig 3 — mechanisms
# ------------------------------------------------------------------
def fig3():
    d = pd.read_csv(f"{U.V11}/results/t_diag_single_column.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    mcol = {"label restatement": PAL["red"],
            "study bias + literature absorption": PAL["orange"],
            "network position + absorption": PAL["blue"],
            "perturbation variability (clean)": PAL["green"],
            "perturbation effect (clean)": PAL["green"],
            "vintage-clean study bias": PAL["teal"],
            "study-bias carrier": PAL["grey"],
            "knowledge-graph channel": PAL["purple"],
            "knowledge channel": PAL["grey"],
            "knowledge channel (weak)": PAL["grey"],
            "omics (weak)": PAL["grey"],
            "other": PAL["grey"]}
    top = d.head(13).copy()
    top["auc_plot"] = [max(a, 1 - a) for a in top["auroc_2017plus"]]
    ax = axes[0][0]
    style_ax(ax, grid=False)
    ax.grid(axis="x", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["auc_plot"] - 0.5, left=0.5, height=0.68,
            color=[mcol.get(m, PAL["grey"]) for m in top["mechanism"]],
            edgecolor="white", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f if len(f) < 18 else f[:17] + "." for f in
                        top["feature"]], fontsize=6.2)
    ax.axvline(0.5, lw=0.8, color=FRAME)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC on future gold (direction-removed)")
    ax.set_title("Single features on 2017+ gold", loc="left", style="italic")
    panel(ax, "a")

    ax = axes[0][1]
    style_ax(ax)
    bars = [("BioGRID\n2013 degree", 0.7561, PAL["teal"], "clean"),
            ("Hetionet\n2016 degree", 0.6982, PAL["teal"], "clean"),
            ("today's\ndegree", 0.7241, PAL["grey"], "current"),
            ("today's\nlit proxy", 0.9687, PAL["orange"], "current")]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], 0.58, color=[b[2] for b in bars],
           edgecolor="white", lw=0.5)
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + 0.012, f"{b[1]:.3f}", ha="center", fontsize=6.6,
                color=b[2], fontweight="bold")
    ax.axhline(0.5, lw=0.8, ls=":", color=PAL["grey"])
    ax.annotate("", xy=(3, 0.94), xytext=(0.5, 0.79),
                arrowprops=dict(arrowstyle="-|>", color=PAL["red"], lw=1.2))
    ax.text(1.75, 0.87, "absorption  ≈ +0.21", fontsize=6.6, color=PAL["red"],
            ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.2)
    ax.set_ylabel("AUROC on future gold")
    ax.set_ylim(0.5, 1.03)
    ax.set_title("Study bias vs absorption", loc="left", style="italic")
    panel(ax, "b")

    ax = axes[1][0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Three routes to 'foresight'", loc="left", style="italic")
    rows = [
        ("label restatement", "today's SL-database counts already\n"
         "include the future discoveries", "0.97", PAL["red"], "#FBEAE7"),
        ("literature prominence", "the field studies the genes it will\n"
         "discover next, and writes about them\nafterwards", "0.97",
         PAL["orange"], "#FDF3EC"),
        ("feature-vintage absorption", "databases re-absorb discoveries\n"
         "into edges, weights and counts", "+0.10", PAL["blue"], "#E9F4F7"),
    ]
    for i, (t, sub, val, col, fill) in enumerate(rows):
        y0 = 0.80 - i * 0.29
        rbox(ax, 0.02, y0 - 0.12, 0.96, 0.24, fill, col, 1.1)
        ax.text(0.06, y0 + 0.05, t, fontsize=7, va="center", color=col,
                fontweight="bold")
        ax.text(0.06, y0 - 0.045, sub, fontsize=5.8, va="center",
                color="#555555")
        ax.text(0.93, y0 - 0.01, val, fontsize=9, va="center", ha="right",
                color=col, fontweight="bold")
    panel(ax, "c", dx=0.0, dy=1.10)
    # (d) negative-pool composition dumbbell
    ax = axes[1][1]
    style_ax(ax, grid=False)
    b6 = pd.read_csv(f"{U.V11}/results/b6_negative_pool.csv")
    t1 = pd.read_csv(f"{U.V11}/results/t1_prospective_leaderboard.csv")
    excl = b6[b6.variant == "excl_broad"].set_index("method")["auroc"]
    base16 = t1[(t1.cutoff == 2016)].set_index("method")["auroc"]
    show = ["PPI_GBM", "Fusion", "PPI_Proximity", "CRISPR_Only", "SLant"]
    labd = {"PPI_GBM": "PPI (GBM)", "Fusion": "Fusion",
            "PPI_Proximity": "PPI proximity", "CRISPR_Only": "CRISPR-only",
            "SLant": "SLant (label-free)"}
    from matplotlib.lines import Line2D
    for i, m in enumerate(show):
        a, b_ = float(base16.get(m, np.nan)), float(excl.get(m, np.nan))
        ax.plot([a, b_], [i, i], color="#DDDDDD", lw=2.5, zorder=1)
        ax.scatter(a, i, s=26, color=PAL["navy"], zorder=3,
                   edgecolor="white", lw=0.6)
        ax.scatter(b_, i, s=26, color=PAL["red"], zorder=3,
                   edgecolor="white", lw=0.6)
        ax.text(a - 0.012, i, fmt3(a), ha="right", va="center",
                fontsize=6.0, color=PAL["navy"])
        ax.text(b_ + 0.012, i, fmt3(b_), ha="left", va="center",
                fontsize=6.0, color=PAL["red"])
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([labd[m] for m in show], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0.50, 1.07)
    ax.set_xlabel("prospective AUROC at cutoff 2016")
    ax.set_title("Negative-pool composition", loc="left", style="italic")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5,
                              color=PAL["navy"],
                              label="curated genes kept as negatives"),
                       Line2D([], [], marker="o", ls="", ms=5,
                              color=PAL["red"],
                              label="curated genes removed")],
              frameon=False, fontsize=5.6, loc="lower right")
    panel(ax, "d")
    fig.subplots_adjust(wspace=0.42, hspace=0.55)
    save(fig, "fig3_mechanisms")


# ------------------------------------------------------------------
# Fig 4 — vintage chain (curve + bubble + growth + stability matrix)
# ------------------------------------------------------------------
def fig4():
    t2 = pd.read_csv(f"{V11}/results/t2_vintage_chain.csv")
    t2b = pd.read_csv(f"{V11}/results/t2b_biogrid_dose.csv")
    stab = json.load(open(f"{U.V11}/results/t2c_rank_stability.json"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    ax = axes[0][0]
    style_ax(ax)
    chain = pd.concat([
        t2b[t2b["network"] == "BioGRID_3.2.121"],
        t2b[t2b["network"] == "BioGRID_3.4.136"],
        t2[t2["network"] == "Hetionet_2016"],
        t2[t2["network"] == "STRING_2021"],
        t2b[t2b["network"] == "BioGRID_4.4.249"],
    ])
    for model, col, mk in (("PPI_vintage", PAL["navy"], "o"),
                           ("Fusion_vintage", PAL["orange"], "s")):
        g = chain[chain["model"] == model]
        ax.fill_between(g["vintage_year"], g["ci_lo"], g["ci_hi"],
                        color=col, alpha=0.15, lw=0)
        ax.plot(g["vintage_year"], g["auroc"], mk + "-", color=col, ms=4.5,
                lw=1.6, label=("PPI (3-feature)" if model == "PPI_vintage"
                               else "Fusion (+CRISPR)"))
    ax.axhline(0.5, color=PAL["grey"], lw=0.8, ls=":")
    ax.set_xlabel("network vintage (year)")
    ax.set_ylabel("AUROC on 2017+ gold")
    ax.set_ylim(0.55, 0.90)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax.annotate("STRING pair: +0.096\nrank stability 0.47",
                xy=(2018.6, 0.735), fontsize=6.3, color=PAL["red"],
                ha="center")
    ax.set_title("Network vintage curve", loc="left", style="italic")
    panel(ax, "a")

    ax = axes[0][1]
    style_ax(ax)
    pts = [
        (0.786, 0.057, "BioGRID 15\u219224", PAL["navy"], 4.2),
        (0.853, 0.048, "BioGRID 13\u219215", PAL["navy"], 1.4),
        (0.465, 0.096, "Hetionet\u2192STRING21", PAL["red"], 1.4),
        (0.794, 0.022, "GO \u226416\u2192current", PAL["green"], 2.2),
        (0.515, 0.054, "STRING v10\u219211.5", PAL["wine"], 1.9),
        (0.877, -0.010, "STRING 11.5\u219212", PAL["wine"], 1.1),
        (0.957, 0.024, "PubMed counts", PAL["teal"], 3.5),
    ]
    for x, y_, lab_, col, gr in pts:
        ax.scatter(x, y_, s=30 * gr + 18, color=col, zorder=3,
                   edgecolor="white", lw=0.7, alpha=0.85)
        ax.annotate(lab_, (x, y_), textcoords="offset points",
                    xytext=(6, 5), fontsize=6.3, color=col)
    ax.scatter([], [], s=30 * 1.4 + 18, color="#BBBBBB", edgecolor="white",
               label="bubble size = growth factor")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.axhline(0, color=PAL["grey"], lw=0.8, ls=":")
    ax.set_xlabel("rank stability between vintages (Spearman)")
    ax.set_ylabel("absorption (\u0394AUROC)")
    ax.set_ylim(-0.04, 0.16)
    ax.set_xlim(0.42, 1.00)
    ax.set_title("Absorption vs rank stability", loc="left", style="italic")
    panel(ax, "b")

    ax = axes[1][0]
    style_ax(ax)
    t2s = json.load(open(f"{U.V11}/results/t2b_summary.json"))
    vers = ["3.2.121", "3.4.136", "4.4.249"]
    ng = [t2s["new_gold_degree_mean"][v] for v in vers]
    cg = [t2s["candidate_degree_mean"][v] for v in vers]
    x = np.arange(3)
    ax.bar(x - 0.19, ng, 0.36, color=PAL["red"], label="2017+ gold genes",
           edgecolor="white", lw=0.5)
    ax.bar(x + 0.19, cg, 0.36, color=PAL["grey"], label="all candidates",
           edgecolor="white", lw=0.5)
    for xi, a, b_ in zip(x, ng, cg):
        ax.text(xi - 0.19, a + 4, f"{a:.0f}", ha="center", fontsize=6.3,
                color=PAL["red"], fontweight="bold")
        ax.text(xi + 0.19, b_ + 4, f"{b_:.0f}", ha="center", fontsize=6.3,
                color="#777777")
    ax.set_xticks(x)
    ax.set_xticklabels(["BioGRID 2013", "2015", "2024"])
    ax.set_xlabel("BioGRID release")
    ax.set_ylabel("mean degree")
    ax.set_ylim(0, 285)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.text(1.55, 225, "\u00d75.7 edges,\nrank order stable\n\u2192 no absorption",
            fontsize=6.3, color=PAL["navy"], ha="center")
    ax.set_title("Degree growth of future gold", loc="left", style="italic")
    panel(ax, "c")

    # (d) 5x5 rank-stability matrix
    from matplotlib.colors import LinearSegmentedColormap
    cmap_st = LinearSegmentedColormap.from_list(
        "st", ["#E64B35", "#F7F7F7", "#00A087"])
    cmap_st.set_bad("#F2F2F2")
    names = stab["networks"]
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for b, v in stab["matrix"][a].items():
            M[i, names.index(b)] = v
    ax = axes[1][1]
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap_st, vmin=0.4, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(M[i, j]):
                txt = "1" if M[i, j] == 1.0 else fmt3(M[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                        color="#333333")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["BG13", "BG15", "Het16", "STR21", "BG24"], fontsize=6.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(["BG13", "BG15", "Het16", "STR21", "BG24"], fontsize=6.6)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("degree rank stability", fontsize=6.2)
    ax.set_title("STRING 2021 reorders against all", loc="left",
                 style="italic")
    panel(ax, "d")
    fig.tight_layout(w_pad=2.2, h_pad=2.4)
    save(fig, "fig4_vintage_chain")


# ------------------------------------------------------------------
# Fig 5 — label-side decomposition
# ------------------------------------------------------------------
def fig5():
    t4 = pd.read_csv(f"{U.V11}/results/t4_label_decomposition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    style_ax(ax)
    models = ["PPI_GBM", "CRISPR_Only", "Fusion"]
    cols = {"PPI_GBM": PAL["navy"], "CRISPR_Only": PAL["green"],
            "Fusion": PAL["orange"]}
    x = np.arange(3)
    for k, m in enumerate(models):
        sub = t4[t4.model == m].sort_values("cutoff")
        bars = ax.bar(x + (k - 1) * 0.27, sub["label_leakage_gain"], 0.25,
                      color=cols[m], label=m, edgecolor="white", lw=0.5)
        for bxi, v in zip(x + (k - 1) * 0.27, sub["label_leakage_gain"]):
            ax.text(bxi, v + 0.006, f"+{v:.3f}", ha="center", fontsize=5.6,
                    color=cols[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["@2012", "@2014", "@2016"])
    ax.set_xlabel("training cutoff")
    ax.set_ylabel("AUROC gain from label leakage")
    ax.set_ylim(0, 0.39)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Label-side leakage is a constant", loc="left",
                 style="italic")
    panel(ax, "a")

    ax = axes[1]
    style_ax(ax)
    sub = t4[t4.cutoff == 2016]
    x = np.arange(len(sub))
    ax.bar(x - 0.19, sub["auroc_kb_restricted"], 0.36,
           color=PAL["navy"], label="≤2016 gold only (honest)",
           edgecolor="white", lw=0.5)
    ax.bar(x + 0.19, sub["auroc_allgold_trained"], 0.36,
           color=PAL["red"], label="all-gold twin (CV-style)",
           edgecolor="white", lw=0.5)
    for xi, (a, b_) in enumerate(zip(sub["auroc_kb_restricted"],
                                     sub["auroc_allgold_trained"])):
        ax.text(xi - 0.19, a + 0.008, f"{a:.3f}", ha="center", fontsize=6.2,
                color=PAL["navy"])
        ax.text(xi + 0.19, b_ + 0.008, f"{b_:.3f}", ha="center", fontsize=6.2,
                color=PAL["red"])
        ax.annotate("", xy=(xi + 0.02, b_ - 0.012), xytext=(xi + 0.02, a + 0.012),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"])
    ax.set_ylabel("AUROC on 2017+ gold")
    ax.set_ylim(0.6, 1.08)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("Twin models @2016", loc="left", style="italic")
    panel(ax, "b")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig5_label_decomposition")


# ------------------------------------------------------------------
# Fig 6 — shortlist (rank strip + recall@K curves + hits)
# ------------------------------------------------------------------
def fig6():
    """End-of-2016 shortlist on the DISCOVERY list (status unknown at cutoff).

    Corrected framing: the full-candidate top 100 is crowded by the 630 known
    gold genes, so the informative pool is the 13,553 candidates whose status
    was unknown at the cutoff. Base-rate expectation at depth K is K*n+/N.
    """
    per_gene = pd.read_csv(f"{U.V11}/results/t7_per_gene_scores.csv")
    full = U.load_scores()
    assert (full["gene"].to_numpy() == per_gene["gene"].to_numpy()).all()
    old_pos, new_pos, _ = U.era_masks(full, 2016)
    pool = ~old_pos
    N = int(pool.sum())
    n_new = int(new_pos.sum())
    yr = U.first_evidence_year()
    genes = full["gene"].to_numpy()
    order = ["Fusion_current", "PPI_GBM_current", "CRISPR_Only",
             "Fusion_vintage2016", "PPI_vintage2016"]
    short = {"Fusion_current": "Fusion (current)",
             "PPI_GBM_current": "PPI (current)",
             "CRISPR_Only": "CRISPR-only",
             "Fusion_vintage2016": "Fusion (2016 honest)",
             "PPI_vintage2016": "PPI (2016 honest)"}

    pool_idx = np.where(pool)[0]
    rfull, med, hits100 = {}, {}, {}
    for v in order:
        s = per_gene[v].to_numpy()
        ranked = pool_idx[np.argsort(-s[pool])]
        r = np.full(len(full), -1, dtype=int)
        r[ranked] = np.arange(1, len(ranked) + 1)
        rfull[v] = r
        nr = r[new_pos]
        med[v] = int(np.median(nr))
        hits100[v] = int((nr <= 100).sum())

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          width_ratios=[1.35, 1.0], hspace=0.5, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    style_ax(ax, grid=False)
    rng = np.random.default_rng(0)
    ax.axvspan(0, 100, color=PAL["gold"], alpha=0.18, zorder=0)
    for i, v in enumerate(order):
        nr = rfull[v][new_pos]
        ax.hlines(i, 1, N, color="#EDEDED", lw=5, zorder=1)
        n1k = int((nr <= 1000).sum())
        xs = rng.uniform(1, 6000, max(n1k, 1))
        ys = rng.uniform(i - 0.22, i + 0.22, max(n1k, 1))
        ax.scatter(xs, ys, s=6, color=PAL["red"], alpha=0.55, zorder=2,
                   edgecolor="none")
        ax.scatter([med[v]], [i], marker="D", s=26, color=PAL["navy"],
                   zorder=3, edgecolor="white", lw=0.5)
        ax.text(9500, i, f"median {med[v]}   top100 {hits100[v]}",
                fontsize=6.2, va="center", color=PAL["navy"])
    ax.text(50, 4.62, "top 100\nE[H]=0.43", fontsize=6.0, ha="center",
            color="#8A6D0B")
    ax.set_xscale("symlog")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short[v] for v in order], fontsize=6.5)
    ax.set_xlabel(f"rank among {N:,} status-unknown candidates (symlog)")
    ax.set_xlim(1, 40000)
    ax.set_ylim(-0.6, 4.9)
    ax.invert_yaxis()
    ax.set_title(f"Ranks of the {n_new} future discoveries", loc="left",
                 style="italic")
    panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax)
    K = np.unique(np.logspace(0, 4.13, 120).astype(int))
    base = K / N * n_new
    ax.plot(K, base, ls="--", lw=0.9, color="#999999", label="base rate")
    pool_new = new_pos[pool]
    positions = np.arange(1, len(pool_new) + 1)
    for v, col in (("Fusion_current", PAL["navy"]),
                   ("Fusion_vintage2016", PAL["orange"]),
                   ("CRISPR_Only", PAL["green"]),
                   ("PPI_vintage2016", PAL["purple"])):
        s = per_gene[v].to_numpy()[pool]
        oi = np.argsort(-s)
        cum = np.cumsum(pool_new[oi].astype(int))
        idx = np.searchsorted(positions, K, side="right") - 1
        ax.plot(K, cum[idx], lw=1.4, color=col, label=short[v])
    ax.set_xscale("log")
    ax.axvline(100, color=PAL["gold"], lw=0.9, ls=":")
    ax.text(105, 30.2, "K=100: 1-9 hits (E=0.43)", fontsize=5.6,
            color="#8A6D0B", ha="left")
    ax.set_xlabel("shortlist depth K (log)")
    ax.set_ylabel(f"discoveries captured (of {n_new})")
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    ax.set_title("Recall at depth K", loc="left", style="italic")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("CRISPR-only leading hits", loc="left", style="italic")
    s = per_gene["CRISPR_Only"].to_numpy()
    hits = [(genes[i], int(rfull["CRISPR_Only"][i]), int(yr.get(genes[i], 0)))
            for i in np.argsort(-s) if new_pos[i]][:8]
    rbox(ax, 0.02, 0.10, 0.96, 0.80, "#F0F7F4", PAL["green"], 1.0)
    ax.text(0.08, 0.82, "gene", fontsize=6.5, fontweight="bold",
            color=PAL["dark"])
    ax.text(0.48, 0.82, "rank", fontsize=6.5, fontweight="bold",
            color=PAL["dark"])
    ax.text(0.72, 0.82, "first evidence", fontsize=6.5, fontweight="bold",
            color=PAL["dark"])
    ax.plot([0.08, 0.92], [0.775, 0.775], lw=0.7, color=PAL["green"])
    for i, (gene, r, yv) in enumerate(hits):
        yy = 0.70 - i * 0.075
        col = PAL["dark"]
        ax.text(0.08, yy, gene, fontsize=6.8, color=col,
                fontweight="bold" if gene == "GMPS" else "normal")
        ax.text(0.48, yy, str(r), fontsize=6.8, color=col)
        ax.text(0.72, yy, str(int(yv)), fontsize=6.8, color=col)
    panel(ax, "c", dx=0.0, dy=1.10)
    save(fig, "fig6_shortlist")

# ------------------------------------------------------------------
# Fig 7 — CircAudit architecture (layered, in the reference papers' style)
# ------------------------------------------------------------------
def fig7():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- column bands ----------------
    ax.text(0.125, 0.965, "INPUTS", ha="center", fontsize=8,
            fontweight="bold", color="#555555")
    ax.text(0.50, 0.965, "EVALUATION PIPELINE", ha="center", fontsize=8,
            fontweight="bold", color="#555555")
    ax.text(0.875, 0.965, "REPORT", ha="center", fontsize=8,
            fontweight="bold", color="#555555")

    # ---------------- input cards (left) ----------------
    def input_card(y, title, sub, col, fill):
        rbox(ax, 0.015, y, 0.215, 0.215, fill, col, 1.1)
        ax.text(0.1225, y + 0.185, title, ha="center", fontsize=6.6,
                fontweight="bold", color=col)
        ax.text(0.1225, y + 0.03, sub, ha="center", fontsize=5.6,
                color="#666666")

    input_card(0.70, "model predictions", "per-gene / per-pair scores  S(x)\n"
               "feature table  X", PAL["navy"], "#E8EFF7")
    # mini feature-matrix grid
    rng = np.random.default_rng(3)
    for r_ in range(3):
        for c_ in range(9):
            v = rng.random()
            fc = (PAL["red"] if v > 0.82 else
                  "#B8CCE4" if v > 0.45 else "#E4E9F0")
            ax.add_patch(plt.Rectangle((0.045 + c_ * 0.0185, 0.845 - r_ * 0.022),
                                       0.0155, 0.017, fc=fc, ec="none"))

    input_card(0.42, "dated knowledge base", "evidence years from PubMed\n"
               "+ archived snapshots", PAL["red"], "#FBEAE7")
    for i, (yr, yy) in enumerate([("2013", 0.565), ("2015", 0.545),
                                  ("2016", 0.525), ("2021", 0.505),
                                  ("2024", 0.485)]):
        col = PAL["red"] if yr in ("2021",) else "#C9C9C9"
        rbox(ax, 0.075, yy - 0.008, 0.026, 0.019, col, col, 0.5, r=0.004)
        ax.text(0.108, yy, yr, fontsize=5.2, va="center", color="#666666")

    input_card(0.14, "feature provenance", "per-column source labels",
               PAL["green"], "#E9F6F1")
    for i, (lab, col) in enumerate([("SL-DB", PAL["red"]),
                                    ("knowledge", PAL["orange"]),
                                    ("topology", PAL["blue"]),
                                    ("omics", PAL["green"])]):
        yy = 0.255 - i * 0.028
        rbox(ax, 0.055, yy - 0.007, 0.02, 0.016, col, col, 0.5, r=0.004)
        ax.text(0.083, yy, lab, fontsize=5.4, va="center", color="#555555")

    # ---------------- engine container (middle) ----------------
    ax.add_patch(FancyBboxPatch((0.275, 0.115), 0.45, 0.795,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc="#FCFCFC", ec="#999999", lw=1.0,
                                ls=(0, (4, 3))))

    def module(x, y, w, h, num, title, col, hl=False):
        rbox(ax, x, y, w, h, "white", col, 1.8 if hl else 1.0)
        ax.text(x + 0.012, y + h - 0.028, num, fontsize=6.4,
                fontweight="bold", color=col)
        ax.text(x + 0.036, y + h - 0.028, title, fontsize=6.4,
                fontweight="bold", color=col)
        if hl:
            rbox(ax, x + w - 0.062, y + h - 0.042, 0.052, 0.022, PAL["red"],
                 PAL["red"], 0.8, r=0.006)
            ax.text(x + w - 0.036, y + h - 0.031, "new", fontsize=5.0,
                    color="white", ha="center", va="center",
                    fontweight="bold")

    # M1 feature exposure (mini bars)
    module(0.295, 0.585, 0.20, 0.145, "1", "feature exposure", PAL["navy"])
    for i, (v, col) in enumerate([(0.96, PAL["red"]), (0.72, PAL["orange"]),
                                  (0.47, PAL["blue"]), (0.05, PAL["green"])]):
        yy = 0.685 - i * 0.023
        ax.plot([0.315, 0.315 + v * 0.115], [yy, yy], lw=3.2, color=col,
                solid_capstyle="round")
    ax.text(0.438, 0.685, "1.0", fontsize=4.8, color="#999999")
    ax.text(0.315, 0.608, "|Spearman| vs dated positives", fontsize=4.8,
            color="#888888")

    # M2 label overlap (mini venn)
    module(0.515, 0.585, 0.19, 0.145, "2", "label overlap", PAL["navy"])
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0.575, 0.663), 0.021, fc="#B8CCE4", ec=PAL["navy"],
                        lw=0.8))
    ax.add_patch(Circle((0.597, 0.663), 0.021, fc="#F4C7C3", ec=PAL["red"],
                        lw=0.8))
    ax.text(0.545, 0.632, "test ∩ KB-train", fontsize=4.8, color="#888888")

    # M3 vintage rebuild + rank stability (HIGHLIGHTED, mini bubble)
    module(0.295, 0.14, 0.20, 0.26, "3", "vintage rebuild +\nrank stability",
           PAL["red"], hl=True)
    ax_mini = ax.inset_axes([0.315, 0.175, 0.16, 0.13])
    ax_mini.scatter([0.853, 0.786, 0.794], [0.048, 0.057, 0.022], s=16,
                    color=[PAL["navy"], PAL["navy"], PAL["green"]],
                    edgecolor="none", zorder=3)
    ax_mini.scatter([0.465], [0.096], s=30, color=PAL["red"],
                    edgecolor="none", zorder=3)
    ax_mini.set_xlim(0.35, 1.0); ax_mini.set_ylim(0, 0.13)
    ax_mini.set_xticks([]); ax_mini.set_yticks([])
    for s_ in ax_mini.spines.values():
        s_.set_color("#CCCCCC"); s_.set_linewidth(0.6)
    ax_mini.set_xlabel("rank stability", fontsize=4.8, labelpad=1,
                       color="#888888")
    ax_mini.set_ylabel("absorption", fontsize=4.8, labelpad=1,
                       color="#888888")

    # M4 performance decomposition (mini dumbbell)
    module(0.515, 0.14, 0.19, 0.26, "4", "performance\ndecomposition",
           PAL["navy"])
    for i, (a_, b_, col) in enumerate([(0.90, 0.90, PAL["navy"]),
                                       (0.66, 0.73, PAL["green"]),
                                       (0.58, 0.60, PAL["purple"])]):
        yy = 0.325 - i * 0.032
        ax.plot([0.535 + a_ * 0.15, 0.535 + b_ * 0.15], [yy, yy], lw=1.4,
                color="#BBBBBB", solid_capstyle="round")
        ax.scatter([0.535 + a_ * 0.15], [yy], s=13, color="white",
                   edgecolor="#888888", lw=0.8, zorder=3)
        ax.scatter([0.535 + b_ * 0.15], [yy], s=17, color=col,
                   edgecolor="white", lw=0.5, zorder=4)
    ax.text(0.61, 0.245, "all features vs\nnon-knowledge features", fontsize=4.8,
            color="#888888", ha="center")

    # internal engine arrows
    arrow(ax, 0.395, 0.585, 0.395, 0.40, "#AAAAAA", lw=0.9)
    arrow(ax, 0.61, 0.585, 0.61, 0.40, "#AAAAAA", lw=0.9)

    # ---------------- report card (right) ----------------
    rbox(ax, 0.755, 0.46, 0.235, 0.45, "white", PAL["navy"], 1.4)
    rbox(ax, 0.755, 0.855, 0.235, 0.055, PAL["navy"], PAL["navy"], 1.4)
    ax.text(0.8725, 0.8825, "evaluation report", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold")
    rows = [("knowledge exposure ρ", "0.262", PAL["orange"]),
            ("max exposure", "0.48", PAL["red"]),
            ("label overlap", "none", "#777777"),
            ("rank stability", "0.47", PAL["red"]),
            ("label-side gain", "+0.08", PAL["orange"])]
    for i, (k, v, col) in enumerate(rows):
        yy = 0.815 - i * 0.045
        ax.text(0.768, yy, k, fontsize=6.0, color="#555555")
        ax.text(0.978, yy, v, fontsize=6.4, color=col, ha="right",
                fontweight="bold")
        ax.plot([0.768, 0.978], [yy - 0.016, yy - 0.016], lw=0.4,
                color="#E5E5E5")
    rbox(ax, 0.80, 0.495, 0.145, 0.045, PAL["orange"], PAL["orange"], 0.8,
         r=0.008)
    ax.text(0.8725, 0.5175, "moderate · absorption-flagged", ha="center",
            va="center", fontsize=5.6, color="white", fontweight="bold")

    # 7-rule protocol strip
    ax.text(0.8725, 0.40, "graduated 7-rule protocol", ha="center",
            fontsize=6.2, fontweight="bold", color="#555555")
    for i in range(7):
        rbox(ax, 0.775 + (i % 4) * 0.055, 0.345 - (i // 4) * 0.042, 0.046,
             0.032, "#E9F6F1", PAL["green"], 0.8, r=0.006)
        ax.text(0.775 + (i % 4) * 0.055 + 0.023,
                0.345 - (i // 4) * 0.042 + 0.016, f"R{i+1}", ha="center",
                va="center", fontsize=5.2, color=PAL["green"],
                fontweight="bold")

    # ---------------- flow arrows ----------------
    for ysrc in (0.81, 0.53, 0.25):
        arrow(ax, 0.232, ysrc, 0.272, ysrc, "#999999", lw=1.1)
    arrow(ax, 0.727, 0.51, 0.753, 0.51, "#999999", lw=1.4)

    # ---------------- bottom release strip ----------------
    rbox(ax, 0.015, 0.015, 0.975, 0.062, "#F5F5F5", "#CCCCCC", 0.8, r=0.008)
    ax.text(0.5, 0.046, "released with the paper", ha="center", fontsize=5.8,
            fontweight="bold", color="#777777")
    ax.text(0.5, 0.028, "pip install circ-audit  ·  dated gene + pair gold "
            "standards  ·  snapshot rebuilders (BioGRID, Hetionet, STRING, "
            "dated GOA)  ·  per-gene + per-pair scores  ·  result tables  ·  "
            "public during review  ·  MIT licence",
            ha="center", fontsize=5.4, color="#888888")

    save(fig, "fig7_tool")


def fig_abstract():
    """Graphical abstract: time machine (left) + the law (right)."""
    fig = plt.figure(figsize=(9.6, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.945, "Feature-side inflation follows rank instability, "
            "not database growth", ha="center", fontsize=15,
            fontweight="bold", color=PAL["dark"])
    ax.text(0.5, 0.895, "A temporal evaluation of synthetic lethality "
            "prediction", ha="center", fontsize=10.5, color="#777777",
            style="italic")
    ax.add_patch(plt.Rectangle((0.025, 0.14), 0.29, 0.64, fc="#EDF3F9", ec="none"))
    ax.add_patch(plt.Rectangle((0.35, 0.14), 0.115, 0.64, fc="#FBEBE8", ec="none"))
    ax.text(0.17, 0.82, "past", ha="center", fontsize=9, color="#7A94B8", style="italic")
    ax.text(0.4075, 0.82, "future", ha="center", fontsize=9, color="#C9958C", style="italic")
    ax.plot([0.335, 0.335], [0.18, 0.78], color=PAL["red"], lw=2.4, ls=(0, (4, 3)))
    ax.text(0.348, 0.735, "T", fontsize=13, fontweight="bold", color=PAL["red"])
    ax.text(0.326, 0.46, "freeze", rotation=90, ha="right", va="center", fontsize=7.5, color=PAL["red"])
    rbox(ax, 0.04, 0.46, 0.26, 0.28, "#FFFFFF", PAL["navy"], 1.4)
    ax.text(0.17, 0.665, "knowledge at T", ha="center", fontsize=10, fontweight="bold", color=PAL["navy"])
    ax.text(0.17, 0.585, "dated labels", ha="center", fontsize=8, color="#8A6D0B")
    ax.scatter(np.linspace(0.075, 0.265, 6), [0.545] * 6, s=22, color=PAL["gold"], edgecolor="white", lw=0.6, zorder=3)
    ax.text(0.17, 0.505, "vintage features", ha="center", fontsize=8, color="#5B7DA8")
    rbox(ax, 0.075, 0.185, 0.19, 0.16, "#FFFFFF", PAL["grey"], 1.4)
    ax.text(0.17, 0.265, "train model", ha="center", fontsize=9.5, fontweight="bold", color=PAL["dark"])
    npos = [(0.115, 0.235), (0.155, 0.222), (0.195, 0.235), (0.135, 0.208), (0.175, 0.208)]
    for (x1, y1), (x2, y2) in ((npos[0], npos[1]), (npos[1], npos[2]), (npos[0], npos[3]), (npos[2], npos[4]), (npos[3], npos[4])):
        ax.plot([x1, x2], [y1, y2], color="#9AA5B1", lw=1.0, zorder=2)
    ax.scatter(*zip(*npos), s=20, color=PAL["navy"], edgecolor="white", lw=0.6, zorder=3)
    rbox(ax, 0.362, 0.44, 0.095, 0.22, "#FFFFFF", PAL["red"], 1.4)
    ax.text(0.4095, 0.60, "new gold", ha="center", fontsize=8.5, fontweight="bold", color=PAL["red"])
    ax.scatter(np.linspace(0.372, 0.447, 4), [0.50] * 4, s=34, color=PAL["red"], marker="*", edgecolor="white", lw=0.5, zorder=3)
    arrow(ax, 0.17, 0.46, 0.17, 0.35, PAL["navy"], lw=1.8)
    arrow(ax, 0.265, 0.265, 0.385, 0.435, PAL["red"], lw=1.8)
    ax.text(0.335, 0.34, "score", fontsize=8.5, color=PAL["red"], rotation=48, ha="center")
    ax.text(0.245, 0.095, "labels and features frozen at T", ha="center", fontsize=8, color="#888888", style="italic")
    axs = ax.inset_axes([0.505, 0.16, 0.44, 0.56])
    style_ax(axs, grid=False)
    axs.grid(axis="y", ls=(0, (2, 3)), lw=0.5, color="#C9C9C9", alpha=0.8)
    axs.axvspan(0.40, 0.60, color="#FBEBE8", zorder=0)
    axs.axvspan(0.60, 1.00, color="#EDF6F2", zorder=0)
    axs.axhline(0, color=PAL["grey"], lw=1.0, ls=":")
    pts = [(0.786, 0.057, "BioGRID 15→24", PAL["navy"], 4.2),
           (0.853, 0.048, "BioGRID 13→15", PAL["navy"], 1.4),
           (0.794, 0.022, "GO", PAL["green"], 2.2),
           (0.465, 0.096, "Hetion→STRING21", PAL["red"], 1.4),
           (0.515, 0.054, "STRING v10→11.5", PAL["wine"], 1.9),
           (0.877, -0.010, "STRING 11.5→12", PAL["wine"], 1.1),
           (0.957, 0.024, "PubMed counts", PAL["teal"], 3.5)]
    for x, y_, lab_, col, gr in pts:
        axs.scatter(x, y_, s=42 * gr + 24, color=col, zorder=3, edgecolor="white", lw=0.9, alpha=0.9)
        axs.annotate(lab_, (x, y_), textcoords="offset points", xytext=(7, 5), fontsize=7.5, color=col)
    axs.set_xlim(0.40, 1.02)
    axs.set_ylim(-0.045, 0.135)
    axs.set_xlabel("rank stability between vintages (Spearman Ρ)", fontsize=9)
    axs.set_ylabel("absorption (ΔAUROC)", fontsize=9)
    axs.tick_params(labelsize=8)
    axs.text(0.50, 0.122, "reorders → absorbs", fontsize=9.5, color=PAL["red"], ha="center", fontweight="bold")
    axs.text(0.86, -0.033, "stable order → no absorption", fontsize=9.5, color="#0B6E55", ha="center", fontweight="bold")
    ax.text(0.5, 0.035, "label-side restriction alone: 0.97 → 0.97   ·   BioGRID ×5.8 edges: +0.009   ·   honest top-100 of the discovery list: 9× enrichment", ha="center", fontsize=9, color="#666666")
    save(fig, "graphical_abstract")


if __name__ == "__main__":
    print("making v11 figures (Elsevier-style)...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("done ->", FIG)
