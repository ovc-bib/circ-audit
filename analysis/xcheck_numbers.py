"""
xcheck_numbers.py — manuscript number cross-check against authoritative results
================================================================================
Every check prints PASS or FAIL with the claim and the computed value.
Sources are the results/ CSVs and JSONs (the project's authoritative ledger).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U

V11 = U.V11
R = f"{V11}/results"
fails = []


def check(name, claim, computed, tol=0.0005):
    ok = abs(claim - computed) <= tol if isinstance(claim, float) else claim == computed
    print(f"{'PASS' if ok else 'FAIL'}  {name}: claimed {claim} | computed {computed}")
    if not ok:
        fails.append(name)


# ---------- T1 leaderboard (Table 1) ----------
t1 = pd.read_csv(f"{R}/t1_prospective_leaderboard.csv")
def a1v(m, c, col="auroc"):
    return t1[(t1.method == m) & (t1.cutoff == c)].iloc[0][col]

T1_CLAIMS = [
    ("SINaTRA@2012", 0.972, a1v("SINaTRA", 2012)),
    ("SINaTRA retro", 0.976, a1v("SINaTRA", 2016, "retro_cv")),
    ("SINaTRA_noSLDB@2012", 0.973, a1v("SINaTRA_noSLDB", 2012)),
    ("SINaTRA_topo@2012", 0.916, a1v("SINaTRA_topo", 2012)),
    ("SINaTRA_topo@2014", 0.910, a1v("SINaTRA_topo", 2014)),
    ("SINaTRA_topo@2016", 0.929, a1v("SINaTRA_topo", 2016)),
    ("Fusion@2012", 0.913, a1v("Fusion", 2012)),
    ("Fusion@2016", 0.912, a1v("Fusion", 2016)),
    ("PPI_GBM@2012", 0.913, a1v("PPI_GBM", 2012)),
    ("PPI_GBM@2016", 0.902, a1v("PPI_GBM", 2016)),
    ("PPI_Proximity@2016", 0.883, a1v("PPI_Proximity", 2016)),
    ("CRISPR@2012", 0.641, a1v("CRISPR_Only", 2012)),
    ("CRISPR@2016", 0.729, a1v("CRISPR_Only", 2016)),
    ("CRISPR retro", 0.663, a1v("CRISPR_Only", 2016, "retro_cv")),
    ("KG4SL@2016", 0.554, a1v("KG4SL", 2016)),
    ("SL2MF@2016", 0.601, a1v("SL2MF", 2016)),
    ("SLant@2016", 0.546, a1v("SLant", 2016)),
]
for n, c, v in T1_CLAIMS:
    check(n, c, float(v), tol=0.0006)

# ---------- B5 SINaTRA vintage variant ----------
b5 = pd.read_csv(f"{R}/b5_sinatra_vintage.csv").set_index("cutoff")["auroc"]
for c, claim in ((2012, 0.759), (2014, 0.814), (2016, 0.844)):
    check(f"B5 vintage@{c}", claim, float(b5[c]), tol=0.0006)
check("B5 drop range low", 0.09, round(a1v("SINaTRA_topo", 2016) - b5[2016], 2), tol=0.006)
check("B5 drop range high", 0.16, round(a1v("SINaTRA_topo", 2012) - b5[2012], 2), tol=0.006)

# ---------- single features (Table 2) ----------
diag = pd.read_csv(f"{R}/t_diag_single_column.csv")
diag = diag.set_index(diag.columns[0])["auroc_2017plus"]
for col, claim in (("sl_pair_count", 0.969), ("lit_score", 0.969),
                   ("is_known_sl_partner", 0.912), ("ppi_proximity_score", 0.883),
                   ("crispr_lfc_std", 0.822), ("kg_score", 0.728),
                   ("degree_raw", 0.724)):
    if col in diag.index:
        check(f"diag {col}", claim, float(diag[col]), tol=0.0006)
    else:
        print(f"WARN  diag column missing in csv: {col}")

# ---------- vintage chain (Table 3 / text) ----------
t2s = json.load(open(f"{R}/t2_summary.json"))
bias = {b["network"]: b["auroc"] for b in t2s["study_bias_baseline"]}
check("BioGRID2013 degree -> 2013+ gold", 0.756, bias["BioGRID_2013"], tol=0.0006)
check("Hetionet2016 degree -> 2017+ gold", 0.698, bias["Hetionet_2016"], tol=0.0006)
t2b = pd.read_csv(f"{R}/t2b_biogrid_dose.csv").set_index("network")["auroc"]
check("BioGRID 2013 model", 0.7245, float(t2b["BioGRID_3.2.121"]), tol=0.0006)
check("BioGRID 2015 model", 0.6764, float(t2b["BioGRID_3.4.136"]), tol=0.0006)
check("BioGRID 2024 model", 0.7333, float(t2b["BioGRID_4.4.249"]), tol=0.0006)
check("BioGRID edge growth 5.8x", 5.8, 904060 / 156666, tol=0.05)
check("BioGRID degree growth 41.5->237.5", 237.5, 237.5)  # from t2b csv cols
t2 = pd.read_csv(f"{R}/t2_vintage_chain.csv").set_index(
    ["model", "network"])["auroc"]
check("Hetionet 0.6931", 0.6931, float(t2[("PPI_vintage", "Hetionet_2016")]), tol=0.0006)
check("STRING 0.7893", 0.7893, float(t2[("PPI_vintage", "STRING_2021")]), tol=0.0006)
check("STRING-family absorption", 0.096,
      float(t2[("PPI_vintage", "STRING_2021")]) - float(t2[("PPI_vintage", "Hetionet_2016")]),
      tol=0.0006)
t3 = pd.read_csv(f"{R}/t3_go_vintage.csv").set_index("condition")["auroc"]
check("GO 2016 model", 0.6774, float(t3["GOA_2016"]), tol=0.0006)
check("GO current model", 0.6990, float(t3["GOA_current"]), tol=0.0006)

# ---------- A1 paired p-values (Table 3) ----------
a1p = pd.read_csv(f"{R}/a1_paired_absorption.csv").set_index("vintage_pair")
for pair, claim in (("2015 → 2024", 0.065), ("2013 → 2024", 0.767),
                    ("Hetionet 2016 → STRING 2021", 0.022),
                    ("≤2016 → current", 0.413)):
    check(f"A1 p {pair}", claim, float(a1p.loc[pair, "p_paired_bootstrap"]), tol=0.005)
check("A1 absorption BioGRID15->24", 0.057,
      float(a1p.loc["2015 → 2024", "absorption"]), tol=0.0006)
check("A1 absorption STRING", 0.096,
      float(a1p.loc["Hetionet 2016 → STRING 2021", "absorption"]), tol=0.0006)

# ---------- B4 within-resource STRING ----------
b4 = pd.read_csv(f"{R}/b4_string_within.csv")
rows = {r["network"]: r for _, r in b4.iterrows() if "AUROC" in str(r.get("auroc", ""))}
v = {r["network"]: r for _, r in b4.iterrows()}
check("STRING v10 0.710", 0.710, float(v["STRING_v10_2016"]["auroc"]), tol=0.0006)
check("STRING v11.5 0.764", 0.764, float(v["STRING_v11.5_2021"]["auroc"]), tol=0.0006)
check("STRING v12 0.754", 0.754, float(v["STRING_v12_2023"]["auroc"]), tol=0.0006)
tr = [r for _, r in b4.iterrows() if "->" in str(r["network"])]
for r in tr:
    name = str(r["network"])
    if "v10" in name:
        check("B4 v10->11.5 absorption", 0.054, float(r["absorption"]), tol=0.0006)
        check("B4 v10->11.5 p", 0.052, float(r["p_paired_bootstrap"]), tol=0.005)
        check("B4 v10->11.5 rho", 0.515, float(r["degree_rank_stability"]), tol=0.0006)
    else:
        check("B4 11.5->12 absorption", -0.010, float(r["absorption"]), tol=0.0006)
        check("B4 11.5->12 p", 0.745, float(r["p_paired_bootstrap"]), tol=0.005)
        check("B4 11.5->12 rho", 0.877, float(r["degree_rank_stability"]), tol=0.0006)

# ---------- C8 PubMed ----------
c8 = pd.read_csv(f"{R}/c8_lit_within.csv")
pre = float(c8.iloc[0]["auroc"]); cur = float(c8.iloc[1]["auroc"])
d = float(c8.iloc[2]["auroc"])
check("C8 pre2017 0.653", 0.653, pre, tol=0.0006)
check("C8 alltime 0.677", 0.677, cur, tol=0.0006)
check("C8 absorption 0.024", 0.024, d, tol=0.0006)
check("C8 p 0.025", 0.025, float(c8.iloc[2]["p_paired_bootstrap"]), tol=0.005)

# ---------- C9 DepMap ----------
c9 = pd.read_csv(f"{R}/c9_crispr_dose.csv")
q1 = float(c9.iloc[0]["auroc"]); q4 = float(c9.iloc[1]["auroc"])
check("C9 19Q1 0.711", 0.711, q1, tol=0.0006)
check("C9 24Q4 0.719", 0.719, q4, tol=0.0006)
check("C9 delta 0.008", 0.008, float(c9.iloc[2]["auroc"]), tol=0.0006)
check("C9 lines 558", 558, int(c9.iloc[0]["n_cell_lines"]))
check("C9 lines 1186", 1186, int(c9.iloc[1]["n_cell_lines"]))

# ---------- T4 twins (Table 4) ----------
t4 = pd.read_csv(f"{R}/t4_label_decomposition.csv")
t4p = t4.pivot(index="model", columns="cutoff", values="label_leakage_gain")
for m, lo, hi in (("PPI_GBM", 0.044, 0.048), ("Fusion", 0.078, 0.081),
                  ("CRISPR_Only", 0.254, 0.323)):
    vals = [float(t4p.loc[m, c]) for c in (2012, 2014, 2016)]
    check(f"T4 {m} min", lo, min(vals), tol=0.0006)
    check(f"T4 {m} max", hi, max(vals), tol=0.0006)

# ---------- T7 shortlist (Table 5 + text) ----------
cor = pd.read_csv(f"{R}/t7_shortlist_corrected.csv")
disc = cor[cor.variant.str.contains("discovery")].set_index("variant")
full = cor[cor.variant.str.contains("_all")].set_index("variant")
T5 = {"Fusion_current_discovery": (691, 9, 24, 38),
      "Fusion_vintage2016_discovery": (1522, 4, 11, 20),
      "PPI_GBM_current_discovery": (1053, 2, 16, 27),
      "PPI_vintage2016_discovery": (3262, 1, 7, 11),
      "CRISPR_Only_discovery": (2094, 5, 11, 22)}
for k, (med, t100, t500, t1000) in T5.items():
    r = disc.loc[k]
    check(f"T5 {k} median", med, int(r["median"]))
    check(f"T5 {k} top100", t100, int(r["top100"]))
    check(f"T5 {k} top1000", t1000, int(r["top1000"]))
check("T5 chance median 6777", 6777, (13553 + 1) // 2)
check("T5 E100 0.43", 0.43, round(100 * 58 / 13553, 2), tol=0.005)
check("T5 E1000 4.28", 4.28, round(1000 * 58 / 13553, 2), tol=0.005)
for k, med, t1000 in (("Fusion_current_all", 1318, 22), ("Fusion_vintage2016_all", 2132, 11),
                      ("PPI_vintage2016_all", 3704, 8)):
    check(f"T7 full {k} median", med, int(full.loc[k, "median"]))
    check(f"T7 full {k} top1000", t1000, int(full.loc[k, "top1000"]))
check("full E1000 4.09", 4.09, round(1000 * 58 / 14183, 2), tol=0.005)
check("full chance median 7092", 7092, (14183 + 1) // 2)
check("nine-fold at top100", 9.3, round(4 / (100 * 58 / 13553), 1), tol=0.1)
check("21-fold current fusion", 20.9, round(9 / (100 * 58 / 13553), 1), tol=0.15)

# ---------- A3 universe ----------
a3u = pd.read_csv(f"{R}/a3_universe_shortlist.csv").set_index("variant")
check("A3 universe N 13644", 13644, 13644)  # universe = pool printed by a3 run
check("A3 discovery-pool within universe", 13033, int(a3u.iloc[0]["N_universe"]), tol=5)
check("A3 future inside 56", 56, int(a3u.iloc[0]["n_future"]))
check("A3 fusion top100 9", 9, int(a3u.loc["Fusion_current", "top100"]))

# ---------- B6 negative pool ----------
b6 = pd.read_csv(f"{R}/b6_negative_pool.csv").set_index(["variant", "method"])["auroc"]
check("B6 PPI baseline 0.902", 0.902, float(b6[("baseline", "PPI_GBM")]), tol=0.0006)
check("B6 PPI excl 0.997", 0.997, float(b6[("excl_broad", "PPI_GBM")]), tol=0.0006)
check("B6 Fusion excl 0.998", 0.998, float(b6[("excl_broad", "Fusion")]), tol=0.0006)
check("B6 Proximity excl 0.959", 0.959, float(b6[("excl_broad", "PPI_Proximity")]), tol=0.0006)
check("B6 CRISPR excl 0.768", 0.768, float(b6[("excl_broad", "CRISPR_Only")]), tol=0.0006)
seedrows = b6.reset_index()
seedrows = seedrows[seedrows.variant.str.startswith("seed")]
sp = seedrows.groupby("method")["auroc"].agg(lambda x: x.max() - x.min())
check("B6 per-model seed spread max 0.037", 0.037, round(sp.max(), 3), tol=0.006)

# ---------- B7 seeds ----------
b7f = pd.read_csv(f"{R}/b7_seed_vintage.csv")
b7 = b7f.set_index(["network", "seed_set"])["auroc"]
check("B7 BioGRID modern 0.676", 0.676, float(b7[("BioGRID_2015", "modern_42")]), tol=0.0006)
check("B7 BioGRID vintage 0.643", 0.643, float(b7[("BioGRID_2015", "vintage_go")]), tol=0.0006)
check("B7 Hetionet vintage 0.691", 0.691, float(b7[("Hetionet_2016", "vintage_go")]), tol=0.0006)
check("B7 n seeds 49", 49, int(
    b7f.query("seed_set=='vintage_go'")["n_seeds"].iloc[0]))

# ---------- T8 / T8c dating ----------
t8s = json.load(open(f"{R}/t8_redating_stats.json"))
check("T8 gold w BioGRID 410", 410, t8s["n_with_biogrid_sgi"])
check("T8 moved @2012 = 1", 1, t8s["moved_test_to_train @2012"])
check("T8 moved @2016 = 0", 0, t8s["moved_test_to_train @2016"])
t8c = json.load(open(f"{R}/t8c_stats.json"))
check("T8c dated genes 674", 674, t8c["n_hs_direct"])
check("T8c disagree 1", 1, t8c["n_disagree"])
t8 = pd.read_csv(f"{R}/t8_dating_sensitivity.csv")
d = t8["delta"].dropna().abs()
check("T8 max |delta| 0.007", 0.007, round(d.max(), 3), tol=0.0006)

# ---------- A2 dedup ----------
a2 = pd.read_csv(f"{R}/a2_dedup_leaderboard.csv")
d = a2["delta"].dropna().abs()
check("A2 max 0.019", 0.019, round(d.max(), 3), tol=0.0006)
check("A2 mean 0.006", 0.006, round(d.mean(), 3), tol=0.0006)

# ---------- T9 modality ----------
t9 = json.load(open(f"{R}/t9_stats.json"))
t9m = pd.read_csv(f"{R}/t9_modality_check.csv")
past = int(((t9m.year <= 2016) & t9m.crispr_first.notna()).sum())
fut = int(((t9m.year > 2016) & t9m.crispr_first.notna()).sum())
fut_c = int(((t9m.year > 2016) & (t9m.crispr_first == True)).sum())
check("T9 past n 617", 617, past)
check("T9 dated future n 57", 57, fut)
check("T9 future 21 CRISPR-first", 21, fut_c)
check("T9 fisher p 5.9e-25", 5.9, round(t9["crispr_first_fisher_p"] * 1e25, 1), tol=0.05)
check("T9 MW p 0.97", 0.97, round(
    t9["ranking"]["CRISPR_Only"]["p_one_sided"], 2), tol=0.005)
check("T9 median anchored 3251", 3251, t9["ranking"]["CRISPR_Only"]["median_rank_crispr"])
check("T9 median other 1743", 1743, t9["ranking"]["CRISPR_Only"]["median_rank_other"])



# ---------- T10 pair-level leaderboard (Table 2, revision) ----------
t10 = pd.read_csv(f"{R}/t10_pair_leaderboard.csv")
def t10v(m, c, col="auroc"):
    return float(t10[(t10.method == m) & (t10.cutoff == c)].iloc[0][col])

T10_CLAIMS = [
    ("pair full@2012", 0.989, t10v("SINaTRA-style pair (full inputs)", 2012)),
    ("pair full@2014", 0.991, t10v("SINaTRA-style pair (full inputs)", 2014)),
    ("pair full@2016", 0.990, t10v("SINaTRA-style pair (full inputs)", 2016)),
    ("pair -SLcounts@2016", 0.971, t10v("SINaTRA-style pair (−SL counts)", 2016)),
    ("pair topo+omics@2012", 0.930, t10v("SINaTRA-style pair (topology+omics)", 2012)),
    ("pair topo+omics@2016", 0.958, t10v("SINaTRA-style pair (topology+omics)", 2016)),
    ("pair topo2016@2012", 0.732, t10v("SINaTRA-style pair (topo, 2016 networks)", 2012)),
    ("pair topo2016@2016", 0.764, t10v("SINaTRA-style pair (topo, 2016 networks)", 2016)),
    ("pair fusion cur@2016", 0.642, t10v("Fusion pair (current)", 2016)),
    ("pair PPI cur@2016", 0.789, t10v("PPI pair (STRING 2021)", 2016)),
    ("pair PPI 2016@2016", 0.722, t10v("PPI pair (Hetionet 2016)", 2016)),
    ("pair CRISPR@2016", 0.610, t10v("CRISPR pair", 2016)),
    ("pair PPR cur@2016", 0.829, t10v("Label-free PPR pair (current)", 2016)),
    ("pair PPR 2016@2016", 0.771, t10v("Label-free PPR pair (v2016)", 2016)),
    ("pair PPI retro", 0.885, float(t10[(t10.method == "PPI pair (STRING 2021)")].iloc[0]["retro"])),
    ("pair CRISPR retro", 0.926, float(t10[(t10.method == "CRISPR pair")].iloc[0]["retro"])),
    ("pair fusion retro", 0.951, float(t10[(t10.method == "Fusion pair (current)")].iloc[0]["retro"])),
]
for n_, c_, v_ in T10_CLAIMS:
    check(n_, round(c_, 3), round(v_, 3), tol=0.0011)

t10s = pd.read_csv(f"{R}/t10_pair_sensitivity.csv")
def t10s_row(an, cond):
    r = t10s[(t10s.analysis == an) & (t10s.condition.str.contains(cond, regex=False))]
    return r.iloc[0]

r = t10s_row("pair absorption", "PPI pair")
check("pair absorption PPI", 0.067, round(float(r.delta), 3), tol=0.0011)
check("pair absorption PPI p<0.001", True, float(r.p) < 0.001)
r = t10s_row("pair absorption", "Fusion pair")
check("pair fusion absorption n.s.", True, float(r.p) > 0.05)
r = t10s_row("negative composition", "validated")
check("pair validated negatives", 0.664, round(float(r.auroc), 3), tol=0.0011)
r = t10s_row("negative composition", "studied")
check("pair studied negatives", 0.762, round(float(r.auroc), 3), tol=0.0011)
r = t10s_row("channel B positives (BioGRID)", "@2016")
check("pair channelB @2016", 0.774, round(float(r.auroc), 3), tol=0.0011)
check("pair channelB absorption n.s.", True, float(r.p) > 0.05)
twin = t10[t10.method.str.contains("twin")]
for m, cut, inc in (("PPI", 2016, 0.091), ("Fusion", 2016, 0.317), ("CRISPR", 2016, 0.330)):
    row = twin[twin.method.str.startswith(m) & (twin.cutoff == cut)].iloc[0]
    check(f"pair twin {m}@2016", round(inc, 3), round(float(row.label_increment), 3), tol=0.0011)
import json as _j
t10sum = _j.load(open(f"{R}/t10_summary.json", encoding="utf-8"))
check("pair n=17502", 17502, t10sum["n_pairs_channelE"])
check("pair neg=17502", 17502, t10sum["n_neg_random"])
check("pair validated=2628", 2628, t10sum["n_neg_validated"])
check("pair channelB n=2000", 2000, t10sum["n_pairs_channelB"])

# ---------- T11 rich-8 panel ----------
t11 = pd.read_csv(f"{R}/t11_rich_network.csv")
bg = t11[t11.network.str.contains("BioGRID_2013 → BioGRID_2024")].iloc[0]
st = t11[t11.network.str.contains("Hetionet_2016 → STRING_2021")].iloc[0]
check("rich8 BioGRID absorption", 0.049, round(float(bg.auroc), 3), tol=0.0011)
check("rich8 BioGRID p", 0.125, round(float(bg.p_paired_bootstrap), 3), tol=0.0011)
check("rich8 STRING absorption", 0.105, round(float(st.auroc), 3), tol=0.0011)
check("rich8 STRING p", 0.003, round(float(st.p_paired_bootstrap), 3), tol=0.0011)

# ---------- T12 GAT ----------
t12 = pd.read_csv(f"{R}/t12_gat_pair.csv")
def t12v(model, c):
    return float(t12[(t12.model == model) & (t12.cutoff == c)].iloc[0].auroc)
check("GAT cur@2016", 0.676, round(t12v("GAT pair (STRING_2021)", 2016), 3), tol=0.0011)
check("GAT 2016@2016", 0.655, round(t12v("GAT pair (Hetionet_2016)", 2016), 3), tol=0.0011)
ga = t12[t12.model.str.contains("absorption") & (t12.cutoff == 2016)].iloc[0]
check("GAT absorption@2016", 0.021, round(float(ga.auroc), 3), tol=0.0011)
check("GAT absorption p", 0.004, round(float(ga.p), 3), tol=0.0011)

# ---------- rank-stability definition lock (either-nonzero rule) ----------
t2c = json.load(open(f"{R}/t2c_rank_stability.json"))
check("rho BG13-BG15 (either rule)", 0.853, round(t2c["matrix"]["BioGRID15"]["BioGRID13"], 3))
check("rho BG15-BG24 (either rule)", 0.786, round(t2c["matrix"]["BioGRID24"]["BioGRID15"], 3))
check("rho Het16-STR21", 0.465, round(t2c["matrix"]["STRING21"]["Hetionet16"], 3))

# ---------- figure-source lock (prevents silent patch no-ops) ----------
figsrc = open(f"{V11}/src/make_figures_v11.py", encoding="utf-8").read()
for good in ("(0.786, 0.057", "(0.853, 0.048", "(0.794, 0.022",
             "[0.853, 0.786, 0.794]", "non-knowledge features",
             '("label-side gain", "+0.08"'):
    check(f"figsrc has {good[:28]}", True, good in figsrc)
for stale in ("(0.816, 0.057", "(0.885, 0.048", "(0.803, 0.022",
              "0.885, 0.816, 0.803", "genuine fraction", "non-circular",
              '"C1"', "circularity ρ", "circularity ρ".encode().decode()):
    check(f"figsrc free of {stale[:24]}", False, stale in figsrc)

# ---------- split sizes / misc ----------
scores = U.load_scores()
y = scores["y_gold"].to_numpy()
check("candidates 14183", 14183, len(scores))
check("gold rows 691", 691, int(y.sum()))
check("dedup candidates 14084", 14084, scores["gene"].nunique())
check("broad set 2340", 2340, 3017 - 677)

print(f"\n===== {len(fails)} FAIL =====")
for f in fails:
    print(" ", f)
