"""
t8_biogrid_dating.py — multi-source dating sensitivity, stage 1
================================================================
The paper's evidence dating resolves SynLethDB evidence rows to PubMed years,
while the gold standard itself is defined from BioGRID and HumanSL.  A gene
whose earliest record lives only in BioGRID would be dated late by the
shipped split, exaggerating prospective scores.  This stage extracts an
independent dating channel, namely human genetic interactions in the local
BioGRID 4.4.249 archive, resolves their publication years, and quantifies how
the merged (two-source) dating would move genes across the 2012/2014/2016
training/test boundary.

Outputs: results/t8_biogrid_dating.csv (per-gene years + flags),
         results/t8_redating_stats.json
"""
import io
import json
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U

BG_ZIP = "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/data/BIOGRID-ORGANISM-4.4.249.tab2.zip"
BG_MEMBER = "BIOGRID-ORGANISM-Homo_sapiens-4.4.249.tab2.txt"
V10 = U.V10
V11 = U.V11
SGI_SYSTEMS = {"Synthetic Genetic Interaction", "Synthetic Growth Defect",
               "Synthetic Lethality", "Dosage Rescue", "Dosage Lethality",
               "Phenotypic Suppression", "Phenotypic Enhancement",
               "Suppression", "Enhancement"}
# strict subset = the systems BioGRID files under genetic interaction screens
STRICT_SYSTEMS = {"Synthetic Genetic Interaction", "Synthetic Lethality",
                  "Synthetic Growth Defect"}


def extract_gene_pmids():
    """Stream the human tab2 and collect gene -> set(pubmed ids) for SGI rows."""
    zf = zipfile.ZipFile(BG_ZIP)
    with zf.open(BG_MEMBER) as fh:
        header = fh.readline().decode("utf-8").rstrip("\n").split("\t")
        idx = {name.lstrip("#"): i for i, name in enumerate(header)}
        need = ["Official Symbol Interactor A", "Official Symbol Interactor B",
                "Experimental System", "Pubmed ID", "Experimental System Type"]
        for n in need:
            if n not in idx:
                raise KeyError(f"column {n} missing; header={header}")
        gene_pm = {}
        n_rows = n_sgi = 0
        types = {}
        for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            f = raw.rstrip("\n").split("\t")
            n_rows += 1
            if f[idx["Experimental System"]] not in STRICT_SYSTEMS:
                continue
            n_sgi += 1
            types[f[idx["Experimental System Type"]]] = types.get(
                f[idx["Experimental System Type"]], 0) + 1
            pmids = {p for p in f[idx["Pubmed ID"]].split("|")
                     if p.isdigit()}
            for g in (f[idx["Official Symbol Interactor A"]], f[idx["Official Symbol Interactor B"]]):
                if g == "-" or not pmids:
                    continue
                gene_pm.setdefault(g, set()).update(pmids)
        print(f"biogrid rows={n_rows:,} SGI rows={n_sgi:,} genes={len(gene_pm):,}")
        print("interaction types:", dict(sorted(types.items(), key=lambda kv: -kv[1])))
        return gene_pm


def fetch_missing_years(pmids, cache):
    """Resolve publication years via NCBI esummary in batches of 200."""
    missing = [p for p in pmids if p not in cache]
    print(f"pmids total={len(pmids)} cached={len(pmids)-len(missing)} "
          f"to_fetch={len(missing)}")
    for i in range(0, len(missing), 200):
        batch = missing[i:i + 200]
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
               f"?db=pubmed&id={','.join(batch)}&retmode=json")
        for attempt in range(3):
            r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time",
                                "60", url], capture_output=True, text=True)
            try:
                d = json.loads(r.stdout)["result"]
                break
            except Exception:
                time.sleep(3 + attempt * 3)
        else:
            print(f"  batch {i}: FAILED, skipping")
            continue
        for uid in d.get("uids", []):
            pubdate = d[uid].get("pubdate", "")
            year = int("".join(c for c in pubdate[:4] if c.isdigit()) or 0)
            if year:
                cache[uid] = year
        if (i // 200) % 5 == 0:
            print(f"  fetched {min(i+200, len(missing))}/{len(missing)}")
        time.sleep(0.4)
    return cache


def main():
    scores = U.load_scores()
    y = scores["y_gold"].to_numpy()
    gold_rows = np.where(y == 1)[0]
    gold_genes = set(scores["gene"].to_numpy()[gold_rows])

    gene_pm = extract_gene_pmids()

    # existing SynLethDB dating
    split = pd.read_csv(f"{V10}/data/gold_temporal_split.csv")
    sl_year = dict(zip(split["gene"], split["first_year_experimental"]))

    cache_df = pd.read_csv(f"{V10}/data/pubmed_years.csv", dtype={"pmid": str})
    cache = {str(r.pmid): int(r.year) for r in cache_df.itertuples()
             if pd.notna(r.year)}
    involved = set().union(*[gene_pm[g] for g in gold_genes if g in gene_pm])
    print(f"gold genes with biogrid SGI rows: "
          f"{sum(1 for g in gold_genes if g in gene_pm)}/{len(gold_genes)}")
    cache = fetch_missing_years(involved, cache)
    pd.DataFrame({"pmid": list(cache), "year": [cache[p] for p in cache]}
                 ).to_csv(f"{V10}/data/pubmed_years.csv", index=False)

    bg_year = {}
    for g in gold_genes:
        if g not in gene_pm:
            continue
        ys = [cache[p] for p in gene_pm[g] if p in cache and cache[p] > 1900]
        if ys:
            bg_year[g] = min(ys)

    rows = []
    for g in sorted(gold_genes):
        sy = sl_year.get(g, np.nan)
        by = bg_year.get(g, np.nan)
        mv = np.nanmin([sy, by]) if pd.notna(sy) or pd.notna(by) else np.nan
        rows.append({"gene": g, "synlethdb_year": sy, "biogrid_year": by,
                     "merged_year": mv,
                     "earlier_source": ("biogrid" if pd.notna(by) and
                                        (pd.isna(sy) or by < sy) else
                                        "synlethdb" if pd.notna(sy) else "")})
    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/t8_biogrid_dating.csv", index=False)

    stats = {"n_gold_genes": int(len(gold_genes)),
             "n_with_biogrid_sgi": int(df["biogrid_year"].notna().sum()),
             "n_biogrid_earlier": int((df["earlier_source"] == "biogrid").sum()),
             "pmid_cache_size": len(cache)}
    # per-cutoff movement under merged dating
    per_gene = dict(zip(df["gene"], df["merged_year"]))
    for c in (2012, 2014, 2016):
        moved = []
        for r in df.itertuples():
            sy, mv = r.synlethdb_year, r.merged_year
            if pd.isna(mv):
                continue
            was_test = pd.notna(sy) and sy > c
            now_test = mv > c
            if was_test and not now_test:
                moved.append(r.gene)
        stats[f"moved_test_to_train @{c}"] = len(moved)
        stats[f"moved_genes @{c}"] = moved[:20]
    with open(f"{V11}/results/t8_redating_stats.json", "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items()
                      if not str(k).startswith("moved_genes")}, indent=2))


if __name__ == "__main__":
    main()
