"""
t13_gcatsl_inputs.py — T13: GCATSL original-pipeline temporal inputs (R1 revision)
====================================================================================
Builds the input files consumed by the authors' original GCATSL TensorFlow
implementation (github.com/lichenbiostat/GCATSL, master, MIT licence,
TensorFlow 1.13.1) under the temporal pair protocol of T10:

  adj.txt               all evidence-dated gold pairs, 1-based gene indices,
                        label 1, rows sorted by (year, a, b)
  test_arr_0.txt        row indices of post-cutoff pairs (the original main.py
                        reads a supplied test_arr file and trains on the rest,
                        so the fold loop degenerates to the temporal split)
  feature_1.txt         128-d node2vec embedding of the vintage PPI graph
                        "hetionet2016": Hetionet GiG (2015-16 curation)
                        "string2021"  : STRING v11.5, combined score >= 400
  feature_2.txt         128-d node2vec embedding of the vintage GO BP
                        co-annotation graph (GOA dated <= 2016-12-31 or the
                        current release), edges = top-50 Jaccard neighbours
  feature_3.txt         same for GO CC

The authors' released example supplies 6375 x 128 node-feature matrices for
the same three sources (PPI, GO BP, GO CC) built on undated SynLethDB-era
data; temporal evaluation requires rebuilding them at each knowledge vintage,
so each feature graph is embedded here with seeded node2vec (PyG, seed 42,
128 dimensions, matching the released example's dimensionality). The model
code, hyperparameters, loss and training loop are used unmodified.

Universe: all genes spanning the dated gold pairs (kept in full) plus a
seed-42 subsample of negative-pool genes, capped at N_MAX = 6,400, matching
the original implementation's dense-matrix scale (its released example uses
6,375 nodes; the published GCATSL benchmark of Feng et al. likewise sets
num_nodes to its own gene count). The universe is frozen once and shared by
every cutoff, vintage and feature file.

Output: external/runs/<cutoff>_<vintage>/{adj.txt,feature_*.txt,test_arr_0.txt}
        external/runs/universe_genes.csv, test_pools_universe.csv
        results/t13_universe.json (manifest)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t10_pair_level import export_pairs
from t2_vintage_chain import graph_hetionet, graph_string2021
from t3_go_vintage import goa_terms

V11 = U.V11
EXT = f"{V11}/external"
RUNS = f"{EXT}/runs"
N_MAX = 6400
SEED = 42
EMB_DIM = 128
GO_TERM_MAX_GENES = 300
GO_TOPK = 50
CUTOFFS = U.CUTOFFS
VINTAGE_PPI = {"hetionet2016": "hetionet", "string2021": "string"}
VINTAGE_GOA = {"hetionet2016": "2016-12-31", "string2021": None}


def build_universe():
    """Gold-pair genes in full + seed-42 pool-gene subsample, capped at N_MAX."""
    scores = U.load_scores()
    cand = scores["gene"].tolist()
    cand_set = set(cand)
    posE = export_pairs(cand_set)
    pos_genes = sorted(set(posE["a"]) | set(posE["b"]))

    pools = pd.read_csv(f"{V11}/results/t10_pools.csv")
    rnd = pools[pools["pool"] == "random"]
    pool_genes = sorted((set(rnd["a"]) | set(rnd["b"])) - set(pos_genes))

    budget = max(N_MAX - len(pos_genes), 0)
    rng = np.random.default_rng(SEED)
    keep = sorted(rng.choice(len(pool_genes), budget, replace=False))
    pool_keep = [pool_genes[i] for i in keep]

    universe = sorted(pos_genes + pool_keep)
    print(f"universe: {len(universe)} genes "
          f"({len(pos_genes)} gold-pair + {len(pool_keep)} pool-sample of "
          f"{len(pool_genes)})")
    return universe, posE, rnd


def ppi_graph_edges(graph, universe):
    pos = {g: i for i, g in enumerate(universe)}
    edges = set()
    for g, nbrs in graph.items():
        if g not in pos:
            continue
        for nb in nbrs:
            if nb in pos and nb != g:
                edges.add((min(pos[g], pos[nb]), max(pos[g], pos[nb])))
    print(f"  PPI on universe: {len(edges)} undirected edges, "
          f"{len({v for e in edges for v in e})} genes covered")
    return sorted(edges)


def gosim_graph_edges(terms_by_gene, universe, aspect):
    """Top-K Jaccard co-annotation neighbours (one aspect), rare-term filter."""
    n = len(universe)
    pos = {g: i for i, g in enumerate(universe)}
    term_genes = {}
    for g, asp in terms_by_gene.items():
        if g not in pos:
            continue
        for t in asp.get(aspect, ()):
            term_genes.setdefault(t, []).append(pos[g])
    kept = {t: gs for t, gs in term_genes.items() if len(gs) <= GO_TERM_MAX_GENES}
    size = np.zeros(n, dtype=np.int64)
    indices, col_of = [], {}
    indptr_list = [0]
    for ci, (t, gs) in enumerate(sorted(kept.items())):
        col_of[t] = ci
        for g in gs:
            size[g] += 1
    rows = []
    for g in universe:
        ts = [col_of[t] for t in terms_by_gene.get(g, {}).get(aspect, ())
              if t in col_of]
        rows.append(sorted(set(ts)))
    for r in rows:
        indices.extend(r)
        indptr_list.append(len(indices))
    indptr = np.array(indptr_list, dtype=np.int64)
    T = len(col_of)
    G = sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(n, T))
    S = (G @ G.T).tocoo()
    union = size[S.row] + size[S.col] - S.data
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, S.data / union, 0.0)
    ok = jac > 0
    R, C, V = S.row[ok], S.col[ok], jac[ok]
    edges = set()
    for i, j, v in zip(R.tolist(), C.tolist(), V.tolist()):
        a, b = (i, j) if i < j else (j, i)
        if a != b:
            edges.add((a, b, v))
    # top-K per gene over the sparse similarity entries
    nbrs = {}
    for a, b, v in edges:
        nbrs.setdefault(a, []).append((v, b))
        nbrs.setdefault(b, []).append((v, a))
    top_edges = set()
    for g, lst in nbrs.items():
        lst.sort(reverse=True)
        for v, nb in lst[:GO_TOPK]:
            top_edges.add((min(g, nb), max(g, nb)))
    print(f"  GOsim {aspect}: {T} terms kept, {len(edges)} sim pairs, "
          f"top-{GO_TOPK} graph -> {len(top_edges)} edges")
    return sorted(top_edges)


def embed_edges(n, edges, tag):
    """128-d uniform-random-walk skip-gram embedding of an undirected graph
    (node2vec with p = q = 1, i.e. DeepWalk-equivalent), fully seeded, CPU.

    walks 10 per node, length 20, context window 5, 5 negative samples,
    6 epochs of minibatch SGNS (Adam, lr 0.03). torch-cluster is not
    available for this torch build, so the sampler is implemented here in
    vectorised numpy; with p = q = 1 it draws the same distribution as the
    reference sampler.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    if not edges:
        return np.zeros((n, EMB_DIM), dtype=np.float32)
    src = np.array([e[0] for e in edges] + [e[1] for e in edges],
                   dtype=np.int64)
    dst = np.array([e[1] for e in edges] + [e[0] for e in edges],
                   dtype=np.int64)
    order = np.argsort(src, kind="stable")
    src_s, dst_s = src[order], dst[order]
    starts = np.searchsorted(src_s, np.arange(n))
    deg = np.bincount(src, minlength=n)

    rng = np.random.default_rng(SEED)
    PER, LEN = 10, 20
    walks = np.full((n * PER, LEN), -1, dtype=np.int64)
    cur = np.tile(np.arange(n, dtype=np.int64), PER)
    walks[:, 0] = cur
    stopped = deg[cur] == 0
    for step in range(1, LEN):
        idx = np.where(~stopped)[0]
        if len(idx) == 0:
            break
        c = cur[idx]
        d = deg[c]
        draw = (rng.random(len(idx)) * d).astype(np.int64)
        cur[idx] = dst_s[starts[c] + draw]
        walks[idx, step] = cur[idx]
        newly = idx[deg[cur[idx]] == 0]
        stopped[newly] = True

    # (center, context) pairs within window 5 on each walk
    WIN = 5
    ci, cj = [], []
    for off in range(1, WIN + 1):
        a = walks[:, :-off].reshape(-1)
        b = walks[:, off:].reshape(-1)
        ok = (a >= 0) & (b >= 0)
        ci.append(a[ok])
        cj.append(b[ok])
    centers = np.concatenate(ci + cj)
    contexts = np.concatenate(cj + ci)
    P = len(centers)
    print(f"    [{tag}] walks done, {P} (center,context) pairs")

    torch.manual_seed(SEED)
    emb_c = nn.Embedding(n, EMB_DIM)
    nn.init.normal_(emb_c.weight, std=1.0 / np.sqrt(EMB_DIM))
    opt = torch.optim.Adam(emb_c.parameters(), lr=0.03)
    tc = torch.tensor(centers)
    tx = torch.tensor(contexts)
    BS = 8192
    NEGE = 5
    t = time.time()
    steps_per_epoch = (P + BS - 1) // BS
    g = torch.Generator().manual_seed(SEED)
    for ep in range(6):
        perm = torch.randperm(P, generator=g)
        tot = 0.0
        for s in range(steps_per_epoch):
            sel = perm[s * BS:(s + 1) * BS]
            c = tc[sel]
            x = tx[sel]
            neg = torch.randint(0, n, (len(sel), NEGE), generator=g)
            vc = emb_c(c)                             # B x D
            vx = emb_c(x)                             # B x D
            vn = emb_c(neg)                           # B x E x D
            pos_l = F.logsigmoid((vc * vx).sum(-1))
            neg_l = F.logsigmoid(-(vc.unsqueeze(1) * vn).sum(-1)).sum(-1)
            loss = -(pos_l + neg_l).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        print(f"    [{tag}] epoch {ep + 1} loss {tot / P:.4f} "
              f"({time.time() - t:.0f}s)")

    emb = emb_c.weight.detach().numpy().astype(np.float32)
    # deterministic sign convention
    r0 = emb[int(np.argmax(np.abs(emb[:, 0])))]
    if r0[0] < 0:
        emb = -emb
    return emb


def main():
    t0 = time.time()
    os.makedirs(RUNS, exist_ok=True)
    universe, posE, rnd = build_universe()
    n = len(universe)
    universe_set = set(universe)
    idx = {g: i + 1 for i, g in enumerate(universe)}

    df = posE.copy()
    df["ia"] = [idx[a] for a in df["a"]]
    df["ib"] = [idx[b] for b in df["b"]]
    df = df.sort_values(["year", "a", "b"]).reset_index(drop=True)
    adj = np.hstack([df[["ia", "ib"]].to_numpy(),
                     np.ones((len(df), 1), dtype=int)])
    years = df["year"].to_numpy()

    pd.DataFrame({"index": np.arange(1, n + 1), "gene": universe,
                  "in_gold_pairs": [g in set(df["a"]) | set(df["b"])
                                    for g in universe]}
                 ).to_csv(f"{RUNS}/universe_genes.csv", index=False)

    pools_in = rnd[(rnd["a"].isin(universe_set)) & (rnd["b"].isin(universe_set))]
    print(f"pool pairs inside universe: {len(pools_in)} "
          f"{pools_in.groupby('split').size().to_dict()}")
    pools_in.to_csv(f"{RUNS}/test_pools_universe.csv", index=False)

    print("building feature graphs ...")
    ppi_cur = ppi_graph_edges(graph_string2021(), universe)
    ppi_2016 = ppi_graph_edges(graph_hetionet(), universe)
    goa_2016 = goa_terms(date_limit="2016-12-31")
    goa_cur = goa_terms(date_limit=None)
    gosim = {"BP": {"hetionet2016": gosim_graph_edges(goa_2016, universe, "BP"),
                    "string2021": gosim_graph_edges(goa_cur, universe, "BP")},
             "CC": {"hetionet2016": gosim_graph_edges(goa_2016, universe, "CC"),
                    "string2021": gosim_graph_edges(goa_cur, universe, "CC")}}

    ppi_edges = {"hetionet2016": ppi_2016, "string2021": ppi_cur}
    feats = {}
    for vname in ("hetionet2016", "string2021"):
        feats[vname] = [
            embed_edges(n, ppi_edges[vname], f"ppi_{vname}"),
            embed_edges(n, gosim["BP"][vname], f"bp_{vname}"),
            embed_edges(n, gosim["CC"][vname], f"cc_{vname}"),
        ]

    manifest = {"n_max": N_MAX, "seed": SEED, "emb_dim": EMB_DIM,
                "go_term_max_genes": GO_TERM_MAX_GENES, "go_topk": GO_TOPK,
                "n_universe": n, "ppi_edges": {k: len(v) for k, v in
                                               ppi_edges.items()},
                "runs": {}}

    for vname, mats in feats.items():
        for c in CUTOFFS:
            d = f"{RUNS}/{c}_{vname}"
            os.makedirs(d, exist_ok=True)
            np.savetxt(f"{d}/adj.txt", adj, fmt="%d")
            te = np.where(years > c)[0]
            np.savetxt(f"{d}/test_arr_0.txt", te, fmt="%d")
            for fi, M in enumerate(mats, start=1):
                np.savetxt(f"{d}/feature_{fi}.txt", M, fmt="%.6g")
            manifest["runs"][f"{c}_{vname}"] = {
                "n_train_pairs": int((years <= c).sum()),
                "n_test_pairs": int(len(te))}
            print(f"  {c}_{vname}: train {int((years <= c).sum())} / "
                  f"test {len(te)}")

    json.dump(manifest, open(f"{V11}/results/t13_universe.json", "w"),
              indent=1)
    print(f"T13 inputs done in {(time.time() - t0) / 60:.1f} min -> {RUNS}")


if __name__ == "__main__":
    main()
