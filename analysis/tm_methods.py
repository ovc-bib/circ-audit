"""
tm_methods.py — third-party methods adapted for time-split retraining
======================================================================
Adapted from v10_bioinformatics/src/e3_regenerate_scores.py (same seeds,
same hyperparameters) with two changes required by the time-machine design:

  1. label-dependent parts (SVM training pairs, gold centroids) are split
     from label-free parts (NMF factors, DeepWalk-style embeddings) so the
     expensive label-free computation runs once and only the cheap part
     is repeated per cutoff;
  2. negative sampling excludes ALL strict gold (not just the training-era
     gold), per DESIGN.md §5 rule 1.
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr  # noqa: F401  (kept for parity with e3)
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = "E:/崔雷/博士/设计/ovc_project"
V10 = f"{ROOT}/circ_aware_bib/v10_bioinformatics"
V11 = f"{ROOT}/circ_aware_bib/v11_timemachine"
SEED = 42


# ------------------------------------------------------------------
# SINaTRA — network topology features + SVM (retrained per cutoff)
# ------------------------------------------------------------------
def _saarf_feature_table(saarf_df, candidate_genes):
    feature_cols = [c for c in saarf_df.columns
                    if c not in ("gene",) and saarf_df[c].dtype in
                    ("float32", "float64", "int32", "int64")]
    gene_to_saarf = {g: i for i, g in enumerate(saarf_df["gene"])}
    table = {}
    for gene in candidate_genes:
        if gene in gene_to_saarf:
            table[gene] = np.nan_to_num(
                saarf_df.iloc[gene_to_saarf[gene]][feature_cols]
                .values.astype(float), nan=0.0)
    return table, feature_cols


# saarf columns that COUNT the SL database itself — for the gene-level task
# these restate the label (gold membership = >=2 experimental SL evidence rows),
# so they carry post-discovery information regardless of any label split.
SLDB_COLUMNS = ["sl_pair_count", "is_known_sl_partner"]

# saarf columns derived from curated KNOWLEDGE (literature mining, KG, CTD,
# pathway databases) — these track how studied a gene is today and therefore
# blend study bias with retro-absorption of post-discovery curation.
KNOWLEDGE_COLUMNS = ["sl_pair_count", "is_known_sl_partner", "lit_score",
                     "kg_score", "kg_source_score", "kg_non_hetionet_score",
                     "ctd_score", "pathway_score",
                     "pathway_enriched_score", "pathway_count"]


def sinatra_retrain(candidate_genes, saarf_df, context_genes,
                    train_gold_set, neg_exclude, seed=SEED,
                    exclude_cols=None):
    """SINaTRA retrained with only <=cutoff gold as positives.

    neg_exclude: genes that must never be sampled as negatives
    (all strict gold, cutoff-independent).
    exclude_cols: input columns to drop (e.g. SLDB_COLUMNS for the
    label-leakage-free variant).
    """
    if exclude_cols:
        saarf_df = saarf_df.drop(columns=[c for c in exclude_cols
                                          if c in saarf_df.columns])
    gene_features, _ = _saarf_feature_table(saarf_df, candidate_genes)

    gold_list = sorted(train_gold_set & set(candidate_genes))
    context_list = [g for g in context_genes if g in gene_features]

    pos_pairs = [(g, c) for g in gold_list for c in context_list if g != c]
    rng = np.random.RandomState(seed)
    non_gold = [g for g in candidate_genes
                if g not in neg_exclude and g in gene_features]
    n_neg = min(len(pos_pairs) * 5, len(non_gold) * len(context_list))
    neg_pairs = [(rng.choice(non_gold), rng.choice(context_list))
                 for _ in range(n_neg)]

    def pair_feat(g1, g2):
        f1, f2 = gene_features.get(g1), gene_features.get(g2)
        if f1 is None or f2 is None:
            return None
        cos = np.dot(f1, f2) / max(np.linalg.norm(f1) * np.linalg.norm(f2), 1e-8)
        return np.concatenate([f1 + f2, np.abs(f1 - f2), f1 * f2, [cos]])

    X_pos = [pf for pf in (pair_feat(g, c) for g, c in pos_pairs) if pf is not None]
    X_neg = [pf for pf in (pair_feat(g, c) for g, c in neg_pairs) if pf is not None]
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = LinearSVC(dual="auto", random_state=seed, max_iter=5000)
    svm.fit(X_scaled, y)

    scores = np.zeros(len(candidate_genes))
    for i, gene in enumerate(candidate_genes):
        pair_feats = [pair_feat(gene, c) for c in context_list]
        pair_feats = [pf for pf in pair_feats if pf is not None]
        if pair_feats:
            X_g = scaler.transform(np.array(pair_feats))
            scores[i] = float(np.max(svm.decision_function(X_g)))
    return scores


# ------------------------------------------------------------------
# SL2MF — multi-view NMF (label-free fit + per-cutoff gold centroid)
# ------------------------------------------------------------------
def sl2mf_fit(candidate_genes, merged_df, saarf_df, seed=SEED):
    """Label-free NMF factors on CRISPR / coess / PPI views."""
    gene_to_merged = {g: i for i, g in enumerate(merged_df["gene"])}
    gene_to_saarf = {g: i for i, g in enumerate(saarf_df["gene"])}
    comp_cols = [c for c in merged_df.columns if c.startswith("comp_")]
    coess_cols = [c for c in merged_df.columns if c.startswith("coess_")]
    ppi_cols = [c for c in saarf_df.columns
                if c.startswith("ppi_") or c.startswith("ppr_")]

    views = []
    for cols, df_src, g2i in [(comp_cols, merged_df, gene_to_merged),
                              (coess_cols, merged_df, gene_to_merged),
                              (ppi_cols, saarf_df, gene_to_saarf)]:
        if not cols:
            continue
        mat = np.zeros((len(candidate_genes), len(cols)), dtype=np.float32)
        for i, gene in enumerate(candidate_genes):
            if gene in g2i:
                mat[i] = np.nan_to_num(
                    df_src.iloc[g2i[gene]][cols].values.astype(float), nan=0.0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = mat / norms
        shifted = normalized - normalized.min() + 1e-8
        views.append(shifted)
    concatenated = np.hstack(views)
    n_factors = min(32, len(candidate_genes) - 1)
    model = NMF(n_components=n_factors, init="random", random_state=seed,
                max_iter=500)
    W = model.fit_transform(concatenated)
    return W


def sl2mf_score(W, gold_row_indices):
    """1/(1+distance to gold centroid) — centroid from training-era gold only."""
    centroid = W[gold_row_indices].mean(axis=0)
    return 1.0 / (1.0 + np.linalg.norm(W - centroid, axis=1))


# ------------------------------------------------------------------
# KG4SL — graph embedding (label-free, cached) + per-cutoff centroid
# ------------------------------------------------------------------
def kg4sl_embed(candidate_genes, hetionet_edges, hetionet_nodes,
                cache_path=None, seed=SEED):
    """DeepWalk-style embedding on the Hetionet gene-gene graph (label-free)."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return dict(zip(z["genes"].tolist(), z["emb"])), z["genes"].tolist()

    gene_nodes = hetionet_nodes[hetionet_nodes["kind"] == "Gene"]
    hetio_id_to_symbol = dict(zip(gene_nodes["id"], gene_nodes["name"]))
    gene_gene = hetionet_edges[
        hetionet_edges["source"].str.startswith("Gene") &
        hetionet_edges["target"].str.startswith("Gene")
    ]
    candidate_set = set(candidate_genes)
    adj_dict = {}
    for s, t in zip(gene_gene["source"], gene_gene["target"]):
        a = hetio_id_to_symbol.get(s)
        b = hetio_id_to_symbol.get(t)
        if a is None or b is None:
            continue
        if a in candidate_set or b in candidate_set:
            adj_dict.setdefault(a, set()).add(b)
            adj_dict.setdefault(b, set()).add(a)

    relevant = set(candidate_set)
    for g in candidate_set:
        if g in adj_dict:
            relevant.update(adj_dict[g])
    gene_list = sorted(relevant)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    N = len(gene_list)

    row_idx, col_idx = [], []
    for g, nbrs in adj_dict.items():
        if g in gene_to_idx:
            for n in nbrs:
                if n in gene_to_idx:
                    row_idx.append(gene_to_idx[g])
                    col_idx.append(gene_to_idx[n])
    data = np.ones(len(row_idx), dtype=np.float64)
    adj = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    print(f"  KG4SL graph: {N} nodes, {adj.nnz} edges")

    rng = np.random.RandomState(seed)
    neighbors_list = []
    for i in range(N):
        nbrs = adj.getrow(i).indices.tolist()
        neighbors_list.append(nbrs if nbrs else [i])

    n_walks, walk_length = 10, 40
    walks = []
    for _ in range(n_walks):
        for start in range(N):
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                current = rng.choice(neighbors_list[current])
                walk.append(current)
            walks.append(walk)

    embed_dim = 64
    cooccur = np.zeros((N, N), dtype=np.float32)
    window = 5
    for walk in walks:
        for i, node in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    cooccur[node, walk[j]] += 1.0 / abs(i - j)

    svd = TruncatedSVD(n_components=min(embed_dim, N - 1), random_state=seed)
    embeddings = svd.fit_transform(cooccur)

    emb = {g: embeddings[gene_to_idx[g]] for g in candidate_genes
           if g in gene_to_idx}
    if cache_path:
        np.savez_compressed(cache_path, emb=np.array(list(emb.values()),
                               dtype=object),
                            genes=np.array(list(emb.keys())))
    return emb, gene_list


def kg4sl_score(emb, candidate_genes, gold_genes):
    """Cosine to the gold centroid — centroid from training-era gold only."""
    gold_vecs = [emb[g] for g in gold_genes if g in emb]
    if not gold_vecs:
        return np.random.RandomState(SEED).random(len(candidate_genes))
    centroid = np.mean(gold_vecs, axis=0)
    cn = np.linalg.norm(centroid)
    scores = np.zeros(len(candidate_genes))
    for i, g in enumerate(candidate_genes):
        v = emb.get(g)
        if v is not None and cn > 0:
            vn = np.linalg.norm(v)
            if vn > 0:
                scores[i] = float(np.dot(v, centroid) / (vn * cn))
    return scores
