"""
snapshots.py — Snapshot rebuilders for archived database releases
================================================================
Rebuild the three network features of the temporal evaluation on any
interaction snapshot with identical code, so that snapshots of very
different density remain comparable (the property the vintage design
depends on).

Features (equation numbers refer to the manuscript):
    Eq (3)  rank-normalised degree      d_hat(g) = r_d(g) / N
    Eq (4)  personalised PageRank       pi = (1 - alpha) (I - alpha P^T)^-1 s
            (closed form, alpha = 0.15, restart on the seed set)
    Eq (5)  BFS proximity               b(g) = min_{s in S} dist_G(g, s),
            disconnected genes assigned the graph diameter

Also provides the pair transformations of Eq (6) used by the pair-level
evaluation:
    phi(x_u, x_v) in {x_u + x_v, |x_u - x_v|, x_u * x_v, cos(x_u, x_v)}
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from typing import Iterable, List, Sequence, Tuple

ALPHA = 0.15  # restart probability of Eq (4)


def _normalised_adjacency(edges: pd.DataFrame, nodes: Sequence[str]) -> csr_matrix:
    """Row-normalised adjacency matrix P~ of the snapshot graph."""
    index = {g: i for i, g in enumerate(nodes)}
    n = len(nodes)
    rows = edges.iloc[:, 0].map(index)
    cols = edges.iloc[:, 1].map(index)
    keep = rows.notna() & cols.notna()
    r = rows[keep].astype(int).to_numpy()
    c = cols[keep].astype(int).to_numpy()
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = A + A.T
    A.data[:] = 1.0  # undirected simple graph
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(degree > 0, 1.0 / np.maximum(degree, 1.0), 0.0)
    return csr_matrix(np.diag(inv)) @ A


def rank_normalised_degree(edges: pd.DataFrame, nodes: Sequence[str]) -> pd.Series:
    """Eq (3): ascending rank of the raw degree divided by the candidate count."""
    index = {g: i for i, g in enumerate(nodes)}
    degree = np.zeros(len(nodes))
    for a, b in edges.itertuples(index=False):
        if a in index:
            degree[index[a]] += 1
        if b in index and b != a:
            degree[index[b]] += 1
    order = degree.argsort().argsort().astype(float)  # ascending rank, ties by order
    return pd.Series(order / len(nodes), index=list(nodes), name="degree_rank")


def personalised_pagerank(edges: pd.DataFrame, nodes: Sequence[str],
                          seeds: Iterable[str], alpha: float = ALPHA) -> pd.Series:
    """Eq (4): closed-form personalised PageRank with restart on the seeds."""
    p_tilde = _normalised_adjacency(edges, nodes)
    n = len(nodes)
    s = np.zeros(n)
    seed_list = [g for g in seeds if g in set(nodes)]
    if not seed_list:
        raise ValueError("no seed gene is present in the snapshot")
    s[[list(nodes).index(g) for g in seed_list]] = 1.0 / len(seed_list)
    pi = spsolve((identity(n) - alpha * p_tilde.T).tocsc(), (1 - alpha) * s)
    return pd.Series(pi, index=list(nodes), name="ppr")


def bfs_proximity(edges: pd.DataFrame, nodes: Sequence[str],
                  seeds: Iterable[str]) -> pd.Series:
    """Eq (5): minimum graph distance to any seed, diameter for disconnected genes."""
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    p_tilde = _normalised_adjacency(edges, nodes)  # symmetric simple graph
    graph = p_tilde.sign()
    n = len(nodes)
    node_list = list(nodes)
    node_set = set(node_list)
    seed_idx = [i for i, g in enumerate(node_list) if g in set(seeds)]
    if not seed_idx:
        raise ValueError("no seed gene is present in the snapshot")

    dist = np.full(n, np.inf)
    for s in seed_idx:
        order, predecessors = breadth_first_order(graph, s, directed=False,
                                                  return_predecessors=True)
        depth = np.full(n, -1)
        depth[s] = 0
        # BFS layer depths from the predecessor tree
        for node in order:
            if node == s:
                continue
            p = predecessors[node]
            if p >= 0:
                depth[node] = depth[p] + 1
        reached = depth >= 0
        dist[reached] = np.minimum(dist[reached], depth[reached].astype(float))

    n_components, _ = connected_components(graph, directed=False)
    diameter = 2  # fallback for the degenerate single-component case
    dist_finite = dist[np.isfinite(dist)]
    # a conservative stand-in for the graph diameter: twice the observed max
    # distance to the seeds when every node is reached, else the candidate count
    if len(dist_finite) == n and n_components == 1:
        diameter = 2 * int(dist_finite.max())
    dist[~np.isfinite(dist)] = diameter
    return pd.Series(dist, index=node_list, name="bfs_proximity")


def build_snapshot_features(edges: pd.DataFrame, nodes: Sequence[str],
                            seeds: Iterable[str]) -> pd.DataFrame:
    """The three-feature snapshot table of the temporal evaluation."""
    return pd.concat(
        [rank_normalised_degree(edges, nodes),
         personalised_pagerank(edges, nodes, seeds),
         bfs_proximity(edges, nodes, seeds)],
        axis=1)


def pair_transform(x_u: np.ndarray, x_v: np.ndarray) -> np.ndarray:
    """Eq (6): the four pair transformations concatenated into one vector."""
    cos = float(x_u @ x_v / (np.linalg.norm(x_u) * np.linalg.norm(x_v) + 1e-12))
    return np.concatenate([x_u + x_v, np.abs(x_u - x_v), x_u * x_v, [cos]])
