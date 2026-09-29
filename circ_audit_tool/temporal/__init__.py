"""
temporal — the temporal-evaluation instruments of the circ-audit package
========================================================================
 dating         evidence dating pipeline (Eq 1)
 snapshots      snapshot rebuilders for archived releases (Eq 3-6)
 rank_stability vintage rank stability and absorption (Eq 7-9)
"""

from .dating import first_evidence_year, cutoff_split
from .rank_stability import absorption, auroc_pairwise, rank_stability
from .snapshots import (ALPHA, bfs_proximity, build_snapshot_features,
                        pair_transform, personalised_pagerank,
                        rank_normalised_degree)

__all__ = [
    "first_evidence_year", "cutoff_split",
    "rank_normalised_degree", "personalised_pagerank", "bfs_proximity",
    "build_snapshot_features", "pair_transform", "ALPHA",
    "auroc_pairwise", "rank_stability", "absorption",
]
