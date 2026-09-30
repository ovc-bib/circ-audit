"""
dating.py — Evidence dating pipeline
====================================
Dates every gold entry by the publication year of its first experimental
evidence, the definition the temporal evaluation is built on (Eq (1) of
the manuscript):

    y(g) = min over p in E_exp(g) of year(p)

Computational-prediction and text-mining source categories count as
non-experimental and are excluded from the minimum.
"""

from typing import Optional

import numpy as np
import pandas as pd

NON_EXPERIMENTAL = ("prediction", "text-mining", "text_mining", "computational")
_SOURCE_CANDIDATES = ("source", "source_category", "category", "evidence_type", "evidence")


def first_evidence_year(evidence: pd.DataFrame,
                        gene_col: str = "gene",
                        year_col: str = "year",
                        source_col: Optional[str] = None) -> pd.Series:
    """Eq (1): minimum publication year over a gene's experimental evidence rows.

    Parameters
    ----------
    evidence : DataFrame with one row per evidence record; a PubMed-resolved
        publication year per row is expected in ``year_col``.
    source_col : column whose values name the source category; rows whose
        category matches a non-experimental keyword are dropped. When None
        (default), a source column is auto-detected among the common names
        and used if present, so the experimental-evidence discipline of the
        temporal evaluation is the default behaviour.
    """
    rows = evidence.dropna(subset=[gene_col, year_col]).copy()
    if source_col is None:
        source_col = next(
            (c for c in _SOURCE_CANDIDATES if c in rows.columns), None)
    if source_col is not None and source_col in rows.columns:
        mask = ~rows[source_col].astype(str).str.lower().apply(
            lambda s: any(k in s for k in NON_EXPERIMENTAL))
        rows = rows[mask]
    years = rows.groupby(gene_col)[year_col].min()
    return years.sort_index().rename("first_evidence_year")


def cutoff_split(years: pd.Series, cutoff: int) -> pd.Series:
    """Split a dated gold standard into training (<= cutoff) and test (> cutoff)."""
    return pd.Series(
        np.where(years <= cutoff, "train", "test"),
        index=years.index, name=f"split@{cutoff}")
