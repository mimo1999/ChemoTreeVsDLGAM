"""
M-GAM binarization + missingness augmentation (no external repo dependency).

A clean reimplementation of the per-fold pipeline introduced in Donnelly et al.,
"Interpretable Generalized Additive Models for Datasets with Missing Values"
(arXiv:2412.02646). The reference code in ``M-GAM-main/binarizer.py`` is
intertwined with paper-specific defaults (e.g. survival miss_vals like ``-7, -8,
-9``); this file extracts the same idea into two stateless functions:

    fit_quantile_cutoffs(X_train, numerical_cols, quantiles)
        Compute per-column quantile thresholds from observed train values.

    mgam_binarize(X, numerical_cols, cutoffs, categorical_levels=..., ...)
        Apply the cutoffs to produce a wide 0/1 matrix. Optional missingness
        intercepts and missingness x bin interactions follow the M-GAM paper.

Train fold calls ``fit_quantile_cutoffs`` then ``mgam_binarize``; val/test
folds reuse the train-derived cutoffs via the second call only. The returned
matrices have aligned columns by construction.

No dependency on the M-GAM repo. ``numpy + pandas`` only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: fit cutoffs (called once on the training fold)
# ---------------------------------------------------------------------------

def fit_quantile_cutoffs(
    X_train: pd.DataFrame,
    numerical_cols: Iterable[str],
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
) -> Dict[str, List[float]]:
    """Compute deduplicated per-column quantile cutoffs from observed values.

    Missing cells (NaN) are dropped before quantile estimation, matching
    M-GAM's behaviour. Returns a dict mapping column name to a sorted list of
    cutoff values (deduplicated so identical quantiles don't emit two
    redundant binary columns downstream).
    """
    cutoffs: Dict[str, List[float]] = {}
    qarr = np.asarray(list(quantiles), dtype=float)
    for col in numerical_cols:
        if col not in X_train.columns:
            continue
        vals = X_train[col].dropna()
        if vals.empty:
            continue
        cuts = sorted(set(float(v) for v in vals.quantile(qarr).tolist()))
        if cuts:
            cutoffs[col] = cuts
    return cutoffs


def fit_categorical_levels(
    X_train: pd.DataFrame,
    categorical_cols: Iterable[str],
) -> Dict[str, List]:
    """Record observed category levels per column (train-only) for dummy expansion."""
    levels: Dict[str, List] = {}
    for col in categorical_cols:
        if col not in X_train.columns:
            continue
        ls = [v for v in X_train[col].dropna().unique()]
        if ls:
            levels[col] = ls
    return levels


# ---------------------------------------------------------------------------
# Step 2: binarize (called for train AND val/test using the same cutoffs)
# ---------------------------------------------------------------------------

def mgam_binarize(
    X: pd.DataFrame,
    *,
    numerical_cols: Iterable[str],
    cutoffs: Dict[str, List[float]],
    categorical_cols: Optional[Iterable[str]] = None,
    categorical_levels: Optional[Dict[str, List]] = None,
    miss_vals: Optional[Sequence[float]] = None,
    specific_mi_intercept: bool = True,
    overall_mi_intercept: bool = False,
    specific_mi_ixn: bool = False,
    overall_mi_ixn: bool = False,
) -> pd.DataFrame:
    """Apply the M-GAM transformation to ``X`` using pre-fit cutoffs.

    Parameters
    ----------
    X : DataFrame
        Wide tabular matrix. Numerical columns can contain NaN (or values in
        ``miss_vals``) to indicate missingness.
    numerical_cols : iterable of str
        Columns that will be quantile-binarized.
    cutoffs : dict[str, list[float]]
        Per-column cutoffs (from ``fit_quantile_cutoffs`` on the train fold).
    categorical_cols, categorical_levels : optional
        If provided, each categorical column emits one binary ``<col> == v``
        dummy per training-observed level.
    miss_vals : sequence of float, optional
        Values that count as "missing". NaN is always treated as missing in
        addition to any explicit sentinels listed here.
    specific_mi_intercept : bool, default True
        Emit one binary indicator per (numerical column, missing-value token)
        pair that has at least one observation in X.
    overall_mi_intercept : bool, default False
        Emit a single ``<col> missing`` indicator that is the union across
        ``miss_vals``.
    specific_mi_ixn, overall_mi_ixn : bool, default False
        Emit missingness-by-cutoff interaction terms. These explode the
        column count (O(N^2) on numerical_cols); off by default.

    Returns
    -------
    DataFrame with columns
        ``<col> <= <v>`` for every (col, v in cutoffs[col])
        ``<col> == NaN`` and/or ``<col> == <miss_val>``  (if specific_mi_intercept)
        ``<col> missing``                                (if overall_mi_intercept)
        ``<col> == <level>``  for each categorical column level
        ``<col_outer> missing & <col_inner> <= v``       (if overall_mi_ixn)
        ``<col_outer>_missing_<m> & <col_inner> <= v``   (if specific_mi_ixn)
    """
    miss_vals = list(miss_vals) if miss_vals else []
    categorical_cols = list(categorical_cols) if categorical_cols else []
    categorical_levels = categorical_levels or {}

    n_rows = len(X)
    idx = X.index
    out: Dict[str, np.ndarray] = {}

    # Per-column missingness mask: True wherever the cell was NaN or in miss_vals
    def _missing_mask(col):
        s = X[col]
        m = s.isna().to_numpy()
        for v in miss_vals:
            if pd.notna(v):
                m = m | (s == v).to_numpy()
        return m

    # -------- main quantile-cutoff bins --------
    for col, cuts in cutoffs.items():
        if col not in X.columns:
            continue
        vals = X[col].to_numpy()
        for v in cuts:
            name = f"{col}__le__{v}"
            arr = np.zeros(n_rows, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                arr[np.where(vals <= v)[0]] = 1.0
            out[name] = arr

    # -------- per-(col, miss_val) intercept --------
    if specific_mi_intercept:
        for col in cutoffs.keys():
            if col not in X.columns:
                continue
            s = X[col]
            # NaN as a "missing value" is always considered
            for m_val in [*miss_vals, np.nan]:
                if pd.isna(m_val):
                    arr = s.isna().to_numpy().astype(np.float64)
                    name = f"{col}__eq__NaN"
                else:
                    arr = (s == m_val).to_numpy().astype(np.float64)
                    name = f"{col}__eq__{m_val}"
                if arr.sum() > 0:
                    out[name] = arr

    # -------- overall <col> missing intercept --------
    if overall_mi_intercept:
        for col in cutoffs.keys():
            if col not in X.columns:
                continue
            m = _missing_mask(col).astype(np.float64)
            if m.sum() > 0:
                out[f"{col}__missing"] = m

    # -------- categorical dummies --------
    for col in categorical_cols:
        if col not in X.columns:
            continue
        levels = categorical_levels.get(col, [])
        for lvl in levels:
            arr = (X[col] == lvl).to_numpy().astype(np.float64)
            out[f"{col}__eq__{lvl}"] = arr

    # -------- overall missingness x cutoff interactions --------
    if overall_mi_ixn:
        for c_outer in cutoffs.keys():
            if c_outer not in X.columns:
                continue
            outer_miss = _missing_mask(c_outer)
            if outer_miss.sum() == 0:
                continue
            for c_inner, cuts in cutoffs.items():
                if c_inner == c_outer or c_inner not in X.columns:
                    continue
                inner_vals = X[c_inner].to_numpy()
                for v in cuts:
                    arr = np.zeros(n_rows, dtype=np.float64)
                    with np.errstate(invalid="ignore"):
                        arr[outer_miss & (inner_vals <= v)] = 1.0
                    if arr.sum() > 0:
                        out[f"{c_outer}__missing_AND_{c_inner}__le__{v}"] = arr

    # -------- specific missingness-token x cutoff interactions --------
    if specific_mi_ixn:
        for c_outer in cutoffs.keys():
            if c_outer not in X.columns:
                continue
            for m_val in [*miss_vals, np.nan]:
                outer_miss = (
                    X[c_outer].isna().to_numpy()
                    if pd.isna(m_val)
                    else (X[c_outer] == m_val).to_numpy()
                )
                if outer_miss.sum() == 0:
                    continue
                tag = "NaN" if pd.isna(m_val) else str(m_val)
                for c_inner, cuts in cutoffs.items():
                    if c_inner == c_outer or c_inner not in X.columns:
                        continue
                    inner_vals = X[c_inner].to_numpy()
                    for v in cuts:
                        arr = np.zeros(n_rows, dtype=np.float64)
                        with np.errstate(invalid="ignore"):
                            arr[outer_miss & (inner_vals <= v)] = 1.0
                        if arr.sum() > 0:
                            out[f"{c_outer}__missing_{tag}_AND_{c_inner}__le__{v}"] = arr

    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------------------
# Convenience: one-shot helper that emulates the M-GAM Binarizer call style
# ---------------------------------------------------------------------------

def mgam_binarize_train_and_apply(
    X_train: pd.DataFrame,
    X_other: Optional[pd.DataFrame] = None,
    *,
    numerical_cols: Iterable[str],
    categorical_cols: Optional[Iterable[str]] = None,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    miss_vals: Optional[Sequence[float]] = None,
    specific_mi_intercept: bool = True,
    overall_mi_intercept: bool = False,
    specific_mi_ixn: bool = False,
    overall_mi_ixn: bool = False,
):
    """Fit cutoffs on X_train, then binarize X_train (and optionally X_other).

    Returns
    -------
    If X_other is None : binarized X_train
    Otherwise          : (binarized X_train, binarized X_other)
    Both frames share an identical column set.
    """
    num_cols = list(numerical_cols)
    cat_cols = list(categorical_cols) if categorical_cols else []
    cutoffs = fit_quantile_cutoffs(X_train, num_cols, quantiles=quantiles)
    cat_levels = fit_categorical_levels(X_train, cat_cols)

    kw = dict(
        numerical_cols=num_cols,
        cutoffs=cutoffs,
        categorical_cols=cat_cols,
        categorical_levels=cat_levels,
        miss_vals=miss_vals,
        specific_mi_intercept=specific_mi_intercept,
        overall_mi_intercept=overall_mi_intercept,
        specific_mi_ixn=specific_mi_ixn,
        overall_mi_ixn=overall_mi_ixn,
    )

    Xtr_bin = mgam_binarize(X_train, **kw)
    if X_other is None:
        return Xtr_bin
    Xo_bin = mgam_binarize(X_other, **kw)
    # Align columns: X_other may not have observed certain bin/missing patterns
    # that X_train did (or vice versa). Use X_train's columns as canonical.
    Xo_bin = Xo_bin.reindex(columns=Xtr_bin.columns, fill_value=0.0)
    return Xtr_bin, Xo_bin
