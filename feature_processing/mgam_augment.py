"""
M-GAM-style missingness augmentation (no external repo dependency).

Inspired by Donnelly et al. — "Interpretable Generalized Additive Models for
Datasets with Missing Values" (https://arxiv.org/abs/2412.02646). The full
M-GAM transformation consists of three parts:

    (a) per-feature quantile-cutoff binarization
    (b) per-feature missingness intercept indicators
    (c) per-(missing-feature, other-feature-bin) missingness x bin interactions

This module implements **only part (b)** — the missingness indicators —
because

    * (a) the quantile-cutoff binarization is not needed when M-GAM is layered
      on top of an already-aggregated representation like ``feat_type=A``
      (the A features are themselves shape-friendly numerics);
    * (c) the interaction terms scale O(features^2) and would explode the
      column count on wide tabular cohorts.

The single public function ``add_missingness_indicators`` takes a feature
matrix and a parallel boolean missingness mask, and appends one float-valued
``_missing`` flag per target column. No imports outside numpy / pandas.

The Binarizer logic in ``M-GAM-main/binarizer.py`` was used as the
reference; this file is a clean re-implementation with no code carried over
verbatim and no dependency on that repo.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_missingness_indicators(
    X: pd.DataFrame,
    *,
    target_cols: Iterable[str],
    missing_mask: pd.DataFrame,
    suffix: str = "_missing",
) -> pd.DataFrame:
    """Append one 0/1 missingness indicator per ``target_cols`` entry to X.

    Parameters
    ----------
    X : DataFrame
        Base feature matrix.  ``X.index`` is preserved on the output.
    target_cols : iterable of str
        Names of columns whose missingness should be encoded as an extra
        binary column. Need not be a subset of ``X.columns`` — only need to
        align with ``missing_mask``.
    missing_mask : DataFrame
        Same row index as ``X``. Boolean (or 0/1) matrix indicating whether
        each (row, target_col) was missing in the source data. Must contain
        every name in ``target_cols``.
    suffix : str, default "_missing"
        Appended to each target column name to form indicator column names.

    Returns
    -------
    DataFrame
        Concatenation of ``X`` and the new indicator columns (each cast to
        float64 so downstream scalers / models don't have to handle bool).
    """
    target_cols = list(target_cols)
    if not target_cols:
        return X.copy()

    if not isinstance(missing_mask, pd.DataFrame):
        raise TypeError("missing_mask must be a pandas DataFrame")
    if not X.index.equals(missing_mask.index):
        raise ValueError(
            "X and missing_mask must share the same row index "
            f"(X: {len(X)} rows, mask: {len(missing_mask)} rows)"
        )

    missing_target = set(target_cols) - set(missing_mask.columns)
    if missing_target:
        raise KeyError(
            f"missing_mask lacks columns: {sorted(missing_target)[:3]}"
            + ("..." if len(missing_target) > 3 else "")
        )

    blocks = {
        f"{col}{suffix}": missing_mask[col].astype(bool).astype(np.float64).to_numpy()
        for col in target_cols
    }
    miss_df = pd.DataFrame(blocks, index=X.index)
    return pd.concat([X, miss_df], axis=1)


def missingness_mask_from_observed_fraction(
    X: pd.DataFrame,
    observed_fraction_suffix: str = "_observed_fraction",
    target_suffix: str = "_mean",
) -> pd.DataFrame:
    """Build a missingness mask from feat_type=A's ``*_observed_fraction`` cols.

    For each ``<prefix>_observed_fraction`` column in ``X``, emit a boolean
    column named ``<prefix>{target_suffix}`` set to True where the original
    bin had zero observations (i.e. ``observed_fraction == 0``).

    The output mask is indexed by ``X.index`` and has one column per
    ``*_observed_fraction`` source column, named to match the corresponding
    ``*_mean`` feature so it can be fed directly into
    ``add_missingness_indicators(..., target_cols=mean_cols, missing_mask=...)``.
    """
    blocks = {}
    for col in X.columns:
        if not col.endswith(observed_fraction_suffix):
            continue
        prefix = col[: -len(observed_fraction_suffix)]
        target_name = f"{prefix}{target_suffix}"
        blocks[target_name] = (X[col] == 0.0).to_numpy()
    return pd.DataFrame(blocks, index=X.index)
