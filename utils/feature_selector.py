"""
Shared feature-selection helper used by the lightweight ebm / gaminet / gam
wrappers in this repo.

Pipeline:

    SelectKBest(score_func=mutual_info_classif, k=max_features)
        -> StandardScaler

Imputation is intentionally omitted: input data is expected to be
NaN-free after the upstream feature-generation step.

The class is stateless across calls; it carries the fitted selector
/ scaler instances so the same transformation can be applied to val / test
splits.

It is purely a wrapper module: any downstream model (EBM, GAMINET, GAM, LR,
...) can drop this in front of itself and behave consistently.
"""

import re
from functools import partial
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Column-name patterns used by RecencyFeatureSelector
# ---------------------------------------------------------------------------
#
# The feat_types in this repo encode the temporal index (bin or B-bin) in the
# column name itself. RecencyFeatureSelector parses these to drop early-phase
# columns without touching values.
#
#   V / standard / M / D / VMD       : "<itemid>_<bin>"            e.g. "50912_0"
#   V_MGAM cutoff column             : "<itemid>_<bin>__le__<v>"   e.g. "50912_3__le__1.2"
#   V_MGAM missingness intercept     : "<itemid>_<bin>__eq__NaN"   e.g. "50912_3__eq__NaN"
#   V_MGAM interaction               : "<col>__missing_AND_<col>__le__<v>"
#   A / AM "five-feature" columns    : "<itemid>_B<n>_<feature>"   e.g. "50912_B1_mean"
#   AM missingness flag              : "<itemid>_B<n>_mean_missing"
#   demographics                     : "age", "gender"
#
# The first regex finds the LAST numeric token preceded by "_" and either
# followed by "__" or the end of string, which captures the bin index for
# every <itemid>_<bin>-style schema (including the V_MGAM derivatives where
# the bin precedes a "__le__" / "__eq__" suffix).

_BIN_RE = re.compile(r"_(\d+)(?:__|$)")
_A_BIN_RE = re.compile(r"_B(\d+)_")


def _picklable_mi_score(X, y, random_state=42):
    """Top-level wrapper so the SelectKBest score_func is picklable.

    A lambda or a closure captured inside ``fit_transform`` is not picklable
    by ``joblib.dump`` / ``cloudpickle``-without-helpers — every saved fold
    model would otherwise hit a ``PicklingError``. Defining the function at
    module scope lets pickle resolve it by qualified name.
    """
    return mutual_info_classif(X, y, random_state=random_state)


class TopKFeatureSelector:
    """Mutual-information top-K select -> standard-scale.

    Parameters
    ----------
    max_features : int, default 200
        Upper bound on the number of columns kept. If the input has fewer
        columns, all of them are kept (``k`` is clamped to ``X.shape[1]``).
    random_state : int, default 42
        Forwarded to ``mutual_info_classif`` for reproducible scoring.
    """

    def __init__(self, max_features: int = 200, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_dataframe(
        self, X, feature_names: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=list(feature_names))
        df = X.copy()
        if feature_names is not None:
            df.columns = list(feature_names)
        return df

    # ------------------------------------------------------------------
    # sklearn-style API (fit_transform / transform)
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X,
        y,
        feature_names: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        X_df = self._as_dataframe(X, feature_names=feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)

        # 1) mutual-information top-K selection — use functools.partial of a
        #    module-level helper (picklable; lambdas / inline closures are not)
        k = int(min(self.max_features, X_df.shape[1]))
        self.selector_ = SelectKBest(
            score_func=partial(_picklable_mi_score, random_state=self.random_state),
            k=k,
        )
        X_sel = self.selector_.fit_transform(X_df, y)

        mask = self.selector_.get_support()
        self.selected_features_: List[str] = [
            name for name, keep in zip(self.feature_names_in_, mask) if keep
        ]

        # 2) standardize
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "selector_"):
            raise RuntimeError(
                "TopKFeatureSelector.transform called before fit_transform"
            )
        X_df = self._as_dataframe(X, feature_names=self.feature_names_in_)
        X_sel = self.selector_.transform(X_df)
        return self.scaler_.transform(X_sel)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))


# ---------------------------------------------------------------------------
# RecencyFeatureSelector — drop early-phase columns by bin / B-bin index
# ---------------------------------------------------------------------------

class RecencyFeatureSelector:
    """Drop columns that encode "early observation window" measurements.

    A non-statistical alternative to :class:`TopKFeatureSelector` — instead of
    scoring columns by mutual information with the label, this selector
    parses the temporal index embedded in the column name and keeps only the
    later bins. Useful when the prediction task assumes the most recent
    measurements carry the strongest signal (e.g. short-horizon discharge
    outcomes).

    Parameters
    ----------
    drop_bins : sequence of int, optional
        Bin indices to drop for ``<itemid>_<bin>``-style columns (the V, M,
        D, VMD, V_MGAM, standard feat_types). For a 14-bin layout, passing
        ``[0, 1, 2, 3, 4]`` drops the first 5 days.
    drop_b_bins : sequence of int, optional
        B-bin indices (1-based) to drop for ``<itemid>_B<n>_<feature>``-style
        columns (A, AM feat_types). For Approach A's three-bin partition,
        ``[1]`` drops B1 (days 1-5), ``[1, 2]`` drops B1 and B2.
    drop_unmatched : bool, default False
        If True, also drop columns that match neither pattern. Off by default
        so demographics (``age``, ``gender``) and any other pass-through
        columns survive.

    Notes
    -----
    This selector is **stateless** w.r.t. the data — it doesn't need a fit
    pass over X or y. ``fit_transform`` records the surviving column names
    on the train fold so :meth:`transform` can apply the same column subset
    to val/test. The selector silently keeps every column on val/test that
    was kept on train; columns present only on test are dropped.
    """

    def __init__(
        self,
        drop_bins: Optional[Sequence[int]] = None,
        drop_b_bins: Optional[Sequence[int]] = None,
        drop_unmatched: bool = False,
    ):
        self.drop_bins = set(int(b) for b in (drop_bins or ()))
        self.drop_b_bins = set(int(b) for b in (drop_b_bins or ()))
        self.drop_unmatched = bool(drop_unmatched)

    # ------------------------------------------------------------------
    # internal column-decision logic
    # ------------------------------------------------------------------

    def _keep_column(self, col: str) -> bool:
        # Approach-A naming takes precedence (it can also greedily match the
        # plain ``_<digits>`` regex via the feature-name tail, so check first)
        m_a = _A_BIN_RE.search(col)
        if m_a is not None:
            return int(m_a.group(1)) not in self.drop_b_bins
        m = _BIN_RE.search(col)
        if m is not None:
            return int(m.group(1)) not in self.drop_bins
        return not self.drop_unmatched

    # ------------------------------------------------------------------
    # sklearn-style API (parity with TopKFeatureSelector)
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X,
        y=None,
        feature_names: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Decide which columns to drop based on names; return filtered X.

        ``y`` is accepted for API parity with :class:`TopKFeatureSelector`
        but is ignored — recency filtering is label-independent.
        """
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError(
                    "RecencyFeatureSelector.fit_transform on an ndarray "
                    "requires feature_names (column names hold the temporal index)."
                )
            X_df = pd.DataFrame(X, columns=list(feature_names))
        else:
            X_df = X
            if feature_names is not None:
                X_df = X_df.copy()
                X_df.columns = list(feature_names)

        self.feature_names_in_: List[str] = list(X_df.columns)
        self.selected_features_: List[str] = [
            c for c in self.feature_names_in_ if self._keep_column(c)
        ]
        self.dropped_features_: List[str] = [
            c for c in self.feature_names_in_ if c not in set(self.selected_features_)
        ]
        return X_df.loc[:, self.selected_features_]

    def transform(self, X) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "RecencyFeatureSelector.transform called before fit_transform"
            )
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X
        # Keep only columns that survived fit AND are present in X (silently
        # drop test-only columns).
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df.loc[:, cols]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_dropped_features(self) -> List[str]:
        return list(getattr(self, "dropped_features_", []))


# ---------------------------------------------------------------------------
# FeatureSelector — unified facade that dispatches on `score_func`
# ---------------------------------------------------------------------------

# Default early-phase drop list for the recency path. Drops bin indices
# 0..4 (the first 5 days of the 14-day observation window) for V/M/D/VMD/
# V_MGAM/standard, and B1 (Approach-A's days 1-5 bucket) for A/AM.
_RECENCY_DEFAULT_DROP_BINS = (0, 1, 2, 3, 4)
_RECENCY_DEFAULT_DROP_B_BINS = (1,)


def _normalise_score_func_name(score_func) -> str:
    """Map a ``score_func`` argument (callable or string) to a canonical key.

    Accepts the actual sklearn function ``mutual_info_classif`` as well as
    the string aliases ``"mutual_info_classif"`` / ``"mi"`` / ``"recency"``.
    """
    if callable(score_func):
        name = getattr(score_func, "__name__", "")
        if name == "mutual_info_classif":
            return "mutual_info_classif"
        if name == "_picklable_mi_score":
            return "mutual_info_classif"
        raise ValueError(
            f"Unknown callable score_func: {score_func!r}. "
            "Supported callables: mutual_info_classif."
        )
    if isinstance(score_func, str):
        s = score_func.lower().strip()
        if s in ("mutual_info_classif", "mi", "mutual_info", "mutualinfo"):
            return "mutual_info_classif"
        if s in ("recency", "early_phase", "drop_early"):
            return "recency"
        raise ValueError(
            f"Unknown score_func string {score_func!r}. "
            "Supported: 'mutual_info_classif' (or 'mi'), 'recency'."
        )
    raise TypeError(
        f"score_func must be a callable or string, got {type(score_func)}"
    )


class FeatureSelector:
    """Unified facade over :class:`TopKFeatureSelector` and :class:`RecencyFeatureSelector`.

    Pick the underlying strategy via the ``score_func`` argument:

    * ``mutual_info_classif`` (callable) **or** ``"mutual_info_classif"`` /
      ``"mi"`` (string)  → median-impute + ``SelectKBest(MI)`` + StandardScaler
      (the existing :class:`TopKFeatureSelector` pipeline).
    * ``"recency"`` (string, default) → drop early-phase columns by parsing
      the temporal index out of the column names. No statistical fit.

    Parameters
    ----------
    score_func : callable or str, default ``"recency"``
        Selector strategy.
    max_features : int, default 200
        Top-K cap for the MI strategy. Ignored for ``"recency"``.
    random_state : int, default 42
        Forwarded to MI's k-NN estimator. Ignored for ``"recency"``.
    drop_bins : sequence of int, optional
        Bin indices to drop in ``"recency"`` mode. Defaults to ``(0, 1, 2,
        3, 4)`` — the first 5 days of the 14-day window. Ignored for MI.
    drop_b_bins : sequence of int, optional
        B-bin indices to drop in ``"recency"`` mode. Defaults to ``(1,)`` —
        Approach-A's B1 bucket (days 1-5). Ignored for MI.
    drop_unmatched : bool, default False
        ``"recency"`` mode only. If True, columns matching neither bin
        pattern (e.g. ``age``, ``gender``) are also dropped.
    """

    def __init__(
        self,
        score_func="mutual_info_classif",
        max_features: int = 200,
        random_state: int = 42,
        drop_bins: Optional[Sequence[int]] = None,
        drop_b_bins: Optional[Sequence[int]] = None,
        drop_unmatched: bool = False,
    ):
        self.score_func = score_func
        self.max_features = max_features
        self.random_state = random_state
        self.drop_bins = drop_bins
        self.drop_b_bins = drop_b_bins
        self.drop_unmatched = drop_unmatched
        self._mode = _normalise_score_func_name(score_func)

        if self._mode == "mutual_info_classif":
            self._inner = TopKFeatureSelector(
                max_features=self.max_features,
                random_state=self.random_state,
            )
        elif self._mode == "recency":
            self._inner = RecencyFeatureSelector(
                drop_bins=(
                    self.drop_bins
                    if self.drop_bins is not None
                    else _RECENCY_DEFAULT_DROP_BINS
                ),
                drop_b_bins=(
                    self.drop_b_bins
                    if self.drop_b_bins is not None
                    else _RECENCY_DEFAULT_DROP_B_BINS
                ),
                drop_unmatched=self.drop_unmatched,
            )

    # ------------------------------------------------------------------
    # sklearn-style API — same shape as TopKFeatureSelector
    # ------------------------------------------------------------------

    def fit_transform(self, X, y=None, feature_names: Optional[Iterable[str]] = None):
        return self._inner.fit_transform(X, y, feature_names=feature_names)

    def transform(self, X):
        return self._inner.transform(X)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> List[str]:
        return self._inner.get_selected_features()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def inner(self):
        """Expose the underlying selector for any backend-specific calls."""
        return self._inner
