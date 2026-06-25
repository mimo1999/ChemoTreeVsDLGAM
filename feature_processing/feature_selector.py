"""
Centralized feature-selection helpers for the ML pipeline.

Six selectors sharing a common fit_transform / transform / get_selected_features API:

    TopKFeatureSelector  (mi)        — mutual-information SelectKBest
    TreeFeatureSelector  (tree)      — RandomForest MDI importance
    CorrelationFeatureSelector       — |feature-target Pearson correlation|
    XGBGainSelector      (xgb_gain)  — XGBoost gain importance
    StabilityNetSelector (stab_net)  — elastic-net stability selection
    RecencyFeatureSelector           — drop early-phase time bins by column name

Use build_selector(selector_type, max_features, ...) to construct the right selector,
or use FeatureSelector as a unified facade.
"""

import re
from collections.abc import Iterable
from functools import partial
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel, SelectKBest, f_classif, mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Column-name patterns used by RecencyFeatureSelector
# ---------------------------------------------------------------------------
#
#   V / standard / M / D / VMD       : "<itemid>_<bin>"            e.g. "50912_0"
#   V_MGAM cutoff column             : "<itemid>_<bin>__le__<v>"   e.g. "50912_3__le__1.2"
#   V_MGAM missingness intercept     : "<itemid>_<bin>__eq__NaN"   e.g. "50912_3__eq__NaN"
#   V_MGAM interaction               : "<col>__missing_AND_<col>__le__<v>"
#   A / AM "five-feature" columns    : "<itemid>_B<n>_<feature>"   e.g. "50912_B1_mean"
#   AM missingness flag              : "<itemid>_B<n>_mean_missing"
#   demographics                     : "age", "gender"

_BIN_RE = re.compile(r"_(\d+)(?:__|$)")
_A_BIN_RE = re.compile(r"_B(\d+)_")

_RECENCY_DEFAULT_DROP_BINS = (0, 1, 2, 3, 4)
_RECENCY_DEFAULT_DROP_B_BINS = (1,)


def _picklable_mi_score(X, y, random_state=42):
    """Top-level wrapper so the SelectKBest score_func is picklable.

    A lambda or closure captured inside fit_transform is not picklable by
    joblib.dump — every saved fold model would hit a PicklingError.
    """
    return mutual_info_classif(X, y, random_state=random_state)


def _as_dataframe(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=list(feature_names))
    df = X.copy()
    if feature_names is not None:
        df.columns = list(feature_names)
    return df


# ---------------------------------------------------------------------------
# TopKFeatureSelector — mutual-information SelectKBest
# ---------------------------------------------------------------------------

class TopKFeatureSelector:
    """Median-impute -> mutual-information top-K select -> standard-scale.

    Parameters
    ----------
    max_features : int, default 250
        Upper bound on columns kept. Clamped to X.shape[1].
    random_state : int, default 42
    """

    def __init__(self, max_features: int = 250, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_df)

        k = int(min(self.max_features, X_imp.shape[1]))
        self.selector_ = SelectKBest(
            score_func=partial(_picklable_mi_score, random_state=self.random_state),
            k=k,
        )
        X_sel = self.selector_.fit_transform(X_imp, y)

        mask = self.selector_.get_support()
        self.selected_features_: List[str] = [
            name for name, keep in zip(self.feature_names_in_, mask) if keep
        ]

        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(X_sel)

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("TopKFeatureSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        X_imp = self.imputer_.transform(X_df)
        X_sel = self.selector_.transform(X_imp)
        return self.scaler_.transform(X_sel)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))


# ---------------------------------------------------------------------------
# TreeFeatureSelector — RandomForest MDI importance
# ---------------------------------------------------------------------------

class TreeFeatureSelector:
    """Median-impute -> RandomForest importance top-K select -> standard-scale.

    Parameters
    ----------
    max_features : int, default 250
    n_estimators : int, default 100
    max_depth : int, default 5
    random_state : int, default 42
    """

    def __init__(
        self,
        max_features: int = 250,
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_df)

        k = int(min(self.max_features, X_imp.shape[1]))
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )
        rf.fit(X_imp, y)

        importances = rf.feature_importances_
        sorted_importances = np.sort(importances)[::-1]
        threshold = sorted_importances[k - 1] if k <= len(sorted_importances) else 0.0

        self.selector_ = SelectFromModel(
            estimator=rf, threshold=threshold, prefit=True, max_features=k,
        )
        X_sel = self.selector_.transform(X_imp)

        mask = self.selector_.get_support()
        self.selected_features_: List[str] = [
            name for name, keep in zip(self.feature_names_in_, mask) if keep
        ]

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        print(
            f"[TreeSelector] {X_imp.shape[1]} input -> {X_scaled.shape[1]} selected "
            f"(top-{k} RF importance)"
        )
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("TreeFeatureSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        X_imp = self.imputer_.transform(X_df)
        X_sel = self.selector_.transform(X_imp)
        return self.scaler_.transform(X_sel)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_feature_importances(self) -> pd.Series:
        if not hasattr(self, "selector_"):
            raise RuntimeError("TreeFeatureSelector.get_feature_importances called before fit_transform")
        importances = self.selector_.estimator.feature_importances_
        mask = self.selector_.get_support()
        return pd.Series(importances[mask], index=self.selected_features_).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# CorrelationFeatureSelector — |feature-target Pearson/Spearman correlation|
# ---------------------------------------------------------------------------

class CorrelationFeatureSelector:
    """Median-impute -> top-K by |feature-target correlation| -> standard-scale.

    Parameters
    ----------
    max_features : int, default 250
    method : {"pearson", "spearman"}, default "pearson"
    random_state : int, default 42
        Unused (kept for consistent constructor signature).
    """

    def __init__(
        self,
        max_features: int = 250,
        method: str = "pearson",
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.method = method
        self.random_state = random_state

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(
            self.imputer_.fit_transform(X_df), columns=self.feature_names_in_
        )

        y_ser = pd.Series(np.ravel(y), index=X_imp.index)
        scores = X_imp.corrwith(y_ser, method=self.method).abs().fillna(0.0)
        self.scores_ = scores

        k = int(min(self.max_features, X_imp.shape[1]))
        keep = set(scores.sort_values(ascending=False).index[:k])
        self.selected_features_: List[str] = [c for c in self.feature_names_in_ if c in keep]

        X_sel = X_imp[self.selected_features_].values
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        print(
            f"[CorrelationSelector] {X_imp.shape[1]} input -> "
            f"{X_scaled.shape[1]} selected (top-{k} |{self.method}| corr)"
        )
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("CorrelationFeatureSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        X_imp = pd.DataFrame(self.imputer_.transform(X_df), columns=self.feature_names_in_)
        return self.scaler_.transform(X_imp[self.selected_features_].values)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_feature_scores(self) -> pd.Series:
        if not hasattr(self, "scores_"):
            raise RuntimeError("CorrelationFeatureSelector.get_feature_scores called before fit_transform")
        return self.scores_[self.selected_features_].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# XGBGainSelector — XGBoost gain importance
# ---------------------------------------------------------------------------

class XGBGainSelector:
    """Median-impute -> XGBoost gain importance top-K -> scale.

    Parameters
    ----------
    max_features : int, default 250
    n_estimators : int, default 200
    max_depth : int, default 4
    learning_rate : float, default 0.05
    random_state : int, default 42
    """

    def __init__(
        self,
        max_features: int = 250,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        import xgboost as xgb

        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_df)
        y_arr = np.ravel(y)

        pos = y_arr.sum()
        neg = len(y_arr) - pos
        clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=self.random_state,
            verbosity=0,
        )
        clf.fit(X_imp, y_arr)

        gain_map = clf.get_booster().get_score(importance_type="gain")
        scores = np.array([gain_map.get(f"f{i}", 0.0) for i in range(X_imp.shape[1])])
        self.importance_scores_ = pd.Series(scores, index=self.feature_names_in_)

        k = int(min(self.max_features, X_imp.shape[1]))
        keep = set(np.array(self.feature_names_in_)[np.argsort(scores)[::-1][:k]])
        self.selected_features_: List[str] = [c for c in self.feature_names_in_ if c in keep]

        X_sel = X_imp[:, [self.feature_names_in_.index(c) for c in self.selected_features_]]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        print(
            f"[XGBGainSelector] {X_imp.shape[1]} input -> "
            f"{X_scaled.shape[1]} selected (top-{k} XGB gain)"
        )
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("XGBGainSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        X_imp = self.imputer_.transform(X_df)
        X_sel = X_imp[:, [self.feature_names_in_.index(c) for c in self.selected_features_]]
        return self.scaler_.transform(X_sel)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_feature_importances(self) -> pd.Series:
        if not hasattr(self, "importance_scores_"):
            raise RuntimeError("XGBGainSelector.get_feature_importances called before fit_transform")
        return self.importance_scores_[self.selected_features_].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# StabilityNetSelector — elastic-net stability selection
# ---------------------------------------------------------------------------

class StabilityNetSelector:
    """F-test screen -> elastic-net stability selection -> top-K by selection frequency.

    Parameters
    ----------
    max_features : int, default 250
    screen : int, default 2000
        F-test screen size before the elastic-net stage.
    n_bootstrap : int, default 25
    sample_fraction : float, default 0.6
    C : float, default 0.5
    l1_ratio : float, default 0.5
    max_iter : int, default 200
    random_state : int, default 42
    """

    def __init__(
        self,
        max_features: int = 250,
        screen: int = 2000,
        n_bootstrap: int = 25,
        sample_fraction: float = 0.6,
        C: float = 0.5,
        l1_ratio: float = 0.5,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.screen = screen
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)
        n_feats = X_df.shape[1]

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_df)
        X_std = StandardScaler().fit_transform(X_imp)
        y_arr = np.ravel(y)

        f_scores = np.nan_to_num(f_classif(X_std, y_arr)[0])
        screen_n = int(min(self.screen, n_feats))
        screen_idx = np.argsort(f_scores)[::-1][:screen_n]
        screen_names = [self.feature_names_in_[i] for i in screen_idx]
        X_screen = X_std[:, screen_idx]

        n_samples = X_screen.shape[0]
        boot_n = int(self.sample_fraction * n_samples)
        sel_count = np.zeros(screen_n)
        coef_sum = np.zeros(screen_n)
        for b in range(self.n_bootstrap):
            rng = np.random.RandomState(self.random_state + b)
            idx = rng.choice(n_samples, boot_n, replace=False)
            if len(np.unique(y_arr[idx])) < 2:
                continue
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio,
                C=self.C, class_weight="balanced", max_iter=self.max_iter,
                tol=1e-3, random_state=self.random_state,
            )
            clf.fit(X_screen[idx], y_arr[idx])
            coef = np.abs(clf.coef_.ravel())
            sel_count += (coef > 0)
            coef_sum += coef

        freq = sel_count / max(self.n_bootstrap, 1)
        mean_coef = coef_sum / max(self.n_bootstrap, 1)
        self.selection_frequency_ = pd.Series(freq, index=screen_names)

        order = np.lexsort((-mean_coef, -freq))
        k = int(min(self.max_features, screen_n))
        selected_top = set(np.array(screen_names)[order[:k]])
        self.selected_features_: List[str] = [
            c for c in self.feature_names_in_ if c in selected_top
        ]

        X_sel = X_imp[:, [self.feature_names_in_.index(c) for c in self.selected_features_]]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        print(
            f"[StabilityNetSelector] {n_feats} input -> {screen_n} F-screened -> "
            f"{X_scaled.shape[1]} selected (top-{k} by selection freq)"
        )
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("StabilityNetSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        X_imp = self.imputer_.transform(X_df)
        X_sel = X_imp[:, [self.feature_names_in_.index(c) for c in self.selected_features_]]
        return self.scaler_.transform(X_sel)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_selection_frequency(self) -> pd.Series:
        if not hasattr(self, "selection_frequency_"):
            raise RuntimeError("StabilityNetSelector.get_selection_frequency called before fit_transform")
        sel = [c for c in self.selected_features_ if c in self.selection_frequency_.index]
        return self.selection_frequency_[sel].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# RecencyFeatureSelector — drop early-phase time bins by column name
# ---------------------------------------------------------------------------

class RecencyFeatureSelector:
    """Drop columns encoding early observation-window measurements.

    Non-statistical: parses the temporal index embedded in the column name
    and keeps only the later bins. No fit over X or y.

    Parameters
    ----------
    drop_bins : sequence of int, optional
        Bin indices to drop for ``<itemid>_<bin>``-style columns (V, M, D,
        VMD, V_MGAM, standard feat_types). Default: (0,1,2,3,4).
    drop_b_bins : sequence of int, optional
        B-bin indices (1-based) to drop for ``<itemid>_B<n>_<feature>``-style
        columns (A, AM feat_types). Default: (1,).
    drop_unmatched : bool, default False
        If True, also drop columns matching neither pattern.
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

    def _keep_column(self, col: str) -> bool:
        m_a = _A_BIN_RE.search(col)
        if m_a is not None:
            return int(m_a.group(1)) not in self.drop_b_bins
        m = _BIN_RE.search(col)
        if m is not None:
            return int(m.group(1)) not in self.drop_bins
        return not self.drop_unmatched

    def fit_transform(self, X, y=None, feature_names=None) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError(
                    "RecencyFeatureSelector.fit_transform on an ndarray requires feature_names."
                )
            X_df = pd.DataFrame(X, columns=list(feature_names))
        else:
            X_df = X.copy()
            if feature_names is not None:
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
            raise RuntimeError("RecencyFeatureSelector.transform called before fit_transform")
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df.loc[:, cols]

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_dropped_features(self) -> List[str]:
        return list(getattr(self, "dropped_features_", []))


# ---------------------------------------------------------------------------
# build_selector — factory function
# ---------------------------------------------------------------------------

_SELECTOR_TYPES = ("mi", "tree", "correlation", "xgb_gain", "stab_net", "recency")


def build_selector(
    selector_type: str = "mi",
    max_features: int = 250,
    random_state: int = 42,
    drop_bins: Optional[Sequence[int]] = None,
    drop_b_bins: Optional[Sequence[int]] = None,
    **kwargs,
):
    """Construct a selector by name.

    Parameters
    ----------
    selector_type : str
        One of 'mi', 'tree', 'correlation', 'xgb_gain', 'stab_net', 'recency'.
    max_features : int, default 250
        Top-K cap (ignored for 'recency').
    random_state : int, default 42
    drop_bins, drop_b_bins : sequence of int, optional
        Forwarded to RecencyFeatureSelector when selector_type='recency'.
    **kwargs
        Extra keyword arguments forwarded to the selector constructor.
    """
    st = selector_type.lower().strip()
    if st in ("mi", "mutual_info_classif", "mutual_info"):
        return TopKFeatureSelector(max_features=max_features, random_state=random_state, **kwargs)
    if st == "tree":
        return TreeFeatureSelector(max_features=max_features, random_state=random_state, **kwargs)
    if st == "correlation":
        return CorrelationFeatureSelector(max_features=max_features, random_state=random_state, **kwargs)
    if st == "xgb_gain":
        return XGBGainSelector(max_features=max_features, random_state=random_state, **kwargs)
    if st == "stab_net":
        return StabilityNetSelector(max_features=max_features, random_state=random_state, **kwargs)
    if st in ("recency", "early_phase", "drop_early"):
        return RecencyFeatureSelector(
            drop_bins=drop_bins if drop_bins is not None else _RECENCY_DEFAULT_DROP_BINS,
            drop_b_bins=drop_b_bins if drop_b_bins is not None else _RECENCY_DEFAULT_DROP_B_BINS,
        )
    raise ValueError(
        f"Unknown selector_type '{selector_type}'. "
        f"Available: {_SELECTOR_TYPES}"
    )


# ---------------------------------------------------------------------------
# FeatureSelector — unified facade (used by model wrappers)
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Unified facade over all selector strategies.

    Parameters
    ----------
    selector_type : str, default 'mi'
        One of 'mi', 'tree', 'correlation', 'xgb_gain', 'stab_net', 'recency'.
        Also accepts legacy score_func aliases 'mutual_info_classif' and 'mi'.
    max_features : int, default 250
        Top-K cap (ignored for 'recency').
    random_state : int, default 42
    drop_bins, drop_b_bins : sequence of int, optional
        Forwarded to RecencyFeatureSelector when selector_type='recency'.
    drop_unmatched : bool, default False
        RecencyFeatureSelector only.
    """

    def __init__(
        self,
        selector_type: str = "mi",
        max_features: int = 250,
        random_state: int = 42,
        drop_bins: Optional[Sequence[int]] = None,
        drop_b_bins: Optional[Sequence[int]] = None,
        drop_unmatched: bool = False,
        # backward-compat alias
        score_func: Optional[str] = None,
    ):
        # score_func was the legacy param name used in older wrappers
        if score_func is not None and selector_type == "mi":
            selector_type = score_func

        self.selector_type = selector_type
        self.max_features = max_features
        self.random_state = random_state
        self.drop_bins = drop_bins
        self.drop_b_bins = drop_b_bins
        self.drop_unmatched = drop_unmatched

        self._inner = build_selector(
            selector_type=selector_type,
            max_features=max_features,
            random_state=random_state,
            drop_bins=drop_bins,
            drop_b_bins=drop_b_bins,
        )

    def fit_transform(self, X, y=None, feature_names=None):
        return self._inner.fit_transform(X, y, feature_names=feature_names)

    def transform(self, X):
        return self._inner.transform(X)

    def get_selected_features(self) -> List[str]:
        return self._inner.get_selected_features()

    @property
    def mode(self) -> str:
        return self.selector_type

    @property
    def inner(self):
        return self._inner
