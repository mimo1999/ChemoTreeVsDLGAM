"""
Centralized feature-selection helpers for the ML pipeline.

Five selectors sharing a common fit_transform / transform / get_selected_features API:

    TopKFeatureSelector  (mi)          — mutual-information SelectKBest
    TreeFeatureSelector  (tree)        — RandomForest MDI importance
    CorrelationFeatureSelector         — |feature-target Pearson correlation|
    XGBGainSelector      (xgb_gain)    — XGBoost gain importance
    LassoFeatureSelector (lasso)       — L1-regularized logistic regression

Use build_selector(selector_type, max_features, ...) to construct the right selector,
or use FeatureSelector as a unified facade.
"""

from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
# LassoFeatureSelector — L1-regularized logistic regression
# ---------------------------------------------------------------------------

class LassoFeatureSelector:
    """Standardize -> L1 logistic regression -> top-K by |coefficient|.

    Embedded feature selection using L1-regularized logistic regression.

    Parameters
    ----------
    max_features : int, default 250
    C : float, default 0.1
    random_state : int, default 42
    max_iter : int, default 1000
    """

    def __init__(
        self,
        max_features: int = 250,
        C: float = 0.1,
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        self.max_features = max_features
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        X_df = _as_dataframe(X, feature_names)
        self.feature_names_in_: List[str] = list(X_df.columns)
        X_arr = X_df.to_numpy()

        self.pre_scaler_ = StandardScaler()
        X_std = self.pre_scaler_.fit_transform(X_arr)

        clf = LogisticRegression(
            penalty="l1", solver="saga", C=self.C, class_weight="balanced",
            max_iter=self.max_iter, random_state=self.random_state, n_jobs=-1,
        )
        clf.fit(X_std, np.ravel(y))

        coef = np.abs(clf.coef_.ravel())
        order = np.argsort(coef)[::-1]
        k = int(min(self.max_features, len(order)))
        selected = sorted(order[:k])

        self.selected_features_: List[str] = [self.feature_names_in_[i] for i in selected]
        self.importance_scores_ = pd.Series(coef, index=self.feature_names_in_)

        X_sel = X_arr[:, selected]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_sel)
        print(
            f"[LassoSelector] {X_arr.shape[1]} input -> "
            f"{X_scaled.shape[1]} selected (top-{k} |L1 coefficient|)"
        )
        return X_scaled

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("LassoFeatureSelector.transform called before fit_transform")
        X_df = _as_dataframe(X, self.feature_names_in_)
        idx = [self.feature_names_in_.index(c) for c in self.selected_features_]
        X_sel = X_df.to_numpy()[:, idx]
        return self.scaler_.transform(X_sel)

    def get_selected_features(self) -> List[str]:
        return list(getattr(self, "selected_features_", []))

    def get_feature_importances(self) -> pd.Series:
        if not hasattr(self, "importance_scores_"):
            raise RuntimeError("LassoFeatureSelector.get_feature_importances called before fit_transform")
        return self.importance_scores_[self.selected_features_].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# build_selector — factory function
# ---------------------------------------------------------------------------

_SELECTOR_TYPES = ("mi", "tree", "correlation", "xgb_gain", "lasso")


def build_selector(
    selector_type: str = "mi",
    max_features: int = 250,
    random_state: int = 42,
    **kwargs,
):
    """Construct a selector by name.

    Parameters
    ----------
    selector_type : str
        One of 'mi', 'tree', 'correlation', 'xgb_gain', 'lasso'.
    max_features : int, default 250
    random_state : int, default 42
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
    if st == "lasso":
        return LassoFeatureSelector(max_features=max_features, random_state=random_state, **kwargs)
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
        One of 'mi', 'tree', 'correlation', 'xgb_gain', 'lasso'.
        Also accepts legacy score_func aliases 'mutual_info_classif' and 'mi'.
    max_features : int, default 250
    random_state : int, default 42
    """

    def __init__(
        self,
        selector_type: str = "mi",
        max_features: int = 250,
        random_state: int = 42,
        # backward-compat alias
        score_func: Optional[str] = None,
    ):
        # score_func was the legacy param name used in older wrappers
        if score_func is not None and selector_type == "mi":
            selector_type = score_func

        self.selector_type = selector_type
        self.max_features = max_features
        self.random_state = random_state

        self._inner = build_selector(
            selector_type=selector_type,
            max_features=max_features,
            random_state=random_state,
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
