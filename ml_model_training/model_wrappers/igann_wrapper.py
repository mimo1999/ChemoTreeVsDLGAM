"""
Lightweight IGANN wrapper.

IGANN is an "Interpretable Generalized Additive Neural Network" — see
Kraus et al., https://github.com/MathiasKraus/igann. It fits one ELM
shape function per feature plus optional interaction terms, using boosting
across the per-feature ELMs.

Feature selection is handled by the centralized FeatureSelector
(feature_processing.feature_selector), consistent with the other wrappers.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from igann import IGANN

from feature_processing.feature_selector import FeatureSelector


class IGANNWrapperClassifier(BaseEstimator, ClassifierMixin):
    """IGANN wrapped with centralized feature selection + StandardScaler.

    Parameters
    ----------
    selector_type : str, default 'mi'
        Feature selector strategy. One of 'mi', 'tree', 'correlation',
        'xgb_gain', 'stab_net'. See feature_processing.feature_selector.
    max_features : int, default 250
        Top-K selection cap.
    n_estimators : int, default 5000
    boost_rate : float, default 0.1
    elm_scale : float, default 1.0
    elm_alpha : float, default 1.0
    n_hid : int, default 10
    init_reg : float, default 1.0
    act : str, default "elu"
    early_stopping : int, default 50
    device : str, default "cpu"
    verbose : int, default 0
    random_state : int, default 1
    """

    def __init__(
        self,
        selector_type: str = "mi",
        max_features: int = 250,
        n_estimators: int = 5000,
        boost_rate: float = 0.1,
        elm_scale: float = 1.0,
        elm_alpha: float = 1.0,
        n_hid: int = 10,
        init_reg: float = 1.0,
        act: str = "elu",
        early_stopping: int = 50,
        device: str = "cpu",
        verbose: int = 0,
        random_state: int = 1,
    ):
        self.selector_type = selector_type
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.boost_rate = boost_rate
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.n_hid = n_hid
        self.init_reg = init_reg
        self.act = act
        self.early_stopping = early_stopping
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names: Optional[Iterable[str]] = None,
        cat_feature_names=None,
    ):
        self.preproc_ = FeatureSelector(
            selector_type=self.selector_type,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        X_scaled = self.preproc_.fit_transform(X, y, feature_names=feature_names)
        self.selected_features_ = self.preproc_.get_selected_features()

        print(
            f"[IGANN] Training with {X_scaled.shape[1]} selected features "
            f"(selector={self.selector_type}) | "
            f"n_estimators={self.n_estimators} | boost_rate={self.boost_rate} | "
            f"n_hid={self.n_hid} | early_stopping={self.early_stopping}"
        )

        self.igann_ = IGANN(
            task="classification",
            n_hid=self.n_hid,
            n_estimators=self.n_estimators,
            boost_rate=self.boost_rate,
            init_reg=self.init_reg,
            elm_scale=self.elm_scale,
            elm_alpha=self.elm_alpha,
            act=self.act,
            early_stopping=self.early_stopping,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        # IGANN's fit calls X.iloc internally — wrap into DataFrame.
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.selected_features_)
        y_series = pd.Series(np.asarray(y).ravel())
        self.igann_.fit(X_scaled_df, y_series)
        self.classes_ = np.array([0, 1])
        return self

    def _transform_X(self, X):
        check_is_fitted(self, "igann_")
        X_scaled = self.preproc_.transform(X)
        return pd.DataFrame(X_scaled, columns=self.selected_features_)

    def predict_proba(self, X):
        proba = np.asarray(self.igann_.predict_proba(self._transform_X(X)))
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_selected_features(self):
        check_is_fitted(self, "igann_")
        return list(self.selected_features_)

    def get_igann(self):
        check_is_fitted(self, "igann_")
        return self.igann_
