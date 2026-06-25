from typing import Iterable, Optional

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from interpret.glassbox import ExplainableBoostingClassifier

from feature_processing.feature_selector import FeatureSelector


class EBMClassifier(BaseEstimator, ClassifierMixin):
    """EBM wrapped with centralized feature selection + StandardScaler.

    Parameters
    ----------
    selector_type : str, default 'mi'
        Feature selector strategy. One of 'mi', 'tree', 'correlation',
        'xgb_gain', 'stab_net'. See feature_processing.feature_selector.
    max_features : int, default 250
        Top-K selection cap.
    max_bins : int, default 128
    interactions : int, default 2
    outer_bags : int, default 4
    max_rounds : int, default 100
    learning_rate : float, default 0.1
    smoothing_rounds : int, default 25
    interaction_smoothing_rounds : int, default 25
    early_stopping_rounds : int, default 30
    early_stopping_tolerance : float, default 1e-4
    inner_bags : int, default 0
    random_state : int, default 42
    """

    def __init__(
        self,
        selector_type: str = "mi",
        max_features: int = 250,
        max_bins: int = 128,
        interactions: int = 2,
        outer_bags: int = 4,
        max_rounds: int = 100,
        learning_rate: float = 0.1,
        smoothing_rounds: int = 25,
        interaction_smoothing_rounds: int = 25,
        early_stopping_rounds: int = 30,
        early_stopping_tolerance: float = 1e-4,
        inner_bags: int = 0,
        random_state: int = 42,
    ):
        self.selector_type = selector_type
        self.max_features = max_features
        self.max_bins = max_bins
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.smoothing_rounds = smoothing_rounds
        self.interaction_smoothing_rounds = interaction_smoothing_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.inner_bags = inner_bags
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
            f"[EBM] Training with {X_scaled.shape[1]} selected features "
            f"(selector={self.selector_type}) | "
            f"max_bins={self.max_bins} | interactions={self.interactions} | "
            f"outer_bags={self.outer_bags} | max_rounds={self.max_rounds} | "
            f"lr={self.learning_rate}"
        )

        self.ebm_ = ExplainableBoostingClassifier(
            max_bins=self.max_bins,
            interactions=self.interactions,
            outer_bags=self.outer_bags,
            inner_bags=self.inner_bags,
            max_rounds=self.max_rounds,
            learning_rate=self.learning_rate,
            smoothing_rounds=self.smoothing_rounds,
            interaction_smoothing_rounds=self.interaction_smoothing_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            random_state=self.random_state,
            feature_names=self.selected_features_,
        )
        self.ebm_.fit(X_scaled, y)
        self.classes_ = np.array([0, 1])
        return self

    def _transform_X(self, X):
        check_is_fitted(self, "ebm_")
        return self.preproc_.transform(X)

    def predict_proba(self, X):
        return self.ebm_.predict_proba(self._transform_X(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_selected_features(self):
        check_is_fitted(self, "ebm_")
        return list(self.selected_features_)

    def get_ebm(self):
        check_is_fitted(self, "ebm_")
        return self.ebm_
