"""Shared sklearn-API base class for the model wrappers in this package.

Every wrapper here (GAM, EBM, GAMINET, IGANN, MGAM, NAM, DGAM, HE-EBM,
IG-EBM, SSTGAM) implements predict() and predict_log_proba() as the
same derivation from predict_proba(): threshold class-1 probability at 0.5,
or take the log of the clipped probabilities. get_selected_features() is
almost always just "return the feature_names the wrapper was fit with".
BaseWrapperClassifier factors that out so subclasses only need to implement
fit() and predict_proba(), and override get_selected_features() only when
their notion of "selected features" differs from `self.selected_features_`
(e.g. MGAM's interaction-augmented column list, NAM's zero-variance-filtered
list).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class BaseWrapperClassifier(ClassifierMixin, BaseEstimator):
    """Common predict/predict_log_proba/get_selected_features for the wrappers.

    Subclasses must, in fit():
      - set `self.classes_` (typically `np.array([0, 1])`)
      - set `self.selected_features_` to the feature names used for training,
        unless they override get_selected_features() themselves
    and must implement predict_proba(X) returning an (n_samples, 2) array.
    """

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_selected_features(self) -> list:
        check_is_fitted(self, "selected_features_")
        return list(self.selected_features_)
