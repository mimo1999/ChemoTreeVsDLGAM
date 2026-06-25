"""
GAMINET wrapper for the ChemoTreeVsDL pipeline.

Pipeline: MinMax[0,1] -> GAMINetClassifier

MinMax scaling is required because GAMINET's internal binner assumes a finite
support; heavy-tailed lab features can produce extreme z-scores that NaN-out
the loss at the first epoch when StandardScaler is used instead.

Feature selection and NaN-filling are handled upstream by DataLoader.
"""

from collections.abc import Iterable
from typing import Tuple

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# piml's GAMINetClassifier still calls BaseEstimator._validate_data, which was
# removed in sklearn 1.6. Reinstate it as a thin wrapper around the new API.
if not hasattr(BaseEstimator, "_validate_data"):
    def _validate_data_compat(self, X="no_validation", y="no_validation", **kwargs):
        from sklearn.utils.validation import validate_data as _vd
        return _vd(self, X=X, y=y, **kwargs)

    BaseEstimator._validate_data = _validate_data_compat  # type: ignore[attr-defined]

from piml.models import GAMINetClassifier


class GAMINetWrapperClassifier(ClassifierMixin, BaseEstimator):
    """GAMINET wrapped with MinMax [0,1] scaling.

    Parameters
    ----------
    interact_num : int, default 2
        Number of pairwise interaction subnets.
    batch_size : int, default 1024
    activation_func : str, default "ReLU"
    reg_clarity : float, default 0.1
    max_epochs : tuple[int, int, int], default (50, 50, 50)
        (main-effects, interactions, fine-tune) epoch caps per stage.
    device : str, default "cpu"
    random_state : int, default 0
    """

    def __init__(
        self,
        interact_num: int = 2,
        batch_size: int = 1024,
        activation_func: str = "ReLU",
        reg_clarity: float = 0.1,
        max_epochs: Tuple[int, int, int] = (50, 50, 50),
        device: str = "cpu",
        random_state: int = 0,
    ):
        self.interact_num = interact_num
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.reg_clarity = reg_clarity
        self.max_epochs = max_epochs
        self.device = device
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names: Iterable[str] | None = None,
        sample_weight=None,
    ):
        if feature_names is None:
            feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        self.selected_features_ = list(feature_names)

        X_scaled = np.asarray(X, dtype=np.float32)

        col_min = X_scaled.min(axis=0)
        col_max = X_scaled.max(axis=0)
        col_range = col_max - col_min
        self._gaminet_keep_mask_ = col_range > 1e-8
        if not self._gaminet_keep_mask_.all():
            dropped = int((~self._gaminet_keep_mask_).sum())
            print(f"[GAMINET] dropping {dropped} zero-variance columns post-selection")
            X_scaled = X_scaled[:, self._gaminet_keep_mask_]
            col_min = col_min[self._gaminet_keep_mask_]
            col_range = col_range[self._gaminet_keep_mask_]
        self._gaminet_col_min_ = col_min
        self._gaminet_col_range_ = col_range
        X_scaled = np.clip((X_scaled - col_min) / col_range, 0.0, 1.0).astype(np.float32)

        print(
            f"[GAMINET] Training with {X_scaled.shape[1]} selected features | "
            f"interact_num={self.interact_num} | "
            f"max_epochs={self.max_epochs} | batch_size={self.batch_size}"
        )

        # yaml lists arrive as Python lists; piml expects a 3-tuple.
        max_epochs = tuple(self.max_epochs) if not isinstance(self.max_epochs, tuple) else self.max_epochs

        self.gaminet_ = GAMINetClassifier(
            batch_size=self.batch_size,
            interact_num=self.interact_num,
            activation_func=self.activation_func,
            reg_clarity=self.reg_clarity,
            warm_start=False,
            max_epochs=max_epochs,
            verbose=True,
            device=self.device,
            random_state=self.random_state,
        )
        self.gaminet_.fit(X_scaled, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def _transform_X(self, X):
        check_is_fitted(self, "gaminet_")
        Xt = np.asarray(X, dtype=np.float32)
        if hasattr(self, "_gaminet_keep_mask_"):
            Xt = Xt[:, self._gaminet_keep_mask_]
        Xt = (Xt - self._gaminet_col_min_) / self._gaminet_col_range_
        return np.clip(Xt, 0.0, 1.0).astype(np.float32)

    def predict_proba(self, X):
        proba = np.asarray(self.gaminet_.predict_proba(self._transform_X(X)))
        # GAMINET can emit NaNs if its fine-tune stage diverged.
        if not np.all(np.isfinite(proba)):
            proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_selected_features(self):
        check_is_fitted(self, "gaminet_")
        return list(self.selected_features_)

    def get_gaminet(self):
        check_is_fitted(self, "gaminet_")
        return self.gaminet_
