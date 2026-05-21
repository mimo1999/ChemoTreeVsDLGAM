"""
Lightweight GAMINET wrapper that mirrors the pattern of
``ChemoTreeVsDL-main/ml_model_training/gam_wrapper.py``:

    SimpleImputer -> SelectKBest(MI, k) -> StandardScaler -> GAMINetClassifier

Designed for wide, sparse tabular cohorts where the library default
``max_epochs=(3000, 1000, 1000)`` is prohibitively expensive. Defaults below
keep training time within minutes per fold.

Lightweight defaults
--------------------
    interact_num=2, batch_size=1024, activation_func="ReLU",
    reg_clarity=0.1, max_epochs=(50, 50, 50), device="cpu"
"""
from typing import Iterable, Optional, Tuple

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------
# Compatibility patch for piml + sklearn 1.6+
# ---------------------------------------------------------------------
# sklearn 1.6 removed BaseEstimator._validate_data; piml's GAMINetClassifier
# still calls it. Reinstate it as a thin wrapper around the new functional
# API so the model fits without monkey-patching at the caller. The import
# is performed lazily inside the method to avoid any free-variable / stale-
# bytecode issues with closures captured at module load time.
if not hasattr(BaseEstimator, "_validate_data"):
    def _validate_data_compat(self, X="no_validation", y="no_validation", **kwargs):
        from sklearn.utils.validation import validate_data as _vd  # local import
        return _vd(self, X=X, y=y, **kwargs)

    BaseEstimator._validate_data = _validate_data_compat  # type: ignore[attr-defined]

from piml.models import GAMINetClassifier

from utils.feature_selector import FeatureSelector


class GAMINetWrapperClassifier(BaseEstimator, ClassifierMixin):
    """GAMINET wrapped with a configurable feature selector + StandardScaler.

    Parameters
    ----------
    score_func : str or callable, default ``"mutual_info_classif"``
        Strategy passed to :class:`utils.feature_selector.FeatureSelector`.
        - ``"mutual_info_classif"`` / ``"mi"`` / sklearn ``mutual_info_classif``
          → median-impute + ``SelectKBest(MI)`` + StandardScaler.
        - ``"recency"`` → drop early-phase columns by parsing the temporal
          index out of the column names (no statistical fit).
    max_features : int, default 200
        Top-K MI selection cap. Ignored when ``score_func="recency"``.
    drop_bins, drop_b_bins : sequence of int, optional
        Forwarded to the recency selector when ``score_func="recency"``.
        Defaults to ``(0,1,2,3,4)`` and ``(1,)`` respectively (drop the
        first 5 days for V/M/D/VMD/V_MGAM/standard and B1 for A/AM).
    interact_num : int, default 2
        Number of pairwise interaction subnets.
    batch_size : int, default 1024
    activation_func : str, default "ReLU"
    reg_clarity : float, default 0.1
    max_epochs : tuple[int, int, int], default (50, 50, 50)
        (main-effects, interactions, fine-tune) — each stage is capped here.
    device : str, default "cpu"
    random_state : int, default 0
    """

    def __init__(
        self,
        score_func="mutual_info_classif",
        max_features: int = 200,
        drop_bins=None,
        drop_b_bins=None,
        interact_num: int = 2,
        batch_size: int = 1024,
        activation_func: str = "ReLU",
        reg_clarity: float = 0.1,
        max_epochs: Tuple[int, int, int] = (50, 50, 50),
        device: str = "cpu",
        random_state: int = 0,
    ):
        self.score_func = score_func
        self.max_features = max_features
        self.drop_bins = drop_bins
        self.drop_b_bins = drop_b_bins
        self.interact_num = interact_num
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.reg_clarity = reg_clarity
        self.max_epochs = max_epochs
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        feature_names: Optional[Iterable[str]] = None,
        cat_feature_names=None,  # accepted for signature parity; unused
    ):
        self.preproc_ = FeatureSelector(
            score_func=self.score_func,
            max_features=self.max_features,
            random_state=42,
            drop_bins=self.drop_bins,
            drop_b_bins=self.drop_b_bins,
        )
        X_scaled = self.preproc_.fit_transform(
            X, y, feature_names=feature_names
        )
        self.selected_features_ = self.preproc_.get_selected_features()

        # GAMINET expects bounded inputs (its internal binner assumes a finite
        # support). After StandardScaler, heavy-tailed A+ features can produce
        # extreme values that NaN-out the loss at epoch 1. Apply a min-max
        # rescale to [0, 1] per column and drop zero-variance columns.
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
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
        X_scaled = (X_scaled - col_min) / col_range
        X_scaled = np.clip(X_scaled, 0.0, 1.0).astype(np.float32)

        # FeatureSelector returns ndarray for MI and DataFrame for recency;
        # GAMINET fits on either. Capture the shape consistently for logging.
        if hasattr(X_scaled, "shape"):
            n_cols = X_scaled.shape[1]
        else:
            n_cols = len(self.selected_features_)
        print(
            f"[GAMINET] Training with {n_cols} selected features "
            f"(mode={self.preproc_.mode}) | "
            f"interact_num={self.interact_num} | "
            f"max_epochs={self.max_epochs} | batch_size={self.batch_size}"
        )

        # yaml lists arrive as python lists; piml expects a 3-tuple.
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
        self.gaminet_.fit(X_scaled, y)
        self.classes_ = np.array([0, 1])
        return self

    def _transform_X(self, X):
        check_is_fitted(self, "gaminet_")
        Xt = self.preproc_.transform(X)
        Xt = np.asarray(Xt, dtype=np.float32)
        # Apply the same column filter + min-max scaling fit at training time.
        if hasattr(self, "_gaminet_keep_mask_"):
            Xt = Xt[:, self._gaminet_keep_mask_]
        Xt = (Xt - self._gaminet_col_min_) / self._gaminet_col_range_
        return np.clip(Xt, 0.0, 1.0).astype(np.float32)

    def predict_proba(self, X):
        proba = np.asarray(self.gaminet_.predict_proba(self._transform_X(X)))
        # NaN guard — GAMINET can emit NaNs if its fine-tune stage diverged.
        if not np.all(np.isfinite(proba)):
            proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
        # Some piml versions return 1-D; normalise to 2-D for sklearn parity.
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self):
        check_is_fitted(self, "gaminet_")
        return list(self.selected_features_)

    def get_gaminet(self):
        check_is_fitted(self, "gaminet_")
        return self.gaminet_
