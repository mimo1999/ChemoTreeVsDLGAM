"""
GAM wrapper for the ChemoTreeVsDL pipeline.

Pipeline: StandardScaler -> LogisticGAM

Feature selection and NaN-filling are handled upstream by DataLoader.
"""

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix as _csr
from scipy.sparse import csc_matrix as _csc

# ---------------------------------------------------------------------
# Compatibility patches for pygam 0.8
# ---------------------------------------------------------------------

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

for _sp in (_csr, _csc):
    if not hasattr(_sp, "A"):
        _sp.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from pygam import LogisticGAM, s, f, l, terms as pt
from pygam.utils import OptimizationError as _GAMOptimizationError

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class LogisticGAMClassifier(ClassifierMixin, BaseEstimator):
    """
    Simplified GAM classifier for clinical tabular data.

    Feature selection and NaN-filling are handled upstream by DataLoader.
    This wrapper performs standard scaling, then fits a LogisticGAM with
    linear terms for all features except continuous demographics which get
    smooth spline terms.

    Parameters
    ----------
    lam : float
        GAM regularization strength. Higher values improve stability for
        high-dimensional data.
    n_splines : int
        Number of spline basis functions per feature.
    spline_order : int
        B-spline order (3 = cubic).
    max_iter : int
        Maximum PIRLS optimization iterations.
    """

    def __init__(
        self,
        lam=40.0,
        n_splines=4,
        spline_order=3,
        max_iter=150,
    ):
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.max_iter = max_iter

    # -----------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------

    # Substrings that identify continuous demographic features that benefit
    # from a smooth spline term. Everything else gets a linear term (l),
    # which is cheaper, more stable in PIRLS, and avoids spurious
    # nonlinearities on lab-value features that are effectively linear.
    _SPLINE_FEATURES = ("age", "los")

    def _build_terms(self, feature_names, cat_indices):

        term_list = []

        for i, fname in enumerate(feature_names):

            if i in cat_indices:
                term_list.append(f(i))

            elif any(key in fname.lower() for key in self._SPLINE_FEATURES):
                term_list.append(
                    s(
                        i,
                        n_splines=self.n_splines,
                        spline_order=self.spline_order,
                    )
                )

            else:
                # Lab values and all other features: linear term.
                # Keeps the PIRLS design matrix at n_features columns (not
                # n_features × n_splines), which is far more numerically stable.
                term_list.append(l(i))

        return pt.TermList(*term_list)

    # -----------------------------------------------------------------
    # sklearn API
    # -----------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        feature_names=None,
        cat_feature_names=None,
    ):
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            if feature_names is None:
                feature_names = list(X_df.columns)

        self.feature_names_in_ = list(feature_names)
        cat_feature_names = cat_feature_names or []
        self.selected_features_ = list(feature_names)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_df)

        cat_indices = [
            idx for idx, fname in enumerate(self.selected_features_)
            if fname in cat_feature_names
        ]

        terms = self._build_terms(
            feature_names=self.selected_features_,
            cat_indices=cat_indices,
        )

        print(
            f"[GAM] Training with {X_scaled.shape[1]} features | "
            f"lam={self.lam} | n_splines={self.n_splines}"
        )

        # Retry with increasing lam if PIRLS diverges.
        lam = self.lam
        _MAX_RETRIES = 2
        fitted = False
        for attempt in range(_MAX_RETRIES):
            self.gam_ = LogisticGAM(
                terms,
                lam=lam,
                max_iter=self.max_iter,
            )
            try:
                self.gam_.fit(X_scaled, y)
                if attempt > 0:
                    print(f"[GAM] Converged with lam={lam:.1f} (attempt {attempt + 1})")
                fitted = True
                break
            except (_GAMOptimizationError, np.linalg.LinAlgError):
                next_lam = lam * 50
                print(
                    f"[GAM] PIRLS diverged at lam={lam:.4g} — "
                    f"retrying with lam={next_lam:.4g}"
                )
                if hasattr(self.gam_, "coef_"):
                    del self.gam_.coef_
                lam = next_lam

        if not fitted:
            raise RuntimeError(
                f"[GAM] PIRLS failed to converge after {_MAX_RETRIES} retries "
                f"(final lam={lam:.4g}). Consider increasing self.lam or reducing "
                f"n_splines."
            )

        self.classes_ = np.array([0, 1])
        return self

    # -----------------------------------------------------------------
    # Prediction utilities
    # -----------------------------------------------------------------

    def _transform_X(self, X):
        check_is_fitted(self, "gam_")

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X.copy()

        return self.scaler_.transform(X_df)

    def predict_proba(self, X):
        X_scaled = self._transform_X(X)
        pos = self.gam_.predict_proba(X_scaled)
        # pygam can return NaN (degenerate coefs) or values outside [0,1]
        # due to float overflow; np.clip passes NaN through so sanitise first.
        pos = np.where(np.isfinite(pos), pos, 0.5)
        pos = np.clip(pos, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1))

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_selected_features(self):
        check_is_fitted(self, "gam_")
        return self.selected_features_

    def get_gam_statistics(self):
        check_is_fitted(self, "gam_")
        return self.gam_.statistics_
