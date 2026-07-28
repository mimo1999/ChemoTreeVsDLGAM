"""
Logistic GAM wrapper for the ChemoTreeVsDL pipeline, backed by pygam.

Pipeline: StandardScaler (numeric columns only) -> LogisticGAM

Each feature gets one term: a factor term for categorical columns (default
"gender"), a smooth spline for continuous demographics (default "age"), and
a plain linear term for everything else (lab values), which keeps the PIRLS
design matrix at n_features columns instead of n_features x n_splines. PIRLS
occasionally diverges on wide, correlated lab matrices; fit() retries with
escalating regularization (lam) when that happens.
"""

import numpy as np

from config.constants import RANDOM_SEED, STATIC_SPLINE_FEATURES, STATIC_CATEGORICAL_FEATURES

from pygam import LogisticGAM, s, f, l, terms as pt
from pygam.utils import OptimizationError as _GAMOptimizationError

from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ._base import BaseWrapperClassifier


class LogisticGAMClassifier(BaseWrapperClassifier):
    """
    Simplified GAM classifier for clinical tabular data.

    Parameters
    ----------
    lam : float
        GAM regularization strength.
        Higher values improve stability for high-dimensional data.

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
        random_state: int = RANDOM_SEED,
    ):
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.max_iter = max_iter
        self.random_state = random_state

    # -----------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------

    def _build_terms(self, feature_names, cat_indices, lam):

        term_list = []

        for i, fname in enumerate(feature_names):

            if i in cat_indices:
                # Categorical: factor (step-function) term only
                term_list.append(f(i, lam=lam))

            elif any(key in fname.lower() for key in STATIC_SPLINE_FEATURES):
                # Continuous demographic: smooth spline
                term_list.append(
                    s(
                        i,
                        n_splines=self.n_splines,
                        spline_order=self.spline_order,
                        lam=lam,
                    )
                )

            else:
                # Lab values and all other features: linear term. Keeps the
                # PIRLS design matrix at n_features columns instead of
                # n_features × n_splines, which is more numerically stable.
                # lam is passed explicitly since pygam terms default to their
                # own lam=0.6 and don't inherit LogisticGAM's lam= kwarg.
                term_list.append(l(i, lam=lam))

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
        """
        Fit GAM model.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix

        y : array-like
            Binary labels

        feature_names : list[str] or None
            Column names for X. Defaults to X.columns when not given (e.g.
            when scikit-learn's GridSearchCV calls fit() without it).

        cat_feature_names : list[str] or None
            Names of categorical columns. Defaults to
            config.constants.STATIC_CATEGORICAL_FEATURES (currently
            ["gender"]) when not given.
        """
        X_df = X.copy()
        if feature_names is None:
            feature_names = list(X_df.columns)

        self.feature_names_in_ = list(feature_names)
        cat_feature_names = (
            list(STATIC_CATEGORICAL_FEATURES) if cat_feature_names is None else cat_feature_names
        )

        self.selected_features_ = list(feature_names)

        # -------------------------------------------------------------
        # Identify categorical feature indices
        # -------------------------------------------------------------

        cat_indices = []

        for idx, fname in enumerate(self.selected_features_):

            if fname in cat_feature_names:
                cat_indices.append(idx)

        self.cat_indices_ = cat_indices
        self.numeric_indices_ = [
            i for i in range(len(self.selected_features_)) if i not in cat_indices
        ]

        # -------------------------------------------------------------
        # Standardize numeric (non-categorical) columns. Raw VMD columns mix
        # ~[0,1] flags/ratios with lab values spanning into the tens of
        # thousands (e.g. NT-proBNP), which overflows PIRLS's exp() on the
        # first iteration. Categorical columns are left raw since f() factor
        # terms expect integer category codes, not standardized floats.
        # -------------------------------------------------------------

        X_arr = X_df.values.astype(float)
        self.scaler_ = StandardScaler()
        if self.numeric_indices_:
            X_arr[:, self.numeric_indices_] = self.scaler_.fit_transform(
                X_arr[:, self.numeric_indices_]
            )

        # -------------------------------------------------------------
        # Build GAM terms
        # -------------------------------------------------------------

        print(
            f"[GAM] Training with "
            f"{X_arr.shape[1]} features | "
            f"lam={self.lam} | "
            f"n_splines={self.n_splines}"
        )

        # -------------------------------------------------------------
        # Train GAM  (retry with increasing lam if PIRLS diverges)
        # -------------------------------------------------------------

        np.random.seed(self.random_state)
        lam = self.lam
        _MAX_RETRIES = 2          # lam grows as: lam * 50^attempt
        fitted = False
        for attempt in range(_MAX_RETRIES):
            terms = self._build_terms(
                feature_names=self.selected_features_,
                cat_indices=cat_indices,
                lam=lam,
            )
            self.gam_ = LogisticGAM(
                terms,
                lam=lam,
                max_iter=self.max_iter,
            )
            try:
                self.gam_.fit(X_arr, y)
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
        X_arr = X.values.astype(float)
        if self.numeric_indices_:
            X_arr[:, self.numeric_indices_] = self.scaler_.transform(
                X_arr[:, self.numeric_indices_]
            )

        return X_arr

    # -----------------------------------------------------------------
    # Predict API
    # -----------------------------------------------------------------

    def predict_proba(self, X):

        X_scaled = self._transform_X(X)

        pos = self.gam_.predict_proba(X_scaled)

        # pygam can return NaN (PIRLS converged to degenerate coefs) or values
        # just outside [0, 1] due to float overflow in the link function;
        # np.clip passes NaN through unchanged so we sanitise first.
        pos = np.where(np.isfinite(pos), pos, 0.5)
        pos = np.clip(pos, 1e-6, 1.0 - 1e-6)

        return np.column_stack([1 - pos, pos])

    # predict() / predict_log_proba() / get_selected_features() are inherited
    # from BaseWrapperClassifier.

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_gam_statistics(self):
        check_is_fitted(self, "gam_")
        return self.gam_.statistics_
