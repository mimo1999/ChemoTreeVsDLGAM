import numpy as np

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
from sklearn.utils.validation import check_is_fitted

from utils.feature_selector import FeatureSelector


class LogisticGAMClassifier(BaseEstimator, ClassifierMixin):
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

    max_features : int
        Number of top features retained using mutual information.

    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lam=40.0,
        n_splines=4,
        spline_order=3,
        max_iter=150,
        max_features=200,
        random_state=42,
    ):
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.max_iter = max_iter
        self.max_features = max_features
        self.random_state = random_state

    # -----------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------

    def _build_terms(self, feature_names, cat_indices):

        term_list = []

        for i, fname in enumerate(feature_names):

            # categorical
            if i in cat_indices:
                term_list.append(f(i))

            # smooth only for demographics
            elif fname in ["age", "los"]:
                term_list.append(
                    s(
                        i,
                        n_splines=self.n_splines,
                        spline_order=self.spline_order,
                    )
                )

            # all labs/features linear
            else:
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
        """
        Fit GAM model.

        Parameters
        ----------
        X : array-like
            Feature matrix

        y : array-like
            Binary labels

        feature_names : list[str]
            Column names for X

        cat_feature_names : list[str]
            Names of categorical columns
            Example: ["gender"]
        """

        cat_feature_names = cat_feature_names or []

        # -------------------------------------------------------------
        # Feature selection + scaling (MI top-K → StandardScaler)
        # Delegates to the shared FeatureSelector so the GAM uses the
        # same picklable, fold-consistent pipeline as EBM / GAMINET.
        # -------------------------------------------------------------

        self.preproc_ = FeatureSelector(
            score_func="mutual_info_classif",
            max_features=self.max_features,
            random_state=self.random_state,
        )
        X_scaled = self.preproc_.fit_transform(X, y, feature_names=feature_names)
        self.selected_features_ = self.preproc_.get_selected_features()
        # Expose input feature names for diagnostics / _transform_X
        self.feature_names_in_ = list(self.preproc_.inner.feature_names_in_)

        # -------------------------------------------------------------
        # Identify categorical feature indices (post-selection)
        # -------------------------------------------------------------

        cat_indices = [
            idx for idx, fname in enumerate(self.selected_features_)
            if fname in cat_feature_names
        ]

        # -------------------------------------------------------------
        # Build GAM terms
        # -------------------------------------------------------------

        terms = self._build_terms(
            feature_names=self.selected_features_,
            cat_indices=cat_indices,
        )

        print(
            f"[GAM] Training with {X_scaled.shape[1]} selected features "
            f"(top-{self.max_features} by MI) | "
            f"lam={self.lam} | n_splines={self.n_splines}"
        )

        # -------------------------------------------------------------
        # Train GAM
        # -------------------------------------------------------------

        self.gam_ = LogisticGAM(
            terms,
            lam=self.lam,
            max_iter=self.max_iter,
        )

        self.gam_.fit(X_scaled, y)

        self.classes_ = np.array([0, 1])

        return self

    # -----------------------------------------------------------------
    # Prediction utilities
    # -----------------------------------------------------------------

    def _transform_X(self, X):

        check_is_fitted(self, "gam_")
        return self.preproc_.transform(X)

    # -----------------------------------------------------------------
    # Predict API
    # -----------------------------------------------------------------

    def predict_proba(self, X):

        X_scaled = self._transform_X(X)

        pos = self.gam_.predict_proba(X_scaled)
        pos = np.where(np.isfinite(pos), pos, 0.5)
        pos = np.clip(pos, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1 - pos, pos])

    def predict(self, X):

        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)

    def predict_log_proba(self, X):

        return np.log(
            np.clip(self.predict_proba(X), 1e-10, 1)
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_selected_features(self):
        """
        Return selected feature names.
        """

        check_is_fitted(self, "gam_")

        return self.selected_features_

    def get_gam_statistics(self):
        """
        Return pygam fit statistics.
        """

        check_is_fitted(self, "gam_")

        return self.gam_.statistics_