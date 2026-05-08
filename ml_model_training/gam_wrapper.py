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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


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
        lam=100.0,
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

        # -------------------------------------------------------------
        # Convert to DataFrame
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Imputation
        # -------------------------------------------------------------

        self.imputer_ = SimpleImputer(strategy="median")

        X_imp = pd.DataFrame(
            self.imputer_.fit_transform(X_df),
            columns=feature_names,
        )

        # -------------------------------------------------------------
        # Feature selection
        # -------------------------------------------------------------

        k = min(self.max_features, X_imp.shape[1])

        self.selector_ = SelectKBest(
            score_func=mutual_info_classif,
            k=k,
        )

        X_sel = self.selector_.fit_transform(X_imp, y)

        selected_mask = self.selector_.get_support()

        self.selected_features_ = [
            f for f, keep in zip(feature_names, selected_mask) if keep
        ]

        # -------------------------------------------------------------
        # Scaling
        # -------------------------------------------------------------

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X_sel)

        # -------------------------------------------------------------
        # Identify categorical feature indices
        # -------------------------------------------------------------

        cat_indices = []

        for idx, fname in enumerate(self.selected_features_):

            if fname in cat_feature_names:
                cat_indices.append(idx)

        # -------------------------------------------------------------
        # Build GAM terms
        # -------------------------------------------------------------

        terms = self._build_terms(
            feature_names=self.selected_features_,
            cat_indices=cat_indices,
        )

        print(
            f"[GAM] Training with "
            f"{X_scaled.shape[1]} selected features | "
            f"lam={self.lam} | "
            f"n_splines={self.n_splines}"
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

        if isinstance(X, np.ndarray):

            X_df = pd.DataFrame(
                X,
                columns=self.feature_names_in_,
            )

        else:
            X_df = X.copy()

        X_imp = self.imputer_.transform(X_df)

        X_sel = self.selector_.transform(X_imp)

        X_scaled = self.scaler_.transform(X_sel)

        return X_scaled

    # -----------------------------------------------------------------
    # Predict API
    # -----------------------------------------------------------------

    def predict_proba(self, X):

        X_scaled = self._transform_X(X)

        pos = self.gam_.predict_proba(X_scaled)

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