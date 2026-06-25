"""
M-GAM (Missingness-Aware GAM) wrapper for the ChemoTreeVsDL pipeline.

Designed for VMD feature matrices where columns follow the convention:
    X_{itemid}_{bin}  — V: mean value, 0 when absent
    D_{itemid}_{bin}  — D: delta/trend, 0 when absent
    M_{itemid}_{bin}  — M: missingness indicator, 1 when absent

The VMD encoding already encodes absence (V=0, D=0, M=1), so no imputation is
needed for lab columns.  VD are the continuous base features; M columns are
explicit missingness intercepts.

M-GAM interaction terms
-----------------------
For each itemid i and each pair of bins (b_miss, b_obs) where b_miss != b_obs,
emit:

    M_{i, b_miss} × V_{i, b_obs}   — "item i absent in bin b_miss; what was
    M_{i, b_miss} × D_{i, b_obs}     its value / trend in bin b_obs?"

These within-item temporal interactions capture whether missingness in one
time window is informative given the measurement in another window.  Same-bin
interactions (b_miss == b_obs) are identically zero under VMD convention and
are skipped.

Demographic columns not matching the VMD prefix pattern pass through as
continuous features and are median-imputed when NaN.

Backend: fastsparsegams L0L2-penalized logistic regression on continuous VMD
features, called directly in-process.  CARMA (the C++ numpy bridge) requires
arrays to be F-contiguous AND own their memory — sliced views of F-contiguous
arrays are non-owning strides and are rejected.  All inputs are materialized
via np.asfortranarray() before each fit() call.  Retries with escalating
gamma_max + shrinking support/num_lambda on crashes caused by degenerate fold
splits.

Lambda selection
----------------
fastsparsegams.fit() returns a full regularization path (num_lambda solutions).
fit() holds out val_fraction of the training data, scores each solution by
AUC-ROC on that split, and returns the best lambda_0 + coefficients.

Pipeline: SimpleImputer (demographics only) → M×VD interaction augmentation
          → fastsparsegams L0L2 logistic → coeff
"""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import fastsparsegams

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


def _fit_fastsparsegams(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    gamma_max: float,
    **kwargs,
) -> dict:
    """Fit fastsparsegams in-process and return selected coefficients.

    Arrays must be materialized as new F-contiguous copies before calling —
    sliced views of F-contiguous arrays are non-owning strides that CARMA
    (fastsparsegams' C++ bridge) rejects with a RuntimeError.
    """
    X_tr = np.asfortranarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)

    fit_model = fastsparsegams.fit(
        X_tr, y_tr, loss="Logistic", penalty="L0L2",
        gamma_max=gamma_max, gamma_min=gamma_max * 0.1,
        **kwargs,
    )

    lambdas  = fit_model.lambda_0[0]
    y_val_01 = ((y_val + 1) // 2).astype(int)
    best_auc = -1.0
    best_lam = lambdas[-1]

    for lam in lambdas:
        proba = np.asarray(fit_model.predict(X_val, lam)).ravel()
        if not np.isfinite(proba).all():
            continue
        auc = roc_auc_score(y_val_01, proba)
        if auc > best_auc:
            best_auc = auc
            best_lam = lam

    lam_idx = list(lambdas).index(best_lam)
    support = fit_model.support_size[0][lam_idx]
    coeff   = np.asarray(fit_model.coeff(best_lam, include_intercept=True).todense()).ravel()

    return {"coeff": coeff, "best_lambda": best_lam, "best_val_auc": best_auc, "support": support}


class MGAMClassifier(ClassifierMixin, BaseEstimator):
    """M-GAM missingness-aware L0-penalized classifier for VMD feature matrices.

    Parameters
    ----------
    max_support_size : int, default 40
        Maximum number of non-zero coefficients.  Acts as the sparsity ceiling —
        smaller values produce sparser, more interpretable models.
    num_lambda : int, default 50
        Number of lambda values in the regularization path.
    algorithm : str, default "CD"
        "CD" is coordinate descent.  "CDPSI" adds a local combinatorial
        search for higher quality solutions at the cost of stability.
    val_fraction : float, default 0.15
        Fraction of training data held out (stratified) for lambda selection
        by AUC-ROC.  Not used for coefficient estimation.
    max_iter : int, default 200
        Maximum CD iterations per grid point.
    random_state : int, default 42
    """

    def __init__(
        self,
        max_support_size: int = 40,
        num_lambda: int = 50,
        algorithm: str = "CD",
        val_fraction: float = 0.15,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.max_support_size = max_support_size
        self.num_lambda = num_lambda
        self.algorithm = algorithm
        self.val_fraction = val_fraction
        self.max_iter = max_iter
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Column parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vmd_columns(feature_names):
        """Split column names into V, D, M maps and non-VMD remainder.

        Returns
        -------
        v_map : dict[(itemid_int, bin_str)] -> col_name   (X_ prefix)
        d_map : dict[(itemid_int, bin_str)] -> col_name   (D_ prefix)
        m_map : dict[(itemid_int, bin_str)] -> col_name   (M_ prefix)
        other : list of column names that don't follow the VMD convention
        """
        v_map: dict = {}
        d_map: dict = {}
        m_map: dict = {}
        other: list = []

        for col in feature_names:
            parts = col.split("_")
            if len(parts) >= 3 and parts[0] in ("X", "D", "M"):
                try:
                    itemid = int(parts[1])
                    bin_id = parts[2]
                except (ValueError, IndexError):
                    other.append(col)
                    continue
                key = (itemid, bin_id)
                if parts[0] == "X":
                    v_map[key] = col
                elif parts[0] == "D":
                    d_map[key] = col
                else:
                    m_map[key] = col
            else:
                other.append(col)

        return v_map, d_map, m_map, other

    # ------------------------------------------------------------------
    # Interaction construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_interactions(
        X_df: pd.DataFrame,
        v_map: dict,
        d_map: dict,
        m_map: dict,
    ) -> pd.DataFrame:
        """Emit M_{i,b_miss} × V_{i,b_obs} and M_{i,b_miss} × D_{i,b_obs}.

        Only cross-bin pairs (b_miss != b_obs) within the same itemid are
        considered.  Interaction columns that are entirely zero (itemid never
        missing, or complementary bin never observed) are dropped.
        """
        itemid_bins: dict[int, set] = defaultdict(set)
        for (itemid, bin_id) in m_map:
            itemid_bins[itemid].add(bin_id)

        cols: dict[str, np.ndarray] = {}

        for itemid, bins in itemid_bins.items():
            if len(bins) < 2:
                continue
            for b_miss in sorted(bins):
                m_col = m_map.get((itemid, b_miss))
                if m_col is None or m_col not in X_df.columns:
                    continue
                m_vals = X_df[m_col].to_numpy(dtype=np.float64)
                if m_vals.sum() == 0:
                    continue
                for b_obs in sorted(bins):
                    if b_obs == b_miss:
                        continue
                    v_col = v_map.get((itemid, b_obs))
                    if v_col is not None and v_col in X_df.columns:
                        arr = m_vals * X_df[v_col].to_numpy(dtype=np.float64)
                        if arr.sum() != 0:
                            cols[f"ixn_M{itemid}_{b_miss}_X{itemid}_{b_obs}"] = arr
                    d_col = d_map.get((itemid, b_obs))
                    if d_col is not None and d_col in X_df.columns:
                        arr = m_vals * X_df[d_col].to_numpy(dtype=np.float64)
                        if arr.sum() != 0:
                            cols[f"ixn_M{itemid}_{b_miss}_D{itemid}_{b_obs}"] = arr

        if not cols:
            return pd.DataFrame(index=X_df.index)
        return pd.DataFrame(cols, index=X_df.index, dtype=np.float64)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        feature_names: Iterable[str] | None = None,
        sample_weight=None,       # accepted for API compat; fastsparsegams has no sample_weight
    ):
        if feature_names is None:
            feature_names = (
                list(X.columns) if hasattr(X, "columns")
                else [f"f{i}" for i in range(X.shape[1])]
            )
        self.feature_names_in_ = list(feature_names)

        X_df = pd.DataFrame(
            np.asarray(X, dtype=np.float64),
            columns=self.feature_names_in_,
        )

        v_map, d_map, m_map, other_cols = self._parse_vmd_columns(self.feature_names_in_)
        self.v_map_      = v_map
        self.d_map_      = d_map
        self.m_map_      = m_map
        self.other_cols_ = other_cols

        if other_cols:
            self.imputer_ = SimpleImputer(strategy="median")
            X_df[other_cols] = self.imputer_.fit_transform(X_df[other_cols])
        else:
            self.imputer_ = None

        ixn_df = self._build_interactions(X_df, v_map, d_map, m_map)
        self.interaction_col_names_ = list(ixn_df.columns)

        X_aug = (
            pd.concat([X_df, ixn_df], axis=1)
            if not ixn_df.empty
            else X_df.copy()
        )
        self.augmented_col_names_ = list(X_aug.columns)

        X_arr = np.asarray(X_aug, dtype=np.float64)

        # fastsparsegams requires labels in {-1, +1}
        y_pm = (np.asarray(y).ravel() * 2 - 1).astype(int)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_pm,
            test_size=self.val_fraction,
            stratify=y_pm,
            random_state=self.random_state,
        )

        # fastsparsegams crashes when max_support_size >= n_features; cap it.
        effective_support = min(self.max_support_size, X_arr.shape[1] - 1)

        print(
            f"[MGAM] {len(self.feature_names_in_)} base features + "
            f"{len(self.interaction_col_names_)} M×VD interactions "
            f"= {X_arr.shape[1]} total | "
            f"max_support_size={effective_support} | algorithm={self.algorithm}"
        )

        # Retry schedule: escalate gamma AND reduce solver complexity on each crash.
        # gamma_max=1e-4 is negligible vs L0; larger gamma adds L2 convexity.
        # Reducing num_lambda and max_support_size shrinks the search space to
        # avoid degenerate coordinate-descent states on difficult fold splits.
        _retry_schedule = [
            dict(gamma_max=1e-4, num_lambda=self.num_lambda,  max_support_size=effective_support),
            dict(gamma_max=1e-2, num_lambda=30,               max_support_size=min(effective_support, 30)),
            dict(gamma_max=1e-1, num_lambda=20,               max_support_size=min(effective_support, 20)),
            dict(gamma_max=5e-1, num_lambda=10,               max_support_size=min(effective_support, 10)),
        ]
        result = None
        for attempt in _retry_schedule:
            try:
                result = _fit_fastsparsegams(
                    X_tr, X_val, y_tr, y_val,
                    gamma_max=attempt["gamma_max"],
                    algorithm=self.algorithm,
                    max_support_size=attempt["max_support_size"],
                    num_lambda=attempt["num_lambda"],
                    num_gamma=1,
                    max_iter=self.max_iter,
                    intercept=True,
                )
                break
            except RuntimeError:
                print(
                    f"[MGAM] fit crashed "
                    f"(gamma={attempt['gamma_max']:.0e}, "
                    f"support={attempt['max_support_size']}, "
                    f"nlam={attempt['num_lambda']}), retrying..."
                )
        if result is None:
            raise RuntimeError("[MGAM] fastsparsegams crashed on all retry attempts.")

        self.coeff_        = result["coeff"]          # shape (n_features + 1,) — intercept first
        self.best_lambda_  = result["best_lambda"]
        self.best_val_auc_ = result["best_val_auc"]
        self.support_      = result["support"]
        self.classes_      = np.array([0, 1])

        print(
            f"[MGAM] Selected lambda={self.best_lambda_:.4f} "
            f"(support={self.support_}, val AUC={self.best_val_auc_:.4f})"
        )
        return self

    def _augment(self, X) -> np.ndarray:
        check_is_fitted(self, "coeff_")
        X_df = pd.DataFrame(
            np.asarray(X, dtype=np.float64),
            columns=self.feature_names_in_,
        )
        if self.imputer_ is not None and self.other_cols_:
            X_df[self.other_cols_] = self.imputer_.transform(X_df[self.other_cols_])
        ixn_df = self._build_interactions(X_df, self.v_map_, self.d_map_, self.m_map_)
        if not ixn_df.empty:
            ixn_df = ixn_df.reindex(columns=self.interaction_col_names_, fill_value=0.0)
            X_aug = pd.concat([X_df, ixn_df], axis=1)
        else:
            X_aug = X_df
        return np.asarray(X_aug, dtype=np.float64)

    def predict_proba(self, X):
        X_scaled = self._augment(X)
        # coeff_ layout: [intercept, w_1, ..., w_p] — prepend ones column
        X_with_intercept = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
        activations = X_with_intercept @ self.coeff_
        pos = np.clip(1.0 / (1.0 + np.exp(-activations)), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> list:
        """Return all column names after M×VD interaction augmentation."""
        check_is_fitted(self, "coeff_")
        return list(self.augmented_col_names_)

    def get_interaction_terms(self) -> list:
        check_is_fitted(self, "coeff_")
        return list(self.interaction_col_names_)

    def get_nonzero_features(self) -> list:
        """Return names of features with non-zero L0 coefficients."""
        check_is_fitted(self, "coeff_")
        # coeff_[0] is the intercept; remaining align with augmented_col_names_
        weights = self.coeff_[1:]
        return [n for n, w in zip(self.augmented_col_names_, weights) if w != 0.0]
