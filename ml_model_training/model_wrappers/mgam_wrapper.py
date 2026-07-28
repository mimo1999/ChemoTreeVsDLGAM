"""
M-GAM (Missingness-Aware GAM) wrapper for the ChemoTreeVsDL pipeline.

Column convention
-----------------
    X_{itemid}_{bin}  — mean value in bin (0 when absent)
    D_{itemid}_{bin}  — time since last observation in bin (0 iff present, >0 while missing)
    M_{itemid}_{bin}  — explicit missingness flag (optional; not used in interaction construction)

Since D == 0 iff the item was observed, D directly encodes missingness.  No
separate M column is required for the interaction terms below.  If M columns
are present (e.g. feat_type=VMD) they pass through as base features.

Interaction terms
-----------------
For each lab item i and each cross-bin pair (b_miss, b_obs) where b_miss ≠ b_obs:

    [D_{i, b_miss} > 0] × V_{i, b_obs}

This captures: "item i was missing in window b_miss — what was its value in
window b_obs?"  Same-bin pairs are always zero under this convention and are
skipped.  Zero-valued interaction columns (item never missing, or the paired
bin never observed) are also dropped.

Backend: fastsparsegams L0L2-penalized logistic regression.
    • CARMA (fastsparsegams' C++ bridge) requires arrays that are F-contiguous
      AND own their memory.  Sliced views are rejected; every array is
      materialized via np.asfortranarray() before fitting.
    • fastsparsegams crashes on degenerate fold splits.  fit() retries with
      escalating L2 regularization (gamma_max) and shrinking solver complexity.

Lambda selection: fastsparsegams returns a full regularization path.  The best
lambda is chosen by AUC-ROC on a stratified hold-out split (val_fraction of the
training data).

Pipeline:  D-gated V interaction augmentation
          → fastsparsegams L0L2 logistic → coeff_
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import fastsparsegams

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from config.constants import RANDOM_SEED
from utils.feature_naming import parse_vmd_column
from ._base import BaseWrapperClassifier

# ---------------------------------------------------------------------------
# fastsparsegams solver
# ---------------------------------------------------------------------------

def _fit_and_select_lambda(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    gamma_max: float,
    **kwargs,
) -> tuple[np.ndarray, float, float, int]:
    """Fit one fastsparsegams run and return the best-AUC lambda's solution.

    X_tr must be F-contiguous and own its memory (CARMA requirement).
    y_tr / y_val must be in {-1, +1} (fastsparsegams convention).

    Returns (coeff, best_lambda, best_val_auc, support_size).
    coeff shape: (n_features + 1,) with the intercept as the first element.
    """
    X_tr = np.asfortranarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)

    path = fastsparsegams.fit(
        X_tr, y_tr, loss="Logistic", penalty="L0L2",
        gamma_max=gamma_max, gamma_min=gamma_max * 0.1,
        **kwargs,
    )

    lambdas   = path.lambda_0[0]
    y_val_bin = (y_val > 0).astype(int)   # {-1,+1} → {0,1} for roc_auc_score
    best_auc  = -1.0
    best_lam  = lambdas[-1]

    for lam in lambdas:
        proba = np.asarray(path.predict(X_val, lam)).ravel()
        if not np.isfinite(proba).all():
            continue
        auc = roc_auc_score(y_val_bin, proba)
        if auc > best_auc:
            best_auc = auc
            best_lam = lam

    lam_idx = list(lambdas).index(best_lam)
    support = path.support_size[0][lam_idx]
    coeff   = np.asarray(path.coeff(best_lam, include_intercept=True).todense()).ravel()

    return coeff, best_lam, best_auc, support


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class MGAMClassifier(BaseWrapperClassifier):
    """M-GAM missingness-aware L0-penalized classifier for VMD feature matrices.

    Parameters
    ----------
    max_support_size : int, default 40
        Maximum number of non-zero coefficients (sparsity ceiling).
    num_lambda : int, default 50
        Number of lambda values in the regularization path.
    algorithm : str, default "CD"
        "CD" = coordinate descent.  "CDPSI" adds a local combinatorial search
        for higher-quality solutions at the cost of stability.
    val_fraction : float, default 0.15
        Fraction of training data held out (stratified) for lambda selection.
        Not used in coefficient estimation.
    max_iter : int, default 200
        Maximum coordinate-descent iterations per grid point.
    random_state : int, default RANDOM_SEED
    """

    def __init__(
        self,
        max_support_size: int = 40,
        num_lambda: int = 50,
        algorithm: str = "CD",
        val_fraction: float = 0.15,
        max_iter: int = 200,
        random_state: int = RANDOM_SEED,
    ):
        self.max_support_size = max_support_size
        self.num_lambda       = num_lambda
        self.algorithm        = algorithm
        self.val_fraction     = val_fraction
        self.max_iter         = max_iter
        self.random_state     = random_state

    # ------------------------------------------------------------------
    # Column parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vmd_columns(feature_names):
        """Partition column names into V-map, D-map, and non-VMD remainder.

        Returns
        -------
        v_map  : {(itemid, bin) -> col_name}  columns with X_ prefix
        d_map  : {(itemid, bin) -> col_name}  columns with D_ prefix
        other  : list of column names that don't follow the VD/VMD pattern
        """
        v_map = {}
        d_map = {}
        other = []

        for col in feature_names:
            parsed = parse_vmd_column(col)
            if parsed is None or parsed[2] is None:
                # Unmatched, or a channel-matched column with no bin (e.g. an
                # aggregate feature_method's X_<itemid>): no cross-bin axis
                # for the D-gated interaction terms below.
                other.append(col)
                continue
            channel, itemid, bin_id = parsed
            key = (itemid, bin_id)
            if channel == "X":
                v_map[key] = col
            elif channel == "D":
                d_map[key] = col
            # M columns: recognised as VMD but not stored — pass through as base features

        return v_map, d_map, other

    # ------------------------------------------------------------------
    # Interaction construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_interactions(X_df: pd.DataFrame, v_map: dict, d_map: dict) -> pd.DataFrame:
        """Build [D_{i,b_miss} > 0] × V_{i,b_obs} cross-bin interaction columns.

        Only cross-bin pairs (b_miss ≠ b_obs) within the same lab item are
        considered.  Interaction columns that are entirely zero are dropped.
        """
        # Collect all D-observed bins per item
        item_bins = defaultdict(set)
        for itemid, bin_id in d_map:
            item_bins[itemid].add(bin_id)

        interactions = {}

        for itemid, bins in item_bins.items():
            if len(bins) < 2:
                continue
            for b_miss in sorted(bins):
                d_col = d_map.get((itemid, b_miss))
                if d_col is None or d_col not in X_df.columns:
                    continue
                miss_mask = (X_df[d_col].to_numpy(dtype=np.float64) > 0).astype(np.float64)
                if miss_mask.sum() == 0:
                    continue
                for b_obs in sorted(bins):
                    if b_obs == b_miss:
                        continue
                    v_col = v_map.get((itemid, b_obs))
                    if v_col is not None and v_col in X_df.columns:
                        values = miss_mask * X_df[v_col].to_numpy(dtype=np.float64)
                        if values.sum() != 0:
                            interactions[f"ixn_D{itemid}_{b_miss}_X{itemid}_{b_obs}"] = values

        if not interactions:
            return pd.DataFrame(index=X_df.index)
        return pd.DataFrame(interactions, index=X_df.index, dtype=np.float64)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, feature_names=None, sample_weight=None):
        # sample_weight accepted for API compatibility; fastsparsegams has no sample_weight support.
        if feature_names is None:
            feature_names = list(X.columns)
        self.feature_names_in_ = list(feature_names)

        X_df = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=self.feature_names_in_)

        v_map, d_map, _ = self._parse_vmd_columns(self.feature_names_in_)
        self.v_map_ = v_map
        self.d_map_ = d_map

        ixn_df = self._build_interactions(X_df, v_map, d_map)
        self.interaction_col_names_ = list(ixn_df.columns)

        X_aug = pd.concat([X_df, ixn_df], axis=1) if not ixn_df.empty else X_df
        self.augmented_col_names_ = list(X_aug.columns)

        X_arr = np.asarray(X_aug, dtype=np.float64)
        # fastsparsegams requires labels in {-1, +1}
        y_pm  = (np.asarray(y).ravel() * 2 - 1).astype(int)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_pm,
            test_size=self.val_fraction,
            stratify=y_pm,
            random_state=self.random_state,
        )

        # Cap support at n_features - 1; fastsparsegams crashes if support >= n_features.
        max_support = min(self.max_support_size, X_arr.shape[1] - 1)

        print(
            f"[MGAM] {len(self.feature_names_in_)} base features + "
            f"{len(self.interaction_col_names_)} interactions "
            f"= {X_arr.shape[1]} total | "
            f"max_support={max_support} | algorithm={self.algorithm}"
        )

        # Retry schedule: each attempt escalates L2 regularization (gamma_max adds
        # convexity to stabilize CD) and shrinks the path to reduce degenerate states.
        retry_schedule = [
            (1e-4, self.num_lambda, max_support),
            (1e-2, 30,              min(max_support, 30)),
            (1e-1, 20,              min(max_support, 20)),
            (5e-1, 10,              min(max_support, 10)),
        ]

        fitted = False
        for gamma, n_lam, n_support in retry_schedule:
            try:
                coeff, best_lam, best_auc, support = _fit_and_select_lambda(
                    X_tr, X_val, y_tr, y_val,
                    gamma_max=gamma,
                    algorithm=self.algorithm,
                    max_support_size=n_support,
                    num_lambda=n_lam,
                    num_gamma=1,
                    max_iter=self.max_iter,
                    intercept=True,
                )
                fitted = True
                break
            except RuntimeError:
                print(f"[MGAM] fit crashed (gamma={gamma:.0e}, support={n_support}, nlam={n_lam}), retrying...")

        if not fitted:
            raise RuntimeError("[MGAM] fastsparsegams crashed on all retry attempts.")

        self.coeff_        = coeff        # shape (n_features + 1,), intercept first
        self.best_lambda_  = best_lam
        self.best_val_auc_ = best_auc
        self.support_      = support
        self.classes_      = np.array([0, 1])

        print(f"[MGAM] Selected lambda={self.best_lambda_:.4f} (support={self.support_}, val AUC={self.best_val_auc_:.4f})")
        return self

    def _prepare_X(self, X) -> np.ndarray:
        """Apply interaction augmentation to inference data, aligned to the training schema."""
        check_is_fitted(self, "coeff_")
        X_df   = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=self.feature_names_in_)
        ixn_df = self._build_interactions(X_df, self.v_map_, self.d_map_)
        if not ixn_df.empty:
            ixn_df = ixn_df.reindex(columns=self.interaction_col_names_, fill_value=0.0)
            X_aug  = pd.concat([X_df, ixn_df], axis=1)
        else:
            X_aug  = X_df
        return np.asarray(X_aug, dtype=np.float64)

    def predict_proba(self, X):
        X_aug = self._prepare_X(X)
        # coeff_ layout: [intercept, w_1, ..., w_p] — prepend a ones column
        logits = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug]) @ self.coeff_
        pos    = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - pos, pos])

    # predict() / predict_log_proba() are inherited from BaseWrapperClassifier.

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> list:
        """Return all column names after D×V interaction augmentation.

        Overrides BaseWrapperClassifier's default since MGAM's "selected
        features" includes the generated interaction columns, not just the
        base feature_names it was fit with.
        """
        check_is_fitted(self, "coeff_")
        return list(self.augmented_col_names_)
