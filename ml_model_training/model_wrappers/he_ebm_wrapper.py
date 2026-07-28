"""
Hierarchical Expert-Based Explainable Boosting Machine (HE-EBM).

Architecture
------------
Up to five EBM experts are trained on disjoint feature subsets partitioned by
specimen × category from d_labitems:

    Expert 0  –  Blood × Chemistry
    Expert 1  –  Blood × Hematology
    Expert 2  –  Urine (any category)
    Expert 3  –  Blood Gas (any specimen)
    Expert 4  –  Other Fluids + demographics (catch-all)

Each expert runs its own internal TreeFeatureSelector (top-k RF importance)
before training, so the EBM only sees the most predictive features within its
group.  This is designed to run with global feature selection disabled
(--feature_selection 0), so all group features are available to each selector.

Combining the experts (two-step; no expert is ever hard-excluded)
-----------------------------------------------------------------
Step 1  Cross-fit each expert (K-fold) to produce out-of-fold (OOF) logits for
        every training row.  These are honest, test-like predictions — an
        expert that only *overfits* its group cannot look good here.  The fold
        models are retained for bagged inference.

Step 2  Fine-tune a small combiner on the OOF logits:

            final_logit = b + Σ_i  w_i · expert_logit_i ,     w_i ≥ 0

        The weights are fit by convex non-negative logistic regression
        (weighted BCE + L2). Non-negativity keeps every w_i a "trust this
        expert more/less" knob that can never invert an expert's signal; an
        expert that adds nothing on honest OOF logits simply collapses toward
        w_i ≈ 0.

Final prediction
----------------
    expert logits come from the bagged OOF fold models (their distribution
    matches what the combiner was trained on), then:

        final_logit = b + Σ_i w_i · expert_logit_i
        p = sigmoid(final_logit)
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from interpret.glassbox import ExplainableBoostingClassifier

from config.constants import RANDOM_SEED, PROJECT_ROOT, BEST_PARAMS
from utils.feature_selector import TreeFeatureSelector
from utils.feature_naming import parse_vmd_column
from ._base import BaseWrapperClassifier


# ── Constants ────────────────────────────────────────────────────────────────

N_EXPERTS = 5
EXPERT_NAMES = [
    "Blood Chemistry",
    "Blood Hematology",
    "Urine",
    "Blood Gas",
    "Other Fluids",
]


def _expert_itemid(col_name) -> "int | None":
    """Extract the itemid used for expert assignment, or None for a static
    covariate (e.g. age/gender). Ignores channel and bin -- expert
    partitioning only cares which lab item a column belongs to, whether it's
    a per-bin (X_<itemid>_<bin>, feature_method=concatenate) or an
    already-aggregated (X_<itemid>, feature_method=aggregate) column."""
    parsed = parse_vmd_column(col_name)
    return parsed[1] if parsed is not None else None


# ── Feature partitioning ──────────────────────────────────────────────────────

def _default_labitems_path() -> Path:
    return Path(PROJECT_ROOT) / "MIMIC_IV" / "saved_data" / "features" / "d_labitems.csv.gz"


@lru_cache(maxsize=8)
def _load_labitems_lookup(labitems_path: str) -> dict[int, tuple[str, str]]:
    """
    Parse d_labitems.csv.gz into an {itemid: (fluid, category)} lookup, cached
    per path. The table is static across folds/models, so this avoids
    re-reading and re-parsing it on every HEEBMClassifier.fit() call.
    """
    df = pd.read_csv(labitems_path)
    return {row.itemid: (row.fluid, row.category) for row in df.itertuples()}


def _assign_expert(fluid: str, category: str) -> int:
    """Map a (fluid, category) pair to one of 5 expert indices. First match wins."""
    if fluid == "Blood" and category == "Chemistry":
        return 0
    if fluid == "Blood" and category == "Hematology":
        return 1
    if fluid == "Urine":
        return 2
    if category == "Blood Gas":
        return 3
    return 4


def build_expert_masks(feature_names: list[str], labitems_path) -> list[list[int]]:
    """
    Partition global column indices into 5 expert groups.

    Non-lab columns (age, gender) and itemids absent from d_labitems fall into
    Expert 4 (catch-all).
    """
    lookup = _load_labitems_lookup(str(labitems_path))

    masks: list[list[int]] = [[] for _ in range(N_EXPERTS)]
    for col_idx, col in enumerate(feature_names):
        itemid = _expert_itemid(col)
        if itemid is not None:
            fluid_cat = lookup.get(itemid)
            expert_id = _assign_expert(*fluid_cat) if fluid_cat is not None else 4
        else:
            expert_id = 4  # demographics (age, gender)
        masks[expert_id].append(col_idx)

    return masks


# ── Combiner: non-negative logistic regression over expert OOF logits ─────────

def fit_expert_weights(
    oof_logits: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    l2: float = 1e-2,
) -> tuple[float, np.ndarray]:
    """
    Fit  p = sigmoid(b + Z @ w)  with  w ≥ 0,  minimising weighted BCE + l2·‖w‖².

    Convex in (b, w) with a box-constrained w, so L-BFGS-B reaches the global
    optimum deterministically — no warm-init, no learning-rate/epoch tuning.

    Non-negativity is the key inductive bias: each w_i is a "how much to trust
    expert i" knob. On honest OOF logits an unhelpful expert cannot reduce the
    loss, so its weight is driven toward 0.

    Returns
    -------
    bias : float
    weights : np.ndarray[float32], shape [n_experts], all ≥ 0
    """
    Z = np.asarray(oof_logits, dtype=float)
    n, k = Z.shape
    y = np.asarray(y, dtype=float)
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    sw = sw / sw.mean()

    def nll_and_grad(theta):
        b, w = theta[0], theta[1:]
        p = 1.0 / (1.0 + np.exp(-(b + Z @ w)))
        eps = 1e-12
        nll = -np.sum(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        nll += l2 * np.sum(w * w)
        d = sw * (p - y)
        grad = np.concatenate([[d.sum()], Z.T @ d + 2 * l2 * w])
        return nll, grad

    theta0 = np.concatenate([[0.0], np.full(k, 0.5)])
    bounds = [(None, None)] + [(0.0, None)] * k  # bias free, weights ≥ 0
    res = minimize(nll_and_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    return float(res.x[0]), res.x[1:].astype(np.float32)


# ── Main classifier ───────────────────────────────────────────────────────────

class HEEBMClassifier(BaseWrapperClassifier):
    """
    Hierarchical Expert-Based EBM classifier.

    Parameters
    ----------
    cohort : str
        Every expert EBM is configured with BEST_PARAMS['EBM'][cohort] — the
        same tuned hyperparameters as the production flat EBM. There is no
        separate fallback hyperparameter set: HE-EBM only makes sense to run
        for a cohort whose flat-EBM backbone has already been tuned, so a
        missing entry raises immediately rather than silently training on
        arbitrary untuned defaults.

    top_k_per_expert : int, default 150
        Maximum number of features each expert's internal TreeFeatureSelector
        retains from its group.  Clamped to the group size automatically.
        Set very large (e.g. 9999) to skip intra-group FS.

    cv_folds : int, default 5
        Number of cross-fit folds used to build honest OOF expert logits for
        the combiner and the bagged inference ensemble.

    combiner_l2 : float, default 1e-2
        L2 penalty on the non-negative expert weights.  Larger → smoother,
        more conservative weighting across experts.

    d_labitems_path : str | None
        Path to d_labitems.csv.gz.  Auto-detected from PROJECT_ROOT if None.

    random_state : int, default RANDOM_SEED
    """

    def __init__(
        self,
        cohort: str | None = None,
        top_k_per_expert: int = 150,
        cv_folds: int = 5,
        combiner_l2: float = 1e-2,
        d_labitems_path: str | None = None,
        random_state: int = RANDOM_SEED,
    ):
        self.cohort = cohort
        self.top_k_per_expert = top_k_per_expert
        self.cv_folds = cv_folds
        self.combiner_l2 = combiner_l2
        self.d_labitems_path = d_labitems_path
        self.random_state = random_state

    # ── EBM construction ─────────────────────────────────────────────────────

    def _ebm_params(self) -> dict:
        """Look up the tuned flat-EBM hyperparameters for `cohort`. Raises if
        absent — HE-EBM has no independent hyperparameter tuning of its own."""
        best = BEST_PARAMS.get("EBM", {}).get(self.cohort) if self.cohort else None
        if not best:
            raise ValueError(
                f"[HE-EBM] No BEST_PARAMS['EBM'] entry for cohort={self.cohort!r}. "
                "HE-EBM reuses the flat EBM's tuned hyperparameters; tune and "
                "register that cohort under BEST_PARAMS['EBM'] first."
            )
        print(f"[HE-EBM] Using best EBM params for cohort '{self.cohort}'")
        return dict(best)

    def _make_ebm(self, params: dict) -> ExplainableBoostingClassifier:
        return ExplainableBoostingClassifier(random_state=self.random_state, **params)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _proba1(proba: np.ndarray) -> np.ndarray:
        """Extract the class-1 probability column from a predict_proba output."""
        if proba.ndim == 1 or proba.shape[1] == 1:
            return proba.ravel()
        return proba[:, 1]

    @classmethod
    def _to_logit(cls, proba: np.ndarray) -> np.ndarray:
        p = np.clip(cls._proba1(proba), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).astype(np.float32)

    @staticmethod
    def _safe_roc_prc(y: np.ndarray, p1: np.ndarray) -> tuple[float | None, float | None]:
        """Compute (AUC-ROC, AUC-PRC); (None, None) if scoring is impossible for
        this fold (e.g. an empty split, or NaN logits from a diverged expert)."""
        try:
            return roc_auc_score(y, p1), average_precision_score(y, p1)
        except ValueError:
            return None, None

    def _run_intra_group_fs(
        self,
        X_group: np.ndarray,
        y: np.ndarray,
        feat_names: list[str],
        expert_id: int,
    ) -> tuple[list[int], list[str]]:
        """
        Apply TreeFeatureSelector within one expert's feature group.

        Returns local indices (into X_group) and names of the selected features.
        TreeFeatureSelector imputes/scales internally for RF ranking only; the
        raw (unscaled, NaN-bearing) columns are what the EBM actually trains on.
        """
        n_group = X_group.shape[1]
        k = min(self.top_k_per_expert, n_group)

        if k == n_group:
            print(f"  [FS Expert {expert_id}] {n_group} features — keeping all "
                  f"(<= top_k_per_expert={self.top_k_per_expert})")
            return list(range(n_group)), feat_names

        selector = TreeFeatureSelector(
            max_features=k, n_estimators=100, max_depth=5, random_state=self.random_state,
        )
        selector.fit_transform(X_group, y, feature_names=feat_names)
        sel_names = selector.get_selected_features()

        name_to_local = {name: i for i, name in enumerate(feat_names)}
        local_sel_idx = [name_to_local[n] for n in sel_names]

        print(f"  [FS Expert {expert_id} ({EXPERT_NAMES[expert_id]})] "
              f"{n_group} -> {len(sel_names)} features (top-{k} RF importance)")
        return local_sel_idx, sel_names

    def _compute_oof_logits(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        ebm_params: dict,
        sample_weight=None,
    ) -> np.ndarray:
        """
        Cross-fit expert EBMs to produce out-of-fold (OOF) logits.

        For each inner CV fold an EBM is trained per active expert on the
        inner-train rows (reusing the full-train FS column masks so meta inputs
        stay column-aligned) and predicts on the held-out rows.  Yields an
        honest, test-like logit for every training row, and stores the fold
        models per expert for bagged inference.
        """
        n_active = len(self.active_experts_)
        oof = np.zeros((X_arr.shape[0], n_active), dtype=np.float32)
        self.oof_fold_models_: list[list] = [[] for _ in range(n_active)]
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)

        for fold_no, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
            for idx_e, i in enumerate(self.active_experts_):
                cols = self.expert_fs_masks_[i]
                X_tr_sub = X_arr[np.ix_(tr_idx, cols)]
                X_val_sub = X_arr[np.ix_(val_idx, cols)]
                ebm = self._make_ebm(ebm_params)
                ebm.feature_names = self.expert_fs_feat_names_[i]
                ebm.fit(X_tr_sub, y_arr[tr_idx], sample_weight=sw_tr)
                oof[val_idx, idx_e] = self._to_logit(ebm.predict_proba(X_val_sub))
                self.oof_fold_models_[idx_e].append(ebm)
            print(f"  [HE-EBM] OOF fold {fold_no+1}/{self.cv_folds} done")
        return oof

    def _expert_logit_matrix_bagged(self, X_arr: np.ndarray) -> np.ndarray:
        """
        Build the [n_samples, n_active] expert-logit matrix by averaging each
        expert's OOF fold models in probability space, then converting to a
        logit.  Keeps inference inputs on the same distribution the combiner
        was trained on.
        """
        n_active = len(self.active_experts_)
        mat = np.zeros((X_arr.shape[0], n_active), dtype=np.float32)
        for idx_e, i in enumerate(self.active_experts_):
            X_sub = X_arr[:, self.expert_fs_masks_[i]]
            models = self.oof_fold_models_[idx_e]
            probas = sum(self._proba1(m.predict_proba(X_sub)) for m in models) / len(models)
            mat[:, idx_e] = self._to_logit(probas)
        return mat

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X,
        y,
        feature_names: Iterable[str] | None = None,
        sample_weight=None,
    ):
        np.random.seed(self.random_state)

        if feature_names is None:
            feature_names = list(X.columns)
        self.selected_features_ = list(feature_names)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        labitems_path = self.d_labitems_path or _default_labitems_path()

        # ── Step 1: partition all features into expert groups (no exclusion) ──
        self.expert_masks_ = build_expert_masks(self.selected_features_, labitems_path)
        self.active_experts_ = [i for i in range(N_EXPERTS) if self.expert_masks_[i]]

        print("\n[HE-EBM] Feature partitioning (pre-FS):")
        for i in range(N_EXPERTS):
            n = len(self.expert_masks_[i])
            tag = "  (SKIPPED — no features)" if n == 0 else ""
            print(f"  Expert {i} ({EXPERT_NAMES[i]}): {n} features{tag}")

        # ── Step 2: intra-group feature selection per expert ──────────────────
        print(f"\n[HE-EBM] Intra-group feature selection (top_k_per_expert={self.top_k_per_expert}):")
        self.expert_fs_masks_: list[list[int]] = [[] for _ in range(N_EXPERTS)]
        self.expert_fs_feat_names_: list[list[str]] = [[] for _ in range(N_EXPERTS)]

        for i in self.active_experts_:
            group_global_idx = self.expert_masks_[i]
            X_group = X_arr[:, group_global_idx]
            group_feat_names = [self.selected_features_[c] for c in group_global_idx]

            local_sel_idx, sel_names = self._run_intra_group_fs(
                X_group, y_arr, group_feat_names, i
            )
            self.expert_fs_masks_[i] = [group_global_idx[j] for j in local_sel_idx]
            self.expert_fs_feat_names_[i] = sel_names

        # ── Step 3: cross-fit OOF expert logits (also stores bagged models) ──
        ebm_params = self._ebm_params()
        print(f"\n[HE-EBM] Cross-fitting OOF expert logits ({self.cv_folds}-fold) …")
        oof_logits = self._compute_oof_logits(X_arr, y_arr, ebm_params, sample_weight)

        print("\n[HE-EBM] OOF per-expert AUC (honest, test-like):")
        for idx_e, i in enumerate(self.active_experts_):
            roc, prc = self._safe_roc_prc(y_arr, oof_logits[:, idx_e])
            if roc is not None:
                print(f"  {EXPERT_NAMES[i]}: OOF AUC-ROC={roc:.4f}  AUC-PRC={prc:.4f}")
            else:
                print(f"  {EXPERT_NAMES[i]}: OOF AUC unavailable")

        # ── Step 4: fine-tune non-negative combiner on honest OOF logits ─────
        print("\n[HE-EBM] Fitting non-negative combiner on OOF logits …")
        self.combiner_bias_, w = fit_expert_weights(
            oof_logits, y_arr, sample_weight, l2=self.combiner_l2
        )
        self.expert_weights_ = {EXPERT_NAMES[i]: float(w[idx_e])
                                for idx_e, i in enumerate(self.active_experts_)}
        self._weight_vec_ = w  # aligned with active_experts_

        print("[HE-EBM] Learned expert weights (data-driven relevance, >= 0):")
        for i in self.active_experts_:
            print(f"  {EXPERT_NAMES[i]}: {self.expert_weights_[EXPERT_NAMES[i]]:.3f}")
        print(f"[HE-EBM] Combiner bias: {self.combiner_bias_:.3f}")

        # Report the combined OOF fit quality (honest, in-sample only for the combiner)
        combined_oof = self.combiner_bias_ + oof_logits @ w
        roc, prc = self._safe_roc_prc(y_arr, combined_oof)
        if roc is not None:
            print(f"[HE-EBM] Combined OOF AUC-ROC={roc:.4f}  AUC-PRC={prc:.4f}")

        self.classes_ = np.array([0, 1])
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def _combined_logit(self, X_arr: np.ndarray) -> np.ndarray:
        Z = self._expert_logit_matrix_bagged(X_arr)
        return self.combiner_bias_ + Z @ self._weight_vec_

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["oof_fold_models_", "_weight_vec_"])
        X_arr = np.asarray(X, dtype=float)
        logit = self._combined_logit(X_arr)
        p1 = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - p1, p1])

    # predict() / predict_log_proba() / get_selected_features() are inherited
    # from BaseWrapperClassifier.

    # ── Per-expert diagnostics ────────────────────────────────────────────────

    def predict_expert_probas(self, X) -> dict[str, np.ndarray]:
        """
        Return each expert's independent (bagged) predicted class-1 probability.

        Keys are EXPERT_NAMES strings; values are 1-D float arrays.  The ensemble
        probability from predict_proba() applies the learned weights and bias on
        top of these.
        """
        check_is_fitted(self, "oof_fold_models_")
        X_arr = np.asarray(X, dtype=float)
        Z = self._expert_logit_matrix_bagged(X_arr)
        out: dict[str, np.ndarray] = {}
        for idx_e, i in enumerate(self.active_experts_):
            p = 1.0 / (1.0 + np.exp(-Z[:, idx_e]))
            out[EXPERT_NAMES[i]] = p.astype(np.float32)
        return out

    # ── Explainability ────────────────────────────────────────────────────────

    def explain_prediction(self, X, sample_idx: int = 0) -> dict:
        """
        Return a per-expert breakdown for one sample:

            'expert_logits'   : {expert_name: float}
            'expert_weights'  : {expert_name: float}   learned relevance weights
            'weighted_logits' : {expert_name: float}   w_i * logit_i contribution
            'bias'            : float
            'final_logit'     : float
            'final_prob'      : float
            'fs_feature_counts': {expert_name: int}
        """
        check_is_fitted(self, "oof_fold_models_")
        X_arr = np.asarray(X, dtype=float)
        Z = self._expert_logit_matrix_bagged(X_arr[[sample_idx]])[0]

        expert_logits, weighted_logits = {}, {}
        total = float(self.combiner_bias_)
        for idx_e, i in enumerate(self.active_experts_):
            name = EXPERT_NAMES[i]
            w_i = float(self._weight_vec_[idx_e])
            expert_logits[name] = float(Z[idx_e])
            weighted_logits[name] = w_i * float(Z[idx_e])
            total += weighted_logits[name]

        return {
            "expert_logits": expert_logits,
            "expert_weights": dict(self.expert_weights_),
            "weighted_logits": weighted_logits,
            "bias": float(self.combiner_bias_),
            "final_logit": total,
            "final_prob": float(1 / (1 + np.exp(-total))),
            "fs_feature_counts": {
                EXPERT_NAMES[i]: len(self.expert_fs_masks_[i])
                for i in self.active_experts_
            },
        }
