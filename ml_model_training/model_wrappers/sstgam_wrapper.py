"""
SSTGAM wrapper — adapts the use-case-agnostic :class:`sstgam.SSTGAM` library
(vendored via git subtree at ``ml_model_training/model_wrappers/sstgam``,
mirrored from the standalone ``sstgam`` repo) to this pipeline's
``fit_method``-keyed construction convention and its VMD feature layout.

The ``sstgam`` library knows nothing about VMD naming (``X_<itemid>_<bin>``
value / ``M_<itemid>_<bin>`` missingness / ``D_<itemid>_<bin>`` delta) — its
only input contract is the generic ``<feature_id>_<time_id>`` pattern (see
``sstgam/_base.py``). This wrapper is the one place that bridges the two:

  * translates the legacy ``fit_method="boost"|"tree"`` call convention onto
    the library's current ``SSTGAM(method="spline"|"tree")`` (``"joint"``
    raises a clear error, since it has no AdamW trainer);
  * does the VMD preprocessing the library won't: keeps the ``X_`` value
    channel (stripping the prefix -> ``<itemid>_<bin>`` = ``<feature_id>_
    <time_id>``), drops the ``M_`` / ``D_`` channels, passes non-VMD columns
    through as static.

Keeping the VMD-specific bits here is what lets ``sstgam`` port cleanly to
other use-cases.

All of ``sstgam.SSTGAM``'s constructor parameters are named explicitly below
(mirrored from ``sstgam/classifier.py``) instead of forwarded through
``**kwargs``: scikit-learn's ``get_params``/``clone`` (used internally by
``GridSearchCV``) require every parameter to appear by name in ``__init__``'s
signature, and construction of the underlying model is deferred to ``fit()``
(also required by the sklearn estimator contract — ``__init__`` must only
assign attributes, never validate or build objects).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.utils.validation import check_is_fitted

# Vendored package lives at sstgam/src/sstgam (src-layout, no longer pip
# installed) -- put its src/ on sys.path so `import sstgam` resolves here.
_SSTGAM_SRC = str(Path(__file__).parent / "sstgam" / "src")
if _SSTGAM_SRC not in sys.path:
    sys.path.insert(0, _SSTGAM_SRC)

from sstgam import SSTGAM

from utils.feature_naming import parse_vmd_column
from ._base import BaseWrapperClassifier

# Legacy fit_method -> current method mapping. This dict was documented in
# the module docstring above but missing from __init__ entirely (found
# 2026-07-25 when fit_method="boost" crashed with "method must be 'spline'
# or 'tree', got 'boost'" -- SSTGAM.__init__ never learned the legacy name).
_FIT_METHOD_TO_METHOD = {"boost": "spline", "tree": "tree"}


def _vmd_to_generic(feature_names):
    """VMD column names -> (kept_col_indices, generic_names_for_kept).

    X_<id>_<bin> -> keep, rename to "<id>_<bin>"; M_/D_ -> drop; anything
    else (including a bin-less X_<id>, e.g. from feature_method="aggregate")
    -> keep unchanged as a static covariate, since SSTGAM's temporal
    <feature>_<time> contract needs an actual time axis.
    """
    keep_idx: list[int] = []
    new_names: list[str] = []
    for i, fn in enumerate(feature_names):
        parsed = parse_vmd_column(fn)
        if parsed is not None and parsed[2] is not None:
            channel, itemid, bin_id = parsed
            if channel == "X":
                keep_idx.append(i)
                new_names.append(f"{itemid}_{bin_id}")
                continue
            continue  # M_/D_ with a bin: dropped, no generic-library analog.
        keep_idx.append(i)
        new_names.append(fn)
    return keep_idx, new_names


class SSTGAMClassifier(BaseWrapperClassifier):
    """VMD adapter + legacy ``fit_method`` shim around :class:`SSTGAM`.

    Every parameter except ``fit_method`` is forwarded verbatim to
    :class:`sstgam.SSTGAM` (see its docstring for what each one does).
    ``fit_method`` maps the legacy "boost"/"tree" naming onto SSTGAM's
    ``method`` parameter ("boost" -> "spline", "tree" -> "tree"; "joint"
    raises, since the sstgam package has no AdamW trainer).

    Same ``fit``/``predict_proba`` interface the pipeline expects. Use
    ``get_sstgam()`` to reach the fitted inner SSTGAM's own interpretability
    methods (``value_effect``, ``temporal_surface``, ``value_time_table``).
    """

    def __init__(
        self,
        fit_method: str = "boost",
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        n_value_bins: int = 16,
        tree_depth: int = 3,
        value_transform: str = "quantile",
        temporal_penalty: float = 1.0,
        static_penalty: float = 1.0,
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        early_stopping_rounds: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.fit_method = fit_method
        self.n_value_bases = n_value_bases
        self.n_time_bases = n_time_bases
        self.n_static_bases = n_static_bases
        self.n_value_bins = n_value_bins
        self.tree_depth = tree_depth
        self.value_transform = value_transform
        self.temporal_penalty = temporal_penalty
        self.static_penalty = static_penalty
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self) -> SSTGAM:
        """Validate fit_method and construct the inner SSTGAM. Called from
        fit(), not __init__, per the sklearn estimator contract."""
        if self.fit_method == "joint":
            raise ValueError(
                "[SSTGAM] fit_method='joint' (the AdamW joint trainer) is not "
                "supported by the sstgam package — it never won a real-data "
                "comparison against boosting. Use fit_method='boost' (-> "
                "method='spline') or 'tree' instead."
            )
        try:
            method = _FIT_METHOD_TO_METHOD[self.fit_method]
        except KeyError:
            raise ValueError(
                f"[SSTGAM] unknown fit_method={self.fit_method!r} (expected 'boost' or 'tree')"
            ) from None
        return SSTGAM(
            method=method,
            n_value_bases=self.n_value_bases,
            n_time_bases=self.n_time_bases,
            n_static_bases=self.n_static_bases,
            n_value_bins=self.n_value_bins,
            tree_depth=self.tree_depth,
            value_transform=self.value_transform,
            temporal_penalty=self.temporal_penalty,
            static_penalty=self.static_penalty,
            learning_rate=self.learning_rate,
            max_rounds=self.max_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    @staticmethod
    def _as_array(X):
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y, feature_names=None, sample_weight=None):
        if feature_names is None:
            feature_names = list(X.columns)
        self.selected_features_ = list(feature_names)
        keep_idx, generic_names = _vmd_to_generic(feature_names)
        self._keep_idx = keep_idx
        Xa = self._as_array(X)[:, keep_idx]
        self.model_ = self._build_model()
        self.model_.fit(Xa, y, feature_names=generic_names, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(self._as_array(X)[:, self._keep_idx])

    # predict() / predict_log_proba() / get_selected_features() are inherited
    # from BaseWrapperClassifier. get_selected_features() returns
    # selected_features_ as set above -- the original, pre-VMD-strip names
    # (not the narrower, renamed columns actually handed to the inner
    # SSTGAM), matching this wrapper's previous behavior.

    def get_sstgam(self) -> SSTGAM:
        """Return the fitted inner SSTGAM, e.g. for its `value_effect`,
        `temporal_surface`, or `value_time_table` interpretability methods."""
        check_is_fitted(self, "model_")
        return self.model_
