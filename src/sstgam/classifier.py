"""Public SSTGAM classifier."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import _SSTGAMBase
from .boosting import SharedShapeTemporalGAMBoost
from .tree_boost import SharedShapeTemporalGAMTreeBoost


class SSTGAM(_SSTGAMBase):
    """Explainable binary classifier for wide longitudinal data.

    Temporal columns use the name ``<feature_name>_<time_id>``. For example,
    ``temperature_0`` and ``temperature_6`` are two observations of the same
    feature at time coordinates 0 and 6. Columns without an integer suffix are
    static covariates.

    Each temporal feature has one shared value-time effect. The model sums the
    effects of observed temporal cells, adds optional static feature effects,
    and maps the result to a binary probability through a logistic link.

    Parameters
    ----------
    method : {"spline", "tree"}, default="spline"
        ``"spline"`` fits smooth tensor-product value-time surfaces.
        ``"tree"`` fits discrete value-bin by time-bin tables.
    n_value_bases, n_time_bases, n_static_bases : int
        Number of RBF bases used by the spline method.
    n_value_bins, tree_depth : int
        Value bin count and shallow-tree depth used by the tree method.
    value_transform : {"quantile", "zscore"}, default="quantile"
        Per-feature value normalization for the spline method.
    temporal_penalty, static_penalty : float
        Smoothness penalties for spline temporal and static effects.
    learning_rate, max_rounds, early_stopping_rounds
        Boosting controls shared by both methods.
    outer_bags : int, default=4
        Number of bootstrap-resampled fits (the validation split is fixed
        across bags) whose coefficients/tables are averaged at the end --
        the same bagging EBM uses for variance reduction. ``outer_bags=1``
        disables bagging (a single fit on the full training split).
    val_fraction, random_state, verbose
        Validation split and runtime controls.
    """

    def __init__(
        self,
        method: str = "spline",
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
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        if method not in {"spline", "tree"}:
            raise ValueError(f"method must be 'spline' or 'tree', got {method!r}")
        if min(n_value_bases, n_time_bases, n_static_bases, n_value_bins, tree_depth) < 1:
            raise ValueError("Basis counts, bin count, and tree_depth must be positive.")
        if min(temporal_penalty, static_penalty, learning_rate) <= 0:
            raise ValueError("Penalties and learning_rate must be positive.")
        if max_rounds < 1 or early_stopping_rounds < 1 or outer_bags < 1:
            raise ValueError("max_rounds, early_stopping_rounds, and outer_bags must be positive.")
        super().__init__(val_fraction=val_fraction, random_state=random_state, verbose=verbose)
        self.method = method
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
        self.outer_bags = outer_bags
        self.early_stopping_rounds = early_stopping_rounds

    def _fit_trainer(self, values, observed, static_values, time_coordinates,
                     target, sample_weight):
        trainer = self._make_trainer()
        return trainer.fit(
            values, observed, static_values, time_coordinates, target,
            sample_weight=sample_weight,
        )

    def _make_trainer(self):
        common_arguments = dict(
            learning_rate=self.learning_rate,
            max_rounds=self.max_rounds,
            outer_bags=self.outer_bags,
            early_stopping_rounds=self.early_stopping_rounds,
            val_fraction=self.val_fraction,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        if self.method == "tree":
            return SharedShapeTemporalGAMTreeBoost(
                n_value_bins=self.n_value_bins,
                max_depth=self.tree_depth,
                **common_arguments,
            )
        return SharedShapeTemporalGAMBoost(
            n_value_bases=self.n_value_bases,
            n_time_bases=self.n_time_bases,
            n_static_bases=self.n_static_bases,
            value_transform=self.value_transform,
            temporal_penalty=self.temporal_penalty,
            static_penalty=self.static_penalty,
            **common_arguments,
        )

    def value_effect(self, feature: str, values) -> np.ndarray:
        """Return the fitted spline effect of values averaged across time."""
        if self.method != "spline":
            raise ValueError("value_effect is only available when method='spline'.")
        check_is_fitted(self, "model_")
        feature_index = self.temporal_features_.index(feature)
        normalized_values = self._normalise_feature_values(feature_index, values)
        return self.model_.mean_value_effect_normalized(feature_index, normalized_values)

    def temporal_surface(self, feature: str, values, time_coordinates) -> np.ndarray:
        """Evaluate a spline feature's value-time log-odds surface."""
        if self.method != "spline":
            raise ValueError("temporal_surface is only available when method='spline'.")
        check_is_fitted(self, "model_")
        feature_index = self.temporal_features_.index(feature)
        normalized_values = self._normalise_feature_values(feature_index, values)
        normalized_time = self._normalise_time(time_coordinates)
        return self.model_.temporal_surface_normalized(
            feature_index, normalized_values, normalized_time
        )

    def value_time_table(self, feature: str):
        """Return the tree backend's fitted value-bin by time-bin table."""
        if self.method != "tree":
            raise ValueError("value_time_table is only available when method='tree'.")
        check_is_fitted(self, "model_")
        return self.model_.value_time_table(self.temporal_features_.index(feature))
