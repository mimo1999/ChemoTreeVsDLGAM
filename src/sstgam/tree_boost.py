"""Histogram-based value-time GAM backend."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from ._training import make_validation_split

_MIN_HESSIAN = 1e-6

# Penalty (weighted-SSE-gain units) applied to a single-axis candidate
# split's achieved gain per log-candidate-count, used by the adaptive
# per-round axis choice to correct the classic CART bias toward whichever
# axis has more candidate thresholds (usually the value axis, which has far
# more quantile bins than there are distinct time positions). Not a fit
# hyperparameter -- a Bonferroni/BIC-style multiple-comparisons correction,
# same rationale as before this mechanism was briefly removed and
# reintroduced 2026-07-26.
_AXIS_PENALTY_LAMBDA = 2.0

# Depth given to whichever axis is chosen to split FIRST each round (the
# remaining max_depth - _FIRST_AXIS_DEPTH levels go to the other axis).
# Matches the value that used to be exposed as ``day_split_depth`` before
# this trainer's constructor was simplified; kept as an internal constant
# here rather than re-exposed, since it was never independently tuned away
# from 2 in this project's history.
_FIRST_AXIS_DEPTH = 2


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Create deduplicated quantile bin edges with at least one bin."""
    if values.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges.astype(np.float64)


class SharedShapeTemporalGAMTreeBoost:
    """Cyclic boosted GAM with one value-bin by time-bin table per feature.

    The table is updated through shallow regression trees fitted to aggregated
    Newton statistics. Values are binned per feature using pooled training
    observations.

    Each per-feature (joint value x time) update is fit via an ADAPTIVE axis
    choice: a depth-1 candidate split is tried on the value axis and on the
    time axis separately, each scored by weighted-SSE gain minus a
    candidate-count penalty, and whichever axis's corrected score is higher
    splits first -- so the tree can put weight on the time axis when the
    data actually supports it, rather than defaulting to whichever axis has
    more candidate thresholds (see ``_AXIS_PENALTY_LAMBDA``). Static (1-D)
    terms have only one axis and use a plain single-stage tree.

    ``outer_bags`` bootstrap-resamples the training rows independently
    ``outer_bags`` times (the validation split is fixed across bags); each
    bag's fitted tables are averaged at the end -- the same bagging EBM
    uses for variance reduction.
    """

    def __init__(
        self,
        n_value_bins: int = 16,
        n_static_bins: int = 16,
        max_depth: int = 3,
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_value_bins = n_value_bins
        self.n_static_bins = n_static_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.outer_bags = outer_bags
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)

    def _fit_bins(self, values, observed, static_values) -> None:
        self._value_edges = []
        for feature_index in range(values.shape[1]):
            feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
            self._value_edges.append(_quantile_edges(feature_values, self.n_value_bins))

        self._static_edges = []
        self._static_fill = np.zeros(static_values.shape[1], dtype=np.float64)
        for column_index in range(static_values.shape[1]):
            column = static_values[:, column_index]
            finite_values = column[np.isfinite(column)]
            self._static_fill[column_index] = np.median(finite_values) if finite_values.size else 0.0
            self._static_edges.append(_quantile_edges(finite_values, self.n_static_bins))

    def _feature_bins(self, values) -> list[np.ndarray]:
        return [self._bin(np.nan_to_num(values[:, feature_index, :], nan=0.0), edges)
                for feature_index, edges in enumerate(self._value_edges)]

    def _static_bins(self, static_values) -> list[np.ndarray]:
        filled = np.where(np.isfinite(static_values), static_values, self._static_fill)
        return [self._bin(filled[:, column_index], edges)
                for column_index, edges in enumerate(self._static_edges)]

    def _tree_update(self, coordinates, gradient_sum, hessian_sum, rng):
        """Plain single-stage tree update -- used for 1-D (static) terms."""
        active = hessian_sum > _MIN_HESSIAN
        if active.sum() < 2:
            return None
        targets = gradient_sum[active] / hessian_sum[active]
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=rng.randint(0, 2**31 - 1),
        )
        tree.fit(coordinates[active], targets, sample_weight=hessian_sum[active])
        return tree.predict(coordinates)

    @staticmethod
    def _weighted_gain(target, weight, prediction) -> float:
        """Reduction in weighted SSE from the weighted mean to ``prediction``."""
        if weight.sum() <= 0:
            return 0.0
        mean = float(np.average(target, weights=weight))
        loss_root = float(np.sum(weight * (target - mean) ** 2))
        loss_leaf = float(np.sum(weight * (target - prediction) ** 2))
        return max(loss_root - loss_leaf, 0.0)

    def _adaptive_tree_update(self, coordinates, gradient_sum, hessian_sum, rng):
        """Joint (value_bin, normalized_time) update with a data-driven
        choice of which axis splits first each round.

        ``coordinates`` columns are ``[value_bin, normalized_time]``.
        """
        active = hessian_sum > _MIN_HESSIAN
        if active.sum() < 2:
            return None
        targets = np.divide(gradient_sum, hessian_sum,
                            out=np.zeros_like(gradient_sum), where=hessian_sum > 0)
        value_column, time_column = coordinates[:, 0:1], coordinates[:, 1:2]

        def fit_depth1(column):
            candidate_count = max(len(np.unique(column[active])), 2)
            tree = DecisionTreeRegressor(max_depth=1, random_state=rng.randint(0, 2**31 - 1))
            tree.fit(column[active], targets[active], sample_weight=hessian_sum[active])
            prediction = tree.predict(column[active])
            gain = self._weighted_gain(targets[active], hessian_sum[active], prediction)
            return tree, gain - _AXIS_PENALTY_LAMBDA * np.log(candidate_count)

        time_tree1, time_score = fit_depth1(time_column)
        value_tree1, value_score = fit_depth1(value_column)

        first_depth = max(1, min(_FIRST_AXIS_DEPTH, self.max_depth - 1))
        second_depth = max(1, self.max_depth - first_depth)
        if time_score >= value_score:
            first_column, candidate_tree = time_column, time_tree1
            second_column = value_column
        else:
            first_column, candidate_tree = value_column, value_tree1
            second_column = time_column

        if first_depth == 1:
            # Reuse the depth-1 candidate tree already fit above instead of
            # refitting an identical tree on the same axis/target/weights.
            first_tree = candidate_tree
        else:
            first_tree = DecisionTreeRegressor(max_depth=first_depth, random_state=rng.randint(0, 2**31 - 1))
            first_tree.fit(first_column[active], targets[active], sample_weight=hessian_sum[active])
        groups = first_tree.apply(first_column)
        prediction = first_tree.predict(first_column)

        update = prediction.copy()
        for group in np.unique(groups):
            mask = groups == group
            group_active = mask & active
            if group_active.sum() < 2:
                continue
            second_tree = DecisionTreeRegressor(max_depth=second_depth, random_state=rng.randint(0, 2**31 - 1))
            second_tree.fit(second_column[group_active], targets[group_active], sample_weight=hessian_sum[group_active])
            update[mask] = second_tree.predict(second_column[mask])
        return update

    def _update_temporal_term(self, gradient, hessian, cell_ids, observed,
                              coordinates, n_cells, rng):
        weighted_gradient = (gradient[:, None] * observed).ravel()
        weighted_hessian = (hessian[:, None] * observed).ravel()
        flat_cells = cell_ids.ravel()
        gradient_sum = np.bincount(flat_cells, weights=weighted_gradient, minlength=n_cells)
        hessian_sum = np.bincount(flat_cells, weights=weighted_hessian, minlength=n_cells)
        return self._adaptive_tree_update(coordinates, gradient_sum, hessian_sum, rng)

    def _update_static_term(self, gradient, hessian, bins, coordinates, rng):
        n_bins = coordinates.shape[0]
        gradient_sum = np.bincount(bins, weights=gradient, minlength=n_bins)
        hessian_sum = np.bincount(bins, weights=hessian, minlength=n_bins)
        return self._tree_update(coordinates, gradient_sum, hessian_sum, rng)

    @staticmethod
    def _log_loss(target, probability, weight) -> float:
        probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
        return float(-np.average(
            target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability),
            weights=weight,
        ))

    def fit(self, values, observed, static_values, time_coordinates, target,
            sample_weight=None):
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).ravel()
        n_samples, n_features, n_times = values.shape
        static_values = np.asarray(static_values, dtype=np.float32)
        n_static = static_values.shape[1]
        weights = (np.ones(n_samples, dtype=np.float64) if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float64).ravel())

        self._n_times = n_times
        self._fit_bins(values, observed, static_values)
        feature_bins = self._feature_bins(values)
        static_bins = self._static_bins(static_values)
        n_value_bins = [len(edges) - 1 for edges in self._value_edges]
        self.effective_value_bins_ = np.asarray(n_value_bins)

        time_coordinates = np.asarray(time_coordinates, dtype=np.float64)
        time_scale = max(float(time_coordinates.max() - time_coordinates.min()), 1.0)
        normalized_time = (time_coordinates - time_coordinates.min()) / time_scale
        temporal_coordinates = [
            np.column_stack((
                np.repeat(np.arange(bin_count), n_times),
                np.tile(normalized_time, bin_count),
            ))
            for bin_count in n_value_bins
        ]
        temporal_cell_ids = [
            bins * n_times + np.arange(n_times, dtype=np.int32)[None, :]
            for bins in feature_bins
        ]
        static_coordinates = [np.arange(len(edges) - 1, dtype=np.float64)[:, None]
                              for edges in self._static_edges]

        # Validation split fixed across all bags -- only the bootstrap-
        # resampled training rows differ bag to bag.
        train_indices, validation_indices = make_validation_split(
            target, self.val_fraction, self.random_state
        )

        rng = np.random.RandomState(self.random_state)
        accumulated_intercept = 0.0
        accumulated_feature_tables = None
        accumulated_static_tables = None
        self.best_rounds_ = []
        for bag in range(self.outer_bags):
            fit_indices = (train_indices if self.outer_bags == 1
                          else rng.choice(train_indices, size=train_indices.size, replace=True))
            intercept, feature_tables, static_tables, best_round = self._fit_one_bag(
                weights, target, observed, fit_indices, validation_indices,
                n_features, n_static, n_value_bins, n_times,
                static_bins, temporal_coordinates, temporal_cell_ids,
                static_coordinates, rng,
            )
            self.best_rounds_.append(best_round)
            accumulated_intercept += intercept
            if accumulated_feature_tables is None:
                accumulated_feature_tables = feature_tables
                accumulated_static_tables = static_tables
            else:
                accumulated_feature_tables = [a + b for a, b in zip(accumulated_feature_tables, feature_tables)]
                accumulated_static_tables = [a + b for a, b in zip(accumulated_static_tables, static_tables)]

        self._intercept = accumulated_intercept / self.outer_bags
        self._feature_tables = [table / self.outer_bags for table in accumulated_feature_tables]
        self._static_tables = [table / self.outer_bags for table in accumulated_static_tables]
        return self

    def _fit_one_bag(self, weights, target, observed, fit_indices, validation_indices,
                     n_features, n_static, n_value_bins, n_times,
                     static_bins, temporal_coordinates, temporal_cell_ids,
                     static_coordinates, rng):
        base_rate = np.clip(np.average(target[fit_indices], weights=weights[fit_indices]), 1e-6, 1 - 1e-6)
        intercept = float(np.log(base_rate / (1.0 - base_rate)))
        feature_tables = [np.zeros(bin_count * n_times) for bin_count in n_value_bins]
        static_tables = [np.zeros(len(edges) - 1) for edges in self._static_edges]

        fit_score = np.full(fit_indices.size, intercept)
        validation_score = np.full(validation_indices.size, intercept)
        best_loss = np.inf
        best_state = None
        best_round = 0
        rounds_without_improvement = 0

        for round_index in range(1, self.max_rounds + 1):
            probability = 1.0 / (1.0 + np.exp(-fit_score))
            gradient = weights[fit_indices] * (target[fit_indices] - probability)
            hessian = weights[fit_indices] * probability * (1.0 - probability)
            intercept_update = gradient.sum() / max(hessian.sum(), _MIN_HESSIAN)
            intercept += self.learning_rate * intercept_update
            fit_score += self.learning_rate * intercept_update
            validation_score += self.learning_rate * intercept_update

            for feature_index in range(n_features):
                probability = 1.0 / (1.0 + np.exp(-fit_score))
                gradient = weights[fit_indices] * (target[fit_indices] - probability)
                hessian = weights[fit_indices] * probability * (1.0 - probability)
                update = self._update_temporal_term(
                    gradient, hessian,
                    temporal_cell_ids[feature_index][fit_indices],
                    observed[fit_indices, feature_index, :],
                    temporal_coordinates[feature_index],
                    n_value_bins[feature_index] * n_times, rng,
                )
                if update is None:
                    continue
                feature_tables[feature_index] += self.learning_rate * update
                fit_score += self.learning_rate * (
                    update[temporal_cell_ids[feature_index][fit_indices]] *
                    observed[fit_indices, feature_index, :]
                ).sum(axis=1)
                validation_score += self.learning_rate * (
                    update[temporal_cell_ids[feature_index][validation_indices]] *
                    observed[validation_indices, feature_index, :]
                ).sum(axis=1)

            for column_index in range(n_static):
                probability = 1.0 / (1.0 + np.exp(-fit_score))
                gradient = weights[fit_indices] * (target[fit_indices] - probability)
                hessian = weights[fit_indices] * probability * (1.0 - probability)
                update = self._update_static_term(
                    gradient, hessian, static_bins[column_index][fit_indices],
                    static_coordinates[column_index], rng,
                )
                if update is None:
                    continue
                static_tables[column_index] += self.learning_rate * update
                fit_score += self.learning_rate * update[static_bins[column_index][fit_indices]]
                validation_score += self.learning_rate * update[static_bins[column_index][validation_indices]]

            validation_probability = 1.0 / (1.0 + np.exp(-validation_score))
            validation_loss = self._log_loss(
                target[validation_indices], validation_probability, weights[validation_indices]
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = (intercept, [table.copy() for table in feature_tables],
                              [table.copy() for table in static_tables])
                best_round = round_index
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break

        if best_state is None:
            best_state = (intercept, feature_tables, static_tables)
        return (*best_state, best_round)

    def predict_proba(self, values, observed, static_values):
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        score = np.full(values.shape[0], self._intercept)
        for feature_index, bins in enumerate(self._feature_bins(values)):
            cell_ids = bins * self._n_times + np.arange(self._n_times)[None, :]
            score += (self._feature_tables[feature_index][cell_ids] *
                      observed[:, feature_index, :]).sum(axis=1)
        if self._static_tables:
            for column_index, bins in enumerate(self._static_bins(static_values)):
                score += self._static_tables[column_index][bins]
        probability = 1.0 / (1.0 + np.exp(-score))
        return np.clip(probability, 1e-6, 1.0 - 1e-6)

    def value_time_table(self, feature_index: int):
        """Return ``(value_bin_edges, log_odds_table)`` for one feature."""
        return (self._value_edges[feature_index],
                self._feature_tables[feature_index].reshape(-1, self._n_times))

    def static_effect_table(self, column_index: int):
        """Return ``(bin_edges, log_odds_effect)`` for one static covariate."""
        return self._static_edges[column_index], self._static_tables[column_index]
