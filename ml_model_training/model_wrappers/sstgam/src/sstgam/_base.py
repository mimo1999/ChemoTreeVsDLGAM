"""Shared sklearn interface for wide longitudinal data."""

import re
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

_TEMPORAL_COLUMN_PATTERN = re.compile(r"^(.+)_(\d+)$")


class _SSTGAMBase(ClassifierMixin, BaseEstimator):
    """Parse wide longitudinal inputs and provide the sklearn classifier API.

    A temporal column is named ``<feature_name>_<time_id>``. All other columns
    are treated as static covariates. Missing temporal values are represented
    by an observation mask; they do not contribute to the temporal surface.
    """

    def __init__(self, val_fraction: float = 0.15, random_state: int = 42,
                 verbose: int = 0) -> None:
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _parse_columns(column_names: list[str]):
        temporal_columns: dict[str, dict[int, int]] = {}
        static_columns: list[int] = []
        for column_index, column_name in enumerate(column_names):
            match = _TEMPORAL_COLUMN_PATTERN.match(column_name)
            if match is None:
                static_columns.append(column_index)
                continue
            feature_name, time_id = match.group(1), int(match.group(2))
            feature_columns = temporal_columns.setdefault(feature_name, {})
            if time_id in feature_columns:
                raise ValueError(f"Duplicate temporal column: {column_name!r}")
            feature_columns[time_id] = column_index
        return temporal_columns, static_columns

    @staticmethod
    def _as_float_matrix(X) -> np.ndarray:
        matrix = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        return matrix

    def _prepare_fit_input(self, X, feature_names: Iterable[str] | None):
        matrix = self._as_float_matrix(X)
        if feature_names is None:
            names = list(X.columns) if hasattr(X, "columns") else [
                f"feature_{index}" for index in range(matrix.shape[1])
            ]
        else:
            names = list(feature_names)
        if len(names) != matrix.shape[1]:
            raise ValueError("feature_names must contain one name per column in X.")
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique.")
        return matrix, names

    def _prepare_predict_input(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            received = list(X.columns)
            missing = [name for name in self.feature_names_in_ if name not in received]
            extra = [name for name in received if name not in self.feature_names_in_]
            if missing or extra:
                raise ValueError(
                    f"Prediction columns differ from fitted columns; missing={missing}, extra={extra}."
                )
            X = X.loc[:, self.feature_names_in_]
        matrix = self._as_float_matrix(X)
        if matrix.shape[1] != len(self.feature_names_in_):
            raise ValueError("X has a different number of columns than the fitted data.")
        return matrix

    def _build_tensors(self, matrix: np.ndarray):
        n_samples = matrix.shape[0]
        n_features = len(self.temporal_features_)
        n_times = len(self.time_ids_)
        values = np.zeros((n_samples, n_features, n_times), dtype=np.float32)
        observed = np.zeros_like(values)
        time_positions = {time_id: index for index, time_id in enumerate(self.time_ids_)}

        for feature_index, feature_name in enumerate(self.temporal_features_):
            for time_id, column_index in self._temporal_columns[feature_name].items():
                column = matrix[:, column_index]
                is_observed = np.isfinite(column)
                time_index = time_positions[time_id]
                values[:, feature_index, time_index] = np.where(is_observed, column, 0.0)
                observed[:, feature_index, time_index] = is_observed

        if self._static_column_indices:
            static_values = matrix[:, self._static_column_indices]
        else:
            static_values = np.empty((n_samples, 0), dtype=np.float32)
        return values, observed, static_values.astype(np.float32, copy=False)

    def fit(self, X, y, feature_names: Iterable[str] | None = None, sample_weight=None):
        matrix, names = self._prepare_fit_input(X, feature_names)
        target = np.asarray(y, dtype=np.float64).ravel()
        if target.shape[0] != matrix.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if not np.isin(target, (0.0, 1.0)).all():
            raise ValueError("SSTGAM supports binary targets encoded as 0 and 1.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            if (sample_weight.shape[0] != matrix.shape[0] or np.any(sample_weight < 0)
                    or sample_weight.sum() <= 0):
                raise ValueError(
                    "sample_weight must be non-negative, have positive total weight, and match X rows."
                )

        temporal_columns, static_columns = self._parse_columns(names)
        if not temporal_columns:
            raise ValueError("At least one '<feature_name>_<time_id>' column is required.")

        self.feature_names_in_ = names
        self._temporal_columns = temporal_columns
        self._static_column_indices = static_columns
        self.temporal_features_ = sorted(temporal_columns)
        self.time_ids_ = sorted({
            time_id for feature_name in self.temporal_features_
            for time_id in temporal_columns[feature_name]
        })

        values, observed, static_values = self._build_tensors(matrix)
        self.model_ = self._fit_trainer(
            values, observed, static_values,
            np.asarray(self.time_ids_, dtype=np.float32), target, sample_weight,
        )
        self.classes_ = np.array([0, 1])
        return self

    def _fit_trainer(self, values, observed, static_values, time_coordinates,
                     target, sample_weight):
        raise NotImplementedError

    def _positive_probability(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        values, observed, static_values = self._build_tensors(self._prepare_predict_input(X))
        return self.model_.predict_proba(values, observed, static_values)

    def predict_proba(self, X):
        probability = self._positive_probability(X)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X):
        return (self._positive_probability(X) >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_feature_names(self) -> list[str]:
        """Return all fitted input column names in their required order."""
        check_is_fitted(self, "model_")
        return list(self.feature_names_in_)

    def get_temporal_features(self) -> list[str]:
        """Return temporal feature names in model order."""
        check_is_fitted(self, "model_")
        return list(self.temporal_features_)

    def _normalise_feature_values(self, feature_index: int, raw_values) -> np.ndarray:
        values = np.asarray(raw_values, dtype=np.float32)
        if self.value_transform == "quantile":
            source_values, source_ranks = self.model_._quantile_tables[feature_index]
            return (np.interp(values, source_values, source_ranks) * self.model_.clip).astype(np.float32)
        scaled = ((values - self.model_._feature_center[feature_index]) /
                  self.model_._feature_scale[feature_index])
        return np.clip(scaled, -self.model_.clip, self.model_.clip)

    def _normalise_time(self, time_values) -> np.ndarray:
        lower, upper = float(self.time_ids_[0]), float(self.time_ids_[-1])
        return (np.asarray(time_values, dtype=np.float32) - lower) / max(upper - lower, 1.0)
