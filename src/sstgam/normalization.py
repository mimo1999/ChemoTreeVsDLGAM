"""Normalization helpers for longitudinal and static values."""

import numpy as np


def robust_center_scale(values: np.ndarray, observed: np.ndarray):
    """Return per-feature median centres and robust scales."""
    n_features = values.shape[1]
    centers = np.zeros(n_features, dtype=np.float32)
    scales = np.ones(n_features, dtype=np.float32)
    for feature_index in range(n_features):
        feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
        feature_values = feature_values[np.isfinite(feature_values)]
        if feature_values.size == 0:
            continue
        centers[feature_index] = np.median(feature_values)
        lower, upper = np.percentile(feature_values, [25, 75])
        scale = (upper - lower) / 1.349
        if scale <= 1e-8:
            scale = feature_values.std()
        scales[feature_index] = scale if scale > 1e-8 else 1.0
    return centers, scales


def fit_quantile_tables(values: np.ndarray, observed: np.ndarray):
    """Fit empirical CDF tables that map each feature into ``(-1, 1)``."""
    tables = []
    for feature_index in range(values.shape[1]):
        feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
        feature_values = feature_values[np.isfinite(feature_values)]
        if feature_values.size < 2:
            tables.append((np.array([-1.0, 1.0]), np.array([-1.0, 1.0])))
            continue

        sorted_values = np.sort(feature_values)
        ranks = (np.arange(sorted_values.size) + 0.5) / sorted_values.size
        unique_values, inverse = np.unique(sorted_values, return_inverse=True)
        rank_sums = np.zeros(unique_values.size)
        rank_counts = np.zeros(unique_values.size)
        np.add.at(rank_sums, inverse, ranks)
        np.add.at(rank_counts, inverse, 1.0)
        unique_ranks = rank_sums / rank_counts
        tables.append((unique_values.astype(np.float64), (2.0 * unique_ranks - 1.0).astype(np.float64)))
    return tables


def apply_quantile_tables(values: np.ndarray, tables, clip: float) -> np.ndarray:
    """Apply fitted empirical CDF tables to a value tensor."""
    transformed = np.empty(values.shape, dtype=np.float32)
    for feature_index, (x_values, y_values) in enumerate(tables):
        transformed[:, feature_index, :] = (
            np.interp(values[:, feature_index, :], x_values, y_values) * clip
        ).astype(np.float32)
    return transformed


def fit_value_normalization(values: np.ndarray, observed: np.ndarray, transform: str, clip: float):
    """Fit and apply either quantile or robust z-score normalization."""
    n_features = values.shape[1]
    if transform == "quantile":
        tables = fit_quantile_tables(values, observed)
        return (
            apply_quantile_tables(values, tables, clip),
            np.zeros(n_features, dtype=np.float32),
            np.ones(n_features, dtype=np.float32),
            tables,
        )
    if transform == "zscore":
        centers, scales = robust_center_scale(values, observed)
        return values, centers, scales, None
    raise ValueError(f"value_transform must be 'zscore' or 'quantile', got {transform!r}")


def apply_value_normalization(values: np.ndarray, transform: str, tables, clip: float) -> np.ndarray:
    """Apply a normalization fitted by :func:`fit_value_normalization`."""
    return apply_quantile_tables(values, tables, clip) if transform == "quantile" else values


def static_stats(values: np.ndarray):
    """Return per-column finite-value means and standard deviations."""
    means = np.zeros(values.shape[1], dtype=np.float32)
    scales = np.ones(values.shape[1], dtype=np.float32)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        column = column[np.isfinite(column)]
        if column.size:
            means[column_index] = column.mean()
            standard_deviation = column.std()
            scales[column_index] = standard_deviation if standard_deviation > 1e-8 else 1.0
    return means, scales
