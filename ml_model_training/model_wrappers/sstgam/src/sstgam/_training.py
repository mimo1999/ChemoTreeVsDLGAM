"""Small shared helpers used by both boosting backends."""

import numpy as np
from sklearn.model_selection import train_test_split


def make_validation_split(target: np.ndarray, fraction: float, random_state: int):
    """Create a reproducible train/validation split.

    Stratification is used when both classes have enough samples. Otherwise a
    random split is used so small or highly imbalanced datasets remain usable.
    """
    target = np.asarray(target).ravel()
    n_samples = target.size
    if n_samples < 2:
        raise ValueError("At least two samples are required to create a validation split.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    indices = np.arange(n_samples)
    n_validation = min(max(1, int(round(n_samples * fraction))), n_samples - 1)
    _, counts = np.unique(target, return_counts=True)
    can_stratify = counts.size > 1 and counts.min() >= 2 and n_validation >= counts.size
    if can_stratify:
        return train_test_split(
            indices,
            test_size=n_validation,
            random_state=random_state,
            stratify=target,
        )

    generator = np.random.RandomState(random_state)
    validation_indices = generator.choice(indices, size=n_validation, replace=False)
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[validation_indices] = False
    return indices[train_mask], validation_indices


def weighted_log_loss(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    """Weighted binary cross-entropy loss."""
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(-np.average(
        target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability),
        weights=weight,
    ))
