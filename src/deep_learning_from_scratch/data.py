"""Small deterministic datasets for scratch experiments."""

from __future__ import annotations

import numpy as np


def make_regression(
    *,
    sample_count: int = 128,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create ``y = 2x - 1 + noise`` for linear-regression practice."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=(sample_count, 1))
    targets = 2.0 * features[:, [0]] - 1.0
    targets += rng.normal(0.0, noise, size=targets.shape)
    return features.astype(float), targets.astype(float)


def make_spiral(
    *,
    points_per_class: int = 100,
    class_count: int = 3,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a small spiral classification dataset without external packages."""

    if points_per_class <= 0:
        raise ValueError("points_per_class must be positive.")
    if class_count <= 1:
        raise ValueError("class_count must be greater than one.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    sample_count = points_per_class * class_count
    features = np.zeros((sample_count, 2), dtype=float)
    labels = np.zeros(sample_count, dtype=int)

    for class_index in range(class_count):
        row_slice = slice(
            points_per_class * class_index,
            points_per_class * (class_index + 1),
        )
        radius = np.linspace(0.0, 1.0, points_per_class)
        theta = np.linspace(
            class_index * 4.0,
            (class_index + 1) * 4.0,
            points_per_class,
        )
        theta += rng.normal(0.0, noise, points_per_class)
        features[row_slice] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        labels[row_slice] = class_index

    return features, labels


def one_hot(labels: np.ndarray, class_count: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if labels.size == 0:
        raise ValueError("labels must not be empty.")

    inferred_class_count = int(labels.max()) + 1
    class_count = inferred_class_count if class_count is None else class_count
    if class_count < inferred_class_count:
        raise ValueError("class_count is smaller than the largest label.")

    encoded = np.zeros((labels.size, class_count), dtype=float)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded
