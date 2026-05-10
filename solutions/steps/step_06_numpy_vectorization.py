from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def normalize_columns(features: Array) -> Array:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")

    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True)
    safe_stds = np.where(stds == 0, 1.0, stds)
    return (features - means) / safe_stds


def relu(values: Array) -> Array:
    return np.maximum(values, 0.0)


def dense_forward(features: Array, weights: Array, bias: Array) -> Array:
    return features @ weights + bias


def mse_loss(predictions: Array, targets: Array) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")

    return float(np.mean((predictions - targets) ** 2))
