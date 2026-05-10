from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def softmax(logits: Array) -> Array:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(logits: Array, labels: NDArray[np.int_]) -> float:
    probabilities = softmax(logits)
    sample_indices = np.arange(labels.shape[0])
    correct_probabilities = probabilities[sample_indices, labels]
    return float(-np.mean(np.log(np.clip(correct_probabilities, 1e-12, 1.0))))


def softmax_cross_entropy_gradient(logits: Array, labels: NDArray[np.int_]) -> Array:
    probabilities = softmax(logits)
    sample_indices = np.arange(labels.shape[0])
    probabilities[sample_indices, labels] -= 1.0
    return probabilities / labels.shape[0]
