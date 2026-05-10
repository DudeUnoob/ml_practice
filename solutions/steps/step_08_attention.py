from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def causal_mask(sequence_length: int) -> NDArray[np.bool_]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    return np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)


def attention(
    query: Array,
    key: Array,
    value: Array,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Array, Array]:
    scores = query @ key.T / np.sqrt(query.shape[1])
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    outputs = weights @ value
    return outputs, weights


def sinusoidal_position_encoding(sequence_length: int, width: int) -> Array:
    positions = np.arange(sequence_length)[:, None]
    dimensions = np.arange(width)[None, :]
    rates = 1 / np.power(10000, (2 * (dimensions // 2)) / width)
    angles = positions * rates

    encodings = np.zeros((sequence_length, width), dtype=float)
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    return encodings
