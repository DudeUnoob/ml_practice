"""Attention primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def causal_mask(sequence_length: int) -> NDArray[np.bool_]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    return np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Array, Array]:
    """Compute attention outputs and attention weights.

    Shapes:
    - query: ``(query_count, width)``
    - key: ``(key_count, width)``
    - value: ``(key_count, value_width)``
    - mask: optional boolean array broadcastable to ``(query_count, key_count)``
    """

    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("query, key, and value must be two-dimensional.")
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key must have the same feature width.")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value must have the same number of rows.")

    scores = query @ key.T / np.sqrt(query.shape[1])
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = _row_softmax(scores)
    outputs = weights @ value
    return outputs, weights


def sinusoidal_position_encoding(sequence_length: int, width: int) -> Array:
    """Create deterministic transformer-style position encodings."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if width <= 0:
        raise ValueError("width must be positive.")

    positions = np.arange(sequence_length)[:, None]
    dimensions = np.arange(width)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / width)
    angles = positions * angle_rates

    encodings = np.zeros((sequence_length, width), dtype=float)
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    return encodings


def _row_softmax(values: Array) -> Array:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
