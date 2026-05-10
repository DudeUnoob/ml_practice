from __future__ import annotations

import numpy as np

from deep_learning_from_scratch.attention import (
    causal_mask,
    scaled_dot_product_attention,
    sinusoidal_position_encoding,
)


def test_scaled_dot_product_attention_rows_sum_to_one() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    key = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([[10.0, 0.0], [0.0, 20.0]])

    outputs, weights = scaled_dot_product_attention(query, key, value)

    assert outputs.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), np.ones(2))
    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 1] > weights[1, 0]


def test_causal_mask_blocks_future_positions() -> None:
    sequence = np.eye(3, dtype=float)

    _, weights = scaled_dot_product_attention(
        sequence,
        sequence,
        sequence,
        mask=causal_mask(3),
    )

    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 0.0
    assert weights[1, 2] == 0.0
    assert weights[2, 0] > 0.0


def test_position_encoding_shape_and_first_row() -> None:
    encodings = sinusoidal_position_encoding(sequence_length=4, width=6)

    assert encodings.shape == (4, 6)
    assert np.allclose(encodings[0, 0::2], np.zeros(3))
    assert np.allclose(encodings[0, 1::2], np.ones(3))
