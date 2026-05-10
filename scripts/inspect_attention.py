"""Print a small attention example that is easy to inspect by hand."""

from __future__ import annotations

import numpy as np

from deep_learning_from_scratch.attention import (
    causal_mask,
    scaled_dot_product_attention,
    sinusoidal_position_encoding,
)


def main() -> None:
    tokens = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        dtype=float,
    )
    positions = sinusoidal_position_encoding(sequence_length=tokens.shape[0], width=4)
    sequence = tokens + 0.1 * positions

    outputs, weights = scaled_dot_product_attention(sequence, sequence, sequence)
    masked_outputs, masked_weights = scaled_dot_product_attention(
        sequence,
        sequence,
        sequence,
        mask=causal_mask(sequence.shape[0]),
    )

    np.set_printoptions(precision=3, suppress=True)
    print("attention_weights")
    print(weights)
    print("outputs")
    print(outputs)
    print("causal_attention_weights")
    print(masked_weights)
    print("causal_outputs")
    print(masked_outputs)


if __name__ == "__main__":
    main()
