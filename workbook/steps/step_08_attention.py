"""Step 08: scaled dot-product attention.

Attention computes content-based weighted averages.
Run:

    python3 tools/coach.py check 08
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def causal_mask(sequence_length: int) -> NDArray[np.bool_]:
    """Return True where attention should be blocked."""

    raise NotImplementedError("TODO: block positions above the main diagonal.")


def attention(
    query: Array,
    key: Array,
    value: Array,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Array, Array]:
    """Return attention outputs and weights."""

    raise NotImplementedError("TODO: scores -> optional mask -> softmax -> weighted values.")


def sinusoidal_position_encoding(sequence_length: int, width: int) -> Array:
    """Return transformer-style sinusoidal position encodings."""

    raise NotImplementedError("TODO: alternate sin and cos columns.")
