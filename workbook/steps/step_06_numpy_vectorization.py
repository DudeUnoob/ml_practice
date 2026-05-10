"""Step 06: NumPy arrays and vectorized dense layers.

Now you move from scalar/list code to batched array code.
Run:

    python3 tools/coach.py check 06
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def normalize_columns(features: Array) -> Array:
    """Return features with each column standardized to mean 0 and std 1."""

    raise NotImplementedError("TODO: use axis=0 means and standard deviations.")


def relu(values: Array) -> Array:
    """Return max(values, 0) elementwise."""

    raise NotImplementedError("TODO: use np.maximum.")


def dense_forward(features: Array, weights: Array, bias: Array) -> Array:
    """Return features @ weights + bias."""

    raise NotImplementedError("TODO: implement a vectorized dense layer.")


def mse_loss(predictions: Array, targets: Array) -> float:
    """Return mean squared error for arrays with the same shape."""

    raise NotImplementedError("TODO: average squared differences.")
