"""Step 07: logits, softmax, and cross entropy.

Classification models usually produce logits, not probabilities. You turn logits
into probabilities with softmax, then train with cross entropy.
Run:

    python3 tools/coach.py check 07
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def softmax(logits: Array) -> Array:
    """Return row-wise stable softmax probabilities."""

    raise NotImplementedError("TODO: subtract each row max before exponentiating.")


def cross_entropy(logits: Array, labels: NDArray[np.int_]) -> float:
    """Return mean negative log probability of the correct class."""

    raise NotImplementedError("TODO: gather true-class probabilities and average -log.")


def softmax_cross_entropy_gradient(logits: Array, labels: NDArray[np.int_]) -> Array:
    """Return dloss/dlogits for softmax + cross entropy."""

    raise NotImplementedError("TODO: probabilities minus one-hot labels, divided by batch size.")
