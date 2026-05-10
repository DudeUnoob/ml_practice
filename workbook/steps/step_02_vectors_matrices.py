"""Step 02: vectors and matrices using plain Python lists.

No NumPy yet. Shapes should be visible and explicit.
Run:

    python3 tools/coach.py check 02
"""

from __future__ import annotations


def dot(left: list[float], right: list[float]) -> float:
    """Return the dot product of two equal-length vectors."""

    raise NotImplementedError("TODO: multiply matching items and sum them.")


def vector_add(left: list[float], right: list[float]) -> list[float]:
    """Return elementwise left + right."""

    raise NotImplementedError("TODO: add matching items.")


def scalar_multiply(scalar: float, values: list[float]) -> list[float]:
    """Return a new vector with every item multiplied by scalar."""

    raise NotImplementedError("TODO: multiply each value by scalar.")


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Flip rows and columns."""

    raise NotImplementedError("TODO: convert rows into columns.")


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    """Return matrix multiplication left @ right."""

    raise NotImplementedError("TODO: each output item is a row-column dot product.")
