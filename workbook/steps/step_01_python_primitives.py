"""Step 01: Python numbers, losses, and finite differences.

Implement these functions using only plain Python. Do not import NumPy here.
Run:

    python3 tools/coach.py check 01
"""

from __future__ import annotations

from collections.abc import Callable


def add(left: float, right: float) -> float:
    """Return left + right."""

    raise NotImplementedError("TODO: return the sum of left and right.")


def multiply(left: float, right: float) -> float:
    """Return left * right."""

    raise NotImplementedError("TODO: return the product of left and right.")


def mean(values: list[float]) -> float:
    """Return the average value.

    Think about the empty-list case before writing the happy path.
    """

    raise NotImplementedError("TODO: validate values and return their average.")


def squared_error(prediction: float, target: float) -> float:
    """Return (prediction - target) squared."""

    raise NotImplementedError("TODO: compute one squared prediction error.")


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    """Return the average squared error for matching prediction/target lists."""

    raise NotImplementedError("TODO: combine squared_error and mean.")


def finite_difference(fn: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Approximate the derivative of fn at x using a centered difference."""

    raise NotImplementedError("TODO: evaluate fn at x + h and x - h.")


def quadratic(x: float) -> float:
    """Return 3*x^2 - 2*x + 5."""

    raise NotImplementedError("TODO: implement the quadratic from the docstring.")


def quadratic_gradient(x: float) -> float:
    """Return the exact derivative of quadratic(x)."""

    raise NotImplementedError("TODO: derive d/dx of 3*x^2 - 2*x + 5.")
