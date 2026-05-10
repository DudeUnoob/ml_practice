from __future__ import annotations

from collections.abc import Callable


def add(left: float, right: float) -> float:
    return left + right


def multiply(left: float, right: float) -> float:
    return left * right


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty.")

    return sum(values) / len(values)


def squared_error(prediction: float, target: float) -> float:
    error = prediction - target
    return error * error


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    return mean(
        [
            squared_error(prediction, target)
            for prediction, target in zip(predictions, targets, strict=True)
        ]
    )


def finite_difference(fn: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")

    return (fn(x + h) - fn(x - h)) / (2 * h)


def quadratic(x: float) -> float:
    return 3 * x * x - 2 * x + 5


def quadratic_gradient(x: float) -> float:
    return 6 * x - 2
