"""Step 03: linear regression and gradient descent.

You now have enough pieces to train the simplest model:

    prediction = dot(features, weights) + bias

Run:

    python3 tools/coach.py check 03
"""

from __future__ import annotations


def predict_one(features: list[float], weights: list[float], bias: float) -> float:
    """Predict one target from one feature vector."""

    raise NotImplementedError("TODO: compute dot(features, weights) + bias.")


def predict_batch(
    feature_rows: list[list[float]],
    weights: list[float],
    bias: float,
) -> list[float]:
    """Predict one value for each feature row."""

    raise NotImplementedError("TODO: call predict_one for every row.")


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    """Return average squared error."""

    raise NotImplementedError("TODO: average (prediction - target)^2.")


def linear_regression_gradients(
    feature_rows: list[list[float]],
    targets: list[float],
    weights: list[float],
    bias: float,
) -> dict[str, list[float] | float]:
    """Return gradients for weights and bias.

    For MSE, dloss/dprediction = 2 * (prediction - target) / sample_count.
    """

    raise NotImplementedError("TODO: compute gradients for every weight and bias.")


def gradient_descent_step(
    weights: list[float],
    bias: float,
    gradients: dict[str, list[float] | float],
    learning_rate: float,
) -> tuple[list[float], float]:
    """Return updated weights and bias."""

    raise NotImplementedError("TODO: parameter = parameter - learning_rate * gradient.")
