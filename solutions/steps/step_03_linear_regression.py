from __future__ import annotations


def predict_one(features: list[float], weights: list[float], bias: float) -> float:
    if len(features) != len(weights):
        raise ValueError("features and weights must have the same length.")

    return sum(feature * weight for feature, weight in zip(features, weights, strict=True)) + bias


def predict_batch(
    feature_rows: list[list[float]],
    weights: list[float],
    bias: float,
) -> list[float]:
    return [predict_one(features, weights, bias) for features in feature_rows]


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    if not predictions:
        raise ValueError("predictions must not be empty.")

    errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    return sum(errors) / len(errors)


def linear_regression_gradients(
    feature_rows: list[list[float]],
    targets: list[float],
    weights: list[float],
    bias: float,
) -> dict[str, list[float] | float]:
    if len(feature_rows) != len(targets):
        raise ValueError("feature_rows and targets must have the same length.")
    if not feature_rows:
        raise ValueError("feature_rows must not be empty.")

    predictions = predict_batch(feature_rows, weights, bias)
    sample_count = len(feature_rows)
    weight_grads = [0.0 for _ in weights]
    bias_grad = 0.0

    for features, prediction, target in zip(feature_rows, predictions, targets, strict=True):
        dloss_dprediction = 2 * (prediction - target) / sample_count
        bias_grad += dloss_dprediction
        for index, feature in enumerate(features):
            weight_grads[index] += dloss_dprediction * feature

    return {"weights": weight_grads, "bias": bias_grad}


def gradient_descent_step(
    weights: list[float],
    bias: float,
    gradients: dict[str, list[float] | float],
    learning_rate: float,
) -> tuple[list[float], float]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    weight_gradients = gradients["weights"]
    bias_gradient = gradients["bias"]
    if not isinstance(weight_gradients, list) or not isinstance(bias_gradient, float):
        raise TypeError("gradients must contain list 'weights' and float 'bias'.")

    updated_weights = [
        weight - learning_rate * gradient
        for weight, gradient in zip(weights, weight_gradients, strict=True)
    ]
    updated_bias = bias - learning_rate * bias_gradient
    return updated_weights, updated_bias
