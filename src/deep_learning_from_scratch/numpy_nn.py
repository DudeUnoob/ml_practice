"""Vectorized neural-network pieces implemented directly with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ForwardPass:
    features: Array
    hidden_pre_activation: Array
    hidden_activation: Array
    logits: Array


@dataclass(frozen=True)
class TrainHistory:
    losses: list[float]
    accuracies: list[float]


def relu(values: Array) -> Array:
    return np.maximum(values, 0.0)


def softmax(logits: Array) -> Array:
    """Convert logits to probabilities using a numerically stable shift."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (sample_count, class_count).")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(logits: Array, labels: NDArray[np.int_]) -> tuple[float, Array]:
    """Return the mean cross-entropy loss and its gradient with respect to logits."""

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must contain the same number of samples.")

    probabilities = softmax(logits)
    sample_indices = np.arange(labels.shape[0])
    clipped = np.clip(probabilities[sample_indices, labels], 1e-12, 1.0)
    loss = float(-np.mean(np.log(clipped)))

    gradient = probabilities.copy()
    gradient[sample_indices, labels] -= 1.0
    gradient /= labels.shape[0]
    return loss, gradient


def initialize_mlp(
    *,
    input_size: int,
    hidden_size: int,
    output_size: int,
    seed: int = 0,
) -> dict[str, Array]:
    """Initialize a one-hidden-layer MLP with He-style scaling."""

    if min(input_size, hidden_size, output_size) <= 0:
        raise ValueError("All layer sizes must be positive.")

    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size)),
        "b1": np.zeros((1, hidden_size), dtype=float),
        "w2": rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size, output_size)),
        "b2": np.zeros((1, output_size), dtype=float),
    }


def forward_mlp(parameters: dict[str, Array], features: Array) -> ForwardPass:
    hidden_pre_activation = features @ parameters["w1"] + parameters["b1"]
    hidden_activation = relu(hidden_pre_activation)
    logits = hidden_activation @ parameters["w2"] + parameters["b2"]
    return ForwardPass(
        features=features,
        hidden_pre_activation=hidden_pre_activation,
        hidden_activation=hidden_activation,
        logits=logits,
    )


def l2_penalty(parameters: dict[str, Array], strength: float) -> float:
    if strength < 0:
        raise ValueError("strength must be non-negative.")

    return float(
        0.5
        * strength
        * (np.sum(parameters["w1"] * parameters["w1"]) + np.sum(parameters["w2"] * parameters["w2"]))
    )


def backward_mlp(
    parameters: dict[str, Array],
    forward_pass: ForwardPass,
    labels: NDArray[np.int_],
    *,
    l2_strength: float = 0.0,
) -> tuple[float, dict[str, Array]]:
    data_loss, dlogits = cross_entropy(forward_pass.logits, labels)
    loss = data_loss + l2_penalty(parameters, l2_strength)

    dw2 = forward_pass.hidden_activation.T @ dlogits + l2_strength * parameters["w2"]
    db2 = np.sum(dlogits, axis=0, keepdims=True)

    dhidden = dlogits @ parameters["w2"].T
    dhidden_pre_activation = dhidden * (forward_pass.hidden_pre_activation > 0)
    dw1 = forward_pass.features.T @ dhidden_pre_activation + l2_strength * parameters["w1"]
    db1 = np.sum(dhidden_pre_activation, axis=0, keepdims=True)

    return loss, {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


def update_parameters(
    parameters: dict[str, Array],
    gradients: dict[str, Array],
    *,
    learning_rate: float,
) -> dict[str, Array]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    for name, gradient in gradients.items():
        parameters[name] -= learning_rate * gradient
    return parameters


def accuracy(logits: Array, labels: NDArray[np.int_]) -> float:
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def train_mlp(
    *,
    parameters: dict[str, Array],
    features: Array,
    labels: NDArray[np.int_],
    epochs: int,
    learning_rate: float,
    l2_strength: float = 0.0,
    log_every: int = 50,
) -> TrainHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    losses: list[float] = []
    accuracies: list[float] = []
    for epoch in range(1, epochs + 1):
        forward_pass = forward_mlp(parameters, features)
        loss, gradients = backward_mlp(
            parameters,
            forward_pass,
            labels,
            l2_strength=l2_strength,
        )
        update_parameters(parameters, gradients, learning_rate=learning_rate)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            losses.append(loss)
            accuracies.append(accuracy(forward_pass.logits, labels))

    return TrainHistory(losses=losses, accuracies=accuracies)
