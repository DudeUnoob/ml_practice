"""Neural-network building blocks built on the scalar autograd engine."""

from __future__ import annotations

import random
from collections.abc import Iterable

from deep_learning_from_scratch.autograd import Value


def zero_grad(parameters: Iterable[Value]) -> None:
    for parameter in parameters:
        parameter.grad = 0.0


def sgd_step(parameters: Iterable[Value], learning_rate: float) -> None:
    for parameter in parameters:
        parameter.data -= learning_rate * parameter.grad


class Neuron:
    """A single fully connected neuron."""

    def __init__(self, input_count: int, *, activation: str = "tanh") -> None:
        if input_count <= 0:
            raise ValueError("input_count must be positive.")

        self.weights = [
            Value(random.uniform(-1, 1), label=f"w_{index}")
            for index in range(input_count)
        ]
        self.bias = Value(0.0, label="bias")
        self.activation = activation

    def __call__(self, inputs: list[Value] | list[float]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, received {len(inputs)}."
            )

        values = [item if isinstance(item, Value) else Value(item) for item in inputs]
        pre_activation = sum(
            (weight * value for weight, value in zip(self.weights, values, strict=True)),
            self.bias,
        )

        if self.activation == "linear":
            return pre_activation
        if self.activation == "relu":
            return pre_activation.relu()
        if self.activation == "tanh":
            return pre_activation.tanh()

        raise ValueError(f"Unsupported activation: {self.activation}")

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]


class Layer:
    """A dense layer made of independent neurons."""

    def __init__(
        self,
        input_count: int,
        output_count: int,
        *,
        activation: str = "tanh",
    ) -> None:
        if output_count <= 0:
            raise ValueError("output_count must be positive.")

        self.neurons = [
            Neuron(input_count, activation=activation) for _ in range(output_count)
        ]

    def __call__(self, inputs: list[Value] | list[float]) -> list[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class MLP:
    """A small multilayer perceptron.

    ``layer_sizes`` contains every layer width, including the input width.
    For example, ``MLP([2, 8, 8, 1])`` builds a 2-input network with two hidden
    layers and one output.
    """

    def __init__(self, layer_sizes: list[int], *, seed: int | None = None) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Provide at least an input and output size.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive.")
        if seed is not None:
            random.seed(seed)

        pairs = zip(layer_sizes[:-1], layer_sizes[1:], strict=True)
        self.layers = [
            Layer(
                input_count,
                output_count,
                activation="linear" if index == len(layer_sizes) - 2 else "tanh",
            )
            for index, (input_count, output_count) in enumerate(pairs)
        ]

    def __call__(self, inputs: list[float] | list[Value]) -> list[Value]:
        values = inputs
        for layer in self.layers:
            values = layer(values)
        return values

    def parameters(self) -> list[Value]:
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]


def mean_squared_error(predictions: list[Value], targets: list[float]) -> Value:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    if not predictions:
        raise ValueError("At least one prediction is required.")

    losses = [(prediction - target) ** 2 for prediction, target in zip(predictions, targets, strict=True)]
    return sum(losses, Value(0.0)) / len(losses)
