from __future__ import annotations

from solutions.steps.step_04_scalar_autograd import Value


def tanh_neuron(inputs: list[Value], weights: list[Value], bias: Value) -> Value:
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have the same length.")

    activation = sum(
        (input_value * weight for input_value, weight in zip(inputs, weights, strict=True)),
        bias,
    )
    return activation.tanh()


def layer(inputs: list[Value], weight_rows: list[list[Value]], biases: list[Value]) -> list[Value]:
    if len(weight_rows) != len(biases):
        raise ValueError("weight_rows and biases must have the same length.")

    return [
        tanh_neuron(inputs, weights, bias)
        for weights, bias in zip(weight_rows, biases, strict=True)
    ]


def mlp_forward(
    inputs: list[float],
    layer_weights: list[list[list[Value]]],
    layer_biases: list[list[Value]],
) -> list[Value]:
    values: list[Value] = [Value(input_value) for input_value in inputs]
    for weight_rows, biases in zip(layer_weights, layer_biases, strict=True):
        values = layer(values, weight_rows, biases)
    return values


def parameters(
    layer_weights: list[list[list[Value]]],
    layer_biases: list[list[Value]],
) -> list[Value]:
    return [
        parameter
        for weight_rows, biases in zip(layer_weights, layer_biases, strict=True)
        for parameter in [item for row in weight_rows for item in row] + biases
    ]


def sgd_step(params: list[Value], learning_rate: float) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    for param in params:
        param.data -= learning_rate * param.grad
