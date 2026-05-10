"""Step 05: neurons and multilayer perceptrons.

This step uses plain Python lists and the Value class you wrote in step 04.
Run:

    python3 tools/coach.py check 05
"""

from __future__ import annotations

from workbook.steps.step_04_scalar_autograd import Value


def tanh_neuron(inputs: list[Value], weights: list[Value], bias: Value) -> Value:
    """Return tanh(dot(inputs, weights) + bias)."""

    raise NotImplementedError("TODO: combine weighted inputs, add bias, apply tanh.")


def layer(inputs: list[Value], weight_rows: list[list[Value]], biases: list[Value]) -> list[Value]:
    """Return outputs for a dense layer.

    weight_rows contains one list of weights per output neuron.
    """

    raise NotImplementedError("TODO: call tanh_neuron once per output neuron.")


def mlp_forward(
    inputs: list[float],
    layer_weights: list[list[list[Value]]],
    layer_biases: list[list[Value]],
) -> list[Value]:
    """Run inputs through multiple dense tanh layers."""

    raise NotImplementedError("TODO: convert floats to Values and apply each layer.")


def parameters(
    layer_weights: list[list[list[Value]]],
    layer_biases: list[list[Value]],
) -> list[Value]:
    """Flatten every weight and bias into one list."""

    raise NotImplementedError("TODO: collect all trainable Values.")


def sgd_step(params: list[Value], learning_rate: float) -> None:
    """Update parameters in place with stochastic gradient descent."""

    raise NotImplementedError("TODO: subtract learning_rate * grad from each parameter.")
