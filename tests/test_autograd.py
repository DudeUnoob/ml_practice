from __future__ import annotations

import math

import pytest

from deep_learning_from_scratch.autograd import Value
from deep_learning_from_scratch.nn import MLP, mean_squared_error, sgd_step


def finite_difference(fn, value: float, *, h: float = 1e-6) -> float:
    return (fn(value + h) - fn(value - h)) / (2 * h)


def test_value_gradient_matches_finite_difference() -> None:
    x = Value(1.7)
    y = ((x * x) + (3 * x) - 2).tanh()
    y.backward()

    expected = finite_difference(lambda input_value: math.tanh(input_value**2 + 3 * input_value - 2), 1.7)

    assert x.grad == pytest.approx(expected, rel=1e-5)


def test_gradients_accumulate_when_value_is_reused() -> None:
    x = Value(3.0)
    y = x * x + x
    y.backward()

    assert x.grad == pytest.approx(7.0)


def test_log_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        Value(0.0).log()


def test_mlp_parameters_receive_gradients() -> None:
    model = MLP([2, 4, 1], seed=1)
    predictions = [model([1.0, -1.0])[0], model([-1.0, 1.0])[0]]
    loss = mean_squared_error(predictions, [1.0, -1.0])
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(abs(gradient) > 0 for gradient in gradients)

    before = [parameter.data for parameter in model.parameters()]
    sgd_step(model.parameters(), learning_rate=0.05)
    after = [parameter.data for parameter in model.parameters()]
    assert before != after
