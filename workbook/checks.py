"""Executable checks for the interactive workbook."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from types import ModuleType

import numpy as np

STEP_MODULES = {
    "01": "step_01_python_primitives",
    "02": "step_02_vectors_matrices",
    "03": "step_03_linear_regression",
    "04": "step_04_scalar_autograd",
    "05": "step_05_neural_networks",
    "06": "step_06_numpy_vectorization",
    "07": "step_07_softmax_classifier",
    "08": "step_08_attention",
}


def step_ids() -> list[str]:
    return list(STEP_MODULES)


def run_check(step_id: str, *, module_root: str = "workbook.steps") -> None:
    if step_id == "all":
        for current_step_id in step_ids():
            run_check(current_step_id, module_root=module_root)
        return

    if step_id not in CHECKS:
        raise ValueError(f"Unknown step: {step_id}")

    module = _load_module(step_id, module_root)
    CHECKS[step_id](module)


def _load_module(step_id: str, module_root: str) -> ModuleType:
    module_name = STEP_MODULES[step_id]
    return importlib.import_module(f"{module_root}.{module_name}")


def _close(actual: float, expected: float, *, tolerance: float = 1e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (
        f"expected {expected}, got {actual}"
    )


def _check_step_01(module: ModuleType) -> None:
    _close(module.add(2, 3), 5)
    _close(module.multiply(-4, 2.5), -10)
    _close(module.mean([2, 4, 6]), 4)
    _close(module.squared_error(3, -1), 16)
    _close(module.mean_squared_error([1, 3, 5], [1, 1, 2]), (0 + 4 + 9) / 3)

    derivative = module.finite_difference(module.quadratic, 2.0)
    _close(module.quadratic(2), 13)
    _close(module.quadratic_gradient(2), 10)
    _close(derivative, module.quadratic_gradient(2.0), tolerance=1e-4)


def _check_step_02(module: ModuleType) -> None:
    _close(module.dot([1, 2, 3], [4, 5, 6]), 32)
    assert module.vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert module.scalar_multiply(3, [1, -2, 4]) == [3, -6, 12]
    assert module.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert module.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [
        [19, 22],
        [43, 50],
    ]


def _check_step_03(module: ModuleType) -> None:
    features = [[1.0, 2.0], [2.0, 0.0], [-1.0, 1.0]]
    targets = [1.0, 4.0, -2.0]
    weights = [0.5, -0.5]
    bias = 0.25

    predictions = module.predict_batch(features, weights, bias)
    assert len(predictions) == 3
    _close(predictions[0], -0.25)
    _close(module.mean_squared_error([1, 2], [1, 4]), 2)

    gradients = module.linear_regression_gradients(features, targets, weights, bias)
    expected_weight_grads = [-5.333333333333333, -0.8333333333333334]
    expected_bias_grad = -1.8333333333333333
    assert set(gradients) == {"weights", "bias"}
    for actual, expected in zip(gradients["weights"], expected_weight_grads, strict=True):
        _close(actual, expected)
    _close(gradients["bias"], expected_bias_grad)

    updated_weights, updated_bias = module.gradient_descent_step(
        weights,
        bias,
        gradients,
        learning_rate=0.1,
    )
    _close(updated_weights[0], 1.0333333333333332)
    _close(updated_weights[1], -0.41666666666666663)
    _close(updated_bias, 0.43333333333333335)


def _check_step_04(module: ModuleType) -> None:
    x = module.Value(3.0)
    y = x * x + x
    y.backward()
    _close(y.data, 12)
    _close(x.grad, 7)

    a = module.Value(1.7)
    b = ((a * a) + (3 * a) - 2).tanh()
    b.backward()
    expected = _finite_difference(lambda value: math.tanh(value * value + 3 * value - 2), 1.7)
    _close(a.grad, expected, tolerance=1e-4)


def _check_step_05(module: ModuleType) -> None:
    values = [module.Value(1.0), module.Value(-2.0)]
    weights = [module.Value(0.5), module.Value(-1.0)]
    bias = module.Value(0.1)
    output = module.tanh_neuron(values, weights, bias)
    _close(output.data, math.tanh(2.6))
    output.backward()
    assert abs(weights[0].grad) > 0
    assert abs(weights[1].grad) > 0

    layer_weights = [
        [
            [module.Value(0.2), module.Value(-0.1)],
            [module.Value(0.5), module.Value(0.3)],
        ],
        [[module.Value(0.7), module.Value(-0.4)]],
    ]
    layer_biases = [[module.Value(0.0), module.Value(0.1)], [module.Value(-0.2)]]
    outputs = module.mlp_forward([1.0, -1.0], layer_weights, layer_biases)
    params = module.parameters(layer_weights, layer_biases)
    assert len(outputs) == 1
    assert len(params) == 9
    loss = (outputs[0] - 1.0) ** 2
    loss.backward()
    before = params[0].data
    module.sgd_step(params, learning_rate=0.05)
    assert params[0].data != before


def _check_step_06(module: ModuleType) -> None:
    features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalized = module.normalize_columns(features)
    assert np.allclose(normalized.mean(axis=0), np.zeros(2))
    assert np.allclose(normalized.std(axis=0), np.ones(2))
    assert np.array_equal(module.relu(np.array([[-1.0, 2.0]])), np.array([[0.0, 2.0]]))

    dense = module.dense_forward(
        np.array([[1.0, 2.0]]),
        np.array([[3.0, 4.0], [5.0, 6.0]]),
        np.array([[0.5, -0.5]]),
    )
    assert np.allclose(dense, np.array([[13.5, 15.5]]))
    _close(module.mse_loss(np.array([[1.0], [3.0]]), np.array([[2.0], [1.0]])), 2.5)


def _check_step_07(module: ModuleType) -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    probabilities = module.softmax(logits)
    assert np.allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.all(probabilities >= 0)

    labels = np.array([2, 0])
    loss = module.cross_entropy(logits, labels)
    gradient = module.softmax_cross_entropy_gradient(logits, labels)
    assert loss > 0
    assert gradient.shape == logits.shape
    assert np.allclose(gradient.sum(axis=1), np.zeros(2))


def _check_step_08(module: ModuleType) -> None:
    sequence = np.eye(3, dtype=float)
    outputs, weights = module.attention(sequence, sequence, sequence)
    assert outputs.shape == (3, 3)
    assert np.allclose(weights.sum(axis=1), np.ones(3))

    _, masked_weights = module.attention(
        sequence,
        sequence,
        sequence,
        mask=module.causal_mask(3),
    )
    assert masked_weights[0, 1] == 0
    assert masked_weights[0, 2] == 0
    assert masked_weights[1, 2] == 0

    encodings = module.sinusoidal_position_encoding(4, 6)
    assert encodings.shape == (4, 6)
    assert np.allclose(encodings[0, 0::2], np.zeros(3))
    assert np.allclose(encodings[0, 1::2], np.ones(3))


def _finite_difference(fn: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (fn(x + h) - fn(x - h)) / (2 * h)


CHECKS: dict[str, Callable[[ModuleType], None]] = {
    "01": _check_step_01,
    "02": _check_step_02,
    "03": _check_step_03,
    "04": _check_step_04,
    "05": _check_step_05,
    "06": _check_step_06,
    "07": _check_step_07,
    "08": _check_step_08,
}
