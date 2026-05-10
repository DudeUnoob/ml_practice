from __future__ import annotations

import numpy as np
import pytest

from deep_learning_from_scratch.data import make_spiral, one_hot
from deep_learning_from_scratch.numpy_nn import (
    accuracy,
    cross_entropy,
    forward_mlp,
    initialize_mlp,
    softmax,
    train_mlp,
)


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    probabilities = softmax(logits)

    assert np.allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.all(probabilities >= 0)


def test_cross_entropy_gradient_shape_matches_logits() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    labels = np.array([0, 2])
    loss, gradient = cross_entropy(logits, labels)

    assert loss > 0
    assert gradient.shape == logits.shape
    assert np.allclose(gradient.sum(axis=1), np.zeros(2))


def test_one_hot_encodes_labels() -> None:
    encoded = one_hot(np.array([2, 0, 1]), class_count=3)

    assert encoded.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_train_mlp_improves_spiral_accuracy() -> None:
    features, labels = make_spiral(points_per_class=40, class_count=3, seed=3)
    parameters = initialize_mlp(input_size=2, hidden_size=24, output_size=3, seed=3)
    initial_accuracy = accuracy(forward_mlp(parameters, features).logits, labels)

    history = train_mlp(
        parameters=parameters,
        features=features,
        labels=labels,
        epochs=250,
        learning_rate=0.08,
        l2_strength=0.001,
        log_every=50,
    )
    final_accuracy = accuracy(forward_mlp(parameters, features).logits, labels)

    assert history.losses[-1] < history.losses[0]
    assert final_accuracy > initial_accuracy
    assert final_accuracy > 0.45


def test_cross_entropy_validates_label_shape() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        cross_entropy(np.zeros((2, 2)), np.zeros((2, 1), dtype=int))
