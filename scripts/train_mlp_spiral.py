"""Train a one-hidden-layer NumPy MLP on a spiral dataset."""

from __future__ import annotations

import argparse

from deep_learning_from_scratch.data import make_spiral
from deep_learning_from_scratch.numpy_nn import (
    accuracy,
    forward_mlp,
    initialize_mlp,
    train_mlp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2-strength", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels = make_spiral(points_per_class=120, class_count=3, seed=args.seed)
    parameters = initialize_mlp(
        input_size=2,
        hidden_size=args.hidden,
        output_size=3,
        seed=args.seed,
    )
    history = train_mlp(
        parameters=parameters,
        features=features,
        labels=labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        log_every=max(1, args.epochs // 10),
    )
    final_pass = forward_mlp(parameters, features)
    final_accuracy = accuracy(final_pass.logits, labels)

    print("epoch_checkpoints:", len(history.losses))
    for index, (loss, logged_accuracy) in enumerate(
        zip(history.losses, history.accuracies, strict=True),
        start=1,
    ):
        print(
            f"checkpoint={index:02d} loss={loss:.4f} "
            f"accuracy={logged_accuracy:.3f}"
        )
    print(f"final_accuracy={final_accuracy:.3f}")


if __name__ == "__main__":
    main()
