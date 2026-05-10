"""Command-line coach for the interactive deep-learning workbook."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from workbook.checks import run_check, step_ids


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    file_path: str
    goal: str
    tasks: tuple[str, ...]
    hints: tuple[str, ...]


STEPS: dict[str, Step] = {
    "01": Step(
        id="01",
        title="Python numbers, losses, and finite differences",
        file_path="workbook/steps/step_01_python_primitives.py",
        goal="Make loss and derivative intuition concrete with plain Python numbers.",
        tasks=(
            "Implement add and multiply.",
            "Implement mean with an empty-list guard.",
            "Implement squared error and mean squared error.",
            "Implement centered finite differences.",
            "Compare a numerical derivative with the exact quadratic derivative.",
        ),
        hints=(
            "A mean is sum(values) divided by len(values).",
            "Squared error is not abs(prediction - target); it is error * error.",
            "Centered finite difference uses both sides: (f(x+h) - f(x-h)) / (2*h).",
        ),
    ),
    "02": Step(
        id="02",
        title="Vectors and matrices before NumPy",
        file_path="workbook/steps/step_02_vectors_matrices.py",
        goal="Build shape intuition with Python lists before arrays hide the loops.",
        tasks=(
            "Implement dot product.",
            "Implement elementwise vector addition.",
            "Implement scalar-vector multiplication.",
            "Implement transpose.",
            "Implement matrix multiplication as row-column dot products.",
        ),
        hints=(
            "The dot product multiplies matching positions, then sums.",
            "Transposing a 2x3 matrix creates a 3x2 matrix.",
            "For matmul, transpose the right matrix so columns are easy to loop over.",
        ),
    ),
    "03": Step(
        id="03",
        title="Linear regression and gradient descent",
        file_path="workbook/steps/step_03_linear_regression.py",
        goal="Train the simplest model by computing a loss, gradients, and updates.",
        tasks=(
            "Implement one prediction.",
            "Apply prediction to a batch.",
            "Compute mean squared error.",
            "Derive gradients for weights and bias.",
            "Update parameters with gradient descent.",
        ),
        hints=(
            "For one weight, dloss/dweight = dloss/dprediction * feature.",
            "For MSE over a batch, dloss/dprediction = 2 * error / sample_count.",
            "Gradient descent subtracts the gradient; adding it usually increases loss.",
        ),
    ),
    "04": Step(
        id="04",
        title="Scalar autograd",
        file_path="workbook/steps/step_04_scalar_autograd.py",
        goal="Implement backpropagation by chaining local derivatives.",
        tasks=(
            "Implement addition, multiplication, negation, power, and tanh.",
            "Store parent values for every operation.",
            "Write local _backward functions that accumulate gradients.",
            "Topologically sort the graph in backward.",
            "Seed the final output gradient with 1.0.",
        ),
        hints=(
            "For z = x + y, dz/dx = 1 and dz/dy = 1.",
            "For z = x * y, dz/dx = y and dz/dy = x.",
            "Use += for gradients because one value can influence the loss through multiple paths.",
        ),
    ),
    "05": Step(
        id="05",
        title="Neurons and multilayer perceptrons",
        file_path="workbook/steps/step_05_neural_networks.py",
        goal="Compose scalar autograd Values into trainable neural networks.",
        tasks=(
            "Implement a tanh neuron.",
            "Build a layer as many neurons.",
            "Run inputs through multiple layers.",
            "Collect every trainable parameter.",
            "Apply SGD updates in place.",
        ),
        hints=(
            "A neuron is tanh(sum(input_i * weight_i) + bias).",
            "A layer returns one output per weight row.",
            "Parameters are all weights from all rows plus all biases.",
        ),
    ),
    "06": Step(
        id="06",
        title="NumPy vectorization",
        file_path="workbook/steps/step_06_numpy_vectorization.py",
        goal="Move from scalar/list code to batched array operations.",
        tasks=(
            "Normalize each column with axis=0.",
            "Implement ReLU elementwise.",
            "Implement a dense layer as features @ weights + bias.",
            "Compute array mean squared error.",
        ),
        hints=(
            "Use keepdims=True so column means broadcast back over rows.",
            "np.maximum(values, 0.0) applies ReLU to the whole array.",
            "If features is (batch, input) and weights is (input, output), the result is (batch, output).",
        ),
    ),
    "07": Step(
        id="07",
        title="Softmax classification",
        file_path="workbook/steps/step_07_softmax_classifier.py",
        goal="Understand logits, probabilities, cross entropy, and their gradient.",
        tasks=(
            "Implement stable row-wise softmax.",
            "Gather the probability assigned to the correct label.",
            "Compute mean negative log probability.",
            "Implement the softmax-cross-entropy gradient.",
        ),
        hints=(
            "Subtract the row max before exp; probabilities stay the same but overflow risk drops.",
            "Use np.arange(batch_size) with labels to index true-class probabilities.",
            "The gradient is probabilities, then subtract 1 at each true class, then divide by batch size.",
        ),
    ),
    "08": Step(
        id="08",
        title="Attention",
        file_path="workbook/steps/step_08_attention.py",
        goal="Build the transformer core operation as weighted retrieval.",
        tasks=(
            "Create a causal mask.",
            "Compute scaled query-key scores.",
            "Apply an optional mask.",
            "Softmax scores into weights.",
            "Compute weighted sums of values.",
            "Create sinusoidal position encodings.",
        ),
        hints=(
            "Scores are query @ key.T divided by sqrt(width).",
            "A causal mask is True above the diagonal.",
            "Attention output is weights @ value.",
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List workbook steps.")

    show_parser = subparsers.add_parser("show", help="Show instructions for a step.")
    show_parser.add_argument("step_id", choices=step_ids())

    hint_parser = subparsers.add_parser("hint", help="Show a hint for a step.")
    hint_parser.add_argument("step_id", choices=step_ids())
    hint_parser.add_argument("hint_number", type=int, nargs="?", default=1)

    solution_parser = subparsers.add_parser("solution", help="Print the reference solution path.")
    solution_parser.add_argument("step_id", choices=step_ids())

    check_parser = subparsers.add_parser("check", help="Run checks for a step.")
    check_parser.add_argument("step_id", choices=[*step_ids(), "all"])
    check_parser.add_argument(
        "--solution",
        action="store_true",
        help="Check reference solutions instead of learner workbook files.",
    )

    args = parser.parse_args()

    if args.command == "list":
        _print_step_list()
        return
    if args.command == "show":
        _print_step(STEPS[args.step_id])
        return
    if args.command == "hint":
        _print_hint(STEPS[args.step_id], args.hint_number)
        return
    if args.command == "solution":
        _print_solution(args.step_id)
        return
    if args.command == "check":
        module_root = "solutions.steps" if args.solution else "workbook.steps"
        _run_check_with_message(args.step_id, module_root=module_root)
        return

    raise ValueError(f"Unsupported command: {args.command}")


def _print_step_list() -> None:
    print("Interactive deep-learning workbook")
    print()
    for step in STEPS.values():
        print(f"{step.id}: {step.title}")
    print()
    print("Next: python3 tools/coach.py show 01")


def _print_step(step: Step) -> None:
    print(f"Step {step.id}: {step.title}")
    print(f"Goal: {step.goal}")
    print(f"Edit: {step.file_path}")
    print()
    print("Tasks:")
    for index, task in enumerate(step.tasks, start=1):
        print(f"  {index}. {task}")
    print()
    print(f"Check: python3 tools/coach.py check {step.id}")
    print(f"Hint:  python3 tools/coach.py hint {step.id} 1")


def _print_hint(step: Step, hint_number: int) -> None:
    if hint_number < 1 or hint_number > len(step.hints):
        raise SystemExit(f"Hint number must be between 1 and {len(step.hints)}.")

    print(f"Step {step.id} hint {hint_number}: {step.hints[hint_number - 1]}")


def _print_solution(step_id: str) -> None:
    module_name = step_id_to_module_name(step_id)
    path = Path("solutions") / "steps" / f"{module_name}.py"
    print(path)
    print()
    print("Read this only after attempting the workbook step.")


def _run_check_with_message(step_id: str, *, module_root: str) -> None:
    try:
        run_check(step_id, module_root=module_root)
    except NotImplementedError as error:
        raise SystemExit(f"Not implemented yet: {error}") from error
    except AssertionError as error:
        message = str(error) or "an assertion failed"
        raise SystemExit(f"Check failed: {message}") from error

    target = "reference solutions" if module_root == "solutions.steps" else "workbook"
    print(f"Passed {target} check for step {step_id}.")


def step_id_to_module_name(step_id: str) -> str:
    return {
        "01": "step_01_python_primitives",
        "02": "step_02_vectors_matrices",
        "03": "step_03_linear_regression",
        "04": "step_04_scalar_autograd",
        "05": "step_05_neural_networks",
        "06": "step_06_numpy_vectorization",
        "07": "step_07_softmax_classifier",
        "08": "step_08_attention",
    }[step_id]


if __name__ == "__main__":
    main()
