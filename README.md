# Deep Learning From Nothing

This repo is now organized as an interactive workshop for someone starting from
zero. You do not begin by reading finished code. You begin by editing tiny
workbook files, running a checker, reading a hint only when stuck, and repeating
until the idea becomes mechanical.

## Start here

1. Open [`START_HERE.md`](START_HERE.md).
2. Run:

   ```bash
   python3 -m pip install -e ".[dev]"
   python3 tools/coach.py list
   python3 tools/coach.py show 01
   ```

3. Edit the file shown by the coach.
4. Check your work:

   ```bash
   python3 tools/coach.py check 01
   ```

The workbook checks are expected to fail at first. That is the point: each
failure tells you exactly what concept to implement next.

## Learning method

Each step follows the same loop:

```text
1. Build intuition with a concrete example.
2. Predict the output by hand.
3. Implement the smallest possible function.
4. Run the checker.
5. Read one hint if stuck.
6. Compare with the reference solution only after trying.
7. Write a short explanation in your own words.
```

This is designed to feel like a guided coding tutor, not a textbook dump.

## Repository map

```text
START_HERE.md                    Beginner onboarding and exact first commands
curriculum/ROADMAP.md            Full ordered path from Python numbers to transformers
workbook/steps/                  Files you edit by hand
workbook/checks.py               Step-specific checks used by the coach
solutions/steps/                 Reference implementations to compare after trying
lessons/                         Concept notes for each major idea
src/deep_learning_from_scratch/   Finished reference implementation
scripts/                         Runnable experiments
tests/                           Repo-level tests for the reference and tooling
```

## The two tracks

### Track A: interactive workbook

This is the main learning path. You implement:

1. Python arithmetic, means, losses, finite differences.
2. Vectors and matrices from plain Python lists.
3. Linear regression and gradient descent.
4. A tiny scalar autograd engine.
5. Neurons and multilayer perceptrons.
6. NumPy arrays, broadcasting, and vectorized dense layers.
7. Softmax classification.
8. Attention.

Run:

```bash
python3 tools/coach.py list
```

### Track B: reference implementation

After struggling with the workbook, inspect the finished implementation in
`src/deep_learning_from_scratch/`. This gives you clean code to compare against
your own implementation.

Run:

```bash
python3 -m pytest
python3 scripts/train_mlp_spiral.py --epochs 200 --hidden 16 --learning-rate 0.08
python3 scripts/inspect_attention.py
```

## Why this order?

Deep learning can look huge because people introduce everything at once:
tensors, frameworks, GPUs, optimizers, initialization, backpropagation, datasets,
and architecture names. This repo separates those ideas.

You first learn one number. Then a list of numbers. Then a matrix. Then a loss.
Then a gradient. Then an update. Then backpropagation. Then layers. Then
vectorization. Then attention.

Every advanced concept is built as a small extension of a previous concept.

## What to do when stuck

Use hints in increasing order:

```bash
python3 tools/coach.py hint 03 1
python3 tools/coach.py hint 03 2
python3 tools/coach.py hint 03 3
```

Then compare with:

```bash
python3 tools/coach.py solution 03
```

Do not start with the solution. The learning happens while your implementation
is incomplete and you are debugging it.

## Verification for maintainers

The normal test suite validates the finished reference implementation and the
coach tooling:

```bash
python3 -m pytest
python3 tools/coach.py check all --solution
```

The default workbook checks target `workbook/steps/` and may fail until a learner
fills in the TODOs.
