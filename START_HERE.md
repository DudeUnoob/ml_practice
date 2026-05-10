# Start Here: Your First Hour

You said you are starting from nothing. Good. This path assumes only that you can
open a file, type Python, and run a terminal command.

The goal is to make you *build* deep learning one small piece at a time.

## Step 0: install and check the coach

```bash
python3 -m pip install -e ".[dev]"
python3 tools/coach.py list
```

You should see numbered steps. Each step has:

- a file to edit,
- a concept to learn,
- checks to run,
- hints,
- a reference solution.

## Step 1: work on one tiny file

Run:

```bash
python3 tools/coach.py show 01
```

Open the file it prints:

```text
workbook/steps/step_01_python_primitives.py
```

You will see functions with TODOs. Implement only the first function. Then run:

```bash
python3 tools/coach.py check 01
```

It may still fail. Read the next failure. Implement the next function. Repeat.

## The rule

Do not read the reference solution first.

You are allowed to:

- run the checker many times,
- print intermediate values,
- use hints,
- write messy code first,
- rewrite it after it works.

You are not trying to look smart. You are trying to make the machine and your
intuition agree.

## What each stage teaches

| Step | You implement | Why it matters |
| --- | --- | --- |
| 01 | numbers, means, squared error, finite differences | loss and derivative intuition |
| 02 | vectors and matrices using lists | shape thinking before NumPy |
| 03 | linear regression and gradient descent | the full training loop |
| 04 | scalar autograd | backpropagation from first principles |
| 05 | neurons and MLPs | networks as composed functions |
| 06 | NumPy vectorization | why real training uses arrays |
| 07 | softmax classifier | classification losses and logits |
| 08 | attention | the core transformer operation |

## How a step should feel

For each step, write answers to these questions in your own notes:

1. What are the inputs?
2. What should the output be for one tiny example?
3. What shape is each value?
4. What mistake did the checker catch?
5. How would I explain this to someone one step behind me?

## If the checker fails

Read the failure literally. For example:

```text
expected 5.0, got 4.0
```

This is not a judgment. It is a microscope. Make the smallest change that could
explain the difference.

## When to peek at solutions

Only after:

1. You tried at least twice.
2. You used hints.
3. You wrote down what you think the solution should do.

Then run:

```bash
python3 tools/coach.py solution 01
```

Compare structure, not just final answers.

## After the workbook

Once you finish a step, inspect the cleaner reference implementation in
`src/deep_learning_from_scratch/`. The reference code shows how the same idea can
be organized after you understand it.
