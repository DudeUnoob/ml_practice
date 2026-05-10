# Lesson 00 - Learning method and setup

Deep learning is best learned as a stack of mechanisms. Each mechanism answers
one question:

- How do numbers flow forward through a model?
- How does a loss measure error?
- How do gradients assign responsibility for that error?
- How do updates change future predictions?

## The study loop

For every topic in this repo:

1. Read the lesson once without coding.
2. Rewrite the core equation by hand.
3. Predict the shapes of every input and output.
4. Implement the smallest useful version.
5. Run tests.
6. Break the implementation deliberately and explain the failure.

The deliberate break is important. You learn backpropagation faster by seeing
what happens when a gradient is missing than by only reading the correct formula.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Minimum Python comfort

You should be able to:

- Read list comprehensions.
- Write small functions.
- Understand NumPy array shapes.
- Use a terminal command and interpret a test failure.

If a concept feels too abstract, make it smaller. Use one scalar, then one
vector, then a batch.

## Your first checkpoint

Run:

```bash
pytest tests/test_autograd.py
```

Then open `src/deep_learning_from_scratch/autograd.py` and trace one expression:

```python
from deep_learning_from_scratch.autograd import Value

x = Value(2.0)
y = (x * x + 3 * x).tanh()
y.backward()
print(x.grad)
```

Explain: which operations happen forward, and which local derivatives happen
backward?
