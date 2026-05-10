# Exercises

The exercises are written as checkpoints instead of long assignments. Do them in
order and keep a small notes file with your answers.

## Checkpoint 1: finite differences

Write a function:

```python
def finite_difference(fn, x, h=1e-5):
    ...
```

Use it to approximate gradients for:

- `x**2`
- `3*x**2 - 2*x + 5`
- `tanh(x)`

Compare against analytic derivatives.

## Checkpoint 2: scalar autograd

Open `src/deep_learning_from_scratch/autograd.py`.

Tasks:

1. Trace `Value.__mul__` line by line.
2. Explain why `other.grad += self.data * out.grad`.
3. Add a `sigmoid` helper using existing `Value` operations.
4. Verify it with finite differences.

## Checkpoint 3: tiny MLP

Train an `MLP([2, 4, 1])` on four XOR-like examples:

```text
[-1, -1] -> -1
[-1,  1] ->  1
[ 1, -1] ->  1
[ 1,  1] -> -1
```

Questions:

- Why is a linear model insufficient?
- What happens if the hidden layer has one neuron?
- What happens if you remove the activation function?

## Checkpoint 4: vectorized classifier

Run:

```bash
python scripts/train_mlp_spiral.py --epochs 300 --hidden 32 --learning-rate 0.08
```

Then modify one thing at a time:

- hidden width
- learning rate
- L2 regularization
- random seed

Record the final loss and accuracy.

## Checkpoint 5: attention inspection

Run:

```bash
python scripts/inspect_attention.py
```

Then answer:

- Which key does the first query attend to most?
- How do the weights change when the causal mask is enabled?
- Why do the rows of the attention matrix sum to one?

## Mastery rubric

You are ready to move on when you can explain:

- How data flows forward through a model.
- How a scalar loss creates gradients for many parameters.
- Why parameter updates use the negative gradient.
- Why non-linear activations are necessary.
- Why attention is weighted retrieval.
