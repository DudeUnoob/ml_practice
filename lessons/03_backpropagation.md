# Lesson 03 - Reverse-mode automatic differentiation

Backpropagation is efficient bookkeeping for the chain rule.

If:

```text
z = x * y
loss = z^2
```

then `x` affects `loss` through `z`. The chain rule says:

```text
dloss/dx = dloss/dz * dz/dx
```

Reverse-mode autodiff stores each operation during the forward pass, then walks
backward from the final loss.

## What `Value` stores

`deep_learning_from_scratch.autograd.Value` stores:

- `data`: the scalar number from the forward pass.
- `grad`: the derivative of the final output with respect to this value.
- `_prev`: parent values used to compute this value.
- `_backward`: a function that applies the local derivative.

## Why topological order matters

A value can feed into later values. We must run backward functions only after all
downstream gradients have arrived. A topological sort gives that order.

## Example

```python
from deep_learning_from_scratch.autograd import Value

x = Value(2.0).with_label("x")
y = Value(-3.0).with_label("y")
loss = (x * y + x**2).tanh()
loss.backward()
```

After `backward`, `x.grad` answers:

```text
If x changes a tiny amount, how much does loss change?
```

## Exercises

1. Implement a new operation, such as `sigmoid`, using existing operations.
2. Verify one gradient with finite differences.
3. Create a graph where one value is reused twice, such as `z = x * x + x`.
   Explain why gradients must accumulate instead of being overwritten.

## Reflection prompt

Backpropagation feels like magic until you track one path. Pick one path through
a computation graph and write every local derivative in order.
