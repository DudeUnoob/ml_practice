# Lesson 01 - Scalars, vectors, matrices, and derivatives

Deep learning is applied calculus with a lot of bookkeeping. The same pattern
appears everywhere:

```text
inputs -> function with parameters -> prediction -> loss
```

Training changes the parameters so the loss becomes smaller.

## Scalars

A scalar is one number. Start here because derivatives are easiest to see:

```text
f(x) = x^2
f'(x) = 2x
```

At `x = 3`, a tiny increase in `x` changes `f(x)` about six times as much. That
local sensitivity is the gradient.

## Vectors

A vector is a list of numbers. A data point is often a vector:

```text
house = [bedrooms, bathrooms, square_feet]
```

A linear model combines features with weights:

```text
y = x1*w1 + x2*w2 + x3*w3 + b
```

## Matrices

A matrix lets us process many examples at once:

```text
X shape: (sample_count, feature_count)
W shape: (feature_count, output_count)
Y shape: (sample_count, output_count)

Y = X @ W + b
```

The `@` operator is not just syntax. It is the vectorized version of many dot
products.

## Finite differences

Before trusting backpropagation, approximate a derivative by nudging a value:

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

This is slower than backpropagation but useful for checking your intuition.

## Exercises

1. Write a function for `f(x) = 3x^2 - 2x + 5`.
2. Approximate its derivative at `x = -2, 0, 2`.
3. Compare the finite-difference result with the analytic derivative.
4. Create a matrix `X` with shape `(4, 3)` and weights `W` with shape `(3, 2)`.
   Predict the output shape before running the code.

## Reflection prompt

What does a gradient tell you locally, and why does that local information still
help train a model over many steps?
