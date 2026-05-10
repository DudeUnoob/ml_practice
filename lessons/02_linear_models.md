# Lesson 02 - Linear models and gradient descent

A linear model is the smallest useful training system:

```text
prediction = X @ W + b
loss = average error between prediction and target
```

It teaches the full training loop without hidden layers.

## Regression

For a numeric target, use mean squared error:

```text
loss = mean((prediction - target)^2)
```

If the prediction is too high, the gradient should push it down. If it is too
low, the gradient should push it up.

## Classification

For class labels, use logits and cross entropy:

```text
logits -> softmax probabilities -> negative log probability of true class
```

Softmax turns arbitrary scores into probabilities. Cross entropy penalizes the
model when it assigns low probability to the correct class.

## Gradient descent

The parameter update is:

```text
parameter = parameter - learning_rate * gradient
```

The learning rate controls step size. Too small learns slowly. Too large can
overshoot and increase the loss.

## Implementation checklist

- Create predictions.
- Compute a scalar loss.
- Compute gradients.
- Update parameters.
- Repeat and record the loss.

## Experiments

1. Train on `make_regression` from `deep_learning_from_scratch.data`.
2. Try learning rates `0.001`, `0.01`, `0.1`, and `1.0`.
3. Record which rates converge, crawl, or explode.

## Reflection prompt

Why does a loss need to be a single scalar even when the model makes many
predictions?
