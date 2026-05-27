# Lesson 04 - Multilayer perceptrons

A neural network is a composition of simple functions:

```text
layer_1 = activation(X @ W1 + b1)
layer_2 = activation(layer_1 @ W2 + b2)
prediction = layer_2 @ W3 + b3
```

The hidden layers learn intermediate representations. Without a non-linear
activation, stacked linear layers collapse into one linear layer.

## Neurons

A neuron computes:

```text
activation(w1*x1 + w2*x2 + ... + b)
```

In `deep_learning_from_scratch.nn`, each weight and bias is a `Value`, so the
autograd engine can compute gradients through the whole network.

## Why activation functions matter

Try removing `tanh` or `relu`. The model loses the ability to bend decision
boundaries. Non-linearity is what lets depth matter.

## Training an MLP

The loop is the same as before:

1. Forward pass.
2. Loss.
3. Backward pass.
4. Parameter update.
5. Repeat.

The only difference is that there are more parameters and more intermediate
values.

## Exercises

1. Train `MLP([2, 8, 1])` on a tiny non-linear problem.
2. Increase the hidden width. What changes first: training speed, final loss, or
   stability?
3. Replace `tanh` with `relu` in hidden layers and compare behavior.
4. Print the number of parameters for `MLP([2, 16, 16, 1])`.

## Reflection prompt

If every layer learns a transformation of the previous layer, what might the
first, middle, and final layers be responsible for?
