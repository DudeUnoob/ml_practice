# Lesson 05 - Training loops, initialization, and generalization

Once the basic loop works, most practical skill comes from diagnosing training
behavior.

## Initialization

If weights are too small, signals shrink. If weights are too large, activations
and gradients can explode. The NumPy MLP uses He-style initialization:

```text
W ~ Normal(0, sqrt(2 / input_width))
```

This keeps activation scale more stable for ReLU networks.

## Generalization

A model can memorize training data and fail on new data. Useful checks:

- Compare train and validation loss.
- Reduce model size.
- Add L2 regularization.
- Add more data or noise.

## Optimization symptoms

| Symptom | Likely cause | Try |
| --- | --- | --- |
| Loss is flat | learning rate too low, dead activations | larger rate, different initialization |
| Loss explodes | learning rate too high | smaller rate |
| Train loss low, validation loss high | overfitting | regularization, smaller model |
| Accuracy stuck near chance | bug, bad labels, too little capacity | inspect data and gradients |

## Code to inspect

- `deep_learning_from_scratch.numpy_nn.initialize_mlp`
- `deep_learning_from_scratch.numpy_nn.forward_mlp`
- `deep_learning_from_scratch.numpy_nn.backward_mlp`
- `deep_learning_from_scratch.numpy_nn.train_mlp`

## Exercises

1. Run `scripts/train_mlp_spiral.py`.
2. Change `--hidden`, `--learning-rate`, and `--l2-strength`.
3. Find one setting that underfits and one setting that trains well.
4. Add a train/validation split and report both accuracies.

## Reflection prompt

What is the difference between an implementation bug and a modeling choice that
does not fit the data?
