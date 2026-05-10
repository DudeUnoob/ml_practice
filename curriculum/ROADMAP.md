# Full Roadmap: From Nothing to Deep Learning Intuition

This roadmap is the long-form path. The workbook is where you code. The roadmap
explains why each step exists and what mastery looks like.

## Phase 1: Programming and numbers

### Goal

Become comfortable turning simple math into Python functions.

### Build

- addition and multiplication,
- averages,
- squared error,
- finite differences.

### You understand it when

You can explain why a loss is a number and why changing an input slightly can
estimate a derivative.

## Phase 2: Vectors and matrices

### Goal

Think in shapes before relying on NumPy.

### Build

- dot product,
- vector addition,
- scalar multiplication,
- transpose,
- matrix multiplication.

### You understand it when

You can predict the output shape of `X @ W` before running code.

## Phase 3: Linear regression

### Goal

Train the simplest useful model.

### Build

- `prediction = x1*w1 + x2*w2 + ... + b`,
- mean squared error,
- gradients for weights and bias,
- gradient descent updates.

### You understand it when

You can look at a bad prediction and explain which direction the weights should
move.

## Phase 4: Backpropagation

### Goal

Stop hand-writing every full derivative and let local derivatives compose.

### Build

- a scalar `Value`,
- operation history,
- local backward functions,
- topological ordering,
- gradient accumulation.

### You understand it when

You can explain why `x * x + x` gives `x.grad = 2*x + 1`, and why reuse requires
`+=` instead of `=`.

## Phase 5: Neural networks

### Goal

See a neural network as repeated linear combinations plus non-linearity.

### Build

- neuron,
- layer,
- MLP,
- tanh/ReLU activation,
- parameter collection,
- training loop.

### You understand it when

You can explain why stacking linear layers without activations is still just a
linear model.

## Phase 6: Vectorization with NumPy

### Goal

Replace slow scalar loops with array operations while keeping the same mental
model.

### Build

- dense layer forward pass,
- ReLU over a matrix,
- batch loss,
- vectorized gradients.

### You understand it when

You can map a scalar formula to its batch matrix equivalent.

## Phase 7: Classification

### Goal

Train models that choose among categories.

### Build

- logits,
- stable softmax,
- cross entropy,
- softmax gradient.

### You understand it when

You can explain why adding the same constant to every logit does not change the
probabilities.

## Phase 8: Training mechanics

### Goal

Diagnose why a model does or does not learn.

### Build and test

- learning-rate sweeps,
- initialization changes,
- L2 regularization,
- train/validation split,
- accuracy metrics.

### You understand it when

You can distinguish an optimization problem from an overfitting problem.

## Phase 9: Attention

### Goal

Understand the central operation behind transformers.

### Build

- scaled dot-product attention,
- causal mask,
- positional encoding.

### You understand it when

You can describe attention as content-based weighted retrieval.

## Phase 10: Capstone

### Goal

Use the pieces together on a small end-to-end problem.

### Options

- Train a spiral classifier from scratch.
- Build a tiny character-level language model.
- Re-implement one scratch model in PyTorch and compare.

### Final proof of understanding

For any component, you should be able to:

1. write a tiny forward pass,
2. name every shape,
3. test or derive the gradient,
4. describe common failure modes,
5. connect it to a framework implementation.
