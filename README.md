# Deep Learning From Scratch

This repository is a guided path for learning deep learning from the ground up by
deriving the ideas, implementing them with small amounts of Python/NumPy, and
testing each piece until the mechanics feel concrete.

The goal is not to memorize framework APIs. The goal is to build the mental
model behind them:

1. What is a model?
2. What is a loss?
3. Why do gradients point toward useful parameter updates?
4. How does backpropagation avoid recomputing the same work?
5. Why do initialization, normalization, optimization, and regularization matter?
6. How do modern layers such as convolutions and attention fit the same pattern?

## How to use this repo

Use the loop below for each lesson:

1. **Read** the short lesson note.
2. **Predict** what the implementation should do before opening the code.
3. **Implement or modify** the referenced function.
4. **Run tests** for fast feedback.
5. **Run an experiment** and inspect failure modes.
6. **Write one paragraph** explaining the mechanism in your own words.

This read -> derive -> implement -> test -> reflect loop is slower than watching
videos, but it builds intuition that transfers to new architectures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The implementation intentionally depends only on NumPy for the core learning
path. PyTorch/JAX/TensorFlow are useful later, but they hide the machinery this
repo is designed to expose.

## Repository map

```text
lessons/                         Conceptual notes in the recommended order
exercises/                       Guided prompts and checkpoints
scripts/                         Runnable experiments
src/deep_learning_from_scratch/   Small teaching implementation
tests/                           Executable specifications for the core ideas
```

## Learning path

### Stage 0: Orientation

- [Lesson 00 - Learning method and setup](lessons/00_learning_method.md)
- Outcome: you know how to run tests and how each activity reinforces intuition.

### Stage 1: Mathematical primitives

- [Lesson 01 - Scalars, vectors, matrices, and derivatives](lessons/01_math_primitives.md)
- Implement: scalar arithmetic, finite differences, vectorized data transforms.
- Outcome: you can explain a gradient as local sensitivity.

### Stage 2: Optimization before neural networks

- [Lesson 02 - Linear models and gradient descent](lessons/02_linear_models.md)
- Implement: predictions, mean squared error, cross entropy, gradient descent.
- Outcome: you can train a model by changing parameters to reduce a loss.

### Stage 3: Backpropagation

- [Lesson 03 - Reverse-mode automatic differentiation](lessons/03_backpropagation.md)
- Code: `deep_learning_from_scratch.autograd.Value`
- Outcome: you can build a computation graph and push gradients backward.

### Stage 4: Neural networks as composed functions

- [Lesson 04 - Multilayer perceptrons](lessons/04_multilayer_perceptrons.md)
- Code: `deep_learning_from_scratch.nn`
- Outcome: you can train an MLP from scratch on non-linear data.

### Stage 5: Practical training mechanics

- [Lesson 05 - Training loops, initialization, and generalization](lessons/05_training_mechanics.md)
- Code: `deep_learning_from_scratch.numpy_nn`
- Outcome: you can diagnose underfitting, overfitting, unstable updates, and bad
  initialization.

### Stage 6: Modern building blocks

- [Lesson 06 - Attention and sequence modeling](lessons/06_attention.md)
- Code: `deep_learning_from_scratch.attention`
- Outcome: you can explain attention as content-based weighted averaging and
  understand why transformers scale.

### Stage 7: Capstone

- [Lesson 07 - Capstone projects](lessons/07_capstone.md)
- Outcome: you build a small model end-to-end, write a short report, and compare
  your scratch implementation with a production framework.

## Local GPT (PyTorch)

This repo includes a decoder-only GPT transformer you can train and query on
local hardware, including AMD Radeon GPUs such as the RX 5700 XT (8 GB VRAM).

### AMD GPU setup (Linux + ROCm)

Install PyTorch built for ROCm, then install the GPT extras:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python -m pip install -e ".[gpt,dev]"
```

On Linux, ROCm exposes the GPU through `torch.cuda`, so training and inference
use the same commands as NVIDIA CUDA builds.

### Train

```bash
python scripts/train_gpt.py --preset rx5700xt --max-iters 5000
```

Presets: `tiny` (smoke tests), `rx5700xt` (recommended for 8 GB VRAM),
`local-large` (heavier model, mainly for inference).

The first run downloads the Shakespeare corpus into `data/gpt/shakespeare.txt`.
Question-answer examples live in `data/gpt/instruct.txt`.

### Chat / query

```bash
python scripts/chat_gpt.py
python scripts/chat_gpt.py --prompt "What does GPT stand for?"
python scripts/chat_gpt.py --greedy --prompt "What is 2 + 2?"
```

The chat script formats your input as `Question: ...\nAnswer:` and extracts the
generated answer text. Use `--greedy` for deterministic decoding when querying
facts from the instruct corpus.

### HTTP API (optional)

```bash
python -m pip install -e ".[gpt,api]"
python scripts/serve_gpt.py
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What does GPT stand for?","greedy":true}'
```

## First commands to run

```bash
pytest tests/test_autograd.py
python scripts/train_mlp_spiral.py --epochs 200 --hidden 16 --learning-rate 0.08
python scripts/inspect_attention.py
```

## What "from scratch" means here

This repo avoids deep learning frameworks while learning core mechanics. It does
use:

- Python for control flow.
- NumPy for arrays and linear algebra.
- Pytest for executable checks.

You will still write the forward pass, backward pass, parameter update, loss
calculation, and attention operation yourself.

## Suggested study cadence

For each lesson, aim to produce three artifacts:

1. A passing test.
2. A small experiment result.
3. A written explanation of the main idea.

If you can make a broken training run work and explain why it was broken, you
are learning the important part.
