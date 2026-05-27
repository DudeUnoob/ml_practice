# ML Practice

A personal machine learning workspace with two complementary tracks:

1. **Deep learning from scratch** — build intuition for neural networks, backprop,
   and transformers using small NumPy implementations and guided lessons.
2. **Classical ML notebooks** — scikit-learn practice on tabular, text, and
   time-series datasets.

The first track teaches *mechanisms* (what gradients, layers, and attention actually
do). The second track teaches *workflow* (EDA, model selection, hyperparameter
tuning, and serving predictions).

---

## Quick start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/DudeUnoob/ml_practice.git
cd ml_practice
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the core package and run tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

### Optional extras

| Extra | Install command | Use case |
|-------|-----------------|----------|
| `dev` | `pip install -e ".[dev]"` | pytest (included above) |
| `notebooks` | `pip install -e ".[notebooks]"` | Jupyter + scikit-learn practice |
| `gpt` | `pip install -e ".[gpt,dev]"` | Local PyTorch GPT training |
| `api` | `pip install -e ".[api]"` | FastAPI serving (GPT or Iris example) |

Combine extras as needed, e.g. `pip install -e ".[notebooks,api,gpt,dev]"`.

---

## Repository layout

```text
ml_practice/
├── lessons/                          Guided deep-learning curriculum (start here)
├── exercises/                        Checkpoints and mastery rubric
├── src/deep_learning_from_scratch/   NumPy teaching implementations + PyTorch GPT
├── scripts/                          Runnable experiments and GPT CLI tools
├── tests/                            Executable specs for core ideas
├── docs/                             Extended guides (local GPT)
├── data/gpt/                         Sample training text for the GPT module
└── notebooks/                        scikit-learn Jupyter projects (separate track)
```

---

## Track 1: Deep learning from scratch

The goal is not to memorize framework APIs. The goal is to build the mental model
behind them:

1. What is a model?
2. What is a loss?
3. Why do gradients point toward useful parameter updates?
4. How does backpropagation avoid recomputing the same work?
5. Why do initialization, normalization, optimization, and regularization matter?
6. How do modern layers such as attention fit the same pattern?

### Study loop

For each lesson:

1. **Read** the short lesson note.
2. **Predict** what the implementation should do before opening the code.
3. **Implement or modify** the referenced function.
4. **Run tests** for fast feedback.
5. **Run an experiment** and inspect failure modes.
6. **Write one paragraph** explaining the mechanism in your own words.

This read → derive → implement → test → reflect loop is slower than watching
videos, but it builds intuition that transfers to new architectures.

### Learning path

| Stage | Lesson | Code module | Outcome |
|-------|--------|-------------|---------|
| 0 | [Learning method](lessons/00_learning_method.md) | — | Run tests; know the study loop |
| 1 | [Math primitives](lessons/01_math_primitives.md) | — | Explain a gradient as local sensitivity |
| 2 | [Linear models](lessons/02_linear_models.md) | — | Train by minimizing a loss |
| 3 | [Backpropagation](lessons/03_backpropagation.md) | `autograd.Value` | Push gradients through a computation graph |
| 4 | [MLPs](lessons/04_multilayer_perceptrons.md) | `nn` | Train a multilayer network on non-linear data |
| 5 | [Training mechanics](lessons/05_training_mechanics.md) | `numpy_nn` | Diagnose under/overfitting and unstable training |
| 6 | [Attention](lessons/06_attention.md) | `attention` | Explain attention as weighted retrieval |
| 7 | [Capstone](lessons/07_capstone.md) | scripts + GPT | End-to-end project with a short report |

See [exercises/README.md](exercises/README.md) for checkpoint prompts and a
mastery rubric.

### First commands

```bash
pytest tests/test_autograd.py
python scripts/train_mlp_spiral.py --epochs 200 --hidden 16 --learning-rate 0.08
python scripts/inspect_attention.py
```

### What "from scratch" means here

Core mechanics use only **Python** and **NumPy**. You write the forward pass,
backward pass, parameter updates, loss, and attention yourself. Pytest provides
executable checks.

PyTorch appears only in the optional **GPT module** — a production-style
transformer that applies the same ideas at scale on local hardware.

---

## Track 2: Classical ML notebooks

Jupyter notebooks for classification, regression, time series, and a competition
template live under [notebooks/](notebooks/). See [notebooks/README.md](notebooks/README.md)
for the full project index and setup.

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Notable projects:

- **iris_dataset** — gentle multi-class classification intro
- **lasso_regression** — OLS, Lasso, Ridge, Random Forest (mlcourse.ai material)
- **logistic_regression_classification** — binary classification with GridSearchCV
- **iris_api** — train a Random Forest and serve it with FastAPI

---

## Local GPT (PyTorch)

Train and query a decoder-only GPT on your own GPU, including AMD Radeon cards
such as the RX 5700 XT (8 GB VRAM).

**Full guide:** [docs/local_gpt.md](docs/local_gpt.md) — architecture, AMD/NVIDIA
setup, training, chat, HTTP API, troubleshooting, and extensions.

### Quick start

```bash
# Install PyTorch for your platform first, then:
python -m pip install -e ".[gpt,dev,api]"

# Train (5700 XT preset)
python scripts/train_gpt.py --preset rx5700xt --max-iters 5000

# Chat
python scripts/chat_gpt.py --greedy --prompt "What does GPT stand for?"

# HTTP API
python scripts/serve_gpt.py
```

On Linux with ROCm, install PyTorch from the ROCm wheel index; the GPU is exposed
via `torch.cuda`. See the [hardware section](docs/local_gpt.md#hardware-and-pytorch-builds)
in the GPT docs.

---

## Scripts reference

| Script | Description |
|--------|-------------|
| `scripts/train_mlp_spiral.py` | Train a NumPy MLP on 2D spiral data |
| `scripts/inspect_attention.py` | Print attention weights for a toy sequence |
| `scripts/train_gpt.py` | Train the local character-level GPT |
| `scripts/chat_gpt.py` | Interactive or one-shot GPT queries |
| `scripts/serve_gpt.py` | REST API for the trained GPT |

---

## Suggested study cadence

For each deep-learning lesson, aim to produce three artifacts:

1. A passing test.
2. A small experiment result.
3. A written explanation of the main idea.

If you can fix a broken training run and explain why it was broken, you are
learning the important part.

---

## License

Lesson content and notebook adaptations may include third-party material (e.g.
[mlcourse.ai](https://mlcourse.ai), Kaggle competitions). Check individual
notebooks for attribution and license details.
