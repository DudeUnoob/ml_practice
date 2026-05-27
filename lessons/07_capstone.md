# Lesson 07 - Capstone projects

The capstone is where you turn mechanisms into judgment. Choose one project and
write a short report with:

- Problem statement.
- Dataset description.
- Model architecture.
- Training curve.
- Failure cases.
- What you would change next.

## Project A: classify spiral data

Build on `scripts/train_mlp_spiral.py`.

Required changes:

- Add a validation split.
- Compare at least three hidden sizes.
- Compare at least three learning rates.
- Explain the best and worst runs.

## Project B: tiny character model

Implement a small next-character predictor.

Suggested order:

1. Build a vocabulary.
2. Convert text to integer token IDs.
3. Create input/target pairs.
4. Start with a bigram model.
5. Add an embedding table.
6. Add one attention block.

Keep the dataset tiny. The goal is understanding, not benchmark performance.

## Project C: compare scratch code with a framework

Re-implement the spiral classifier in PyTorch after finishing the scratch
version.

Answer:

- Which lines did the framework remove?
- Which concepts stayed the same?
- Which bugs became easier or harder to find?

## Final reflection

You understand a component when you can:

1. Write its forward pass.
2. Explain the shape of every tensor.
3. Derive or test its backward pass.
4. Predict how it fails.
5. Connect it to the same component in a production framework.
