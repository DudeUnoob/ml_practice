# Lesson 06 - Attention and sequence modeling

Attention lets each item in a sequence choose what other items to read from.

For each query, compare it with every key:

```text
scores = query @ key.T / sqrt(width)
weights = softmax(scores)
output = weights @ value
```

The output is a weighted average of values. The weights depend on content, not
just position.

## Query, key, value intuition

- Query: what am I looking for?
- Key: what information do I contain?
- Value: what should be copied if I am relevant?

This separation lets a token decide what to attend to and what information to
retrieve.

## Why scale by `sqrt(width)`?

Dot products get larger as vector width grows. Large scores make softmax overly
sharp, which can cause tiny gradients. Scaling keeps scores in a friendlier
range.

## Causal masks

Language models should not read future tokens while predicting the next token.
A causal mask hides positions to the right of the current token.

## Positional information

Attention alone does not know token order. Positional encodings add order
signals to token embeddings before attention.

## Exercises

1. Run `scripts/inspect_attention.py`.
2. Create a sequence where one query strongly matches one key.
3. Apply a causal mask and verify future positions receive zero probability.
4. Add sinusoidal position encodings to random token vectors and compare
   attention weights before and after.

## Reflection prompt

In what way is attention similar to nearest-neighbor lookup, and in what way is
it more flexible?
