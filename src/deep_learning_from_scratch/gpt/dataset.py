"""Text dataset utilities for GPT training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer


@dataclass(frozen=True)
class TextCorpus:
    train_ids: np.ndarray
    val_ids: np.ndarray
    tokenizer: CharTokenizer


def load_text_corpus(
    *,
    data_paths: list[str | Path],
    train_split: float = 0.9,
    seed: int = 42,
    repeat_paths: dict[str | Path, int] | None = None,
) -> TextCorpus:
    if not data_paths:
        raise ValueError("At least one data path is required.")
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")

    chunks: list[str] = []
    for data_path in data_paths:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        repeats = 1
        if repeat_paths is not None:
            repeats = repeat_paths.get(path, repeat_paths.get(str(path), 1))
        chunks.extend([path.read_text(encoding="utf-8")] * repeats)

    text = "\n\n".join(chunks)
    tokenizer = CharTokenizer.train(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.uint16)

    split_index = int(train_split * len(token_ids))
    train_ids = token_ids[:split_index]
    val_ids = token_ids[split_index:]
    return TextCorpus(train_ids=train_ids, val_ids=val_ids, tokenizer=tokenizer)


def get_batch(
    *,
    token_ids: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= block_size + 1:
        raise ValueError("Corpus is too small for the requested block size.")

    start_indices = np.random.randint(0, len(token_ids) - block_size - 1, size=batch_size)
    inputs = np.stack([token_ids[start : start + block_size] for start in start_indices])
    targets = np.stack([token_ids[start + 1 : start + block_size + 1] for start in start_indices])
    input_tensor = torch.from_numpy(inputs.astype(np.int64)).to(device)
    target_tensor = torch.from_numpy(targets.astype(np.int64)).to(device)
    return input_tensor, target_tensor


@torch.no_grad()
def estimate_loss(
    *,
    model: torch.nn.Module,
    token_ids: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(
            token_ids=token_ids,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )
        _, loss = model(inputs, targets=targets)
        if loss is None:
            raise RuntimeError("Model did not return a loss during evaluation.")
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))
