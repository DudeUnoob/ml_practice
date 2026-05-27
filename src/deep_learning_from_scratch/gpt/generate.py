"""Text generation helpers for GPT models."""

from __future__ import annotations

import torch

from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer


def generate_text(
    *,
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    do_sample: bool = True,
    device: torch.device | None = None,
) -> str:
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    device = device or next(model.parameters()).device
    model.eval()
    token_ids = tokenizer.encode_tensor(prompt, device=device).unsqueeze(0)
    output_ids = model.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
    )
    return tokenizer.decode(output_ids[0])
