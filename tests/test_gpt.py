from __future__ import annotations

import torch

from deep_learning_from_scratch.gpt.config import GPTConfig
from deep_learning_from_scratch.gpt.generate import generate_text
from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.train("hello world")
    token_ids = tokenizer.encode("hello")
    assert tokenizer.decode(token_ids) == "hello"
    assert tokenizer.vocab_size >= 3


def test_gpt_forward_and_loss() -> None:
    config = GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    config = GPTConfig(
        block_size=config.block_size,
        vocab_size=10,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = GPT(config)
    token_ids = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))
    logits, loss = model(token_ids, targets=targets)
    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert loss.ndim == 0


def test_gpt_generate_smoke() -> None:
    text = "Question: What is 2 + 2?\nAnswer: 4.\n\n"
    tokenizer = CharTokenizer.train(text * 20)
    config = GPTConfig(
        block_size=32,
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Question: What is 2 + 2?\nAnswer:",
        max_new_tokens=8,
        temperature=1.0,
        top_k=5,
    )
    assert output.startswith("Question:")
