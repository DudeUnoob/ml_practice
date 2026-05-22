from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deep_learning_from_scratch.gpt.config import PRESETS, GPTConfig, TrainConfig
from deep_learning_from_scratch.gpt.generate import generate_text
from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer
from deep_learning_from_scratch.gpt.train import train_gpt


def test_train_gpt_writes_checkpoint_and_tokenizer(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "Question: What is 2 + 2?\nAnswer: 4.\n\n" * 50,
        encoding="utf-8",
    )

    output_dir = tmp_path / "model"
    result = train_gpt(
        data_paths=[corpus_path],
        output_dir=output_dir,
        model_config=PRESETS["tiny"],
        train_config=TrainConfig(
            max_iters=120,
            eval_interval=60,
            eval_iters=5,
            batch_size=4,
            use_amp=False,
        ),
    )

    assert result.checkpoint_path.exists()
    assert (output_dir / "tokenizer.json").exists()
    assert result.train_loss > 0.0
    assert result.val_loss > 0.0

    model = GPT.from_checkpoint(str(result.checkpoint_path), map_location="cpu")
    tokenizer = CharTokenizer.load(output_dir / "tokenizer.json")
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Question: What is 2 + 2?\nAnswer:",
        max_new_tokens=12,
        temperature=0.1,
        top_k=5,
        do_sample=False,
        device=torch.device("cpu"),
    )
    assert output.startswith("Question:")


def test_chat_helpers_format_and_extract() -> None:
    from scripts.chat_gpt import extract_answer, format_question

    assert format_question("What is water?") == "Question: What is water?\nAnswer:"
    assert format_question("Question: Already formatted") == "Question: Already formatted\nAnswer:"

    raw = "Question: What is 2 + 2?\nAnswer: 4.\n\nQuestion: Next?"
    assert extract_answer(raw) == "4."


def test_serve_app_health_and_query(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from scripts.serve_gpt import create_app

    corpus = "Question: abcd\nAnswer: abcd.\n\n" * 20
    tokenizer = CharTokenizer.train(corpus)
    config = GPTConfig(
        block_size=32,
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    device = torch.device("cpu")

    app = create_app(model=model, tokenizer=tokenizer, device=device)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/query",
        json={"prompt": "abcd", "max_new_tokens": 8, "greedy": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt"] == "abcd"
    assert "answer" in payload
    assert "raw_text" in payload
