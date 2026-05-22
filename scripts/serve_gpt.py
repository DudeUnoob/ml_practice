"""Serve a trained GPT model over HTTP for querying."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from deep_learning_from_scratch.gpt.device import get_device
from deep_learning_from_scratch.gpt.generate import generate_text
from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError as error:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Install FastAPI extras first: python -m pip install -e \".[gpt,api]\""
    ) from error


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=120, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_k: int = Field(default=20, ge=1, le=200)
    greedy: bool = False


class QueryResponse(BaseModel):
    prompt: str
    answer: str
    raw_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gpt/checkpoint.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/gpt/tokenizer.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def format_question(question: str) -> str:
    question = question.strip()
    if question.lower().startswith("question:"):
        return f"{question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def extract_answer(text: str) -> str:
    marker = "Answer:"
    if marker not in text:
        return text.strip()
    answer = text.split(marker, maxsplit=1)[1]
    if "\n\nQuestion:" in answer:
        answer = answer.split("\n\nQuestion:", maxsplit=1)[0]
    return answer.strip()


def create_app(*, model: GPT, tokenizer: CharTokenizer, device: torch.device) -> FastAPI:
    app = FastAPI(title="Local GPT", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": str(device)}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        prompt = format_question(request.prompt)
        try:
            raw_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                do_sample=not request.greedy,
                device=device,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return QueryResponse(
            prompt=request.prompt,
            answer=extract_answer(raw_text),
            raw_text=raw_text,
        )

    return app


def main() -> None:
    args = parse_args()
    if args.cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not args.checkpoint.exists() or not args.tokenizer.exists():
        raise FileNotFoundError("Train the model first with: python scripts/train_gpt.py")

    device = get_device()
    model = GPT.from_checkpoint(str(args.checkpoint), map_location=device).to(device)
    tokenizer = CharTokenizer.load(args.tokenizer)
    app = create_app(model=model, tokenizer=tokenizer, device=device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
