"""Query a trained local GPT model in an interactive chat loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deep_learning_from_scratch.gpt.device import get_device
from deep_learning_from_scratch.gpt.generate import generate_text
from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gpt/checkpoint.pt"),
        help="Path to a trained checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/gpt/tokenizer.json"),
        help="Path to the character tokenizer JSON file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt; omit for REPL.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for sharper answers.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    return parser.parse_args()


def format_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
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


def respond(
    *,
    model: GPT,
    tokenizer: CharTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    do_sample: bool,
    device: torch.device,
) -> str:
    prompt = format_question(question)
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        device=device,
    )
    return extract_answer(generated)


def main() -> None:
    args = parse_args()
    if args.cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {args.checkpoint}. Train first with: python scripts/train_gpt.py"
        )
    if not args.tokenizer.exists():
        raise FileNotFoundError(
            f"Missing tokenizer at {args.tokenizer}. Train first with: python scripts/train_gpt.py"
        )

    device = get_device()
    model = GPT.from_checkpoint(str(args.checkpoint), map_location=device).to(device)
    tokenizer = CharTokenizer.load(args.tokenizer)

    if args.prompt is not None:
        answer = respond(
            model=model,
            tokenizer=tokenizer,
            question=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.greedy,
            device=device,
        )
        print(answer)
        return

    print("Local GPT chat. Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer = respond(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.greedy,
            device=device,
        )
        print(f"GPT> {answer}")


if __name__ == "__main__":
    main()
