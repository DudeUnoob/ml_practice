"""Train a local GPT model on bundled text corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_learning_from_scratch.gpt.config import PRESETS, GPTConfig, TrainConfig
from deep_learning_from_scratch.gpt.train import train_gpt

DEFAULT_DATA = [
    Path("data/gpt/instruct.txt"),
    Path("data/gpt/shakespeare.txt"),
]
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="rx5700xt",
        help="Model size preset tuned for local hardware.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/gpt"),
        help="Directory for checkpoints and tokenizer files.",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Use only the files passed via --data (skip default corpora).",
    )
    parser.add_argument(
        "--instruct-repeats",
        type=int,
        default=80,
        help="How many times to repeat the instruct corpus during training.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="*",
        default=None,
        help="Optional extra corpus files to include in training.",
    )
    return parser.parse_args()


def ensure_corpus_files() -> list[Path]:
    import urllib.request

    data_paths = list(DEFAULT_DATA)
    shakespeare_path = Path("data/gpt/shakespeare.txt")
    shakespeare_path.parent.mkdir(parents=True, exist_ok=True)
    if not shakespeare_path.exists():
        print(f"Downloading Shakespeare corpus to {shakespeare_path}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, shakespeare_path)
    return data_paths


def main() -> None:
    args = parse_args()
    if args.cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model_config = PRESETS[args.preset]
    if args.block_size is not None:
        model_config = GPTConfig(
            block_size=args.block_size,
            vocab_size=model_config.vocab_size,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_embd=model_config.n_embd,
            dropout=model_config.dropout,
            bias=model_config.bias,
        )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16 if args.preset == "local-large" else 32

    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        max_iters=args.max_iters,
        lr_decay_iters=args.max_iters,
        seed=args.seed,
        use_amp=not args.cpu,
    )

    if args.data_only:
        if not args.data:
            raise SystemExit("--data-only requires at least one --data path.")
        data_paths = list(args.data)
        repeat_paths = None
    else:
        data_paths = ensure_corpus_files()
        if args.data:
            data_paths.extend(args.data)
        repeat_paths = {Path("data/gpt/instruct.txt"): args.instruct_repeats}

    result = train_gpt(
        data_paths=data_paths,
        output_dir=args.output_dir,
        model_config=model_config,
        train_config=train_config,
        preset_name=args.preset,
        repeat_paths=repeat_paths,
    )
    print(f"checkpoint={result.checkpoint_path}")
    print(f"train_loss={result.train_loss:.4f} val_loss={result.val_loss:.4f}")
    print(f"device={result.device}")


if __name__ == "__main__":
    main()
