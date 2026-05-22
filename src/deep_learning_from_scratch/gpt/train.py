"""Training loop for local GPT models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from deep_learning_from_scratch.gpt.config import GPTConfig, TrainConfig
from deep_learning_from_scratch.gpt.dataset import TextCorpus, estimate_loss, get_batch, load_text_corpus
from deep_learning_from_scratch.gpt.device import autocast_context, get_device, resolve_dtype
from deep_learning_from_scratch.gpt.model import GPT


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    train_loss: float
    val_loss: float
    iterations: int
    device: str


def _learning_rate_schedule(*, iteration: int, config: TrainConfig) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / max(1, config.warmup_iters)
    if iteration > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iteration - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coefficient * (config.learning_rate - config.min_lr)


def train_gpt(
    *,
    data_paths: list[str | Path],
    output_dir: str | Path,
    model_config: GPTConfig,
    train_config: TrainConfig,
    preset_name: str | None = None,
    repeat_paths: dict[str | Path, int] | None = None,
) -> TrainResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(train_config.seed)
    device = get_device()
    dtype = resolve_dtype(device=device, use_amp=train_config.use_amp)

    corpus = load_text_corpus(
        data_paths=data_paths,
        train_split=train_config.train_split,
        seed=train_config.seed,
        repeat_paths=repeat_paths,
    )
    model_config = GPTConfig(
        block_size=model_config.block_size,
        vocab_size=corpus.tokenizer.vocab_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        dropout=model_config.dropout,
        bias=model_config.bias,
    )
    corpus.tokenizer.save(output_path / "tokenizer.json")

    model = GPT(model_config).to(device)
    if train_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=dtype != torch.float32 and device.type == "cuda")

    parameter_millions = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Model parameters: {parameter_millions:.2f}M")
    if preset_name:
        print(f"Preset: {preset_name}")

    start_time = time.time()
    best_val_loss = float("inf")
    checkpoint_path = output_path / "checkpoint.pt"

    for iteration in range(train_config.max_iters):
        learning_rate = _learning_rate_schedule(iteration=iteration, config=train_config)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = get_batch(
            token_ids=corpus.train_ids,
            block_size=model_config.block_size,
            batch_size=train_config.batch_size,
            device=device,
        )

        with autocast_context(device=device, dtype=dtype):
            _, loss = model(inputs, targets=targets)
        if loss is None:
            raise RuntimeError("Training step did not produce a loss.")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (iteration + 1) % train_config.eval_interval == 0 or iteration == 0:
            train_loss = estimate_loss(
                model=model,
                token_ids=corpus.train_ids,
                block_size=model_config.block_size,
                batch_size=train_config.batch_size,
                device=device,
                eval_iters=train_config.eval_iters,
            )
            val_loss = estimate_loss(
                model=model,
                token_ids=corpus.val_ids,
                block_size=model_config.block_size,
                batch_size=train_config.batch_size,
                device=device,
                eval_iters=train_config.eval_iters,
            )
            elapsed = time.time() - start_time
            print(
                f"iter {iteration + 1:5d} | lr {learning_rate:.2e} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(
                    str(checkpoint_path),
                    extra={"train_loss": train_loss, "val_loss": val_loss, "iteration": iteration + 1},
                )

    if not checkpoint_path.exists():
        model.save_checkpoint(str(checkpoint_path), extra={"iteration": train_config.max_iters})

    final_train_loss = estimate_loss(
        model=model,
        token_ids=corpus.train_ids,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        device=device,
        eval_iters=train_config.eval_iters,
    )
    final_val_loss = estimate_loss(
        model=model,
        token_ids=corpus.val_ids,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        device=device,
        eval_iters=train_config.eval_iters,
    )
    return TrainResult(
        checkpoint_path=checkpoint_path,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        iterations=train_config.max_iters,
        device=str(device),
    )
