"""GPT model and training configuration presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters for a decoder-only GPT transformer."""

    block_size: int = 256
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    def parameter_count(self) -> int:
        """Approximate trainable parameter count (excluding tied embeddings)."""
        head_dim = self.n_embd // self.n_head
        if head_dim * self.n_head != self.n_embd:
            raise ValueError("n_embd must be divisible by n_head.")

        attention = 4 * self.n_embd * self.n_embd
        mlp = 8 * self.n_embd * self.n_embd
        block = attention + mlp + 4 * self.n_embd
        return self.vocab_size * self.n_embd + self.n_layer * block + self.n_embd

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int | float | bool]) -> GPTConfig:
        return cls(**payload)


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 3e-5
    train_split: float = 0.9
    seed: int = 42
    use_amp: bool = True
    compile_model: bool = False


PRESETS: dict[str, GPTConfig] = {
    # Fast smoke tests and laptops without a GPU.
    "tiny": GPTConfig(
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    ),
    # Recommended default for AMD Radeon RX 5700 XT (8 GB VRAM).
    "rx5700xt": GPTConfig(
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
    ),
    # Larger local model when you have headroom on 8 GB VRAM (inference-first).
    "local-large": GPTConfig(
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
    ),
}
