"""Decoder-only GPT transformer implemented in PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_learning_from_scratch.gpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query, key, value = self.c_attn(hidden_states).split(self.n_embd, dim=2)
        query = query.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        attention_weights = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention_weights = attention_weights.transpose(1, 2).contiguous()
        attention_weights = attention_weights.view(batch_size, sequence_length, self.n_embd)
        return self.resid_dropout(self.c_proj(attention_weights))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class GPT(nn.Module):
    """GPT-style causal language model with tied input/output embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.config.block_size}."
            )

        positions = torch.arange(0, sequence_length, device=token_ids.device)
        hidden_states = self.transformer.wte(token_ids) + self.transformer.wpe(positions)
        hidden_states = self.transformer.drop(hidden_states)
        for block in self.transformer.h:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        do_sample: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return token_ids

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

            if do_sample:
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids

    def configure_optimizers(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
    ) -> torch.optim.AdamW:
        decay_parameters: set[str] = set()
        no_decay_parameters: set[str] = set()
        for parameter_name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2:
                decay_parameters.add(parameter_name)
            else:
                no_decay_parameters.add(parameter_name)

        parameter_groups = [
            {
                "params": [parameter for name, parameter in self.named_parameters() if name in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [parameter for name, parameter in self.named_parameters() if name in no_decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(parameter_groups, lr=learning_rate, betas=betas)

    @classmethod
    def from_checkpoint(cls, path: str, *, map_location: str | torch.device = "cpu") -> GPT:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = GPTConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model"])
        return model

    def save_checkpoint(self, path: str, *, extra: dict | None = None) -> None:
        payload = {
            "config": self.config.to_dict(),
            "model": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
