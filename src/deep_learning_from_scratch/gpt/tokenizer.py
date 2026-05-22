"""Character-level tokenizer for small local GPT models."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class CharTokenizer:
    """Map UTF-8 characters to integer token ids."""

    def __init__(self, *, stoi: dict[str, int], itos: dict[int, str]) -> None:
        if not stoi or not itos:
            raise ValueError("Tokenizer vocabulary cannot be empty.")
        if len(stoi) != len(itos):
            raise ValueError("stoi and itos must have the same length.")

        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def train(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("Cannot train tokenizer on empty text.")

        characters = sorted(set(text))
        stoi = {character: index for index, character in enumerate(characters)}
        itos = {index: character for character, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {key: int(value) for key, value in payload["stoi"].items()}
        itos = {int(key): value for key, value in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}, indent=2),
            encoding="utf-8",
        )

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi.get("\ufffd")
        if unknown is None:
            missing = sorted({character for character in text if character not in self.stoi})
            if missing:
                preview = "".join(missing[:8])
                raise ValueError(f"Text contains unknown characters: {preview!r}")
        return [self.stoi.get(character, unknown) for character in text]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self.itos[int(token_id)] for token_id in token_ids)

    def encode_tensor(self, text: str, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)
