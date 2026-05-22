"""PyTorch GPT-style transformer for local training and inference."""

from deep_learning_from_scratch.gpt.config import GPTConfig, PRESETS
from deep_learning_from_scratch.gpt.device import get_device, resolve_dtype
from deep_learning_from_scratch.gpt.generate import generate_text
from deep_learning_from_scratch.gpt.model import GPT
from deep_learning_from_scratch.gpt.tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "GPT",
    "GPTConfig",
    "PRESETS",
    "generate_text",
    "get_device",
    "resolve_dtype",
]
