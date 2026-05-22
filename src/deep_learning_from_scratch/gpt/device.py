"""Device and mixed-precision helpers for AMD, NVIDIA, Apple, and CPU."""

from __future__ import annotations

import torch


def get_device(*, prefer_gpu: bool = True) -> torch.device:
    """Pick the best available compute device.

    On Linux with ROCm, ``torch.cuda.is_available()`` is True for AMD GPUs such
    as the Radeon RX 5700 XT. Install the ROCm build of PyTorch for GPU training.
    """

    if prefer_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
        return torch.device("cuda")

    if prefer_gpu and torch.backends.mps.is_available():
        print("Using Apple Metal (MPS).")
        return torch.device("mps")

    print("Using CPU. For AMD GPU support, install PyTorch with ROCm.")
    return torch.device("cpu")


def resolve_dtype(*, device: torch.device, use_amp: bool) -> torch.dtype:
    if not use_amp:
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast_context(*, device: torch.device, dtype: torch.dtype):
    """Return a context manager for mixed precision, or a no-op on CPU."""

    if device.type == "cpu" or dtype == torch.float32:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)
